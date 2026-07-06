use std::{path::PathBuf, sync::Arc};

use anyhow::{Context, Result};
use futures::{stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    agents::{CorrectionAgent, CorrectionAgentInput, DsrsCorrectionAgent},
    artifacts::extract_instruction_from_path,
    config::{ProxyConfig, UpstreamConfig},
    model_profile::{builtin_profiles, resolve_profile, ModelProfile},
    openai::{ChatMessage, OpenAiTool},
    response_interpreter::{ResponseFailure, ToolIntent},
};

#[derive(Debug, Clone)]
pub struct CorrectionEvalConfig {
    pub dataset_path: PathBuf,
    pub output_path: Option<PathBuf>,
    pub upstream: UpstreamConfig,
    pub proxy_config: ProxyConfig,
    pub target_model: String,
    pub profile: Option<String>,
    pub correction_agent_artifact: Option<PathBuf>,
    pub dataset_model_filter: Option<String>,
    pub dataset_profile_filter: Option<String>,
    pub max_examples: Option<usize>,
    pub concurrency: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionEvalReport {
    pub dataset_path: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_path: Option<PathBuf>,
    pub target_model: String,
    pub profile: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correction_agent_artifact: Option<String>,
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub skipped: usize,
    pub results: Vec<CorrectionEvalCaseResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionEvalCaseResult {
    pub trace_id: String,
    pub source_model: String,
    pub source_profile: String,
    pub passed: bool,
    pub reason: String,
    pub expected: Value,
    pub predicted: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_output: Option<String>,
}

pub async fn run_correction_eval(config: CorrectionEvalConfig) -> Result<CorrectionEvalReport> {
    let Some(api_key) = config.upstream.api_key.clone() else {
        anyhow::bail!(
            "OPENROUTER_API_KEY or ATTUNE_UPSTREAM_API_KEY is required for correction-agent eval"
        );
    };
    let mut profile = if let Some(profile_name) = &config.profile {
        profile_by_name(profile_name, &config.proxy_config.model_profiles)
            .with_context(|| format!("unknown correction eval profile {profile_name:?}"))?
    } else {
        resolve_profile(&config.target_model, &config.proxy_config.model_profiles)
    };
    if let Some(artifact_path) = &config.correction_agent_artifact {
        let content = tokio::fs::read_to_string(artifact_path)
            .await
            .with_context(|| {
                format!(
                    "failed to read correction-agent artifact {}",
                    artifact_path.display()
                )
            })?;
        let instruction = extract_instruction_from_path(artifact_path, &content)?;
        profile.correction_agent_artifact = Some(artifact_path.display().to_string());
        profile.correction_instruction = Some(instruction);
    }

    let rows = read_dataset_rows(&config.dataset_path).await?;
    let agent = Arc::new(DsrsCorrectionAgent::new(
        config.upstream.clone(),
        correction_config_for_target(&config),
    ));

    let mut eval_rows = Vec::new();
    let mut skipped = 0usize;
    for row in rows {
        if !row_matches_eval_filter(&row, &config) {
            skipped += 1;
            continue;
        }
        if let Some(max_examples) = config.max_examples {
            if eval_rows.len() >= max_examples {
                skipped += 1;
                continue;
            }
        }
        eval_rows.push(row);
    }

    let total_rows = eval_rows.len();
    let concurrency = config.concurrency.max(1);
    eprintln!("correction eval: evaluating {total_rows} rows with concurrency {concurrency}");
    let eval_target_model = config.target_model.clone();
    let eval_profile = profile.clone();
    let mut pending = stream::iter(eval_rows.into_iter().enumerate().map(|(index, row)| {
        let target_model = eval_target_model.clone();
        let profile = eval_profile.clone();
        let api_key = api_key.clone();
        let agent = Arc::clone(&agent);
        async move {
            let result =
                evaluate_correction_row(&row, &target_model, &profile, api_key, &agent).await;
            (index, result)
        }
    }))
    .buffer_unordered(concurrency);

    let mut result_slots = vec![None; total_rows];
    let mut completed = 0usize;
    while let Some((index, result)) = pending.next().await {
        result_slots[index] = Some(result?);
        completed += 1;
        if completed == total_rows || completed == 1 || completed % 10 == 0 {
            eprintln!("correction eval: completed {completed}/{total_rows}");
        }
    }
    let results = result_slots
        .into_iter()
        .map(|result| result.context("correction eval result slot was not filled"))
        .collect::<Result<Vec<_>>>()?;

    let passed = results.iter().filter(|result| result.passed).count();
    let failed = results.len().saturating_sub(passed);
    let report = CorrectionEvalReport {
        dataset_path: config.dataset_path,
        output_path: config.output_path.clone(),
        target_model: config.target_model,
        profile: profile.name,
        correction_agent_artifact: profile.correction_agent_artifact,
        total: results.len(),
        passed,
        failed,
        skipped,
        results,
    };
    if let Some(output_path) = &config.output_path {
        if let Some(parent) = output_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        tokio::fs::write(output_path, serde_json::to_vec_pretty(&report)?)
            .await
            .with_context(|| format!("failed to write {}", output_path.display()))?;
    }
    Ok(report)
}

async fn read_dataset_rows(path: &PathBuf) -> Result<Vec<Value>> {
    let content = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("failed to read {}", path.display()))?;
    content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str::<Value>(line).context("failed to parse dataset JSONL row")
        })
        .collect()
}

fn correction_config_for_target(config: &CorrectionEvalConfig) -> crate::config::CorrectionConfig {
    let mut correction = config.proxy_config.correction.clone();
    correction.default_model = Some(config.target_model.clone());
    correction
}

fn profile_by_name(name: &str, configured: &[ModelProfile]) -> Option<ModelProfile> {
    configured
        .iter()
        .cloned()
        .chain(builtin_profiles())
        .find(|profile| profile.name == name)
}

fn row_matches_eval_filter(row: &Value, config: &CorrectionEvalConfig) -> bool {
    matches_optional_text_filter(
        row.get("model").and_then(Value::as_str).unwrap_or_default(),
        config.dataset_model_filter.as_deref(),
    ) && matches_optional_text_filter(
        row.get("profile")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        config.dataset_profile_filter.as_deref(),
    )
}

fn matches_optional_text_filter(actual: &str, filter: Option<&str>) -> bool {
    filter
        .map(|filter| {
            actual
                .to_ascii_lowercase()
                .contains(&filter.to_ascii_lowercase())
        })
        .unwrap_or(true)
}

async fn evaluate_correction_row(
    row: &Value,
    target_model: &str,
    profile: &ModelProfile,
    api_key: String,
    agent: &Arc<DsrsCorrectionAgent>,
) -> Result<CorrectionEvalCaseResult> {
    let trace_id = row
        .get("trace_id")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let source_model = row
        .get("model")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let source_profile = row
        .get("profile")
        .and_then(Value::as_str)
        .unwrap_or("unknown")
        .to_string();
    let expected = row.get("expected_repair").cloned().unwrap_or(Value::Null);
    let input = correction_input_from_row(row, target_model, profile.clone(), api_key)?;
    let output = agent.correct(input).await;
    let (predicted, raw_output) = match output {
        Ok(Some(output)) => (correction_output_value(&output), output.raw_output),
        Ok(None) => (
            json!({
                "possible": false,
                "content": "",
                "tool_calls": [],
                "error": "correction agent unavailable"
            }),
            None,
        ),
        Err(error) => (
            json!({
                "possible": false,
                "content": "",
                "tool_calls": [],
                "error": error.to_string()
            }),
            None,
        ),
    };
    let (passed, reason) = score_repair(&expected, &predicted);

    Ok(CorrectionEvalCaseResult {
        trace_id,
        source_model,
        source_profile,
        passed,
        reason,
        expected: repair_scoring_value(&expected),
        predicted: repair_scoring_value(&predicted),
        raw_output,
    })
}

fn correction_input_from_row(
    row: &Value,
    target_model: &str,
    profile: ModelProfile,
    api_key: String,
) -> Result<CorrectionAgentInput> {
    Ok(CorrectionAgentInput {
        tools: serde_json::from_value(row_value(row, "available_tools")?)
            .context("failed to parse available_tools")?,
        recent_messages: serde_json::from_value(row_value(row, "recent_messages")?)
            .context("failed to parse recent_messages")?,
        malformed_response: string_field(row, "malformed_response"),
        parser_events: serde_json::from_value(row_value(row, "parser_events")?)
            .context("failed to parse parser_events")?,
        response_failures: serde_json::from_value(row_value(row, "response_failures")?)
            .context("failed to parse response_failures")?,
        model: target_model.to_string(),
        profile,
        api_key_override: Some(api_key),
    })
}

fn row_value(row: &Value, field: &str) -> Result<Value> {
    row.get(field)
        .cloned()
        .with_context(|| format!("dataset row missing {field}"))
}

fn string_field(row: &Value, field: &str) -> String {
    match row.get(field) {
        Some(Value::String(text)) => text.clone(),
        Some(other) => other.to_string(),
        None => String::new(),
    }
}

fn correction_output_value(output: &crate::agents::CorrectionAgentOutput) -> Value {
    json!({
        "possible": output.possible,
        "confidence": output.confidence,
        "content": output.content.clone().unwrap_or_default(),
        "tool_calls": output.tool_calls.iter().map(tool_intent_value).collect::<Vec<_>>()
    })
}

fn tool_intent_value(intent: &ToolIntent) -> Value {
    json!({
        "name": intent.name,
        "arguments": intent.arguments.clone().unwrap_or_else(|| parse_arguments(&intent.raw_arguments))
    })
}

fn score_repair(expected: &Value, predicted: &Value) -> (bool, String) {
    let expected_possible = expected.get("possible").and_then(Value::as_bool);
    if expected_possible == Some(false) {
        let predicted_possible = predicted.get("possible").and_then(Value::as_bool);
        let predicted_calls = normalized_tool_calls(predicted);
        let predicted_content = repair_content(predicted);
        if predicted_possible == Some(false)
            && predicted_calls.is_empty()
            && predicted_content.is_empty()
        {
            return (true, "expected and predicted not_possible".to_string());
        }
        return (
            false,
            "expected not_possible, but prediction attempted a repair".to_string(),
        );
    }

    let predicted_possible = predicted.get("possible").and_then(Value::as_bool);
    if predicted_possible == Some(false) {
        return (
            false,
            "expected possible repair, but prediction marked not_possible".to_string(),
        );
    }

    let expected_calls = normalized_tool_calls(expected);
    let predicted_calls = normalized_tool_calls(predicted);
    let expected_content = repair_content(expected);
    let predicted_content = repair_content(predicted);
    if !expected_calls.is_empty() {
        if expected_calls != predicted_calls {
            return (
                false,
                format!(
                    "tool call mismatch; expected {}; predicted {}",
                    json!(expected_calls),
                    json!(predicted_calls)
                ),
            );
        }
        if expected_content.is_empty() || expected_content == predicted_content {
            return (
                true,
                "predicted tool calls matched expected repair".to_string(),
            );
        }
        return (
            false,
            format!(
                "content mismatch; expected {expected_content:?}; predicted {predicted_content:?}"
            ),
        );
    }

    if !predicted_calls.is_empty() {
        return (
            false,
            format!(
                "expected no tool calls; predicted {}",
                json!(predicted_calls)
            ),
        );
    }

    if expected_content.is_empty() || expected_content == predicted_content {
        return (
            true,
            "content-only repair matched expected repair".to_string(),
        );
    }

    (
        false,
        format!("content mismatch; expected {expected_content:?}; predicted {predicted_content:?}"),
    )
}

fn repair_scoring_value(value: &Value) -> Value {
    json!({
        "possible": value.get("possible").cloned().unwrap_or(Value::Bool(true)),
        "content": repair_content(value),
        "tool_calls": normalized_tool_calls(value),
    })
}

fn repair_content(value: &Value) -> String {
    value
        .get("content")
        .and_then(Value::as_str)
        .or_else(|| {
            value
                .pointer("/final_message/content")
                .and_then(Value::as_str)
        })
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn normalized_tool_calls(value: &Value) -> Vec<Value> {
    value
        .get("tool_calls")
        .and_then(Value::as_array)
        .map(|calls| {
            calls
                .iter()
                .filter_map(normalized_tool_call)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn normalized_tool_call(call: &Value) -> Option<Value> {
    if let Some(function) = call.get("function") {
        let name = function.get("name").and_then(Value::as_str)?.to_string();
        let arguments = function
            .get("arguments")
            .cloned()
            .map(normalize_arguments_value)
            .unwrap_or_else(|| json!({}));
        return Some(json!({ "name": name, "arguments": arguments }));
    }

    let name = call.get("name").and_then(Value::as_str)?.to_string();
    let arguments = call
        .get("arguments")
        .cloned()
        .map(normalize_arguments_value)
        .unwrap_or_else(|| json!({}));
    Some(json!({ "name": name, "arguments": arguments }))
}

fn normalize_arguments_value(value: Value) -> Value {
    match value {
        Value::String(text) => normalize_argument_strings(parse_arguments(&text)),
        Value::Array(values) => Value::Array(
            values
                .into_iter()
                .map(normalize_arguments_value)
                .collect::<Vec<_>>(),
        ),
        Value::Object(object) => Value::Object(
            object
                .into_iter()
                .map(|(key, value)| (key, normalize_arguments_value(value)))
                .collect(),
        ),
        Value::Null => json!({}),
        other => normalize_argument_strings(other),
    }
}

fn parse_arguments(raw: &str) -> Value {
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

fn normalize_argument_strings(value: Value) -> Value {
    match value {
        Value::String(text) => Value::String(text.replace("\\r\\n", "\n").replace("\\n", "\n")),
        Value::Array(values) => Value::Array(
            values
                .into_iter()
                .map(normalize_argument_strings)
                .collect::<Vec<_>>(),
        ),
        Value::Object(object) => Value::Object(
            object
                .into_iter()
                .map(|(key, value)| (key, normalize_argument_strings(value)))
                .collect(),
        ),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_repair_rejects_not_possible_when_repair_expected() {
        let expected = json!({
            "possible": true,
            "content": "",
            "tool_calls": [{"name":"search","arguments":{"query":"redis"}}]
        });
        let predicted = json!({
            "possible": false,
            "content": "",
            "tool_calls": [{"name":"search","arguments":{"query":"redis"}}]
        });

        let (passed, reason) = score_repair(&expected, &predicted);

        assert!(!passed);
        assert_eq!(
            reason,
            "expected possible repair, but prediction marked not_possible"
        );
    }

    #[test]
    fn score_repair_rejects_missing_required_content_with_tool_calls() {
        let expected = json!({
            "possible": true,
            "content": "I'll search Redis configuration.",
            "tool_calls": [{"name":"search","arguments":{"query":"redis"}}]
        });
        let predicted = json!({
            "possible": true,
            "content": "",
            "tool_calls": [{"name":"search","arguments":{"query":"redis"}}]
        });

        let (passed, reason) = score_repair(&expected, &predicted);

        assert!(!passed);
        assert_eq!(
            reason,
            "content mismatch; expected \"I'll search Redis configuration.\"; predicted \"\""
        );
    }

    #[test]
    fn score_repair_allows_empty_expected_content_with_tool_calls() {
        let expected = json!({
            "possible": true,
            "content": "",
            "tool_calls": [{"name":"search","arguments":{"query":"redis"}}]
        });
        let predicted = json!({
            "possible": true,
            "content": "Searching Redis configuration.",
            "tool_calls": [{"name":"search","arguments":{"query":"redis"}}]
        });

        let (passed, reason) = score_repair(&expected, &predicted);

        assert!(passed);
        assert_eq!(reason, "predicted tool calls matched expected repair");
    }
}

#[allow(dead_code)]
fn _assert_correction_row_shapes(
    _tools: Vec<OpenAiTool>,
    _messages: Vec<ChatMessage>,
    _failures: Vec<ResponseFailure>,
) {
}
