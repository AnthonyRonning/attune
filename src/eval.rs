use std::{collections::BTreeMap, path::PathBuf};

use anyhow::{Context, Result};
use dspy_rs::{example, Example, GEPA};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    agents::NoopCorrectionAgent,
    config::ProxyConfig,
    model_profile::resolve_profile,
    normalizer::normalize_request,
    openai::{ChatCompletionRequest, ChatCompletionResponse},
    repair::repair_response,
    response_interpreter::interpret_response,
};

#[derive(Debug, Clone)]
pub struct RegressionConfig {
    pub suite_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionCase {
    pub name: String,
    pub request: ChatCompletionRequest,
    pub upstream_response: ChatCompletionResponse,
    #[serde(default)]
    pub expected_tool_calls: Vec<ExpectedToolCall>,
    #[serde(default)]
    pub expected_content_contains: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedToolCall {
    pub name: String,
    #[serde(default)]
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionReport {
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub failures: Vec<RegressionFailure>,
    pub metrics: CorrectionMetrics,
    pub by_profile: BTreeMap<String, ProfileMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionFailure {
    pub case: String,
    pub reason: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CorrectionMetrics {
    pub tool_call_recovery_rate: f32,
    pub schema_valid_final_response_rate: f32,
    pub deterministic_repair_count: usize,
    pub correction_agent_count: usize,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProfileMetrics {
    pub total: usize,
    pub passed: usize,
}

pub async fn run_regression_suite(config: RegressionConfig) -> Result<RegressionReport> {
    let content = tokio::fs::read_to_string(&config.suite_path)
        .await
        .with_context(|| format!("failed to read {}", config.suite_path.display()))?;
    let cases = parse_cases(&content)
        .with_context(|| format!("failed to parse {}", config.suite_path.display()))?;
    evaluate_cases(cases).await
}

pub async fn evaluate_cases(cases: Vec<RegressionCase>) -> Result<RegressionReport> {
    let proxy_config = ProxyConfig::default();
    let correction_agent = NoopCorrectionAgent;
    let mut failures = Vec::new();
    let mut by_profile = BTreeMap::<String, ProfileMetrics>::new();
    let mut deterministic_repair_count = 0usize;
    let mut correction_agent_count = 0usize;
    let mut recovered_tool_cases = 0usize;
    let mut expected_tool_cases = 0usize;
    let mut schema_valid = 0usize;

    for case in &cases {
        let normalized = normalize_request(case.request.clone())?;
        let profile = resolve_profile(&normalized.model, &[]);
        let interpreted = interpret_response(&case.upstream_response, &normalized.tools);
        let outcome = repair_response(
            &proxy_config,
            &normalized,
            &profile,
            &case.upstream_response,
            &interpreted,
            &correction_agent,
        )
        .await?;

        deterministic_repair_count += outcome
            .actions
            .iter()
            .filter(|action| {
                action.action.contains("json")
                    || action.action.contains("extracted")
                    || action.action.contains("tool_name")
            })
            .count();
        correction_agent_count += outcome
            .actions
            .iter()
            .filter(|action| action.action.contains("correction_agent"))
            .count();

        let profile_metrics = by_profile.entry(profile.name.clone()).or_default();
        profile_metrics.total += 1;

        let validation = validate_case(case, &outcome.final_response);
        if let Err(reason) = validation {
            failures.push(RegressionFailure {
                case: case.name.clone(),
                reason,
            });
        } else {
            profile_metrics.passed += 1;
        }

        if !case.expected_tool_calls.is_empty() {
            expected_tool_cases += 1;
            if outcome
                .final_response
                .choices
                .first()
                .and_then(|choice| choice.message.tool_calls.as_ref())
                .is_some_and(|calls| !calls.is_empty())
            {
                recovered_tool_cases += 1;
            }
        }

        if serde_json::to_value(&outcome.final_response).is_ok() {
            schema_valid += 1;
        }
    }

    let total = cases.len();
    let failed = failures.len();
    let passed = total.saturating_sub(failed);
    Ok(RegressionReport {
        total,
        passed,
        failed,
        failures,
        metrics: CorrectionMetrics {
            tool_call_recovery_rate: ratio(recovered_tool_cases, expected_tool_cases),
            schema_valid_final_response_rate: ratio(schema_valid, total),
            deterministic_repair_count,
            correction_agent_count,
        },
        by_profile,
    })
}

pub fn correction_gepa_config() -> GEPA {
    GEPA::builder()
        .num_iterations(3)
        .minibatch_size(3)
        .temperature(0.7)
        .track_stats(true)
        .maybe_max_lm_calls(Some(64))
        .build()
}

pub fn traces_to_gepa_examples(dataset_rows: &[Value]) -> Vec<Example> {
    dataset_rows
        .iter()
        .map(|row| {
            example! {
                "available_tools": "input" => row.get("available_tools").cloned().unwrap_or(Value::Null),
                "recent_messages": "input" => row.get("recent_messages").cloned().unwrap_or(Value::Null),
                "malformed_response": "input" => row.get("malformed_response").cloned().unwrap_or(Value::Null),
                "parser_events": "input" => row.get("parser_events").cloned().unwrap_or(Value::Null),
                "expected_repair": "output" => row.get("expected_repair").cloned().unwrap_or(Value::Null)
            }
        })
        .collect()
}

fn parse_cases(content: &str) -> Result<Vec<RegressionCase>> {
    if content.trim_start().starts_with('[') {
        Ok(serde_json::from_str(content)?)
    } else {
        content
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| Ok(serde_json::from_str::<RegressionCase>(line)?))
            .collect()
    }
}

fn validate_case(
    case: &RegressionCase,
    response: &ChatCompletionResponse,
) -> std::result::Result<(), String> {
    let choice = response
        .choices
        .first()
        .ok_or_else(|| "response contained no choices".to_string())?;

    if !case.expected_tool_calls.is_empty() {
        let calls = choice
            .message
            .tool_calls
            .as_ref()
            .ok_or_else(|| "expected tool calls but response had none".to_string())?;
        if calls.len() != case.expected_tool_calls.len() {
            return Err(format!(
                "expected {} tool calls, got {}",
                case.expected_tool_calls.len(),
                calls.len()
            ));
        }
        for (actual, expected) in calls.iter().zip(&case.expected_tool_calls) {
            if actual.function.name != expected.name {
                return Err(format!(
                    "expected tool {}, got {}",
                    expected.name, actual.function.name
                ));
            }
            let actual_args: Value = serde_json::from_str(&actual.function.arguments)
                .map_err(|error| format!("tool arguments were not valid JSON: {error}"))?;
            if expected.arguments != Value::Null && actual_args != expected.arguments {
                return Err(format!(
                    "expected args {}, got {}",
                    expected.arguments, actual_args
                ));
            }
        }
    }

    if let Some(expected) = &case.expected_content_contains {
        let content = choice.message.content_text().unwrap_or_default();
        if !content.contains(expected) {
            return Err(format!(
                "expected content to contain {expected:?}, got {content:?}"
            ));
        }
    }

    Ok(())
}

fn ratio(numerator: usize, denominator: usize) -> f32 {
    if denominator == 0 {
        1.0
    } else {
        numerator as f32 / denominator as f32
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Map};

    use super::*;
    use crate::openai::{ChatChoice, ChatMessage, OpenAiFunctionTool, OpenAiTool};

    #[tokio::test]
    async fn evaluates_xml_regression_case() {
        let case = RegressionCase {
            name: "xml".to_string(),
            request: ChatCompletionRequest {
                model: "qwen".to_string(),
                messages: vec![ChatMessage::new("user", "read")],
                tools: Some(vec![OpenAiTool {
                    tool_type: "function".to_string(),
                    function: OpenAiFunctionTool {
                        name: "read_file".to_string(),
                        description: None,
                        parameters: json!({"type":"object"}),
                    },
                }]),
                tool_choice: None,
                parallel_tool_calls: Some(false),
                stream: None,
                temperature: None,
                top_p: None,
                max_tokens: None,
                max_completion_tokens: None,
                response_format: None,
                extra: Map::new(),
            },
            upstream_response: ChatCompletionResponse {
                id: "1".to_string(),
                object: "chat.completion".to_string(),
                created: 0,
                model: "qwen".to_string(),
                choices: vec![ChatChoice {
                    index: 0,
                    message: ChatMessage::new(
                        "assistant",
                        r#"<tool_call name="read_file">{"path":"x"}</tool_call>"#,
                    ),
                    finish_reason: Some("stop".to_string()),
                    logprobs: None,
                    extra: Map::new(),
                }],
                usage: None,
                extra: Map::new(),
            },
            expected_tool_calls: vec![ExpectedToolCall {
                name: "read_file".to_string(),
                arguments: json!({"path":"x"}),
            }],
            expected_content_contains: None,
        };

        let report = evaluate_cases(vec![case]).await.unwrap();
        assert_eq!(report.passed, 1);
        assert_eq!(report.metrics.tool_call_recovery_rate, 1.0);
    }
}
