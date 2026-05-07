use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::{fs::OpenOptions, io::AsyncWriteExt};

use crate::{
    model_profile::{DsrsHistoryFormat, ProfileMetadata},
    openai::{ChatMessage, OpenAiToolCall},
    trace::{read_trace_records, TraceRecord},
};

#[derive(Debug, Clone)]
pub struct DatasetExportConfig {
    pub trace_path: PathBuf,
    pub output_path: PathBuf,
    pub filter: DatasetExportFilter,
}

#[derive(Debug, Clone, Default)]
pub struct DatasetExportFilter {
    pub trace_ids: Vec<String>,
    pub model: Option<String>,
    pub profile: Option<String>,
    pub failure_kinds: Vec<String>,
    pub repair_actions: Vec<String>,
    pub correction_results: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct RequestAdapterDatasetExportConfig {
    pub trace_path: PathBuf,
    pub output_path: PathBuf,
    pub filter: DatasetExportFilter,
    pub expected_output: Option<Value>,
    pub use_final_response: bool,
    pub allow_unlabeled: bool,
    pub append: bool,
    pub observed_failure_kind: Option<String>,
    pub observed_problem: Option<String>,
    pub prompt_goal: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetExportReport {
    pub traces_seen: usize,
    pub rows_written: usize,
    pub rows_skipped_by_filter: usize,
    pub output_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestAdapterDatasetExportReport {
    pub traces_seen: usize,
    pub rows_written: usize,
    pub rows_skipped_by_filter: usize,
    pub rows_unlabeled: usize,
    pub output_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionDatasetRow {
    pub trace_id: String,
    pub model: String,
    pub profile: String,
    pub profile_revision: Option<u32>,
    pub profile_source: Option<String>,
    pub request_adapter_artifact: Option<String>,
    pub correction_agent_artifact: Option<String>,
    pub available_tools: Value,
    pub recent_messages: Value,
    pub malformed_response: Value,
    pub parser_events: Value,
    pub response_failures: Value,
    pub upstream_assistant_content: Value,
    pub upstream_assistant_reasoning: Value,
    pub interpreted_content: Value,
    pub correction_attempts: Value,
    pub expected_repair: Value,
    pub repair_actions: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RequestAdapterDatasetRow {
    pub dataset_type: String,
    pub trace_id: String,
    pub started_at: String,
    pub completed_at: Option<String>,
    pub model: String,
    pub profile: String,
    pub profile_revision: Option<u32>,
    pub profile_source: Option<String>,
    pub request_adapter_artifact: Option<String>,
    pub correction_agent_artifact: Option<String>,
    pub dsrs_history_format: Option<DsrsHistoryFormat>,
    pub request: Value,
    pub messages: Value,
    pub system_context: String,
    pub conversation: String,
    pub available_tools: Value,
    pub tool_choice: Value,
    pub parallel_tool_calls: bool,
    pub adapted_request: Option<Value>,
    pub observed_upstream_content: Value,
    pub observed_upstream_reasoning: Value,
    pub observed_finish_reason: Option<String>,
    pub observed_failure_kinds: Vec<String>,
    pub observed_problem: Option<String>,
    pub observed_failure_kind: Option<String>,
    pub expected_output: Value,
    pub expected_adapter_policy: Value,
}

pub async fn export_dataset(config: DatasetExportConfig) -> Result<()> {
    let report = export_dataset_file(config).await?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

pub async fn export_request_adapter_dataset(
    config: RequestAdapterDatasetExportConfig,
) -> Result<()> {
    let report = export_request_adapter_dataset_file(config).await?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

pub async fn export_dataset_file(config: DatasetExportConfig) -> Result<DatasetExportReport> {
    let records = read_trace_records(&config.trace_path).await?;
    if let Some(parent) = config.output_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let mut file = OpenOptions::new()
        .create(true)
        .truncate(true)
        .write(true)
        .open(&config.output_path)
        .await
        .with_context(|| format!("failed to open {}", config.output_path.display()))?;

    let mut rows_written = 0usize;
    let mut rows_skipped_by_filter = 0usize;
    for record in &records {
        let Some(normalized) = record.normalized.as_ref() else {
            continue;
        };
        let Some(interpreted) = record.interpreted.as_ref() else {
            continue;
        };
        let Some(final_response) = record.final_response.as_ref() else {
            continue;
        };
        if record.repair_actions.is_empty() && interpreted.tool_intents.is_empty() {
            continue;
        }
        let profile_name = record
            .profile
            .as_ref()
            .map(|profile| profile.name.clone())
            .unwrap_or_else(|| "unknown".to_string());
        if !record_matches_filter(
            &config.filter,
            record,
            normalized,
            &profile_name,
            interpreted,
        ) {
            rows_skipped_by_filter += 1;
            continue;
        }
        let metadata = record_profile_metadata(record);

        let row = CorrectionDatasetRow {
            trace_id: record.trace_id.clone(),
            model: normalized.model.clone(),
            profile: profile_name,
            profile_revision: metadata.as_ref().map(|metadata| metadata.profile_revision),
            profile_source: metadata
                .as_ref()
                .map(|metadata| metadata.profile_source.clone()),
            request_adapter_artifact: metadata
                .as_ref()
                .and_then(|metadata| metadata.request_adapter_artifact.clone()),
            correction_agent_artifact: metadata
                .as_ref()
                .and_then(|metadata| metadata.correction_agent_artifact.clone()),
            available_tools: serde_json::to_value(&normalized.tools)?,
            recent_messages: serde_json::to_value(&normalized.messages)?,
            malformed_response: malformed_response_value(record, interpreted),
            parser_events: serde_json::to_value(&interpreted.parse_events)?,
            response_failures: serde_json::to_value(&interpreted.failures)?,
            upstream_assistant_content: upstream_assistant_content(record)
                .map(Value::String)
                .unwrap_or(Value::Null),
            upstream_assistant_reasoning: upstream_assistant_reasoning(record)
                .map(Value::String)
                .unwrap_or(Value::Null),
            interpreted_content: interpreted
                .content
                .clone()
                .map(Value::String)
                .unwrap_or(Value::Null),
            correction_attempts: serde_json::to_value(&record.correction_attempts)?,
            expected_repair: json!({
                "final_message": final_response.choices.first().map(|choice| &choice.message),
                "tool_calls": final_response
                    .choices
                    .first()
                    .and_then(|choice| choice.message.tool_calls.as_ref())
                    .cloned()
                    .unwrap_or_default()
            }),
            repair_actions: serde_json::to_value(&record.repair_actions)?,
        };

        let mut line = serde_json::to_vec(&row)?;
        line.push(b'\n');
        file.write_all(&line).await?;
        rows_written += 1;
    }

    Ok(DatasetExportReport {
        traces_seen: records.len(),
        rows_written,
        rows_skipped_by_filter,
        output_path: config.output_path,
    })
}

pub async fn export_request_adapter_dataset_file(
    config: RequestAdapterDatasetExportConfig,
) -> Result<RequestAdapterDatasetExportReport> {
    let records = read_trace_records(&config.trace_path).await?;
    if let Some(parent) = config.output_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let mut options = OpenOptions::new();
    options.create(true).write(true);
    if config.append {
        options.append(true);
    } else {
        options.truncate(true);
    }
    let mut file = options
        .open(&config.output_path)
        .await
        .with_context(|| format!("failed to open {}", config.output_path.display()))?;

    let mut rows_written = 0usize;
    let mut rows_skipped_by_filter = 0usize;
    let mut rows_unlabeled = 0usize;
    for record in &records {
        if !request_adapter_record_matches_filter(&config.filter, record) {
            rows_skipped_by_filter += 1;
            continue;
        }
        let expected_output = request_adapter_expected_output(&config, record)?;
        if expected_output.is_null() {
            rows_unlabeled += 1;
        }

        let row = request_adapter_dataset_row(&config, record, expected_output)?;
        let mut line = serde_json::to_vec(&row)?;
        line.push(b'\n');
        file.write_all(&line).await?;
        rows_written += 1;
    }

    Ok(RequestAdapterDatasetExportReport {
        traces_seen: records.len(),
        rows_written,
        rows_skipped_by_filter,
        rows_unlabeled,
        output_path: config.output_path,
    })
}

fn record_matches_filter(
    filter: &DatasetExportFilter,
    record: &TraceRecord,
    normalized: &crate::normalizer::NormalizedRequest,
    profile_name: &str,
    interpreted: &crate::response_interpreter::InterpretedResponse,
) -> bool {
    if !matches_trace_id_filter(&record.trace_id, &filter.trace_ids) {
        return false;
    }
    if !matches_optional_text_filter(&normalized.model, filter.model.as_deref()) {
        return false;
    }
    if !matches_optional_text_filter(profile_name, filter.profile.as_deref()) {
        return false;
    }
    if !matches_any_normalized(
        interpreted
            .failures
            .iter()
            .map(|failure| format!("{:?}", failure.kind)),
        &filter.failure_kinds,
    ) {
        return false;
    }
    if !matches_any_normalized(
        record
            .repair_actions
            .iter()
            .map(|action| action.action.clone()),
        &filter.repair_actions,
    ) {
        return false;
    }
    matches_any_normalized(
        record
            .correction_attempts
            .iter()
            .map(|attempt| attempt.result.clone()),
        &filter.correction_results,
    )
}

fn request_adapter_record_matches_filter(
    filter: &DatasetExportFilter,
    record: &TraceRecord,
) -> bool {
    let profile_name = record
        .profile
        .as_ref()
        .map(|profile| profile.name.as_str())
        .or_else(|| {
            record
                .profile_metadata
                .as_ref()
                .map(|metadata| metadata.profile_id.as_str())
        })
        .unwrap_or("unknown");
    let failure_kinds = record
        .interpreted
        .as_ref()
        .map(|interpreted| {
            interpreted
                .failures
                .iter()
                .map(|failure| format!("{:?}", failure.kind))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    matches_trace_id_filter(&record.trace_id, &filter.trace_ids)
        && matches_optional_text_filter(&record.request.model, filter.model.as_deref())
        && matches_optional_text_filter(profile_name, filter.profile.as_deref())
        && matches_any_normalized(failure_kinds, &filter.failure_kinds)
        && matches_any_normalized(
            record
                .repair_actions
                .iter()
                .map(|action| action.action.clone()),
            &filter.repair_actions,
        )
        && matches_any_normalized(
            record
                .correction_attempts
                .iter()
                .map(|attempt| attempt.result.clone()),
            &filter.correction_results,
        )
}

fn matches_trace_id_filter(actual: &str, trace_ids: &[String]) -> bool {
    trace_ids.is_empty() || trace_ids.iter().any(|trace_id| trace_id == actual)
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

fn matches_any_normalized(values: impl IntoIterator<Item = String>, filters: &[String]) -> bool {
    if filters.is_empty() {
        return true;
    }
    let values = values
        .into_iter()
        .map(|value| normalize_filter_value(&value))
        .collect::<Vec<_>>();
    filters.iter().any(|filter| {
        let filter = normalize_filter_value(filter);
        values.iter().any(|value| value == &filter)
    })
}

fn normalize_filter_value(value: &str) -> String {
    value
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

fn record_profile_metadata(record: &crate::trace::TraceRecord) -> Option<ProfileMetadata> {
    record
        .profile_metadata
        .clone()
        .or_else(|| record.profile.as_ref().map(|profile| profile.metadata()))
}

fn request_adapter_dataset_row(
    config: &RequestAdapterDatasetExportConfig,
    record: &TraceRecord,
    expected_output: Value,
) -> Result<RequestAdapterDatasetRow> {
    let metadata = record_profile_metadata(record);
    let profile_name = metadata
        .as_ref()
        .map(|metadata| metadata.profile_id.clone())
        .or_else(|| record.profile.as_ref().map(|profile| profile.name.clone()))
        .unwrap_or_else(|| "unknown".to_string());
    let messages = serde_json::to_value(&record.request.messages)?;
    let request = serde_json::to_value(&record.request)?;
    let available_tools =
        serde_json::to_value(record.request.tools.as_ref().cloned().unwrap_or_default())?;
    let tool_choice = record
        .request
        .tool_choice
        .clone()
        .unwrap_or_else(|| Value::String("auto".to_string()));
    let parallel_tool_calls = record.request.parallel_tool_calls.unwrap_or(true);

    Ok(RequestAdapterDatasetRow {
        dataset_type: "request_adapter_prompt_gepa/v1".to_string(),
        trace_id: record.trace_id.clone(),
        started_at: record.started_at.to_rfc3339(),
        completed_at: record.completed_at.map(|time| time.to_rfc3339()),
        model: record.request.model.clone(),
        profile: profile_name,
        profile_revision: metadata.as_ref().map(|metadata| metadata.profile_revision),
        profile_source: metadata
            .as_ref()
            .map(|metadata| metadata.profile_source.clone()),
        request_adapter_artifact: metadata
            .as_ref()
            .and_then(|metadata| metadata.request_adapter_artifact.clone()),
        correction_agent_artifact: metadata
            .as_ref()
            .and_then(|metadata| metadata.correction_agent_artifact.clone()),
        dsrs_history_format: metadata
            .as_ref()
            .map(|metadata| metadata.dsrs_history_format),
        request,
        messages: messages.clone(),
        system_context: render_dataset_system_context(&record.request.messages)?,
        conversation: render_dataset_conversation(&record.request.messages)?,
        available_tools,
        tool_choice,
        parallel_tool_calls,
        adapted_request: record
            .adapted_request
            .as_ref()
            .map(serde_json::to_value)
            .transpose()?,
        observed_upstream_content: upstream_assistant_content(record)
            .map(Value::String)
            .unwrap_or(Value::Null),
        observed_upstream_reasoning: upstream_assistant_reasoning(record)
            .map(Value::String)
            .unwrap_or(Value::Null),
        observed_finish_reason: record
            .upstream_response
            .as_ref()
            .and_then(|response| response.choices.first())
            .and_then(|choice| choice.finish_reason.clone()),
        observed_failure_kinds: record
            .interpreted
            .as_ref()
            .map(|interpreted| {
                interpreted
                    .failures
                    .iter()
                    .map(|failure| format!("{:?}", failure.kind))
                    .collect()
            })
            .unwrap_or_default(),
        observed_problem: config.observed_problem.clone(),
        observed_failure_kind: config.observed_failure_kind.clone(),
        expected_output,
        expected_adapter_policy: request_adapter_policy_value(config),
    })
}

fn request_adapter_expected_output(
    config: &RequestAdapterDatasetExportConfig,
    record: &TraceRecord,
) -> Result<Value> {
    if let Some(expected) = &config.expected_output {
        return Ok(expected.clone());
    }
    if config.use_final_response {
        return final_response_expected_output(record);
    }
    if config.allow_unlabeled {
        return Ok(Value::Null);
    }
    anyhow::bail!(
        "request-adapter dataset export requires --expected-output-json, --expected-output-path, --use-final-response, or --allow-unlabeled"
    );
}

fn final_response_expected_output(record: &TraceRecord) -> Result<Value> {
    let Some(message) = record
        .final_response
        .as_ref()
        .and_then(|response| response.choices.first())
        .map(|choice| &choice.message)
    else {
        anyhow::bail!(
            "trace {} did not contain a final response to use as expected output",
            record.trace_id
        );
    };

    Ok(json!({
        "content": message.content_text().unwrap_or_default(),
        "tool_calls": message
            .tool_calls
            .as_deref()
            .map(openai_tool_calls_to_request_adapter_calls)
            .transpose()?
            .unwrap_or_default()
    }))
}

fn openai_tool_calls_to_request_adapter_calls(calls: &[OpenAiToolCall]) -> Result<Vec<Value>> {
    calls
        .iter()
        .map(|call| {
            let arguments = serde_json::from_str::<Value>(&call.function.arguments)
                .unwrap_or_else(|_| Value::String(call.function.arguments.clone()));
            Ok(json!({
                "name": call.function.name,
                "arguments": arguments
            }))
        })
        .collect()
}

fn request_adapter_policy_value(config: &RequestAdapterDatasetExportConfig) -> Value {
    let mut policy = serde_json::Map::new();
    if let Some(prompt_goal) = &config.prompt_goal {
        policy.insert(
            "prompt_goal".to_string(),
            Value::String(prompt_goal.clone()),
        );
    }
    Value::Object(policy)
}

fn render_dataset_system_context(messages: &[ChatMessage]) -> Result<String> {
    let system_messages = messages
        .iter()
        .filter(|message| message.role == "system" || message.role == "developer")
        .collect::<Vec<_>>();
    if system_messages.is_empty() {
        return Ok("No system or developer messages.".to_string());
    }

    let mut rendered = String::new();
    for (index, message) in system_messages.iter().enumerate() {
        rendered.push_str(&format!("[{index}] role: {}\n", message.role));
        if let Some(name) = &message.name {
            rendered.push_str(&format!("name: {name}\n"));
        }
        rendered.push_str("content:\n");
        rendered.push_str(&render_dataset_message_content(message)?);
        rendered.push('\n');
    }
    Ok(rendered.trim_end().to_string())
}

fn render_dataset_conversation(messages: &[ChatMessage]) -> Result<String> {
    let conversation = messages
        .iter()
        .filter(|message| message.role != "system" && message.role != "developer")
        .collect::<Vec<_>>();
    if conversation.is_empty() {
        return Ok("No non-system conversation messages.".to_string());
    }

    let mut rendered = String::new();
    for (index, message) in conversation.iter().enumerate() {
        rendered.push_str(&format!("[{index}] role: {}\n", message.role));
        if let Some(name) = &message.name {
            rendered.push_str(&format!("name: {name}\n"));
        }
        if let Some(tool_call_id) = &message.tool_call_id {
            rendered.push_str(&format!("tool_call_id: {tool_call_id}\n"));
        }
        rendered.push_str("content:\n");
        rendered.push_str(&render_dataset_message_content(message)?);
        rendered.push('\n');
        if let Some(tool_calls) = &message.tool_calls {
            rendered.push_str("assistant_tool_calls:\n");
            rendered.push_str(&serde_json::to_string_pretty(
                &openai_tool_calls_to_request_adapter_calls(tool_calls)?,
            )?);
            rendered.push('\n');
        }
        rendered.push('\n');
    }
    Ok(rendered.trim_end().to_string())
}

fn render_dataset_message_content(message: &ChatMessage) -> Result<String> {
    match &message.content {
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Array(parts)) => {
            let text = parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<String>();
            if text.is_empty() {
                Ok(serde_json::to_string_pretty(parts)?)
            } else {
                Ok(text)
            }
        }
        Some(Value::Null) | None => Ok(String::new()),
        Some(value) => Ok(serde_json::to_string_pretty(value)?),
    }
}

fn malformed_response_value(
    record: &crate::trace::TraceRecord,
    interpreted: &crate::response_interpreter::InterpretedResponse,
) -> Value {
    let content = upstream_assistant_content(record)
        .or_else(|| interpreted.content.clone())
        .unwrap_or_default();
    let reasoning = upstream_assistant_reasoning(record).or_else(|| interpreted.reasoning.clone());

    if let Some(reasoning) = reasoning.filter(|reasoning| !reasoning.trim().is_empty()) {
        Value::String(format!(
            "assistant_content:\n{content}\n\nassistant_reasoning:\n{reasoning}"
        ))
    } else {
        Value::String(content)
    }
}

fn upstream_assistant_content(record: &crate::trace::TraceRecord) -> Option<String> {
    record
        .upstream_response
        .as_ref()
        .and_then(|response| response.choices.first())
        .and_then(|choice| choice.message.content_text())
}

fn upstream_assistant_reasoning(record: &crate::trace::TraceRecord) -> Option<String> {
    record
        .upstream_response
        .as_ref()
        .and_then(|response| response.choices.first())
        .and_then(|choice| choice.message.reasoning_text())
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Map};

    use super::*;
    use crate::{
        openai::{
            ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage,
            OpenAiFunctionTool, OpenAiTool,
        },
        repair::RepairAction,
        response_interpreter::InterpretedResponse,
        trace::TraceRecord,
    };

    #[tokio::test]
    async fn exports_repaired_trace_rows() {
        let dir = tempfile::tempdir().unwrap();
        let trace_path = dir.path().join("trace.jsonl");
        let output_path = dir.path().join("dataset.jsonl");
        let request = ChatCompletionRequest {
            model: "test".to_string(),
            messages: vec![ChatMessage::new("user", "hello")],
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            stream: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            max_completion_tokens: None,
            response_format: None,
            extra: Map::new(),
        };
        let mut record = TraceRecord::new(request.clone(), json!({}));
        record.normalized = Some(crate::normalizer::normalize_request(request).unwrap());
        record.interpreted = Some(InterpretedResponse {
            content: Some("bad json".to_string()),
            reasoning: None,
            finish_reason: Some("stop".to_string()),
            tool_intents: Vec::new(),
            parse_events: vec!["bad".to_string()],
            failures: Vec::new(),
            suspicious_stop: false,
        });
        let mut final_response = ChatCompletionResponse::empty_for_model("test");
        final_response.choices.push(ChatChoice {
            index: 0,
            message: ChatMessage::new("assistant", "fixed"),
            finish_reason: Some("stop".to_string()),
            logprobs: None,
            extra: Map::new(),
        });
        record.final_response = Some(final_response);
        record.repair_actions = vec![RepairAction {
            action: "x".to_string(),
            confidence: 1.0,
            reason: "test".to_string(),
        }];
        tokio::fs::write(
            &trace_path,
            format!("{}\n", serde_json::to_string(&record).unwrap()),
        )
        .await
        .unwrap();

        let report = export_dataset_file(DatasetExportConfig {
            trace_path,
            output_path: output_path.clone(),
            filter: DatasetExportFilter::default(),
        })
        .await
        .unwrap();

        assert_eq!(report.rows_written, 1);
        assert!(tokio::fs::read_to_string(output_path)
            .await
            .unwrap()
            .contains("bad json"));
    }

    #[tokio::test]
    async fn filters_dataset_rows_by_model_profile_and_repair_action() {
        let dir = tempfile::tempdir().unwrap();
        let trace_path = dir.path().join("trace.jsonl");
        let output_path = dir.path().join("dataset.jsonl");
        let request = ChatCompletionRequest {
            model: "qwen/qwen3.5-9b".to_string(),
            messages: vec![ChatMessage::new("user", "hello")],
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            stream: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            max_completion_tokens: None,
            response_format: None,
            extra: Map::new(),
        };
        let mut record = TraceRecord::new(request.clone(), json!({}));
        record.normalized = Some(crate::normalizer::normalize_request(request).unwrap());
        record.profile = Some(crate::model_profile::ModelProfile::qwen());
        record.profile_metadata = record.profile.as_ref().map(|profile| profile.metadata());
        record.interpreted = Some(InterpretedResponse {
            content: Some("bad json".to_string()),
            reasoning: None,
            finish_reason: Some("stop".to_string()),
            tool_intents: Vec::new(),
            parse_events: vec!["bad".to_string()],
            failures: Vec::new(),
            suspicious_stop: false,
        });
        let mut final_response = ChatCompletionResponse::empty_for_model("test");
        final_response.choices.push(ChatChoice {
            index: 0,
            message: ChatMessage::new("assistant", "fixed"),
            finish_reason: Some("stop".to_string()),
            logprobs: None,
            extra: Map::new(),
        });
        record.final_response = Some(final_response);
        record.repair_actions = vec![RepairAction {
            action: "content_tool_call_extracted".to_string(),
            confidence: 1.0,
            reason: "test".to_string(),
        }];
        tokio::fs::write(
            &trace_path,
            format!("{}\n", serde_json::to_string(&record).unwrap()),
        )
        .await
        .unwrap();

        let report = export_dataset_file(DatasetExportConfig {
            trace_path,
            output_path: output_path.clone(),
            filter: DatasetExportFilter {
                model: Some("qwen".to_string()),
                profile: Some("qwen-dsrs".to_string()),
                repair_actions: vec!["content_tool_call_extracted".to_string()],
                ..DatasetExportFilter::default()
            },
        })
        .await
        .unwrap();

        assert_eq!(report.rows_written, 1);
        let exported = tokio::fs::read_to_string(output_path).await.unwrap();
        let exported_row: Value = serde_json::from_str(exported.lines().next().unwrap()).unwrap();
        assert_eq!(exported_row["profile_revision"], json!(1));
        assert_eq!(exported_row["profile_source"], json!("builtin"));
    }

    #[tokio::test]
    async fn exports_request_adapter_rows_from_exact_trace_request() {
        let dir = tempfile::tempdir().unwrap();
        let trace_path = dir.path().join("trace.jsonl");
        let output_path = dir.path().join("request-adapter.jsonl");
        let request = ChatCompletionRequest {
            model: "google/gemma-4-26b-a4b-it".to_string(),
            messages: vec![
                ChatMessage::new("system", "Use tools for project questions."),
                ChatMessage::new("user", "tell me about the packages"),
            ],
            tools: Some(vec![OpenAiTool {
                tool_type: "function".to_string(),
                function: OpenAiFunctionTool {
                    name: "read".to_string(),
                    description: Some("Read a file".to_string()),
                    parameters: json!({
                        "type": "object",
                        "required": ["path"],
                        "properties": {"path": {"type": "string"}}
                    }),
                },
            }]),
            tool_choice: Some(json!("auto")),
            parallel_tool_calls: Some(false),
            stream: Some(true),
            temperature: Some(0.1),
            top_p: None,
            max_tokens: None,
            max_completion_tokens: None,
            response_format: None,
            extra: Map::new(),
        };
        let mut record = TraceRecord::new(request.clone(), json!({}));
        record.profile = Some(crate::model_profile::ModelProfile::gemma());
        record.profile_metadata = record.profile.as_ref().map(|profile| profile.metadata());
        let mut upstream_response = ChatCompletionResponse::empty_for_model(&request.model);
        upstream_response.choices.push(ChatChoice {
            index: 0,
            message: ChatMessage::new(
                "assistant",
                "[[ ## content ## ]]\n[]\n[[ ## tool_calls ## ]]\n[]\n[[ ## completed ## ]]",
            ),
            finish_reason: Some("stop".to_string()),
            logprobs: None,
            extra: Map::new(),
        });
        record.upstream_response = Some(upstream_response);
        let trace_id = record.trace_id.clone();
        tokio::fs::write(
            &trace_path,
            format!("{}\n", serde_json::to_string(&record).unwrap()),
        )
        .await
        .unwrap();

        let expected_output = json!({
            "content": "",
            "tool_calls": [{"name":"read","arguments":{"path":"packages/coding-agent/docs/packages.md"}}]
        });
        let report = export_request_adapter_dataset_file(RequestAdapterDatasetExportConfig {
            trace_path,
            output_path: output_path.clone(),
            filter: DatasetExportFilter {
                trace_ids: vec![trace_id.clone()],
                ..DatasetExportFilter::default()
            },
            expected_output: Some(expected_output.clone()),
            use_final_response: false,
            allow_unlabeled: false,
            append: false,
            observed_failure_kind: Some("empty_dsrs_output".to_string()),
            observed_problem: Some("model stopped with placeholder output".to_string()),
            prompt_goal: Some("call read instead of stopping empty".to_string()),
        })
        .await
        .unwrap();

        assert_eq!(report.rows_written, 1);
        let exported = tokio::fs::read_to_string(output_path).await.unwrap();
        let row: Value = serde_json::from_str(exported.lines().next().unwrap()).unwrap();
        assert_eq!(row["dataset_type"], json!("request_adapter_prompt_gepa/v1"));
        assert_eq!(row["trace_id"], json!(trace_id));
        assert_eq!(
            row["request"]["messages"],
            serde_json::to_value(request.messages).unwrap()
        );
        assert_eq!(row["messages"], row["request"]["messages"]);
        assert_eq!(row["available_tools"][0]["function"]["name"], json!("read"));
        assert_eq!(row["parallel_tool_calls"], json!(false));
        assert_eq!(row["dsrs_history_format"], json!("append_only"));
        assert_eq!(row["expected_output"], expected_output);
        assert_eq!(
            row["expected_adapter_policy"]["prompt_goal"],
            json!("call read instead of stopping empty")
        );
    }
}
