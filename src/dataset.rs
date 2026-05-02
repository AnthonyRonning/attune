use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::{fs::OpenOptions, io::AsyncWriteExt};

use crate::trace::read_trace_records;

#[derive(Debug, Clone)]
pub struct DatasetExportConfig {
    pub trace_path: PathBuf,
    pub output_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetExportReport {
    pub traces_seen: usize,
    pub rows_written: usize,
    pub output_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionDatasetRow {
    pub trace_id: String,
    pub model: String,
    pub profile: String,
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

pub async fn export_dataset(config: DatasetExportConfig) -> Result<()> {
    let report = export_dataset_file(config).await?;
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

        let row = CorrectionDatasetRow {
            trace_id: record.trace_id.clone(),
            model: normalized.model.clone(),
            profile: record
                .profile
                .as_ref()
                .map(|profile| profile.name.clone())
                .unwrap_or_else(|| "unknown".to_string()),
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
        output_path: config.output_path,
    })
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
        openai::{ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage},
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
        })
        .await
        .unwrap();

        assert_eq!(report.rows_written, 1);
        assert!(tokio::fs::read_to_string(output_path)
            .await
            .unwrap()
            .contains("bad json"));
    }
}
