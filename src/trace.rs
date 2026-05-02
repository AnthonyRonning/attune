use std::{path::PathBuf, sync::Arc};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::{fs::OpenOptions, io::AsyncWriteExt, sync::Mutex};
use uuid::Uuid;

use crate::{
    model_profile::ModelProfile,
    normalizer::NormalizedRequest,
    openai::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage},
    prompt_adapter::AdaptedRequest,
    repair::{CorrectionAttemptTrace, PolicyDecisionTrace, RepairAction},
    response_interpreter::InterpretedResponse,
};

#[derive(Clone)]
pub struct TraceStore {
    path: PathBuf,
    lock: Arc<Mutex<()>>,
    enabled: bool,
}

impl TraceStore {
    pub fn new(path: PathBuf, enabled: bool) -> Self {
        Self {
            path,
            lock: Arc::new(Mutex::new(())),
            enabled,
        }
    }

    pub async fn append(&self, mut record: TraceRecord) -> Result<String> {
        let trace_id = record.trace_id.clone();
        if !self.enabled {
            return Ok(trace_id);
        }

        if let Some(parent) = self.path.parent() {
            tokio::fs::create_dir_all(parent).await.with_context(|| {
                format!("failed to create trace directory {}", parent.display())
            })?;
        }

        let _guard = self.lock.lock().await;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .await
            .with_context(|| format!("failed to open trace file {}", self.path.display()))?;
        record.completed_at = Some(Utc::now());
        record.summary = Some(TraceSummary::from_record(&record));
        let mut line = serde_json::to_vec(&record)?;
        line.push(b'\n');
        file.write_all(&line).await?;
        Ok(trace_id)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceRecord {
    pub trace_id: String,
    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
    pub request: ChatCompletionRequest,
    pub normalized: Option<NormalizedRequest>,
    pub profile: Option<ModelProfile>,
    pub adapted_request: Option<AdaptedRequest>,
    pub upstream_response: Option<ChatCompletionResponse>,
    pub interpreted: Option<InterpretedResponse>,
    #[serde(default)]
    pub correction_attempts: Vec<CorrectionAttemptTrace>,
    #[serde(default)]
    pub policy_decisions: Vec<PolicyDecisionTrace>,
    pub repair_actions: Vec<RepairAction>,
    pub final_response: Option<ChatCompletionResponse>,
    pub error: Option<String>,
    pub metadata: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub summary: Option<TraceSummary>,
}

impl TraceRecord {
    pub fn new(request: ChatCompletionRequest, metadata: Value) -> Self {
        Self {
            trace_id: format!("trace_{}", Uuid::new_v4().simple()),
            started_at: Utc::now(),
            completed_at: None,
            request,
            normalized: None,
            profile: None,
            adapted_request: None,
            upstream_response: None,
            interpreted: None,
            correction_attempts: Vec::new(),
            policy_decisions: Vec::new(),
            repair_actions: Vec::new(),
            final_response: None,
            error: None,
            metadata,
            summary: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceSummary {
    pub trace_id: String,
    pub model: String,
    pub profile: Option<String>,
    pub adapter_mode: Option<String>,
    pub latest_user: Option<String>,
    pub request_messages: usize,
    pub request_tools: usize,
    pub upstream_finish_reason: Option<String>,
    pub upstream_content_len: usize,
    pub upstream_content_preview: Option<String>,
    pub upstream_reasoning_len: usize,
    pub interpreted_content_len: usize,
    pub interpreted_content_preview: Option<String>,
    pub parse_events: Vec<String>,
    #[serde(default)]
    pub failure_kinds: Vec<String>,
    pub suspicious_stop: Option<bool>,
    pub tool_intents: Vec<String>,
    #[serde(default)]
    pub correction_attempts: Vec<String>,
    #[serde(default)]
    pub policy_decisions: Vec<String>,
    pub repair_actions: Vec<String>,
    pub final_finish_reason: Option<String>,
    pub final_content_len: usize,
    pub final_content_preview: Option<String>,
    pub final_tool_calls: usize,
    pub error: Option<String>,
}

impl TraceSummary {
    pub fn from_record(record: &TraceRecord) -> Self {
        let upstream_message = record
            .upstream_response
            .as_ref()
            .and_then(|response| response.choices.first())
            .map(|choice| &choice.message);
        let upstream_content = upstream_message.and_then(ChatMessage::content_text);
        let upstream_reasoning = upstream_message.and_then(ChatMessage::reasoning_text);
        let interpreted_content = record
            .interpreted
            .as_ref()
            .and_then(|interpreted| interpreted.content.clone());
        let final_message = record
            .final_response
            .as_ref()
            .and_then(|response| response.choices.first())
            .map(|choice| &choice.message);
        let final_content = final_message.and_then(ChatMessage::content_text);

        Self {
            trace_id: record.trace_id.clone(),
            model: record.request.model.clone(),
            profile: record.profile.as_ref().map(|profile| profile.name.clone()),
            adapter_mode: record
                .adapted_request
                .as_ref()
                .map(|adapted| format!("{:?}", adapted.mode)),
            latest_user: latest_user_text(&record.request.messages).map(|text| preview(&text)),
            request_messages: record.request.messages.len(),
            request_tools: record.request.tools.as_ref().map_or(0, Vec::len),
            upstream_finish_reason: record
                .upstream_response
                .as_ref()
                .and_then(first_finish_reason),
            upstream_content_len: upstream_content.as_ref().map_or(0, String::len),
            upstream_content_preview: upstream_content.as_deref().map(preview),
            upstream_reasoning_len: upstream_reasoning.as_ref().map_or(0, String::len),
            interpreted_content_len: interpreted_content.as_ref().map_or(0, String::len),
            interpreted_content_preview: interpreted_content.as_deref().map(preview),
            parse_events: record
                .interpreted
                .as_ref()
                .map(|interpreted| interpreted.parse_events.clone())
                .unwrap_or_default(),
            failure_kinds: record
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
            suspicious_stop: record
                .interpreted
                .as_ref()
                .map(|interpreted| interpreted.suspicious_stop),
            tool_intents: record
                .interpreted
                .as_ref()
                .map(|interpreted| {
                    interpreted
                        .tool_intents
                        .iter()
                        .map(|intent| format!("{}:{:?}", intent.name, intent.source))
                        .collect()
                })
                .unwrap_or_default(),
            correction_attempts: record
                .correction_attempts
                .iter()
                .map(|attempt| {
                    format!(
                        "{}:{} accepted={} confidence={:?}",
                        attempt.correction_model,
                        attempt.result,
                        attempt.accepted,
                        attempt.confidence
                    )
                })
                .collect(),
            policy_decisions: record
                .policy_decisions
                .iter()
                .map(|decision| format!("{}:{}", decision.stage, decision.decision))
                .collect(),
            repair_actions: record
                .repair_actions
                .iter()
                .map(|action| action.action.clone())
                .collect(),
            final_finish_reason: record.final_response.as_ref().and_then(first_finish_reason),
            final_content_len: final_content.as_ref().map_or(0, String::len),
            final_content_preview: final_content.as_deref().map(preview),
            final_tool_calls: final_message
                .and_then(|message| message.tool_calls.as_ref())
                .map_or(0, Vec::len),
            error: record.error.clone(),
        }
    }
}

pub fn trace_summaries(records: &[TraceRecord], limit: usize) -> Vec<TraceSummary> {
    let mut summaries = records
        .iter()
        .rev()
        .take(limit)
        .map(|record| {
            record
                .summary
                .clone()
                .unwrap_or_else(|| TraceSummary::from_record(record))
        })
        .collect::<Vec<_>>();
    summaries.reverse();
    summaries
}

pub async fn read_trace_records(path: &PathBuf) -> Result<Vec<TraceRecord>> {
    let content = match tokio::fs::read_to_string(path).await {
        Ok(content) => content,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => {
            return Err(error).with_context(|| format!("failed to read {}", path.display()))
        }
    };

    let mut records = Vec::new();
    for (line_no, line) in content.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let record = serde_json::from_str::<TraceRecord>(line)
            .with_context(|| format!("failed to parse trace line {}", line_no + 1))?;
        records.push(record);
    }
    Ok(records)
}

fn latest_user_text(messages: &[ChatMessage]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .and_then(ChatMessage::content_text)
}

fn first_finish_reason(response: &ChatCompletionResponse) -> Option<String> {
    response
        .choices
        .first()
        .and_then(|choice| choice.finish_reason.clone())
}

fn preview(text: &str) -> String {
    let normalized = text.split_whitespace().collect::<Vec<_>>().join(" ");
    const MAX: usize = 240;
    if normalized.chars().count() <= MAX {
        return normalized;
    }

    let mut out = normalized.chars().take(MAX).collect::<String>();
    out.push_str("...");
    out
}
