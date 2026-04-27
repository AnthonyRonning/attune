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
    openai::{ChatCompletionRequest, ChatCompletionResponse},
    prompt_adapter::AdaptedRequest,
    repair::RepairAction,
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
    pub repair_actions: Vec<RepairAction>,
    pub final_response: Option<ChatCompletionResponse>,
    pub error: Option<String>,
    pub metadata: Value,
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
            repair_actions: Vec::new(),
            final_response: None,
            error: None,
            metadata,
        }
    }
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
