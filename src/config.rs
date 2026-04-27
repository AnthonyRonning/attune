use std::{path::PathBuf, time::Duration};

use serde::{Deserialize, Serialize};

use crate::{model_profile::ModelProfile, policy::PolicyConfig};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProxyConfig {
    pub upstream: UpstreamConfig,
    pub correction: CorrectionConfig,
    pub trace: TraceConfig,
    pub policy: PolicyConfig,
    #[serde(default)]
    pub model_profiles: Vec<ModelProfile>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpstreamConfig {
    pub base_url: String,
    pub api_key: Option<String>,
    pub timeout_seconds: u64,
}

impl UpstreamConfig {
    pub fn timeout(&self) -> Duration {
        Duration::from_secs(self.timeout_seconds)
    }
}

impl Default for UpstreamConfig {
    fn default() -> Self {
        Self {
            base_url: "https://openrouter.ai/api/v1".to_string(),
            api_key: None,
            timeout_seconds: 120,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionConfig {
    pub enabled: bool,
    pub default_model: Option<String>,
    pub max_context_messages: usize,
    pub min_confidence: f32,
}

impl Default for CorrectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_model: None,
            max_context_messages: 12,
            min_confidence: 0.65,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceConfig {
    pub enabled: bool,
    pub path: PathBuf,
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            path: PathBuf::from("traces/model-correction-proxy.jsonl"),
        }
    }
}
