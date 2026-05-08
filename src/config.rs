use std::{
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::{
    artifacts::extract_instruction_from_path, builtin_defaults, model_profile::ModelProfile,
    policy::PolicyConfig,
};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ProxyConfig {
    #[serde(default)]
    pub upstream: UpstreamConfig,
    #[serde(default)]
    pub correction: CorrectionConfig,
    #[serde(default)]
    pub trace: TraceConfig,
    #[serde(default)]
    pub policy: PolicyConfig,
    #[serde(default)]
    pub model_profiles: Vec<ModelProfile>,
}

impl ProxyConfig {
    pub async fn from_optional_path(path: Option<&Path>) -> Result<Self> {
        match path {
            Some(path) => Self::from_path(path).await,
            None => Ok(Self::default()),
        }
    }

    pub async fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let content = tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("failed to read config file {}", path.display()))?;
        let mut config = parse_config_content(path, &content)?;
        let base_dir = path.parent().unwrap_or_else(|| Path::new("."));
        hydrate_profile_artifacts(&mut config, base_dir).await?;
        for profile in &mut config.model_profiles {
            profile.mark_config_source();
        }
        Ok(config)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpstreamConfig {
    #[serde(default = "default_upstream_base_url")]
    pub base_url: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default = "default_upstream_timeout_seconds")]
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
            base_url: default_upstream_base_url(),
            api_key: None,
            timeout_seconds: default_upstream_timeout_seconds(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionConfig {
    #[serde(default = "default_correction_enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub default_model: Option<String>,
    #[serde(default = "default_max_context_messages")]
    pub max_context_messages: usize,
    #[serde(default = "default_min_confidence")]
    pub min_confidence: f32,
}

impl Default for CorrectionConfig {
    fn default() -> Self {
        Self {
            enabled: default_correction_enabled(),
            default_model: None,
            max_context_messages: default_max_context_messages(),
            min_confidence: default_min_confidence(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceConfig {
    #[serde(default = "default_trace_enabled")]
    pub enabled: bool,
    #[serde(default = "default_trace_path")]
    pub path: PathBuf,
    #[serde(default = "default_correction_trace_path")]
    pub correction_path: PathBuf,
}

impl Default for TraceConfig {
    fn default() -> Self {
        Self {
            enabled: default_trace_enabled(),
            path: default_trace_path(),
            correction_path: default_correction_trace_path(),
        }
    }
}

fn parse_config_content(path: &Path, content: &str) -> Result<ProxyConfig> {
    match path.extension().and_then(|extension| extension.to_str()) {
        Some("toml") => toml::from_str(content).context("failed to parse TOML proxy config"),
        Some("json") => serde_json::from_str(content).context("failed to parse JSON proxy config"),
        Some("json5") => json5::from_str(content).context("failed to parse JSON5 proxy config"),
        _ => toml::from_str(content)
            .or_else(|_| serde_json::from_str(content))
            .or_else(|_| json5::from_str(content))
            .context("failed to parse proxy config as TOML, JSON, or JSON5"),
    }
}

async fn hydrate_profile_artifacts(config: &mut ProxyConfig, base_dir: &Path) -> Result<()> {
    for profile in &mut config.model_profiles {
        if let Some(artifact) = profile.request_adapter_artifact.clone() {
            let instruction = read_instruction_artifact(base_dir, &artifact)
                .await
                .with_context(|| {
                    format!(
                        "failed to load request adapter artifact {} for profile {}",
                        artifact, profile.name
                    )
                })?;
            profile.tool_instruction = instruction;
        }

        if let Some(artifact) = profile.correction_agent_artifact.clone() {
            let instruction = read_instruction_artifact(base_dir, &artifact)
                .await
                .with_context(|| {
                    format!(
                        "failed to load correction agent artifact {} for profile {}",
                        artifact, profile.name
                    )
                })?;
            profile.correction_instruction = Some(instruction);
        }
    }
    Ok(())
}

fn resolve_artifact_path(base_dir: &Path, artifact: &str) -> PathBuf {
    let path = PathBuf::from(artifact);
    if path.is_absolute() {
        path
    } else {
        base_dir.join(path)
    }
}

async fn read_instruction_artifact(base_dir: &Path, artifact: &str) -> Result<String> {
    if let Some(instruction) = builtin_defaults::instruction_for_reference(artifact)? {
        return Ok(instruction);
    }

    let path = resolve_artifact_path(base_dir, artifact);
    let content = tokio::fs::read_to_string(&path)
        .await
        .with_context(|| format!("failed to read instruction artifact {}", path.display()))?;
    extract_instruction_from_path(&path, &content)
}

fn default_upstream_base_url() -> String {
    "https://openrouter.ai/api/v1".to_string()
}

fn default_upstream_timeout_seconds() -> u64 {
    120
}

fn default_correction_enabled() -> bool {
    true
}

fn default_max_context_messages() -> usize {
    12
}

fn default_min_confidence() -> f32 {
    0.65
}

fn default_trace_enabled() -> bool {
    true
}

fn default_trace_path() -> PathBuf {
    PathBuf::from("traces/model-correction-proxy.jsonl")
}

fn default_correction_trace_path() -> PathBuf {
    PathBuf::from("traces/model-correction-proxy-corrections.jsonl")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_profile::{DsrsHistoryFormat, ToolFormat, ToolMode};

    #[test]
    fn parses_partial_toml_config_with_defaults() {
        let config = parse_config_content(
            Path::new("proxy.toml"),
            r#"
                [[model_profiles]]
                name = "custom-default"
                model_patterns = ["custom"]
                [model_profiles.provider]
                ignore = ["venice"]
                allow_fallbacks = false
            "#,
        )
        .unwrap();

        assert_eq!(config.upstream.base_url, "https://openrouter.ai/api/v1");
        assert!(config.correction.enabled);
        assert_eq!(config.model_profiles.len(), 1);
        assert_eq!(config.model_profiles[0].tool_mode, ToolMode::ProxyOwned);
        assert_eq!(config.model_profiles[0].tool_format, ToolFormat::Dsrs);
        assert_eq!(
            config.model_profiles[0].dsrs_history_format,
            DsrsHistoryFormat::AppendOnly
        );
        assert!(config.model_profiles[0]
            .tool_instruction
            .contains("Tool calls are an application contract"));
        let provider = config.model_profiles[0].provider.as_ref().unwrap();
        assert_eq!(provider.ignore, vec!["venice"]);
        assert_eq!(provider.allow_fallbacks, Some(false));
    }

    #[tokio::test]
    async fn loads_config_profiles_and_artifacts_relative_to_file() {
        let dir = tempfile::tempdir().unwrap();
        let request_artifact = dir.path().join("request.json");
        let correction_artifact = dir.path().join("correction.json");
        let config_path = dir.path().join("proxy.toml");
        tokio::fs::write(
            &request_artifact,
            r#"{"best_instruction":"REQUEST ARTIFACT INSTRUCTION"}"#,
        )
        .await
        .unwrap();
        tokio::fs::write(
            &correction_artifact,
            r#"{"best_instruction":"CORRECTION ARTIFACT INSTRUCTION"}"#,
        )
        .await
        .unwrap();
        tokio::fs::write(
            &config_path,
            r#"
                [[model_profiles]]
                name = "artifact-profile"
                model_patterns = ["artifact-model"]
                revision = 3
                dsrs_history_format = "regenerated_context"
                request_adapter_artifact = "request.json"
                correction_agent_artifact = "correction.json"
            "#,
        )
        .await
        .unwrap();

        let config = ProxyConfig::from_path(&config_path).await.unwrap();
        let profile = &config.model_profiles[0];

        assert_eq!(profile.source, "config");
        assert_eq!(profile.revision, 3);
        assert_eq!(
            profile.dsrs_history_format,
            DsrsHistoryFormat::RegeneratedContext
        );
        assert_eq!(profile.tool_instruction, "REQUEST ARTIFACT INSTRUCTION");
        assert_eq!(
            profile.correction_instruction.as_deref(),
            Some("CORRECTION ARTIFACT INSTRUCTION")
        );
    }

    #[tokio::test]
    async fn loads_builtin_artifact_references_from_config() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("proxy.toml");
        tokio::fs::write(
            &config_path,
            r#"
                [[model_profiles]]
                name = "gemma-dsrs-conservative"
                model_patterns = ["gemma"]
                revision = 5
                request_adapter_artifact = "builtin:request-adapter/gemma-dsrs-conservative/r4-append-only-content-tools-json-meta"
            "#,
        )
        .await
        .unwrap();

        let config = ProxyConfig::from_path(&config_path).await.unwrap();
        let profile = &config.model_profiles[0];

        assert_eq!(profile.source, "config");
        assert_eq!(profile.revision, 5);
        assert!(profile
            .tool_instruction
            .contains("MANDATORY INSPECTION FIRST"));
        assert_eq!(
            profile.request_adapter_artifact.as_deref(),
            Some(
                "builtin:request-adapter/gemma-dsrs-conservative/r4-append-only-content-tools-json-meta"
            )
        );
    }
}
