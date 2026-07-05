use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};

use crate::builtin_defaults;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolMode {
    ProxyOwned,
    PassThrough,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolFormat {
    Dsrs,
    Xml,
    TaggedJson,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DsrsHistoryFormat {
    AppendOnly,
    RegeneratedContext,
}

impl fmt::Display for DsrsHistoryFormat {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DsrsHistoryFormat::AppendOnly => formatter.write_str("append_only"),
            DsrsHistoryFormat::RegeneratedContext => formatter.write_str("regenerated_context"),
        }
    }
}

impl FromStr for DsrsHistoryFormat {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "append_only" | "append-only" | "appendonly" => Ok(DsrsHistoryFormat::AppendOnly),
            "regenerated_context" | "regenerated-context" | "regeneratedcontext" | "legacy" => {
                Ok(DsrsHistoryFormat::RegeneratedContext)
            }
            other => Err(format!(
                "unsupported DSRs history format {other:?}; expected append_only or regenerated_context"
            )),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    pub name: String,
    #[serde(default)]
    pub model_patterns: Vec<String>,
    #[serde(default = "default_profile_revision")]
    pub revision: u32,
    #[serde(default = "default_profile_source")]
    pub source: String,
    #[serde(default)]
    pub request_adapter_artifact: Option<String>,
    #[serde(default)]
    pub correction_agent_artifact: Option<String>,
    #[serde(default)]
    pub correction_instruction: Option<String>,
    #[serde(default = "default_tool_mode")]
    pub tool_mode: ToolMode,
    #[serde(default = "default_tool_format")]
    pub tool_format: ToolFormat,
    #[serde(default = "default_dsrs_history_format")]
    pub dsrs_history_format: DsrsHistoryFormat,
    #[serde(default)]
    pub correction_model: Option<String>,
    #[serde(default)]
    pub judge_model: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderRouting>,
    #[serde(default = "default_max_correction_passes")]
    pub max_correction_passes: usize,
    #[serde(default = "default_supports_parallel_tool_calls")]
    pub supports_parallel_tool_calls: bool,
    #[serde(default = "default_dsrs_instruction")]
    pub tool_instruction: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProviderRouting {
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub order: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub only: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ignore: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub allow_fallbacks: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub require_parameters: Option<bool>,
}

impl ProviderRouting {
    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
            && self.only.is_empty()
            && self.ignore.is_empty()
            && self.allow_fallbacks.is_none()
            && self.require_parameters.is_none()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProfileMetadata {
    pub profile_id: String,
    pub profile_revision: u32,
    pub profile_source: String,
    #[serde(default = "default_dsrs_history_format")]
    pub dsrs_history_format: DsrsHistoryFormat,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_adapter_artifact: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correction_agent_artifact: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<ProviderRouting>,
}

impl ModelProfile {
    pub fn default_balanced() -> Self {
        Self {
            name: "balanced-default".to_string(),
            model_patterns: vec!["*".to_string()],
            revision: default_profile_revision(),
            source: builtin_profile_source(),
            request_adapter_artifact: None,
            correction_agent_artifact: None,
            correction_instruction: None,
            tool_mode: ToolMode::ProxyOwned,
            tool_format: ToolFormat::Dsrs,
            dsrs_history_format: default_dsrs_history_format(),
            correction_model: None,
            judge_model: None,
            provider: None,
            max_correction_passes: 1,
            supports_parallel_tool_calls: true,
            tool_instruction: default_dsrs_instruction(),
        }
    }

    pub fn qwen() -> Self {
        Self {
            name: "qwen-dsrs".to_string(),
            model_patterns: vec!["qwen".to_string(), "qwq".to_string()],
            revision: default_profile_revision(),
            source: builtin_profile_source(),
            request_adapter_artifact: None,
            correction_agent_artifact: None,
            correction_instruction: None,
            tool_mode: ToolMode::ProxyOwned,
            tool_format: ToolFormat::Dsrs,
            dsrs_history_format: default_dsrs_history_format(),
            correction_model: None,
            judge_model: None,
            provider: None,
            max_correction_passes: 1,
            supports_parallel_tool_calls: true,
            tool_instruction: default_dsrs_instruction(),
        }
    }

    pub fn kimi() -> Self {
        Self {
            name: "kimi-dsrs".to_string(),
            model_patterns: vec!["kimi".to_string(), "moonshot".to_string()],
            revision: default_profile_revision(),
            source: builtin_profile_source(),
            request_adapter_artifact: None,
            correction_agent_artifact: None,
            correction_instruction: None,
            tool_mode: ToolMode::ProxyOwned,
            tool_format: ToolFormat::Dsrs,
            dsrs_history_format: default_dsrs_history_format(),
            correction_model: None,
            judge_model: None,
            provider: None,
            max_correction_passes: 1,
            supports_parallel_tool_calls: true,
            tool_instruction: format!(
                "{}\nKeep DSRs output compact and avoid prose outside the DSRs fields.",
                default_dsrs_instruction()
            ),
        }
    }

    pub fn glm() -> Self {
        Self {
            name: "glm-dsrs".to_string(),
            model_patterns: vec!["glm".to_string(), "z-ai".to_string()],
            revision: default_profile_revision(),
            source: builtin_profile_source(),
            request_adapter_artifact: None,
            correction_agent_artifact: None,
            correction_instruction: None,
            tool_mode: ToolMode::ProxyOwned,
            tool_format: ToolFormat::Dsrs,
            dsrs_history_format: default_dsrs_history_format(),
            correction_model: None,
            judge_model: None,
            provider: None,
            max_correction_passes: 1,
            supports_parallel_tool_calls: true,
            tool_instruction: default_dsrs_instruction(),
        }
    }

    pub fn llama() -> Self {
        Self {
            name: "llama-dsrs".to_string(),
            model_patterns: vec!["llama".to_string()],
            revision: default_profile_revision(),
            source: builtin_profile_source(),
            request_adapter_artifact: None,
            correction_agent_artifact: None,
            correction_instruction: None,
            tool_mode: ToolMode::ProxyOwned,
            tool_format: ToolFormat::Dsrs,
            dsrs_history_format: default_dsrs_history_format(),
            correction_model: None,
            judge_model: None,
            provider: None,
            max_correction_passes: 1,
            supports_parallel_tool_calls: true,
            tool_instruction: default_dsrs_instruction(),
        }
    }

    pub fn gemma() -> Self {
        apply_builtin_profile_default(Self {
            name: "gemma-dsrs-conservative".to_string(),
            model_patterns: vec!["gemma".to_string()],
            revision: default_profile_revision(),
            source: builtin_profile_source(),
            request_adapter_artifact: None,
            correction_agent_artifact: None,
            correction_instruction: None,
            tool_mode: ToolMode::ProxyOwned,
            tool_format: ToolFormat::Dsrs,
            dsrs_history_format: default_dsrs_history_format(),
            correction_model: None,
            judge_model: None,
            provider: None,
            max_correction_passes: 1,
            supports_parallel_tool_calls: false,
            tool_instruction: format!(
                "{}\nPrefer a single item in the tool_calls field. Do not emit multiple tool calls unless explicitly required.",
                default_dsrs_instruction()
            ),
        })
    }

    pub fn pass_through() -> Self {
        Self {
            name: "pass-through".to_string(),
            model_patterns: Vec::new(),
            revision: default_profile_revision(),
            source: builtin_profile_source(),
            request_adapter_artifact: None,
            correction_agent_artifact: None,
            correction_instruction: None,
            tool_mode: ToolMode::PassThrough,
            tool_format: ToolFormat::TaggedJson,
            dsrs_history_format: default_dsrs_history_format(),
            correction_model: None,
            judge_model: None,
            provider: None,
            max_correction_passes: 1,
            supports_parallel_tool_calls: true,
            tool_instruction: String::new(),
        }
    }

    pub fn metadata(&self) -> ProfileMetadata {
        ProfileMetadata {
            profile_id: self.name.clone(),
            profile_revision: self.revision,
            profile_source: self.source.clone(),
            dsrs_history_format: self.dsrs_history_format,
            request_adapter_artifact: self.request_adapter_artifact.clone(),
            correction_agent_artifact: self.correction_agent_artifact.clone(),
            provider: self.provider.clone(),
        }
    }

    pub fn mark_config_source(&mut self) {
        self.source = "config".to_string();
    }
}

pub fn resolve_profile(model: &str, configured: &[ModelProfile]) -> ModelProfile {
    let lower = model.to_ascii_lowercase();
    configured
        .iter()
        .chain(builtin_profiles().iter())
        .find(|profile| profile_matches(&lower, profile))
        .cloned()
        .unwrap_or_else(ModelProfile::default_balanced)
}

pub fn builtin_profiles() -> Vec<ModelProfile> {
    vec![
        ModelProfile::qwen(),
        ModelProfile::kimi(),
        ModelProfile::glm(),
        ModelProfile::llama(),
        ModelProfile::gemma(),
    ]
    .into_iter()
    .map(apply_builtin_profile_default)
    .collect()
}

fn apply_builtin_profile_default(mut profile: ModelProfile) -> ModelProfile {
    let Some(default) = builtin_defaults::profile_default(&profile.name) else {
        return profile;
    };

    profile.revision = default.revision;
    if !default.model_patterns.is_empty() {
        profile.model_patterns = default
            .model_patterns
            .iter()
            .map(|pattern| (*pattern).to_string())
            .collect();
    }
    if let Some(format) = default.dsrs_history_format {
        profile.dsrs_history_format = format;
    }
    if let Some(artifact_id) = default.request_adapter_artifact {
        profile.request_adapter_artifact = Some(builtin_defaults::artifact_reference(artifact_id));
        profile.tool_instruction = builtin_defaults::instruction_for_id(artifact_id)
            .expect("built-in request adapter artifact must contain an instruction");
    }
    if let Some(artifact_id) = default.correction_agent_artifact {
        profile.correction_agent_artifact = Some(builtin_defaults::artifact_reference(artifact_id));
        profile.correction_instruction = Some(
            builtin_defaults::instruction_for_id(artifact_id)
                .expect("built-in correction agent artifact must contain an instruction"),
        );
    }

    profile
}

fn profile_matches(lower_model: &str, profile: &ModelProfile) -> bool {
    profile.model_patterns.iter().any(|pattern| {
        let pattern = pattern.to_ascii_lowercase();
        pattern == "*" || lower_model.contains(pattern.trim_matches('*'))
    })
}

pub fn default_dsrs_instruction() -> String {
    [
        "Tool calls are an application contract. If a tool is needed, do not narrate the action.",
        "Use the provided DSRs output fields exactly and do not create extra labels.",
        "Answer the latest non-system conversation message; obey system_context but never copy it.",
        "Do not copy or summarize the serialized input fields, tool definitions, or prompt template.",
        "When no tool is needed, put only the user-facing reply in content and set tool_calls to [].",
        r#"When a tool is needed, set tool_calls to a valid JSON array like [{"name":"tool_name","arguments":{"argument":"value"}}]. You may also include brief user-facing content before the tool call when it helps the caller understand what you are doing."#,
        "Never emit both empty content and [] tool_calls for a real user turn; either answer in content or call an available tool.",
        "Every arguments object must match the selected tool parameters.",
        "Do not put scratchpad reasoning, field explanations, or presentation labels in content.",
    ]
    .join("\n")
}

fn default_tool_mode() -> ToolMode {
    ToolMode::ProxyOwned
}

fn default_tool_format() -> ToolFormat {
    ToolFormat::Dsrs
}

fn default_dsrs_history_format() -> DsrsHistoryFormat {
    DsrsHistoryFormat::AppendOnly
}

fn default_max_correction_passes() -> usize {
    1
}

fn default_supports_parallel_tool_calls() -> bool {
    true
}

fn default_profile_revision() -> u32 {
    1
}

fn default_profile_source() -> String {
    "unknown".to_string()
}

fn builtin_profile_source() -> String {
    "builtin".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_models_resolve_to_dsrs_proxy_owned_profile() {
        let profile = resolve_profile("qwen/qwen3-8b", &[]);
        assert_eq!(profile.tool_mode, ToolMode::ProxyOwned);
        assert_eq!(profile.tool_format, ToolFormat::Dsrs);
    }

    #[test]
    fn llama_models_use_dsrs_profile() {
        let profile = resolve_profile("meta-llama/llama-3.2-3b-instruct", &[]);
        assert_eq!(profile.name, "llama-dsrs");
        assert_eq!(profile.tool_format, ToolFormat::Dsrs);
    }

    #[test]
    fn unknown_models_fall_back_to_balanced_default() {
        let profile = resolve_profile("some-new-provider/new-model-1", &[]);

        assert_eq!(profile.name, "balanced-default");
        assert_eq!(profile.tool_mode, ToolMode::ProxyOwned);
        assert_eq!(profile.tool_format, ToolFormat::Dsrs);
        assert_eq!(profile.dsrs_history_format, DsrsHistoryFormat::AppendOnly);
        assert_eq!(profile.source, "builtin");
    }

    #[test]
    fn gemma_builtin_profile_uses_embedded_promoted_default() {
        let profile = resolve_profile("google/gemma-4-26b-a4b-it", &[]);

        assert_eq!(profile.name, "gemma-dsrs-conservative");
        assert_eq!(profile.revision, 9);
        assert_eq!(profile.dsrs_history_format, DsrsHistoryFormat::AppendOnly);
        assert_eq!(
            profile.request_adapter_artifact.as_deref(),
            Some("builtin:request-adapter/gemma-dsrs-conservative/sonnet5-shuffled-r1-append-only")
        );
        assert!(profile
            .tool_instruction
            .contains("MANDATORY INSPECTION FIRST"));
    }

    #[test]
    fn qwen_builtin_profile_uses_embedded_promoted_default() {
        let profile = resolve_profile("qwen/qwen3.5-9b", &[]);

        assert_eq!(profile.name, "qwen-dsrs");
        assert_eq!(profile.revision, 6);
        assert_eq!(
            profile.dsrs_history_format,
            DsrsHistoryFormat::RegeneratedContext
        );
        assert_eq!(
            profile.request_adapter_artifact.as_deref(),
            Some("builtin:request-adapter/qwen-dsrs/sonnet5-shuffled-r1-regenerated-context")
        );
        assert!(profile
            .tool_instruction
            .contains("CRITICAL JSON ESCAPING RULE"));
    }

    #[test]
    fn profile_metadata_records_runtime_artifact_ids() {
        let mut profile = ModelProfile::qwen();
        profile.revision = 7;
        profile.source = "config".to_string();
        profile.request_adapter_artifact = Some("profiles/qwen/request.json".to_string());
        profile.correction_agent_artifact = Some("profiles/qwen/correction.json".to_string());

        let metadata = profile.metadata();

        assert_eq!(metadata.profile_id, "qwen-dsrs");
        assert_eq!(metadata.profile_revision, 7);
        assert_eq!(metadata.profile_source, "config");
        assert_eq!(metadata.dsrs_history_format, DsrsHistoryFormat::AppendOnly);
        assert_eq!(
            metadata.request_adapter_artifact.as_deref(),
            Some("profiles/qwen/request.json")
        );
        assert_eq!(
            metadata.correction_agent_artifact.as_deref(),
            Some("profiles/qwen/correction.json")
        );
    }
}
