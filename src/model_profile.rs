use serde::{Deserialize, Serialize};

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    pub name: String,
    #[serde(default)]
    pub model_patterns: Vec<String>,
    pub tool_mode: ToolMode,
    pub tool_format: ToolFormat,
    pub correction_model: Option<String>,
    pub judge_model: Option<String>,
    pub max_correction_passes: usize,
    pub supports_parallel_tool_calls: bool,
    pub tool_instruction: String,
}

impl ModelProfile {
    pub fn default_balanced() -> Self {
        Self {
            name: "balanced-default".to_string(),
            model_patterns: vec!["*".to_string()],
            tool_mode: ToolMode::ProxyOwned,
            tool_format: ToolFormat::Dsrs,
            correction_model: None,
            judge_model: None,
            max_correction_passes: 1,
            supports_parallel_tool_calls: true,
            tool_instruction: default_dsrs_instruction(),
        }
    }

    pub fn qwen() -> Self {
        Self {
            name: "qwen-dsrs".to_string(),
            model_patterns: vec!["qwen".to_string(), "qwq".to_string()],
            tool_mode: ToolMode::ProxyOwned,
            tool_format: ToolFormat::Dsrs,
            correction_model: None,
            judge_model: None,
            max_correction_passes: 1,
            supports_parallel_tool_calls: true,
            tool_instruction: default_dsrs_instruction(),
        }
    }

    pub fn kimi() -> Self {
        Self {
            name: "kimi-dsrs".to_string(),
            model_patterns: vec!["kimi".to_string(), "moonshot".to_string()],
            tool_mode: ToolMode::ProxyOwned,
            tool_format: ToolFormat::Dsrs,
            correction_model: None,
            judge_model: None,
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
            tool_mode: ToolMode::ProxyOwned,
            tool_format: ToolFormat::Dsrs,
            correction_model: None,
            judge_model: None,
            max_correction_passes: 1,
            supports_parallel_tool_calls: true,
            tool_instruction: default_dsrs_instruction(),
        }
    }

    pub fn llama() -> Self {
        Self {
            name: "llama-dsrs".to_string(),
            model_patterns: vec!["llama".to_string()],
            tool_mode: ToolMode::ProxyOwned,
            tool_format: ToolFormat::Dsrs,
            correction_model: None,
            judge_model: None,
            max_correction_passes: 1,
            supports_parallel_tool_calls: true,
            tool_instruction: default_dsrs_instruction(),
        }
    }

    pub fn gemma() -> Self {
        Self {
            name: "gemma-dsrs-conservative".to_string(),
            model_patterns: vec!["gemma".to_string()],
            tool_mode: ToolMode::ProxyOwned,
            tool_format: ToolFormat::Dsrs,
            correction_model: None,
            judge_model: None,
            max_correction_passes: 1,
            supports_parallel_tool_calls: false,
            tool_instruction: format!(
                "{}\nPrefer a single item in the tool_calls field. Do not emit multiple tool calls unless explicitly required.",
                default_dsrs_instruction()
            ),
        }
    }

    pub fn pass_through() -> Self {
        Self {
            name: "pass-through".to_string(),
            model_patterns: Vec::new(),
            tool_mode: ToolMode::PassThrough,
            tool_format: ToolFormat::TaggedJson,
            correction_model: None,
            judge_model: None,
            max_correction_passes: 1,
            supports_parallel_tool_calls: true,
            tool_instruction: String::new(),
        }
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
}

fn profile_matches(lower_model: &str, profile: &ModelProfile) -> bool {
    profile.model_patterns.iter().any(|pattern| {
        let pattern = pattern.to_ascii_lowercase();
        pattern == "*" || lower_model.contains(pattern.trim_matches('*'))
    })
}

fn default_dsrs_instruction() -> String {
    [
        "Tool calls are an application contract. If a tool is needed, do not narrate the action.",
        "Respond only with DSRs fields generated by the tool-use contract:",
        "[[ ## content ## ]]",
        "assistant text, or empty when using tools",
        "[[ ## tool_calls ## ]]",
        r#"[{"name":"tool_name","arguments":{"argument":"value"}}]"#,
        "[[ ## completed ## ]]",
        "The tool_calls field must be valid JSON and every arguments object must match the selected tool parameters.",
    ]
    .join("\n")
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
}
