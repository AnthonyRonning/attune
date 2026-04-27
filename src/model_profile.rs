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
            tool_format: ToolFormat::Xml,
            correction_model: None,
            judge_model: None,
            max_correction_passes: 1,
            supports_parallel_tool_calls: true,
            tool_instruction: default_xml_instruction(),
        }
    }

    pub fn qwen() -> Self {
        Self {
            name: "qwen-xml".to_string(),
            model_patterns: vec!["qwen".to_string()],
            tool_mode: ToolMode::ProxyOwned,
            tool_format: ToolFormat::Xml,
            correction_model: None,
            judge_model: None,
            max_correction_passes: 1,
            supports_parallel_tool_calls: true,
            tool_instruction: default_xml_instruction(),
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
        .find(|profile| profile_matches(&lower, profile))
        .cloned()
        .unwrap_or_else(|| {
            if lower.contains("qwen") {
                ModelProfile::qwen()
            } else {
                ModelProfile::default_balanced()
            }
        })
}

fn profile_matches(lower_model: &str, profile: &ModelProfile) -> bool {
    profile.model_patterns.iter().any(|pattern| {
        let pattern = pattern.to_ascii_lowercase();
        pattern == "*" || lower_model.contains(pattern.trim_matches('*'))
    })
}

fn default_xml_instruction() -> String {
    [
        "Tool calls are an application contract. If a tool is needed, do not narrate the action.",
        "Emit only XML tool calls using this exact shape:",
        r#"<tool_call name="tool_name">"#,
        r#"{"argument":"value"}"#,
        "</tool_call>",
        "The JSON body must match the selected tool parameters. You may emit multiple consecutive tool_call blocks only when the request allows parallel tool calls.",
    ]
    .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_models_resolve_to_xml_proxy_owned_profile() {
        let profile = resolve_profile("qwen/qwen3-8b", &[]);
        assert_eq!(profile.tool_mode, ToolMode::ProxyOwned);
        assert_eq!(profile.tool_format, ToolFormat::Xml);
    }
}
