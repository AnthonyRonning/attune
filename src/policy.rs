use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PolicyMode {
    Conservative,
    Balanced,
    AggressiveRecovery,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyConfig {
    #[serde(default = "default_policy_mode")]
    pub mode: PolicyMode,
    #[serde(default = "default_enabled")]
    pub deterministic_json_repair: bool,
    #[serde(default = "default_enabled")]
    pub schema_guided_repair: bool,
    #[serde(default = "default_enabled")]
    pub correction_agent: bool,
    #[serde(default = "default_enabled")]
    pub retry_or_continue: bool,
    #[serde(default = "default_hallucinated_tool_match_threshold")]
    pub hallucinated_tool_match_threshold: f64,
    #[serde(default = "default_max_tool_calls_without_parallel")]
    pub max_tool_calls_without_parallel: usize,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            mode: default_policy_mode(),
            deterministic_json_repair: default_enabled(),
            schema_guided_repair: default_enabled(),
            correction_agent: default_enabled(),
            retry_or_continue: default_enabled(),
            hallucinated_tool_match_threshold: default_hallucinated_tool_match_threshold(),
            max_tool_calls_without_parallel: default_max_tool_calls_without_parallel(),
        }
    }
}

fn default_policy_mode() -> PolicyMode {
    PolicyMode::Balanced
}

fn default_enabled() -> bool {
    true
}

fn default_hallucinated_tool_match_threshold() -> f64 {
    0.86
}

fn default_max_tool_calls_without_parallel() -> usize {
    1
}
