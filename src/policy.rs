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
    pub mode: PolicyMode,
    pub deterministic_json_repair: bool,
    pub schema_guided_repair: bool,
    pub correction_agent: bool,
    pub retry_or_continue: bool,
    pub hallucinated_tool_match_threshold: f64,
    pub max_tool_calls_without_parallel: usize,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            mode: PolicyMode::Balanced,
            deterministic_json_repair: true,
            schema_guided_repair: true,
            correction_agent: true,
            retry_or_continue: true,
            hallucinated_tool_match_threshold: 0.86,
            max_tool_calls_without_parallel: 1,
        }
    }
}
