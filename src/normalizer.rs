use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

use crate::openai::{ChatCompletionRequest, ChatMessage, OpenAiTool};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NormalizedRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub tools: Vec<OpenAiTool>,
    pub tool_choice: Option<serde_json::Value>,
    pub parallel_tool_calls: bool,
    pub original: ChatCompletionRequest,
}

pub fn normalize_request(request: ChatCompletionRequest) -> Result<NormalizedRequest> {
    if request.messages.is_empty() {
        bail!("chat completion request must include at least one message");
    }

    Ok(NormalizedRequest {
        model: request.model.clone(),
        messages: request.messages.clone(),
        tools: request.tools.clone().unwrap_or_default(),
        tool_choice: request.tool_choice.clone(),
        parallel_tool_calls: request.parallel_tool_calls.unwrap_or(true),
        original: request,
    })
}

#[cfg(test)]
mod tests {
    use serde_json::Map;

    use super::*;

    #[test]
    fn defaults_parallel_tool_calls_to_true() {
        let req = ChatCompletionRequest {
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

        let normalized = normalize_request(req).unwrap();
        assert!(normalized.parallel_tool_calls);
    }
}
