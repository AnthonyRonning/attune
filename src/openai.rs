use std::collections::BTreeMap;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<ChatMessage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<OpenAiTool>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_choice: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parallel_tool_calls: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_completion_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub response_format: Option<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChatMessage {
    pub role: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAiToolCall>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl ChatMessage {
    pub fn new(role: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: role.into(),
            content: Some(Value::String(content.into())),
            ..Self::default()
        }
    }

    pub fn content_text(&self) -> Option<String> {
        match self.content.as_ref()? {
            Value::String(text) => Some(text.clone()),
            Value::Array(parts) => {
                let mut out = String::new();
                for part in parts {
                    if let Some(text) = part.get("text").and_then(Value::as_str) {
                        out.push_str(text);
                    }
                }
                if out.is_empty() {
                    None
                } else {
                    Some(out)
                }
            }
            Value::Null => None,
            other => Some(other.to_string()),
        }
    }

    pub fn set_content_text(&mut self, text: impl Into<String>) {
        self.content = Some(Value::String(text.into()));
    }

    pub fn set_content_null(&mut self) {
        self.content = Some(Value::Null);
    }

    pub fn reasoning_text(&self) -> Option<String> {
        ["reasoning", "reasoning_content", "thinking"]
            .iter()
            .find_map(|key| {
                self.extra
                    .get(*key)
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiTool {
    #[serde(rename = "type")]
    pub tool_type: String,
    pub function: OpenAiFunctionTool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiFunctionTool {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(default)]
    pub parameters: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenAiToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub call_type: String,
    pub function: OpenAiFunctionCall,
}

impl OpenAiToolCall {
    pub fn function(name: impl Into<String>, arguments: impl Into<String>) -> Self {
        Self {
            id: format!("call_{}", Uuid::new_v4().simple()),
            call_type: "function".to_string(),
            function: OpenAiFunctionCall {
                name: name.into(),
                arguments: arguments.into(),
            },
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenAiFunctionCall {
    pub name: String,
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    #[serde(default)]
    pub choices: Vec<ChatChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

impl ChatCompletionResponse {
    pub fn empty_for_model(model: impl Into<String>) -> Self {
        Self {
            id: format!("chatcmpl-{}", Uuid::new_v4().simple()),
            object: "chat.completion".to_string(),
            created: Utc::now().timestamp(),
            model: model.into(),
            choices: Vec::new(),
            usage: None,
            extra: Map::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatChoice {
    pub index: u32,
    pub message: ChatMessage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Value>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiErrorResponse {
    pub error: OpenAiError,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiError {
    pub message: String,
    #[serde(rename = "type")]
    pub error_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub param: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub code: Option<String>,
}

pub fn redacted_headers_for_trace(headers: &axum::http::HeaderMap) -> BTreeMap<String, String> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            let key = name.as_str().to_ascii_lowercase();
            if key == "authorization" || key == "cookie" || key.contains("key") {
                return None;
            }
            value
                .to_str()
                .ok()
                .map(|value| (key, value.chars().take(256).collect()))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn extracts_text_from_openai_content_parts() {
        let msg = ChatMessage {
            role: "user".to_string(),
            content: Some(json!([
                {"type": "text", "text": "hello "},
                {"type": "text", "text": "world"}
            ])),
            ..ChatMessage::default()
        };

        assert_eq!(msg.content_text().as_deref(), Some("hello world"));
    }
}
