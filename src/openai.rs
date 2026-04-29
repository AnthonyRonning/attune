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
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    #[serde(default)]
    pub choices: Vec<ChatCompletionChunkChoice>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChunkChoice {
    pub index: u32,
    pub delta: ChatCompletionDelta,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<Value>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ChatCompletionDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<OpenAiToolCallDelta>>,
    #[serde(flatten)]
    pub extra: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiToolCallDelta {
    pub index: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type", default, skip_serializing_if = "Option::is_none")]
    pub call_type: Option<String>,
    pub function: OpenAiFunctionCallDelta,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenAiFunctionCallDelta {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub arguments: Option<String>,
}

pub fn chat_completion_response_to_sse(
    response: &ChatCompletionResponse,
    include_usage: bool,
) -> serde_json::Result<String> {
    let mut out = String::new();

    for choice in &response.choices {
        let message = &choice.message;
        let (reasoning_extra, mut remaining_extra) = split_reasoning_extra(&message.extra);

        push_sse_choice_delta(
            &mut out,
            response,
            choice,
            ChatCompletionDelta {
                role: Some(message.role.clone()),
                content: None,
                tool_calls: None,
                extra: reasoning_extra,
            },
            None,
        )?;

        if let Some(content) = message.content_text() {
            push_sse_choice_delta(
                &mut out,
                response,
                choice,
                ChatCompletionDelta {
                    role: None,
                    content: Some(content),
                    tool_calls: None,
                    extra: std::mem::take(&mut remaining_extra),
                },
                None,
            )?;
        }

        if let Some(tool_calls) = &message.tool_calls {
            push_sse_choice_delta(
                &mut out,
                response,
                choice,
                ChatCompletionDelta {
                    role: None,
                    content: None,
                    tool_calls: Some(tool_call_deltas(tool_calls)),
                    extra: std::mem::take(&mut remaining_extra),
                },
                None,
            )?;
        }

        if !remaining_extra.is_empty() {
            push_sse_choice_delta(
                &mut out,
                response,
                choice,
                ChatCompletionDelta {
                    role: None,
                    content: None,
                    tool_calls: None,
                    extra: remaining_extra,
                },
                None,
            )?;
        }

        push_sse_choice_delta(
            &mut out,
            response,
            choice,
            ChatCompletionDelta::default(),
            choice.finish_reason.clone(),
        )?;
    }

    if include_usage {
        if let Some(usage) = response.usage.clone() {
            push_sse_chunk(
                &mut out,
                ChatCompletionChunk {
                    id: response.id.clone(),
                    object: "chat.completion.chunk".to_string(),
                    created: response.created,
                    model: response.model.clone(),
                    choices: Vec::new(),
                    usage: Some(usage),
                },
            )?;
        }
    }

    out.push_str("data: [DONE]\n\n");
    Ok(out)
}

fn push_sse_choice_delta(
    out: &mut String,
    response: &ChatCompletionResponse,
    choice: &ChatChoice,
    delta: ChatCompletionDelta,
    finish_reason: Option<String>,
) -> serde_json::Result<()> {
    push_sse_chunk(
        out,
        ChatCompletionChunk {
            id: response.id.clone(),
            object: "chat.completion.chunk".to_string(),
            created: response.created,
            model: response.model.clone(),
            choices: vec![ChatCompletionChunkChoice {
                index: choice.index,
                delta,
                finish_reason,
                logprobs: choice.logprobs.clone(),
            }],
            usage: None,
        },
    )
}

fn push_sse_chunk(out: &mut String, chunk: ChatCompletionChunk) -> serde_json::Result<()> {
    out.push_str("data: ");
    out.push_str(&serde_json::to_string(&chunk)?);
    out.push_str("\n\n");
    Ok(())
}

fn tool_call_deltas(tool_calls: &[OpenAiToolCall]) -> Vec<OpenAiToolCallDelta> {
    tool_calls
        .iter()
        .enumerate()
        .map(|(index, call)| OpenAiToolCallDelta {
            index: index as u32,
            id: Some(call.id.clone()),
            call_type: Some(call.call_type.clone()),
            function: OpenAiFunctionCallDelta {
                name: Some(call.function.name.clone()),
                arguments: Some(call.function.arguments.clone()),
            },
        })
        .collect()
}

fn split_reasoning_extra(extra: &Map<String, Value>) -> (Map<String, Value>, Map<String, Value>) {
    let mut reasoning = Map::new();
    let mut remaining = Map::new();
    for (key, value) in extra {
        if is_reasoning_key(key) {
            reasoning.insert(key.clone(), value.clone());
        } else {
            remaining.insert(key.clone(), value.clone());
        }
    }
    (reasoning, remaining)
}

fn is_reasoning_key(key: &str) -> bool {
    matches!(
        key,
        "reasoning" | "reasoning_content" | "reasoning_details" | "thinking" | "thinking_details"
    )
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

    #[test]
    fn streams_content_response_as_openai_chunks() {
        let mut response = ChatCompletionResponse::empty_for_model("test-model");
        response.id = "chatcmpl-test".to_string();
        response.created = 1;
        response.choices.push(ChatChoice {
            index: 0,
            message: ChatMessage::new("assistant", "hello"),
            finish_reason: Some("stop".to_string()),
            logprobs: None,
            extra: Map::new(),
        });

        let sse = chat_completion_response_to_sse(&response, false).unwrap();

        assert!(sse.contains(r#""object":"chat.completion.chunk""#));
        assert!(sse.contains(r#""delta":{"role":"assistant"}"#));
        assert!(sse.contains(r#""delta":{"content":"hello"}"#));
        assert!(sse.contains(r#""finish_reason":"stop""#));
        assert!(sse.ends_with("data: [DONE]\n\n"));
    }

    #[test]
    fn streams_tool_calls_as_openai_chunks() {
        let mut response = ChatCompletionResponse::empty_for_model("test-model");
        response.id = "chatcmpl-test".to_string();
        response.created = 1;
        response.choices.push(ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: Some(Value::Null),
                name: None,
                tool_call_id: None,
                tool_calls: Some(vec![OpenAiToolCall::function(
                    "read_file",
                    r#"{"path":"Cargo.toml"}"#,
                )]),
                extra: Map::new(),
            },
            finish_reason: Some("tool_calls".to_string()),
            logprobs: None,
            extra: Map::new(),
        });

        let sse = chat_completion_response_to_sse(&response, false).unwrap();

        assert!(sse.contains(r#""tool_calls":[{"index":0"#));
        assert!(sse.contains(r#""name":"read_file""#));
        assert!(sse.contains(r#""arguments":"{\"path\":\"Cargo.toml\"}""#));
        assert!(sse.contains(r#""finish_reason":"tool_calls""#));
    }

    #[test]
    fn streams_reasoning_before_content() {
        let mut response = ChatCompletionResponse::empty_for_model("test-model");
        response.id = "chatcmpl-test".to_string();
        response.created = 1;
        let mut message = ChatMessage::new("assistant", "final answer");
        message.extra.insert(
            "reasoning_content".to_string(),
            Value::String("thinking first".to_string()),
        );
        message.extra.insert(
            "reasoning_details".to_string(),
            json!([{"type": "reasoning.text", "text": "detail first"}]),
        );
        response.choices.push(ChatChoice {
            index: 0,
            message,
            finish_reason: Some("stop".to_string()),
            logprobs: None,
            extra: Map::new(),
        });

        let sse = chat_completion_response_to_sse(&response, false).unwrap();
        let reasoning_index = sse.find(r#""reasoning_content":"thinking first""#).unwrap();
        let details_index = sse.find(r#""reasoning_details":[{"#).unwrap();
        let content_index = sse.find(r#""content":"final answer""#).unwrap();

        assert!(reasoning_index < content_index);
        assert!(details_index < content_index);
        assert_eq!(sse.matches("reasoning_content").count(), 1);
        assert_eq!(sse.matches("reasoning_details").count(), 1);
    }
}
