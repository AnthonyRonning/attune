use std::panic::{catch_unwind, AssertUnwindSafe};

use anyhow::Result;
use dspy_rs::{adapter::Adapter, example, ChatAdapter, Message, Signature};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    model_profile::ModelProfile,
    normalizer::NormalizedRequest,
    openai::ChatMessage,
    response_interpreter::{ToolIntent, ToolIntentSource},
};

#[Signature]
struct OpenAiToolUseContract {
    /// You are an OpenAI-compatible assistant behind a correction proxy. Honor the
    /// serialized conversation, choose tools only from available_tools, and produce
    /// exactly the DSRs output fields. Put user-facing text in content. Put tool
    /// calls in tool_calls as a JSON array of {"name": string, "arguments": object}.
    /// Use [] when no tool call is needed. If parallel_tool_calls is false, emit at
    /// most one tool call.
    #[input(desc = "Additional profile-specific guidance")]
    pub profile_guidance: String,

    #[input(desc = "Serialized OpenAI messages to answer")]
    pub conversation: String,

    #[input(desc = "Serialized OpenAI tool definitions")]
    pub available_tools: String,

    #[input(desc = "Original OpenAI tool_choice")]
    pub tool_choice: String,

    #[input(desc = "Whether multiple tool calls may be emitted")]
    pub parallel_tool_calls: bool,

    #[output(desc = "Assistant content. Use an empty string when emitting tool calls.")]
    pub content: String,

    #[output(desc = "JSON array of {\"name\": string, \"arguments\": object} tool calls")]
    pub tool_calls: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormattedDsrsContract {
    pub messages: Vec<ChatMessage>,
    pub instruction: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParsedDsrsContract {
    pub content: Option<String>,
    pub tool_intents: Vec<ToolIntent>,
    pub events: Vec<String>,
}

pub fn format_tool_contract(
    normalized: &NormalizedRequest,
    profile: &ModelProfile,
) -> Result<FormattedDsrsContract> {
    let adapter = ChatAdapter;
    let signature = OpenAiToolUseContract::new();
    let conversation = serde_json::to_string_pretty(&normalized.messages)?;
    let available_tools = serde_json::to_string_pretty(&normalized.tools)?;
    let tool_choice = normalized
        .tool_choice
        .as_ref()
        .map(serde_json::to_string)
        .transpose()?
        .unwrap_or_else(|| "auto".to_string());

    let chat = adapter.format(
        &signature,
        example! {
            "profile_guidance": "input" => profile.tool_instruction.clone(),
            "conversation": "input" => conversation,
            "available_tools": "input" => available_tools,
            "tool_choice": "input" => tool_choice,
            "parallel_tool_calls": "input" => normalized.parallel_tool_calls,
        },
    );

    let mut instruction = String::new();
    let messages = chat
        .messages
        .into_iter()
        .map(|message| {
            if let Message::System { content } = &message {
                instruction = content.clone();
            }
            dsrs_message_to_openai(message)
        })
        .collect();

    Ok(FormattedDsrsContract {
        messages,
        instruction,
    })
}

pub fn parse_tool_contract_response(content: &str) -> Option<ParsedDsrsContract> {
    if !contains_dsrs_marker(content) {
        return None;
    }

    let adapter = ChatAdapter;
    let signature = OpenAiToolUseContract::new();
    let parsed = catch_unwind(AssertUnwindSafe(|| {
        adapter.parse_response(&signature, Message::assistant(content.to_string()))
    }));

    let parsed = match parsed {
        Ok(parsed) => parsed,
        Err(_) => {
            return Some(ParsedDsrsContract {
                content: None,
                tool_intents: Vec::new(),
                events: vec!["DSRs contract parser panicked on malformed output".to_string()],
            });
        }
    };

    let mut events = vec!["parsed response through DSRs tool-use contract".to_string()];
    let parsed_content = parsed
        .get("content")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_owned);

    let tool_calls_field = parsed
        .get("tool_calls")
        .and_then(Value::as_str)
        .unwrap_or("[]")
        .trim();

    let tool_intents = parse_tool_calls_field(tool_calls_field, &mut events);

    Some(ParsedDsrsContract {
        content: parsed_content,
        tool_intents,
        events,
    })
}

fn dsrs_message_to_openai(message: Message) -> ChatMessage {
    match message {
        Message::System { content } => ChatMessage::new("system", content),
        Message::User { content } => ChatMessage::new("user", content),
        Message::Assistant { content } => ChatMessage::new("assistant", content),
    }
}

fn contains_dsrs_marker(content: &str) -> bool {
    content.contains("[[ ## content ## ]]") || content.contains("[[ ## tool_calls ## ]]")
}

fn parse_tool_calls_field(field: &str, events: &mut Vec<String>) -> Vec<ToolIntent> {
    if field.is_empty() || field == "[]" {
        return Vec::new();
    }

    let Some(value) = parse_jsonish(field) else {
        events.push("DSRs tool_calls field was not valid JSON".to_string());
        return Vec::new();
    };

    let calls = match value {
        Value::Array(calls) => calls,
        Value::Object(mut object) => match object.remove("tool_calls") {
            Some(Value::Array(calls)) => calls,
            _ => vec![Value::Object(object)],
        },
        _ => {
            events.push("DSRs tool_calls field was not a JSON array or object".to_string());
            return Vec::new();
        }
    };

    calls
        .into_iter()
        .filter_map(|call| tool_intent_from_value(call, events))
        .collect()
}

fn tool_intent_from_value(call: Value, events: &mut Vec<String>) -> Option<ToolIntent> {
    let Value::Object(mut object) = call else {
        events.push("DSRs tool call item was not an object".to_string());
        return None;
    };

    let mut function_arguments = None;
    let function_name = match object.remove("function") {
        Some(Value::Object(mut function)) => {
            function_arguments = function.remove("arguments");
            function
                .remove("name")
                .and_then(|value| value.as_str().map(str::to_owned))
        }
        Some(Value::String(name)) => Some(name),
        _ => None,
    };

    let name = object
        .remove("name")
        .or_else(|| object.remove("tool_name"))
        .and_then(|value| value.as_str().map(str::to_owned))
        .or(function_name)?;

    let arguments = object
        .remove("arguments")
        .or_else(|| object.remove("args"))
        .or(function_arguments)
        .unwrap_or_else(|| Value::Object(object));
    let raw_arguments = match &arguments {
        Value::String(raw) => raw.clone(),
        other => serde_json::to_string(other).unwrap_or_else(|_| "{}".to_string()),
    };
    let parsed_arguments = match arguments {
        Value::String(raw) => parse_jsonish(&raw).unwrap_or(Value::String(raw)),
        other => other,
    };

    Some(ToolIntent {
        name,
        arguments: Some(parsed_arguments),
        raw_arguments,
        source: ToolIntentSource::Dsrs,
        confidence: 0.97,
    })
}

fn parse_jsonish(input: &str) -> Option<Value> {
    let trimmed = input.trim();
    serde_json::from_str(trimmed)
        .ok()
        .or_else(|| json5::from_str(trimmed).ok())
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Map};

    use super::*;
    use crate::openai::{ChatCompletionRequest, OpenAiFunctionTool, OpenAiTool};

    fn normalized_request() -> NormalizedRequest {
        crate::normalizer::normalize_request(ChatCompletionRequest {
            model: "test".to_string(),
            messages: vec![ChatMessage::new("user", "list files")],
            tools: Some(vec![OpenAiTool {
                tool_type: "function".to_string(),
                function: OpenAiFunctionTool {
                    name: "bash".to_string(),
                    description: Some("Run a shell command".to_string()),
                    parameters: json!({
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"]
                    }),
                },
            }]),
            tool_choice: None,
            parallel_tool_calls: Some(false),
            stream: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            max_completion_tokens: None,
            response_format: None,
            extra: Map::new(),
        })
        .unwrap()
    }

    #[test]
    fn formats_request_with_dsrs_field_markers() {
        let formatted = format_tool_contract(&normalized_request(), &ModelProfile::qwen()).unwrap();

        assert_eq!(formatted.messages[0].role, "system");
        assert!(formatted.instruction.contains("[[ ## content ## ]]"));
        assert!(formatted.messages[1]
            .content_text()
            .unwrap()
            .contains("[[ ## conversation ## ]]"));
        assert!(formatted.messages[1]
            .content_text()
            .unwrap()
            .contains("bash"));
    }

    #[test]
    fn parses_dsrs_tool_calls_field() {
        let parsed = parse_tool_contract_response(
            r#"[[ ## content ## ]]

[[ ## tool_calls ## ]]
[{"name":"bash","arguments":{"command":"ls -la packages/"}}]
[[ ## completed ## ]]"#,
        )
        .unwrap();

        assert_eq!(parsed.tool_intents.len(), 1);
        assert_eq!(parsed.tool_intents[0].name, "bash");
        assert_eq!(
            parsed.tool_intents[0].arguments.as_ref().unwrap()["command"],
            "ls -la packages/"
        );
    }
}
