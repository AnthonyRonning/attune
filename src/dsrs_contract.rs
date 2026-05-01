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
    /// most one tool call. Do not put field labels, scratchpad reasoning, or
    /// chain-of-thought in content.
    #[input(desc = "Additional profile-specific guidance")]
    pub profile_guidance: String,

    #[input(desc = "Serialized system/developer instructions to obey but never repeat")]
    pub system_context: String,

    #[input(desc = "Serialized non-system OpenAI messages to answer")]
    pub conversation: String,

    #[input(desc = "Serialized OpenAI tool definitions")]
    pub available_tools: String,

    #[input(desc = "Original OpenAI tool_choice")]
    pub tool_choice: String,

    #[input(desc = "Whether multiple tool calls may be emitted")]
    pub parallel_tool_calls: bool,

    #[output(desc = "Plain user-facing reply text without labels; empty when using tools.")]
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
    let (system_messages, conversation_messages): (Vec<_>, Vec<_>) = normalized
        .messages
        .iter()
        .cloned()
        .partition(|message| message.role == "system" || message.role == "developer");
    let system_context = serde_json::to_string_pretty(&system_messages)?;
    let conversation = serde_json::to_string_pretty(&conversation_messages)?;
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
            "system_context": "input" => system_context,
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
    let fallback = parse_contract_fallback(content);
    let parsed_content = fallback.content.clone().or_else(|| {
        parsed
            .get("content")
            .and_then(Value::as_str)
            .and_then(clean_output_content)
    });

    let parsed_tool_calls = parsed
        .get("tool_calls")
        .and_then(Value::as_str)
        .and_then(clean_tool_calls_field);
    let tool_calls_field = fallback
        .tool_calls
        .as_deref()
        .and_then(clean_tool_calls_field)
        .or(parsed_tool_calls)
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
    content.contains("[[ ## content ## ]]")
        || content.contains("[[ ## tool_calls ## ]]")
        || looks_like_label_free_dsrs(content)
}

#[derive(Debug, Default)]
struct ContractFallback {
    content: Option<String>,
    tool_calls: Option<String>,
}

fn parse_contract_fallback(content: &str) -> ContractFallback {
    if let Some((before_tool_calls, after_tool_calls)) =
        content.rsplit_once("[[ ## tool_calls ## ]]")
    {
        let content_section = before_tool_calls
            .rsplit_once("[[ ## content ## ]]")
            .map(|(_, section)| section)
            .unwrap_or(before_tool_calls);
        return ContractFallback {
            content: clean_contract_content(content_section),
            tool_calls: Some(extract_until_completed(after_tool_calls).to_string()),
        };
    }

    if looks_like_label_free_dsrs(content) {
        return parse_label_free_dsrs(content);
    }

    ContractFallback::default()
}

fn parse_label_free_dsrs(content: &str) -> ContractFallback {
    let mut lines: Vec<&str> = content.lines().collect();
    trim_blank_line_suffix(&mut lines);
    if lines.last().is_some_and(|line| is_completed_line(line)) {
        lines.pop();
    }
    trim_blank_line_suffix(&mut lines);

    if let Some(tool_calls_index) = lines
        .iter()
        .rposition(|line| is_label(line.trim(), "tool_calls"))
    {
        let content = clean_contract_content(&lines[..tool_calls_index].join("\n"));
        let tool_call_lines = &lines[tool_calls_index + 1..];
        let tool_calls = if tool_call_lines.iter().all(|line| line.trim().is_empty()) {
            "[]".to_string()
        } else {
            tool_call_lines
                .iter()
                .map(|line| line.trim())
                .filter(|line| !line.is_empty())
                .collect::<Vec<_>>()
                .join("\n")
        };
        return ContractFallback {
            content,
            tool_calls: Some(tool_calls),
        };
    }

    let mut tool_calls = None;
    if lines
        .last()
        .is_some_and(|line| line.trim().starts_with('[') && line.trim().ends_with(']'))
    {
        tool_calls = lines.pop().map(|line| line.trim().to_string());
    }
    trim_blank_line_suffix(&mut lines);

    ContractFallback {
        content: clean_contract_content(&lines.join("\n")),
        tool_calls,
    }
}

fn trim_blank_line_suffix(lines: &mut Vec<&str>) {
    while lines.last().is_some_and(|line| line.trim().is_empty()) {
        lines.pop();
    }
}

fn extract_until_completed(content: &str) -> &str {
    content
        .split("[[ ## completed ## ]]")
        .next()
        .unwrap_or(content)
        .trim()
}

fn clean_contract_content(content: &str) -> Option<String> {
    let mut cleaned = content
        .replace("[[ ## content ## ]]", "")
        .trim()
        .to_string();
    if let Some(rest) = cleaned
        .strip_prefix("content\n")
        .or_else(|| cleaned.strip_prefix("content:\n"))
        .or_else(|| cleaned.strip_prefix("content\r\n"))
        .or_else(|| cleaned.strip_prefix("content:\r\n"))
    {
        cleaned = rest.trim().to_string();
    }

    clean_output_content(&cleaned)
}

fn clean_output_content(value: &str) -> Option<String> {
    let cleaned = value.trim();
    if cleaned.is_empty() || is_label(cleaned, "content") || is_placeholder_value(cleaned) {
        None
    } else {
        Some(cleaned.to_string())
    }
}

fn clean_tool_calls_field(value: &str) -> Option<&str> {
    let cleaned = value.trim();
    if is_placeholder_value(cleaned) || is_label(cleaned, "tool_calls") {
        None
    } else {
        Some(cleaned)
    }
}

fn looks_like_label_free_dsrs(content: &str) -> bool {
    let mut lines = content
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty());
    let Some(last) = lines.next_back() else {
        return false;
    };
    is_completed_line(last)
        && lines
            .any(|line| line == "[]" || is_label(line, "content") || is_label(line, "tool_calls"))
}

fn is_completed_line(line: &str) -> bool {
    is_label(line, "completed") || line.trim().eq_ignore_ascii_case("[[ ## completed ## ]]")
}

fn is_label(line: &str, label: &str) -> bool {
    let normalized = line.trim().trim_end_matches(':');
    normalized.eq_ignore_ascii_case(label)
}

fn is_placeholder_value(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "content_value" | "tool_calls_value" | "tool_call_value" | "completed_marker"
    )
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
            messages: vec![
                ChatMessage::new("system", "You are a coding assistant. Do not repeat this."),
                ChatMessage::new("user", "list files"),
            ],
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
        let user_message = formatted.messages[1].content_text().unwrap();

        assert_eq!(formatted.messages[0].role, "system");
        assert!(formatted.instruction.contains("[[ ## content ## ]]"));
        assert!(user_message.contains("[[ ## system_context ## ]]"));
        assert!(user_message.contains("[[ ## conversation ## ]]"));
        assert!(user_message.contains("bash"));
        assert!(user_message.contains("Do not repeat this."));
        assert!(!user_message.contains("[[ ## content ## ]]"));
        assert!(!user_message.contains("[[ ## tool_calls ## ]]"));
        assert!(!user_message.contains("assistant text"));
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

    #[test]
    fn parses_partial_dsrs_without_content_marker() {
        let parsed = parse_tool_contract_response(
            "content\nHello!\n\n[[ ## tool_calls ## ]]\n[]\n\n[[ ## completed ## ]]",
        )
        .unwrap();

        assert_eq!(parsed.content.as_deref(), Some("Hello!"));
        assert!(parsed.tool_intents.is_empty());
    }

    #[test]
    fn parses_label_free_empty_tool_calls_tail() {
        let parsed =
            parse_tool_contract_response("Hello! How can I help?\n\n[]\n\ncompleted").unwrap();

        assert_eq!(parsed.content.as_deref(), Some("Hello! How can I help?"));
        assert!(parsed.tool_intents.is_empty());
    }

    #[test]
    fn parses_label_free_named_tool_calls_section() {
        let parsed =
            parse_tool_contract_response("content\nHello!\n\ntool_calls\n[]\n\ncompleted").unwrap();

        assert_eq!(parsed.content.as_deref(), Some("Hello!"));
        assert!(parsed.tool_intents.is_empty());
    }

    #[test]
    fn parses_label_free_output_with_completed_marker() {
        let parsed = parse_tool_contract_response(
            "content\nI can help with this repository.\n\ntool_calls\n[]\n\n[[ ## completed ## ]]",
        )
        .unwrap();

        assert_eq!(
            parsed.content.as_deref(),
            Some("I can help with this repository.")
        );
        assert!(parsed.tool_intents.is_empty());
    }

    #[test]
    fn ignores_placeholder_values_from_dsrs_template_text() {
        let parsed = parse_tool_contract_response(
            "Thinking Process...\n[[ ## content ## ]]\ncontent_value\n\n[[ ## tool_calls ## ]]\ntool_calls_value\n\n[[ ## completed ## ]]",
        )
        .unwrap();

        assert_eq!(parsed.content, None);
        assert!(parsed.tool_intents.is_empty());
    }

    #[test]
    fn extracts_last_dsrs_block_after_preamble() {
        let parsed = parse_tool_contract_response(
            "preamble [[ ## content ## ]]\ncontent_value\n[[ ## tool_calls ## ]]\ntool_calls_value\n[[ ## completed ## ]]\n[[ ## content ## ]]\nHello.\n[[ ## tool_calls ## ]]\n[]\n[[ ## completed ## ]]",
        )
        .unwrap();

        assert_eq!(parsed.content.as_deref(), Some("Hello."));
        assert!(parsed.tool_intents.is_empty());
    }

    #[test]
    fn parses_label_free_multiline_tool_call_section() {
        let parsed = parse_tool_contract_response(
            r#"content

tool_calls
[
  {
    "name": "bash",
    "arguments": {
      "command": "ls"
    }
  }
]

completed"#,
        )
        .unwrap();

        assert_eq!(parsed.content, None);
        assert_eq!(parsed.tool_intents.len(), 1);
        assert_eq!(parsed.tool_intents[0].name, "bash");
        assert_eq!(
            parsed.tool_intents[0].arguments.as_ref().unwrap()["command"],
            "ls"
        );
    }
}
