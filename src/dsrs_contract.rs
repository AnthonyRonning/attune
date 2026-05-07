use std::{
    fmt::Write,
    panic::{catch_unwind, AssertUnwindSafe},
};

use anyhow::Result;
use dspy_rs::{adapter::Adapter, example, ChatAdapter, Message, Signature};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    model_profile::{DsrsHistoryFormat, ModelProfile},
    normalizer::NormalizedRequest,
    openai::ChatMessage,
    response_interpreter::{
        push_failure, ResponseFailure, ResponseFailureKind, ToolIntent, ToolIntentSource,
    },
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
    pub failures: Vec<ResponseFailure>,
}

pub fn format_tool_contract(
    normalized: &NormalizedRequest,
    profile: &ModelProfile,
) -> Result<FormattedDsrsContract> {
    match profile.dsrs_history_format {
        DsrsHistoryFormat::AppendOnly => format_tool_contract_append_only(normalized, profile),
        DsrsHistoryFormat::RegeneratedContext => {
            format_tool_contract_regenerated_context(normalized, profile)
        }
    }
}

fn format_tool_contract_regenerated_context(
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
    let system_context = render_system_context(&system_messages)?;
    let conversation = render_conversation(&conversation_messages)?;
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

fn format_tool_contract_append_only(
    normalized: &NormalizedRequest,
    profile: &ModelProfile,
) -> Result<FormattedDsrsContract> {
    let (system_messages, conversation_messages): (Vec<_>, Vec<_>) = normalized
        .messages
        .iter()
        .cloned()
        .partition(|message| message.role == "system" || message.role == "developer");
    let system_context = render_system_context(&system_messages)?;
    let available_tools = serde_json::to_string_pretty(&normalized.tools)?;
    let tool_choice = normalized
        .tool_choice
        .as_ref()
        .map(serde_json::to_string)
        .transpose()?
        .unwrap_or_else(|| "auto".to_string());
    let instruction = dsrs_instruction(
        profile,
        &system_context,
        "Append-only conversation follows as chat messages.",
        &available_tools,
        &tool_choice,
        normalized.parallel_tool_calls,
    )?;

    let mut messages = vec![
        ChatMessage::new("system", instruction.clone()),
        ChatMessage::new(
            "user",
            render_append_only_runtime_context(
                profile,
                &system_context,
                &available_tools,
                &tool_choice,
                normalized.parallel_tool_calls,
            )?,
        ),
    ];
    for message in &conversation_messages {
        if let Some(rendered) = render_append_only_conversation_message(message)? {
            messages.push(rendered);
        }
    }

    Ok(FormattedDsrsContract {
        messages,
        instruction,
    })
}

fn dsrs_instruction(
    profile: &ModelProfile,
    system_context: &str,
    conversation: &str,
    available_tools: &str,
    tool_choice: &str,
    parallel_tool_calls: bool,
) -> Result<String> {
    let adapter = ChatAdapter;
    let signature = OpenAiToolUseContract::new();
    let chat = adapter.format(
        &signature,
        example! {
            "profile_guidance": "input" => profile.tool_instruction.clone(),
            "system_context": "input" => system_context.to_string(),
            "conversation": "input" => conversation.to_string(),
            "available_tools": "input" => available_tools.to_string(),
            "tool_choice": "input" => tool_choice.to_string(),
            "parallel_tool_calls": "input" => parallel_tool_calls,
        },
    );
    Ok(chat
        .messages
        .into_iter()
        .find_map(|message| match message {
            Message::System { content } => Some(content),
            _ => None,
        })
        .unwrap_or_default())
}

pub fn parse_tool_contract_response(content: &str) -> Option<ParsedDsrsContract> {
    if !contains_dsrs_marker(content) {
        return None;
    }

    let mut failures = classify_contract_violations(content);
    let adapter = ChatAdapter;
    let signature = OpenAiToolUseContract::new();
    let parsed = catch_unwind(AssertUnwindSafe(|| {
        adapter.parse_response(&signature, Message::assistant(content.to_string()))
    }));

    let parsed = match parsed {
        Ok(parsed) => parsed,
        Err(_) => {
            push_failure(
                &mut failures,
                ResponseFailureKind::DsrsContractViolation,
                "DSRs contract parser panicked on malformed output",
            );
            return Some(ParsedDsrsContract {
                content: None,
                tool_intents: Vec::new(),
                events: vec!["DSRs contract parser panicked on malformed output".to_string()],
                failures,
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
    let raw_tool_calls_field = fallback
        .tool_calls
        .as_deref()
        .and_then(clean_tool_calls_field)
        .or(parsed_tool_calls)
        .unwrap_or("[]");
    detect_placeholder_leaks(content, raw_tool_calls_field, &mut failures);
    let tool_calls_field = raw_tool_calls_field.trim();

    let tool_intents = parse_tool_calls_field(tool_calls_field, &mut events, &mut failures);
    if parsed_content
        .as_deref()
        .is_some_and(|content| !content.trim().is_empty())
        && !tool_intents.is_empty()
    {
        push_failure(
            &mut failures,
            ResponseFailureKind::DsrsContractViolation,
            "DSRs output contained user-facing content while also emitting tool calls",
        );
    }

    Some(ParsedDsrsContract {
        content: parsed_content,
        tool_intents,
        events,
        failures,
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

fn classify_contract_violations(content: &str) -> Vec<ResponseFailure> {
    let mut failures = Vec::new();

    if let Some(first_marker) = first_output_marker_index(content) {
        if !content[..first_marker].trim().is_empty() {
            push_failure(
                &mut failures,
                ResponseFailureKind::DsrsContractViolation,
                "DSRs output contained non-whitespace text before the first output marker",
            );
            push_failure(
                &mut failures,
                ResponseFailureKind::DsrsContentOutsideTaggedFields,
                "DSRs output contained content before the tagged fields",
            );
        }
    }

    if let Some((_, after_completed)) = content.rsplit_once("[[ ## completed ## ]]") {
        if !after_completed.trim().is_empty() {
            push_failure(
                &mut failures,
                ResponseFailureKind::DsrsContractViolation,
                "DSRs output contained non-whitespace text after the completed marker",
            );
            push_failure(
                &mut failures,
                ResponseFailureKind::DsrsContentOutsideTaggedFields,
                "DSRs output contained content after the completed marker",
            );
        }
    } else if contains_dsrs_marker(content) {
        push_failure(
            &mut failures,
            ResponseFailureKind::DsrsContractViolation,
            "DSRs output was missing the completed marker",
        );
    }

    if contains_input_marker(content) {
        push_failure(
            &mut failures,
            ResponseFailureKind::PromptEcho,
            "DSRs output repeated input-side prompt markers",
        );
    }

    if contains_template_artifact(content) {
        push_failure(
            &mut failures,
            ResponseFailureKind::TemplateLeak,
            "DSRs output contained prompt template placeholders or artifacts",
        );
    }

    failures
}

fn first_output_marker_index(content: &str) -> Option<usize> {
    ["[[ ## content ## ]]", "[[ ## tool_calls ## ]]"]
        .iter()
        .filter_map(|marker| content.find(marker))
        .min()
}

fn contains_input_marker(content: &str) -> bool {
    [
        "[[ ## profile_guidance ## ]]",
        "[[ ## system_context ## ]]",
        "[[ ## conversation ## ]]",
        "[[ ## available_tools ## ]]",
        "[[ ## tool_choice ## ]]",
        "[[ ## parallel_tool_calls ## ]]",
    ]
    .iter()
    .any(|marker| content.contains(marker))
}

fn contains_template_artifact(content: &str) -> bool {
    let lower = content.to_ascii_lowercase();
    [
        "content_value",
        "tool_calls_value",
        "tool_call_value",
        "completed_marker",
    ]
    .iter()
    .any(|artifact| lower.contains(artifact))
}

fn detect_placeholder_leaks(
    full_content: &str,
    raw_tool_calls_field: &str,
    failures: &mut Vec<ResponseFailure>,
) {
    let placeholder_present =
        contains_template_artifact(full_content) || is_placeholder_value(raw_tool_calls_field);
    if placeholder_present {
        push_failure(
            failures,
            ResponseFailureKind::DsrsPlaceholderOnly,
            "DSRs output contained placeholder-only field values",
        );
        push_failure(
            failures,
            ResponseFailureKind::TemplateLeak,
            "DSRs output contained prompt template placeholders or artifacts",
        );
    }
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

    ContractFallback::default()
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
    if let Some(rest) = strip_field_label_prefix(&cleaned, "content") {
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
    let cleaned = strip_field_label_prefix(value.trim(), "tool_calls")
        .unwrap_or(value)
        .trim();
    if is_placeholder_value(cleaned) || is_label(cleaned, "tool_calls") {
        None
    } else {
        Some(cleaned)
    }
}

fn strip_field_label_prefix<'a>(value: &'a str, label: &str) -> Option<&'a str> {
    let trimmed = value.trim_start();
    let rest = trimmed
        .strip_prefix(label)
        .or_else(|| strip_ascii_case_prefix(trimmed, label))?;
    let rest = rest.strip_prefix(':').unwrap_or(rest);
    if rest.starts_with(char::is_whitespace) || rest.is_empty() {
        Some(rest)
    } else {
        None
    }
}

fn strip_ascii_case_prefix<'a>(value: &'a str, prefix: &str) -> Option<&'a str> {
    value
        .get(..prefix.len())
        .is_some_and(|head| head.eq_ignore_ascii_case(prefix))
        .then(|| &value[prefix.len()..])
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

fn parse_tool_calls_field(
    field: &str,
    events: &mut Vec<String>,
    failures: &mut Vec<ResponseFailure>,
) -> Vec<ToolIntent> {
    if field.is_empty() || field == "[]" {
        return Vec::new();
    }

    let Some(value) = parse_jsonish(field).or_else(|| parse_adjacent_tool_call_values(field))
    else {
        events.push("DSRs tool_calls field was not valid JSON".to_string());
        push_failure(
            failures,
            ResponseFailureKind::DsrsInvalidToolCallsJson,
            "DSRs tool_calls field was not valid JSON",
        );
        push_failure(
            failures,
            ResponseFailureKind::DsrsContractViolation,
            "DSRs tool_calls field violated the tool-use contract",
        );
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
            push_failure(
                failures,
                ResponseFailureKind::DsrsInvalidToolCallsShape,
                "DSRs tool_calls field was not a JSON array or object",
            );
            push_failure(
                failures,
                ResponseFailureKind::DsrsContractViolation,
                "DSRs tool_calls field violated the tool-use contract",
            );
            return Vec::new();
        }
    };

    calls
        .into_iter()
        .filter_map(|call| tool_intent_from_value(call, events, failures))
        .collect()
}

fn parse_adjacent_tool_call_values(field: &str) -> Option<Value> {
    let values = parse_json_sequence(field)?;
    let mut calls = Vec::new();
    for value in values {
        match value {
            Value::Array(items) => calls.extend(items),
            Value::Object(mut object) => match object.remove("tool_calls") {
                Some(Value::Array(items)) => calls.extend(items),
                _ => calls.push(Value::Object(object)),
            },
            _ => return None,
        }
    }
    (!calls.is_empty()).then(|| Value::Array(calls))
}

fn parse_json_sequence(input: &str) -> Option<Vec<Value>> {
    let mut values = Vec::new();
    let mut offset = 0usize;
    let input = input.trim();

    while offset < input.len() {
        let rest = input[offset..].trim_start();
        offset = input.len() - rest.len();
        if rest.is_empty() {
            break;
        }

        let end = json_value_end(rest)?;
        let value_text = &rest[..end];
        let value = parse_jsonish(value_text)?;
        values.push(value);
        offset += end;
    }

    (values.len() > 1).then_some(values)
}

fn json_value_end(input: &str) -> Option<usize> {
    let mut chars = input.char_indices();
    let (_, first) = chars.next()?;
    let (open, close) = match first {
        '[' => ('[', ']'),
        '{' => ('{', '}'),
        _ => return None,
    };
    let mut stack = vec![open];
    let mut in_string = false;
    let mut escaped = false;

    for (index, ch) in chars {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '[' | '{' => stack.push(ch),
            ']' | '}' => {
                let expected = match stack.pop()? {
                    '[' => ']',
                    '{' => '}',
                    _ => return None,
                };
                if ch != expected {
                    return None;
                }
                if stack.is_empty() {
                    return Some(index + ch.len_utf8());
                }
            }
            _ => {}
        }
    }

    let _ = close;
    None
}

fn tool_intent_from_value(
    call: Value,
    events: &mut Vec<String>,
    failures: &mut Vec<ResponseFailure>,
) -> Option<ToolIntent> {
    let Value::Object(mut object) = call else {
        events.push("DSRs tool call item was not an object".to_string());
        push_failure(
            failures,
            ResponseFailureKind::DsrsInvalidToolCallsShape,
            "DSRs tool call item was not an object",
        );
        push_failure(
            failures,
            ResponseFailureKind::DsrsContractViolation,
            "DSRs tool_calls field violated the tool-use contract",
        );
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
        .or(function_name);

    let Some(name) = name else {
        events.push("DSRs tool call item did not contain a tool name".to_string());
        push_failure(
            failures,
            ResponseFailureKind::DsrsInvalidToolCallsShape,
            "DSRs tool call item did not contain a tool name",
        );
        push_failure(
            failures,
            ResponseFailureKind::DsrsContractViolation,
            "DSRs tool_calls field violated the tool-use contract",
        );
        return None;
    };

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

fn render_system_context(messages: &[ChatMessage]) -> Result<String> {
    if messages.is_empty() {
        return Ok("No system or developer messages.".to_string());
    }

    let mut out = String::new();
    for (index, message) in messages.iter().enumerate() {
        writeln!(&mut out, "[{index}] role: {}", message.role)?;
        if let Some(name) = &message.name {
            writeln!(&mut out, "name: {name}")?;
        }
        writeln!(&mut out, "content:\n{}\n", render_message_content(message)?)?;
    }
    Ok(out.trim_end().to_string())
}

fn render_conversation(messages: &[ChatMessage]) -> Result<String> {
    if messages.is_empty() {
        return Ok("No non-system conversation messages.".to_string());
    }

    let mut out = String::new();
    for (index, message) in messages.iter().enumerate() {
        writeln!(&mut out, "[{index}] role: {}", message.role)?;
        if let Some(name) = &message.name {
            writeln!(&mut out, "name: {name}")?;
        }
        if let Some(tool_call_id) = &message.tool_call_id {
            writeln!(&mut out, "tool_call_id: {tool_call_id}")?;
        }
        writeln!(&mut out, "content:\n{}", render_message_content(message)?)?;
        if let Some(tool_calls) = &message.tool_calls {
            let rendered = tool_calls
                .iter()
                .map(|call| {
                    json!({
                        "id": call.id,
                        "name": call.function.name,
                        "arguments": parse_jsonish(&call.function.arguments)
                            .unwrap_or_else(|| Value::String(call.function.arguments.clone()))
                    })
                })
                .collect::<Vec<_>>();
            writeln!(
                &mut out,
                "assistant_tool_calls:\n{}",
                serde_json::to_string_pretty(&rendered)?
            )?;
        }
        writeln!(&mut out)?;
    }
    Ok(out.trim_end().to_string())
}

fn render_append_only_runtime_context(
    profile: &ModelProfile,
    system_context: &str,
    available_tools: &str,
    tool_choice: &str,
    parallel_tool_calls: bool,
) -> Result<String> {
    let mut out = String::new();
    writeln!(&mut out, "[[ ## profile_guidance ## ]]")?;
    writeln!(&mut out, "{}", profile.tool_instruction)?;
    writeln!(&mut out)?;
    writeln!(&mut out, "[[ ## system_context ## ]]")?;
    writeln!(&mut out, "{system_context}")?;
    writeln!(&mut out)?;
    writeln!(&mut out, "[[ ## conversation ## ]]")?;
    writeln!(
        &mut out,
        "Append-only conversation follows as chat messages."
    )?;
    writeln!(&mut out)?;
    writeln!(&mut out, "[[ ## available_tools ## ]]")?;
    writeln!(&mut out, "{available_tools}")?;
    writeln!(&mut out)?;
    writeln!(&mut out, "[[ ## tool_choice ## ]]")?;
    writeln!(&mut out, "{tool_choice}")?;
    writeln!(&mut out)?;
    writeln!(&mut out, "[[ ## parallel_tool_calls ## ]]")?;
    writeln!(&mut out, "{parallel_tool_calls}")?;
    writeln!(&mut out)?;
    writeln!(
        &mut out,
        "The remaining chat messages are append-only conversation history. Prior assistant messages use the same DSRs content/tool_calls output format you must use now. Tool-result messages may use a tool_result field marker and are observations, not an output field you should emit."
    )?;
    Ok(out.trim_end().to_string())
}

fn render_append_only_conversation_message(message: &ChatMessage) -> Result<Option<ChatMessage>> {
    match message.role.as_str() {
        "user" => Ok(Some(render_append_only_user_message(message))),
        "assistant" => Ok(Some(ChatMessage::new(
            "assistant",
            render_assistant_as_dsrs_history(message)?,
        ))),
        "tool" => Ok(Some(ChatMessage::new(
            "user",
            render_tool_result_as_dsrs_history(message)?,
        ))),
        _ => Ok(Some(ChatMessage::new(
            "user",
            render_observed_message_as_history(message)?,
        ))),
    }
}

fn render_append_only_user_message(message: &ChatMessage) -> ChatMessage {
    ChatMessage {
        role: "user".to_string(),
        content: message.content.clone(),
        name: message.name.clone(),
        extra: message.extra.clone(),
        ..ChatMessage::default()
    }
}

fn render_assistant_as_dsrs_history(message: &ChatMessage) -> Result<String> {
    let mut out = String::new();
    writeln!(&mut out, "[[ ## content ## ]]")?;
    writeln!(&mut out, "{}", render_message_content(message)?)?;
    writeln!(&mut out, "[[ ## tool_calls ## ]]")?;
    writeln!(
        &mut out,
        "{}",
        serde_json::to_string_pretty(&render_tool_calls_for_dsrs_history(message)?)?
    )?;
    writeln!(&mut out, "[[ ## completed ## ]]")?;
    Ok(out.trim_end().to_string())
}

fn render_tool_calls_for_dsrs_history(message: &ChatMessage) -> Result<Vec<Value>> {
    let Some(tool_calls) = &message.tool_calls else {
        return Ok(Vec::new());
    };

    tool_calls
        .iter()
        .map(|call| {
            Ok(json!({
                "name": call.function.name,
                "arguments": parse_jsonish(&call.function.arguments)
                    .unwrap_or_else(|| Value::String(call.function.arguments.clone()))
            }))
        })
        .collect()
}

fn render_tool_result_as_dsrs_history(message: &ChatMessage) -> Result<String> {
    let mut out = String::new();
    writeln!(&mut out, "[[ ## tool_result ## ]]")?;
    if let Some(tool_call_id) = &message.tool_call_id {
        writeln!(&mut out, "tool_call_id: {tool_call_id}")?;
    }
    writeln!(&mut out, "content:")?;
    writeln!(&mut out, "{}", render_message_content(message)?)?;
    writeln!(&mut out, "[[ ## completed ## ]]")?;
    Ok(out.trim_end().to_string())
}

fn render_observed_message_as_history(message: &ChatMessage) -> Result<String> {
    let mut out = String::new();
    writeln!(&mut out, "[[ ## observed_message ## ]]")?;
    writeln!(&mut out, "role: {}", message.role)?;
    if let Some(name) = &message.name {
        writeln!(&mut out, "name: {name}")?;
    }
    writeln!(&mut out, "content:")?;
    writeln!(&mut out, "{}", render_message_content(message)?)?;
    writeln!(&mut out, "[[ ## completed ## ]]")?;
    Ok(out.trim_end().to_string())
}

fn render_message_content(message: &ChatMessage) -> Result<String> {
    match &message.content {
        Some(Value::String(text)) => Ok(text.clone()),
        Some(Value::Array(parts)) => {
            let mut text = String::new();
            for part in parts {
                if let Some(part_text) = part.get("text").and_then(Value::as_str) {
                    text.push_str(part_text);
                }
            }
            if text.is_empty() {
                Ok(serde_json::to_string_pretty(parts)?)
            } else {
                Ok(text)
            }
        }
        Some(Value::Null) | None => Ok(String::new()),
        Some(value) => Ok(serde_json::to_string_pretty(value)?),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Map};

    use super::*;
    use crate::openai::{ChatCompletionRequest, OpenAiFunctionTool, OpenAiTool, OpenAiToolCall};

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
    fn formats_prior_assistant_tool_calls_and_tool_results_in_conversation() {
        let mut assistant = ChatMessage::new("assistant", "");
        assistant.tool_calls = Some(vec![OpenAiToolCall::function(
            "read",
            r#"{"path":"README.md"}"#,
        )]);
        let mut tool = ChatMessage::new("tool", "README contents");
        tool.tool_call_id = assistant
            .tool_calls
            .as_ref()
            .map(|calls| calls[0].id.clone());
        let mut request = normalized_request();
        request.messages = vec![
            ChatMessage::new("system", "Do not repeat this."),
            ChatMessage::new("user", "read README"),
            assistant,
            tool,
            ChatMessage::new("user", "summarize it"),
        ];

        let formatted = format_tool_contract(&request, &ModelProfile::qwen()).unwrap();
        let assistant_message = formatted
            .messages
            .iter()
            .find(|message| message.role == "assistant")
            .and_then(ChatMessage::content_text)
            .unwrap();
        let tool_result_message = formatted
            .messages
            .iter()
            .find(|message| {
                message.role == "user"
                    && message
                        .content_text()
                        .is_some_and(|content| content.contains("[[ ## tool_result ## ]]"))
            })
            .and_then(ChatMessage::content_text)
            .unwrap();

        assert!(assistant_message.contains("[[ ## content ## ]]"));
        assert!(assistant_message.contains("[[ ## tool_calls ## ]]"));
        assert!(assistant_message.contains("\"name\": \"read\""));
        assert!(assistant_message.contains("\"path\": \"README.md\""));
        assert!(!assistant_message.contains("assistant_tool_calls"));
        assert!(tool_result_message.contains("tool_call_id:"));
        assert!(tool_result_message.contains("README contents"));
    }

    #[test]
    fn regenerated_context_history_format_preserves_legacy_transcript() {
        let mut assistant = ChatMessage::new("assistant", "");
        assistant.tool_calls = Some(vec![OpenAiToolCall::function(
            "read",
            r#"{"path":"README.md"}"#,
        )]);
        let mut tool = ChatMessage::new("tool", "README contents");
        tool.tool_call_id = assistant
            .tool_calls
            .as_ref()
            .map(|calls| calls[0].id.clone());
        let mut request = normalized_request();
        request.messages = vec![
            ChatMessage::new("system", "Do not repeat this."),
            ChatMessage::new("user", "read README"),
            assistant,
            tool,
            ChatMessage::new("user", "summarize it"),
        ];
        let mut profile = ModelProfile::qwen();
        profile.dsrs_history_format = DsrsHistoryFormat::RegeneratedContext;

        let formatted = format_tool_contract(&request, &profile).unwrap();
        let user_message = formatted.messages[1].content_text().unwrap();

        assert!(user_message.contains("assistant_tool_calls"));
        assert!(user_message.contains("\"name\": \"read\""));
        assert!(user_message.contains("\"path\": \"README.md\""));
        assert!(user_message.contains("role: tool"));
        assert!(user_message.contains("README contents"));
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
    fn parses_tagged_dsrs_with_redundant_inner_field_labels() {
        let parsed = parse_tool_contract_response(
            r#"[[ ## content ## ]]
content:
[[ ## tool_calls ## ]]
tool_calls: [
  {"name":"read","arguments":{"path":"packages/agent/README.md"}},
  {"name":"read","arguments":{"path":"packages/ai/README.md"}}
]
[[ ## completed ## ]]"#,
        )
        .unwrap();

        assert_eq!(parsed.content, None);
        assert_eq!(parsed.tool_intents.len(), 2);
        assert_eq!(parsed.tool_intents[0].name, "read");
        assert_eq!(
            parsed.tool_intents[0].arguments.as_ref().unwrap()["path"],
            "packages/agent/README.md"
        );
    }

    #[test]
    fn parses_tagged_dsrs_with_adjacent_tool_call_arrays() {
        let parsed = parse_tool_contract_response(
            r#"[[ ## content ## ]]

[[ ## tool_calls ## ]]
[{"name":"bash","arguments":{"command":"ls packages"}}]
[{"name":"read","arguments":{"path":"packages/ai/README.md","limit":100}}]
[{"name":"read","arguments":{"path":"packages/tui/README.md","limit":100}}]
[[ ## completed ## ]]"#,
        )
        .unwrap();

        assert_eq!(parsed.content, None);
        assert_eq!(parsed.tool_intents.len(), 3);
        assert_eq!(parsed.tool_intents[0].name, "bash");
        assert_eq!(
            parsed.tool_intents[1].arguments.as_ref().unwrap()["path"],
            "packages/ai/README.md"
        );
    }

    #[test]
    fn reports_content_and_tool_calls_as_contract_violation() {
        let parsed = parse_tool_contract_response(
            r#"[[ ## content ## ]]
I will inspect the package list first.

[[ ## tool_calls ## ]]
[{"name":"bash","arguments":{"command":"ls packages"}}]
[[ ## completed ## ]]"#,
        )
        .unwrap();

        assert_eq!(
            parsed.content.as_deref(),
            Some("I will inspect the package list first.")
        );
        assert_eq!(parsed.tool_intents.len(), 1);
        assert!(parsed.failures.iter().any(|failure| failure.kind
            == ResponseFailureKind::DsrsContractViolation
            && failure
                .detail
                .contains("content while also emitting tool calls")));
    }

    #[test]
    fn reports_content_outside_tagged_fields() {
        let parsed = parse_tool_contract_response(
            "preamble\n[[ ## content ## ]]\nHello.\n[[ ## tool_calls ## ]]\n[]\n[[ ## completed ## ]]",
        )
        .unwrap();

        assert_eq!(parsed.content.as_deref(), Some("Hello."));
        assert!(parsed
            .failures
            .iter()
            .any(|failure| failure.kind == ResponseFailureKind::DsrsContentOutsideTaggedFields));
        assert!(parsed
            .failures
            .iter()
            .any(|failure| failure.kind == ResponseFailureKind::DsrsContractViolation));
    }

    #[test]
    fn reports_invalid_dsrs_tool_calls_json() {
        let parsed = parse_tool_contract_response(
            "[[ ## content ## ]]\n---\n[[ ## tool_calls ## ]]\n[{\"name\":\"bash\",\"arguments\":]\n[[ ## completed ## ]]",
        )
        .unwrap();

        assert!(parsed.tool_intents.is_empty());
        assert!(parsed
            .events
            .iter()
            .any(|event| event.contains("not valid JSON")));
        assert!(parsed
            .failures
            .iter()
            .any(|failure| failure.kind == ResponseFailureKind::DsrsInvalidToolCallsJson));
    }

    #[test]
    fn reports_prompt_echo_and_placeholder_leak() {
        let parsed = parse_tool_contract_response(
            "[[ ## system_context ## ]]\nDo not repeat this.\n[[ ## content ## ]]\ncontent_value\n[[ ## tool_calls ## ]]\ntool_calls_value\n[[ ## completed ## ]]",
        )
        .unwrap();

        assert!(parsed.content.is_none());
        assert!(parsed
            .failures
            .iter()
            .any(|failure| failure.kind == ResponseFailureKind::PromptEcho));
        assert!(parsed
            .failures
            .iter()
            .any(|failure| failure.kind == ResponseFailureKind::DsrsPlaceholderOnly));
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
    fn rejects_label_free_empty_tool_calls_tail() {
        let parsed = parse_tool_contract_response("Hello! How can I help?\n\n[]\n\ncompleted");

        assert!(parsed.is_none());
    }

    #[test]
    fn rejects_label_free_named_tool_calls_section() {
        let parsed = parse_tool_contract_response("content\nHello!\n\ntool_calls\n[]\n\ncompleted");

        assert!(parsed.is_none());
    }

    #[test]
    fn rejects_label_free_output_with_completed_marker() {
        let parsed = parse_tool_contract_response(
            "content\nI can help with this repository.\n\ntool_calls\n[]\n\n[[ ## completed ## ]]",
        );

        assert!(parsed.is_none());
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
    fn rejects_label_free_multiline_tool_call_section() {
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
        );

        assert!(parsed.is_none());
    }
}
