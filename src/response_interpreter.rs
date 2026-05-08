use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    dsrs_contract::parse_tool_contract_response,
    openai::{ChatCompletionResponse, OpenAiTool},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpretedResponse {
    pub content: Option<String>,
    pub reasoning: Option<String>,
    pub finish_reason: Option<String>,
    pub tool_intents: Vec<ToolIntent>,
    pub parse_events: Vec<String>,
    #[serde(default)]
    pub failures: Vec<ResponseFailure>,
    pub suspicious_stop: bool,
}

impl InterpretedResponse {
    pub fn has_failure(&self, kind: ResponseFailureKind) -> bool {
        self.failures.iter().any(|failure| failure.kind == kind)
    }

    pub fn has_any_failure(&self, kinds: &[ResponseFailureKind]) -> bool {
        self.failures
            .iter()
            .any(|failure| kinds.contains(&failure.kind))
    }

    pub fn failure_kinds(&self) -> Vec<ResponseFailureKind> {
        self.failures.iter().map(|failure| failure.kind).collect()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ResponseFailure {
    pub kind: ResponseFailureKind,
    pub detail: String,
}

impl ResponseFailure {
    pub fn new(kind: ResponseFailureKind, detail: impl Into<String>) -> Self {
        Self {
            kind,
            detail: detail.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseFailureKind {
    NoChoices,
    NativeMalformedJsonArguments,
    DsrsContractViolation,
    DsrsContentOutsideTaggedFields,
    DsrsInvalidToolCallsJson,
    DsrsInvalidToolCallsShape,
    EmptyDsrsOutput,
    DsrsPlaceholderOnly,
    EmptyAssistantOutput,
    TemplateLeak,
    PromptEcho,
    PrematureToolStop,
    MalformedKnownToolCall,
    UntaggedDsrsLikeOutput,
    SchemaViolation,
    UnknownTool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolIntent {
    pub name: String,
    pub arguments: Option<Value>,
    pub raw_arguments: String,
    pub source: ToolIntentSource,
    pub confidence: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolIntentSource {
    Native,
    Dsrs,
    Xml,
    TaggedJson,
    MarkdownJson,
    FunctionLikeText,
    CorrectionAgent,
}

pub fn interpret_response(
    response: &ChatCompletionResponse,
    tools: &[OpenAiTool],
) -> InterpretedResponse {
    let Some(choice) = response.choices.first() else {
        return InterpretedResponse {
            content: None,
            reasoning: None,
            finish_reason: None,
            tool_intents: Vec::new(),
            parse_events: vec!["upstream response contained no choices".to_string()],
            failures: vec![ResponseFailure::new(
                ResponseFailureKind::NoChoices,
                "upstream response contained no choices",
            )],
            suspicious_stop: true,
        };
    };

    let message = &choice.message;
    let mut parse_events = Vec::new();
    let mut content = message.content_text();
    let reasoning = message.reasoning_text();
    if content.is_none() && reasoning.is_some() && !tools_available(tools) {
        content = reasoning.clone();
        parse_events
            .push("mapped reasoning/thinking text into content for compatibility".to_string());
    }

    let mut tool_intents = Vec::new();
    let mut failures = Vec::new();
    let mut parsed_dsrs_empty_output = false;
    if let Some(native_calls) = &message.tool_calls {
        for call in native_calls {
            let parsed = parse_jsonish(&call.function.arguments);
            if parsed.is_none() {
                parse_events.push(format!(
                    "native tool call {} had malformed JSON arguments",
                    call.function.name
                ));
                push_failure(
                    &mut failures,
                    ResponseFailureKind::NativeMalformedJsonArguments,
                    format!(
                        "native tool call {} had malformed JSON arguments",
                        call.function.name
                    ),
                );
            }
            tool_intents.push(ToolIntent {
                name: call.function.name.clone(),
                arguments: parsed,
                raw_arguments: call.function.arguments.clone(),
                source: ToolIntentSource::Native,
                confidence: 1.0,
            });
        }
    }

    if let Some(text) = content.clone() {
        let mut parsed_dsrs = false;
        if let Some(parsed) = parse_tool_contract_response(&text) {
            parsed_dsrs = true;
            parse_events.extend(parsed.events);
            failures.extend(parsed.failures);
            let parsed_content_noop = parsed
                .content
                .as_deref()
                .map(dsrs_content_is_empty_or_noop)
                .unwrap_or(true);
            let parsed_tool_intents = parsed.tool_intents;
            parsed_dsrs_empty_output = parsed_tool_intents.is_empty() && parsed_content_noop;
            content = if !parsed_tool_intents.is_empty() && parsed_content_noop {
                None
            } else {
                parsed
                    .content
                    .or_else(|| parsed_tool_intents.is_empty().then(String::new))
            };
            tool_intents.extend(parsed_tool_intents);
        }

        if !parsed_dsrs {
            tool_intents.extend(extract_tool_intents_from_content(
                &text,
                tools,
                &mut parse_events,
            ));
            if tool_intents.is_empty() && tools_available(tools) {
                if let Some(cleaned) = strip_stray_text_label(&text) {
                    content = Some(cleaned);
                    parse_events.push("stripped stray DSRs text label".to_string());
                }
            }
            if looks_like_untagged_dsrs_contract(&text) {
                push_failure(
                    &mut failures,
                    ResponseFailureKind::UntaggedDsrsLikeOutput,
                    "assistant emitted DSRs-like field labels without valid DSRs output markers",
                );
            }
        }
    }

    classify_intent_failures(&tool_intents, tools, &mut failures);

    if parsed_dsrs_empty_output {
        parse_events
            .push("DSRs response contained empty or no-op content and no tool calls".to_string());
        push_failure(
            &mut failures,
            ResponseFailureKind::EmptyDsrsOutput,
            "assistant emitted a valid DSRs envelope with empty or no-op content and empty tool_calls",
        );
    }

    let content_suggests_premature_tool = content
        .as_deref()
        .map(|content| looks_like_premature_tool_narration(content, tools))
        .unwrap_or(false);
    let content_suggests_malformed_tool = content
        .as_deref()
        .map(|content| looks_like_malformed_tool_output(content, tools))
        .unwrap_or(false);
    let finish_reason = choice.finish_reason.as_deref().unwrap_or("stop");
    let suspicious_tool_stop = tool_intents.is_empty()
        && tools_available(tools)
        && finish_reason == "stop"
        && (content_suggests_premature_tool || content_suggests_malformed_tool);
    let empty_visible_output_stop = !parsed_dsrs_empty_output
        && tool_intents.is_empty()
        && tools_available(tools)
        && matches!(finish_reason, "stop" | "length")
        && content
            .as_deref()
            .map(|content| content.trim().is_empty())
            .unwrap_or(true)
        && message
            .tool_calls
            .as_ref()
            .map(|calls| calls.is_empty())
            .unwrap_or(true);
    let suspicious_stop =
        suspicious_tool_stop || parsed_dsrs_empty_output || empty_visible_output_stop;

    if suspicious_tool_stop {
        parse_events.push(
            "assistant appears to contain tool intent but emitted no valid tool call".to_string(),
        );
        if content_suggests_premature_tool {
            push_failure(
                &mut failures,
                ResponseFailureKind::PrematureToolStop,
                "assistant narrated an intent to use a tool but stopped without a tool call",
            );
        }
        if content_suggests_malformed_tool {
            push_failure(
                &mut failures,
                ResponseFailureKind::MalformedKnownToolCall,
                "assistant emitted malformed structured tool-call output",
            );
        }
    }
    if empty_visible_output_stop {
        if reasoning
            .as_deref()
            .is_some_and(|reasoning| !reasoning.trim().is_empty())
        {
            parse_events.push(
                "assistant emitted reasoning/thinking but no visible content or tool calls"
                    .to_string(),
            );
        } else {
            parse_events.push("assistant emitted empty content and no tool calls".to_string());
        }
        push_failure(
            &mut failures,
            ResponseFailureKind::EmptyAssistantOutput,
            "assistant stopped without visible content or tool calls",
        );
    }

    InterpretedResponse {
        content,
        reasoning,
        finish_reason: choice.finish_reason.clone(),
        tool_intents,
        parse_events,
        failures,
        suspicious_stop,
    }
}

pub fn extract_tool_intents_from_content(
    content: &str,
    tools: &[OpenAiTool],
    parse_events: &mut Vec<String>,
) -> Vec<ToolIntent> {
    let mut intents = Vec::new();
    extract_named_xml(content, parse_events, &mut intents);
    extract_nested_xml(content, parse_events, &mut intents);
    extract_wrapped_json_xml(content, parse_events, &mut intents);
    extract_markdown_json(content, parse_events, &mut intents);
    extract_direct_json(
        content,
        ToolIntentSource::TaggedJson,
        parse_events,
        &mut intents,
    );
    extract_function_like(content, tools, parse_events, &mut intents);
    intents
}

fn extract_named_xml(content: &str, parse_events: &mut Vec<String>, intents: &mut Vec<ToolIntent>) {
    let patterns = [
        r#"(?s)<tool_call\s+name=["']([^"']+)["']\s*>(.*?)</tool_call>"#,
        r#"(?s)<tool\s+name=["']([^"']+)["']\s*>(.*?)</tool>"#,
        r#"(?s)<function\s*=\s*["']?([^>"'\s]+)["']?\s*>(.*?)</function>"#,
    ];

    for pattern in patterns {
        let Ok(regex) = Regex::new(pattern) else {
            continue;
        };
        for capture in regex.captures_iter(content) {
            let name = capture.get(1).map(|m| m.as_str()).unwrap_or_default();
            let body = capture
                .get(2)
                .map(|m| m.as_str())
                .unwrap_or_default()
                .trim();
            let arguments = parse_jsonish(body);
            if arguments.is_none() {
                parse_events.push(format!("XML tool call {name} had malformed JSON body"));
            }
            intents.push(ToolIntent {
                name: name.to_string(),
                arguments,
                raw_arguments: body.to_string(),
                source: ToolIntentSource::Xml,
                confidence: 0.95,
            });
        }
    }
}

fn extract_nested_xml(
    content: &str,
    parse_events: &mut Vec<String>,
    intents: &mut Vec<ToolIntent>,
) {
    let Ok(regex) = Regex::new(
        r#"(?s)<tool_call>\s*<name>\s*([^<]+?)\s*</name>\s*<(?:arguments|args)>\s*(.*?)\s*</(?:arguments|args)>\s*</tool_call>"#,
    ) else {
        return;
    };
    for capture in regex.captures_iter(content) {
        let name = capture
            .get(1)
            .map(|m| m.as_str())
            .unwrap_or_default()
            .trim();
        let body = capture
            .get(2)
            .map(|m| m.as_str())
            .unwrap_or_default()
            .trim();
        let arguments = parse_jsonish(body);
        if arguments.is_none() {
            parse_events.push(format!(
                "nested XML tool call {name} had malformed JSON body"
            ));
        }
        intents.push(ToolIntent {
            name: name.to_string(),
            arguments,
            raw_arguments: body.to_string(),
            source: ToolIntentSource::Xml,
            confidence: 0.92,
        });
    }
}

fn extract_wrapped_json_xml(
    content: &str,
    parse_events: &mut Vec<String>,
    intents: &mut Vec<ToolIntent>,
) {
    let Ok(regex) = Regex::new(
        r#"(?s)<(?:tool_call|tool_calls_json)>\s*(.*?)\s*</(?:tool_call|tool_calls_json)>"#,
    ) else {
        return;
    };
    for capture in regex.captures_iter(content) {
        if let Some(body) = capture.get(1).map(|m| m.as_str()) {
            extract_json_payload(body, ToolIntentSource::Xml, parse_events, intents);
        }
    }
}

fn extract_markdown_json(
    content: &str,
    parse_events: &mut Vec<String>,
    intents: &mut Vec<ToolIntent>,
) {
    let Ok(regex) = Regex::new(r#"(?s)```(?:json|tool_calls?|function_call)?\s*(.*?)\s*```"#)
    else {
        return;
    };
    for capture in regex.captures_iter(content) {
        if let Some(body) = capture.get(1).map(|m| m.as_str()) {
            extract_json_payload(body, ToolIntentSource::MarkdownJson, parse_events, intents);
        }
    }
}

fn extract_direct_json(
    content: &str,
    source: ToolIntentSource,
    parse_events: &mut Vec<String>,
    intents: &mut Vec<ToolIntent>,
) {
    let trimmed = content.trim();
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        extract_json_payload(trimmed, source, parse_events, intents);
    }
}

fn extract_json_payload(
    payload: &str,
    source: ToolIntentSource,
    parse_events: &mut Vec<String>,
    intents: &mut Vec<ToolIntent>,
) {
    let Some(value) = parse_jsonish(payload) else {
        parse_events.push("structured JSON-like tool payload could not be parsed".to_string());
        return;
    };

    match value {
        Value::Array(items) => {
            for item in items {
                push_tool_intent_from_value(item, source.clone(), intents);
            }
        }
        Value::Object(mut object) => {
            if let Some(Value::Array(items)) = object.remove("tool_calls") {
                for item in items {
                    push_tool_intent_from_value(item, source.clone(), intents);
                }
            } else if let Some(Value::Object(call)) = object.remove("function_call") {
                push_tool_intent_from_value(Value::Object(call), source, intents);
            } else {
                push_tool_intent_from_value(Value::Object(object), source, intents);
            }
        }
        _ => {}
    }
}

fn push_tool_intent_from_value(
    value: Value,
    source: ToolIntentSource,
    intents: &mut Vec<ToolIntent>,
) {
    let Value::Object(mut object) = value else {
        return;
    };

    let name = object
        .remove("name")
        .or_else(|| object.remove("tool_name"))
        .or_else(|| object.remove("function"))
        .and_then(|value| value.as_str().map(str::to_owned));

    let Some(name) = name else {
        return;
    };

    let arguments = object
        .remove("arguments")
        .or_else(|| object.remove("args"))
        .unwrap_or_else(|| Value::Object(object));
    let raw_arguments = serde_json::to_string(&arguments).unwrap_or_else(|_| "{}".to_string());

    intents.push(ToolIntent {
        name,
        arguments: Some(arguments),
        raw_arguments,
        source,
        confidence: 0.9,
    });
}

fn extract_function_like(
    content: &str,
    tools: &[OpenAiTool],
    parse_events: &mut Vec<String>,
    intents: &mut Vec<ToolIntent>,
) {
    for tool in tools {
        let pattern = format!(
            r#"(?s)\b{}\s*\((\{{.*?\}})\)"#,
            regex::escape(&tool.function.name)
        );
        let Ok(regex) = Regex::new(&pattern) else {
            continue;
        };
        for capture in regex.captures_iter(content) {
            let body = capture.get(1).map(|m| m.as_str()).unwrap_or("{}");
            let arguments = parse_jsonish(body);
            if arguments.is_none() {
                parse_events.push(format!(
                    "function-like tool call {} had malformed JSON",
                    tool.function.name
                ));
            }
            intents.push(ToolIntent {
                name: tool.function.name.clone(),
                arguments,
                raw_arguments: body.to_string(),
                source: ToolIntentSource::FunctionLikeText,
                confidence: 0.75,
            });
        }
    }
}

fn parse_jsonish(input: &str) -> Option<Value> {
    let trimmed = input.trim();
    serde_json::from_str(trimmed)
        .ok()
        .or_else(|| json5::from_str(trimmed).ok())
}

fn dsrs_content_is_empty_or_noop(content: &str) -> bool {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return true;
    }

    parse_jsonish(trimmed).is_some_and(|value| match value {
        Value::Null => true,
        Value::Array(items) => items.is_empty(),
        Value::Object(object) => object.is_empty(),
        _ => false,
    })
}

fn tools_available(tools: &[OpenAiTool]) -> bool {
    !tools.is_empty()
}

fn classify_intent_failures(
    intents: &[ToolIntent],
    tools: &[OpenAiTool],
    failures: &mut Vec<ResponseFailure>,
) {
    for intent in intents {
        let Some(tool) = tools.iter().find(|tool| tool.function.name == intent.name) else {
            if !tools.is_empty() {
                push_failure(
                    failures,
                    ResponseFailureKind::UnknownTool,
                    format!("assistant requested unknown tool {:?}", intent.name),
                );
            }
            continue;
        };

        if intent.arguments.is_none() {
            push_failure(
                failures,
                ResponseFailureKind::MalformedKnownToolCall,
                format!(
                    "assistant requested tool {:?} with malformed arguments",
                    intent.name
                ),
            );
            continue;
        }

        if intent
            .arguments
            .as_ref()
            .is_some_and(|arguments| !arguments_have_required_properties(tool, arguments))
        {
            push_failure(
                failures,
                ResponseFailureKind::SchemaViolation,
                format!(
                    "assistant requested tool {:?} without required schema properties",
                    intent.name
                ),
            );
        }
    }
}

fn arguments_have_required_properties(tool: &OpenAiTool, value: &Value) -> bool {
    let Some(required) = tool
        .function
        .parameters
        .get("required")
        .and_then(Value::as_array)
    else {
        return true;
    };
    if required.is_empty() {
        return true;
    }
    let Some(object) = value.as_object() else {
        return false;
    };
    required.iter().filter_map(Value::as_str).all(|key| {
        object
            .get(key)
            .is_some_and(|value| !value.is_null() && value.as_str().is_none_or(|s| !s.is_empty()))
    })
}

pub(crate) fn push_failure(
    failures: &mut Vec<ResponseFailure>,
    kind: ResponseFailureKind,
    detail: impl Into<String>,
) {
    let detail = detail.into();
    if failures
        .iter()
        .any(|failure| failure.kind == kind && failure.detail == detail)
    {
        return;
    }
    failures.push(ResponseFailure::new(kind, detail));
}

fn looks_like_premature_tool_narration(content: &str, tools: &[OpenAiTool]) -> bool {
    let lower = content.to_ascii_lowercase();
    let action_phrase = [
        "i'll read",
        "i will read",
        "i'm going to read",
        "i’m going to read",
        "let me read",
        "i'll inspect",
        "i will inspect",
        "let me inspect",
        "i'll run",
        "i will run",
        "let me run",
        "call the tool",
        "use the tool",
        "invoke the tool",
    ]
    .iter()
    .any(|needle| lower.contains(needle));

    action_phrase
        || tools.iter().any(|tool| {
            let name = tool.function.name.to_ascii_lowercase();
            [
                format!("call {name}"),
                format!("use {name}"),
                format!("run {name}"),
                format!("invoke {name}"),
            ]
            .iter()
            .any(|needle| lower.contains(needle))
        })
}

fn looks_like_malformed_tool_output(content: &str, tools: &[OpenAiTool]) -> bool {
    let lower = content.to_ascii_lowercase();
    let structured_marker = [
        "<tool_call",
        "<tool ",
        "<function",
        "function=",
        "tool_calls",
        "[[ ## tool_calls ## ]]",
    ]
    .iter()
    .any(|needle| lower.contains(needle));

    structured_marker
        || looks_like_unparsed_tool_call_json(content, tools)
        || looks_like_untagged_dsrs_contract(content)
}

fn looks_like_unparsed_tool_call_json(content: &str, tools: &[OpenAiTool]) -> bool {
    let trimmed = content.trim_start();
    if !(trimmed.starts_with('[') || trimmed.starts_with('{')) {
        return false;
    }

    let lower = trimmed.to_ascii_lowercase();
    let has_tool_shape = lower.contains("\"name\"") && lower.contains("\"arguments\"");
    let references_known_tool = tools.iter().any(|tool| {
        let needle = format!("\"{}\"", tool.function.name.to_ascii_lowercase());
        lower.contains(&needle)
    });
    has_tool_shape && references_known_tool
}

fn looks_like_untagged_dsrs_contract(content: &str) -> bool {
    let lower = content.trim().to_ascii_lowercase();
    (lower.contains("\ntool_calls") || lower.contains("tool_calls:"))
        || ((lower.ends_with("completed") || lower.ends_with("[[ ## completed ## ]]"))
            && lower.lines().any(|line| line.trim() == "[]"))
}

fn strip_stray_text_label(content: &str) -> Option<String> {
    let trimmed = content.trim();
    let lower = trimmed.to_ascii_lowercase();
    let rest = if lower.starts_with("assistant text:") {
        trimmed.get("assistant text:".len()..)?
    } else if lower.starts_with("content:") {
        trimmed.get("content:".len()..)?
    } else if lower.starts_with("content\n") {
        trimmed.get("content\n".len()..)?
    } else if lower.starts_with("content\r\n") {
        trimmed.get("content\r\n".len()..)?
    } else if lower.starts_with("content \"") || lower.starts_with("content '") {
        trimmed.get("content ".len()..)?
    } else {
        return None;
    };

    let cleaned = rest.trim();
    let cleaned = unquote(cleaned).unwrap_or(cleaned).trim();
    (!cleaned.is_empty()).then(|| cleaned.to_string())
}

fn unquote(value: &str) -> Option<&str> {
    let first = value.chars().next()?;
    if first != '"' && first != '\'' {
        return None;
    }
    value
        .strip_prefix(first)
        .and_then(|value| value.strip_suffix(first))
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Map};

    use super::*;
    use crate::openai::{ChatChoice, ChatMessage, OpenAiFunctionTool, OpenAiTool};

    fn tool(name: &str) -> OpenAiTool {
        OpenAiTool {
            tool_type: "function".to_string(),
            function: OpenAiFunctionTool {
                name: name.to_string(),
                description: None,
                parameters: json!({"type":"object"}),
            },
        }
    }

    #[test]
    fn extracts_xml_tool_calls() {
        let mut events = Vec::new();
        let calls = extract_tool_intents_from_content(
            r#"<tool_call name="read_file">{"path":"Cargo.toml"}</tool_call>"#,
            &[tool("read_file")],
            &mut events,
        );

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments.as_ref().unwrap()["path"], "Cargo.toml");
    }

    #[test]
    fn extracts_nested_xml_tool_calls() {
        let mut events = Vec::new();
        let calls = extract_tool_intents_from_content(
            r#"<tool_call><name>read_file</name><arguments>{path:"Cargo.toml"}</arguments></tool_call>"#,
            &[tool("read_file")],
            &mut events,
        );

        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read_file");
        assert_eq!(calls[0].arguments.as_ref().unwrap()["path"], "Cargo.toml");
    }

    #[test]
    fn extracts_tagged_json_multiple_tool_calls() {
        let mut events = Vec::new();
        let calls = extract_tool_intents_from_content(
            r#"<tool_calls_json>{"tool_calls":[{"name":"read_file","arguments":{"path":"a"}},{"name":"read_file","arguments":{"path":"b"}}]}</tool_calls_json>"#,
            &[tool("read_file")],
            &mut events,
        );

        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].arguments.as_ref().unwrap()["path"], "b");
    }

    #[test]
    fn extracts_dsrs_contract_tool_call() {
        let response = ChatCompletionResponse {
            id: "1".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "m".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage::new(
                    "assistant",
                    r#"[[ ## content ## ]]

[[ ## tool_calls ## ]]
[{"name":"read_file","arguments":{"path":"Cargo.toml"}}]
[[ ## completed ## ]]"#,
                ),
                finish_reason: Some("stop".to_string()),
                logprobs: None,
                extra: Map::new(),
            }],
            usage: None,
            extra: Map::new(),
        };

        let interpreted = interpret_response(&response, &[tool("read_file")]);

        assert_eq!(interpreted.content, None);
        assert_eq!(interpreted.tool_intents.len(), 1);
        assert_eq!(interpreted.tool_intents[0].source, ToolIntentSource::Dsrs);
        assert_eq!(
            interpreted.tool_intents[0].arguments.as_ref().unwrap()["path"],
            "Cargo.toml"
        );
    }

    #[test]
    fn preserves_dsrs_content_with_tool_calls() {
        let response = ChatCompletionResponse {
            id: "1".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "m".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage::new(
                    "assistant",
                    r#"[[ ## content ## ]]
I will inspect the repository first.

[[ ## tool_calls ## ]]
[{"name":"bash","arguments":{"command":"ls packages"}}]
[[ ## completed ## ]]"#,
                ),
                finish_reason: Some("stop".to_string()),
                logprobs: None,
                extra: Map::new(),
            }],
            usage: None,
            extra: Map::new(),
        };

        let interpreted = interpret_response(&response, &[tool("bash")]);

        assert_eq!(
            interpreted.content.as_deref(),
            Some("I will inspect the repository first.")
        );
        assert_eq!(interpreted.tool_intents.len(), 1);
        assert!(!interpreted.has_failure(ResponseFailureKind::DsrsContractViolation));
    }

    #[test]
    fn strips_dsrs_content_when_no_tool_call_is_needed() {
        let response = ChatCompletionResponse {
            id: "1".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "m".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage::new(
                    "assistant",
                    "[[ ## content ## ]]\nDone.\n[[ ## tool_calls ## ]]\n[]\n[[ ## completed ## ]]",
                ),
                finish_reason: Some("stop".to_string()),
                logprobs: None,
                extra: Map::new(),
            }],
            usage: None,
            extra: Map::new(),
        };

        let interpreted = interpret_response(&response, &[tool("read_file")]);

        assert_eq!(interpreted.content.as_deref(), Some("Done."));
        assert!(interpreted.tool_intents.is_empty());
    }

    #[test]
    fn marks_empty_dsrs_content_and_tool_calls_as_failure() {
        let response = ChatCompletionResponse {
            id: "1".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "google/gemma-4-26b-a4b-it".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage::new(
                    "assistant",
                    "[[ ## content ## ]]\n[[ ## tool_calls ## ]]\n[]\n[[ ## completed ## ]]",
                ),
                finish_reason: Some("stop".to_string()),
                logprobs: None,
                extra: Map::new(),
            }],
            usage: None,
            extra: Map::new(),
        };

        let interpreted = interpret_response(&response, &[tool("read")]);

        assert_eq!(interpreted.content.as_deref(), Some(""));
        assert!(interpreted.tool_intents.is_empty());
        assert!(interpreted.suspicious_stop);
        assert!(interpreted.has_failure(ResponseFailureKind::EmptyDsrsOutput));
        assert!(interpreted
            .parse_events
            .iter()
            .any(|event| event.contains("empty or no-op content and no tool calls")));
    }

    #[test]
    fn marks_noop_dsrs_content_and_empty_tool_calls_as_failure() {
        let response = ChatCompletionResponse {
            id: "1".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "google/gemma-4-26b-a4b-it".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage::new(
                    "assistant",
                    "[[ ## content ## ]]\n[]\n[[ ## tool_calls ## ]]\n[]\n[[ ## completed ## ]]",
                ),
                finish_reason: Some("stop".to_string()),
                logprobs: None,
                extra: Map::new(),
            }],
            usage: None,
            extra: Map::new(),
        };

        let interpreted = interpret_response(&response, &[tool("read")]);

        assert_eq!(interpreted.content.as_deref(), Some("[]"));
        assert!(interpreted.tool_intents.is_empty());
        assert!(interpreted.suspicious_stop);
        assert!(interpreted.has_failure(ResponseFailureKind::EmptyDsrsOutput));
    }

    #[test]
    fn marks_label_free_dsrs_empty_tool_tail_suspicious() {
        let response = ChatCompletionResponse {
            id: "1".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "m".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage::new("assistant", "Hello! How can I help?\n\n[]\n\ncompleted"),
                finish_reason: Some("stop".to_string()),
                logprobs: None,
                extra: Map::new(),
            }],
            usage: None,
            extra: Map::new(),
        };

        let interpreted = interpret_response(&response, &[tool("read_file")]);

        assert_eq!(
            interpreted.content.as_deref(),
            Some("Hello! How can I help?\n\n[]\n\ncompleted")
        );
        assert!(interpreted.tool_intents.is_empty());
        assert!(interpreted.suspicious_stop);
        assert!(interpreted.has_failure(ResponseFailureKind::UntaggedDsrsLikeOutput));
        assert!(interpreted.has_failure(ResponseFailureKind::MalformedKnownToolCall));
    }

    #[test]
    fn marks_empty_label_free_dsrs_suspicious_without_parsing_as_dsrs() {
        let response = ChatCompletionResponse {
            id: "1".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "m".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage::new("assistant", "content\n\ntool_calls\n\ncompleted"),
                finish_reason: Some("stop".to_string()),
                logprobs: None,
                extra: Map::new(),
            }],
            usage: None,
            extra: Map::new(),
        };

        let interpreted = interpret_response(&response, &[tool("read_file")]);

        assert_eq!(
            interpreted.content.as_deref(),
            Some("tool_calls\n\ncompleted")
        );
        assert!(interpreted.tool_intents.is_empty());
        assert!(interpreted.suspicious_stop);
        assert!(interpreted.has_failure(ResponseFailureKind::UntaggedDsrsLikeOutput));
    }

    #[test]
    fn strips_stray_content_label_without_completed_marker() {
        let response = ChatCompletionResponse {
            id: "1".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "m".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage::new("assistant", "content \"Hello!\""),
                finish_reason: Some("stop".to_string()),
                logprobs: None,
                extra: Map::new(),
            }],
            usage: None,
            extra: Map::new(),
        };

        let interpreted = interpret_response(&response, &[tool("read_file")]);

        assert_eq!(interpreted.content.as_deref(), Some("Hello!"));
        assert!(interpreted.tool_intents.is_empty());
    }

    #[test]
    fn malformed_structured_tool_output_is_suspicious() {
        let response = ChatCompletionResponse {
            id: "1".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "m".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: ChatMessage::new(
                    "assistant",
                    "<tool_call><function=bash><parameter=\"command\">ls</parameter></tool_call>",
                ),
                finish_reason: Some("stop".to_string()),
                logprobs: None,
                extra: Map::new(),
            }],
            usage: None,
            extra: Map::new(),
        };

        let interpreted = interpret_response(&response, &[tool("bash")]);

        assert!(interpreted.tool_intents.is_empty());
        assert!(interpreted.suspicious_stop);
        assert!(interpreted.has_failure(ResponseFailureKind::MalformedKnownToolCall));
    }

    #[test]
    fn maps_reasoning_text_into_content() {
        let mut msg = ChatMessage {
            role: "assistant".to_string(),
            ..ChatMessage::default()
        };
        msg.extra
            .insert("reasoning".to_string(), json!("visible text"));
        let response = ChatCompletionResponse {
            id: "1".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "m".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: msg,
                finish_reason: Some("stop".to_string()),
                logprobs: None,
                extra: Map::new(),
            }],
            usage: None,
            extra: Map::new(),
        };

        let interpreted = interpret_response(&response, &[]);
        assert_eq!(interpreted.content.as_deref(), Some("visible text"));
    }

    #[test]
    fn does_not_map_reasoning_into_content_when_tools_are_available() {
        let mut msg = ChatMessage {
            role: "assistant".to_string(),
            ..ChatMessage::default()
        };
        msg.extra
            .insert("reasoning".to_string(), json!("Thinking Process:\nsecret"));
        let response = ChatCompletionResponse {
            id: "1".to_string(),
            object: "chat.completion".to_string(),
            created: 0,
            model: "m".to_string(),
            choices: vec![ChatChoice {
                index: 0,
                message: msg,
                finish_reason: Some("length".to_string()),
                logprobs: None,
                extra: Map::new(),
            }],
            usage: None,
            extra: Map::new(),
        };

        let interpreted = interpret_response(&response, &[tool("bash")]);

        assert_eq!(interpreted.content, None);
        assert_eq!(
            interpreted.reasoning.as_deref(),
            Some("Thinking Process:\nsecret")
        );
        assert!(interpreted.tool_intents.is_empty());
        assert!(interpreted.suspicious_stop);
        assert!(interpreted.has_failure(ResponseFailureKind::EmptyAssistantOutput));
        assert!(
            interpreted
                .parse_events
                .iter()
                .any(|event| event
                    .contains("reasoning/thinking but no visible content or tool calls"))
        );
    }
}
