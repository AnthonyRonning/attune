use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::openai::{ChatCompletionResponse, OpenAiTool};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpretedResponse {
    pub content: Option<String>,
    pub reasoning: Option<String>,
    pub finish_reason: Option<String>,
    pub tool_intents: Vec<ToolIntent>,
    pub parse_events: Vec<String>,
    pub suspicious_stop: bool,
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
            suspicious_stop: true,
        };
    };

    let message = &choice.message;
    let mut parse_events = Vec::new();
    let mut content = message.content_text();
    let reasoning = message.reasoning_text();
    if content.is_none() && reasoning.is_some() {
        content = reasoning.clone();
        parse_events
            .push("mapped reasoning/thinking text into content for compatibility".to_string());
    }

    let mut tool_intents = Vec::new();
    if let Some(native_calls) = &message.tool_calls {
        for call in native_calls {
            let parsed = parse_jsonish(&call.function.arguments);
            if parsed.is_none() {
                parse_events.push(format!(
                    "native tool call {} had malformed JSON arguments",
                    call.function.name
                ));
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

    if let Some(text) = content.as_deref() {
        tool_intents.extend(extract_tool_intents_from_content(
            text,
            tools,
            &mut parse_events,
        ));
    }

    let suspicious_stop = tool_intents.is_empty()
        && tools_available(tools)
        && choice.finish_reason.as_deref().unwrap_or("stop") == "stop"
        && content
            .as_deref()
            .map(looks_like_premature_tool_narration)
            .unwrap_or(false);

    if suspicious_stop {
        parse_events.push(
            "assistant appears to narrate an imminent tool action but emitted no tool call"
                .to_string(),
        );
    }

    InterpretedResponse {
        content,
        reasoning,
        finish_reason: choice.finish_reason.clone(),
        tool_intents,
        parse_events,
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

fn tools_available(tools: &[OpenAiTool]) -> bool {
    !tools.is_empty()
}

fn looks_like_premature_tool_narration(content: &str) -> bool {
    let lower = content.to_ascii_lowercase();
    [
        "let me",
        "i'll",
        "i will",
        "i’m going to",
        "i am going to",
        "read the",
        "call the",
        "use the",
        "check the",
    ]
    .iter()
    .any(|needle| lower.contains(needle))
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
}
