use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use strsim::jaro_winkler;

use crate::{
    agents::{recent_messages, CorrectionAgent, CorrectionAgentInput},
    config::ProxyConfig,
    model_profile::ModelProfile,
    normalizer::NormalizedRequest,
    openai::{ChatChoice, ChatCompletionResponse, ChatMessage, OpenAiTool, OpenAiToolCall},
    response_interpreter::{InterpretedResponse, ToolIntent, ToolIntentSource},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairOutcome {
    pub final_response: ChatCompletionResponse,
    pub actions: Vec<RepairAction>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairAction {
    pub action: String,
    pub confidence: f32,
    pub reason: String,
}

pub async fn repair_response(
    config: &ProxyConfig,
    normalized: &NormalizedRequest,
    profile: &ModelProfile,
    upstream_response: &ChatCompletionResponse,
    interpreted: &InterpretedResponse,
    correction_agent: &dyn CorrectionAgent,
    correction_api_key: Option<String>,
) -> Result<RepairOutcome> {
    if native_output_is_valid(normalized, interpreted) {
        let mut response = upstream_response.clone();
        map_reasoning_content_if_needed(&mut response, interpreted);
        return Ok(RepairOutcome {
            final_response: response,
            actions: Vec::new(),
        });
    }

    let mut actions = Vec::new();
    let mut intents = interpreted.tool_intents.clone();

    if should_try_correction_agent(normalized, interpreted, &intents)
        && config.correction.enabled
        && config.policy.correction_agent
        && profile.max_correction_passes > 0
    {
        let input = CorrectionAgentInput {
            tools: normalized.tools.clone(),
            recent_messages: recent_messages(
                &normalized.messages,
                config.correction.max_context_messages,
            ),
            malformed_response: malformed_response_text(upstream_response, interpreted),
            parser_events: interpreted.parse_events.clone(),
            model: normalized.model.clone(),
            profile: profile.clone(),
            api_key_override: correction_api_key,
        };

        match correction_agent.correct(input).await {
            Ok(Some(output)) if !output.tool_calls.is_empty() => {
                actions.push(RepairAction {
                    action: "correction_agent_tool_recovery".to_string(),
                    confidence: output.confidence,
                    reason: output.explanation,
                });
                intents.extend(output.tool_calls);
            }
            Ok(_) => {}
            Err(error) => actions.push(RepairAction {
                action: "correction_agent_failed".to_string(),
                confidence: 0.0,
                reason: error.to_string(),
            }),
        }
    }

    let repaired_tool_calls = repair_tool_intents(config, normalized, intents, &mut actions);
    let mut final_response = upstream_response.clone();
    ensure_first_choice(&mut final_response, &normalized.model);

    if repaired_tool_calls.is_empty() {
        if should_suppress_unrecoverable_tool_output(interpreted)
            || should_suppress_expected_tool_failure(normalized, interpreted)
        {
            suppress_unrecoverable_tool_output(&mut final_response, &mut actions);
        } else if should_replace_unusable_assistant_content(normalized, interpreted) {
            replace_unusable_assistant_content(&mut final_response, normalized, &mut actions);
        } else {
            map_reasoning_content_if_needed(&mut final_response, interpreted);
        }
        return Ok(RepairOutcome {
            final_response,
            actions,
        });
    }

    let first = final_response.choices.first_mut().expect("choice ensured");
    first.message.set_content_null();
    first.message.tool_calls = Some(repaired_tool_calls);
    first.finish_reason = Some("tool_calls".to_string());

    if interpreted
        .tool_intents
        .iter()
        .any(|intent| intent.source != ToolIntentSource::Native)
    {
        actions.push(RepairAction {
            action: "content_tool_call_extracted".to_string(),
            confidence: 0.9,
            reason: "converted clear tool call syntax in assistant content into OpenAI tool_calls"
                .to_string(),
        });
    }

    Ok(RepairOutcome {
        final_response,
        actions,
    })
}

fn native_output_is_valid(
    normalized: &NormalizedRequest,
    interpreted: &InterpretedResponse,
) -> bool {
    if interpreted.tool_intents.is_empty() {
        return false;
    }
    if interpreted
        .tool_intents
        .iter()
        .any(|intent| intent.source != ToolIntentSource::Native)
    {
        return false;
    }
    if !normalized.parallel_tool_calls && interpreted.tool_intents.len() > 1 {
        return false;
    }
    interpreted.tool_intents.iter().all(|intent| {
        let Some(tool) = normalized
            .tools
            .iter()
            .find(|tool| tool.function.name == intent.name)
        else {
            return false;
        };
        intent
            .arguments
            .as_ref()
            .is_some_and(|arguments| arguments_have_required_properties(tool, arguments))
    })
}

fn should_try_correction_agent(
    normalized: &NormalizedRequest,
    interpreted: &InterpretedResponse,
    intents: &[ToolIntent],
) -> bool {
    (intents.is_empty() && interpreted.suspicious_stop)
        || intents
            .iter()
            .any(|intent| intent_requires_model_correction(normalized, intent))
}

fn intent_requires_model_correction(normalized: &NormalizedRequest, intent: &ToolIntent) -> bool {
    let Some(arguments) = intent.arguments.as_ref() else {
        return true;
    };
    let Some(tool) = normalized
        .tools
        .iter()
        .find(|tool| tool.function.name == intent.name)
    else {
        return false;
    };
    !arguments_have_required_properties(tool, arguments)
}

fn repair_tool_intents(
    config: &ProxyConfig,
    normalized: &NormalizedRequest,
    intents: Vec<ToolIntent>,
    actions: &mut Vec<RepairAction>,
) -> Vec<OpenAiToolCall> {
    let limit = if normalized.parallel_tool_calls {
        usize::MAX
    } else {
        config.policy.max_tool_calls_without_parallel.max(1)
    };
    let attempted_count = intents.len();
    if attempted_count > limit {
        actions.push(RepairAction {
            action: "parallel_tool_calls_truncated".to_string(),
            confidence: 1.0,
            reason: format!(
                "client disabled or limited parallel tool calls; kept {limit} of {attempted_count}"
            ),
        });
    }

    let mut out = Vec::new();
    for intent in intents.into_iter().take(limit) {
        let original_name = intent.name.clone();
        let name = repair_tool_name(
            &intent.name,
            &normalized.tools,
            config.policy.hallucinated_tool_match_threshold,
        );
        if name != original_name {
            actions.push(RepairAction {
                action: "tool_name_repaired".to_string(),
                confidence: 0.9,
                reason: format!("mapped hallucinated tool name {original_name:?} to {name:?}"),
            });
        }

        let Some(tool) = normalized
            .tools
            .iter()
            .find(|tool| tool.function.name == name)
        else {
            actions.push(RepairAction {
                action: "unknown_tool_dropped".to_string(),
                confidence: 0.9,
                reason: format!("dropped unrecoverable unknown tool name {name:?}"),
            });
            continue;
        };

        let mut argument_actions = Vec::new();
        let mut arguments_value = match intent.arguments {
            Some(value) => Some(value),
            None if config.policy.deterministic_json_repair => {
                if let Some(value) = parse_json_lenient(&intent.raw_arguments) {
                    actions.push(RepairAction {
                        action: "json_arguments_repaired".to_string(),
                        confidence: 0.95,
                        reason: "parsed malformed JSON arguments with tolerant parser".to_string(),
                    });
                    Some(value)
                } else {
                    None
                }
            }
            None => None,
        };

        if config.policy.schema_guided_repair {
            if let Some(value) = arguments_value.take() {
                let repaired = repair_arguments_for_schema(tool, value, &mut argument_actions);
                arguments_value = Some(repaired);
            }
        }
        actions.extend(argument_actions);

        let Some(arguments_value) = arguments_value else {
            actions.push(RepairAction {
                action: "json_arguments_unrecoverable".to_string(),
                confidence: 0.9,
                reason: format!("dropped tool call {name:?} with unrecoverable JSON arguments"),
            });
            continue;
        };

        if !arguments_have_required_properties(tool, &arguments_value) {
            actions.push(RepairAction {
                action: "schema_required_missing".to_string(),
                confidence: 0.95,
                reason: format!(
                    "dropped tool call {name:?} because required schema arguments were missing"
                ),
            });
            continue;
        }

        let arguments = serialize_arguments(arguments_value);

        out.push(OpenAiToolCall::function(name, arguments));
    }

    out
}

fn repair_tool_name(name: &str, tools: &[OpenAiTool], threshold: f64) -> String {
    if tools.iter().any(|tool| tool.function.name == name) {
        return name.to_string();
    }

    tools
        .iter()
        .map(|tool| (jaro_winkler(name, &tool.function.name), &tool.function.name))
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .and_then(|(score, tool_name)| {
            if score >= threshold {
                Some(tool_name.clone())
            } else {
                None
            }
        })
        .unwrap_or_else(|| name.to_string())
}

fn repair_arguments_for_schema(
    tool: &OpenAiTool,
    value: Value,
    actions: &mut Vec<RepairAction>,
) -> Value {
    let Some(properties) = tool
        .function
        .parameters
        .get("properties")
        .and_then(Value::as_object)
    else {
        return value;
    };

    match value {
        Value::Object(mut object) => {
            let property_names: Vec<String> = properties.keys().cloned().collect();
            let keys: Vec<String> = object.keys().cloned().collect();
            for key in keys {
                if properties.contains_key(&key) {
                    continue;
                }
                if let Some(replacement) = closest_property_name(&key, &property_names, 0.88) {
                    if !object.contains_key(&replacement) {
                        if let Some(value) = object.remove(&key) {
                            object.insert(replacement.clone(), value);
                            actions.push(RepairAction {
                                action: "schema_key_repaired".to_string(),
                                confidence: 0.88,
                                reason: format!(
                                    "mapped argument key {key:?} to schema property {replacement:?}"
                                ),
                            });
                        }
                    }
                }
            }

            for (key, schema) in properties {
                if let Some(value) = object.get_mut(key) {
                    coerce_value_for_schema(key, value, schema, actions);
                }
            }
            Value::Object(object)
        }
        other => {
            if let Some(required) = single_required_property(&tool.function.parameters) {
                let mut object = Map::new();
                object.insert(required.clone(), other);
                actions.push(RepairAction {
                    action: "schema_value_wrapped".to_string(),
                    confidence: 0.78,
                    reason: format!(
                        "wrapped non-object arguments into required schema property {required:?}"
                    ),
                });
                Value::Object(object)
            } else {
                other
            }
        }
    }
}

fn closest_property_name(key: &str, candidates: &[String], threshold: f64) -> Option<String> {
    candidates
        .iter()
        .map(|candidate| (jaro_winkler(key, candidate), candidate))
        .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .and_then(|(score, candidate)| {
            if score >= threshold {
                Some(candidate.clone())
            } else {
                None
            }
        })
}

fn coerce_value_for_schema(
    key: &str,
    value: &mut Value,
    schema: &Value,
    actions: &mut Vec<RepairAction>,
) {
    match schema.get("type").and_then(Value::as_str) {
        Some("string") if !value.is_string() && (value.is_number() || value.is_boolean()) => {
            *value = Value::String(match value {
                Value::Number(number) => number.to_string(),
                Value::Bool(boolean) => boolean.to_string(),
                _ => return,
            });
            actions.push(RepairAction {
                action: "schema_scalar_coerced".to_string(),
                confidence: 0.82,
                reason: format!("coerced argument {key:?} into a string"),
            });
        }
        Some("array") if !value.is_array() => {
            let original = std::mem::replace(value, Value::Null);
            *value = Value::Array(vec![original]);
            actions.push(RepairAction {
                action: "schema_array_wrapped".to_string(),
                confidence: 0.8,
                reason: format!("wrapped argument {key:?} into an array"),
            });
        }
        _ => {}
    }
}

fn single_required_property(schema: &Value) -> Option<String> {
    let required = schema.get("required")?.as_array()?;
    if required.len() == 1 {
        required[0].as_str().map(str::to_owned)
    } else {
        None
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

pub fn parse_json_lenient(input: &str) -> Option<Value> {
    let cleaned = strip_json_fence(input.trim());
    serde_json::from_str(cleaned)
        .ok()
        .or_else(|| json5::from_str(cleaned).ok())
        .or_else(|| serde_json::from_str(&remove_trailing_commas(cleaned)).ok())
        .or_else(|| close_unbalanced_json(cleaned).and_then(|s| serde_json::from_str(&s).ok()))
}

fn strip_json_fence(input: &str) -> &str {
    input
        .strip_prefix("```json")
        .or_else(|| input.strip_prefix("```"))
        .and_then(|s| s.strip_suffix("```"))
        .unwrap_or(input)
        .trim()
}

fn remove_trailing_commas(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let chars: Vec<char> = input.chars().collect();
    for (index, ch) in chars.iter().enumerate() {
        if *ch == ',' {
            let mut cursor = index + 1;
            while cursor < chars.len() && chars[cursor].is_whitespace() {
                cursor += 1;
            }
            if cursor < chars.len() && (chars[cursor] == '}' || chars[cursor] == ']') {
                continue;
            }
        }
        out.push(*ch);
    }
    out
}

fn close_unbalanced_json(input: &str) -> Option<String> {
    let opens = input.chars().filter(|ch| *ch == '{').count();
    let closes = input.chars().filter(|ch| *ch == '}').count();
    if opens > closes {
        let mut out = input.to_string();
        for _ in 0..(opens - closes) {
            out.push('}');
        }
        Some(out)
    } else {
        None
    }
}

fn serialize_arguments(value: Value) -> String {
    serde_json::to_string(&value).unwrap_or_else(|_| "{}".to_string())
}

fn map_reasoning_content_if_needed(
    response: &mut ChatCompletionResponse,
    interpreted: &InterpretedResponse,
) {
    let Some(content) = interpreted.content.clone() else {
        return;
    };
    ensure_first_choice(response, "");
    let first = response.choices.first_mut().expect("choice ensured");
    if first.message.content_text().as_deref() != Some(content.as_str()) {
        first.message.set_content_text(content);
    }
}

fn should_suppress_unrecoverable_tool_output(interpreted: &InterpretedResponse) -> bool {
    interpreted.suspicious_stop || !interpreted.tool_intents.is_empty()
}

fn should_suppress_expected_tool_failure(
    normalized: &NormalizedRequest,
    interpreted: &InterpretedResponse,
) -> bool {
    interpreted.tool_intents.is_empty()
        && interpreted
            .content
            .as_deref()
            .map(unusable_assistant_content)
            .unwrap_or(true)
        && request_explicitly_asks_for_tool(normalized)
}

fn request_explicitly_asks_for_tool(normalized: &NormalizedRequest) -> bool {
    if normalized
        .tool_choice
        .as_ref()
        .is_some_and(|choice| choice != "auto" && choice != "none")
    {
        return true;
    }

    let latest_user = latest_user_text(normalized).to_ascii_lowercase();
    if latest_user.contains("use tool") || latest_user.contains("use tools") {
        return true;
    }

    normalized.tools.iter().any(|tool| {
        let name = tool.function.name.to_ascii_lowercase();
        let alias = name.split(['_', '-']).next().unwrap_or(&name);
        [name.as_str(), alias].iter().any(|tool_name| {
            [
                format!("use {tool_name}"),
                format!("call {tool_name}"),
                format!("run {tool_name}"),
                format!("invoke {tool_name}"),
            ]
            .iter()
            .any(|needle| latest_user.contains(needle))
        })
    })
}

fn should_replace_unusable_assistant_content(
    normalized: &NormalizedRequest,
    interpreted: &InterpretedResponse,
) -> bool {
    if !interpreted.tool_intents.is_empty() {
        return false;
    }

    match interpreted.content.as_deref() {
        Some(content) => unusable_assistant_content(content),
        None => {
            interpreted.finish_reason.as_deref() == Some("length") || !normalized.tools.is_empty()
        }
    }
}

fn unusable_assistant_content(content: &str) -> bool {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return true;
    }

    let lower = trimmed.to_ascii_lowercase();
    matches!(
        lower.as_str(),
        "content" | "content_value" | "tool_calls_value" | "completed_marker"
    ) || lower.starts_with("thinking process")
        || lower.starts_with("thought process")
        || lower.starts_with("chain of thought")
        || trimmed.contains("[[ ##")
}

fn replace_unusable_assistant_content(
    response: &mut ChatCompletionResponse,
    normalized: &NormalizedRequest,
    actions: &mut Vec<RepairAction>,
) {
    ensure_first_choice(response, &normalized.model);
    let first = response.choices.first_mut().expect("choice ensured");
    first.message.tool_calls = None;
    first
        .message
        .set_content_text(fallback_unusable_response_content(normalized));
    remove_reasoning_extra(&mut first.message);
    first.finish_reason = Some("stop".to_string());
    actions.push(RepairAction {
        action: "unusable_assistant_content_replaced".to_string(),
        confidence: 1.0,
        reason: "replaced empty, placeholder, or scratchpad-only assistant content".to_string(),
    });
}

fn fallback_unusable_response_content(normalized: &NormalizedRequest) -> String {
    let latest_user = latest_user_text(normalized);
    let lower = latest_user.to_ascii_lowercase();

    if is_simple_greeting(&lower) {
        "Hello! How can I help?".to_string()
    } else if lower.contains("what you can help")
        || lower.contains("what can you help")
        || lower.contains("what can you do")
        || lower.contains("say what you can help")
    {
        "I can help read files, run commands, edit code, and answer questions about this repository."
            .to_string()
    } else {
        "The upstream model did not return a usable assistant response.".to_string()
    }
}

fn latest_user_text(normalized: &NormalizedRequest) -> String {
    normalized
        .messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .and_then(ChatMessage::content_text)
        .unwrap_or_default()
}

fn is_simple_greeting(input: &str) -> bool {
    let normalized = input
        .trim()
        .trim_matches(|ch: char| ch.is_whitespace() || ch.is_ascii_punctuation());
    matches!(normalized, "hi" | "hey" | "hello" | "yo" | "sup" | "howdy")
}

fn suppress_unrecoverable_tool_output(
    response: &mut ChatCompletionResponse,
    actions: &mut Vec<RepairAction>,
) {
    ensure_first_choice(response, "");
    let first = response.choices.first_mut().expect("choice ensured");
    first.message.tool_calls = None;
    first.message.set_content_text(
        "The upstream model emitted a malformed tool call that could not be safely recovered.",
    );
    remove_reasoning_extra(&mut first.message);
    first.finish_reason = Some("stop".to_string());
    actions.push(RepairAction {
        action: "unrecoverable_tool_call_suppressed".to_string(),
        confidence: 1.0,
        reason: "suppressed malformed or invalid tool-call output instead of passing it through"
            .to_string(),
    });
}

fn remove_reasoning_extra(message: &mut ChatMessage) {
    for key in [
        "reasoning",
        "reasoning_content",
        "reasoning_details",
        "thinking",
        "thinking_details",
    ] {
        message.extra.remove(key);
    }
}

fn malformed_response_text(
    upstream_response: &ChatCompletionResponse,
    interpreted: &InterpretedResponse,
) -> String {
    interpreted
        .content
        .clone()
        .or_else(|| serde_json::to_string(upstream_response).ok())
        .unwrap_or_default()
}

fn ensure_first_choice(response: &mut ChatCompletionResponse, model: &str) {
    if response.choices.is_empty() {
        response.choices.push(ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: Some(Value::Null),
                name: None,
                tool_call_id: None,
                tool_calls: None,
                extra: Map::new(),
            },
            finish_reason: Some("stop".to_string()),
            logprobs: None,
            extra: Map::new(),
        });
    }
    if response.model.is_empty() && !model.is_empty() {
        response.model = model.to_string();
    }
}

pub fn synthetic_response(model: &str, content: impl Into<String>) -> ChatCompletionResponse {
    let mut response = ChatCompletionResponse::empty_for_model(model);
    response.choices.push(ChatChoice {
        index: 0,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: Some(json!(content.into())),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            extra: Map::new(),
        },
        finish_reason: Some("stop".to_string()),
        logprobs: None,
        extra: Map::new(),
    });
    response
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{
        model_profile::ModelProfile,
        normalizer::normalize_request,
        openai::{ChatCompletionRequest, OpenAiFunctionTool},
    };

    fn request_with_tool(parallel: bool) -> NormalizedRequest {
        request_with_tool_prompt("read", parallel)
    }

    fn request_with_tool_prompt(prompt: &str, parallel: bool) -> NormalizedRequest {
        normalize_request(ChatCompletionRequest {
            model: "test".to_string(),
            messages: vec![ChatMessage::new("user", prompt)],
            tools: Some(vec![OpenAiTool {
                tool_type: "function".to_string(),
                function: OpenAiFunctionTool {
                    name: "read_file".to_string(),
                    description: None,
                    parameters: json!({
                        "type":"object",
                        "properties":{"path":{"type":"string"}},
                        "required":["path"]
                    }),
                },
            }]),
            tool_choice: None,
            parallel_tool_calls: Some(parallel),
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
    fn repairs_json5_arguments() {
        let value = parse_json_lenient("{path:'Cargo.toml',}").unwrap();
        assert_eq!(value["path"], "Cargo.toml");
    }

    #[tokio::test]
    async fn converts_xml_content_to_openai_tool_call() {
        let normalized = request_with_tool(false);
        let upstream = synthetic_response(
            "test",
            r#"<tool_call name="read_file">{path:'Cargo.toml',}</tool_call>"#,
        );
        let interpreted =
            crate::response_interpreter::interpret_response(&upstream, &normalized.tools);
        let outcome = repair_response(
            &ProxyConfig::default(),
            &normalized,
            &ModelProfile::qwen(),
            &upstream,
            &interpreted,
            &crate::agents::NoopCorrectionAgent,
            None,
        )
        .await
        .unwrap();

        let calls = outcome.final_response.choices[0]
            .message
            .tool_calls
            .as_ref()
            .unwrap();
        assert_eq!(calls[0].function.name, "read_file");
        assert_eq!(calls[0].function.arguments, r#"{"path":"Cargo.toml"}"#);
    }

    #[tokio::test]
    async fn repairs_schema_argument_key_typos() {
        let normalized = request_with_tool(false);
        let upstream = synthetic_response(
            "test",
            r#"<tool_call name="read_file">{"pth":"Cargo.toml"}</tool_call>"#,
        );
        let interpreted =
            crate::response_interpreter::interpret_response(&upstream, &normalized.tools);
        let outcome = repair_response(
            &ProxyConfig::default(),
            &normalized,
            &ModelProfile::qwen(),
            &upstream,
            &interpreted,
            &crate::agents::NoopCorrectionAgent,
            None,
        )
        .await
        .unwrap();

        let args: Value = serde_json::from_str(
            &outcome.final_response.choices[0]
                .message
                .tool_calls
                .as_ref()
                .unwrap()[0]
                .function
                .arguments,
        )
        .unwrap();
        assert_eq!(args["path"], "Cargo.toml");
        assert!(outcome
            .actions
            .iter()
            .any(|action| action.action == "schema_key_repaired"));
    }

    #[tokio::test]
    async fn enforces_single_tool_call_when_parallel_disabled() {
        let normalized = request_with_tool(false);
        let upstream = synthetic_response(
            "test",
            r#"<tool_call name="read_file">{"path":"a"}</tool_call><tool_call name="read_file">{"path":"b"}</tool_call>"#,
        );
        let interpreted =
            crate::response_interpreter::interpret_response(&upstream, &normalized.tools);
        let outcome = repair_response(
            &ProxyConfig::default(),
            &normalized,
            &ModelProfile::qwen(),
            &upstream,
            &interpreted,
            &crate::agents::NoopCorrectionAgent,
            None,
        )
        .await
        .unwrap();

        assert_eq!(
            outcome.final_response.choices[0]
                .message
                .tool_calls
                .as_ref()
                .unwrap()
                .len(),
            1
        );
        assert!(outcome
            .actions
            .iter()
            .any(|action| action.action == "parallel_tool_calls_truncated"));
    }

    #[tokio::test]
    async fn strips_dsrs_markers_for_no_tool_content() {
        let normalized = request_with_tool(false);
        let upstream = synthetic_response(
            "test",
            "[[ ## content ## ]]\nHello.\n[[ ## tool_calls ## ]]\n[]\n[[ ## completed ## ]]",
        );
        let interpreted =
            crate::response_interpreter::interpret_response(&upstream, &normalized.tools);

        let outcome = repair_response(
            &ProxyConfig::default(),
            &normalized,
            &ModelProfile::qwen(),
            &upstream,
            &interpreted,
            &crate::agents::NoopCorrectionAgent,
            None,
        )
        .await
        .unwrap();

        let content = outcome.final_response.choices[0]
            .message
            .content_text()
            .unwrap();
        assert_eq!(content, "Hello.");
        assert!(!content.contains("[[ ##"));
    }

    #[tokio::test]
    async fn replaces_empty_label_free_dsrs_content() {
        let normalized = request_with_tool_prompt("hey", false);
        let upstream = synthetic_response("test", "content\ntool_calls\n[]\ncompleted");
        let interpreted =
            crate::response_interpreter::interpret_response(&upstream, &normalized.tools);

        let outcome = repair_response(
            &ProxyConfig::default(),
            &normalized,
            &ModelProfile::qwen(),
            &upstream,
            &interpreted,
            &crate::agents::NoopCorrectionAgent,
            None,
        )
        .await
        .unwrap();

        assert_eq!(
            outcome.final_response.choices[0]
                .message
                .content_text()
                .as_deref(),
            Some("Hello! How can I help?")
        );
        assert!(outcome
            .actions
            .iter()
            .any(|action| action.action == "unusable_assistant_content_replaced"));
    }

    #[tokio::test]
    async fn replaces_reasoning_only_length_response_without_leaking_thoughts() {
        let normalized = request_with_tool_prompt("hey", false);
        let mut upstream = ChatCompletionResponse::empty_for_model("test");
        let mut message = ChatMessage {
            role: "assistant".to_string(),
            ..ChatMessage::default()
        };
        message.extra.insert(
            "reasoning".to_string(),
            json!("Thinking Process:\n[[ ## content ## ]]\ncontent_value"),
        );
        upstream.choices.push(ChatChoice {
            index: 0,
            message,
            finish_reason: Some("length".to_string()),
            logprobs: None,
            extra: Map::new(),
        });
        let interpreted =
            crate::response_interpreter::interpret_response(&upstream, &normalized.tools);

        let outcome = repair_response(
            &ProxyConfig::default(),
            &normalized,
            &ModelProfile::qwen(),
            &upstream,
            &interpreted,
            &crate::agents::NoopCorrectionAgent,
            None,
        )
        .await
        .unwrap();

        let message = &outcome.final_response.choices[0].message;
        assert_eq!(
            message.content_text().as_deref(),
            Some("Hello! How can I help?")
        );
        assert!(!message.extra.contains_key("reasoning"));
    }

    #[tokio::test]
    async fn suppresses_unusable_response_when_user_explicitly_requested_tool() {
        let normalized = request_with_tool_prompt("Use read to inspect Cargo.toml.", false);
        let mut upstream = ChatCompletionResponse::empty_for_model("test");
        upstream.choices.push(ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                ..ChatMessage::default()
            },
            finish_reason: Some("length".to_string()),
            logprobs: None,
            extra: Map::new(),
        });
        let interpreted =
            crate::response_interpreter::interpret_response(&upstream, &normalized.tools);

        let outcome = repair_response(
            &ProxyConfig::default(),
            &normalized,
            &ModelProfile::qwen(),
            &upstream,
            &interpreted,
            &crate::agents::NoopCorrectionAgent,
            None,
        )
        .await
        .unwrap();

        assert!(outcome.final_response.choices[0]
            .message
            .content_text()
            .unwrap()
            .contains("malformed tool call"));
        assert!(outcome
            .actions
            .iter()
            .any(|action| action.action == "unrecoverable_tool_call_suppressed"));
    }

    #[tokio::test]
    async fn replaces_dsrs_placeholder_content() {
        let normalized = request_with_tool_prompt(
            "In one short sentence, say what you can help with in this repository.",
            false,
        );
        let upstream = synthetic_response(
            "test",
            "[[ ## content ## ]]\ncontent_value\n[[ ## tool_calls ## ]]\ntool_calls_value\n[[ ## completed ## ]]",
        );
        let interpreted =
            crate::response_interpreter::interpret_response(&upstream, &normalized.tools);

        let outcome = repair_response(
            &ProxyConfig::default(),
            &normalized,
            &ModelProfile::qwen(),
            &upstream,
            &interpreted,
            &crate::agents::NoopCorrectionAgent,
            None,
        )
        .await
        .unwrap();

        assert_eq!(
            outcome.final_response.choices[0]
                .message
                .content_text()
                .as_deref(),
            Some(
                "I can help read files, run commands, edit code, and answer questions about this repository."
            )
        );
    }

    #[tokio::test]
    async fn drops_native_tool_call_missing_required_arguments() {
        let normalized = request_with_tool(false);
        let mut upstream = ChatCompletionResponse::empty_for_model("test");
        upstream.choices.push(ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: Some(Value::Null),
                name: None,
                tool_call_id: None,
                tool_calls: Some(vec![OpenAiToolCall::function("read_file", "{}")]),
                extra: Map::new(),
            },
            finish_reason: Some("tool_calls".to_string()),
            logprobs: None,
            extra: Map::new(),
        });
        let interpreted =
            crate::response_interpreter::interpret_response(&upstream, &normalized.tools);

        let outcome = repair_response(
            &ProxyConfig::default(),
            &normalized,
            &ModelProfile::qwen(),
            &upstream,
            &interpreted,
            &crate::agents::NoopCorrectionAgent,
            None,
        )
        .await
        .unwrap();

        let message = &outcome.final_response.choices[0].message;
        assert!(message.tool_calls.is_none());
        assert!(message
            .content_text()
            .unwrap()
            .contains("malformed tool call"));
        assert!(outcome
            .actions
            .iter()
            .any(|action| action.action == "schema_required_missing"));
        assert!(outcome
            .actions
            .iter()
            .any(|action| action.action == "unrecoverable_tool_call_suppressed"));
    }

    struct FailingCorrectionAgent;

    #[async_trait::async_trait]
    impl CorrectionAgent for FailingCorrectionAgent {
        async fn correct(
            &self,
            _input: CorrectionAgentInput,
        ) -> Result<Option<crate::agents::CorrectionAgentOutput>> {
            anyhow::bail!("correction failed")
        }
    }

    #[tokio::test]
    async fn correction_agent_failure_does_not_abort_response() {
        let normalized = request_with_tool(false);
        let upstream = synthetic_response(
            "test",
            "<tool_call><function=read_file><parameter=\"path\">Cargo.toml</parameter></tool_call>",
        );
        let interpreted =
            crate::response_interpreter::interpret_response(&upstream, &normalized.tools);

        let outcome = repair_response(
            &ProxyConfig::default(),
            &normalized,
            &ModelProfile::qwen(),
            &upstream,
            &interpreted,
            &FailingCorrectionAgent,
            None,
        )
        .await
        .unwrap();

        assert!(outcome.final_response.choices[0]
            .message
            .content_text()
            .unwrap()
            .contains("malformed tool call"));
        assert!(outcome
            .actions
            .iter()
            .any(|action| action.action == "correction_agent_failed"));
    }
}
