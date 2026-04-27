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

    if intents.is_empty()
        && interpreted.suspicious_stop
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
            malformed_response: interpreted.content.clone().unwrap_or_default(),
            parser_events: interpreted.parse_events.clone(),
            model: normalized.model.clone(),
            profile: profile.clone(),
        };

        if let Some(output) = correction_agent.correct(input).await? {
            if !output.tool_calls.is_empty() {
                actions.push(RepairAction {
                    action: "correction_agent_tool_recovery".to_string(),
                    confidence: output.confidence,
                    reason: output.explanation,
                });
                intents.extend(output.tool_calls);
            }
        }
    }

    let repaired_tool_calls = repair_tool_intents(config, normalized, intents, &mut actions);
    let mut final_response = upstream_response.clone();
    ensure_first_choice(&mut final_response, &normalized.model);

    if repaired_tool_calls.is_empty() {
        map_reasoning_content_if_needed(&mut final_response, interpreted);
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
        intent.arguments.is_some()
            && normalized
                .tools
                .iter()
                .any(|tool| tool.function.name == intent.name)
    })
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

        let arguments = match intent.arguments {
            Some(value) => serialize_arguments(value),
            None if config.policy.deterministic_json_repair => {
                if let Some(value) = parse_json_lenient(&intent.raw_arguments) {
                    actions.push(RepairAction {
                        action: "json_arguments_repaired".to_string(),
                        confidence: 0.95,
                        reason: "parsed malformed JSON arguments with tolerant parser".to_string(),
                    });
                    serialize_arguments(value)
                } else {
                    intent.raw_arguments
                }
            }
            None => intent.raw_arguments,
        };

        out.push(OpenAiToolCall::function(name, arguments));
    }

    if !normalized.parallel_tool_calls && out.len() < normalized.tools.len() {
        let attempted_count = out.len();
        if attempted_count > limit {
            actions.push(RepairAction {
                action: "parallel_tool_calls_truncated".to_string(),
                confidence: 1.0,
                reason: "client disabled parallel tool calls".to_string(),
            });
        }
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
    if first.message.content_text().is_none() {
        first.message.set_content_text(content);
    }
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
        normalize_request(ChatCompletionRequest {
            model: "test".to_string(),
            messages: vec![ChatMessage::new("user", "read")],
            tools: Some(vec![OpenAiTool {
                tool_type: "function".to_string(),
                function: OpenAiFunctionTool {
                    name: "read_file".to_string(),
                    description: None,
                    parameters: json!({"type":"object"}),
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
}
