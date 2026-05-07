use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use strsim::jaro_winkler;
use uuid::Uuid;

use crate::{
    agents::{
        recent_messages, CorrectionAgent, CorrectionAgentError, CorrectionAgentInput,
        CorrectionAgentOutput,
    },
    config::ProxyConfig,
    model_profile::ModelProfile,
    normalizer::NormalizedRequest,
    openai::{ChatChoice, ChatCompletionResponse, ChatMessage, OpenAiTool, OpenAiToolCall},
    response_interpreter::{
        InterpretedResponse, ResponseFailure, ResponseFailureKind, ToolIntent, ToolIntentSource,
    },
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairOutcome {
    pub final_response: ChatCompletionResponse,
    pub actions: Vec<RepairAction>,
    #[serde(default)]
    pub correction_attempts: Vec<CorrectionAttemptTrace>,
    #[serde(default)]
    pub policy_decisions: Vec<PolicyDecisionTrace>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RepairAction {
    pub action: String,
    pub confidence: f32,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionAttemptTrace {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attempt_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<DateTime<Utc>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<DateTime<Utc>>,
    pub model: String,
    pub profile: String,
    #[serde(default)]
    pub profile_revision: u32,
    #[serde(default)]
    pub profile_source: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub request_adapter_artifact: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub correction_agent_artifact: Option<String>,
    pub correction_model: String,
    #[serde(default)]
    pub tools: Vec<OpenAiTool>,
    #[serde(default)]
    pub recent_messages: Vec<ChatMessage>,
    #[serde(default)]
    pub malformed_response: String,
    pub parser_events: Vec<String>,
    #[serde(default)]
    pub response_failures: Vec<ResponseFailure>,
    pub failure_kinds: Vec<ResponseFailureKind>,
    pub malformed_response_preview: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_output: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub possible: Option<bool>,
    #[serde(default)]
    pub output_tool_calls: Vec<ToolIntent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub output_content: Option<String>,
    pub result: String,
    pub accepted: bool,
    pub confidence: Option<f32>,
    pub explanation: Option<String>,
    pub error: Option<String>,
    pub tool_calls: usize,
    pub content_len: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyDecisionTrace {
    pub stage: String,
    pub decision: String,
    pub reason: String,
    pub failure_kinds: Vec<ResponseFailureKind>,
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
            correction_attempts: Vec::new(),
            policy_decisions: vec![PolicyDecisionTrace {
                stage: "repair".to_string(),
                decision: "pass_through_native_tool_calls".to_string(),
                reason: "native OpenAI tool_calls were valid for the requested schema".to_string(),
                failure_kinds: interpreted.failure_kinds(),
            }],
        });
    }

    let mut actions = Vec::new();
    let mut correction_attempts = Vec::new();
    let mut policy_decisions = Vec::new();
    let mut intents = interpreted.tool_intents.clone();
    let mut using_correction_tool_calls = false;
    let mut corrected_content: Option<(String, f32, String)> = None;
    let should_try_correction = should_try_correction_agent(normalized, interpreted, &intents);
    let correction_available = config.correction.enabled
        && config.policy.correction_agent
        && profile.max_correction_passes > 0;
    policy_decisions.push(PolicyDecisionTrace {
        stage: "repair".to_string(),
        decision: if should_try_correction && correction_available {
            "attempt_correction_agent".to_string()
        } else if should_try_correction {
            "correction_agent_unavailable".to_string()
        } else {
            "skip_correction_agent".to_string()
        },
        reason: correction_policy_reason(
            config,
            normalized,
            profile,
            interpreted,
            &intents,
            correction_available,
        ),
        failure_kinds: interpreted.failure_kinds(),
    });

    if should_try_correction && correction_available {
        let malformed_response = malformed_response_text(upstream_response, interpreted);
        let correction_model = correction_model_for(config, normalized, profile);
        let attempt_id = format!("correction_{}", Uuid::new_v4().simple());
        let attempt_started_at = Utc::now();
        let input = CorrectionAgentInput {
            tools: normalized.tools.clone(),
            recent_messages: recent_messages(
                &normalized.messages,
                config.correction.max_context_messages,
            ),
            malformed_response: malformed_response.clone(),
            parser_events: interpreted.parse_events.clone(),
            response_failures: interpreted.failures.clone(),
            model: normalized.model.clone(),
            profile: profile.clone(),
            api_key_override: correction_api_key,
        };

        match correction_agent.correct(input.clone()).await {
            Ok(Some(output))
                if correction_output_is_accepted(config, &output)
                    && !output.tool_calls.is_empty() =>
            {
                let confidence = output.confidence;
                let explanation = output.explanation.clone();
                correction_attempts.push(build_correction_attempt_trace(
                    &attempt_id,
                    attempt_started_at,
                    correction_model.clone(),
                    &input,
                    "tool_recovery",
                    true,
                    Some(&output),
                    None,
                ));
                actions.push(RepairAction {
                    action: "correction_agent_tool_recovery".to_string(),
                    confidence,
                    reason: explanation,
                });
                if !intents.is_empty() {
                    actions.push(RepairAction {
                        action: "correction_agent_replaced_parser_tool_intents".to_string(),
                        confidence,
                        reason: format!(
                            "accepted correction-agent tool calls replaced {} parser-recovered intent(s)",
                            intents.len()
                        ),
                    });
                }
                intents = output.tool_calls;
                using_correction_tool_calls = true;
            }
            Ok(Some(output)) => {
                if correction_output_is_accepted(config, &output) {
                    if let Some(content) = output
                        .content
                        .clone()
                        .filter(|content| !content.trim().is_empty())
                    {
                        let confidence = output.confidence;
                        let explanation = output.explanation.clone();
                        correction_attempts.push(build_correction_attempt_trace(
                            &attempt_id,
                            attempt_started_at,
                            correction_model.clone(),
                            &input,
                            "content_recovery",
                            true,
                            Some(&output),
                            None,
                        ));
                        actions.push(RepairAction {
                            action: "correction_agent_content_recovery".to_string(),
                            confidence,
                            reason: explanation.clone(),
                        });
                        corrected_content = Some((content, confidence, explanation));
                    } else {
                        correction_attempts.push(build_correction_attempt_trace(
                            &attempt_id,
                            attempt_started_at,
                            correction_model.clone(),
                            &input,
                            "empty_recovery",
                            false,
                            Some(&output),
                            None,
                        ));
                    }
                } else {
                    let result = if !output.possible {
                        "not_possible"
                    } else {
                        "low_confidence"
                    };
                    correction_attempts.push(build_correction_attempt_trace(
                        &attempt_id,
                        attempt_started_at,
                        correction_model.clone(),
                        &input,
                        result,
                        false,
                        Some(&output),
                        None,
                    ));
                }
            }
            Ok(_) => correction_attempts.push(build_correction_attempt_trace(
                &attempt_id,
                attempt_started_at,
                correction_model.clone(),
                &input,
                "no_recovery",
                false,
                None,
                None,
            )),
            Err(error) => {
                let raw_output = correction_error_raw_output(&error);
                let error = error.to_string();
                let mut attempt = build_correction_attempt_trace(
                    &attempt_id,
                    attempt_started_at,
                    correction_model.clone(),
                    &input,
                    "failed",
                    false,
                    None,
                    Some(error.clone()),
                );
                attempt.raw_output = raw_output;
                correction_attempts.push(attempt);
                actions.push(RepairAction {
                    action: "correction_agent_failed".to_string(),
                    confidence: 0.0,
                    reason: error,
                });
            }
        }
    }

    let repaired_tool_calls = repair_tool_intents(config, normalized, intents, &mut actions);
    let mut final_response = upstream_response.clone();
    ensure_first_choice(&mut final_response, &normalized.model);

    if repaired_tool_calls.is_empty() {
        if let Some((content, confidence, reason)) = corrected_content {
            apply_corrected_content(&mut final_response, content, confidence, reason);
            policy_decisions.push(PolicyDecisionTrace {
                stage: "repair".to_string(),
                decision: "emit_corrected_content".to_string(),
                reason: "correction agent recovered user-facing content".to_string(),
                failure_kinds: interpreted.failure_kinds(),
            });
            return Ok(RepairOutcome {
                final_response,
                actions,
                correction_attempts,
                policy_decisions,
            });
        }

        if should_suppress_unrecoverable_tool_output(interpreted)
            || should_suppress_expected_tool_failure(normalized, interpreted)
        {
            suppress_unrecoverable_tool_output(&mut final_response, &mut actions);
            policy_decisions.push(PolicyDecisionTrace {
                stage: "repair".to_string(),
                decision: "suppress_unrecoverable_tool_output".to_string(),
                reason: "malformed tool output could not be safely recovered".to_string(),
                failure_kinds: interpreted.failure_kinds(),
            });
        } else if should_replace_unusable_assistant_content(normalized, interpreted) {
            replace_unusable_assistant_content(&mut final_response, normalized, &mut actions);
            policy_decisions.push(PolicyDecisionTrace {
                stage: "repair".to_string(),
                decision: "replace_unusable_assistant_content".to_string(),
                reason: "assistant content was empty, placeholder-only, or scratchpad-only"
                    .to_string(),
                failure_kinds: interpreted.failure_kinds(),
            });
        } else {
            map_reasoning_content_if_needed(&mut final_response, interpreted);
            policy_decisions.push(PolicyDecisionTrace {
                stage: "repair".to_string(),
                decision: "pass_through_interpreted_content".to_string(),
                reason: "no safe repair action was required after interpretation".to_string(),
                failure_kinds: interpreted.failure_kinds(),
            });
        }
        return Ok(RepairOutcome {
            final_response,
            actions,
            correction_attempts,
            policy_decisions,
        });
    }

    let first = final_response.choices.first_mut().expect("choice ensured");
    first.message.set_content_null();
    first.message.tool_calls = Some(repaired_tool_calls);
    first.finish_reason = Some("tool_calls".to_string());

    if !using_correction_tool_calls
        && interpreted
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
        correction_attempts,
        policy_decisions: {
            policy_decisions.push(PolicyDecisionTrace {
                stage: "repair".to_string(),
                decision: "emit_repaired_tool_calls".to_string(),
                reason: "repaired tool intents were emitted as OpenAI tool_calls".to_string(),
                failure_kinds: interpreted.failure_kinds(),
            });
            policy_decisions
        },
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
    interpreter_failures_require_model_correction(interpreted)
        || (intents.is_empty() && interpreted.suspicious_stop)
        || intents
            .iter()
            .any(|intent| intent_requires_model_correction(normalized, intent))
}

fn interpreter_failures_require_model_correction(interpreted: &InterpretedResponse) -> bool {
    interpreted.has_any_failure(&[
        ResponseFailureKind::NativeMalformedJsonArguments,
        ResponseFailureKind::DsrsContractViolation,
        ResponseFailureKind::DsrsContentOutsideTaggedFields,
        ResponseFailureKind::DsrsInvalidToolCallsJson,
        ResponseFailureKind::DsrsInvalidToolCallsShape,
        ResponseFailureKind::EmptyDsrsOutput,
        ResponseFailureKind::DsrsPlaceholderOnly,
        ResponseFailureKind::TemplateLeak,
        ResponseFailureKind::PromptEcho,
        ResponseFailureKind::PrematureToolStop,
        ResponseFailureKind::MalformedKnownToolCall,
        ResponseFailureKind::UntaggedDsrsLikeOutput,
        ResponseFailureKind::SchemaViolation,
    ])
}

fn correction_policy_reason(
    config: &ProxyConfig,
    normalized: &NormalizedRequest,
    profile: &ModelProfile,
    interpreted: &InterpretedResponse,
    intents: &[ToolIntent],
    correction_available: bool,
) -> String {
    if !correction_available
        && (interpreter_failures_require_model_correction(interpreted)
            || (intents.is_empty() && interpreted.suspicious_stop)
            || intents
                .iter()
                .any(|intent| intent_requires_model_correction(normalized, intent)))
    {
        return format!(
            "correction would be useful but is unavailable: enabled={} policy_enabled={} max_passes={}",
            config.correction.enabled, config.policy.correction_agent, profile.max_correction_passes
        );
    }
    if interpreter_failures_require_model_correction(interpreted) {
        return "typed response failures require model-based correction".to_string();
    }
    if intents.is_empty() && interpreted.suspicious_stop {
        return "interpreter found a suspicious stop with no valid tool intents".to_string();
    }
    if intents
        .iter()
        .any(|intent| intent_requires_model_correction(normalized, intent))
    {
        return "one or more tool intents have malformed or schema-incomplete arguments"
            .to_string();
    }
    "response can be handled without correction agent".to_string()
}

fn correction_model_for(
    config: &ProxyConfig,
    normalized: &NormalizedRequest,
    profile: &ModelProfile,
) -> String {
    profile
        .correction_model
        .clone()
        .or_else(|| config.correction.default_model.clone())
        .unwrap_or_else(|| normalized.model.clone())
}

fn correction_output_is_accepted(config: &ProxyConfig, output: &CorrectionAgentOutput) -> bool {
    output.possible && output.confidence >= config.correction.min_confidence
}

fn build_correction_attempt_trace(
    attempt_id: &str,
    started_at: DateTime<Utc>,
    correction_model: String,
    input: &CorrectionAgentInput,
    result: &str,
    accepted: bool,
    output: Option<&CorrectionAgentOutput>,
    error: Option<String>,
) -> CorrectionAttemptTrace {
    let output_tool_calls = output
        .map(|output| output.tool_calls.clone())
        .unwrap_or_default();
    let output_content = output.and_then(|output| output.content.clone());
    CorrectionAttemptTrace {
        attempt_id: Some(attempt_id.to_string()),
        started_at: Some(started_at),
        completed_at: Some(Utc::now()),
        model: input.model.clone(),
        profile: input.profile.name.clone(),
        profile_revision: input.profile.revision,
        profile_source: input.profile.source.clone(),
        request_adapter_artifact: input.profile.request_adapter_artifact.clone(),
        correction_agent_artifact: input.profile.correction_agent_artifact.clone(),
        correction_model,
        tools: input.tools.clone(),
        recent_messages: input.recent_messages.clone(),
        malformed_response: input.malformed_response.clone(),
        parser_events: input.parser_events.clone(),
        response_failures: input.response_failures.clone(),
        failure_kinds: input
            .response_failures
            .iter()
            .map(|failure| failure.kind)
            .collect(),
        malformed_response_preview: preview(&input.malformed_response),
        raw_output: output.and_then(|output| output.raw_output.clone()),
        possible: output.map(|output| output.possible),
        output_tool_calls,
        output_content: output_content.clone(),
        result: result.to_string(),
        accepted,
        confidence: output.map(|output| output.confidence),
        explanation: output.map(|output| output.explanation.clone()),
        error,
        tool_calls: output.map_or(0, |output| output.tool_calls.len()),
        content_len: output_content.as_ref().map_or(0, String::len),
    }
}

fn correction_error_raw_output(error: &anyhow::Error) -> Option<String> {
    error
        .downcast_ref::<CorrectionAgentError>()
        .and_then(|error| error.raw_output().map(str::to_string))
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
    !interpreted.tool_intents.is_empty()
        || interpreted
            .content
            .as_deref()
            .map(contains_unrecoverable_tool_output)
            .unwrap_or(false)
        || (interpreted.has_any_failure(&[
            ResponseFailureKind::DsrsInvalidToolCallsJson,
            ResponseFailureKind::DsrsInvalidToolCallsShape,
            ResponseFailureKind::MalformedKnownToolCall,
            ResponseFailureKind::UntaggedDsrsLikeOutput,
        ]) && interpreted
            .content
            .as_deref()
            .map(unusable_assistant_content)
            .unwrap_or(true))
}

fn contains_unrecoverable_tool_output(content: &str) -> bool {
    let lower = content.to_ascii_lowercase();
    lower.contains("<tool_call")
        || lower.contains("<tool ")
        || lower.contains("<function")
        || lower.contains("function=")
        || lower.contains("tool_calls")
        || lower.contains("[\"content\"")
        || lower.contains("[\"tool_calls\"")
        || lower.contains("[## completed ##]")
        || lower.contains("![## completed ##]")
        || (lower.contains("\"name\"") && lower.contains("\"arguments\""))
        || lower.contains("[[ ## tool_calls ## ]]")
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
        "-" | "--"
            | "---"
            | "..."
            | "content"
            | "content_value"
            | "tool_calls_value"
            | "completed_marker"
    ) || lower.starts_with("tool_calls")
        || lower.contains("\ntool_calls")
        || (lower.contains("\n[]") && lower.ends_with("completed"))
        || lower.starts_with("thinking process")
        || lower.starts_with("thought process")
        || lower.starts_with("chain of thought")
        || trimmed.contains("[[ ##")
        || empty_json_response_content(trimmed)
}

fn empty_json_response_content(content: &str) -> bool {
    parse_json_lenient(content).is_some_and(|value| match value {
        Value::Null => true,
        Value::Array(items) => items.is_empty(),
        Value::Object(object) => object.is_empty(),
        _ => false,
    })
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

fn apply_corrected_content(
    response: &mut ChatCompletionResponse,
    content: String,
    confidence: f32,
    reason: String,
) {
    ensure_first_choice(response, "");
    let first = response.choices.first_mut().expect("choice ensured");
    first.message.tool_calls = None;
    first.message.set_content_text(content);
    remove_reasoning_extra(&mut first.message);
    first.finish_reason = Some("stop".to_string());
    tracing::debug!(
        confidence,
        reason = %reason,
        "applied correction-agent content recovery"
    );
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
    let content = upstream_response
        .choices
        .first()
        .and_then(|choice| choice.message.content_text())
        .or_else(|| interpreted.content.clone())
        .or_else(|| serde_json::to_string(upstream_response).ok())
        .unwrap_or_default();
    let reasoning = upstream_response
        .choices
        .first()
        .and_then(|choice| choice.message.reasoning_text())
        .or_else(|| interpreted.reasoning.clone());

    if let Some(reasoning) = reasoning.filter(|reasoning| !reasoning.trim().is_empty()) {
        format!("assistant_content:\n{content}\n\nassistant_reasoning:\n{reasoning}")
    } else {
        content
    }
}

fn preview(value: &str) -> String {
    const LIMIT: usize = 1000;
    let mut out = value.chars().take(LIMIT).collect::<String>();
    if value.chars().count() > LIMIT {
        out.push_str("...");
    }
    out
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
        request_with_named_tool("read_file", prompt, parallel)
    }

    fn request_with_named_tool(name: &str, prompt: &str, parallel: bool) -> NormalizedRequest {
        normalize_request(ChatCompletionRequest {
            model: "test".to_string(),
            messages: vec![ChatMessage::new("user", prompt)],
            tools: Some(vec![OpenAiTool {
                tool_type: "function".to_string(),
                function: OpenAiFunctionTool {
                    name: name.to_string(),
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
    async fn suppresses_empty_label_free_dsrs_content_when_correction_is_unavailable() {
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
            Some("The upstream model emitted a malformed tool call that could not be safely recovered.")
        );
        assert!(outcome
            .actions
            .iter()
            .any(|action| action.action == "unrecoverable_tool_call_suppressed"));
    }

    struct ContentCorrectionAgent;

    #[async_trait::async_trait]
    impl CorrectionAgent for ContentCorrectionAgent {
        async fn correct(
            &self,
            input: CorrectionAgentInput,
        ) -> Result<Option<crate::agents::CorrectionAgentOutput>> {
            assert!(input.malformed_response.contains("tool_calls: []"));
            Ok(Some(crate::agents::CorrectionAgentOutput {
                tool_calls: Vec::new(),
                content: Some("Just coding everything. How about you?".to_string()),
                possible: true,
                confidence: 0.94,
                explanation: "recovered content from malformed DSRs-like output".to_string(),
                raw_output: Some("content correction fixture".to_string()),
            }))
        }
    }

    #[tokio::test]
    async fn correction_agent_can_recover_content_only_contract_violation() {
        let normalized = request_with_tool_prompt("hey what's up?", false);
        let upstream = synthetic_response(
            "test",
            "content: Just coding everything. How about you?\n\ntool_calls: []\n\n[[ ## completed ## ]]",
        );
        let interpreted =
            crate::response_interpreter::interpret_response(&upstream, &normalized.tools);

        assert!(interpreted.suspicious_stop);
        assert!(interpreted.tool_intents.is_empty());

        let outcome = repair_response(
            &ProxyConfig::default(),
            &normalized,
            &ModelProfile::qwen(),
            &upstream,
            &interpreted,
            &ContentCorrectionAgent,
            None,
        )
        .await
        .unwrap();

        assert_eq!(
            outcome.final_response.choices[0]
                .message
                .content_text()
                .as_deref(),
            Some("Just coding everything. How about you?")
        );
        assert!(outcome.final_response.choices[0]
            .message
            .tool_calls
            .is_none());
        assert!(outcome
            .actions
            .iter()
            .any(|action| action.action == "correction_agent_content_recovery"));
    }

    struct EmptyDsrsCorrectionAgent;

    #[async_trait::async_trait]
    impl CorrectionAgent for EmptyDsrsCorrectionAgent {
        async fn correct(
            &self,
            input: CorrectionAgentInput,
        ) -> Result<Option<crate::agents::CorrectionAgentOutput>> {
            assert!(input.malformed_response.contains("[[ ## content ## ]]"));
            assert!(input
                .response_failures
                .iter()
                .any(|failure| failure.kind == ResponseFailureKind::EmptyDsrsOutput));
            assert!(input
                .parser_events
                .iter()
                .any(|event| event.contains("empty or no-op content and no tool calls")));
            Ok(Some(crate::agents::CorrectionAgentOutput {
                tool_calls: vec![ToolIntent {
                    name: "read_file".to_string(),
                    arguments: Some(json!({"path":"README.md"})),
                    raw_arguments: r#"{"path":"README.md"}"#.to_string(),
                    source: ToolIntentSource::CorrectionAgent,
                    confidence: 0.93,
                }],
                content: None,
                possible: true,
                confidence: 0.93,
                explanation: "empty DSRs output should inspect README for project context"
                    .to_string(),
                raw_output: Some("empty DSRs correction fixture".to_string()),
            }))
        }
    }

    #[tokio::test]
    async fn empty_dsrs_output_routes_to_correction_agent() {
        let normalized =
            request_with_tool_prompt("can you tell me more about this project?", false);
        let upstream = synthetic_response(
            "google/gemma-4-26b-a4b-it",
            "[[ ## content ## ]]\n[[ ## tool_calls ## ]]\n[]\n[[ ## completed ## ]]",
        );
        let interpreted =
            crate::response_interpreter::interpret_response(&upstream, &normalized.tools);

        assert!(interpreted.suspicious_stop);
        assert!(interpreted.has_failure(ResponseFailureKind::EmptyDsrsOutput));

        let outcome = repair_response(
            &ProxyConfig::default(),
            &normalized,
            &ModelProfile::gemma(),
            &upstream,
            &interpreted,
            &EmptyDsrsCorrectionAgent,
            None,
        )
        .await
        .unwrap();

        let calls = outcome.final_response.choices[0]
            .message
            .tool_calls
            .as_ref()
            .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read_file");
        assert_eq!(calls[0].function.arguments, r#"{"path":"README.md"}"#);
        assert!(outcome
            .actions
            .iter()
            .any(|action| action.action == "correction_agent_tool_recovery"));
        assert!(outcome.policy_decisions.iter().any(|decision| {
            decision.decision == "attempt_correction_agent"
                && decision
                    .failure_kinds
                    .contains(&ResponseFailureKind::EmptyDsrsOutput)
        }));
    }

    struct DuplicateReadmeCorrectionAgent;

    #[async_trait::async_trait]
    impl CorrectionAgent for DuplicateReadmeCorrectionAgent {
        async fn correct(
            &self,
            input: CorrectionAgentInput,
        ) -> Result<Option<crate::agents::CorrectionAgentOutput>> {
            assert!(input.malformed_response.contains("README.md"));
            assert!(input
                .response_failures
                .iter()
                .any(|failure| failure.kind == ResponseFailureKind::DsrsContractViolation));
            Ok(Some(crate::agents::CorrectionAgentOutput {
                tool_calls: vec![ToolIntent {
                    name: "read".to_string(),
                    arguments: Some(json!({"path":"README.md"})),
                    raw_arguments: r#"{"path":"README.md"}"#.to_string(),
                    source: ToolIntentSource::CorrectionAgent,
                    confidence: 0.97,
                }],
                content: None,
                possible: true,
                confidence: 0.97,
                explanation: "added the missing DSRs completed marker around the read call"
                    .to_string(),
                raw_output: Some("duplicate read correction fixture".to_string()),
            }))
        }
    }

    #[tokio::test]
    async fn correction_tool_recovery_replaces_parser_intents_without_duplication() {
        let normalized = request_with_named_tool(
            "read",
            "what do you know about this project? can you dive into the readme?",
            false,
        );
        let upstream = synthetic_response(
            "google/gemma-4-26b-a4b-it",
            r#"[[ ## content ## ]]
[[ ## tool_calls ## ]]
[
  {
    "name": "read",
    "arguments": {
      "path": "README.md"
    }
  }
]"#,
        );
        let interpreted =
            crate::response_interpreter::interpret_response(&upstream, &normalized.tools);

        assert_eq!(interpreted.tool_intents.len(), 1);
        assert!(interpreted.has_failure(ResponseFailureKind::DsrsContractViolation));

        let outcome = repair_response(
            &ProxyConfig::default(),
            &normalized,
            &ModelProfile::gemma(),
            &upstream,
            &interpreted,
            &DuplicateReadmeCorrectionAgent,
            None,
        )
        .await
        .unwrap();

        let calls = outcome.final_response.choices[0]
            .message
            .tool_calls
            .as_ref()
            .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read");
        assert_eq!(calls[0].function.arguments, r#"{"path":"README.md"}"#);
        assert!(outcome
            .actions
            .iter()
            .any(|action| action.action == "correction_agent_tool_recovery"));
        assert!(outcome
            .actions
            .iter()
            .any(|action| action.action == "correction_agent_replaced_parser_tool_intents"));
        assert!(!outcome
            .actions
            .iter()
            .any(|action| action.action == "content_tool_call_extracted"));
    }

    struct ReadmeCorrectionAgent;

    #[async_trait::async_trait]
    impl CorrectionAgent for ReadmeCorrectionAgent {
        async fn correct(
            &self,
            input: CorrectionAgentInput,
        ) -> Result<Option<crate::agents::CorrectionAgentOutput>> {
            assert!(input.malformed_response.contains("packages/ai/README.md"));
            assert!(input
                .parser_events
                .iter()
                .any(|event| event.contains("assistant appears")));
            Ok(Some(crate::agents::CorrectionAgentOutput {
                tool_calls: vec![
                    ToolIntent {
                        name: "read".to_string(),
                        arguments: Some(json!({"path":"packages/ai/README.md"})),
                        raw_arguments: r#"{"path":"packages/ai/README.md"}"#.to_string(),
                        source: ToolIntentSource::CorrectionAgent,
                        confidence: 0.95,
                    },
                    ToolIntent {
                        name: "read".to_string(),
                        arguments: Some(json!({"path":"packages/agent/README.md"})),
                        raw_arguments: r#"{"path":"packages/agent/README.md"}"#.to_string(),
                        source: ToolIntentSource::CorrectionAgent,
                        confidence: 0.95,
                    },
                ],
                content: None,
                possible: true,
                confidence: 0.95,
                explanation: "recovered adjacent JSON tool-call array".to_string(),
                raw_output: Some("tool correction fixture".to_string()),
            }))
        }
    }

    #[tokio::test]
    async fn adjacent_json_tool_payload_routes_to_correction_agent() {
        let normalized = request_with_named_tool(
            "read",
            "do each of them have a readme that says a little more than that though?",
            true,
        );
        let upstream = synthetic_response(
            "qwen/qwen3.5-9b",
            r#"[]

[
  {"name":"read","arguments":{"path":"packages/ai/README.md"}},
  {"name":"read","arguments":{"path":"packages/agent/README.md"}}
]"#,
        );
        let interpreted =
            crate::response_interpreter::interpret_response(&upstream, &normalized.tools);

        assert!(interpreted.suspicious_stop);
        assert!(interpreted.tool_intents.is_empty());

        let outcome = repair_response(
            &ProxyConfig::default(),
            &normalized,
            &ModelProfile::qwen(),
            &upstream,
            &interpreted,
            &ReadmeCorrectionAgent,
            None,
        )
        .await
        .unwrap();

        let calls = outcome.final_response.choices[0]
            .message
            .tool_calls
            .as_ref()
            .unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].function.name, "read");
        assert_eq!(
            calls[0].function.arguments,
            r#"{"path":"packages/ai/README.md"}"#
        );
        assert!(outcome.final_response.choices[0]
            .message
            .content_text()
            .is_none());
        assert!(outcome
            .actions
            .iter()
            .any(|action| action.action == "correction_agent_tool_recovery"));
    }

    struct ContractViolationCorrectionAgent;

    #[async_trait::async_trait]
    impl CorrectionAgent for ContractViolationCorrectionAgent {
        async fn correct(
            &self,
            input: CorrectionAgentInput,
        ) -> Result<Option<crate::agents::CorrectionAgentOutput>> {
            assert!(input.malformed_response.contains("packages/ai/README.md"));
            assert!(input.response_failures.iter().any(|failure| {
                failure.kind == ResponseFailureKind::DsrsContentOutsideTaggedFields
            }));
            assert!(input
                .response_failures
                .iter()
                .any(|failure| failure.kind == ResponseFailureKind::PromptEcho));
            Ok(Some(crate::agents::CorrectionAgentOutput {
                tool_calls: vec![ToolIntent {
                    name: "read".to_string(),
                    arguments: Some(json!({"path":"packages/ai/README.md"})),
                    raw_arguments: r#"{"path":"packages/ai/README.md"}"#.to_string(),
                    source: ToolIntentSource::CorrectionAgent,
                    confidence: 0.96,
                }],
                content: None,
                possible: true,
                confidence: 0.96,
                explanation: "recovered tool call from malformed tagged output".to_string(),
                raw_output: Some("contract correction fixture".to_string()),
            }))
        }
    }

    #[tokio::test]
    async fn dsrs_contract_violation_routes_to_correction_agent_with_typed_failures() {
        let normalized = request_with_named_tool(
            "read",
            "very cool. can you dive into each package and let me know more information about each?",
            true,
        );
        let upstream = synthetic_response(
            "qwen/qwen3.5-9b",
            r#"I should inspect the package README files.

[
  {"name":"read","arguments":{"path":"packages/ai/README.md"}}
]

[[ ## system_context ## ]]
Do not repeat this.

[[ ## content ## ]]
---
[[ ## tool_calls ## ]]
[]
[[ ## completed ## ]]"#,
        );
        let interpreted =
            crate::response_interpreter::interpret_response(&upstream, &normalized.tools);

        assert!(interpreted.has_failure(ResponseFailureKind::DsrsContractViolation));
        assert!(interpreted.has_failure(ResponseFailureKind::DsrsContentOutsideTaggedFields));
        assert!(interpreted.has_failure(ResponseFailureKind::PromptEcho));

        let outcome = repair_response(
            &ProxyConfig::default(),
            &normalized,
            &ModelProfile::qwen(),
            &upstream,
            &interpreted,
            &ContractViolationCorrectionAgent,
            None,
        )
        .await
        .unwrap();

        let calls = outcome.final_response.choices[0]
            .message
            .tool_calls
            .as_ref()
            .unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "read");
        assert_eq!(
            calls[0].function.arguments,
            r#"{"path":"packages/ai/README.md"}"#
        );
        assert!(outcome
            .actions
            .iter()
            .any(|action| action.action == "correction_agent_tool_recovery"));
        assert!(outcome
            .correction_attempts
            .iter()
            .any(|attempt| attempt.result == "tool_recovery" && attempt.accepted));
        assert!(outcome.policy_decisions.iter().any(|decision| {
            decision.decision == "attempt_correction_agent"
                && decision
                    .failure_kinds
                    .contains(&ResponseFailureKind::DsrsContractViolation)
        }));
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
        assert!(interpreted.has_failure(ResponseFailureKind::SchemaViolation));

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

    #[tokio::test]
    async fn suppresses_array_pair_contract_violation_when_correction_fails() {
        let normalized = request_with_named_tool(
            "read",
            "very cool. can you dive into each package and let me know more information about each?",
            true,
        );
        let mut upstream = synthetic_response(
            "qwen/qwen3.5-9b",
            "[\"content\",\"\"]\n\n[\"tool_calls\",\"[]\\\"]\n\n![## completed ##]",
        );
        upstream.choices[0].message.extra.insert(
            "reasoning".to_string(),
            json!("I should read package README files before answering."),
        );
        let interpreted =
            crate::response_interpreter::interpret_response(&upstream, &normalized.tools);

        assert!(interpreted.suspicious_stop);

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

        let content = outcome.final_response.choices[0]
            .message
            .content_text()
            .unwrap();
        assert!(content.contains("malformed tool call"));
        assert!(!content.contains("[\"content\""));
        assert!(outcome
            .actions
            .iter()
            .any(|action| action.action == "correction_agent_failed"));
        assert!(outcome
            .actions
            .iter()
            .any(|action| action.action == "unrecoverable_tool_call_suppressed"));
    }
}
