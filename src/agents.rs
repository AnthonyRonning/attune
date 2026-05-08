use std::{
    collections::HashMap,
    fmt,
    panic::{self, AssertUnwindSafe},
};

use anyhow::{Context, Result};
use async_trait::async_trait;
use dspy_rs::{
    adapter::Adapter, example, ChatAdapter, Message, MetaSignature, Prediction, Signature, LM,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    config::{CorrectionConfig, UpstreamConfig},
    model_profile::ModelProfile,
    openai::{ChatMessage, OpenAiTool},
    response_interpreter::{ResponseFailure, ToolIntent, ToolIntentSource},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionAgentInput {
    pub tools: Vec<OpenAiTool>,
    pub recent_messages: Vec<ChatMessage>,
    pub malformed_response: String,
    pub parser_events: Vec<String>,
    #[serde(default)]
    pub response_failures: Vec<ResponseFailure>,
    pub model: String,
    pub profile: ModelProfile,
    #[serde(skip)]
    pub api_key_override: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionAgentOutput {
    pub tool_calls: Vec<ToolIntent>,
    pub content: Option<String>,
    #[serde(default = "default_possible")]
    pub possible: bool,
    pub confidence: f32,
    pub explanation: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_output: Option<String>,
}

fn default_possible() -> bool {
    true
}

#[derive(Debug)]
pub struct CorrectionAgentError {
    message: String,
    raw_output: Option<String>,
}

impl CorrectionAgentError {
    fn invalid_dsrs_output(raw_output: String, detail: impl Into<String>) -> Self {
        let detail = detail.into();
        Self {
            message: format!("correction agent returned invalid DSRs output: {detail}"),
            raw_output: Some(raw_output),
        }
    }

    pub fn raw_output(&self) -> Option<&str> {
        self.raw_output.as_deref()
    }
}

impl fmt::Display for CorrectionAgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for CorrectionAgentError {}

#[async_trait]
pub trait CorrectionAgent: Send + Sync {
    async fn correct(&self, input: CorrectionAgentInput) -> Result<Option<CorrectionAgentOutput>>;
}

#[derive(Debug, Default)]
pub struct NoopCorrectionAgent;

#[async_trait]
impl CorrectionAgent for NoopCorrectionAgent {
    async fn correct(&self, _input: CorrectionAgentInput) -> Result<Option<CorrectionAgentOutput>> {
        Ok(None)
    }
}

#[derive(Clone)]
pub struct DsrsCorrectionAgent {
    upstream: UpstreamConfig,
    correction: CorrectionConfig,
}

impl DsrsCorrectionAgent {
    pub fn new(upstream: UpstreamConfig, correction: CorrectionConfig) -> Self {
        Self {
            upstream,
            correction,
        }
    }

    fn model_for(&self, input: &CorrectionAgentInput) -> String {
        input
            .profile
            .correction_model
            .clone()
            .or_else(|| self.correction.default_model.clone())
            .unwrap_or_else(|| input.model.clone())
    }
}

#[Signature]
struct CorrectMalformedToolResponse {
    /// You are a strict DSRs model-response correction agent. Recover a malformed
    /// assistant response by filling exactly the requested DSRs output fields.
    /// If clear tool calls were intended, put them in tool_calls and optionally
    /// preserve brief user-facing content. If no tool call was intended, put the
    /// user-facing text in content and use an empty tool_calls array. An empty DSRs
    /// response with empty content and [] tool_calls is malformed; recover only when
    /// the conversation and tools make the next action or answer clear. If the
    /// malformed response is ambiguous or unsafe to repair, set possible to false.
    /// Do not invent tools, arguments, or facts. Do not emit prose outside the DSRs
    /// field markers.
    #[input(desc = "OpenAI-compatible tool definitions")]
    pub available_tools: String,

    #[input(desc = "Recent conversation context")]
    pub recent_messages: String,

    #[input(desc = "Malformed assistant response to correct")]
    pub malformed_response: String,

    #[input(desc = "Parser diagnostics from deterministic parsing")]
    pub parser_events: String,

    #[input(desc = "Typed response failure diagnostics from deterministic parsing")]
    pub response_failures: String,

    #[output(desc = "Whether a safe repair is possible. Emit true or false.")]
    pub possible: bool,

    #[output(desc = "Repair confidence as a number from 0.0 to 1.0.")]
    pub confidence: f32,

    #[output(desc = "Brief explanation of the repair decision.")]
    pub explanation: String,

    #[output(desc = "Plain user-facing content; may be empty when only tool calls are needed.")]
    pub content: String,

    #[output(
        desc = "JSON array of {\"name\": string, \"arguments\": object}; [] when no tool call is intended."
    )]
    pub tool_calls: Vec<CorrectionSignatureToolCall>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct CorrectionSignatureToolCall {
    name: String,
    #[serde(default)]
    arguments: Value,
}

impl CorrectionSignatureToolCall {
    fn into_tool_intent(self, confidence: f32) -> Option<ToolIntent> {
        let arguments = self.arguments;
        let raw_arguments = match &arguments {
            Value::String(raw) => raw.clone(),
            other => serde_json::to_string(other).unwrap_or_else(|_| "{}".to_string()),
        };

        Some(ToolIntent {
            name: self.name,
            arguments: Some(arguments),
            raw_arguments,
            source: ToolIntentSource::CorrectionAgent,
            confidence,
        })
    }
}

#[async_trait]
impl CorrectionAgent for DsrsCorrectionAgent {
    async fn correct(&self, input: CorrectionAgentInput) -> Result<Option<CorrectionAgentOutput>> {
        let Some(api_key) = input
            .api_key_override
            .clone()
            .or_else(|| self.upstream.api_key.clone())
        else {
            return Ok(None);
        };

        let model = self.model_for(&input);
        let lm = LM::builder()
            .base_url(self.upstream.base_url.clone())
            .api_key(api_key)
            .model(model)
            .temperature(0.1)
            .build()
            .await
            .context("failed to build DSRs correction LM")?;

        let mut signature = CorrectMalformedToolResponse::new();
        if let Some(instruction) = input
            .profile
            .correction_instruction
            .as_deref()
            .map(str::trim)
            .filter(|instruction| !instruction.is_empty())
        {
            signature
                .update_instruction(instruction.to_string())
                .context("failed to apply correction-agent profile instruction")?;
        }
        let adapter = ChatAdapter;
        let available_tools = serde_json::to_string_pretty(&input.tools)?;
        let recent_messages = serde_json::to_string_pretty(&input.recent_messages)?;
        let parser_events = input.parser_events.join("\n");
        let response_failures = serde_json::to_string_pretty(&input.response_failures)?;

        let chat = adapter.format(
            &signature,
            example! {
                "available_tools": "input" => available_tools,
                "recent_messages": "input" => recent_messages,
                "malformed_response": "input" => input.malformed_response,
                "parser_events": "input" => parser_events,
                "response_failures": "input" => response_failures
            },
        );
        let response = lm
            .call(chat, Vec::new())
            .await
            .context("DSRs correction LM call failed")?;
        let raw_output = response.output.content().to_string();
        let parsed =
            parse_correction_prediction_data(&adapter, &signature, response.output, &raw_output)?;
        let prediction = Prediction::new(parsed, response.usage);

        correction_output_from_prediction(&prediction, raw_output).map(Some)
    }
}

fn parse_correction_prediction_data(
    adapter: &ChatAdapter,
    signature: &CorrectMalformedToolResponse,
    response: Message,
    raw_output: &str,
) -> Result<HashMap<String, Value>> {
    match parse_chat_adapter_response_silently(adapter, signature, response) {
        Ok(parsed) => Ok(parsed),
        Err(adapter_detail) => parse_correction_output_fallback(raw_output).map_err(|detail| {
            CorrectionAgentError::invalid_dsrs_output(
                raw_output.to_string(),
                format!("{adapter_detail}; fallback DSRs parser failed: {detail}"),
            )
            .into()
        }),
    }
}

fn parse_chat_adapter_response_silently(
    adapter: &ChatAdapter,
    signature: &CorrectMalformedToolResponse,
    response: Message,
) -> std::result::Result<HashMap<String, Value>, &'static str> {
    let previous_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let parsed = panic::catch_unwind(AssertUnwindSafe(|| {
        adapter.parse_response(signature, response)
    }));
    panic::set_hook(previous_hook);

    parsed.map_err(|_| "typed DSRs fields could not be parsed by dspy-rs ChatAdapter")
}

fn parse_correction_output_fallback(
    raw_output: &str,
) -> std::result::Result<HashMap<String, Value>, String> {
    let mut data = HashMap::new();

    let possible = parse_bool_field(required_dsrs_field(raw_output, "possible")?)
        .ok_or_else(|| "possible field was not a boolean".to_string())?;
    let confidence = parse_number_field(required_dsrs_field(raw_output, "confidence")?)
        .ok_or_else(|| "confidence field was not a number".to_string())?
        .clamp(0.0, 1.0);
    let explanation = required_dsrs_field(raw_output, "explanation")?
        .trim()
        .to_string();
    let content = required_dsrs_field(raw_output, "content")?
        .trim()
        .to_string();
    let tool_calls = parse_json_lenient(required_dsrs_field(raw_output, "tool_calls")?)
        .ok_or_else(|| "tool_calls field was not valid JSON or JSON5".to_string())?;
    if !tool_calls.is_array() {
        return Err("tool_calls field was not a JSON array".to_string());
    }

    data.insert("possible".to_string(), Value::Bool(possible));
    data.insert("confidence".to_string(), serde_json::json!(confidence));
    data.insert("explanation".to_string(), Value::String(explanation));
    data.insert("content".to_string(), Value::String(content));
    data.insert("tool_calls".to_string(), tool_calls);
    Ok(data)
}

fn required_dsrs_field<'a>(
    raw_output: &'a str,
    field_name: &str,
) -> std::result::Result<&'a str, String> {
    extract_dsrs_field(raw_output, field_name)
        .ok_or_else(|| format!("missing [[ ## {field_name} ## ]] field"))
}

fn extract_dsrs_field<'a>(raw_output: &'a str, field_name: &str) -> Option<&'a str> {
    let marker = format!("[[ ## {field_name} ## ]]");
    let (_, after_marker) = raw_output.split_once(&marker)?;
    let end = after_marker.find("[[ ## ").unwrap_or(after_marker.len());
    Some(after_marker[..end].trim())
}

fn parse_bool_field(value: &str) -> Option<bool> {
    serde_json::from_str::<bool>(value.trim())
        .ok()
        .or_else(|| value.trim().parse::<bool>().ok())
}

fn parse_number_field(value: &str) -> Option<f64> {
    serde_json::from_str::<f64>(value.trim())
        .ok()
        .or_else(|| value.trim().parse::<f64>().ok())
}

fn parse_json_lenient(input: &str) -> Option<Value> {
    let cleaned = strip_json_fence(input.trim());
    serde_json::from_str(cleaned)
        .ok()
        .or_else(|| json5::from_str(cleaned).ok())
        .or_else(|| serde_json::from_str(&remove_trailing_commas(cleaned)).ok())
}

fn strip_json_fence(input: &str) -> &str {
    input
        .strip_prefix("```json")
        .or_else(|| input.strip_prefix("```"))
        .and_then(|value| value.strip_suffix("```"))
        .unwrap_or(input)
        .trim()
}

fn remove_trailing_commas(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let chars = input.chars().collect::<Vec<_>>();
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

fn correction_output_from_prediction(
    prediction: &Prediction,
    raw_output: String,
) -> Result<CorrectionAgentOutput> {
    let possible = prediction
        .data
        .get("possible")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let confidence = prediction
        .data
        .get("confidence")
        .and_then(Value::as_f64)
        .unwrap_or(0.0)
        .clamp(0.0, 1.0) as f32;
    let explanation = prediction
        .data
        .get("explanation")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    let content = prediction
        .data
        .get("content")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|content| !content.is_empty())
        .map(str::to_string);
    let tool_calls_value = prediction
        .data
        .get("tool_calls")
        .cloned()
        .unwrap_or_else(|| Value::Array(Vec::new()));
    let signature_calls: Vec<CorrectionSignatureToolCall> =
        serde_json::from_value(tool_calls_value).map_err(|error| {
            CorrectionAgentError::invalid_dsrs_output(
                raw_output.clone(),
                format!("tool_calls field did not match the DSRs correction schema: {error}"),
            )
        })?;
    let tool_calls = signature_calls
        .into_iter()
        .filter_map(|call| call.into_tool_intent(confidence))
        .collect();

    Ok(CorrectionAgentOutput {
        tool_calls,
        content,
        possible,
        confidence,
        explanation,
        raw_output: Some(raw_output),
    })
}

pub fn recent_messages(messages: &[ChatMessage], max: usize) -> Vec<ChatMessage> {
    messages
        .iter()
        .skip(messages.len().saturating_sub(max))
        .cloned()
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use dspy_rs::{LmUsage, Message};

    fn parse_dsrs_correction_output(raw_output: &str) -> Result<CorrectionAgentOutput> {
        let signature = CorrectMalformedToolResponse::new();
        let adapter = ChatAdapter;
        let parsed = parse_correction_prediction_data(
            &adapter,
            &signature,
            Message::assistant(raw_output),
            raw_output,
        )?;
        let prediction = Prediction::new(parsed, LmUsage::default());
        correction_output_from_prediction(&prediction, raw_output.to_string())
    }

    #[test]
    fn parses_typed_dsrs_correction_tool_calls() {
        let parsed = parse_dsrs_correction_output(
            r#"[[ ## possible ## ]]
true

[[ ## confidence ## ]]
0.95

[[ ## explanation ## ]]
Recovered a clear read call.

[[ ## content ## ]]


[[ ## tool_calls ## ]]
[{"name":"read","arguments":{"path":"README.md"}}]

[[ ## completed ## ]]"#,
        )
        .unwrap();

        assert!(parsed.possible);
        assert_eq!(parsed.confidence, 0.95);
        assert_eq!(parsed.tool_calls[0].name, "read");
        assert_eq!(
            parsed.tool_calls[0].arguments.as_ref().unwrap()["path"],
            "README.md"
        );
        assert!(parsed.content.is_none());
    }

    #[test]
    fn parses_typed_dsrs_correction_content() {
        let parsed = parse_dsrs_correction_output(
            r#"[[ ## possible ## ]]
true

[[ ## confidence ## ]]
0.91

[[ ## explanation ## ]]
Recovered a text answer.

[[ ## content ## ]]
Hello.

[[ ## tool_calls ## ]]
[]

[[ ## completed ## ]]"#,
        )
        .unwrap();

        assert!(parsed.possible);
        assert_eq!(parsed.content.as_deref(), Some("Hello."));
        assert!(parsed.tool_calls.is_empty());
    }

    #[test]
    fn rejects_non_dsrs_correction_tool_call_shape() {
        let parsed = parse_dsrs_correction_output(
            r#"[[ ## possible ## ]]
true

[[ ## confidence ## ]]
0.91

[[ ## explanation ## ]]
This used the wrong tool-call schema.

[[ ## content ## ]]


[[ ## tool_calls ## ]]
[{"function":{"name":"read","arguments":{"path":"README.md"}}}]

[[ ## completed ## ]]"#,
        );

        assert!(parsed.is_err());
    }

    #[test]
    fn parses_typed_dsrs_correction_tool_calls_with_trailing_commas() {
        let parsed = parse_dsrs_correction_output(
            r#"[[ ## possible ## ]]
true

[[ ## confidence ## ]]
0.95

[[ ## explanation ## ]]
Recovered a clear write call.

[[ ## content ## ]]
Writing the file now.

[[ ## tool_calls ## ]]
[
  {
    "name": "write_file",
    "arguments": {
      "path": "redis_api_cache/middleware.py",
      "content": "hello",
    },
  },
]

[[ ## completed ## ]]"#,
        )
        .unwrap();

        assert!(parsed.possible);
        assert_eq!(parsed.content.as_deref(), Some("Writing the file now."));
        assert_eq!(parsed.tool_calls[0].name, "write_file");
        assert_eq!(
            parsed.tool_calls[0].arguments.as_ref().unwrap()["path"],
            "redis_api_cache/middleware.py"
        );
    }
}
