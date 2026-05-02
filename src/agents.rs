use std::{fmt, sync::Arc};

use anyhow::{Context, Result};
use async_trait::async_trait;
use dspy_rs::{example, Predict, Predictor, Signature, LM};
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
    fn non_json(raw_output: String) -> Self {
        Self {
            message: format!("correction agent returned non-JSON output: {raw_output}"),
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
    /// You are a strict model-response correction agent. Recover a malformed
    /// assistant response into the OpenAI-compatible result the client should have
    /// received. Return valid JSON only, with no markdown and no prose outside the
    /// JSON object. If clear tool calls were intended, put them in tool_calls and
    /// leave content null. If no tool call was intended, put the user-facing text in
    /// content and leave tool_calls empty. If the malformed response is ambiguous or
    /// unsafe to repair, set possible to false.

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

    #[output(
        desc = "A JSON object with possible, confidence, explanation, content, and tool_calls"
    )]
    pub corrected_json: String,
}

#[derive(Debug, Deserialize)]
struct CorrectionEnvelope {
    #[serde(default)]
    possible: bool,
    #[serde(default)]
    confidence: f32,
    #[serde(default)]
    explanation: String,
    #[serde(default)]
    content: Option<Value>,
    #[serde(default)]
    tool_calls: Vec<CorrectionEnvelopeToolCall>,
}

#[derive(Debug, Deserialize)]
struct CorrectionEnvelopeToolCall {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Value,
    #[serde(default)]
    function: Option<CorrectionEnvelopeFunctionCall>,
}

#[derive(Debug, Deserialize)]
struct CorrectionEnvelopeFunctionCall {
    name: String,
    #[serde(default)]
    arguments: Value,
}

impl CorrectionEnvelopeToolCall {
    fn into_tool_intent(self, confidence: f32) -> Option<ToolIntent> {
        let (name, arguments) = if let Some(function) = self.function {
            (function.name, normalize_arguments(function.arguments))
        } else {
            (self.name?, normalize_arguments(self.arguments))
        };
        let raw_arguments = match &arguments {
            Value::String(raw) => raw.clone(),
            other => serde_json::to_string(other).unwrap_or_else(|_| "{}".to_string()),
        };

        Some(ToolIntent {
            name,
            arguments: Some(arguments),
            raw_arguments,
            source: ToolIntentSource::CorrectionAgent,
            confidence,
        })
    }
}

impl CorrectionEnvelope {
    fn content_text(&self) -> Option<String> {
        self.content
            .as_ref()
            .and_then(Value::as_str)
            .map(str::to_owned)
    }

    fn into_tool_intents(self) -> Vec<ToolIntent> {
        let confidence = self.confidence;
        let mut calls = self.tool_calls;

        if calls.is_empty() {
            if let Some(Value::Object(mut object)) = self.content {
                if let Some(tool_calls) = object.remove("tool_calls") {
                    if let Ok(parsed) =
                        serde_json::from_value::<Vec<CorrectionEnvelopeToolCall>>(tool_calls)
                    {
                        calls = parsed;
                    }
                }
            }
        }

        calls
            .into_iter()
            .filter_map(|call| call.into_tool_intent(confidence))
            .collect()
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

        let predictor = Predict::new(CorrectMalformedToolResponse::new());
        let available_tools = serde_json::to_string_pretty(&input.tools)?;
        let recent_messages = serde_json::to_string_pretty(&input.recent_messages)?;
        let parser_events = input.parser_events.join("\n");
        let response_failures = serde_json::to_string_pretty(&input.response_failures)?;

        let prediction = predictor
            .forward_with_config(
                example! {
                    "available_tools": "input" => available_tools,
                    "recent_messages": "input" => recent_messages,
                    "malformed_response": "input" => input.malformed_response,
                    "parser_events": "input" => parser_events,
                    "response_failures": "input" => response_failures
                },
                Arc::new(lm),
            )
            .await
            .context("DSRs correction predictor failed")?;

        let corrected = prediction
            .get("corrected_json", None)
            .as_str()
            .unwrap_or_default()
            .trim()
            .to_string();

        let envelope = parse_correction_envelope(&corrected)
            .map_err(|_| CorrectionAgentError::non_json(corrected.clone()))?;

        let possible = envelope.possible;
        let confidence = envelope.confidence;
        let explanation = envelope.explanation.clone();
        let content = envelope.content_text();
        let tool_calls = envelope.into_tool_intents();

        Ok(Some(CorrectionAgentOutput {
            tool_calls,
            content,
            possible,
            confidence,
            explanation,
            raw_output: Some(corrected),
        }))
    }
}

fn normalize_arguments(arguments: Value) -> Value {
    match arguments {
        Value::String(raw) => serde_json::from_str(&raw)
            .or_else(|_| json5::from_str(&raw))
            .unwrap_or(Value::String(raw)),
        other => other,
    }
}

fn parse_correction_envelope(output: &str) -> Result<CorrectionEnvelope> {
    let trimmed = output.trim();
    let fenced = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .and_then(|s| s.strip_suffix("```"))
        .unwrap_or(trimmed)
        .trim();
    let dsrs_field = extract_dsrs_corrected_json(fenced);
    let json_object = extract_json_object(fenced);

    for candidate in [Some(fenced), dsrs_field, json_object.as_deref()]
        .into_iter()
        .flatten()
    {
        if let Ok(envelope) =
            serde_json::from_str(candidate).or_else(|_| json5::from_str(candidate))
        {
            return Ok(envelope);
        }
    }

    anyhow::bail!("failed to parse correction envelope")
}

fn extract_dsrs_corrected_json(output: &str) -> Option<&str> {
    let (_, after_marker) = output.split_once("[[ ## corrected_json ## ]]")?;
    Some(
        after_marker
            .split("[[ ## completed ## ]]")
            .next()
            .unwrap_or(after_marker)
            .trim(),
    )
}

fn extract_json_object(output: &str) -> Option<String> {
    let mut start = None;
    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (index, ch) in output.char_indices() {
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
            '{' => {
                if start.is_none() {
                    start = Some(index);
                }
                depth += 1;
            }
            '}' if depth > 0 => {
                depth -= 1;
                if depth == 0 {
                    let start = start?;
                    return Some(output[start..=index].to_string());
                }
            }
            _ => {}
        }
    }
    None
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

    #[test]
    fn parses_fenced_correction_envelope() {
        let parsed = parse_correction_envelope(
            r#"```json
{"possible":true,"confidence":0.9,"tool_calls":[{"name":"read_file","arguments":{"path":"x"}}]}
```"#,
        )
        .unwrap();

        assert!(parsed.possible);
        assert_eq!(parsed.tool_calls[0].name.as_deref(), Some("read_file"));
    }

    #[test]
    fn parses_openai_style_correction_tool_call() {
        let parsed = parse_correction_envelope(
            r#"{
  "possible": true,
  "confidence": 0.95,
  "tool_calls": [
    {
      "id": "call_1",
      "type": "function",
      "function": {
        "name": "bash",
        "arguments": {"command": "find packages -maxdepth 1 -type d"}
      }
    }
  ]
}"#,
        )
        .unwrap();
        let intent = parsed
            .tool_calls
            .into_iter()
            .next()
            .unwrap()
            .into_tool_intent(parsed.confidence)
            .unwrap();

        assert_eq!(intent.name, "bash");
        assert_eq!(
            intent.arguments.as_ref().unwrap()["command"],
            "find packages -maxdepth 1 -type d"
        );
    }

    #[test]
    fn parses_tool_calls_nested_in_content_object() {
        let parsed = parse_correction_envelope(
            r#"{
  "possible": true,
  "confidence": 0.95,
  "content": {
    "tool_calls": [
      {
        "name": "read",
        "arguments": {"path": "Cargo.toml"}
      }
    ]
  }
}"#,
        )
        .unwrap();

        let intents = parsed.into_tool_intents();

        assert_eq!(intents.len(), 1);
        assert_eq!(intents[0].name, "read");
        assert_eq!(intents[0].arguments.as_ref().unwrap()["path"], "Cargo.toml");
    }

    #[test]
    fn parses_correction_envelope_from_dsrs_field() {
        let parsed = parse_correction_envelope(
            r#"[[ ## corrected_json ## ]]
{"possible":true,"confidence":0.91,"content":"Hello.","tool_calls":[]}
[[ ## completed ## ]]"#,
        )
        .unwrap();

        assert!(parsed.possible);
        assert_eq!(parsed.content_text().as_deref(), Some("Hello."));
    }

    #[test]
    fn parses_correction_envelope_embedded_in_prose() {
        let parsed = parse_correction_envelope(
            r#"Here is the corrected response:
{"possible":true,"confidence":0.9,"tool_calls":[{"name":"read","arguments":{"path":"README.md"}}]}
Done."#,
        )
        .unwrap();

        assert!(parsed.possible);
        assert_eq!(parsed.tool_calls[0].name.as_deref(), Some("read"));
    }
}
