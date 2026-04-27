use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use dspy_rs::{example, Predict, Predictor, Signature, LM};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    config::{CorrectionConfig, UpstreamConfig},
    model_profile::ModelProfile,
    openai::{ChatMessage, OpenAiTool},
    response_interpreter::{ToolIntent, ToolIntentSource},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionAgentInput {
    pub tools: Vec<OpenAiTool>,
    pub recent_messages: Vec<ChatMessage>,
    pub malformed_response: String,
    pub parser_events: Vec<String>,
    pub model: String,
    pub profile: ModelProfile,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionAgentOutput {
    pub tool_calls: Vec<ToolIntent>,
    pub content: Option<String>,
    pub confidence: f32,
    pub explanation: String,
}

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
    /// You are a strict model-response correction agent. Recover only clear tool-call
    /// intent from the malformed response. Return valid JSON only.

    #[input(desc = "OpenAI-compatible tool definitions")]
    pub available_tools: String,

    #[input(desc = "Recent conversation context")]
    pub recent_messages: String,

    #[input(desc = "Malformed assistant response to correct")]
    pub malformed_response: String,

    #[input(desc = "Parser diagnostics from deterministic parsing")]
    pub parser_events: String,

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
    content: Option<String>,
    #[serde(default)]
    tool_calls: Vec<CorrectionEnvelopeToolCall>,
}

#[derive(Debug, Deserialize)]
struct CorrectionEnvelopeToolCall {
    name: String,
    #[serde(default)]
    arguments: Value,
}

#[async_trait]
impl CorrectionAgent for DsrsCorrectionAgent {
    async fn correct(&self, input: CorrectionAgentInput) -> Result<Option<CorrectionAgentOutput>> {
        let Some(api_key) = self.upstream.api_key.clone() else {
            return Ok(None);
        };

        let model = self.model_for(&input);
        let lm = LM::builder()
            .base_url(self.upstream.base_url.clone())
            .api_key(api_key)
            .model(model)
            .temperature(0.1)
            .max_tokens(700)
            .build()
            .await
            .context("failed to build DSRs correction LM")?;

        let predictor = Predict::new(CorrectMalformedToolResponse::new());
        let available_tools = serde_json::to_string_pretty(&input.tools)?;
        let recent_messages = serde_json::to_string_pretty(&input.recent_messages)?;
        let parser_events = input.parser_events.join("\n");

        let prediction = predictor
            .forward_with_config(
                example! {
                    "available_tools": "input" => available_tools,
                    "recent_messages": "input" => recent_messages,
                    "malformed_response": "input" => input.malformed_response,
                    "parser_events": "input" => parser_events
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
            .with_context(|| format!("correction agent returned non-JSON output: {corrected}"))?;

        if !envelope.possible || envelope.confidence < self.correction.min_confidence {
            return Ok(None);
        }

        let tool_calls = envelope
            .tool_calls
            .into_iter()
            .map(|call| {
                let raw_arguments =
                    serde_json::to_string(&call.arguments).unwrap_or_else(|_| "{}".to_string());
                ToolIntent {
                    name: call.name,
                    arguments: Some(call.arguments),
                    raw_arguments,
                    source: ToolIntentSource::CorrectionAgent,
                    confidence: envelope.confidence,
                }
            })
            .collect();

        Ok(Some(CorrectionAgentOutput {
            tool_calls,
            content: envelope.content,
            confidence: envelope.confidence,
            explanation: envelope.explanation,
        }))
    }
}

fn parse_correction_envelope(output: &str) -> Result<CorrectionEnvelope> {
    let trimmed = output.trim();
    let unwrapped = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .and_then(|s| s.strip_suffix("```"))
        .unwrap_or(trimmed)
        .trim();
    serde_json::from_str(unwrapped)
        .or_else(|_| json5::from_str(unwrapped))
        .context("failed to parse correction envelope")
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
        assert_eq!(parsed.tool_calls[0].name, "read_file");
    }
}
