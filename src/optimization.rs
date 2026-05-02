use std::path::PathBuf;

use anyhow::{Context, Result};
use dspy_rs::{
    configure, example, ChatAdapter, Evaluator, Example, FeedbackEvaluator, FeedbackMetric,
    GEPAResult, Module, Optimizable, Predict, Prediction, Predictor, Signature, GEPA, LM,
};
use futures::FutureExt;
use indexmap::IndexMap;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

#[derive(Debug, Clone)]
pub struct GepaOptimizationConfig {
    pub dataset_path: PathBuf,
    pub output_path: PathBuf,
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub iterations: usize,
    pub max_examples: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaOptimizationReport {
    pub examples_loaded: usize,
    pub best_instruction: String,
    pub best_average_score: f32,
    pub total_rollouts: usize,
    pub total_lm_calls: usize,
    pub output_path: PathBuf,
}

#[Signature]
struct CorrectionPromptSignature {
    /// You are a strict DSRs model-response correction agent. Recover only clear
    /// tool-call intent or user-facing content by filling the requested DSRs output
    /// fields. Do not invent tools, arguments, or facts. If a safe repair is not
    /// possible, set possible to false.
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

    #[output(desc = "Plain user-facing content, or an empty string when using tools.")]
    pub content: String,

    #[output(
        desc = "JSON array of {\"name\": string, \"arguments\": object}; [] when no tool call is intended."
    )]
    pub tool_calls: Vec<CorrectionPromptToolCall>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct CorrectionPromptToolCall {
    name: String,
    #[serde(default)]
    arguments: Value,
}

pub struct CorrectionPromptProgram {
    predictor: Predict,
}

impl Default for CorrectionPromptProgram {
    fn default() -> Self {
        Self {
            predictor: Predict::new(CorrectionPromptSignature::new()),
        }
    }
}

impl Module for CorrectionPromptProgram {
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        match std::panic::AssertUnwindSafe(self.predictor.forward(inputs))
            .catch_unwind()
            .await
        {
            Ok(result) => result,
            Err(_) => Ok(Prediction::default()),
        }
    }
}

impl Optimizable for CorrectionPromptProgram {
    fn parameters(&mut self) -> IndexMap<String, &mut dyn Optimizable> {
        let mut parameters = IndexMap::new();
        parameters.insert(
            "correction_prompt".to_string(),
            &mut self.predictor as &mut dyn Optimizable,
        );
        parameters
    }
}

impl Evaluator for CorrectionPromptProgram {
    async fn metric(&self, example: &Example, prediction: &Prediction) -> f32 {
        self.feedback_metric(example, prediction).await.score
    }
}

impl FeedbackEvaluator for CorrectionPromptProgram {
    async fn feedback_metric(&self, example: &Example, prediction: &Prediction) -> FeedbackMetric {
        let expected = example.get("expected_repair", None);
        let predicted = prediction_to_repair_value(prediction);
        score_correction_prediction(&expected, &predicted)
    }
}

pub async fn optimize_correction_prompt(
    config: GepaOptimizationConfig,
) -> Result<GepaOptimizationReport> {
    let Some(api_key) = config.api_key.clone() else {
        anyhow::bail!(
            "OPENROUTER_API_KEY or --api-key equivalent is required for GEPA optimization"
        );
    };

    let rows = read_dataset_rows(&config.dataset_path).await?;
    let rows: Vec<Value> = rows.into_iter().take(config.max_examples).collect();
    let examples = traces_to_gepa_examples(&rows);
    if examples.is_empty() {
        anyhow::bail!("dataset contained no optimization examples");
    }

    let lm = LM::builder()
        .base_url(config.base_url.clone())
        .api_key(api_key)
        .model(config.model.clone())
        .temperature(0.2)
        .build()
        .await
        .context("failed to build GEPA LM")?;
    configure(lm.clone(), ChatAdapter);

    let gepa = GEPA::builder()
        .num_iterations(config.iterations)
        .minibatch_size(examples.len().clamp(1, 3))
        .temperature(0.7)
        .track_stats(true)
        .maybe_prompt_model(Some(lm))
        .maybe_max_lm_calls(Some((config.iterations.max(1) * 16) + 16))
        .build();

    let mut program = CorrectionPromptProgram::default();
    let result: GEPAResult = gepa
        .compile_with_feedback(&mut program, examples.clone())
        .await
        .context("GEPA correction prompt optimization failed")?;

    if let Some(parent) = config.output_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let report = GepaOptimizationReport {
        examples_loaded: examples.len(),
        best_instruction: result.best_candidate.instruction.clone(),
        best_average_score: result.best_candidate.average_score(),
        total_rollouts: result.total_rollouts,
        total_lm_calls: result.total_lm_calls,
        output_path: config.output_path.clone(),
    };
    tokio::fs::write(&config.output_path, serde_json::to_vec_pretty(&report)?)
        .await
        .with_context(|| format!("failed to write {}", config.output_path.display()))?;
    Ok(report)
}

pub fn traces_to_gepa_examples(dataset_rows: &[Value]) -> Vec<Example> {
    dataset_rows
        .iter()
        .map(|row| {
            let expected = row.get("expected_repair").unwrap_or(&Value::Null);
            let expected_fields = expected_repair_fields(expected);
            example! {
                "available_tools": "input" => stringify_field(row.get("available_tools")),
                "recent_messages": "input" => stringify_field(row.get("recent_messages")),
                "malformed_response": "input" => stringify_field(row.get("malformed_response")),
                "parser_events": "input" => stringify_field(row.get("parser_events")),
                "response_failures": "input" => stringify_field(row.get("response_failures")),
                "expected_repair": "output" => expected.clone(),
                "possible": "output" => expected_fields.possible,
                "confidence": "output" => expected_fields.confidence,
                "explanation": "output" => expected_fields.explanation,
                "content": "output" => expected_fields.content,
                "tool_calls": "output" => expected_fields.tool_calls
            }
        })
        .collect()
}

pub fn correction_gepa_config(iterations: usize) -> GEPA {
    GEPA::builder()
        .num_iterations(iterations)
        .minibatch_size(3)
        .temperature(0.7)
        .track_stats(true)
        .maybe_max_lm_calls(Some(64))
        .build()
}

async fn read_dataset_rows(path: &PathBuf) -> Result<Vec<Value>> {
    let content = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("failed to read {}", path.display()))?;
    content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            serde_json::from_str::<Value>(line).context("failed to parse dataset JSONL row")
        })
        .collect()
}

fn stringify_field(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(text)) => text.clone(),
        Some(value) => serde_json::to_string(value).unwrap_or_else(|_| "null".to_string()),
        None => "null".to_string(),
    }
}

#[derive(Debug, Clone)]
struct ExpectedRepairFields {
    possible: bool,
    confidence: f32,
    explanation: String,
    content: String,
    tool_calls: Value,
}

fn expected_repair_fields(expected: &Value) -> ExpectedRepairFields {
    let expected_calls = expected
        .get("tool_calls")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let normalized_calls = normalize_tool_calls(&expected_calls);
    let content = expected
        .get("content")
        .and_then(Value::as_str)
        .or_else(|| {
            expected
                .pointer("/final_message/content")
                .and_then(Value::as_str)
        })
        .unwrap_or_default()
        .to_string();
    ExpectedRepairFields {
        possible: expected
            .get("possible")
            .and_then(Value::as_bool)
            .unwrap_or(true),
        confidence: expected
            .get("confidence")
            .and_then(Value::as_f64)
            .unwrap_or(1.0)
            .clamp(0.0, 1.0) as f32,
        explanation: expected
            .get("explanation")
            .and_then(Value::as_str)
            .unwrap_or("Matched expected repair")
            .to_string(),
        content,
        tool_calls: Value::Array(normalized_calls),
    }
}

fn prediction_to_repair_value(prediction: &Prediction) -> Value {
    let tool_calls = match prediction.data.get("tool_calls") {
        Some(Value::Array(calls)) => Value::Array(normalize_tool_calls(calls)),
        Some(Value::String(text)) => serde_json::from_str::<Value>(text)
            .ok()
            .and_then(|value| value.as_array().cloned())
            .map(|calls| Value::Array(normalize_tool_calls(&calls)))
            .unwrap_or_else(|| Value::Array(Vec::new())),
        _ => Value::Array(Vec::new()),
    };

    json!({
        "possible": prediction.data.get("possible").and_then(Value::as_bool).unwrap_or(false),
        "confidence": prediction.data.get("confidence").and_then(Value::as_f64).unwrap_or(0.0),
        "explanation": prediction.data.get("explanation").and_then(Value::as_str).unwrap_or_default(),
        "content": prediction.data.get("content").and_then(Value::as_str).unwrap_or_default(),
        "tool_calls": tool_calls
    })
}

fn score_correction_prediction(expected: &Value, predicted: &Value) -> FeedbackMetric {
    let expected_calls = expected
        .get("tool_calls")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let predicted_calls = predicted
        .get("tool_calls")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    if expected_calls.is_empty() && predicted_calls.is_empty() {
        let expected_content = expected
            .get("content")
            .and_then(Value::as_str)
            .or_else(|| {
                expected
                    .pointer("/final_message/content")
                    .and_then(Value::as_str)
            })
            .unwrap_or_default()
            .trim();
        let predicted_content = predicted
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .trim();
        if expected_content.is_empty() || expected_content == predicted_content {
            return FeedbackMetric::new(1.0, "No tool calls expected or predicted");
        }
        return FeedbackMetric::new(
            0.45,
            format!(
                "Predicted no tool calls, but content differed. Expected {expected_content:?}; predicted {predicted_content:?}"
            ),
        );
    }

    let expected_norm = normalize_tool_calls(&expected_calls);
    let predicted_norm = normalize_tool_calls(&predicted_calls);
    if expected_norm == predicted_norm {
        FeedbackMetric::new(1.0, "Predicted tool calls exactly matched expected repair")
    } else if !predicted_norm.is_empty() {
        FeedbackMetric::new(
            0.45,
            format!(
                "Predicted DSRs tool calls, but mismatch. Expected {}; predicted {}",
                json!(expected_norm),
                json!(predicted_norm)
            ),
        )
    } else {
        FeedbackMetric::new(
            0.2,
            format!(
                "Predicted DSRs fields but no matching tool calls. Expected {}",
                json!(expected_norm)
            ),
        )
    }
}

fn normalize_tool_calls(calls: &[Value]) -> Vec<Value> {
    calls
        .iter()
        .map(|call| {
            if let Some(function) = call.get("function") {
                json!({
                    "name": function.get("name").cloned().unwrap_or(Value::Null),
                    "arguments": parse_arguments(function.get("arguments").cloned().unwrap_or(Value::Null))
                })
            } else {
                json!({
                    "name": call.get("name").cloned().unwrap_or(Value::Null),
                    "arguments": call.get("arguments").cloned().unwrap_or(Value::Null)
                })
            }
        })
        .collect()
}

fn parse_arguments(value: Value) -> Value {
    match value {
        Value::String(text) => serde_json::from_str(&text).unwrap_or(Value::String(text)),
        other => other,
    }
}

pub fn dsrs_is_linked() -> &'static str {
    std::any::type_name::<GEPA>()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_rows_to_gepa_examples() {
        let rows = vec![json!({
            "available_tools": [{"function":{"name":"read_file"}}],
            "recent_messages": [{"role":"user","content":"read"}],
            "malformed_response": "<tool_call name=\"read_file\">{}</tool_call>",
            "parser_events": ["parsed"],
            "expected_repair": {"tool_calls":[{"function":{"name":"read_file","arguments":"{}"}}]}
        })];

        let examples = traces_to_gepa_examples(&rows);
        assert_eq!(examples.len(), 1);
        assert!(examples[0].get("available_tools", None).as_str().is_some());
        assert!(examples[0].get("possible", None).as_bool().unwrap());
        assert_eq!(
            examples[0].get("tool_calls", None),
            json!([{"name":"read_file","arguments":{}}])
        );
        assert!(dsrs_is_linked().contains("GEPA"));
    }

    #[test]
    fn scores_matching_tool_call_predictions() {
        let expected = json!({
            "tool_calls":[{"function":{"name":"read_file","arguments":"{\"path\":\"x\"}"}}]
        });
        let predicted = json!({
            "tool_calls":[{"name":"read_file","arguments":{"path":"x"}}]
        });
        let feedback = score_correction_prediction(&expected, &predicted);
        assert_eq!(feedback.score, 1.0);
    }
}
