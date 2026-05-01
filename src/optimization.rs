use std::path::PathBuf;

use anyhow::{Context, Result};
use dspy_rs::{
    configure, example, ChatAdapter, Evaluator, Example, FeedbackEvaluator, FeedbackMetric,
    GEPAResult, Module, Optimizable, Predict, Prediction, Predictor, Signature, GEPA, LM,
};
use indexmap::IndexMap;
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
    /// You are a strict model-response correction agent. Recover only clear tool-call
    /// intent and return JSON only.

    #[input(desc = "OpenAI-compatible tool definitions")]
    pub available_tools: String,

    #[input(desc = "Recent conversation context")]
    pub recent_messages: String,

    #[input(desc = "Malformed assistant response to correct")]
    pub malformed_response: String,

    #[input(desc = "Parser diagnostics from deterministic parsing")]
    pub parser_events: String,

    #[output(desc = "JSON with possible, confidence, explanation, content, and tool_calls")]
    pub corrected_json: String,
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
        self.predictor.forward(inputs).await
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
        let predicted = prediction
            .get("corrected_json", None)
            .as_str()
            .unwrap_or_default()
            .to_string();
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
            example! {
                "available_tools": "input" => stringify_field(row.get("available_tools")),
                "recent_messages": "input" => stringify_field(row.get("recent_messages")),
                "malformed_response": "input" => stringify_field(row.get("malformed_response")),
                "parser_events": "input" => stringify_field(row.get("parser_events")),
                "expected_repair": "output" => row.get("expected_repair").cloned().unwrap_or(Value::Null)
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

fn score_correction_prediction(expected: &Value, predicted: &str) -> FeedbackMetric {
    let parsed = serde_json::from_str::<Value>(predicted).or_else(|_| json5::from_str(predicted));
    let Ok(parsed) = parsed else {
        return FeedbackMetric::new(
            0.0,
            format!("Correction output was not parseable JSON: {predicted}"),
        );
    };

    let expected_calls = expected
        .get("tool_calls")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let predicted_calls = parsed
        .get("tool_calls")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    if expected_calls.is_empty() && predicted_calls.is_empty() {
        return FeedbackMetric::new(1.0, "No tool calls expected or predicted");
    }

    let expected_norm = normalize_tool_calls(&expected_calls);
    let predicted_norm = normalize_tool_calls(&predicted_calls);
    if expected_norm == predicted_norm {
        FeedbackMetric::new(1.0, "Predicted tool calls exactly matched expected repair")
    } else if !predicted_norm.is_empty() {
        FeedbackMetric::new(
            0.45,
            format!(
                "Predicted valid JSON and tool calls, but mismatch. Expected {}; predicted {}",
                json!(expected_norm),
                json!(predicted_norm)
            ),
        )
    } else {
        FeedbackMetric::new(
            0.2,
            format!(
                "Predicted valid JSON but no matching tool calls. Expected {}",
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
        assert!(dsrs_is_linked().contains("GEPA"));
    }

    #[test]
    fn scores_matching_tool_call_predictions() {
        let expected = json!({
            "tool_calls":[{"function":{"name":"read_file","arguments":"{\"path\":\"x\"}"}}]
        });
        let feedback = score_correction_prediction(
            &expected,
            r#"{"tool_calls":[{"name":"read_file","arguments":{"path":"x"}}]}"#,
        );
        assert_eq!(feedback.score, 1.0);
    }
}
