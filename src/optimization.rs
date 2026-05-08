use std::{collections::HashMap, path::PathBuf, sync::Arc};

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use dspy_rs::{
    adapter::Adapter, configure, example, Chat, ChatAdapter, Evaluator, Example, FeedbackEvaluator,
    FeedbackMetric, GEPAResult, LmUsage, Message, MetaSignature, Module, Optimizable, Predict,
    Prediction, Predictor, Signature, GEPA, LM,
};
use futures::FutureExt;
use indexmap::IndexMap;
use rig::tool::ToolDyn;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    artifacts::extract_instruction_from_path,
    dsrs_contract::{format_tool_contract, parse_tool_contract_response},
    model_profile::{builtin_profiles, resolve_profile, DsrsHistoryFormat, ModelProfile},
    normalizer::normalize_request,
    openai::{ChatCompletionRequest, ChatMessage, OpenAiFunctionTool, OpenAiTool},
};

pub const DEFAULT_GEPA_LM_MAX_TOKENS: u32 = 100_000;
pub const DEFAULT_GEPA_REFLECTION_MODEL: &str = "anthropic/claude-sonnet-4.6";
pub const DEFAULT_GEPA_JUDGE_MODEL: &str = "anthropic/claude-sonnet-4.6";

fn default_gepa_lm_max_tokens() -> u32 {
    DEFAULT_GEPA_LM_MAX_TOKENS
}

fn default_gepa_judge_model_string() -> String {
    DEFAULT_GEPA_JUDGE_MODEL.to_string()
}

#[derive(Debug, Default, Clone)]
struct GepaJsonAdapter;

#[async_trait]
impl Adapter for GepaJsonAdapter {
    fn format(&self, signature: &dyn MetaSignature, inputs: Example) -> Chat {
        let input_fields = signature.input_fields();
        let output_fields = signature.output_fields();
        let output_keys = field_names(&output_fields);
        let input_values = field_values(&input_fields, &inputs);
        let instruction = if signature.instruction().trim().is_empty() {
            format!(
                "Given the input fields {}, produce the output fields {}.",
                field_names(&input_fields)
                    .into_iter()
                    .map(|name| format!("`{name}`"))
                    .collect::<Vec<_>>()
                    .join(", "),
                output_keys
                    .iter()
                    .map(|name| format!("`{name}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        } else {
            signature.instruction()
        };

        let system = format!(
            "You are a strict JSON adapter for GEPA prompt optimization.\n\
Return only one valid JSON object with exactly the requested output keys.\n\
Do not wrap the response in Markdown fences. Do not use DSRs field markers as the outer response format.\n\
Literal DSRs marker text such as [[ ## content ## ]] may appear inside JSON string values; preserve it as normal string content.\n\n\
Input fields:\n{}\n\
Output fields:\n{}\n\
Objective:\n{}",
            describe_fields(&input_fields),
            describe_fields(&output_fields),
            instruction
        );
        let user = format!(
            "Input values:\n{}\n\nReturn a JSON object with these output keys, in this order: {}",
            serde_json::to_string_pretty(&input_values).unwrap_or_else(|_| "{}".to_string()),
            output_keys
                .iter()
                .map(|name| format!("\"{name}\""))
                .collect::<Vec<_>>()
                .join(", ")
        );
        Chat::new(vec![Message::system(system), Message::user(user)])
    }

    fn parse_response(
        &self,
        signature: &dyn MetaSignature,
        response: Message,
    ) -> HashMap<String, Value> {
        let raw = response.content();
        let Some(object) = extract_json_object(&raw).and_then(|value| value.as_object().cloned())
        else {
            return HashMap::new();
        };

        let output_fields = signature.output_fields();
        field_names(&output_fields)
            .into_iter()
            .filter_map(|field_name| {
                object.get(&field_name).map(|value| {
                    let field = output_fields
                        .get(&field_name)
                        .cloned()
                        .unwrap_or(Value::Null);
                    (field_name, coerce_json_adapter_value(value.clone(), &field))
                })
            })
            .collect()
    }

    async fn call(
        &self,
        lm: Arc<LM>,
        signature: &dyn MetaSignature,
        inputs: Example,
        tools: Vec<Arc<dyn ToolDyn>>,
    ) -> Result<Prediction> {
        let messages = self.format(signature, inputs);
        let response = match lm.call(messages, tools).await {
            Ok(response) => response,
            Err(_) => return Ok(Prediction::new(HashMap::new(), LmUsage::default())),
        };
        Ok(Prediction::new(
            self.parse_response(signature, response.output),
            response.usage,
        ))
    }
}

fn field_names(fields: &Value) -> Vec<String> {
    fields
        .as_object()
        .map(|object| object.keys().cloned().collect())
        .unwrap_or_default()
}

fn describe_fields(fields: &Value) -> String {
    fields
        .as_object()
        .map(|object| {
            object
                .iter()
                .enumerate()
                .map(|(index, (name, field))| {
                    let ty = field.get("type").and_then(Value::as_str).unwrap_or("Value");
                    let desc = field.get("desc").and_then(Value::as_str).unwrap_or("");
                    if desc.is_empty() {
                        format!("{}. `{name}` ({ty})", index + 1)
                    } else {
                        format!("{}. `{name}` ({ty}): {desc}", index + 1)
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default()
}

fn field_values(fields: &Value, inputs: &Example) -> Value {
    let object = field_names(fields)
        .into_iter()
        .map(|name| (name.clone(), inputs.get(&name, None)))
        .collect::<serde_json::Map<_, _>>();
    Value::Object(object)
}

fn coerce_json_adapter_value(value: Value, field: &Value) -> Value {
    let data_type = field
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("String");
    match data_type {
        "String" => match value {
            Value::String(_) => value,
            other => Value::String(other.to_string()),
        },
        "bool" => match value {
            Value::Bool(_) => value,
            Value::String(text) => text
                .parse::<bool>()
                .map(Value::Bool)
                .unwrap_or(Value::String(text)),
            other => other,
        },
        "f32" | "f64" => match value {
            Value::Number(_) => value,
            Value::String(text) => text
                .parse::<f64>()
                .ok()
                .and_then(serde_json::Number::from_f64)
                .map(Value::Number)
                .unwrap_or(Value::String(text)),
            other => other,
        },
        _ => match value {
            Value::String(text) => serde_json::from_str(&text).unwrap_or(Value::String(text)),
            other => other,
        },
    }
}

fn extract_json_object(raw: &str) -> Option<Value> {
    let trimmed = strip_json_fence(raw.trim());
    serde_json::from_str::<Value>(trimmed)
        .ok()
        .or_else(|| balanced_json_object(trimmed).and_then(|json| serde_json::from_str(json).ok()))
}

fn strip_json_fence(raw: &str) -> &str {
    let Some(stripped) = raw.strip_prefix("```") else {
        return raw;
    };
    let stripped = stripped
        .strip_prefix("json")
        .unwrap_or(stripped)
        .trim_start();
    stripped
        .strip_suffix("```")
        .map(str::trim_end)
        .unwrap_or(stripped)
}

fn balanced_json_object(raw: &str) -> Option<&str> {
    let start = raw.find('{')?;
    let mut in_string = false;
    let mut escaped = false;
    let mut depth = 0usize;
    for (offset, ch) in raw[start..].char_indices() {
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
            '{' => depth += 1,
            '}' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    let end = start + offset + ch.len_utf8();
                    return raw.get(start..end);
                }
            }
            _ => {}
        }
    }
    None
}

#[derive(Debug, Clone)]
pub struct GepaOptimizationConfig {
    pub dataset_path: PathBuf,
    pub output_path: PathBuf,
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub judge_model: String,
    pub target_model: Option<String>,
    pub profile: Option<String>,
    pub profile_revision: Option<u32>,
    pub dsrs_history_format: Option<DsrsHistoryFormat>,
    pub artifact_id: Option<String>,
    pub seed_artifact_path: Option<PathBuf>,
    pub iterations: usize,
    pub max_examples: usize,
    pub lm_max_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactWarning {
    pub code: String,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matched: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GepaOptimizationReport {
    pub artifact_id: String,
    pub artifact_type: String,
    pub signature: String,
    pub target_model: Option<String>,
    pub profile: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile_revision: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dsrs_history_format: Option<DsrsHistoryFormat>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed_artifact_path: Option<PathBuf>,
    pub optimizer_model: String,
    #[serde(default = "default_gepa_judge_model_string")]
    pub judge_model: String,
    pub created_at: DateTime<Utc>,
    pub examples_loaded: usize,
    #[serde(default = "default_gepa_lm_max_tokens")]
    pub lm_max_tokens: u32,
    pub best_instruction: String,
    pub best_average_score: f32,
    pub total_rollouts: usize,
    pub total_lm_calls: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifact_warnings: Vec<ArtifactWarning>,
    pub output_path: PathBuf,
}

pub fn artifact_instruction_warnings(
    artifact_type: &str,
    instruction: &str,
) -> Vec<ArtifactWarning> {
    let mut warnings = Vec::new();
    let lower = instruction.to_ascii_lowercase();

    if let Some(matched) = first_matched_phrase(
        &lower,
        &[
            "expected_output",
            "expected output",
            "expected repair",
            "answer key",
            "gold label",
            "labels",
            "labelled",
            "labeled",
            "eval",
            "evaluation",
            "benchmark",
            "test harness",
            "dataset",
            "rollout",
            "score",
            "scoring",
        ],
    ) {
        warnings.push(ArtifactWarning {
            code: "optimizer_meta_language".to_string(),
            message: format!(
                "{artifact_type} instruction appears to mention optimizer/eval metadata; review for benchmark overfit before promotion"
            ),
            matched: Some(matched.to_string()),
        });
    }

    if let Some(matched) = first_matched_phrase(
        &lower,
        &[
            "character-for-character",
            "character for character",
            "exact template",
            "match it exactly",
            "match expected",
            "expected values",
            "expected command",
            "use it as the exact",
        ],
    ) {
        warnings.push(ArtifactWarning {
            code: "exact_label_matching_language".to_string(),
            message: format!(
                "{artifact_type} instruction appears to ask the runtime model to match labels or examples too literally"
            ),
            matched: Some(matched.to_string()),
        });
    }

    if instruction.len() > 2_400 {
        warnings.push(ArtifactWarning {
            code: "long_instruction".to_string(),
            message: format!(
                "{artifact_type} instruction is long enough to merit extra review for prompt drift"
            ),
            matched: Some(format!("{} chars", instruction.len())),
        });
    }

    warnings
}

fn first_matched_phrase<'a>(haystack_lower: &str, phrases: &'a [&str]) -> Option<&'a str> {
    phrases
        .iter()
        .copied()
        .find(|phrase| haystack_lower.contains(phrase))
}

#[derive(Debug, Clone)]
struct GepaJudgeCase {
    expected: Value,
    policy: Value,
    observed_problem: Option<String>,
    observed_failure_kind: Option<String>,
}

type GepaJudgeCases = Arc<HashMap<String, GepaJudgeCase>>;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GepaJudgeDecision {
    score: f32,
    feedback: String,
}

#[Signature]
struct RequestAdapterPromptSignature {
    /// Tool calls are an application contract. You are adapting OpenAI-compatible
    /// tool use into DSRs output fields for a coding-agent model. This instruction
    /// becomes profile_guidance inside the real runtime DSRs formatter during
    /// optimization. Obey system_context without copying it, honor the selected
    /// dsrs_history_format, and produce exactly one completed DSRs response. If
    /// the user asks about a project, repository, package, docs, code, or
    /// filesystem state and the answer is not already present in the conversation,
    /// call an available inspection tool instead of guessing. A completed response
    /// must never be blank or placeholder content. Empty content with [] tool_calls
    /// is invalid. Literal [], {}, null, content_value, tool_calls_value, or
    /// completed_marker in content are invalid answers. When using tools, put valid
    /// calls in tool_calls and optionally include brief user-facing content. When
    /// no tool is needed, put a real user-facing answer in content and set
    /// tool_calls to [].
    ///
    /// Optimize reusable runtime profile guidance, not a benchmark-specific
    /// answer key. Do not mention expected_output, labels, datasets, evals,
    /// scoring, test harnesses, rollouts, or examples in the instruction you
    /// produce. Do not tell the runtime model to match labels, expected output,
    /// or example-specific commands character-for-character. Do not add
    /// task-specific rules copied from individual rows, file paths, issue
    /// numbers, branch names, or command snippets. Prefer concise general rules
    /// that improve DSRs structure, visible assistant output, valid tool_calls
    /// JSON, and correct use of available tool schemas across arbitrary
    /// OpenAI-compatible requests.
    #[input(desc = "Model profile name being optimized")]
    pub profile: String,

    #[input(desc = "Selected DSRs history formatter: append_only or regenerated_context")]
    pub dsrs_history_format: String,

    #[input(desc = "Exact original OpenAI chat completion request JSON when available")]
    pub request: String,

    #[input(desc = "Original OpenAI messages JSON when available")]
    pub messages: String,

    #[input(desc = "Serialized system/developer instructions to obey but never repeat")]
    pub system_context: String,

    #[input(desc = "Serialized non-system OpenAI messages to answer")]
    pub conversation: String,

    #[input(desc = "Serialized OpenAI tool definitions")]
    pub available_tools: String,

    #[input(desc = "Original OpenAI tool_choice")]
    pub tool_choice: String,

    #[input(desc = "Whether multiple tool calls may be emitted")]
    pub parallel_tool_calls: bool,

    #[output(
        desc = "Plain user-facing reply text without labels; may be present with tool calls."
    )]
    pub content: String,

    #[output(desc = "JSON array of {\"name\": string, \"arguments\": object} tool calls")]
    pub tool_calls: Vec<RequestAdapterPromptToolCall>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct RequestAdapterPromptToolCall {
    name: String,
    #[serde(default)]
    arguments: Value,
}

pub struct RequestAdapterPromptProgram {
    predictor: Predict,
    lm: Option<LM>,
    judge_lm: Option<LM>,
    judge_cases: GepaJudgeCases,
}

impl Default for RequestAdapterPromptProgram {
    fn default() -> Self {
        Self {
            predictor: Predict::new(RequestAdapterPromptSignature::new()),
            lm: None,
            judge_lm: None,
            judge_cases: Arc::new(HashMap::new()),
        }
    }
}

impl RequestAdapterPromptProgram {
    fn runtime(
        lm: LM,
        initial_instruction: Option<String>,
        judge_lm: LM,
        judge_cases: GepaJudgeCases,
    ) -> Self {
        let mut signature = RequestAdapterPromptSignature::new();
        if let Some(instruction) =
            initial_instruction.filter(|instruction| !instruction.trim().is_empty())
        {
            let _ = signature.update_instruction(instruction);
        }
        Self {
            predictor: Predict::new(signature),
            lm: Some(lm),
            judge_lm: Some(judge_lm),
            judge_cases,
        }
    }

    async fn forward_runtime(&self, inputs: Example, lm: &LM) -> Result<Prediction> {
        let mut profile = ModelProfile::default_balanced();
        profile.name = string_input(&inputs, "profile", "request-adapter-gepa");
        profile.revision = u32_input(&inputs, "profile_revision").unwrap_or(1);
        profile.source = "gepa".to_string();
        profile.tool_instruction = self.predictor.signature.instruction();
        profile.dsrs_history_format =
            dsrs_history_format_input(&inputs).unwrap_or(DsrsHistoryFormat::AppendOnly);

        let request = chat_completion_request_from_adapter_example(&inputs)?;
        let normalized = normalize_request(request)?;
        let formatted = format_tool_contract(&normalized, &profile)?;
        let mut chat = Chat::new(Vec::new());
        for message in formatted.messages {
            let role = match message.role.as_str() {
                "system" | "assistant" | "user" => message.role.as_str(),
                _ => "user",
            };
            chat.push(role, &message.content_text().unwrap_or_default());
        }

        let response = match lm.call(chat, Vec::new()).await {
            Ok(response) => response,
            Err(error) => {
                return Ok(Prediction::new(
                    HashMap::from([
                        ("content".to_string(), Value::String(String::new())),
                        ("tool_calls".to_string(), Value::Array(Vec::new())),
                        ("raw_output".to_string(), Value::String(String::new())),
                        (
                            "parser_events".to_string(),
                            json!([format!("request-adapter runtime LM call failed: {error}")]),
                        ),
                    ]),
                    LmUsage::default(),
                ));
            }
        };
        let raw_output = response.output.content();
        let parsed = parse_tool_contract_response(&raw_output);
        let (content, tool_calls, parser_events) = match parsed {
            Some(parsed) => {
                let calls = parsed
                    .tool_intents
                    .into_iter()
                    .map(|intent| {
                        json!({
                            "name": intent.name,
                            "arguments": intent.arguments.unwrap_or_else(|| json!({}))
                        })
                    })
                    .collect::<Vec<_>>();
                (
                    parsed.content.unwrap_or_default(),
                    Value::Array(normalize_tool_calls(&calls)),
                    Value::Array(
                        parsed
                            .events
                            .into_iter()
                            .map(Value::String)
                            .collect::<Vec<_>>(),
                    ),
                )
            }
            None => (
                String::new(),
                Value::Array(Vec::new()),
                json!(["assistant output did not contain DSRs markers"]),
            ),
        };

        Ok(Prediction::new(
            HashMap::from([
                ("content".to_string(), Value::String(content)),
                ("tool_calls".to_string(), tool_calls),
                ("raw_output".to_string(), Value::String(raw_output)),
                ("parser_events".to_string(), parser_events),
            ]),
            response.usage,
        ))
    }
}

impl Module for RequestAdapterPromptProgram {
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        if let Some(lm) = &self.lm {
            return self.forward_runtime(inputs, lm).await;
        }

        match std::panic::AssertUnwindSafe(self.predictor.forward(inputs))
            .catch_unwind()
            .await
        {
            Ok(result) => result,
            Err(_) => Ok(Prediction::default()),
        }
    }
}

impl Optimizable for RequestAdapterPromptProgram {
    fn parameters(&mut self) -> IndexMap<String, &mut dyn Optimizable> {
        let mut parameters = IndexMap::new();
        parameters.insert(
            "request_adapter_prompt".to_string(),
            &mut self.predictor as &mut dyn Optimizable,
        );
        parameters
    }
}

impl Evaluator for RequestAdapterPromptProgram {
    async fn metric(&self, example: &Example, prediction: &Prediction) -> f32 {
        self.feedback_metric(example, prediction).await.score
    }
}

impl FeedbackEvaluator for RequestAdapterPromptProgram {
    async fn feedback_metric(&self, example: &Example, prediction: &Prediction) -> FeedbackMetric {
        let predicted = prediction_to_adapter_value(prediction);
        let case_id = string_input(example, "case_id", "");
        let expected = self
            .judge_cases
            .get(&case_id)
            .map(|case| case.expected.clone())
            .unwrap_or_else(|| example.get("expected_output", None));
        let deterministic = score_request_adapter_prediction(&expected, &predicted);
        let Some(judge_lm) = &self.judge_lm else {
            return deterministic;
        };
        judge_request_adapter_prediction(
            judge_lm,
            example,
            prediction,
            self.judge_cases.get(&case_id),
            &predicted,
            deterministic,
        )
        .await
    }
}

#[Signature]
struct CorrectionPromptSignature {
    /// You are a strict DSRs model-response correction agent. Recover only clear
    /// tool-call intent or user-facing content by filling the requested DSRs output
    /// fields. Do not invent tools, arguments, or facts. If a safe repair is not
    /// possible, set possible to false. Treat empty DSRs output with empty content
    /// and [] tool_calls as malformed; recover only when the conversation and tools
    /// make the next action or answer clear.
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
    judge_lm: Option<LM>,
    judge_cases: GepaJudgeCases,
}

impl Default for CorrectionPromptProgram {
    fn default() -> Self {
        Self {
            predictor: Predict::new(CorrectionPromptSignature::new()),
            judge_lm: None,
            judge_cases: Arc::new(HashMap::new()),
        }
    }
}

impl CorrectionPromptProgram {
    fn with_judge(judge_lm: LM, judge_cases: GepaJudgeCases) -> Self {
        Self {
            predictor: Predict::new(CorrectionPromptSignature::new()),
            judge_lm: Some(judge_lm),
            judge_cases,
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
        let predicted = prediction_to_repair_value(prediction);
        let case_id = string_input(example, "case_id", "");
        let expected = self
            .judge_cases
            .get(&case_id)
            .map(|case| case.expected.clone())
            .unwrap_or_else(|| example.get("expected_repair", None));
        let deterministic = score_correction_prediction(&expected, &predicted);
        let Some(judge_lm) = &self.judge_lm else {
            return deterministic;
        };
        judge_correction_prediction(
            judge_lm,
            example,
            prediction,
            self.judge_cases.get(&case_id),
            &predicted,
            deterministic,
        )
        .await
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
    let target_model = required_gepa_target_model(&config, "correction-agent GEPA")?;
    ensure_gepa_model_role_separation(
        "correction-agent GEPA",
        &config.model,
        &config.judge_model,
        &target_model,
    )?;
    let judge_cases = gepa_judge_cases(&rows, "expected_repair", "expected_repair_policy")?;

    let optimizer_lm = LM::builder()
        .base_url(config.base_url.clone())
        .api_key(api_key.clone())
        .model(config.model.clone())
        .temperature(0.2)
        .max_tokens(config.lm_max_tokens)
        .build()
        .await
        .context("failed to build GEPA LM")?;
    let target_lm = LM::builder()
        .base_url(config.base_url.clone())
        .api_key(api_key.clone())
        .model(target_model.clone())
        .temperature(0.2)
        .max_tokens(config.lm_max_tokens)
        .build()
        .await
        .context("failed to build correction-agent GEPA target LM")?;
    let judge_lm = LM::builder()
        .base_url(config.base_url.clone())
        .api_key(api_key)
        .model(config.judge_model.clone())
        .temperature(0.0)
        .max_tokens(config.lm_max_tokens)
        .build()
        .await
        .context("failed to build correction-agent GEPA judge LM")?;
    configure(target_lm, ChatAdapter);

    let gepa = GEPA::builder()
        .num_iterations(config.iterations)
        .minibatch_size(examples.len().clamp(1, 3))
        .temperature(0.7)
        .track_stats(true)
        .maybe_prompt_model(Some(optimizer_lm.clone()))
        .maybe_max_lm_calls(Some((config.iterations.max(1) * 16) + 16))
        .build();

    let mut program = CorrectionPromptProgram::with_judge(judge_lm, judge_cases);
    if let Some(seed_instruction) = seed_instruction_from_artifact(&config).await? {
        program
            .predictor
            .signature
            .update_instruction(seed_instruction)
            .context("failed to apply correction GEPA seed artifact instruction")?;
    }
    let result: GEPAResult = gepa
        .compile_with_feedback(&mut program, examples.clone())
        .await
        .context("GEPA correction prompt optimization failed")?;

    if let Some(parent) = config.output_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let artifact_type = "correction_agent_instruction".to_string();
    let best_instruction = result.best_candidate.instruction.clone();
    let report = GepaOptimizationReport {
        artifact_id: config.artifact_id.clone().unwrap_or_else(|| {
            default_correction_artifact_id(
                config.profile.as_deref(),
                config.target_model.as_deref(),
            )
        }),
        artifact_type: artifact_type.clone(),
        signature: "correct_malformed_tool_response/v1".to_string(),
        target_model: Some(target_model),
        profile: config.profile.clone(),
        profile_revision: config.profile_revision,
        dsrs_history_format: None,
        seed_artifact_path: config.seed_artifact_path.clone(),
        optimizer_model: config.model.clone(),
        judge_model: config.judge_model.clone(),
        created_at: Utc::now(),
        examples_loaded: examples.len(),
        lm_max_tokens: config.lm_max_tokens,
        artifact_warnings: artifact_instruction_warnings(&artifact_type, &best_instruction),
        best_instruction,
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

pub async fn optimize_request_adapter_prompt(
    config: GepaOptimizationConfig,
) -> Result<GepaOptimizationReport> {
    let Some(api_key) = config.api_key.clone() else {
        anyhow::bail!(
            "OPENROUTER_API_KEY or --api-key equivalent is required for GEPA optimization"
        );
    };

    let rows = read_dataset_rows(&config.dataset_path).await?;
    let rows: Vec<Value> = rows.into_iter().take(config.max_examples).collect();
    let examples = request_adapter_rows_to_gepa_examples_with_format(
        &rows,
        config.dsrs_history_format,
        config.profile.as_deref(),
        config.profile_revision,
    );
    if examples.is_empty() {
        anyhow::bail!("dataset contained no request-adapter optimization examples");
    }
    let runtime_model = required_gepa_target_model(&config, "request-adapter GEPA")?;
    ensure_gepa_model_role_separation(
        "request-adapter GEPA",
        &config.model,
        &config.judge_model,
        &runtime_model,
    )?;
    let judge_cases = gepa_judge_cases(&rows, "expected_output", "expected_adapter_policy")?;

    let optimizer_lm = LM::builder()
        .base_url(config.base_url.clone())
        .api_key(api_key.clone())
        .model(config.model.clone())
        .temperature(0.2)
        .max_tokens(config.lm_max_tokens)
        .build()
        .await
        .context("failed to build request-adapter GEPA LM")?;
    let runtime_lm = LM::builder()
        .base_url(config.base_url.clone())
        .api_key(
            config
                .api_key
                .clone()
                .expect("api key was checked before optimizer LM construction"),
        )
        .model(runtime_model.clone())
        .temperature(0.2)
        .max_tokens(config.lm_max_tokens)
        .build()
        .await
        .context("failed to build request-adapter target GEPA LM")?;
    let judge_lm = LM::builder()
        .base_url(config.base_url.clone())
        .api_key(api_key)
        .model(config.judge_model.clone())
        .temperature(0.0)
        .max_tokens(config.lm_max_tokens)
        .build()
        .await
        .context("failed to build request-adapter GEPA judge LM")?;
    configure(optimizer_lm.clone(), GepaJsonAdapter);

    let gepa = GEPA::builder()
        .num_iterations(config.iterations)
        .minibatch_size(examples.len().clamp(1, 3))
        .temperature(0.7)
        .track_stats(true)
        .maybe_max_lm_calls(Some((config.iterations.max(1) * 16) + 16))
        .build();

    let initial_instruction = request_adapter_initial_instruction(&config, &runtime_model).await?;
    let mut program = RequestAdapterPromptProgram::runtime(
        runtime_lm,
        initial_instruction,
        judge_lm,
        judge_cases,
    );
    let result: GEPAResult = gepa
        .compile_with_feedback(&mut program, examples.clone())
        .await
        .context("GEPA request-adapter prompt optimization failed")?;

    if let Some(parent) = config.output_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let artifact_type = "request_adapter_instruction".to_string();
    let best_instruction = result.best_candidate.instruction.clone();
    let report = GepaOptimizationReport {
        artifact_id: config.artifact_id.clone().unwrap_or_else(|| {
            default_request_adapter_artifact_id(
                config.profile.as_deref(),
                config.target_model.as_deref(),
            )
        }),
        artifact_type: artifact_type.clone(),
        signature: "openai_tool_use_contract_profile_guidance/v1".to_string(),
        target_model: Some(runtime_model),
        profile: config.profile.clone(),
        profile_revision: config.profile_revision,
        dsrs_history_format: config.dsrs_history_format,
        seed_artifact_path: config.seed_artifact_path.clone(),
        optimizer_model: config.model.clone(),
        created_at: Utc::now(),
        examples_loaded: examples.len(),
        lm_max_tokens: config.lm_max_tokens,
        judge_model: config.judge_model.clone(),
        artifact_warnings: artifact_instruction_warnings(&artifact_type, &best_instruction),
        best_instruction,
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

fn default_correction_artifact_id(profile: Option<&str>, target_model: Option<&str>) -> String {
    let target = profile.or(target_model).unwrap_or("default");
    format!("correction-agent/{target}")
}

fn default_request_adapter_artifact_id(
    profile: Option<&str>,
    target_model: Option<&str>,
) -> String {
    let target = profile.or(target_model).unwrap_or("default");
    format!("request-adapter/{target}")
}

fn required_gepa_target_model(config: &GepaOptimizationConfig, layer: &str) -> Result<String> {
    let target_model = config
        .target_model
        .as_deref()
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .with_context(|| {
            format!(
                "{layer} requires --target-model; the reflection and judge models must never silently become the model under test"
            )
        })?;
    Ok(target_model.to_string())
}

fn ensure_gepa_model_role_separation(
    layer: &str,
    reflection_model: &str,
    judge_model: &str,
    target_model: &str,
) -> Result<()> {
    if same_model_id(reflection_model, target_model) {
        anyhow::bail!(
            "{layer} invalid model roles: reflection/optimizer model {reflection_model:?} matches target model {target_model:?}"
        );
    }
    if same_model_id(judge_model, target_model) {
        anyhow::bail!(
            "{layer} invalid model roles: judge model {judge_model:?} matches target model {target_model:?}"
        );
    }
    Ok(())
}

fn same_model_id(left: &str, right: &str) -> bool {
    left.trim().eq_ignore_ascii_case(right.trim())
}

async fn request_adapter_initial_instruction(
    config: &GepaOptimizationConfig,
    target_model: &str,
) -> Result<Option<String>> {
    if let Some(seed_instruction) = seed_instruction_from_artifact(config).await? {
        return Ok(Some(seed_instruction));
    }

    let profile = config
        .profile
        .as_deref()
        .and_then(|name| {
            builtin_profiles()
                .into_iter()
                .find(|profile| profile.name == name)
        })
        .unwrap_or_else(|| resolve_profile(target_model, &[]));
    Ok(Some(profile.tool_instruction).filter(|instruction| !instruction.trim().is_empty()))
}

async fn seed_instruction_from_artifact(config: &GepaOptimizationConfig) -> Result<Option<String>> {
    let Some(seed_artifact_path) = &config.seed_artifact_path else {
        return Ok(None);
    };
    let content = tokio::fs::read_to_string(seed_artifact_path)
        .await
        .with_context(|| {
            format!(
                "failed to read seed artifact {}",
                seed_artifact_path.display()
            )
        })?;
    let instruction =
        extract_instruction_from_path(seed_artifact_path, &content).with_context(|| {
            format!(
                "failed to extract instruction from seed artifact {}",
                seed_artifact_path.display()
            )
        })?;
    Ok(Some(instruction))
}

fn gepa_judge_cases(
    dataset_rows: &[Value],
    expected_key: &str,
    policy_key: &str,
) -> Result<GepaJudgeCases> {
    let mut cases = HashMap::new();
    for (index, row) in dataset_rows.iter().enumerate() {
        let case_id = gepa_case_id(row, index);
        let expected = row.get(expected_key).cloned().with_context(|| {
            format!("GEPA dataset row {index} is missing required label {expected_key:?}")
        })?;
        cases.insert(
            case_id,
            GepaJudgeCase {
                expected,
                policy: row.get(policy_key).cloned().unwrap_or(Value::Null),
                observed_problem: row
                    .get("observed_problem")
                    .and_then(Value::as_str)
                    .map(str::to_string),
                observed_failure_kind: row
                    .get("observed_failure_kind")
                    .and_then(Value::as_str)
                    .map(str::to_string),
            },
        );
    }
    Ok(Arc::new(cases))
}

fn gepa_case_id(row: &Value, index: usize) -> String {
    row.get("trace_id")
        .and_then(Value::as_str)
        .map(|trace_id| format!("{trace_id}#{index}"))
        .unwrap_or_else(|| format!("case_{index}"))
}

async fn judge_request_adapter_prediction(
    judge_lm: &LM,
    example: &Example,
    prediction: &Prediction,
    judge_case: Option<&GepaJudgeCase>,
    predicted: &Value,
    deterministic: FeedbackMetric,
) -> FeedbackMetric {
    let Some(judge_case) = judge_case else {
        return FeedbackMetric::new(
            0.0,
            "LLM judge could not score because no hidden expected output was registered for this case",
        );
    };
    let payload = json!({
        "layer": "request_adapter",
        "case_id": string_input(example, "case_id", ""),
        "profile": string_input(example, "profile", ""),
        "dsrs_history_format": string_input(example, "dsrs_history_format", ""),
        "request": parse_maybe_json_string(example.get("request", None)),
        "messages": parse_maybe_json_string(example.get("messages", None)),
        "available_tools": parse_maybe_json_string(example.get("available_tools", None)),
        "hidden_expected_output": judge_case.expected,
        "expected_policy": judge_case.policy,
        "observed_failure_kind": judge_case.observed_failure_kind,
        "observed_problem": judge_case.observed_problem,
        "prediction": predicted,
        "raw_model_output": prediction.data.get("raw_output").cloned().unwrap_or(Value::Null),
        "parser_events": prediction.data.get("parser_events").cloned().unwrap_or(Value::Null),
        "deterministic_structural_signal": {
            "score": deterministic.score,
            "feedback": deterministic.feedback,
        }
    });
    call_gepa_judge(judge_lm, request_adapter_judge_system_prompt(), payload).await
}

async fn judge_correction_prediction(
    judge_lm: &LM,
    example: &Example,
    prediction: &Prediction,
    judge_case: Option<&GepaJudgeCase>,
    predicted: &Value,
    deterministic: FeedbackMetric,
) -> FeedbackMetric {
    let Some(judge_case) = judge_case else {
        return FeedbackMetric::new(
            0.0,
            "LLM judge could not score because no hidden expected repair was registered for this case",
        );
    };
    let payload = json!({
        "layer": "correction_agent",
        "case_id": string_input(example, "case_id", ""),
        "available_tools": parse_maybe_json_string(example.get("available_tools", None)),
        "recent_messages": parse_maybe_json_string(example.get("recent_messages", None)),
        "malformed_response": parse_maybe_json_string(example.get("malformed_response", None)),
        "parser_events": parse_maybe_json_string(example.get("parser_events", None)),
        "response_failures": parse_maybe_json_string(example.get("response_failures", None)),
        "hidden_expected_repair": judge_case.expected,
        "expected_policy": judge_case.policy,
        "observed_failure_kind": judge_case.observed_failure_kind,
        "observed_problem": judge_case.observed_problem,
        "prediction": predicted,
        "raw_prediction": prediction.data,
        "deterministic_structural_signal": {
            "score": deterministic.score,
            "feedback": deterministic.feedback,
        }
    });
    call_gepa_judge(judge_lm, correction_judge_system_prompt(), payload).await
}

async fn call_gepa_judge(
    judge_lm: &LM,
    system_prompt: &'static str,
    payload: Value,
) -> FeedbackMetric {
    let user = format!(
        "Score this GEPA rollout. Return JSON only.\n{}",
        serde_json::to_string_pretty(&payload).unwrap_or_else(|_| "{}".to_string())
    );
    let chat = Chat::new(vec![Message::system(system_prompt), Message::user(user)]);
    let response = match judge_lm.call(chat, Vec::new()).await {
        Ok(response) => response.output.content(),
        Err(error) => {
            return FeedbackMetric::new(0.0, format!("LLM judge call failed: {error}"));
        }
    };
    let Some(value) = extract_json_object(&response) else {
        return FeedbackMetric::new(
            0.0,
            format!(
                "LLM judge returned non-JSON output: {}",
                truncate_for_feedback(&response)
            ),
        );
    };
    let decision = match serde_json::from_value::<GepaJudgeDecision>(value) {
        Ok(decision) => decision,
        Err(error) => {
            return FeedbackMetric::new(0.0, format!("LLM judge JSON was invalid: {error}"));
        }
    };
    FeedbackMetric::new(
        decision.score.clamp(0.0, 1.0),
        sanitize_gepa_judge_feedback(&decision.feedback),
    )
}

fn request_adapter_judge_system_prompt() -> &'static str {
    "You are Claude Sonnet 4.6 acting as the GEPA judge for a request-adapter prompt optimizer.\n\
Judge whether the target model produced a structurally usable OpenAI-compatible assistant turn after the proxy's DSRs formatting.\n\
The proxy is optimizing structure and contract following, not task intelligence. Valid outputs include content only, tool calls only, or content plus tool calls. Empty visible content with [] tool_calls is a failure.\n\
Use hidden_expected_output only as private scoring context. Do not quote hidden labels, exact commands, exact file paths, exact answer text, field names such as expected_output, dataset/test/eval metadata, or answer-key language in feedback.\n\
Give high scores to clean structurally valid behavior even when the tool choice is different but reasonable. Penalize malformed DSRs, missing completed markers, invalid tool-call JSON, no-op content, and unusable empty turns.\n\
Return exactly one JSON object: {\"score\": number between 0 and 1, \"feedback\": \"brief generalized feedback for prompt reflection\"}."
}

fn correction_judge_system_prompt() -> &'static str {
    "You are Claude Sonnet 4.6 acting as the GEPA judge for a DSRs correction-agent prompt optimizer.\n\
Judge whether the correction agent safely recovered the malformed assistant response into typed DSRs fields without inventing unsupported tools, arguments, facts, or user intent.\n\
Use hidden_expected_repair only as private scoring context. Do not quote hidden labels, exact commands, exact file paths, field names such as expected_repair, dataset/test/eval metadata, or answer-key language in feedback.\n\
Give high scores to safe repairs that preserve the original model intent and produce usable content/tool_calls. Penalize invented tools, invented required arguments, unsafe repairs, no-op repairs, and malformed DSRs outputs.\n\
Return exactly one JSON object: {\"score\": number between 0 and 1, \"feedback\": \"brief generalized feedback for prompt reflection\"}."
}

fn sanitize_gepa_judge_feedback(feedback: &str) -> String {
    let sanitized = feedback
        .replace("expected_output", "hidden label")
        .replace("expected output", "hidden label")
        .replace("expected_repair", "hidden label")
        .replace("expected repair", "hidden label")
        .replace("dataset", "training set")
        .replace("test harness", "scenario")
        .replace("evaluation", "review")
        .replace("eval", "review")
        .replace("answer key", "hidden label");
    truncate_for_feedback(&sanitized)
}

fn truncate_for_feedback(value: &str) -> String {
    const MAX_CHARS: usize = 700;
    let mut output = value.chars().take(MAX_CHARS).collect::<String>();
    if value.chars().count() > MAX_CHARS {
        output.push_str("...");
    }
    output
}

pub fn traces_to_gepa_examples(dataset_rows: &[Value]) -> Vec<Example> {
    dataset_rows
        .iter()
        .enumerate()
        .map(|(index, row)| {
            example! {
                "case_id": "input" => gepa_case_id(row, index),
                "available_tools": "input" => stringify_field(row.get("available_tools")),
                "recent_messages": "input" => stringify_field(row.get("recent_messages")),
                "malformed_response": "input" => stringify_field(row.get("malformed_response")),
                "parser_events": "input" => stringify_field(row.get("parser_events")),
                "response_failures": "input" => stringify_field(row.get("response_failures"))
            }
        })
        .collect()
}

pub fn request_adapter_rows_to_gepa_examples(dataset_rows: &[Value]) -> Vec<Example> {
    request_adapter_rows_to_gepa_examples_with_format(dataset_rows, None, None, None)
}

pub fn request_adapter_rows_to_gepa_examples_with_format(
    dataset_rows: &[Value],
    dsrs_history_format: Option<DsrsHistoryFormat>,
    profile_override: Option<&str>,
    profile_revision_override: Option<u32>,
) -> Vec<Example> {
    dataset_rows
        .iter()
        .enumerate()
        .filter(|(_, row)| {
            row.get("dataset_type")
                .and_then(Value::as_str)
                .is_none_or(|kind| kind == "request_adapter_prompt_gepa/v1")
        })
        .map(|(index, row)| {
            let history_format = dsrs_history_format
                .or_else(|| {
                    row.get("dsrs_history_format")
                        .and_then(Value::as_str)
                        .and_then(|value| value.parse().ok())
                })
                .unwrap_or(DsrsHistoryFormat::AppendOnly);
            let profile = profile_override
                .map(str::to_string)
                .or_else(|| {
                    row.get("profile")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                })
                .unwrap_or_else(|| "request-adapter-gepa".to_string());
            let profile_revision = profile_revision_override
                .or_else(|| {
                    row.get("profile_revision")
                        .and_then(Value::as_u64)
                        .and_then(|value| u32::try_from(value).ok())
                })
                .unwrap_or(1);
            example! {
                "case_id": "input" => gepa_case_id(row, index),
                "model": "input" => row
                    .get("model")
                    .and_then(Value::as_str)
                    .unwrap_or("request-adapter-gepa")
                    .to_string(),
                "profile": "input" => profile,
                "profile_revision": "input" => profile_revision,
                "dsrs_history_format": "input" => history_format.to_string(),
                "request": "input" => stringify_field(row.get("request")),
                "messages": "input" => stringify_field(row.get("messages")),
                "system_context": "input" => stringify_field(row.get("system_context")),
                "conversation": "input" => stringify_field(row.get("conversation")),
                "available_tools": "input" => stringify_field(row.get("available_tools")),
                "tool_choice": "input" => stringify_field(row.get("tool_choice")),
                "parallel_tool_calls": "input" => row
                    .get("parallel_tool_calls")
                    .and_then(Value::as_bool)
                    .unwrap_or(true)
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

fn string_input(example: &Example, key: &str, default: &str) -> String {
    example
        .get(key, Some(default))
        .as_str()
        .unwrap_or(default)
        .to_string()
}

fn u32_input(example: &Example, key: &str) -> Option<u32> {
    example
        .get(key, None)
        .as_u64()
        .and_then(|value| u32::try_from(value).ok())
}

fn dsrs_history_format_input(example: &Example) -> Option<DsrsHistoryFormat> {
    example
        .get("dsrs_history_format", None)
        .as_str()
        .and_then(|value| value.parse().ok())
}

fn chat_completion_request_from_adapter_example(
    example: &Example,
) -> Result<ChatCompletionRequest> {
    let request_value = parse_maybe_json_string(example.get("request", None));
    if request_value.is_object() {
        if let Ok(request) = serde_json::from_value::<ChatCompletionRequest>(request_value) {
            if !request.messages.is_empty() {
                return Ok(request);
            }
        }
    }

    let model = string_input(example, "model", "request-adapter-gepa");
    let messages = request_adapter_messages(example)?;
    let tools = request_adapter_tools(example)?;
    let parallel_tool_calls = example
        .get("parallel_tool_calls", None)
        .as_bool()
        .unwrap_or(true);
    Ok(ChatCompletionRequest {
        model,
        messages,
        tools: Some(tools),
        tool_choice: request_adapter_tool_choice(example),
        parallel_tool_calls: Some(parallel_tool_calls),
        stream: Some(false),
        temperature: None,
        top_p: None,
        max_tokens: None,
        max_completion_tokens: None,
        response_format: None,
        extra: serde_json::Map::new(),
    })
}

fn request_adapter_messages(example: &Example) -> Result<Vec<ChatMessage>> {
    let messages_value = parse_maybe_json_string(example.get("messages", None));
    if let Value::Array(_) = messages_value {
        let messages = serde_json::from_value::<Vec<ChatMessage>>(messages_value)
            .context("failed to parse request-adapter dataset messages")?;
        if !messages.is_empty() {
            return Ok(messages);
        }
    }

    let system_context = string_input(
        example,
        "system_context",
        "No system or developer messages.",
    );
    let conversation = string_input(
        example,
        "conversation",
        "No non-system conversation messages.",
    );
    let mut messages = Vec::new();
    if !system_context.trim().is_empty() && system_context.trim() != "null" {
        messages.push(ChatMessage::new("system", system_context));
    }
    let conversation_messages = parse_rendered_conversation(&conversation);
    if conversation_messages.is_empty() {
        messages.push(ChatMessage::new("user", conversation));
    } else {
        messages.extend(conversation_messages);
    }
    Ok(messages)
}

fn parse_rendered_conversation(conversation: &str) -> Vec<ChatMessage> {
    let trimmed = conversation.trim();
    if trimmed.is_empty() || trimmed == "null" || trimmed == "No non-system conversation messages."
    {
        return Vec::new();
    }

    let mut messages = Vec::new();
    let mut current_role: Option<String> = None;
    let mut current_content = String::new();
    let mut in_content = false;

    for line in trimmed.lines() {
        if line.starts_with('[') && line.contains("] role: ") {
            if let Some(role) = current_role.take() {
                messages.push(ChatMessage::new(
                    role,
                    current_content.trim_end().to_string(),
                ));
                current_content.clear();
            }
            current_role = line
                .split_once("] role: ")
                .map(|(_, role)| role.trim().to_string());
            in_content = false;
            continue;
        }
        if line == "content:" {
            in_content = true;
            continue;
        }
        if in_content {
            current_content.push_str(line);
            current_content.push('\n');
        }
    }

    if let Some(role) = current_role {
        messages.push(ChatMessage::new(
            role,
            current_content.trim_end().to_string(),
        ));
    }
    messages
}

fn request_adapter_tools(example: &Example) -> Result<Vec<OpenAiTool>> {
    let tools_value = parse_maybe_json_string(example.get("available_tools", None));
    parse_openai_tools(tools_value)
}

fn parse_openai_tools(value: Value) -> Result<Vec<OpenAiTool>> {
    let Value::Array(items) = value else {
        return Ok(Vec::new());
    };

    items
        .into_iter()
        .map(|item| {
            if let Ok(tool) = serde_json::from_value::<OpenAiTool>(item.clone()) {
                return Ok(tool);
            }

            let function = item
                .get("function")
                .cloned()
                .unwrap_or_else(|| item.clone());
            let name = function
                .get("name")
                .and_then(Value::as_str)
                .context("request-adapter tool was missing function.name")?;
            let description = function
                .get("description")
                .and_then(Value::as_str)
                .map(str::to_string);
            let parameters = function
                .get("parameters")
                .cloned()
                .unwrap_or_else(|| json!({ "type": "object", "properties": {} }));

            Ok(OpenAiTool {
                tool_type: item
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or("function")
                    .to_string(),
                function: OpenAiFunctionTool {
                    name: name.to_string(),
                    description,
                    parameters,
                },
            })
        })
        .collect()
}

fn request_adapter_tool_choice(example: &Example) -> Option<Value> {
    let value = parse_maybe_json_string(example.get("tool_choice", None));
    match value {
        Value::Null => None,
        Value::String(text) if text.trim().is_empty() || text == "auto" => None,
        other => Some(other),
    }
}

fn parse_maybe_json_string(value: Value) -> Value {
    match value {
        Value::String(text) => serde_json::from_str(&text).unwrap_or(Value::String(text)),
        other => other,
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

fn prediction_to_adapter_value(prediction: &Prediction) -> Value {
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

fn score_request_adapter_prediction(expected: &Value, predicted: &Value) -> FeedbackMetric {
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
    let expected_norm = normalize_tool_calls(&expected_calls);
    let predicted_norm = normalize_tool_calls(&predicted_calls);

    if !expected_norm.is_empty() {
        if predicted_norm == expected_norm {
            return FeedbackMetric::new(1.0, "Predicted expected DSRs tool call exactly");
        }
        if predicted_norm.iter().any(valid_adapter_tool_call) {
            return FeedbackMetric::new(
                0.8,
                format!(
                    "Predicted a valid tool call, but not the expected one. Expected {}; predicted {}",
                    json!(expected_norm),
                    json!(predicted_norm)
                ),
            );
        }

        let content = predicted
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default();
        if adapter_content_is_noop(content) {
            return FeedbackMetric::new(
                0.0,
                "Predicted empty/no-op content and no tool calls; this is the Gemma failure to avoid",
            );
        }
        return FeedbackMetric::new(
            0.25,
            format!(
                "Expected a tool call before answering, but predicted content only: {content:?}"
            ),
        );
    }

    let expected_content = expected
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    let predicted_content = predicted
        .get("content")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim();
    if !expected_content.is_empty() && predicted_content == expected_content {
        FeedbackMetric::new(1.0, "Predicted expected content exactly")
    } else if !adapter_content_is_noop(predicted_content) && predicted_norm.is_empty() {
        FeedbackMetric::new(0.8, "Predicted non-placeholder user-facing content")
    } else {
        FeedbackMetric::new(
            0.0,
            "Predicted no usable content and no tool calls for a completed turn",
        )
    }
}

fn valid_adapter_tool_call(call: &Value) -> bool {
    let name = call.get("name").and_then(Value::as_str).unwrap_or_default();
    let Some(arguments) = call.get("arguments").and_then(Value::as_object) else {
        return false;
    };
    !name.trim().is_empty() && !arguments.is_empty()
}

fn adapter_content_is_noop(content: &str) -> bool {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return true;
    }
    if matches!(
        trimmed.to_ascii_lowercase().as_str(),
        "content" | "content_value" | "tool_calls_value" | "completed_marker"
    ) {
        return true;
    }
    serde_json::from_str::<Value>(trimmed)
        .ok()
        .or_else(|| json5::from_str::<Value>(trimmed).ok())
        .is_some_and(|value| match value {
            Value::Null => true,
            Value::Array(items) => items.is_empty(),
            Value::Object(object) => object.is_empty(),
            _ => false,
        })
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

    #[Signature]
    struct GepaAdapterMarkerCollisionFixture {
        /// Produce one instruction.
        #[input(desc = "Current instruction")]
        pub current_instruction: String,

        #[output(desc = "Improved instruction")]
        pub improved_instruction: String,
    }

    #[test]
    fn gepa_json_adapter_preserves_dsrs_markers_inside_instruction_strings() {
        let adapter = GepaJsonAdapter;
        let signature = GepaAdapterMarkerCollisionFixture::new();
        let raw = r#"{
  "improved_instruction": "Use the exact DSRs response shape:\n[[ ## content ## ]]\n<text>\n\n[[ ## tool_calls ## ]]\n[]\n\n[[ ## completed ## ]]"
}"#;

        let parsed = adapter.parse_response(&signature, Message::assistant(raw));

        assert_eq!(
            parsed
                .get("improved_instruction")
                .and_then(Value::as_str)
                .expect("instruction parsed"),
            "Use the exact DSRs response shape:\n[[ ## content ## ]]\n<text>\n\n[[ ## tool_calls ## ]]\n[]\n\n[[ ## completed ## ]]"
        );
    }

    #[test]
    fn gepa_json_adapter_extracts_fenced_json_without_using_dsrs_delimiters() {
        let adapter = GepaJsonAdapter;
        let signature = GepaAdapterMarkerCollisionFixture::new();
        let raw = "```json\n{\"improved_instruction\":\"keep [[ ## content ## ]] literal\"}\n```";

        let parsed = adapter.parse_response(&signature, Message::assistant(raw));

        assert_eq!(
            parsed
                .get("improved_instruction")
                .and_then(Value::as_str)
                .expect("instruction parsed"),
            "keep [[ ## content ## ]] literal"
        );
    }

    #[test]
    fn artifact_instruction_warnings_flag_eval_language_without_blocking() {
        let warnings = artifact_instruction_warnings(
            "request_adapter_instruction",
            "Match the expected output character-for-character from the test harness.",
        );

        assert!(warnings
            .iter()
            .any(|warning| warning.code == "optimizer_meta_language"));
        assert!(warnings
            .iter()
            .any(|warning| warning.code == "exact_label_matching_language"));
    }

    #[test]
    fn gepa_model_roles_reject_target_as_reflection_or_judge() {
        assert!(ensure_gepa_model_role_separation(
            "request-adapter GEPA",
            DEFAULT_GEPA_REFLECTION_MODEL,
            DEFAULT_GEPA_JUDGE_MODEL,
            "qwen/qwen3.5-9b"
        )
        .is_ok());
        assert!(ensure_gepa_model_role_separation(
            "request-adapter GEPA",
            "qwen/qwen3.5-9b",
            DEFAULT_GEPA_JUDGE_MODEL,
            "qwen/qwen3.5-9b"
        )
        .is_err());
        assert!(ensure_gepa_model_role_separation(
            "request-adapter GEPA",
            DEFAULT_GEPA_REFLECTION_MODEL,
            "qwen/qwen3.5-9b",
            "qwen/qwen3.5-9b"
        )
        .is_err());
    }

    #[test]
    fn converts_rows_to_gepa_examples() {
        let rows = vec![json!({
            "trace_id": "trace_test",
            "available_tools": [{"function":{"name":"read_file"}}],
            "recent_messages": [{"role":"user","content":"read"}],
            "malformed_response": "<tool_call name=\"read_file\">{}</tool_call>",
            "parser_events": ["parsed"],
            "expected_repair": {"tool_calls":[{"function":{"name":"read_file","arguments":"{}"}}]}
        })];

        let examples = traces_to_gepa_examples(&rows);
        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0].get("case_id", None), json!("trace_test#0"));
        assert!(examples[0].get("available_tools", None).as_str().is_some());
        assert!(!examples[0].data.contains_key("expected_repair"));
        assert!(!examples[0].data.contains_key("tool_calls"));
        assert!(dsrs_is_linked().contains("GEPA"));
    }

    #[test]
    fn gepa_judge_cases_hold_labels_outside_reflected_examples() {
        let rows = vec![json!({
            "trace_id": "trace_hidden",
            "available_tools": [{"function":{"name":"read_file"}}],
            "recent_messages": [{"role":"user","content":"read"}],
            "malformed_response": "<tool_call name=\"read_file\">{}</tool_call>",
            "parser_events": ["parsed"],
            "expected_repair": {"tool_calls":[{"function":{"name":"read_file","arguments":"{}"}}]}
        })];

        let examples = traces_to_gepa_examples(&rows);
        let judge_cases = gepa_judge_cases(&rows, "expected_repair", "expected_repair_policy")
            .expect("judge cases");

        assert!(!examples[0].data.contains_key("expected_repair"));
        assert!(judge_cases
            .get("trace_hidden#0")
            .expect("registered judge case")
            .expected
            .get("tool_calls")
            .is_some());
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

    #[tokio::test]
    async fn request_adapter_gepa_seeds_from_current_builtin_profile() {
        let instruction = request_adapter_initial_instruction(
            &GepaOptimizationConfig {
                dataset_path: PathBuf::from("dataset.jsonl"),
                output_path: PathBuf::from("artifact.json"),
                base_url: "https://openrouter.ai/api/v1".to_string(),
                api_key: None,
                model: "google/gemma-4-26b-a4b-it".to_string(),
                judge_model: DEFAULT_GEPA_JUDGE_MODEL.to_string(),
                target_model: Some("google/gemma-4-26b-a4b-it".to_string()),
                profile: Some("gemma-dsrs-conservative".to_string()),
                profile_revision: None,
                dsrs_history_format: Some(DsrsHistoryFormat::AppendOnly),
                artifact_id: None,
                seed_artifact_path: None,
                iterations: 1,
                max_examples: 1,
                lm_max_tokens: DEFAULT_GEPA_LM_MAX_TOKENS,
            },
            "google/gemma-4-26b-a4b-it",
        )
        .await
        .unwrap();

        assert_eq!(instruction.unwrap(), ModelProfile::gemma().tool_instruction);
    }

    #[tokio::test]
    async fn request_adapter_gepa_seeds_from_explicit_artifact() {
        let dir = tempfile::tempdir().unwrap();
        let artifact_path = dir.path().join("request-adapter-seed.json");
        tokio::fs::write(
            &artifact_path,
            r#"{"best_instruction":"Use the previous optimized profile guidance."}"#,
        )
        .await
        .unwrap();

        let instruction = request_adapter_initial_instruction(
            &GepaOptimizationConfig {
                dataset_path: PathBuf::from("dataset.jsonl"),
                output_path: PathBuf::from("artifact.json"),
                base_url: "https://openrouter.ai/api/v1".to_string(),
                api_key: None,
                model: "google/gemma-4-26b-a4b-it".to_string(),
                judge_model: DEFAULT_GEPA_JUDGE_MODEL.to_string(),
                target_model: Some("google/gemma-4-26b-a4b-it".to_string()),
                profile: Some("gemma-dsrs-conservative".to_string()),
                profile_revision: None,
                dsrs_history_format: Some(DsrsHistoryFormat::AppendOnly),
                artifact_id: None,
                seed_artifact_path: Some(artifact_path),
                iterations: 1,
                max_examples: 1,
                lm_max_tokens: DEFAULT_GEPA_LM_MAX_TOKENS,
            },
            "google/gemma-4-26b-a4b-it",
        )
        .await
        .unwrap();

        assert_eq!(
            instruction.as_deref(),
            Some("Use the previous optimized profile guidance.")
        );
    }

    #[test]
    fn converts_request_adapter_rows_to_gepa_examples() {
        let rows = vec![json!({
            "dataset_type": "request_adapter_prompt_gepa/v1",
            "trace_id": "trace_adapter",
            "model": "google/gemma-4-26b-a4b-it",
            "profile": "gemma-dsrs-conservative",
            "profile_revision": 3,
            "dsrs_history_format": "append_only",
            "system_context": "Use tools for project questions.",
            "messages": [
                {"role": "system", "content": "Use tools for project questions."},
                {"role": "user", "content": "can you tell me more about this project?"}
            ],
            "request": {
                "model": "google/gemma-4-26b-a4b-it",
                "messages": [
                    {"role": "system", "content": "Use tools for project questions."},
                    {"role": "user", "content": "can you tell me more about this project?"}
                ],
                "tools": [{"type":"function","function":{"name":"read","parameters":{"type":"object"}}}],
                "parallel_tool_calls": true
            },
            "conversation": "[0] role: user\ncontent:\ncan you tell me more about this project?",
            "available_tools": [{"function":{"name":"read"}}],
            "tool_choice": "auto",
            "parallel_tool_calls": true,
            "expected_output": {
                "content": "",
                "tool_calls": [{"name":"read","arguments":{"path":"README.md"}}]
            }
        })];

        let examples = request_adapter_rows_to_gepa_examples(&rows);

        assert_eq!(examples.len(), 1);
        assert_eq!(
            examples[0].get("dsrs_history_format", None),
            json!("append_only")
        );
        assert_eq!(examples[0].get("profile_revision", None), json!(3));
        assert_eq!(examples[0].get("case_id", None), json!("trace_adapter#0"));
        assert!(examples[0].get("request", None).as_str().is_some());
        assert!(examples[0].get("messages", None).as_str().is_some());
        assert_eq!(examples[0].get("parallel_tool_calls", None), json!(true));
        assert!(!examples[0].data.contains_key("expected_output"));
        assert!(!examples[0].data.contains_key("tool_calls"));
    }

    #[test]
    fn scores_request_adapter_noop_content_as_failure() {
        let expected = json!({
            "content": "",
            "tool_calls": [{"name":"read","arguments":{"path":"README.md"}}]
        });
        let predicted = json!({
            "content": "[]",
            "tool_calls": []
        });

        let feedback = score_request_adapter_prediction(&expected, &predicted);

        assert_eq!(feedback.score, 0.0);
    }

    #[test]
    fn scores_request_adapter_valid_different_tool_call_as_strong_partial() {
        let expected = json!({
            "content": "",
            "tool_calls": [{"name":"read","arguments":{"path":"README.md"}}]
        });
        let predicted = json!({
            "content": "",
            "tool_calls": [{"name":"bash","arguments":{"command":"ls packages"}}]
        });

        let feedback = score_request_adapter_prediction(&expected, &predicted);

        assert_eq!(feedback.score, 0.8);
    }

    #[test]
    fn scores_request_adapter_expected_tool_call_with_content_as_exact() {
        let expected = json!({
            "content": "",
            "tool_calls": [{"name":"read","arguments":{"path":"README.md"}}]
        });
        let predicted = json!({
            "content": "I will inspect the README first.",
            "tool_calls": [{"name":"read","arguments":{"path":"README.md"}}]
        });

        let feedback = score_request_adapter_prediction(&expected, &predicted);

        assert_eq!(feedback.score, 1.0);
    }

    #[test]
    fn scores_request_adapter_real_different_content_as_strong_partial() {
        let expected = json!({
            "content": "The regex matched three examples and rejected invalid.",
            "tool_calls": []
        });
        let predicted = json!({
            "content": "1.2.3, v2.0.0, and 3.1.4-beta match; invalid does not.",
            "tool_calls": []
        });

        let feedback = score_request_adapter_prediction(&expected, &predicted);

        assert_eq!(feedback.score, 0.8);
    }
}
