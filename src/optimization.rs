use std::{collections::HashMap, path::PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use dspy_rs::{
    configure, example, Chat, ChatAdapter, Evaluator, Example, FeedbackEvaluator, FeedbackMetric,
    GEPAResult, Module, Optimizable, Predict, Prediction, Predictor, Signature, GEPA, LM,
};
use futures::FutureExt;
use indexmap::IndexMap;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    dsrs_contract::{format_tool_contract, parse_tool_contract_response},
    model_profile::{DsrsHistoryFormat, ModelProfile},
    normalizer::normalize_request,
    openai::{ChatCompletionRequest, ChatMessage, OpenAiFunctionTool, OpenAiTool},
};

#[derive(Debug, Clone)]
pub struct GepaOptimizationConfig {
    pub dataset_path: PathBuf,
    pub output_path: PathBuf,
    pub base_url: String,
    pub api_key: Option<String>,
    pub model: String,
    pub target_model: Option<String>,
    pub profile: Option<String>,
    pub profile_revision: Option<u32>,
    pub dsrs_history_format: Option<DsrsHistoryFormat>,
    pub artifact_id: Option<String>,
    pub iterations: usize,
    pub max_examples: usize,
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
    pub optimizer_model: String,
    pub created_at: DateTime<Utc>,
    pub examples_loaded: usize,
    pub best_instruction: String,
    pub best_average_score: f32,
    pub total_rollouts: usize,
    pub total_lm_calls: usize,
    pub output_path: PathBuf,
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
    /// completed_marker in content are invalid answers. When using tools, leave
    /// content empty and put valid calls in tool_calls. When no tool is needed, put
    /// a real user-facing answer in content and set tool_calls to [].
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

    #[output(desc = "Plain user-facing reply text without labels; empty when using tools.")]
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
}

impl Default for RequestAdapterPromptProgram {
    fn default() -> Self {
        Self {
            predictor: Predict::new(RequestAdapterPromptSignature::new()),
            lm: None,
        }
    }
}

impl RequestAdapterPromptProgram {
    fn runtime(lm: LM) -> Self {
        Self {
            predictor: Predict::new(RequestAdapterPromptSignature::new()),
            lm: Some(lm),
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

        let response = lm.call(chat, Vec::new()).await?;
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
        let expected = example.get("expected_output", None);
        let predicted = prediction_to_adapter_value(prediction);
        score_request_adapter_prediction(&expected, &predicted)
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
        .maybe_prompt_model(Some(lm.clone()))
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
        artifact_id: config.artifact_id.clone().unwrap_or_else(|| {
            default_correction_artifact_id(
                config.profile.as_deref(),
                config.target_model.as_deref(),
            )
        }),
        artifact_type: "correction_agent_instruction".to_string(),
        signature: "correct_malformed_tool_response/v1".to_string(),
        target_model: config.target_model.clone(),
        profile: config.profile.clone(),
        profile_revision: config.profile_revision,
        dsrs_history_format: None,
        optimizer_model: config.model.clone(),
        created_at: Utc::now(),
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

    let lm = LM::builder()
        .base_url(config.base_url.clone())
        .api_key(api_key)
        .model(config.model.clone())
        .temperature(0.2)
        .build()
        .await
        .context("failed to build request-adapter GEPA LM")?;
    configure(lm.clone(), ChatAdapter);

    let gepa = GEPA::builder()
        .num_iterations(config.iterations)
        .minibatch_size(examples.len().clamp(1, 3))
        .temperature(0.7)
        .track_stats(true)
        .maybe_prompt_model(Some(lm.clone()))
        .maybe_max_lm_calls(Some((config.iterations.max(1) * 16) + 16))
        .build();

    let mut program = RequestAdapterPromptProgram::runtime(lm);
    let result: GEPAResult = gepa
        .compile_with_feedback(&mut program, examples.clone())
        .await
        .context("GEPA request-adapter prompt optimization failed")?;

    if let Some(parent) = config.output_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let report = GepaOptimizationReport {
        artifact_id: config.artifact_id.clone().unwrap_or_else(|| {
            default_request_adapter_artifact_id(
                config.profile.as_deref(),
                config.target_model.as_deref(),
            )
        }),
        artifact_type: "request_adapter_instruction".to_string(),
        signature: "openai_tool_use_contract_profile_guidance/v1".to_string(),
        target_model: config.target_model.clone(),
        profile: config.profile.clone(),
        profile_revision: config.profile_revision,
        dsrs_history_format: config.dsrs_history_format,
        optimizer_model: config.model.clone(),
        created_at: Utc::now(),
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
        .filter(|row| {
            row.get("dataset_type")
                .and_then(Value::as_str)
                .is_none_or(|kind| kind == "request_adapter_prompt_gepa/v1")
        })
        .map(|row| {
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
            let expected = row.get("expected_output").cloned().unwrap_or_else(|| {
                json!({
                    "content": "",
                    "tool_calls": []
                })
            });
            let expected_fields = expected_adapter_output_fields(&expected);
            example! {
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
                    .unwrap_or(true),
                "expected_output": "output" => expected.clone(),
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

fn expected_adapter_output_fields(expected: &Value) -> ExpectedRepairFields {
    let expected_calls = expected
        .get("tool_calls")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    ExpectedRepairFields {
        possible: true,
        confidence: 1.0,
        explanation: "Matched expected request-adapter output".to_string(),
        content: expected
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        tool_calls: Value::Array(normalize_tool_calls(&expected_calls)),
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
                0.65,
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
        FeedbackMetric::new(0.7, "Predicted non-placeholder user-facing content")
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

    #[test]
    fn converts_request_adapter_rows_to_gepa_examples() {
        let rows = vec![json!({
            "dataset_type": "request_adapter_prompt_gepa/v1",
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
        assert!(examples[0].get("request", None).as_str().is_some());
        assert!(examples[0].get("messages", None).as_str().is_some());
        assert_eq!(examples[0].get("parallel_tool_calls", None), json!(true));
        assert_eq!(
            examples[0].get("tool_calls", None),
            json!([{"name":"read","arguments":{"path":"README.md"}}])
        );
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
}
