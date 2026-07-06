use std::{
    collections::HashMap,
    panic::{self, AssertUnwindSafe},
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use dspy_rs::{
    adapter::Adapter, configure, example, Chat, ChatAdapter, Evaluator, Example, FeedbackEvaluator,
    FeedbackMetric, GEPACheckpoint, GEPAResult, LmUsage, Message, MetaSignature, Module,
    Optimizable, Predict, Prediction, Predictor, Signature, GEPA, LM,
};
use futures::FutureExt;
use indexmap::IndexMap;
use rig::tool::ToolDyn;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    artifacts::extract_instruction_from_path,
    config::UpstreamConfig,
    model_profile::{
        builtin_profiles, resolve_profile, DsrsHistoryFormat, ModelProfile, ProviderRouting,
    },
    normalizer::normalize_request,
    openai::{
        ChatCompletionRequest, ChatCompletionResponse, ChatMessage, OpenAiFunctionTool, OpenAiTool,
    },
    prompt_adapter::adapt_request,
    response_interpreter::{interpret_response, ResponseFailureKind, ToolIntent},
    upstream::{InboundAuth, UpstreamClient, UpstreamError},
};

pub const DEFAULT_GEPA_LM_MAX_TOKENS: u32 = ANTHROPIC_SONNET_MAX_OUTPUT_TOKENS;
pub const DEFAULT_GEPA_ROLE_BASE_URL: &str = "https://openrouter.ai/api/v1";
pub const DEFAULT_GEPA_REFLECTION_MODEL: &str = "anthropic/claude-sonnet-5";
pub const DEFAULT_GEPA_JUDGE_MODEL: &str = "anthropic/claude-sonnet-5";
pub const DEFAULT_GEPA_REFLECTION_TEMPERATURE: f32 = 1.0;
pub const DEFAULT_GEPA_JUDGE_TEMPERATURE: f32 = 0.0;
pub const DEFAULT_GEPA_TARGET_TIMEOUT_SECS: u64 = 420;
pub const DEFAULT_GEPA_SEED: u64 = 0;
const ANTHROPIC_SONNET_MAX_OUTPUT_TOKENS: u32 = 128_000;
const GEPA_TARGET_LM_MAX_ATTEMPTS: usize = 3;
const GEPA_JUDGE_LM_MAX_ATTEMPTS: usize = 3;

fn default_gepa_lm_max_tokens() -> u32 {
    DEFAULT_GEPA_LM_MAX_TOKENS
}

fn default_gepa_reflection_temperature() -> f32 {
    DEFAULT_GEPA_REFLECTION_TEMPERATURE
}

fn default_gepa_judge_temperature() -> f32 {
    DEFAULT_GEPA_JUDGE_TEMPERATURE
}

fn default_gepa_target_timeout_seconds() -> u64 {
    DEFAULT_GEPA_TARGET_TIMEOUT_SECS
}

fn default_gepa_seed() -> u64 {
    DEFAULT_GEPA_SEED
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
    pub validation_dataset_path: Option<PathBuf>,
    pub output_path: PathBuf,
    pub base_url: String,
    pub api_key: Option<String>,
    pub reflection_base_url: Option<String>,
    pub reflection_api_key: Option<String>,
    pub model: String,
    pub judge_base_url: Option<String>,
    pub judge_api_key: Option<String>,
    pub judge_model: String,
    pub target_model: Option<String>,
    pub profile: Option<String>,
    pub dataset_model_filter: Option<String>,
    pub dataset_profile_filter: Option<String>,
    pub profile_revision: Option<u32>,
    pub dsrs_history_format: Option<DsrsHistoryFormat>,
    pub target_provider: Option<ProviderRouting>,
    pub artifact_id: Option<String>,
    pub seed_artifact_path: Option<PathBuf>,
    pub iterations: usize,
    pub max_examples: usize,
    pub lm_max_tokens: u32,
    pub reflection_temperature: f32,
    pub judge_temperature: f32,
    pub target_timeout_seconds: u64,
    pub max_rollouts: Option<usize>,
    pub seed: u64,
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
    pub dataset_model_filter: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dataset_profile_filter: Option<String>,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub validation_dataset_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "is_zero_usize")]
    pub validation_examples_loaded: usize,
    #[serde(default = "default_gepa_lm_max_tokens")]
    pub lm_max_tokens: u32,
    #[serde(default = "default_gepa_reflection_temperature")]
    pub reflection_temperature: f32,
    #[serde(default = "default_gepa_judge_temperature")]
    pub judge_temperature: f32,
    #[serde(default = "default_gepa_target_timeout_seconds")]
    pub target_timeout_seconds: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_rollouts: Option<usize>,
    #[serde(default = "default_gepa_seed")]
    pub seed: u64,
    pub best_instruction: String,
    pub best_average_score: f32,
    pub total_rollouts: usize,
    pub total_lm_calls: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub artifact_warnings: Vec<ArtifactWarning>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub partial_checkpoint: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stopped_early_reason: Option<String>,
    pub output_path: PathBuf,
}

#[derive(Clone)]
struct GepaArtifactCheckpointSpec {
    artifact_id: String,
    artifact_type: String,
    signature: String,
    target_model: Option<String>,
    profile: Option<String>,
    dataset_model_filter: Option<String>,
    dataset_profile_filter: Option<String>,
    profile_revision: Option<u32>,
    dsrs_history_format: Option<DsrsHistoryFormat>,
    seed_artifact_path: Option<PathBuf>,
    optimizer_model: String,
    judge_model: String,
    examples_loaded: usize,
    validation_dataset_path: Option<PathBuf>,
    validation_examples_loaded: usize,
    lm_max_tokens: u32,
    reflection_temperature: f32,
    judge_temperature: f32,
    target_timeout_seconds: u64,
    max_rollouts: Option<usize>,
    seed: u64,
    output_path: PathBuf,
}

impl GepaArtifactCheckpointSpec {
    fn report(&self, checkpoint: &GEPACheckpoint) -> Option<GepaOptimizationReport> {
        let best_instruction = checkpoint.best_candidate.instruction.clone();
        if best_instruction.trim().is_empty() {
            return None;
        }
        let partial_checkpoint = !checkpoint.completed;
        let stopped_early_reason = checkpoint
            .stopped_early_reason
            .as_deref()
            .map(sanitize_gepa_infrastructure_message);
        Some(GepaOptimizationReport {
            artifact_id: self.artifact_id.clone(),
            artifact_type: self.artifact_type.clone(),
            signature: self.signature.clone(),
            target_model: self.target_model.clone(),
            profile: self.profile.clone(),
            dataset_model_filter: self.dataset_model_filter.clone(),
            dataset_profile_filter: self.dataset_profile_filter.clone(),
            profile_revision: self.profile_revision,
            dsrs_history_format: self.dsrs_history_format,
            seed_artifact_path: self.seed_artifact_path.clone(),
            optimizer_model: self.optimizer_model.clone(),
            judge_model: self.judge_model.clone(),
            created_at: Utc::now(),
            examples_loaded: self.examples_loaded,
            validation_dataset_path: self.validation_dataset_path.clone(),
            validation_examples_loaded: self.validation_examples_loaded,
            lm_max_tokens: self.lm_max_tokens,
            reflection_temperature: self.reflection_temperature,
            judge_temperature: self.judge_temperature,
            target_timeout_seconds: self.target_timeout_seconds,
            max_rollouts: self.max_rollouts,
            seed: self.seed,
            artifact_warnings: gepa_artifact_warnings(
                &self.artifact_type,
                &best_instruction,
                partial_checkpoint,
                stopped_early_reason.as_deref(),
            ),
            best_instruction,
            best_average_score: checkpoint.best_candidate.average_score(),
            total_rollouts: checkpoint.total_rollouts,
            total_lm_calls: checkpoint.total_lm_calls,
            partial_checkpoint,
            stopped_early_reason,
            output_path: self.output_path.clone(),
        })
    }
}

fn gepa_artifact_checkpoint_callback(
    spec: GepaArtifactCheckpointSpec,
) -> Arc<dyn Fn(GEPACheckpoint) -> Result<()> + Send + Sync> {
    Arc::new(move |checkpoint| {
        let Some(report) = spec.report(&checkpoint) else {
            return Ok(());
        };
        write_gepa_checkpoint_report(&report)?;
        eprintln!(
            "  Saved GEPA best artifact checkpoint to {} (score {:.3}, generation {}, partial={})",
            report.output_path.display(),
            report.best_average_score,
            checkpoint.generation,
            report.partial_checkpoint
        );
        Ok(())
    })
}

fn write_gepa_checkpoint_report(report: &GepaOptimizationReport) -> Result<()> {
    if let Some(parent) = report.output_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    std::fs::write(&report.output_path, serde_json::to_vec_pretty(report)?)
        .with_context(|| format!("failed to write {}", report.output_path.display()))
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

fn gepa_artifact_warnings(
    artifact_type: &str,
    instruction: &str,
    partial_checkpoint: bool,
    stopped_early_reason: Option<&str>,
) -> Vec<ArtifactWarning> {
    let mut warnings = artifact_instruction_warnings(artifact_type, instruction);
    if partial_checkpoint {
        let matched = stopped_early_reason
            .map(sanitize_gepa_infrastructure_message)
            .unwrap_or_else(|| "run still in progress".to_string());
        warnings.push(ArtifactWarning {
            code: "gepa_partial_checkpoint".to_string(),
            message: "GEPA artifact was written as a best-so-far checkpoint before the configured run completed; review before promotion".to_string(),
            matched: Some(matched),
        });
    }
    warnings
}

fn sanitize_gepa_infrastructure_message(message: &str) -> String {
    let lower = message.to_ascii_lowercase();
    if lower.contains("insufficient credits")
        || lower.contains("requires more credits")
        || lower.contains("monthly limit")
        || lower.contains("\"code\":402")
        || lower.contains("\"code\":403")
        || lower.contains("http 402")
        || lower.contains("http 403")
    {
        return "OpenRouter returned a credit or key-limit error; add credits, raise the key limit, or lower max_tokens before continuing.".to_string();
    }

    truncate_for_feedback(&redact_urls(message))
}

fn redact_urls(message: &str) -> String {
    message
        .split_whitespace()
        .map(|token| {
            if token.starts_with("http://") || token.starts_with("https://") {
                "[redacted-url]"
            } else {
                token
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
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
type GepaFatalErrors = Arc<Mutex<Vec<String>>>;

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
    target_client: Option<UpstreamClient>,
    target_model: Option<String>,
    target_timeout_seconds: u64,
    target_provider: Option<ProviderRouting>,
    judge_lm: Option<LM>,
    judge_cases: GepaJudgeCases,
    fatal_errors: GepaFatalErrors,
}

impl Default for RequestAdapterPromptProgram {
    fn default() -> Self {
        Self {
            predictor: Predict::new(RequestAdapterPromptSignature::new()),
            target_client: None,
            target_model: None,
            target_timeout_seconds: DEFAULT_GEPA_TARGET_TIMEOUT_SECS,
            target_provider: None,
            judge_lm: None,
            judge_cases: Arc::new(HashMap::new()),
            fatal_errors: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl RequestAdapterPromptProgram {
    fn runtime(
        target_client: UpstreamClient,
        target_model: String,
        target_timeout_seconds: u64,
        target_provider: Option<ProviderRouting>,
        initial_instruction: Option<String>,
        judge_lm: LM,
        judge_cases: GepaJudgeCases,
        fatal_errors: GepaFatalErrors,
    ) -> Self {
        let mut signature = RequestAdapterPromptSignature::new();
        if let Some(instruction) =
            initial_instruction.filter(|instruction| !instruction.trim().is_empty())
        {
            let _ = signature.update_instruction(instruction);
        }
        Self {
            predictor: Predict::new(signature),
            target_client: Some(target_client),
            target_model: Some(target_model),
            target_timeout_seconds,
            target_provider,
            judge_lm: Some(judge_lm),
            judge_cases,
            fatal_errors,
        }
    }

    async fn forward_runtime(
        &self,
        inputs: Example,
        target_client: &UpstreamClient,
        target_model: &str,
    ) -> Result<Prediction> {
        let mut profile = ModelProfile::default_balanced();
        profile.name = string_input(&inputs, "profile", "request-adapter-gepa");
        profile.revision = u32_input(&inputs, "profile_revision").unwrap_or(1);
        profile.source = "gepa".to_string();
        profile.tool_instruction = self.predictor.signature.instruction();
        profile.dsrs_history_format =
            dsrs_history_format_input(&inputs).unwrap_or(DsrsHistoryFormat::AppendOnly);
        profile.provider = self.target_provider.clone();

        let request = chat_completion_request_from_adapter_example(&inputs)?;
        let normalized = normalize_request(request)?;
        let mut request = adapt_request(&normalized, &profile)?.upstream_request;
        request.model = target_model.to_string();

        let response =
            match call_gepa_target(target_client, &request, self.target_timeout_seconds).await {
                Ok(response) => response,
                Err(GepaTargetCallFailure::Scoreable { event, reason }) => {
                    return Ok(failed_request_adapter_target_prediction(event, reason));
                }
                Err(GepaTargetCallFailure::Fatal(error)) => {
                    return Err(error).context("request-adapter GEPA target LM call failed");
                }
            };
        let interpreted = interpret_response(&response, &normalized.tools);
        let raw_output = raw_assistant_output(&response);
        let calls = interpreted_tool_calls(&interpreted.tool_intents);
        let mut parser_events = interpreted.parse_events;
        parser_events.extend(interpreted.failures.iter().map(|failure| {
            format!(
                "{}: {}",
                response_failure_kind_label(failure.kind),
                failure.detail
            )
        }));

        Ok(Prediction::new(
            HashMap::from([
                (
                    "content".to_string(),
                    Value::String(interpreted.content.unwrap_or_default()),
                ),
                (
                    "tool_calls".to_string(),
                    Value::Array(normalize_tool_calls(&calls)),
                ),
                ("raw_output".to_string(), Value::String(raw_output)),
                (
                    "parser_events".to_string(),
                    Value::Array(parser_events.into_iter().map(Value::String).collect()),
                ),
            ]),
            lm_usage_from_openai_usage(response.usage.as_ref()),
        ))
    }
}

#[cfg(test)]
fn gepa_empty_target_response_error(error: &anyhow::Error) -> bool {
    format!("{error:#}").contains("Response contained no message or tool call (empty)")
}

#[cfg(test)]
fn gepa_target_response_decode_error(error: &anyhow::Error) -> bool {
    let details = format!("{error:#}");
    details.contains("JsonError")
        && details.contains("data did not match any variant of untagged enum ApiResponse")
}

fn gepa_target_retry_delay(attempt: usize) -> Duration {
    Duration::from_millis(500 * attempt as u64)
}

fn gepa_target_lm_timeout(timeout_seconds: u64) -> Duration {
    Duration::from_secs(timeout_seconds)
}

fn failed_request_adapter_target_prediction(event: &'static str, reason: String) -> Prediction {
    Prediction::new(
        HashMap::from([
            ("content".to_string(), Value::String(String::new())),
            ("tool_calls".to_string(), Value::Array(Vec::new())),
            ("raw_output".to_string(), Value::String(String::new())),
            (
                "parser_events".to_string(),
                json!([event, format!("target_lm_error: {reason}")]),
            ),
        ]),
        LmUsage::default(),
    )
}

enum GepaTargetCallFailure {
    Scoreable { event: &'static str, reason: String },
    Fatal(anyhow::Error),
}

async fn call_gepa_target(
    target_client: &UpstreamClient,
    request: &ChatCompletionRequest,
    timeout_seconds: u64,
) -> std::result::Result<ChatCompletionResponse, GepaTargetCallFailure> {
    for attempt in 1..=GEPA_TARGET_LM_MAX_ATTEMPTS {
        match tokio::time::timeout(
            gepa_target_lm_timeout(timeout_seconds),
            target_client.chat_completions(request, &InboundAuth::default()),
        )
        .await
        {
            Err(_) if attempt < GEPA_TARGET_LM_MAX_ATTEMPTS => {
                eprintln!(
                    "GEPA target LM call timed out on attempt {attempt}/{GEPA_TARGET_LM_MAX_ATTEMPTS}; retrying"
                );
                tokio::time::sleep(gepa_target_retry_delay(attempt)).await;
            }
            Err(_) => {
                return Err(GepaTargetCallFailure::Scoreable {
                    event: "target LM call timed out after retries",
                    reason: format!("target LM call exceeded {}s timeout", timeout_seconds),
                });
            }
            Ok(Ok(response)) => return Ok(response),
            Ok(Err(error)) if attempt < GEPA_TARGET_LM_MAX_ATTEMPTS => {
                let message = sanitize_gepa_infrastructure_message(&format!("{error:#}"));
                eprintln!(
                    "GEPA target LM call failed on attempt {attempt}/{GEPA_TARGET_LM_MAX_ATTEMPTS}; retrying: {}",
                    message
                );
                tokio::time::sleep(gepa_target_retry_delay(attempt)).await;
            }
            Ok(Err(error)) if gepa_scoreable_upstream_error(&error) => {
                return Err(GepaTargetCallFailure::Scoreable {
                    event: "target LM call failed after retries",
                    reason: sanitize_gepa_infrastructure_message(&format!("{error:#}")),
                });
            }
            Ok(Err(error)) => return Err(GepaTargetCallFailure::Fatal(error.into())),
        }
    }

    Err(GepaTargetCallFailure::Fatal(anyhow::anyhow!(
        "GEPA target LM retry loop ended unexpectedly"
    )))
}

fn gepa_scoreable_upstream_error(error: &UpstreamError) -> bool {
    let _ = error;
    false
}

fn raw_assistant_output(response: &crate::openai::ChatCompletionResponse) -> String {
    response
        .choices
        .first()
        .and_then(|choice| {
            choice
                .message
                .content_text()
                .or_else(|| choice.message.reasoning_text())
        })
        .unwrap_or_default()
}

fn chat_to_openai_messages(chat: Chat) -> Vec<ChatMessage> {
    chat.messages
        .into_iter()
        .map(|message| match message {
            Message::System { content } => ChatMessage::new("system", content),
            Message::User { content } => ChatMessage::new("user", content),
            Message::Assistant { content } => ChatMessage::new("assistant", content),
        })
        .collect()
}

fn interpreted_tool_calls(intents: &[ToolIntent]) -> Vec<Value> {
    intents
        .iter()
        .map(|intent| {
            json!({
                "name": intent.name,
                "arguments": intent.arguments.clone().unwrap_or_else(|| json!({}))
            })
        })
        .collect()
}

fn response_failure_kind_label(kind: ResponseFailureKind) -> &'static str {
    match kind {
        ResponseFailureKind::NoChoices => "no_choices",
        ResponseFailureKind::NativeMalformedJsonArguments => "native_malformed_json_arguments",
        ResponseFailureKind::DsrsContractViolation => "dsrs_contract_violation",
        ResponseFailureKind::DsrsContentOutsideTaggedFields => "dsrs_content_outside_tagged_fields",
        ResponseFailureKind::DsrsInvalidToolCallsJson => "dsrs_invalid_tool_calls_json",
        ResponseFailureKind::DsrsInvalidToolCallsShape => "dsrs_invalid_tool_calls_shape",
        ResponseFailureKind::EmptyDsrsOutput => "empty_dsrs_output",
        ResponseFailureKind::DsrsPlaceholderOnly => "dsrs_placeholder_only",
        ResponseFailureKind::EmptyAssistantOutput => "empty_assistant_output",
        ResponseFailureKind::TemplateLeak => "template_leak",
        ResponseFailureKind::PromptEcho => "prompt_echo",
        ResponseFailureKind::PrematureToolStop => "premature_tool_stop",
        ResponseFailureKind::MalformedKnownToolCall => "malformed_known_tool_call",
        ResponseFailureKind::UntaggedDsrsLikeOutput => "untagged_dsrs_like_output",
        ResponseFailureKind::SchemaViolation => "schema_violation",
        ResponseFailureKind::UnknownTool => "unknown_tool",
    }
}

fn lm_usage_from_openai_usage(usage: Option<&Value>) -> LmUsage {
    let Some(usage) = usage else {
        return LmUsage::default();
    };
    LmUsage {
        prompt_tokens: usage
            .get("prompt_tokens")
            .and_then(Value::as_u64)
            .unwrap_or_default(),
        completion_tokens: usage
            .get("completion_tokens")
            .and_then(Value::as_u64)
            .unwrap_or_default(),
        total_tokens: usage
            .get("total_tokens")
            .and_then(Value::as_u64)
            .unwrap_or_default(),
    }
}

impl Module for RequestAdapterPromptProgram {
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        if let (Some(target_client), Some(target_model)) = (&self.target_client, &self.target_model)
        {
            return self
                .forward_runtime(inputs, target_client, target_model)
                .await;
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
        let judged = judge_request_adapter_prediction(
            judge_lm,
            example,
            prediction,
            self.judge_cases.get(&case_id),
            &predicted,
            deterministic,
        )
        .await;
        record_gepa_fatal_feedback(&self.fatal_errors, &judged);
        log_gepa_debug("request_adapter", &case_id, &predicted, &judged);
        judged
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
    target_client: Option<UpstreamClient>,
    target_model: Option<String>,
    target_timeout_seconds: u64,
    target_provider: Option<ProviderRouting>,
    judge_lm: Option<LM>,
    judge_cases: GepaJudgeCases,
    fatal_errors: GepaFatalErrors,
}

impl Default for CorrectionPromptProgram {
    fn default() -> Self {
        Self {
            predictor: Predict::new(CorrectionPromptSignature::new()),
            target_client: None,
            target_model: None,
            target_timeout_seconds: DEFAULT_GEPA_TARGET_TIMEOUT_SECS,
            target_provider: None,
            judge_lm: None,
            judge_cases: Arc::new(HashMap::new()),
            fatal_errors: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl CorrectionPromptProgram {
    fn with_target_and_judge(
        target_client: UpstreamClient,
        target_model: String,
        target_timeout_seconds: u64,
        target_provider: Option<ProviderRouting>,
        judge_lm: LM,
        judge_cases: GepaJudgeCases,
        fatal_errors: GepaFatalErrors,
    ) -> Self {
        Self {
            predictor: Predict::new(CorrectionPromptSignature::new()),
            target_client: Some(target_client),
            target_model: Some(target_model),
            target_timeout_seconds,
            target_provider,
            judge_lm: Some(judge_lm),
            judge_cases,
            fatal_errors,
        }
    }

    async fn forward_target(
        &self,
        inputs: Example,
        target_client: &UpstreamClient,
        target_model: &str,
    ) -> Result<Prediction> {
        let chat = ChatAdapter.format(self.predictor.signature.as_ref(), inputs);
        let mut request = ChatCompletionRequest {
            model: target_model.to_string(),
            messages: chat_to_openai_messages(chat),
            tools: None,
            tool_choice: None,
            parallel_tool_calls: None,
            stream: Some(false),
            temperature: Some(0.2),
            top_p: None,
            max_tokens: None,
            max_completion_tokens: None,
            response_format: None,
            extra: serde_json::Map::new(),
        };
        if let Some(provider) = &self.target_provider {
            if !provider.is_empty() {
                request
                    .extra
                    .insert("provider".to_string(), serde_json::to_value(provider)?);
            }
        }

        let response =
            match call_gepa_target(target_client, &request, self.target_timeout_seconds).await {
                Ok(response) => response,
                Err(GepaTargetCallFailure::Scoreable { event, reason }) => {
                    return Ok(failed_correction_target_prediction(event, reason));
                }
                Err(GepaTargetCallFailure::Fatal(error)) => return Err(error),
            };
        let raw_output = raw_assistant_output(&response);
        if raw_output.trim().is_empty() {
            return Ok(failed_correction_target_prediction(
                "target LM returned empty assistant response",
                "assistant message had empty content and empty reasoning".to_string(),
            ));
        }
        let parsed = match catch_unwind_without_panic_hook(AssertUnwindSafe(|| {
            ChatAdapter.parse_response(
                self.predictor.signature.as_ref(),
                Message::assistant(raw_output.clone()),
            )
        })) {
            Ok(parsed) => parsed,
            Err(_) => {
                return Ok(failed_correction_target_prediction(
                    "target LM returned unparsable DSRs correction output",
                    "typed DSRs fields could not be parsed by dspy-rs ChatAdapter".to_string(),
                ));
            }
        };
        Ok(Prediction::new(
            parsed,
            lm_usage_from_openai_usage(response.usage.as_ref()),
        ))
    }
}

impl Module for CorrectionPromptProgram {
    async fn forward(&self, inputs: Example) -> Result<Prediction> {
        if let (Some(target_client), Some(target_model)) = (&self.target_client, &self.target_model)
        {
            return self
                .forward_target(inputs, target_client, target_model)
                .await;
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

fn failed_correction_target_prediction(event: &'static str, reason: String) -> Prediction {
    Prediction::new(
        HashMap::from([
            ("possible".to_string(), Value::Bool(false)),
            ("confidence".to_string(), json!(0.0)),
            (
                "explanation".to_string(),
                Value::String(format!("{event}: {}", truncate_for_feedback(&reason))),
            ),
            ("content".to_string(), Value::String(String::new())),
            ("tool_calls".to_string(), Value::Array(Vec::new())),
            ("target_error".to_string(), Value::String(reason)),
        ]),
        LmUsage::default(),
    )
}

fn catch_unwind_without_panic_hook<F, R>(f: F) -> std::thread::Result<R>
where
    F: FnOnce() -> R + panic::UnwindSafe,
{
    static PANIC_HOOK_LOCK: OnceLock<Mutex<()>> = OnceLock::new();

    let _guard = PANIC_HOOK_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let previous_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));
    let result = panic::catch_unwind(f);
    panic::set_hook(previous_hook);
    result
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
        let judged = judge_correction_prediction(
            judge_lm,
            example,
            prediction,
            self.judge_cases.get(&case_id),
            &predicted,
            deterministic,
        )
        .await;
        record_gepa_fatal_feedback(&self.fatal_errors, &judged);
        log_gepa_debug("correction_agent", &case_id, &predicted, &judged);
        judged
    }
}

pub async fn optimize_correction_prompt(
    config: GepaOptimizationConfig,
) -> Result<GepaOptimizationReport> {
    ensure_gepa_temperature("reflection temperature", config.reflection_temperature)?;
    ensure_gepa_temperature("judge temperature", config.judge_temperature)?;
    let Some(target_api_key) = config.api_key.clone() else {
        anyhow::bail!(
            "OPENROUTER_API_KEY or target --api-key equivalent is required for GEPA target-model calls"
        );
    };

    let rows = filter_gepa_dataset_rows(
        read_dataset_rows(&config.dataset_path).await?,
        config.dataset_model_filter.as_deref(),
        config.dataset_profile_filter.as_deref(),
    );
    let rows: Vec<Value> = rows.into_iter().take(config.max_examples).collect();
    let examples = traces_to_gepa_examples(&rows);
    if examples.is_empty() {
        anyhow::bail!("dataset contained no optimization examples");
    }
    let validation_rows = filter_gepa_dataset_rows(
        read_validation_dataset_rows(config.validation_dataset_path.as_ref()).await?,
        config.dataset_model_filter.as_deref(),
        config.dataset_profile_filter.as_deref(),
    );
    let validation_examples = optional_examples(traces_to_gepa_examples(&validation_rows));
    let target_model = required_gepa_target_model(&config, "correction-agent GEPA")?;
    ensure_gepa_model_role_separation(
        "correction-agent GEPA",
        &config.model,
        &config.judge_model,
        &target_model,
    )?;
    let judge_cases = gepa_judge_cases_for_splits(
        &rows,
        &validation_rows,
        "expected_repair",
        "expected_repair_policy",
    )?;

    let optimizer_lm = build_gepa_lm(
        "correction-agent GEPA reflection/proposal",
        &config.model,
        config.reflection_base_url.as_deref(),
        config.reflection_api_key.clone(),
        config.reflection_temperature,
        Some(config.lm_max_tokens),
    )
    .await?;
    let target_client = UpstreamClient::new(UpstreamConfig {
        base_url: config.base_url.clone(),
        api_key: Some(target_api_key),
        timeout_seconds: config.target_timeout_seconds,
    })
    .context("failed to build correction-agent GEPA target upstream client")?;
    let judge_lm = build_gepa_lm(
        "correction-agent GEPA judge",
        &config.judge_model,
        config.judge_base_url.as_deref(),
        config.judge_api_key.clone(),
        config.judge_temperature,
        Some(config.lm_max_tokens),
    )
    .await?;
    let artifact_type = "correction_agent_instruction".to_string();
    let artifact_id = config.artifact_id.clone().unwrap_or_else(|| {
        default_correction_artifact_id(config.profile.as_deref(), config.target_model.as_deref())
    });
    let checkpoint_callback = gepa_artifact_checkpoint_callback(GepaArtifactCheckpointSpec {
        artifact_id: artifact_id.clone(),
        artifact_type: artifact_type.clone(),
        signature: "correct_malformed_tool_response/v1".to_string(),
        target_model: Some(target_model.clone()),
        profile: config.profile.clone(),
        dataset_model_filter: config.dataset_model_filter.clone(),
        dataset_profile_filter: config.dataset_profile_filter.clone(),
        profile_revision: config.profile_revision,
        dsrs_history_format: None,
        seed_artifact_path: config.seed_artifact_path.clone(),
        optimizer_model: config.model.clone(),
        judge_model: config.judge_model.clone(),
        examples_loaded: examples.len(),
        validation_dataset_path: config.validation_dataset_path.clone(),
        validation_examples_loaded: validation_examples.as_ref().map_or(0, Vec::len),
        lm_max_tokens: config.lm_max_tokens,
        reflection_temperature: config.reflection_temperature,
        judge_temperature: config.judge_temperature,
        target_timeout_seconds: config.target_timeout_seconds,
        max_rollouts: config.max_rollouts,
        seed: config.seed,
        output_path: config.output_path.clone(),
    });

    let gepa = GEPA::builder()
        .num_iterations(config.iterations)
        .minibatch_size(examples.len().clamp(1, 3))
        .temperature(config.reflection_temperature)
        .track_stats(true)
        .maybe_prompt_model(Some(optimizer_lm.clone()))
        .maybe_valset(validation_examples.clone())
        .maybe_max_rollouts(config.max_rollouts)
        .maybe_checkpoint_callback(Some(checkpoint_callback))
        .seed(config.seed)
        .build();

    let fatal_errors = Arc::new(Mutex::new(Vec::new()));
    let mut program = CorrectionPromptProgram::with_target_and_judge(
        target_client,
        target_model.clone(),
        config.target_timeout_seconds,
        config.target_provider.clone(),
        judge_lm,
        judge_cases,
        fatal_errors.clone(),
    );
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
    ensure_no_gepa_fatal_errors("GEPA correction prompt optimization", &fatal_errors)?;

    if let Some(parent) = config.output_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let best_instruction = result.best_candidate.instruction.clone();
    if best_instruction.trim().is_empty() {
        anyhow::bail!("GEPA correction-prompt optimization produced an empty best instruction; refusing to write an unusable artifact");
    }
    let stopped_early_reason = result
        .stopped_early_reason
        .as_deref()
        .map(sanitize_gepa_infrastructure_message);
    let report = GepaOptimizationReport {
        artifact_id,
        artifact_type: artifact_type.clone(),
        signature: "correct_malformed_tool_response/v1".to_string(),
        target_model: Some(target_model),
        profile: config.profile.clone(),
        dataset_model_filter: config.dataset_model_filter.clone(),
        dataset_profile_filter: config.dataset_profile_filter.clone(),
        profile_revision: config.profile_revision,
        dsrs_history_format: None,
        seed_artifact_path: config.seed_artifact_path.clone(),
        optimizer_model: config.model.clone(),
        judge_model: config.judge_model.clone(),
        created_at: Utc::now(),
        examples_loaded: examples.len(),
        validation_dataset_path: config.validation_dataset_path.clone(),
        validation_examples_loaded: validation_examples.as_ref().map_or(0, Vec::len),
        lm_max_tokens: config.lm_max_tokens,
        reflection_temperature: config.reflection_temperature,
        judge_temperature: config.judge_temperature,
        target_timeout_seconds: config.target_timeout_seconds,
        max_rollouts: config.max_rollouts,
        seed: config.seed,
        artifact_warnings: gepa_artifact_warnings(
            &artifact_type,
            &best_instruction,
            stopped_early_reason.is_some(),
            stopped_early_reason.as_deref(),
        ),
        best_instruction,
        best_average_score: result.best_candidate.average_score(),
        total_rollouts: result.total_rollouts,
        total_lm_calls: result.total_lm_calls,
        partial_checkpoint: stopped_early_reason.is_some(),
        stopped_early_reason,
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
    ensure_gepa_temperature("reflection temperature", config.reflection_temperature)?;
    ensure_gepa_temperature("judge temperature", config.judge_temperature)?;
    let Some(target_api_key) = config.api_key.clone() else {
        anyhow::bail!(
            "OPENROUTER_API_KEY or target --api-key equivalent is required for GEPA target-model calls"
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
    let validation_rows =
        read_validation_dataset_rows(config.validation_dataset_path.as_ref()).await?;
    let validation_examples = optional_examples(request_adapter_rows_to_gepa_examples_with_format(
        &validation_rows,
        config.dsrs_history_format,
        config.profile.as_deref(),
        config.profile_revision,
    ));
    let runtime_model = required_gepa_target_model(&config, "request-adapter GEPA")?;
    ensure_gepa_model_role_separation(
        "request-adapter GEPA",
        &config.model,
        &config.judge_model,
        &runtime_model,
    )?;
    let judge_cases = gepa_judge_cases_for_splits(
        &rows,
        &validation_rows,
        "expected_output",
        "expected_adapter_policy",
    )?;

    let optimizer_lm = build_gepa_lm(
        "request-adapter GEPA reflection/proposal",
        &config.model,
        config.reflection_base_url.as_deref(),
        config.reflection_api_key.clone(),
        config.reflection_temperature,
        Some(config.lm_max_tokens),
    )
    .await?;
    let target_client = UpstreamClient::new(UpstreamConfig {
        base_url: config.base_url.clone(),
        api_key: Some(target_api_key),
        timeout_seconds: config.target_timeout_seconds,
    })
    .context("failed to build request-adapter GEPA target upstream client")?;
    let judge_lm = build_gepa_lm(
        "request-adapter GEPA judge",
        &config.judge_model,
        config.judge_base_url.as_deref(),
        config.judge_api_key.clone(),
        config.judge_temperature,
        Some(config.lm_max_tokens),
    )
    .await?;
    configure(optimizer_lm.clone(), GepaJsonAdapter);
    let artifact_type = "request_adapter_instruction".to_string();
    let artifact_id = config.artifact_id.clone().unwrap_or_else(|| {
        default_request_adapter_artifact_id(
            config.profile.as_deref(),
            config.target_model.as_deref(),
        )
    });
    let checkpoint_callback = gepa_artifact_checkpoint_callback(GepaArtifactCheckpointSpec {
        artifact_id: artifact_id.clone(),
        artifact_type: artifact_type.clone(),
        signature: "openai_tool_use_contract_profile_guidance/v1".to_string(),
        target_model: Some(runtime_model.clone()),
        profile: config.profile.clone(),
        dataset_model_filter: config.dataset_model_filter.clone(),
        dataset_profile_filter: config.dataset_profile_filter.clone(),
        profile_revision: config.profile_revision,
        dsrs_history_format: config.dsrs_history_format,
        seed_artifact_path: config.seed_artifact_path.clone(),
        optimizer_model: config.model.clone(),
        judge_model: config.judge_model.clone(),
        examples_loaded: examples.len(),
        validation_dataset_path: config.validation_dataset_path.clone(),
        validation_examples_loaded: validation_examples.as_ref().map_or(0, Vec::len),
        lm_max_tokens: config.lm_max_tokens,
        reflection_temperature: config.reflection_temperature,
        judge_temperature: config.judge_temperature,
        target_timeout_seconds: config.target_timeout_seconds,
        max_rollouts: config.max_rollouts,
        seed: config.seed,
        output_path: config.output_path.clone(),
    });

    let gepa = GEPA::builder()
        .num_iterations(config.iterations)
        .minibatch_size(examples.len().clamp(1, 3))
        .temperature(config.reflection_temperature)
        .track_stats(true)
        .maybe_valset(validation_examples.clone())
        .maybe_max_rollouts(config.max_rollouts)
        .maybe_checkpoint_callback(Some(checkpoint_callback))
        .seed(config.seed)
        .build();

    let initial_instruction = request_adapter_initial_instruction(&config, &runtime_model).await?;
    let fatal_errors = Arc::new(Mutex::new(Vec::new()));
    let mut program = RequestAdapterPromptProgram::runtime(
        target_client,
        runtime_model.clone(),
        config.target_timeout_seconds,
        config.target_provider.clone(),
        initial_instruction,
        judge_lm,
        judge_cases,
        fatal_errors.clone(),
    );
    let result: GEPAResult = gepa
        .compile_with_feedback(&mut program, examples.clone())
        .await
        .context("GEPA request-adapter prompt optimization failed")?;
    ensure_no_gepa_fatal_errors("GEPA request-adapter prompt optimization", &fatal_errors)?;

    if let Some(parent) = config.output_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let best_instruction = result.best_candidate.instruction.clone();
    if best_instruction.trim().is_empty() {
        anyhow::bail!("GEPA request-adapter optimization produced an empty best instruction; refusing to write an unusable artifact");
    }
    let stopped_early_reason = result
        .stopped_early_reason
        .as_deref()
        .map(sanitize_gepa_infrastructure_message);
    let report = GepaOptimizationReport {
        artifact_id,
        artifact_type: artifact_type.clone(),
        signature: "openai_tool_use_contract_profile_guidance/v1".to_string(),
        target_model: Some(runtime_model),
        profile: config.profile.clone(),
        dataset_model_filter: config.dataset_model_filter.clone(),
        dataset_profile_filter: config.dataset_profile_filter.clone(),
        profile_revision: config.profile_revision,
        dsrs_history_format: config.dsrs_history_format,
        seed_artifact_path: config.seed_artifact_path.clone(),
        optimizer_model: config.model.clone(),
        created_at: Utc::now(),
        examples_loaded: examples.len(),
        validation_dataset_path: config.validation_dataset_path.clone(),
        validation_examples_loaded: validation_examples.as_ref().map_or(0, Vec::len),
        lm_max_tokens: config.lm_max_tokens,
        reflection_temperature: config.reflection_temperature,
        judge_temperature: config.judge_temperature,
        target_timeout_seconds: config.target_timeout_seconds,
        max_rollouts: config.max_rollouts,
        seed: config.seed,
        judge_model: config.judge_model.clone(),
        artifact_warnings: gepa_artifact_warnings(
            &artifact_type,
            &best_instruction,
            stopped_early_reason.is_some(),
            stopped_early_reason.as_deref(),
        ),
        best_instruction,
        best_average_score: result.best_candidate.average_score(),
        total_rollouts: result.total_rollouts,
        total_lm_calls: result.total_lm_calls,
        partial_checkpoint: stopped_early_reason.is_some(),
        stopped_early_reason,
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

async fn build_gepa_lm(
    role: &str,
    model: &str,
    base_url: Option<&str>,
    api_key: Option<String>,
    temperature: f32,
    max_tokens: Option<u32>,
) -> Result<LM> {
    if let Some(max_tokens) = max_tokens {
        ensure_gepa_lm_max_tokens_supported(role, model, max_tokens)?;
    }

    let base_url = base_url.map(str::trim).filter(|value| !value.is_empty());
    let api_key = api_key
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty());
    let model = gepa_lm_wire_model(model, base_url);

    match (base_url, api_key, max_tokens) {
        (Some(base_url), Some(api_key), Some(max_tokens)) => {
            LM::builder()
                .base_url(base_url.to_string())
                .api_key(api_key)
                .model(model.clone())
                .temperature(temperature)
                .max_tokens(max_tokens)
                .build()
                .await
        }
        (Some(base_url), Some(api_key), None) => {
            LM::builder()
                .base_url(base_url.to_string())
                .api_key(api_key)
                .model(model.clone())
                .temperature(temperature)
                .build()
                .await
        }
        (Some(base_url), None, Some(max_tokens)) => {
            LM::builder()
                .base_url(base_url.to_string())
                .model(model.clone())
                .temperature(temperature)
                .max_tokens(max_tokens)
                .build()
                .await
        }
        (Some(base_url), None, None) => {
            LM::builder()
                .base_url(base_url.to_string())
                .model(model.clone())
                .temperature(temperature)
                .build()
                .await
        }
        (None, Some(api_key), Some(max_tokens)) => {
            LM::builder()
                .api_key(api_key)
                .model(model.clone())
                .temperature(temperature)
                .max_tokens(max_tokens)
                .build()
                .await
        }
        (None, Some(api_key), None) => {
            LM::builder()
                .api_key(api_key)
                .model(model.clone())
                .temperature(temperature)
                .build()
                .await
        }
        (None, None, Some(max_tokens)) => {
            LM::builder()
                .model(model.clone())
                .temperature(temperature)
                .max_tokens(max_tokens)
                .build()
                .await
        }
        (None, None, None) => {
            LM::builder()
                .model(model.clone())
                .temperature(temperature)
                .build()
                .await
        }
    }
    .with_context(|| format!("failed to build {role} LM for model {model:?}"))
}

fn gepa_lm_wire_model(model: &str, base_url: Option<&str>) -> String {
    let trimmed = model.trim();
    if base_url
        .unwrap_or_default()
        .to_ascii_lowercase()
        .contains("openrouter.ai")
    {
        trimmed
            .strip_prefix("openrouter:")
            .unwrap_or(trimmed)
            .to_string()
    } else {
        trimmed.to_string()
    }
}

fn ensure_gepa_lm_max_tokens_supported(role: &str, model: &str, max_tokens: u32) -> Result<()> {
    if let Some(limit) = known_gepa_lm_max_tokens(model) {
        if max_tokens > limit {
            anyhow::bail!(
                "{role} --lm-max-tokens {max_tokens} exceeds the known provider cap {limit} for model {model:?}; lower --lm-max-tokens before starting live GEPA calls"
            );
        }
    }
    Ok(())
}

fn ensure_gepa_temperature(label: &str, temperature: f32) -> Result<()> {
    if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
        anyhow::bail!("GEPA {label} must be a finite value between 0.0 and 2.0, got {temperature}");
    }
    Ok(())
}

fn known_gepa_lm_max_tokens(model: &str) -> Option<u32> {
    match providerless_model_id(model).as_str() {
        "claude-sonnet-4-6" | "claude-sonnet-5" => Some(ANTHROPIC_SONNET_MAX_OUTPUT_TOKENS),
        _ => None,
    }
}

fn providerless_model_id(model: &str) -> String {
    let normalized = model.trim().to_ascii_lowercase().replace('.', "-");
    let mut model = normalized.as_str();
    if let Some(rest) = model.strip_prefix("openrouter:") {
        model = rest;
    }
    if let Some(rest) = model.strip_prefix("codex:") {
        model = rest;
    }
    if let Some(rest) = model
        .strip_prefix("anthropic:")
        .or_else(|| model.strip_prefix("anthropic/"))
    {
        model = rest;
    }
    if let Some((model_without_options, _)) = model.split_once('@') {
        model = model_without_options;
    }
    model.to_string()
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
    normalized_model_id(left) == normalized_model_id(right)
}

fn normalized_model_id(model: &str) -> String {
    providerless_model_id(model).replacen(':', "/", 1)
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

#[cfg(test)]
fn gepa_judge_cases(
    dataset_rows: &[Value],
    expected_key: &str,
    policy_key: &str,
) -> Result<GepaJudgeCases> {
    gepa_judge_cases_for_splits(dataset_rows, &[], expected_key, policy_key)
}

fn gepa_judge_cases_for_splits(
    train_rows: &[Value],
    validation_rows: &[Value],
    expected_key: &str,
    policy_key: &str,
) -> Result<GepaJudgeCases> {
    let mut cases = HashMap::new();
    insert_gepa_judge_cases(&mut cases, train_rows, "training", expected_key, policy_key)?;
    insert_gepa_judge_cases(
        &mut cases,
        validation_rows,
        "validation",
        expected_key,
        policy_key,
    )?;
    Ok(Arc::new(cases))
}

fn insert_gepa_judge_cases(
    cases: &mut HashMap<String, GepaJudgeCase>,
    dataset_rows: &[Value],
    split: &str,
    expected_key: &str,
    policy_key: &str,
) -> Result<()> {
    for (index, row) in dataset_rows.iter().enumerate() {
        let case_id = gepa_case_id(row, index);
        let expected = row.get(expected_key).cloned().with_context(|| {
            format!("GEPA {split} dataset row {index} is missing required label {expected_key:?}")
        })?;
        if cases
            .insert(
                case_id.clone(),
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
            )
            .is_some()
        {
            anyhow::bail!(
                "GEPA {split} dataset row {index} produced duplicate case_id {case_id:?}; trace_id plus row index must be unique across train and validation splits"
            );
        }
    }
    Ok(())
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
        "scoring_contract": {
            "valid_assistant_shapes": [
                "content only with [] tool_calls",
                "tool_calls only with empty content",
                "content plus one or more tool_calls"
            ],
            "hidden_label_content_semantics": "If hidden_expected_output.content is empty while hidden_expected_output.tool_calls is non-empty, that means user-facing content is optional/not required. It does not mean content is forbidden.",
            "content_with_tool_calls_policy": "Do not penalize a prediction solely because it includes brief useful user-facing content alongside valid tool_calls. Penalize only empty/no-op content with [] tool_calls, malformed DSRs, invalid JSON/tool schema, missing completed markers, or content that prevents a required tool action from being represented."
        },
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
    let response = match call_gepa_judge_lm(judge_lm, chat).await {
        Ok(response) => response,
        Err(message) => return FeedbackMetric::new(0.0, message),
    };
    let decision = match parse_gepa_judge_decision(&response) {
        Ok(decision) => decision,
        Err(error) => return FeedbackMetric::new(0.0, error),
    };
    FeedbackMetric::new(
        decision.score.clamp(0.0, 1.0),
        sanitize_gepa_judge_feedback(&decision.feedback),
    )
}

async fn call_gepa_judge_lm(judge_lm: &LM, chat: Chat) -> std::result::Result<String, String> {
    let mut last_error = None;
    for attempt in 1..=GEPA_JUDGE_LM_MAX_ATTEMPTS {
        match judge_lm.call(chat.clone(), Vec::new()).await {
            Ok(response) => return Ok(response.output.content()),
            Err(error) => {
                let message = format!(
                    "LLM judge call failed: {}",
                    sanitize_gepa_infrastructure_message(&error.to_string())
                );
                if attempt < GEPA_JUDGE_LM_MAX_ATTEMPTS {
                    eprintln!(
                        "GEPA judge LM call failed on attempt {attempt}/{GEPA_JUDGE_LM_MAX_ATTEMPTS}; retrying: {}",
                        message
                    );
                    tokio::time::sleep(gepa_target_retry_delay(attempt)).await;
                }
                last_error = Some(message);
            }
        }
    }
    Err(format!(
        "LLM judge infrastructure failure after {GEPA_JUDGE_LM_MAX_ATTEMPTS} attempts: {}",
        last_error.unwrap_or_else(|| "unknown judge error".to_string())
    ))
}

fn parse_gepa_judge_decision(response: &str) -> std::result::Result<GepaJudgeDecision, String> {
    if let Some(value) = extract_json_object(response) {
        return serde_json::from_value::<GepaJudgeDecision>(value)
            .map_err(|error| format!("LLM judge JSON was invalid: {error}"));
    }

    lenient_gepa_judge_decision(response).ok_or_else(|| {
        format!(
            "LLM judge returned non-JSON output: {}",
            truncate_for_feedback(response)
        )
    })
}

fn lenient_gepa_judge_decision(response: &str) -> Option<GepaJudgeDecision> {
    let score = lenient_json_number_field(response, "score")? as f32;
    let feedback = lenient_json_string_field(response, "feedback")
        .unwrap_or_else(|| truncate_for_feedback(response));
    Some(GepaJudgeDecision { score, feedback })
}

fn lenient_json_number_field(response: &str, field: &str) -> Option<f64> {
    let after_colon = after_json_field_colon(response, field)?;
    let end = after_colon
        .find(|ch: char| !(ch.is_ascii_digit() || matches!(ch, '-' | '+' | '.' | 'e' | 'E')))
        .unwrap_or(after_colon.len());
    after_colon[..end].trim().parse().ok()
}

fn lenient_json_string_field(response: &str, field: &str) -> Option<String> {
    let after_colon = after_json_field_colon(response, field)?;
    let value = after_colon.trim_start();
    if !value.starts_with('"') {
        return Some(value.trim_end_matches('}').trim().to_string());
    }

    let last_quote = value
        .char_indices()
        .rev()
        .find_map(|(index, ch)| (ch == '"').then_some(index))?;
    let quoted = value.get(..=last_quote)?;
    serde_json::from_str::<String>(quoted)
        .ok()
        .or_else(|| Some(quoted.trim_matches('"').replace("\\\"", "\"")))
}

fn after_json_field_colon<'a>(response: &'a str, field: &str) -> Option<&'a str> {
    let key = format!("\"{field}\"");
    let after_key = response.split_once(&key)?.1;
    after_key
        .split_once(':')
        .map(|(_, value)| value.trim_start())
}

#[cfg(test)]
fn fatal_gepa_feedback(message: String) -> FeedbackMetric {
    FeedbackMetric::with_metadata(
        0.0,
        message.clone(),
        HashMap::from([("fatal_error".to_string(), Value::String(message))]),
    )
}

fn record_gepa_fatal_feedback(errors: &GepaFatalErrors, feedback: &FeedbackMetric) {
    let Some(message) = feedback.metadata.get("fatal_error").and_then(Value::as_str) else {
        return;
    };
    if let Ok(mut errors) = errors.lock() {
        errors.push(message.to_string());
    }
}

fn ensure_no_gepa_fatal_errors(layer: &str, errors: &GepaFatalErrors) -> Result<()> {
    let errors = errors
        .lock()
        .map(|errors| errors.clone())
        .unwrap_or_else(|poisoned| poisoned.into_inner().clone());
    if errors.is_empty() {
        return Ok(());
    }

    let mut details = errors
        .iter()
        .take(3)
        .map(|error| format!("- {}", truncate_for_feedback(error)))
        .collect::<Vec<_>>()
        .join("\n");
    if errors.len() > 3 {
        details.push_str(&format!("\n- ... {} more", errors.len() - 3));
    }
    anyhow::bail!("{layer} encountered fatal rollout infrastructure errors:\n{details}");
}

fn request_adapter_judge_system_prompt() -> &'static str {
    "You are the configured GEPA judge model for a request-adapter prompt optimizer.\n\
Judge whether the target model produced a structurally usable OpenAI-compatible assistant turn after Attune's DSRs formatting.\n\
Attune is optimizing structure and contract following, not task intelligence. Valid outputs include content only, tool calls only, and content plus tool calls. Empty visible content with [] tool_calls is a failure.\n\
Content plus tool calls is explicitly valid in this project and in OpenAI-compatible assistant turns. Never treat content and tool_calls as mutually exclusive. Do not penalize a prediction solely because it includes brief useful user-facing content alongside valid tool_calls.\n\
Use hidden_expected_output only as private scoring context. If hidden_expected_output.content is empty while hidden_expected_output.tool_calls is non-empty, that means content is optional/not required, not forbidden. Do not infer a tool-only requirement from an empty hidden content label unless expected_policy explicitly says content must be rejected.\n\
Do not quote hidden labels, exact commands, exact file paths, exact answer text, field names such as expected_output, dataset/test/eval metadata, or answer-key language in feedback.\n\
Give high scores to clean structurally valid behavior even when the tool choice is different but reasonable. Penalize malformed DSRs, missing completed markers, invalid tool-call JSON, no-op content, unusable empty turns, and action-announcing content that emits [] tool_calls when a tool action is clearly required.\n\
If parser_events indicate a target LM transport or response-decoding failure after retries, treat that rollout as unusable for scoring but make the feedback say it was an infrastructure/provider failure, not prompt guidance.\n\
Return exactly one JSON object: {\"score\": number between 0 and 1, \"feedback\": \"brief generalized feedback for prompt reflection\"}."
}

fn correction_judge_system_prompt() -> &'static str {
    "You are the configured GEPA judge model for a DSRs correction-agent prompt optimizer.\n\
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

fn log_gepa_debug(layer: &str, case_id: &str, predicted: &Value, feedback: &FeedbackMetric) {
    if std::env::var_os("ATTUNE_GEPA_DEBUG").is_none() {
        return;
    }
    let predicted =
        serde_json::to_string(predicted).unwrap_or_else(|_| "<unserializable>".to_string());
    eprintln!(
        "GEPA_DEBUG layer={layer} case_id={case_id} score={:.3} feedback={} predicted={}",
        feedback.score,
        truncate_for_feedback(&feedback.feedback),
        truncate_for_feedback(&predicted)
    );
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
        .temperature(DEFAULT_GEPA_REFLECTION_TEMPERATURE)
        .track_stats(true)
        .maybe_max_rollouts(Some(64))
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

async fn read_validation_dataset_rows(path: Option<&PathBuf>) -> Result<Vec<Value>> {
    let Some(path) = path else {
        return Ok(Vec::new());
    };
    read_dataset_rows(path).await
}

fn filter_gepa_dataset_rows(
    rows: Vec<Value>,
    model_filter: Option<&str>,
    profile_filter: Option<&str>,
) -> Vec<Value> {
    rows.into_iter()
        .filter(|row| {
            matches_optional_text_filter(
                row.get("model").and_then(Value::as_str).unwrap_or_default(),
                model_filter,
            ) && matches_optional_text_filter(
                row.get("profile")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                profile_filter,
            )
        })
        .collect()
}

fn matches_optional_text_filter(actual: &str, filter: Option<&str>) -> bool {
    filter
        .map(|filter| {
            actual
                .to_ascii_lowercase()
                .contains(&filter.to_ascii_lowercase())
        })
        .unwrap_or(true)
}

fn optional_examples(examples: Vec<Example>) -> Option<Vec<Example>> {
    (!examples.is_empty()).then_some(examples)
}

fn is_zero_usize(value: &usize) -> bool {
    *value == 0
}

fn is_false(value: &bool) -> bool {
    !*value
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
    fn gepa_fatal_feedback_aborts_artifact_flow() {
        let errors = Arc::new(Mutex::new(Vec::new()));
        let feedback =
            fatal_gepa_feedback("LLM judge call failed: network unavailable".to_string());

        record_gepa_fatal_feedback(&errors, &feedback);

        let error = ensure_no_gepa_fatal_errors("GEPA test run", &errors)
            .expect_err("fatal feedback must abort");
        assert!(error
            .to_string()
            .contains("GEPA test run encountered fatal rollout infrastructure errors"));
        assert!(error
            .to_string()
            .contains("LLM judge call failed: network unavailable"));
    }

    #[test]
    fn gepa_nonfatal_feedback_does_not_abort_artifact_flow() {
        let errors = Arc::new(Mutex::new(Vec::new()));
        let feedback = FeedbackMetric::new(0.0, "model produced an empty assistant turn");

        record_gepa_fatal_feedback(&errors, &feedback);

        ensure_no_gepa_fatal_errors("GEPA test run", &errors)
            .expect("scoreable model failures should not abort as infrastructure errors");
    }

    #[test]
    fn gepa_checkpoint_report_marks_partial_unpromoted_best() {
        let spec = GepaArtifactCheckpointSpec {
            artifact_id: "request-adapter/test/checkpoint".to_string(),
            artifact_type: "request_adapter_instruction".to_string(),
            signature: "openai_tool_use_contract_profile_guidance/v1".to_string(),
            target_model: Some("provider/model".to_string()),
            profile: Some("test-profile".to_string()),
            dataset_model_filter: None,
            dataset_profile_filter: None,
            profile_revision: Some(3),
            dsrs_history_format: Some(DsrsHistoryFormat::RegeneratedContext),
            seed_artifact_path: None,
            optimizer_model: "anthropic/claude-sonnet-5".to_string(),
            judge_model: "anthropic/claude-sonnet-5".to_string(),
            examples_loaded: 22,
            validation_dataset_path: None,
            validation_examples_loaded: 0,
            lm_max_tokens: 64000,
            reflection_temperature: 1.0,
            judge_temperature: 0.0,
            target_timeout_seconds: 420,
            max_rollouts: Some(500),
            seed: 0,
            output_path: PathBuf::from("checkpoint.json"),
        };
        let checkpoint = GEPACheckpoint {
            best_candidate: dspy_rs::GEPACandidate {
                id: 7,
                instruction: "Use robust tool-call structure.".to_string(),
                module_name: "predictor".to_string(),
                example_scores: vec![0.7, 0.9],
                parent_id: Some(0),
                generation: 7,
            },
            generation: 7,
            total_rollouts: 123,
            total_lm_calls: 14,
            completed: false,
            stopped_early_reason: None,
        };

        let report = spec.report(&checkpoint).expect("checkpoint report");

        assert!(report.partial_checkpoint);
        assert_eq!(report.stopped_early_reason, None);
        assert!((report.best_average_score - 0.8).abs() < f32::EPSILON);
        assert_eq!(report.total_rollouts, 123);
        assert!(report
            .artifact_warnings
            .iter()
            .any(|warning| warning.code == "gepa_partial_checkpoint"));
    }

    #[test]
    fn gepa_checkpoint_report_sanitizes_provider_stop_reasons() {
        let spec = GepaArtifactCheckpointSpec {
            artifact_id: "request-adapter/test/checkpoint".to_string(),
            artifact_type: "request_adapter_instruction".to_string(),
            signature: "openai_tool_use_contract_profile_guidance/v1".to_string(),
            target_model: Some("provider/model".to_string()),
            profile: Some("test-profile".to_string()),
            dataset_model_filter: None,
            dataset_profile_filter: None,
            profile_revision: Some(3),
            dsrs_history_format: Some(DsrsHistoryFormat::AppendOnly),
            seed_artifact_path: None,
            optimizer_model: "anthropic/claude-sonnet-5".to_string(),
            judge_model: "anthropic/claude-sonnet-5".to_string(),
            examples_loaded: 22,
            validation_dataset_path: None,
            validation_examples_loaded: 0,
            lm_max_tokens: 64000,
            reflection_temperature: 1.0,
            judge_temperature: 0.0,
            target_timeout_seconds: 420,
            max_rollouts: Some(500),
            seed: 0,
            output_path: PathBuf::from("checkpoint.json"),
        };
        let checkpoint = GEPACheckpoint {
            best_candidate: dspy_rs::GEPACandidate {
                id: 7,
                instruction: "Use robust tool-call structure.".to_string(),
                module_name: "predictor".to_string(),
                example_scores: vec![0.7, 0.9],
                parent_id: Some(0),
                generation: 7,
            },
            generation: 7,
            total_rollouts: 123,
            total_lm_calls: 14,
            completed: false,
            stopped_early_reason: Some(
                "ProviderError: This request requires more credits. Visit https://openrouter.ai/workspaces/default/keys/secret-key-id"
                    .to_string(),
            ),
        };

        let report = spec.report(&checkpoint).expect("checkpoint report");

        assert_eq!(
            report.stopped_early_reason.as_deref(),
            Some(
                "OpenRouter returned a credit or key-limit error; add credits, raise the key limit, or lower max_tokens before continuing."
            )
        );
        let serialized = serde_json::to_string(&report).expect("serialize report");
        assert!(!serialized.contains("secret-key-id"));
        assert!(!serialized.contains("openrouter.ai/workspaces"));
    }

    #[test]
    fn quiet_panic_boundary_catches_expected_parse_panics() {
        let result = catch_unwind_without_panic_hook(AssertUnwindSafe(|| {
            panic!("expected parser panic");
        }));

        assert!(result.is_err());
    }

    #[test]
    fn gepa_lm_max_tokens_rejects_known_provider_over_cap_before_live_call() {
        let error = ensure_gepa_lm_max_tokens_supported(
            "request-adapter GEPA judge",
            "anthropic:claude-sonnet-5",
            ANTHROPIC_SONNET_MAX_OUTPUT_TOKENS + 1,
        )
        .expect_err("known invalid token cap should be rejected before LM calls");

        assert!(error.to_string().contains("exceeds the known provider cap"));
        assert!(error.to_string().contains("128000"));
    }

    #[test]
    fn gepa_lm_max_tokens_allows_known_provider_cap() {
        ensure_gepa_lm_max_tokens_supported(
            "request-adapter GEPA judge",
            "anthropic/claude-sonnet-5",
            ANTHROPIC_SONNET_MAX_OUTPUT_TOKENS,
        )
        .expect("provider cap should be accepted");
    }

    #[test]
    fn gepa_temperature_accepts_default_reflection_value() {
        ensure_gepa_temperature(
            "reflection temperature",
            DEFAULT_GEPA_REFLECTION_TEMPERATURE,
        )
        .expect("default reflection temperature should be accepted");
    }

    #[test]
    fn gepa_temperature_rejects_non_finite_value() {
        let error = ensure_gepa_temperature("judge temperature", f32::NAN)
            .expect_err("NaN temperature should be rejected");

        assert!(format!("{error:#}").contains("finite value"));
    }

    #[test]
    fn parses_strict_gepa_judge_json() {
        let decision = parse_gepa_judge_decision(
            r#"{"score":0.4,"feedback":"Valid shape, but leaked text before markers."}"#,
        )
        .expect("strict JSON should parse");

        assert_eq!(decision.score, 0.4);
        assert!(decision.feedback.contains("Valid shape"));
    }

    #[test]
    fn parses_lenient_gepa_judge_feedback_with_unescaped_json_example() {
        let decision = parse_gepa_judge_decision(
            r#"{"score": 0.05, "feedback": "The tool_call blocks contain malformed JSON (using "name": "x": {...} instead of "name": "x", "arguments": {...}), so no valid tool_calls were parsed."}"#,
        )
        .expect("lenient judge parser should recover score and feedback");

        assert_eq!(decision.score, 0.05);
        assert!(decision.feedback.contains("malformed JSON"));
        assert!(decision.feedback.contains("\"arguments\""));
    }

    #[test]
    fn gepa_lm_max_tokens_recognizes_openrouter_prefixed_sonnet_models() {
        assert_eq!(
            known_gepa_lm_max_tokens("openrouter:anthropic/claude-sonnet-5"),
            Some(ANTHROPIC_SONNET_MAX_OUTPUT_TOKENS)
        );
    }

    #[test]
    fn gepa_lm_wire_model_strips_openrouter_alias_for_openrouter_base_url() {
        assert_eq!(
            gepa_lm_wire_model(
                "openrouter:anthropic/claude-sonnet-5",
                Some(DEFAULT_GEPA_ROLE_BASE_URL)
            ),
            "anthropic/claude-sonnet-5"
        );
        assert_eq!(
            gepa_lm_wire_model("anthropic:claude-sonnet-5", None),
            "anthropic:claude-sonnet-5"
        );
    }

    #[test]
    fn providerless_model_id_strips_codex_reasoning_suffix() {
        assert_eq!(providerless_model_id("codex:gpt-5.5@medium"), "gpt-5-5");
    }

    #[test]
    fn gepa_model_roles_reject_codex_wrapper_for_same_target_model() {
        let error = ensure_gepa_model_role_separation(
            "request-adapter GEPA",
            "codex:gpt-5.5@medium",
            "anthropic/claude-sonnet-5",
            "gpt-5.5",
        )
        .expect_err("codex wrapper should still identify the same underlying model");

        assert!(error.to_string().contains("reflection/optimizer model"));
    }

    #[test]
    fn request_adapter_target_empty_response_becomes_scoreable_prediction() {
        let error =
            anyhow::anyhow!("ResponseError: Response contained no message or tool call (empty)");
        assert!(gepa_empty_target_response_error(&error));

        let prediction = failed_request_adapter_target_prediction(
            "target LM returned empty assistant response",
            format!("{error:#}"),
        );
        let predicted = prediction_to_adapter_value(&prediction);
        let expected = json!({
            "content": "",
            "tool_calls": [
                {
                    "name": "read",
                    "arguments": { "path": "README.md" }
                }
            ]
        });
        let feedback = score_request_adapter_prediction(&expected, &predicted);

        assert_eq!(predicted["content"], "");
        assert_eq!(predicted["tool_calls"], json!([]));
        assert_eq!(feedback.score, 0.0);
        assert!(feedback.feedback.contains("empty/no-op content"));
        assert!(prediction
            .data
            .get("parser_events")
            .and_then(Value::as_array)
            .is_some_and(
                |events| events.iter().any(|event| event.as_str().is_some_and(
                    |event| event.contains("target LM returned empty assistant response")
                ))
            ));
    }

    #[test]
    fn request_adapter_target_decode_error_becomes_scoreable_prediction() {
        let error = anyhow::anyhow!(
            "JsonError: data did not match any variant of untagged enum ApiResponse"
        );
        assert!(gepa_target_response_decode_error(&error));

        let prediction = failed_request_adapter_target_prediction(
            "target LM call failed after retries",
            format!("{error:#}"),
        );
        let predicted = prediction_to_adapter_value(&prediction);
        let expected = json!({
            "content": "",
            "tool_calls": [
                {
                    "name": "read",
                    "arguments": { "path": "README.md" }
                }
            ]
        });
        let feedback = score_request_adapter_prediction(&expected, &predicted);

        assert_eq!(predicted["content"], "");
        assert_eq!(predicted["tool_calls"], json!([]));
        assert_eq!(feedback.score, 0.0);
        assert!(prediction
            .data
            .get("parser_events")
            .and_then(Value::as_array)
            .is_some_and(|events| events.iter().any(|event| event
                .as_str()
                .is_some_and(|event| event.contains("target LM call failed after retries")))));
    }

    #[test]
    fn correction_target_decode_error_becomes_scoreable_prediction() {
        let error = anyhow::anyhow!(
            "JsonError: data did not match any variant of untagged enum ApiResponse"
        );

        let prediction = failed_correction_target_prediction(
            "target LM call failed after retries",
            format!("{error:#}"),
        );
        let predicted = prediction_to_repair_value(&prediction);

        assert_eq!(predicted["possible"], false);
        assert_eq!(predicted["confidence"], json!(0.0));
        assert_eq!(predicted["content"], "");
        assert_eq!(predicted["tool_calls"], json!([]));
        assert!(prediction
            .data
            .get("target_error")
            .and_then(Value::as_str)
            .is_some_and(|error| error.contains("ApiResponse")));
    }

    #[test]
    fn gepa_credit_limit_errors_are_not_scoreable_rollouts() {
        for status in [402, 403, 408, 429, 500, 503] {
            let error = UpstreamError::Status {
                status,
                body: "api error".to_string(),
                diagnostics: crate::upstream::UpstreamDiagnostics::default(),
            };

            assert!(
                !gepa_scoreable_upstream_error(&error),
                "HTTP {status} should stop GEPA as infrastructure, not score a rollout"
            );
        }
    }

    #[test]
    fn request_adapter_judge_prompt_allows_content_with_tools() {
        let prompt = request_adapter_judge_system_prompt();

        assert!(prompt.contains("Content plus tool calls is explicitly valid"));
        assert!(prompt.contains("Never treat content and tool_calls as mutually exclusive"));
        assert!(prompt.contains("content is optional/not required, not forbidden"));
        assert!(prompt.contains("infrastructure/provider failure"));
    }

    #[test]
    fn curated_request_adapter_policies_do_not_reintroduce_tool_only_requirement() {
        let datasets = [
            include_str!(
                "../datasets/request-adapter/gemma-dsrs-conservative-trace-faithful.jsonl"
            ),
            include_str!(
                "../datasets/request-adapter/gemma-dsrs-conservative-trace-harness-curated.jsonl"
            ),
        ];
        let stale_policy_fragments = [
            "When tool_calls is non-empty, leave content empty",
            "When a tool call is needed, leave content empty",
            "with empty content",
            "tool-only turns",
            "content must be empty",
        ];

        for dataset in datasets {
            for line in dataset.lines().filter(|line| !line.trim().is_empty()) {
                let row: Value = serde_json::from_str(line).expect("request-adapter dataset row");
                let prompt_goal = row
                    .pointer("/expected_adapter_policy/prompt_goal")
                    .and_then(Value::as_str)
                    .unwrap_or_default();
                for fragment in stale_policy_fragments {
                    assert!(
                        !prompt_goal.contains(fragment),
                        "curated request-adapter prompt_goal still contains stale policy fragment: {fragment}"
                    );
                }
            }
        }
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
                validation_dataset_path: None,
                output_path: PathBuf::from("artifact.json"),
                base_url: "https://openrouter.ai/api/v1".to_string(),
                api_key: None,
                reflection_base_url: None,
                reflection_api_key: None,
                model: "google/gemma-4-26b-a4b-it".to_string(),
                judge_base_url: None,
                judge_api_key: None,
                judge_model: DEFAULT_GEPA_JUDGE_MODEL.to_string(),
                target_model: Some("google/gemma-4-26b-a4b-it".to_string()),
                profile: Some("gemma-dsrs-conservative".to_string()),
                dataset_model_filter: None,
                dataset_profile_filter: None,
                profile_revision: None,
                dsrs_history_format: Some(DsrsHistoryFormat::AppendOnly),
                target_provider: None,
                artifact_id: None,
                seed_artifact_path: None,
                iterations: 1,
                max_examples: 1,
                lm_max_tokens: DEFAULT_GEPA_LM_MAX_TOKENS,
                reflection_temperature: DEFAULT_GEPA_REFLECTION_TEMPERATURE,
                judge_temperature: DEFAULT_GEPA_JUDGE_TEMPERATURE,
                target_timeout_seconds: DEFAULT_GEPA_TARGET_TIMEOUT_SECS,
                max_rollouts: None,
                seed: DEFAULT_GEPA_SEED,
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
                validation_dataset_path: None,
                output_path: PathBuf::from("artifact.json"),
                base_url: "https://openrouter.ai/api/v1".to_string(),
                api_key: None,
                reflection_base_url: None,
                reflection_api_key: None,
                model: "google/gemma-4-26b-a4b-it".to_string(),
                judge_base_url: None,
                judge_api_key: None,
                judge_model: DEFAULT_GEPA_JUDGE_MODEL.to_string(),
                target_model: Some("google/gemma-4-26b-a4b-it".to_string()),
                profile: Some("gemma-dsrs-conservative".to_string()),
                dataset_model_filter: None,
                dataset_profile_filter: None,
                profile_revision: None,
                dsrs_history_format: Some(DsrsHistoryFormat::AppendOnly),
                target_provider: None,
                artifact_id: None,
                seed_artifact_path: Some(artifact_path),
                iterations: 1,
                max_examples: 1,
                lm_max_tokens: DEFAULT_GEPA_LM_MAX_TOKENS,
                reflection_temperature: DEFAULT_GEPA_REFLECTION_TEMPERATURE,
                judge_temperature: DEFAULT_GEPA_JUDGE_TEMPERATURE,
                target_timeout_seconds: DEFAULT_GEPA_TARGET_TIMEOUT_SECS,
                max_rollouts: None,
                seed: DEFAULT_GEPA_SEED,
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
    fn filters_gepa_dataset_rows_by_model_and_profile() {
        let rows = vec![
            json!({
                "trace_id": "kimi",
                "model": "moonshotai/kimi-k2.7-code",
                "profile": "kimi-k27-code-dsrs"
            }),
            json!({
                "trace_id": "glm",
                "model": "z-ai/glm-5.2",
                "profile": "glm52-dsrs"
            }),
            json!({
                "trace_id": "qwen",
                "model": "qwen/qwen3.5-9b",
                "profile": "qwen-dsrs"
            }),
        ];

        let filtered =
            filter_gepa_dataset_rows(rows, Some("kimi-k2.7-code"), Some("kimi-k27-code-dsrs"));

        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0]["trace_id"], json!("kimi"));
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
