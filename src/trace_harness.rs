use std::{
    collections::{hash_map::DefaultHasher, BTreeMap, VecDeque},
    fs::OpenOptions,
    hash::{Hash, Hasher},
    io::Write,
    net::SocketAddr,
    path::{Path, PathBuf},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{anyhow, Context, Result};
use axum::Router;
use futures::{stream, StreamExt};
use regex::Regex;
use reqwest::{
    header::{HeaderMap, HeaderValue, RETRY_AFTER},
    StatusCode,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use tokio::{task::JoinHandle, time::Duration};

use crate::{
    config::ProxyConfig,
    gateway::Gateway,
    model_profile::ProviderRouting,
    openai::{ChatCompletionRequest, ChatMessage, OpenAiFunctionTool, OpenAiTool, OpenAiToolCall},
    repair::is_unrecovered_fallback_content,
    trace::read_trace_records,
};

#[derive(Debug, Clone)]
pub struct TraceHarnessImportConfig {
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub model: String,
    pub max_scenarios: usize,
    pub seed: u64,
    pub append: bool,
    pub max_messages: usize,
    pub max_request_chars: usize,
}

#[derive(Debug, Clone)]
pub struct TraceHarnessRunConfig {
    pub scenarios_path: PathBuf,
    pub output_path: PathBuf,
    pub proxy_url: Option<String>,
    pub model: Option<String>,
    pub limit: usize,
    pub request_timeout_seconds: u64,
    pub retries: usize,
    pub retry_backoff_ms: u64,
    pub parallel: usize,
    pub provider: Option<ProviderRouting>,
    pub proxy_config: ProxyConfig,
}

#[derive(Debug, Clone)]
pub struct TraceHarnessCompareConfig {
    pub scenarios_path: PathBuf,
    pub output_path: PathBuf,
    pub proxy_url: Option<String>,
    pub baseline_base_url: Option<String>,
    pub model: Option<String>,
    pub limit: usize,
    pub request_timeout_seconds: u64,
    pub retries: usize,
    pub retry_backoff_ms: u64,
    pub parallel: usize,
    pub provider: Option<ProviderRouting>,
    pub proxy_config: ProxyConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceHarnessScenario {
    pub schema: String,
    pub id: String,
    pub dataset: String,
    pub source_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_row: Option<u64>,
    pub turn_index: usize,
    #[serde(default)]
    pub observed_kind: String,
    pub request: ChatCompletionRequest,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceHarnessImportReport {
    pub dataset: String,
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub candidates_seen: usize,
    pub scenarios_written: usize,
    pub append: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceHarnessRunReport {
    pub scenarios_path: PathBuf,
    pub output_path: PathBuf,
    pub proxy_url: String,
    pub model: Option<String>,
    pub request_timeout_seconds: u64,
    pub retries: usize,
    pub retry_backoff_ms: u64,
    pub parallel: usize,
    pub total: usize,
    pub passed: usize,
    pub failed: usize,
    pub warnings: usize,
    pub cases: Vec<TraceHarnessCaseReport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceHarnessCaseReport {
    pub scenario_id: String,
    pub dataset: String,
    pub source_id: String,
    pub turn_index: usize,
    pub model: String,
    pub passed: bool,
    pub failures: Vec<String>,
    pub warnings: Vec<String>,
    pub final_content_len: usize,
    pub final_content_preview: Option<String>,
    pub final_tool_calls: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceHarnessCompareReport {
    pub scenarios_path: PathBuf,
    pub output_path: PathBuf,
    pub baseline_url: String,
    pub proxy_url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy_trace_path: Option<PathBuf>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy_correction_trace_path: Option<PathBuf>,
    pub model: Option<String>,
    pub request_timeout_seconds: u64,
    pub retries: usize,
    pub retry_backoff_ms: u64,
    pub parallel: usize,
    pub total: usize,
    pub baseline_passed: usize,
    pub baseline_failed: usize,
    pub proxy_passed: usize,
    pub proxy_failed: usize,
    pub both_passed: usize,
    pub both_failed: usize,
    pub proxy_fixed_baseline_failure: usize,
    pub proxy_regressed_baseline_success: usize,
    pub warnings: usize,
    pub baseline_failure_categories: BTreeMap<String, usize>,
    pub proxy_failure_categories: BTreeMap<String, usize>,
    pub proxy_repair_actions: BTreeMap<String, usize>,
    pub proxy_correction_attempts: usize,
    pub cases: Vec<TraceHarnessCompareCaseReport>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceHarnessCompareCaseReport {
    pub scenario_id: String,
    pub dataset: String,
    pub source_id: String,
    pub turn_index: usize,
    pub model: String,
    pub outcome: String,
    pub baseline: TraceHarnessEndpointReport,
    pub proxy: TraceHarnessEndpointReport,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy_trace_id: Option<String>,
    #[serde(default)]
    pub proxy_repair_actions: Vec<String>,
    #[serde(default)]
    pub proxy_policy_decisions: Vec<String>,
    #[serde(default)]
    pub proxy_correction_attempts: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceHarnessEndpointReport {
    pub passed: bool,
    pub failures: Vec<String>,
    pub warnings: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub http_status: Option<u16>,
    pub elapsed_ms: u128,
    pub final_content_len: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub final_content_preview: Option<String>,
    pub final_tool_calls: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_response_preview: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(default, skip_serializing_if = "is_zero_usize")]
    pub retry_attempts: usize,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub rate_limit_headers: BTreeMap<String, String>,
}

#[derive(Debug, Clone, Copy)]
struct TraceHarnessHttpOptions {
    retries: usize,
    retry_backoff_ms: u64,
}

#[derive(Debug, Clone)]
struct TraceHarnessRequestTag {
    scenario_id: String,
    case_index: usize,
}

pub async fn import_pi_trace_scenarios(
    config: TraceHarnessImportConfig,
) -> Result<TraceHarnessImportReport> {
    let files = jsonl_files(&config.input_path)?;
    let mut candidates = Vec::new();
    for path in files {
        let content = tokio::fs::read_to_string(&path)
            .await
            .with_context(|| format!("failed to read {}", path.display()))?;
        candidates.extend(pi_scenarios_from_jsonl(&content, &path, &config)?);
    }
    write_sampled_scenarios("badlogicgames/pi-mono", candidates, config)
}

pub async fn import_hermes_rows_scenarios(
    config: TraceHarnessImportConfig,
) -> Result<TraceHarnessImportReport> {
    let content = tokio::fs::read_to_string(&config.input_path)
        .await
        .with_context(|| format!("failed to read {}", config.input_path.display()))?;
    let value: Value = serde_json::from_str(&content)
        .with_context(|| format!("failed to parse {}", config.input_path.display()))?;
    let mut candidates = Vec::new();
    for row in value
        .get("rows")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("Hermes rows input must contain a rows array"))?
    {
        let row_idx = row.get("row_idx").and_then(Value::as_u64);
        let Some(row_value) = row.get("row") else {
            continue;
        };
        candidates.extend(hermes_scenarios_from_row(row_value, row_idx, &config)?);
    }
    write_sampled_scenarios("lambda/hermes-agent-reasoning-traces", candidates, config)
}

pub async fn run_trace_harness(config: TraceHarnessRunConfig) -> Result<TraceHarnessRunReport> {
    let scenarios = read_scenarios(&config.scenarios_path).await?;
    let (proxy_url, server_handle) = if let Some(url) = config.proxy_url.clone() {
        (url.trim_end_matches('/').to_string(), None)
    } else {
        start_proxy(config.proxy_config.clone(), &config.output_path).await?
    };

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(config.request_timeout_seconds))
        .build()?;
    let http_options = TraceHarnessHttpOptions {
        retries: config.retries,
        retry_backoff_ms: config.retry_backoff_ms,
    };
    let parallel = config.parallel.max(1);
    let provider = config.provider.clone();
    let model = config.model.clone();
    let cases = stream::iter(scenarios.into_iter().take(config.limit).enumerate().map(
        |(case_index, scenario)| {
            let client = client.clone();
            let proxy_url = proxy_url.clone();
            let provider = provider.clone();
            let model = model.clone();
            async move {
                let mut request = scenario.request.clone();
                if let Some(model) = &model {
                    request.model = model.clone();
                }
                if let Some(provider) = &provider {
                    if !provider.is_empty() {
                        request
                            .extra
                            .insert("provider".to_string(), serde_json::to_value(provider)?);
                    }
                }
                request.stream = Some(false);
                let tag = TraceHarnessRequestTag {
                    scenario_id: scenario.id.clone(),
                    case_index,
                };
                Ok::<_, anyhow::Error>(
                    run_one_scenario(&client, &proxy_url, scenario, request, http_options, tag)
                        .await,
                )
            }
        },
    ))
    .buffered(parallel)
    .collect::<Vec<_>>()
    .await
    .into_iter()
    .collect::<Result<Vec<_>>>()?;

    if let Some(handle) = server_handle {
        handle.abort();
    }

    let passed = cases.iter().filter(|case| case.passed).count();
    let failed = cases.len().saturating_sub(passed);
    let warnings = cases.iter().map(|case| case.warnings.len()).sum();
    let report = TraceHarnessRunReport {
        scenarios_path: config.scenarios_path,
        output_path: config.output_path.clone(),
        proxy_url,
        model: config.model,
        request_timeout_seconds: config.request_timeout_seconds,
        retries: config.retries,
        retry_backoff_ms: config.retry_backoff_ms,
        parallel,
        total: cases.len(),
        passed,
        failed,
        warnings,
        cases,
    };
    write_json_pretty(&config.output_path, &report).await?;
    Ok(report)
}

pub async fn run_trace_harness_compare(
    config: TraceHarnessCompareConfig,
) -> Result<TraceHarnessCompareReport> {
    let scenarios = read_scenarios(&config.scenarios_path).await?;
    let api_key = live_api_key(&config.proxy_config).ok_or_else(|| {
        anyhow!("OPENROUTER_API_KEY or MCP_UPSTREAM_API_KEY is required for live compare runs")
    })?;
    let baseline_base_url = config
        .baseline_base_url
        .clone()
        .unwrap_or_else(|| config.proxy_config.upstream.base_url.clone());
    let baseline_url = endpoint_url(&baseline_base_url, "chat/completions");
    let (proxy_trace_path, proxy_correction_trace_path) =
        trace_paths_for_output(&config.output_path);
    let (proxy_url, server_handle, trace_paths) = if let Some(url) = config.proxy_url.clone() {
        (url.trim_end_matches('/').to_string(), None, None)
    } else {
        let (url, handle) = start_proxy(config.proxy_config.clone(), &config.output_path).await?;
        (
            url,
            handle,
            Some((
                proxy_trace_path.clone(),
                proxy_correction_trace_path.clone(),
            )),
        )
    };
    let proxy_chat_url = format!("{}/v1/chat/completions", proxy_url.trim_end_matches('/'));

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(config.request_timeout_seconds))
        .build()?;
    let http_options = TraceHarnessHttpOptions {
        retries: config.retries,
        retry_backoff_ms: config.retry_backoff_ms,
    };
    let parallel = config.parallel.max(1);
    let provider = config.provider.clone();
    let model = config.model.clone();
    let mut cases = stream::iter(scenarios.into_iter().take(config.limit).enumerate().map(
        |(case_index, scenario)| {
            let client = client.clone();
            let baseline_url = baseline_url.clone();
            let proxy_chat_url = proxy_chat_url.clone();
            let api_key = api_key.clone();
            let provider = provider.clone();
            let model = model.clone();
            async move {
                let mut request = scenario.request.clone();
                if let Some(model) = &model {
                    request.model = model.clone();
                }
                if let Some(provider) = &provider {
                    if !provider.is_empty() {
                        request
                            .extra
                            .insert("provider".to_string(), serde_json::to_value(provider)?);
                    }
                }
                request.stream = Some(false);
                let tag = TraceHarnessRequestTag {
                    scenario_id: scenario.id.clone(),
                    case_index,
                };

                let baseline = call_chat_endpoint(
                    &client,
                    &baseline_url,
                    &request,
                    Some(&api_key),
                    "baseline",
                    http_options,
                    None,
                )
                .await;
                let proxy = call_chat_endpoint(
                    &client,
                    &proxy_chat_url,
                    &request,
                    Some(&api_key),
                    "proxy",
                    http_options,
                    Some(&tag),
                )
                .await;
                let outcome = compare_outcome(baseline.passed, proxy.passed).to_string();

                Ok::<_, anyhow::Error>(TraceHarnessCompareCaseReport {
                    scenario_id: scenario.id,
                    dataset: scenario.dataset,
                    source_id: scenario.source_id,
                    turn_index: scenario.turn_index,
                    model: request.model,
                    outcome,
                    baseline,
                    proxy,
                    proxy_trace_id: None,
                    proxy_repair_actions: Vec::new(),
                    proxy_policy_decisions: Vec::new(),
                    proxy_correction_attempts: Vec::new(),
                })
            }
        },
    ))
    .buffered(parallel)
    .collect::<Vec<_>>()
    .await
    .into_iter()
    .collect::<Result<Vec<_>>>()?;

    if let Some(handle) = server_handle {
        handle.abort();
    }

    if let Some((trace_path, _)) = &trace_paths {
        enrich_compare_cases_with_proxy_traces(&mut cases, trace_path).await;
    }

    let report = build_compare_report(
        config.scenarios_path,
        config.output_path.clone(),
        baseline_url,
        proxy_url,
        trace_paths,
        config.model,
        config.request_timeout_seconds,
        config.retries,
        config.retry_backoff_ms,
        parallel,
        cases,
    );
    write_json_pretty(&config.output_path, &report).await?;
    Ok(report)
}

pub async fn inspect_trace_harness_scenarios(
    scenarios_path: &Path,
    limit: usize,
) -> Result<Vec<TraceHarnessScenarioPreview>> {
    let scenarios = read_scenarios(scenarios_path).await?;
    Ok(scenarios
        .into_iter()
        .take(limit)
        .map(|scenario| TraceHarnessScenarioPreview {
            id: scenario.id,
            dataset: scenario.dataset,
            source_id: scenario.source_id,
            turn_index: scenario.turn_index,
            model: scenario.request.model.clone(),
            messages: scenario.request.messages.len(),
            tools: scenario.request.tools.as_ref().map_or(0, Vec::len),
            latest_user: latest_user_text(&scenario.request.messages).map(|text| preview(&text)),
            observed_kind: scenario.observed_kind,
            observed_tool_calls: scenario
                .metadata
                .get("observed_tool_calls")
                .and_then(Value::as_array)
                .map_or(0, Vec::len),
            request_chars: serde_json::to_string(&scenario.request).map_or(0, |text| text.len()),
        })
        .collect())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceHarnessScenarioPreview {
    pub id: String,
    pub dataset: String,
    pub source_id: String,
    pub turn_index: usize,
    pub model: String,
    pub messages: usize,
    pub tools: usize,
    pub latest_user: Option<String>,
    pub observed_kind: String,
    pub observed_tool_calls: usize,
    pub request_chars: usize,
}

async fn run_one_scenario(
    client: &reqwest::Client,
    proxy_url: &str,
    scenario: TraceHarnessScenario,
    request: ChatCompletionRequest,
    http_options: TraceHarnessHttpOptions,
    tag: TraceHarnessRequestTag,
) -> TraceHarnessCaseReport {
    let model = request.model.clone();
    let endpoint_url = format!("{proxy_url}/v1/chat/completions");
    let endpoint_report = call_chat_endpoint(
        client,
        &endpoint_url,
        &request,
        None,
        "proxy",
        http_options,
        Some(&tag),
    )
    .await;

    TraceHarnessCaseReport {
        scenario_id: scenario.id,
        dataset: scenario.dataset,
        source_id: scenario.source_id,
        turn_index: scenario.turn_index,
        model,
        passed: endpoint_report.passed,
        failures: endpoint_report.failures,
        warnings: endpoint_report.warnings,
        final_content_len: endpoint_report.final_content_len,
        final_content_preview: endpoint_report.final_content_preview,
        final_tool_calls: endpoint_report.final_tool_calls,
    }
}

async fn call_chat_endpoint(
    client: &reqwest::Client,
    endpoint_url: &str,
    request: &ChatCompletionRequest,
    api_key: Option<&str>,
    label: &str,
    http_options: TraceHarnessHttpOptions,
    tag: Option<&TraceHarnessRequestTag>,
) -> TraceHarnessEndpointReport {
    let started = Instant::now();
    let mut report = TraceHarnessEndpointReport {
        passed: false,
        failures: Vec::new(),
        warnings: Vec::new(),
        http_status: None,
        elapsed_ms: 0,
        final_content_len: 0,
        final_content_preview: None,
        final_tool_calls: Vec::new(),
        finish_reason: None,
        raw_response_preview: None,
        error: None,
        retry_attempts: 0,
        rate_limit_headers: BTreeMap::new(),
    };

    for attempt in 0..=http_options.retries {
        let mut builder = client
            .post(endpoint_url)
            .header("content-type", "application/json")
            .header("X-Title", "model-correction-proxy-trace-harness")
            .json(request);
        if let Some(api_key) = api_key {
            builder = builder.bearer_auth(api_key);
        }
        if let Some(tag) = tag {
            builder = builder
                .header("X-Trace-Harness-Scenario-Id", &tag.scenario_id)
                .header("X-Trace-Harness-Case-Index", tag.case_index.to_string());
        }

        let response = builder.send().await;
        match response {
            Ok(response) => {
                report.http_status = Some(response.status().as_u16());
                let status = response.status();
                report.rate_limit_headers = rate_limit_headers(response.headers());
                let retry_delay =
                    retry_delay_for_status(status, response.headers(), attempt, http_options);
                match response.text().await {
                    Ok(body) => {
                        report.raw_response_preview = Some(preview(&body));
                        if let Some(delay) = retry_delay {
                            report.retry_attempts += 1;
                            report.warnings.push(format!(
                                "{label} returned retryable HTTP {status}; retrying in {} ms",
                                delay.as_millis()
                            ));
                            tokio::time::sleep(delay).await;
                            continue;
                        } else if !status.is_success() {
                            report.failures.push(format!(
                                "{label} returned HTTP {status}: {}",
                                preview(&body)
                            ));
                        } else {
                            report.error = None;
                            match serde_json::from_str::<Value>(&body) {
                                Ok(value) => {
                                    validate_proxy_response(
                                        request,
                                        &value,
                                        &mut report.failures,
                                        &mut report.warnings,
                                    );
                                    fill_endpoint_response_summary(&mut report, &value);
                                }
                                Err(error) => report.failures.push(format!(
                                    "{label} returned invalid JSON: {error}: {}",
                                    preview(&body)
                                )),
                            }
                        }
                    }
                    Err(error) => {
                        if attempt < http_options.retries {
                            let delay = retry_backoff_delay(attempt, http_options);
                            report.retry_attempts += 1;
                            report.error = Some(error.to_string());
                            report.warnings.push(format!(
                                "failed to read {label} response body: {error}; retrying in {} ms",
                                delay.as_millis()
                            ));
                            tokio::time::sleep(delay).await;
                            continue;
                        }
                        report.error = Some(error.to_string());
                        report
                            .failures
                            .push(format!("failed to read {label} response body: {error}"));
                    }
                }
            }
            Err(error) => {
                if attempt < http_options.retries {
                    let delay = retry_backoff_delay(attempt, http_options);
                    report.retry_attempts += 1;
                    report.error = Some(error.to_string());
                    report.warnings.push(format!(
                        "{label} request failed: {error}; retrying in {} ms",
                        delay.as_millis()
                    ));
                    tokio::time::sleep(delay).await;
                    continue;
                }
                report.error = Some(error.to_string());
                report
                    .failures
                    .push(format!("{label} request failed: {error}"));
            }
        }
        break;
    }

    report.elapsed_ms = started.elapsed().as_millis();
    report.passed = report.failures.is_empty();
    report
}

fn retry_delay_for_status(
    status: StatusCode,
    headers: &HeaderMap,
    attempt: usize,
    http_options: TraceHarnessHttpOptions,
) -> Option<Duration> {
    if attempt >= http_options.retries || !is_retryable_status(status) {
        return None;
    }
    Some(
        retry_after_delay(headers)
            .or_else(|| rate_limit_reset_delay(headers))
            .unwrap_or_else(|| retry_backoff_delay(attempt, http_options))
            .min(Duration::from_secs(120)),
    )
}

fn is_retryable_status(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS
        || status == StatusCode::REQUEST_TIMEOUT
        || status.as_u16() == 425
        || status.is_server_error()
}

fn retry_backoff_delay(attempt: usize, http_options: TraceHarnessHttpOptions) -> Duration {
    let multiplier = 1_u64 << attempt.min(6);
    Duration::from_millis(
        http_options
            .retry_backoff_ms
            .saturating_mul(multiplier)
            .max(1),
    )
    .min(Duration::from_secs(120))
}

fn retry_after_delay(headers: &HeaderMap) -> Option<Duration> {
    let seconds = header_seconds(headers.get(RETRY_AFTER)?)?;
    Some(Duration::from_secs(seconds))
}

fn rate_limit_reset_delay(headers: &HeaderMap) -> Option<Duration> {
    let value = header_seconds(headers.get("x-ratelimit-reset")?)?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs();
    if value > now && value.saturating_sub(now) <= 300 {
        Some(Duration::from_secs(value - now))
    } else if value <= 300 {
        Some(Duration::from_secs(value))
    } else {
        None
    }
}

fn header_seconds(value: &HeaderValue) -> Option<u64> {
    value.to_str().ok()?.trim().parse::<u64>().ok()
}

fn rate_limit_headers(headers: &HeaderMap) -> BTreeMap<String, String> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            let key = name.as_str().to_ascii_lowercase();
            if key == "retry-after" || key.starts_with("x-ratelimit") {
                value
                    .to_str()
                    .ok()
                    .map(|value| (key, value.chars().take(256).collect()))
            } else {
                None
            }
        })
        .collect()
}

fn is_zero_usize(value: &usize) -> bool {
    *value == 0
}

fn fill_endpoint_response_summary(report: &mut TraceHarnessEndpointReport, value: &Value) {
    report.finish_reason = value
        .pointer("/choices/0/finish_reason")
        .and_then(Value::as_str)
        .map(str::to_string);
    if let Some(content) = value
        .pointer("/choices/0/message/content")
        .and_then(Value::as_str)
    {
        report.final_content_len = content.len();
        report.final_content_preview = Some(preview(content));
    }
    if let Some(calls) = value
        .pointer("/choices/0/message/tool_calls")
        .and_then(Value::as_array)
    {
        for call in calls {
            let name = call
                .pointer("/function/name")
                .and_then(Value::as_str)
                .unwrap_or("<missing>");
            let args = call
                .pointer("/function/arguments")
                .and_then(Value::as_str)
                .unwrap_or("");
            report.final_tool_calls.push(format!("{name} {args}"));
        }
    }
}

fn validate_proxy_response(
    request: &ChatCompletionRequest,
    response: &Value,
    failures: &mut Vec<String>,
    warnings: &mut Vec<String>,
) {
    let Some(message) = response.pointer("/choices/0/message") else {
        failures.push("response missing choices[0].message".to_string());
        return;
    };
    let content = message.get("content");
    let content_text = content.and_then(Value::as_str).unwrap_or_default();
    let tool_calls = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();

    if content_text.contains("[[ ##") || content_text.contains("[[##") {
        failures.push("final content leaked DSRs field markers".to_string());
    }
    let lowered = content_text.trim_start().to_ascii_lowercase();
    if lowered.starts_with("content:") || lowered.starts_with("tool_calls:") {
        failures.push("final content leaked DSRs-style field labels".to_string());
    }
    if tool_calls.is_empty() && content_text.trim().is_empty() {
        failures.push("final assistant message had empty content and no tool calls".to_string());
    }
    if is_unrecovered_fallback_content(content_text) {
        failures.push("proxy returned an unrecovered fallback response".to_string());
    }

    let known_tools = request
        .tools
        .as_ref()
        .map(|tools| {
            tools
                .iter()
                .map(|tool| tool.function.name.as_str())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if tool_calls.is_empty() && !known_tools.is_empty() {
        if looks_like_text_tool_call(content_text) {
            failures.push("assistant emitted tool-like text but no OpenAI tool_calls".to_string());
        }
        if looks_like_premature_tool_action(content_text) {
            failures.push(
                "assistant appeared to announce a tool/action without a tool call".to_string(),
            );
        }
    }
    let mut seen_calls = BTreeMap::<String, usize>::new();
    for call in tool_calls {
        let name = call.pointer("/function/name").and_then(Value::as_str);
        let args = call.pointer("/function/arguments").and_then(Value::as_str);
        let Some(name) = name else {
            failures.push("tool call missing function.name".to_string());
            continue;
        };
        if !known_tools.is_empty() && !known_tools.contains(&name) {
            failures.push(format!("tool call used unknown tool {name:?}"));
        }
        let Some(args) = args else {
            failures.push(format!("tool call {name:?} missing function.arguments"));
            continue;
        };
        match serde_json::from_str::<Value>(args) {
            Ok(Value::Object(_)) => {}
            Ok(_) => failures.push(format!(
                "tool call {name:?} arguments parsed but were not a JSON object"
            )),
            Err(error) => failures.push(format!(
                "tool call {name:?} arguments were not valid JSON: {error}"
            )),
        }
        let key = format!("{name}:{args}");
        *seen_calls.entry(key).or_default() += 1;
    }
    for (key, count) in seen_calls {
        if count > 1 {
            warnings.push(format!(
                "duplicate final tool call appeared {count} times: {key}"
            ));
        }
    }
}

fn looks_like_text_tool_call(content: &str) -> bool {
    let lower = content.to_ascii_lowercase();
    lower.contains("<tool_call")
        || lower.contains("</tool_call>")
        || lower.contains("[[ ## tool_calls ## ]]")
        || lower.contains("[[## tool_calls ##]]")
        || lower.contains("tool_calls:")
        || lower.contains("\"tool_calls\"")
}

fn looks_like_premature_tool_action(content: &str) -> bool {
    let lower = content
        .split_whitespace()
        .take(40)
        .collect::<Vec<_>>()
        .join(" ")
        .to_ascii_lowercase();
    let prefixes = [
        "let me ",
        "i'll ",
        "i will ",
        "i am going to ",
        "i'm going to ",
    ];
    prefixes.iter().any(|prefix| {
        action_phrase_has_tool_verb(&lower, prefix)
            || [". ", "! ", "? ", "\n"]
                .iter()
                .any(|boundary| action_phrase_has_tool_verb(&lower, &format!("{boundary}{prefix}")))
    })
}

fn action_phrase_has_tool_verb(lower: &str, marker: &str) -> bool {
    let Some(index) = lower.find(marker) else {
        return false;
    };
    let mut tail = lower[index + marker.len()..].trim_start();
    for filler in [
        "go ahead and ",
        "now ",
        "first ",
        "try to ",
        "attempt to ",
        "attempting to ",
    ] {
        if let Some(stripped) = tail.strip_prefix(filler) {
            tail = stripped.trim_start();
        }
    }
    let tool_action_verbs = [
        "read",
        "check",
        "compile",
        "inspect",
        "open",
        "list",
        "search",
        "run",
        "execute",
        "call",
        "use the",
        "look at",
        "look into",
        "continue with",
        "create",
        "write",
        "edit",
        "modify",
        "update",
    ];
    tool_action_verbs.iter().any(|verb| tail.starts_with(verb))
}

fn compare_outcome(baseline_passed: bool, proxy_passed: bool) -> &'static str {
    match (baseline_passed, proxy_passed) {
        (true, true) => "both_passed",
        (false, true) => "proxy_fixed_baseline_failure",
        (true, false) => "proxy_regressed_baseline_success",
        (false, false) => "both_failed",
    }
}

async fn enrich_compare_cases_with_proxy_traces(
    cases: &mut [TraceHarnessCompareCaseReport],
    trace_path: &Path,
) {
    let Ok(records) = read_trace_records(&trace_path.to_path_buf()).await else {
        return;
    };
    let records_by_case_index = records
        .iter()
        .filter_map(|record| trace_harness_case_index(record).map(|index| (index, record)))
        .collect::<BTreeMap<_, _>>();
    if !records_by_case_index.is_empty() {
        for (case_index, case) in cases.iter_mut().enumerate() {
            if let Some(record) = records_by_case_index.get(&case_index) {
                enrich_compare_case_with_proxy_trace(case, record);
            }
        }
    } else {
        for (case, record) in cases.iter_mut().zip(records.iter()) {
            enrich_compare_case_with_proxy_trace(case, record);
        }
    }
}

fn trace_harness_case_index(record: &crate::trace::TraceRecord) -> Option<usize> {
    record
        .metadata
        .pointer("/headers/x-trace-harness-case-index")
        .and_then(Value::as_str)?
        .parse()
        .ok()
}

fn enrich_compare_case_with_proxy_trace(
    case: &mut TraceHarnessCompareCaseReport,
    record: &crate::trace::TraceRecord,
) {
    case.proxy_trace_id = Some(record.trace_id.clone());
    case.proxy_repair_actions = record
        .repair_actions
        .iter()
        .map(|action| action.action.clone())
        .collect();
    case.proxy_policy_decisions = record
        .policy_decisions
        .iter()
        .map(|decision| format!("{}:{}", decision.stage, decision.decision))
        .collect();
    case.proxy_correction_attempts = record
        .correction_attempts
        .iter()
        .map(|attempt| attempt.result.clone())
        .collect();
}

fn build_compare_report(
    scenarios_path: PathBuf,
    output_path: PathBuf,
    baseline_url: String,
    proxy_url: String,
    trace_paths: Option<(PathBuf, PathBuf)>,
    model: Option<String>,
    request_timeout_seconds: u64,
    retries: usize,
    retry_backoff_ms: u64,
    parallel: usize,
    cases: Vec<TraceHarnessCompareCaseReport>,
) -> TraceHarnessCompareReport {
    let total = cases.len();
    let baseline_passed = cases.iter().filter(|case| case.baseline.passed).count();
    let proxy_passed = cases.iter().filter(|case| case.proxy.passed).count();
    let both_passed = cases
        .iter()
        .filter(|case| case.outcome == "both_passed")
        .count();
    let both_failed = cases
        .iter()
        .filter(|case| case.outcome == "both_failed")
        .count();
    let proxy_fixed_baseline_failure = cases
        .iter()
        .filter(|case| case.outcome == "proxy_fixed_baseline_failure")
        .count();
    let proxy_regressed_baseline_success = cases
        .iter()
        .filter(|case| case.outcome == "proxy_regressed_baseline_success")
        .count();
    let warnings = cases
        .iter()
        .map(|case| case.baseline.warnings.len() + case.proxy.warnings.len())
        .sum();
    let mut baseline_failure_categories = BTreeMap::new();
    let mut proxy_failure_categories = BTreeMap::new();
    let mut proxy_repair_actions = BTreeMap::new();
    for case in &cases {
        for failure in &case.baseline.failures {
            *baseline_failure_categories
                .entry(failure_category(failure))
                .or_default() += 1;
        }
        for failure in &case.proxy.failures {
            *proxy_failure_categories
                .entry(failure_category(failure))
                .or_default() += 1;
        }
        for action in &case.proxy_repair_actions {
            *proxy_repair_actions.entry(action.clone()).or_default() += 1;
        }
    }
    let proxy_correction_attempts = cases
        .iter()
        .map(|case| case.proxy_correction_attempts.len())
        .sum();
    TraceHarnessCompareReport {
        scenarios_path,
        output_path,
        baseline_url,
        proxy_url,
        proxy_trace_path: trace_paths.as_ref().map(|paths| paths.0.clone()),
        proxy_correction_trace_path: trace_paths.as_ref().map(|paths| paths.1.clone()),
        model,
        request_timeout_seconds,
        retries,
        retry_backoff_ms,
        parallel,
        total,
        baseline_passed,
        baseline_failed: total.saturating_sub(baseline_passed),
        proxy_passed,
        proxy_failed: total.saturating_sub(proxy_passed),
        both_passed,
        both_failed,
        proxy_fixed_baseline_failure,
        proxy_regressed_baseline_success,
        warnings,
        baseline_failure_categories,
        proxy_failure_categories,
        proxy_repair_actions,
        proxy_correction_attempts,
        cases,
    }
}

fn failure_category(failure: &str) -> String {
    let lower = failure.to_ascii_lowercase();
    if lower.contains("tool-like text") {
        "tool_like_text_without_openai_tool_calls".to_string()
    } else if lower.contains("announce a tool/action") {
        "premature_tool_action_without_tool_call".to_string()
    } else if lower.contains("empty content and no tool calls") {
        "empty_content_empty_tool_calls".to_string()
    } else if lower.contains("unrecovered fallback") {
        "unrecovered_proxy_fallback".to_string()
    } else if lower.contains("unknown tool") {
        "unknown_tool_name".to_string()
    } else if lower.contains("not valid json") {
        "invalid_json".to_string()
    } else if lower.contains("not a json object") {
        "tool_arguments_not_object".to_string()
    } else if lower.contains("dsrs") {
        "dsrs_leak_or_contract_violation".to_string()
    } else if lower.contains("http") {
        "http_error".to_string()
    } else {
        "other".to_string()
    }
}

async fn start_proxy(
    mut config: ProxyConfig,
    output_path: &Path,
) -> Result<(String, Option<JoinHandle<()>>)> {
    config.upstream.api_key = live_api_key(&config);
    if config.upstream.api_key.is_none() {
        return Err(anyhow!(
            "OPENROUTER_API_KEY or MCP_UPSTREAM_API_KEY is required for in-process live proxy runs"
        ));
    }
    if config.upstream.base_url.trim().is_empty() {
        config.upstream.base_url = "https://openrouter.ai/api/v1".to_string();
    }
    let (trace_path, correction_path) = trace_paths_for_output(output_path);
    let parent = output_path.parent().unwrap_or_else(|| Path::new("."));
    tokio::fs::create_dir_all(parent).await?;
    config.trace.path = trace_path;
    config.trace.correction_path = correction_path;

    let gateway = Gateway::new(config)?;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr: SocketAddr = listener.local_addr()?;
    let router: Router = gateway.router();
    let handle = tokio::spawn(async move {
        if let Err(error) = axum::serve(listener, router).await {
            tracing::error!(%error, "trace harness proxy server failed");
        }
    });
    Ok((format!("http://{addr}"), Some(handle)))
}

fn trace_paths_for_output(output_path: &Path) -> (PathBuf, PathBuf) {
    let stem = output_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("trace-harness-run");
    let parent = output_path.parent().unwrap_or_else(|| Path::new("."));
    (
        parent.join(format!("{stem}-proxy-traces.local.jsonl")),
        parent.join(format!("{stem}-corrections.local.jsonl")),
    )
}

fn endpoint_url(base_url: &str, endpoint: &str) -> String {
    format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        endpoint.trim_start_matches('/')
    )
}

fn live_api_key(config: &ProxyConfig) -> Option<String> {
    config
        .upstream
        .api_key
        .clone()
        .or_else(|| env_key("OPENROUTER_API_KEY"))
        .or_else(|| read_env_key("OPENROUTER_API_KEY"))
        .or_else(|| env_key("MCP_UPSTREAM_API_KEY"))
        .or_else(|| read_env_key("MCP_UPSTREAM_API_KEY"))
}

fn pi_scenarios_from_jsonl(
    content: &str,
    path: &Path,
    config: &TraceHarnessImportConfig,
) -> Result<Vec<TraceHarnessScenario>> {
    let mut session_id = path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("pi-session")
        .to_string();
    let mut history = vec![ChatMessage::new("system", pi_default_system_prompt())];
    let mut observed_tools = standard_pi_tools();
    let mut candidates = Vec::new();
    let mut turn_index = 0usize;

    for line in content.lines().filter(|line| !line.trim().is_empty()) {
        let event: Value = serde_json::from_str(line)?;
        if event.get("type").and_then(Value::as_str) == Some("session") {
            if let Some(id) = event.get("id").and_then(Value::as_str) {
                session_id = id.to_string();
            }
            continue;
        }
        if event.get("type").and_then(Value::as_str) != Some("message") {
            continue;
        }
        let Some(message_value) = event.get("message") else {
            continue;
        };
        let role = message_value
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or_default();
        match role {
            "user" => history.push(ChatMessage::new("user", parts_text(message_value, true))),
            "assistant" => {
                let assistant = assistant_message_from_parts(message_value);
                add_observed_tools(&mut observed_tools, assistant.tool_calls.as_ref());
                if history.iter().any(|message| message.role == "user") {
                    if let Some(scenario) = build_scenario(
                        "badlogicgames/pi-mono",
                        &session_id,
                        None,
                        turn_index,
                        &config.model,
                        &history,
                        &observed_tools,
                        &assistant,
                        config,
                    )? {
                        candidates.push(scenario);
                    }
                }
                history.push(assistant);
                turn_index += 1;
            }
            "toolResult" => history.push(tool_result_message_from_pi(message_value)),
            _ => {}
        }
    }
    Ok(candidates)
}

fn hermes_scenarios_from_row(
    row: &Value,
    row_idx: Option<u64>,
    config: &TraceHarnessImportConfig,
) -> Result<Vec<TraceHarnessScenario>> {
    let source_id = row
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("hermes-row")
        .to_string();
    let tools = row
        .get("tools")
        .and_then(Value::as_str)
        .and_then(|text| serde_json::from_str::<Value>(text).ok())
        .and_then(tools_from_value)
        .unwrap_or_else(|| vec![generic_tool("terminal")]);
    let mut history = Vec::new();
    let mut pending_tool_call_ids = VecDeque::new();
    let mut candidates = Vec::new();
    let conversations = row
        .get("conversations")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow!("Hermes row {source_id} missing conversations"))?;
    let mut turn_index = 0usize;

    for turn in conversations {
        let from = turn.get("from").and_then(Value::as_str).unwrap_or_default();
        let value = turn
            .get("value")
            .and_then(Value::as_str)
            .unwrap_or_default();
        match from {
            "system" => history.push(ChatMessage::new("system", value)),
            "human" | "user" => history.push(ChatMessage::new("user", value)),
            "gpt" | "assistant" => {
                let assistant = assistant_message_from_tool_call_text(value);
                if history.iter().any(|message| message.role == "user") {
                    if let Some(scenario) = build_scenario(
                        "lambda/hermes-agent-reasoning-traces",
                        &source_id,
                        row_idx,
                        turn_index,
                        &config.model,
                        &history,
                        &tools,
                        &assistant,
                        config,
                    )? {
                        candidates.push(scenario);
                    }
                }
                if let Some(calls) = &assistant.tool_calls {
                    for call in calls {
                        pending_tool_call_ids.push_back(call.id.clone());
                    }
                }
                history.push(assistant);
                turn_index += 1;
            }
            "tool" | "tool_result" | "observation" => {
                let id = pending_tool_call_ids
                    .pop_front()
                    .unwrap_or_else(|| format!("tool_result_{turn_index}"));
                let mut message = ChatMessage::new("tool", value);
                message.tool_call_id = Some(id);
                history.push(message);
            }
            _ => {}
        }
    }
    Ok(candidates)
}

fn build_scenario(
    dataset: &str,
    source_id: &str,
    source_row: Option<u64>,
    turn_index: usize,
    model: &str,
    history: &[ChatMessage],
    tools: &[OpenAiTool],
    observed_assistant: &ChatMessage,
    config: &TraceHarnessImportConfig,
) -> Result<Option<TraceHarnessScenario>> {
    let messages = bounded_messages(history, config.max_messages);
    if latest_user_text(&messages).is_none() {
        return Ok(None);
    }
    let request = ChatCompletionRequest {
        model: model.to_string(),
        messages,
        tools: Some(tools.to_vec()),
        tool_choice: None,
        parallel_tool_calls: Some(true),
        stream: Some(false),
        temperature: None,
        top_p: None,
        max_tokens: None,
        max_completion_tokens: None,
        response_format: None,
        extra: Map::new(),
    };
    let request_chars = serde_json::to_string(&request)?.len();
    if request_chars > config.max_request_chars {
        return Ok(None);
    }
    let observed_calls = observed_assistant
        .tool_calls
        .as_ref()
        .map(|calls| {
            calls
                .iter()
                .map(|call| {
                    json!({
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let observed_content = observed_assistant.content_text().unwrap_or_default();
    let mut metadata = Map::new();
    metadata.insert("request_chars".to_string(), json!(request_chars));
    metadata.insert(
        "observed_tool_calls".to_string(),
        Value::Array(observed_calls),
    );
    metadata.insert(
        "observed_content_preview".to_string(),
        Value::String(preview(&observed_content)),
    );
    Ok(Some(TraceHarnessScenario {
        schema: "model-correction-proxy.trace_harness.scenario/v1".to_string(),
        id: format!("{}:{}:{}", dataset.replace('/', "_"), source_id, turn_index),
        dataset: dataset.to_string(),
        source_id: source_id.to_string(),
        source_row,
        turn_index,
        observed_kind: if observed_assistant
            .tool_calls
            .as_ref()
            .is_some_and(|calls| !calls.is_empty())
        {
            "tool_calls".to_string()
        } else {
            "content".to_string()
        },
        request,
        metadata,
    }))
}

fn bounded_messages(messages: &[ChatMessage], max_messages: usize) -> Vec<ChatMessage> {
    if messages.len() <= max_messages {
        return messages.to_vec();
    }
    let first_system = messages
        .first()
        .filter(|message| message.role == "system")
        .cloned();
    let mut out = Vec::new();
    if let Some(system) = first_system {
        out.push(system);
    }
    let remaining = max_messages.saturating_sub(out.len());
    let start = messages.len().saturating_sub(remaining);
    out.extend(messages[start..].iter().cloned());
    while out
        .get(usize::from(
            out.first().is_some_and(|message| message.role == "system"),
        ))
        .is_some_and(|message| message.role == "tool")
    {
        let index = usize::from(out.first().is_some_and(|message| message.role == "system"));
        out.remove(index);
    }
    out
}

fn write_sampled_scenarios(
    dataset: &str,
    mut candidates: Vec<TraceHarnessScenario>,
    config: TraceHarnessImportConfig,
) -> Result<TraceHarnessImportReport> {
    let candidates_seen = candidates.len();
    candidates.sort_by_key(|scenario| stable_sample_key(&scenario.id, config.seed));
    candidates.truncate(config.max_scenarios);
    write_scenarios(&config.output_path, &candidates, config.append)?;
    Ok(TraceHarnessImportReport {
        dataset: dataset.to_string(),
        input_path: config.input_path,
        output_path: config.output_path,
        candidates_seen,
        scenarios_written: candidates.len(),
        append: config.append,
    })
}

fn write_scenarios(path: &Path, scenarios: &[TraceHarnessScenario], append: bool) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut options = OpenOptions::new();
    options.create(true).write(true);
    if append {
        options.append(true);
    } else {
        options.truncate(true);
    }
    let mut file = options
        .open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    for scenario in scenarios {
        serde_json::to_writer(&mut file, scenario)?;
        file.write_all(b"\n")?;
    }
    Ok(())
}

async fn read_scenarios(path: &Path) -> Result<Vec<TraceHarnessScenario>> {
    let content = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("failed to read {}", path.display()))?;
    content
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str(line).context("failed to parse trace harness scenario"))
        .collect()
}

fn jsonl_files(path: &Path) -> Result<Vec<PathBuf>> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut files = Vec::new();
    for entry in
        std::fs::read_dir(path).with_context(|| format!("failed to list {}", path.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) == Some("jsonl") {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

fn assistant_message_from_parts(message_value: &Value) -> ChatMessage {
    let content = parts_text(message_value, false);
    let tool_calls = message_value
        .get("content")
        .and_then(Value::as_array)
        .map(|parts| {
            parts
                .iter()
                .filter(|part| part.get("type").and_then(Value::as_str) == Some("toolCall"))
                .filter_map(tool_call_from_pi_part)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let mut message = ChatMessage {
        role: "assistant".to_string(),
        ..ChatMessage::default()
    };
    if !content.trim().is_empty() {
        message.set_content_text(content);
    }
    if !tool_calls.is_empty() {
        message.tool_calls = Some(tool_calls);
    }
    message
}

fn assistant_message_from_tool_call_text(text: &str) -> ChatMessage {
    let calls = extract_xml_tool_calls(text);
    let content = strip_tool_call_blocks(text)
        .replace("<think>", "")
        .replace("</think>", "")
        .trim()
        .to_string();
    let mut message = ChatMessage {
        role: "assistant".to_string(),
        ..ChatMessage::default()
    };
    if !content.is_empty() {
        message.set_content_text(content);
    }
    if !calls.is_empty() {
        message.tool_calls = Some(calls);
    }
    message
}

fn tool_result_message_from_pi(message_value: &Value) -> ChatMessage {
    let mut message = ChatMessage::new("tool", parts_text(message_value, true));
    message.tool_call_id = message_value
        .get("toolCallId")
        .and_then(Value::as_str)
        .map(str::to_string);
    message
}

fn parts_text(message_value: &Value, include_non_text_placeholders: bool) -> String {
    let Some(parts) = message_value.get("content").and_then(Value::as_array) else {
        return message_value
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
    };
    let mut out = Vec::new();
    for part in parts {
        match part.get("type").and_then(Value::as_str) {
            Some("text") => {
                if let Some(text) = part.get("text").and_then(Value::as_str) {
                    out.push(text.to_string());
                }
            }
            Some("image") if include_non_text_placeholders => {
                out.push("[image omitted from imported trace]".to_string());
            }
            _ => {}
        }
    }
    out.join("\n")
}

fn tool_call_from_pi_part(part: &Value) -> Option<OpenAiToolCall> {
    let id = part
        .get("id")
        .and_then(Value::as_str)
        .unwrap_or("call_imported");
    let name = part.get("name").and_then(Value::as_str)?;
    let arguments = part
        .get("arguments")
        .cloned()
        .unwrap_or_else(|| Value::Object(Map::new()));
    Some(OpenAiToolCall {
        id: id.to_string(),
        call_type: "function".to_string(),
        function: crate::openai::OpenAiFunctionCall {
            name: name.to_string(),
            arguments: serde_json::to_string(&arguments).ok()?,
        },
    })
}

fn extract_xml_tool_calls(text: &str) -> Vec<OpenAiToolCall> {
    let re = Regex::new(r"(?s)<tool_call>\s*(.*?)\s*</tool_call>").expect("valid regex");
    re.captures_iter(text)
        .filter_map(|capture| capture.get(1).map(|body| body.as_str().trim()))
        .filter_map(parse_tool_call_body)
        .collect()
}

fn parse_tool_call_body(body: &str) -> Option<OpenAiToolCall> {
    let value: Value = serde_json::from_str(body)
        .ok()
        .or_else(|| json5::from_str(body).ok())?;
    let (name, arguments) = if let Some(function) = value.get("function") {
        (
            function.get("name")?.as_str()?.to_string(),
            function
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| Value::Object(Map::new())),
        )
    } else {
        (
            value.get("name")?.as_str()?.to_string(),
            value
                .get("arguments")
                .cloned()
                .unwrap_or_else(|| Value::Object(Map::new())),
        )
    };
    Some(OpenAiToolCall::function(
        name,
        serde_json::to_string(&arguments).ok()?,
    ))
}

fn strip_tool_call_blocks(text: &str) -> String {
    let re = Regex::new(r"(?s)<tool_call>\s*.*?\s*</tool_call>").expect("valid regex");
    re.replace_all(text, "").to_string()
}

fn tools_from_value(value: Value) -> Option<Vec<OpenAiTool>> {
    let array = value.as_array()?;
    let tools = array.iter().filter_map(tool_from_value).collect::<Vec<_>>();
    (!tools.is_empty()).then_some(tools)
}

fn tool_from_value(value: &Value) -> Option<OpenAiTool> {
    if value.get("type").and_then(Value::as_str) == Some("function") {
        let function = value.get("function")?;
        return function_tool_from_value(function);
    }
    function_tool_from_value(value)
}

fn function_tool_from_value(value: &Value) -> Option<OpenAiTool> {
    let name = value.get("name")?.as_str()?.to_string();
    let description = value
        .get("description")
        .and_then(Value::as_str)
        .map(str::to_string);
    let parameters = value
        .get("parameters")
        .cloned()
        .unwrap_or_else(loose_object_schema);
    Some(OpenAiTool {
        tool_type: "function".to_string(),
        function: OpenAiFunctionTool {
            name,
            description,
            parameters,
        },
    })
}

fn standard_pi_tools() -> Vec<OpenAiTool> {
    vec![
        OpenAiTool {
            tool_type: "function".to_string(),
            function: OpenAiFunctionTool {
                name: "read".to_string(),
                description: Some("Read the contents of a file.".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "offset": {"type": "number"},
                        "limit": {"type": "number"}
                    },
                    "required": ["path"]
                }),
            },
        },
        OpenAiTool {
            tool_type: "function".to_string(),
            function: OpenAiFunctionTool {
                name: "bash".to_string(),
                description: Some("Execute a bash command.".to_string()),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "number"}
                    },
                    "required": ["command"]
                }),
            },
        },
        generic_tool("edit"),
        generic_tool("write"),
    ]
}

fn add_observed_tools(tools: &mut Vec<OpenAiTool>, calls: Option<&Vec<OpenAiToolCall>>) {
    let Some(calls) = calls else {
        return;
    };
    for call in calls {
        if !tools
            .iter()
            .any(|tool| tool.function.name == call.function.name)
        {
            tools.push(generic_tool(&call.function.name));
        }
    }
}

fn generic_tool(name: &str) -> OpenAiTool {
    OpenAiTool {
        tool_type: "function".to_string(),
        function: OpenAiFunctionTool {
            name: name.to_string(),
            description: Some(format!("Imported harness tool {name}.")),
            parameters: loose_object_schema(),
        },
    }
}

fn loose_object_schema() -> Value {
    json!({
        "type": "object",
        "properties": {},
        "additionalProperties": true
    })
}

fn pi_default_system_prompt() -> &'static str {
    "You are an expert coding assistant operating inside a generic coding harness. Use the available tools for file and shell work. Be concise, and return either user-facing content or tool calls as appropriate."
}

fn latest_user_text(messages: &[ChatMessage]) -> Option<String> {
    messages
        .iter()
        .rev()
        .find(|message| message.role == "user")
        .and_then(ChatMessage::content_text)
}

fn stable_sample_key(id: &str, seed: u64) -> u64 {
    let mut hasher = DefaultHasher::new();
    seed.hash(&mut hasher);
    id.hash(&mut hasher);
    hasher.finish()
}

fn preview(text: &str) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let preview = collapsed.chars().take(240).collect::<String>();
    if preview.len() < collapsed.len() {
        format!("{preview}...")
    } else {
        preview
    }
}

async fn write_json_pretty<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    let content = serde_json::to_string_pretty(value)?;
    tokio::fs::write(path, content).await?;
    Ok(())
}

fn env_key(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .and_then(|value| normalize_key(&value))
}

fn read_env_key(name: &str) -> Option<String> {
    let env = std::fs::read_to_string(".env").ok()?;
    env.lines().find_map(|line| {
        let line = line.trim();
        let (key, value) = line.split_once('=')?;
        let key = key.trim().strip_prefix("export ").unwrap_or(key.trim());
        if key == name {
            normalize_key(value)
        } else {
            None
        }
    })
}

fn normalize_key(value: &str) -> Option<String> {
    let value = value.trim().trim_matches('"').trim_matches('\'').trim();
    let value = value
        .strip_prefix("Bearer ")
        .or_else(|| value.strip_prefix("bearer "))
        .unwrap_or(value)
        .trim();
    (!value.is_empty()).then(|| value.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn import_config() -> TraceHarnessImportConfig {
        TraceHarnessImportConfig {
            input_path: PathBuf::from("input"),
            output_path: PathBuf::from("output"),
            model: "test/model".to_string(),
            max_scenarios: 10,
            seed: 13,
            append: false,
            max_messages: 24,
            max_request_chars: 60000,
        }
    }

    #[test]
    fn imports_pi_jsonl_as_openai_scenario_prefix() {
        let content = r#"
{"type":"session","id":"session-1"}
{"type":"message","message":{"role":"user","content":[{"type":"text","text":"Read the README."}]}}
{"type":"message","message":{"role":"assistant","content":[{"type":"toolCall","id":"call_readme","name":"read","arguments":{"path":"README.md"}}]}}
{"type":"message","message":{"role":"toolResult","toolCallId":"call_readme","content":[{"type":"text","text":"README contents"}]}}
{"type":"message","message":{"role":"assistant","content":[{"type":"text","text":"The README describes the proxy."}]}}
"#;

        let scenarios =
            pi_scenarios_from_jsonl(content, Path::new("session.jsonl"), &import_config())
                .expect("imports pi scenarios");

        assert_eq!(scenarios.len(), 2);
        let first = &scenarios[0];
        assert_eq!(first.dataset, "badlogicgames/pi-mono");
        assert_eq!(first.source_id, "session-1");
        assert_eq!(first.observed_kind, "tool_calls");
        assert_eq!(first.request.messages.len(), 2);
        assert_eq!(
            first.request.messages[1].content_text().as_deref(),
            Some("Read the README.")
        );
        assert!(first
            .request
            .tools
            .as_ref()
            .expect("tools")
            .iter()
            .any(|tool| tool.function.name == "read"));
    }

    #[test]
    fn imports_hermes_rows_with_tool_call_blocks() {
        let row = json!({
            "id": "hermes-1",
            "tools": r#"[{"type":"function","function":{"name":"read","description":"Read a file","parameters":{"type":"object","properties":{"path":{"type":"string"}},"required":["path"]}}}]"#,
            "conversations": [
                {"from":"system","value":"You are a coding agent."},
                {"from":"human","value":"Open Cargo.toml"},
                {"from":"gpt","value":"<tool_call>{\"name\":\"read\",\"arguments\":{\"path\":\"Cargo.toml\"}}</tool_call>"},
                {"from":"tool","value":"[package]\nname = \"proxy\""},
                {"from":"gpt","value":"Cargo.toml defines the proxy crate."}
            ]
        });

        let scenarios =
            hermes_scenarios_from_row(&row, Some(7), &import_config()).expect("imports hermes row");

        assert_eq!(scenarios.len(), 2);
        let first = &scenarios[0];
        assert_eq!(first.dataset, "lambda/hermes-agent-reasoning-traces");
        assert_eq!(first.source_row, Some(7));
        assert_eq!(first.observed_kind, "tool_calls");
        assert_eq!(first.request.messages.len(), 2);
        assert_eq!(
            first.request.tools.as_ref().expect("tools")[0]
                .function
                .name,
            "read"
        );
        assert_eq!(
            first
                .metadata
                .get("observed_tool_calls")
                .and_then(Value::as_array)
                .expect("observed calls")
                .len(),
            1
        );
    }

    #[test]
    fn validate_proxy_response_fails_generic_unrecovered_fallback() {
        let request = ChatCompletionRequest {
            model: "test/model".to_string(),
            messages: vec![ChatMessage::new("user", "Read README.md")],
            tools: Some(vec![OpenAiTool {
                tool_type: "function".to_string(),
                function: OpenAiFunctionTool {
                    name: "read_file".to_string(),
                    description: None,
                    parameters: json!({
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"]
                    }),
                },
            }]),
            tool_choice: None,
            parallel_tool_calls: None,
            stream: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            max_completion_tokens: None,
            response_format: None,
            extra: Map::new(),
        };
        let response = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "The upstream model did not return a usable assistant response."
                }
            }]
        });
        let mut failures = Vec::new();
        let mut warnings = Vec::new();

        validate_proxy_response(&request, &response, &mut failures, &mut warnings);

        assert!(failures
            .iter()
            .any(|failure| failure.contains("unrecovered fallback")));
        assert!(warnings.is_empty());
    }

    #[test]
    fn validate_proxy_response_fails_tool_like_text_without_tool_calls() {
        let request = ChatCompletionRequest {
            model: "test/model".to_string(),
            messages: vec![ChatMessage::new("user", "Read README.md")],
            tools: Some(vec![OpenAiTool {
                tool_type: "function".to_string(),
                function: OpenAiFunctionTool {
                    name: "read_file".to_string(),
                    description: None,
                    parameters: json!({
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"]
                    }),
                },
            }]),
            tool_choice: None,
            parallel_tool_calls: None,
            stream: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            max_completion_tokens: None,
            response_format: None,
            extra: Map::new(),
        };
        let response = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "<tool_call name=\"read_file\">{\"path\":\"README.md\"}</tool_call>"
                }
            }]
        });
        let mut failures = Vec::new();
        let mut warnings = Vec::new();

        validate_proxy_response(&request, &response, &mut failures, &mut warnings);

        assert!(failures
            .iter()
            .any(|failure| failure.contains("tool-like text")));
        assert!(warnings.is_empty());
    }

    #[test]
    fn validate_proxy_response_fails_announced_tool_action_without_tool_call() {
        let request = ChatCompletionRequest {
            model: "test/model".to_string(),
            messages: vec![ChatMessage::new("user", "Read README.md")],
            tools: Some(vec![OpenAiTool {
                tool_type: "function".to_string(),
                function: OpenAiFunctionTool {
                    name: "read_file".to_string(),
                    description: None,
                    parameters: json!({
                        "type": "object",
                        "properties": {"path": {"type": "string"}},
                        "required": ["path"]
                    }),
                },
            }]),
            tool_choice: None,
            parallel_tool_calls: None,
            stream: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            max_completion_tokens: None,
            response_format: None,
            extra: Map::new(),
        };
        let response = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "Let me read the README first."
                }
            }]
        });
        let mut failures = Vec::new();
        let mut warnings = Vec::new();

        validate_proxy_response(&request, &response, &mut failures, &mut warnings);

        assert!(failures
            .iter()
            .any(|failure| failure.contains("announce a tool/action")));
        assert!(warnings.is_empty());
    }

    #[test]
    fn premature_tool_action_detector_requires_action_verb_near_marker() {
        assert!(looks_like_premature_tool_action(
            "Let me read the README first."
        ));
        assert!(looks_like_premature_tool_action(
            "I'll compile and run the chat server."
        ));
        assert!(!looks_like_premature_tool_action(
            "I attempted to save the API convention to memory, but memory is disabled. I will keep this information in my current context. Additionally, my search for `.graphql` files found nothing."
        ));
    }

    #[test]
    fn compare_outcome_classifies_baseline_vs_proxy() {
        assert_eq!(compare_outcome(true, true), "both_passed");
        assert_eq!(compare_outcome(false, true), "proxy_fixed_baseline_failure");
        assert_eq!(
            compare_outcome(true, false),
            "proxy_regressed_baseline_success"
        );
        assert_eq!(compare_outcome(false, false), "both_failed");
    }

    #[test]
    fn preview_truncates_unicode_on_char_boundaries() {
        let text = "x ".repeat(260) + "🥺";
        let shortened = preview(&text);

        assert!(shortened.ends_with("..."));
        assert!(shortened.len() < text.len());
    }
}
