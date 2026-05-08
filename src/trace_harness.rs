use std::{
    collections::{hash_map::DefaultHasher, BTreeMap, VecDeque},
    fs::OpenOptions,
    hash::{Hash, Hasher},
    io::Write,
    net::SocketAddr,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Context, Result};
use axum::Router;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use tokio::task::JoinHandle;

use crate::{
    config::ProxyConfig,
    gateway::Gateway,
    model_profile::ProviderRouting,
    openai::{ChatCompletionRequest, ChatMessage, OpenAiFunctionTool, OpenAiTool, OpenAiToolCall},
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

    let client = reqwest::Client::new();
    let mut cases = Vec::new();
    for scenario in scenarios.into_iter().take(config.limit) {
        let mut request = scenario.request.clone();
        if let Some(model) = &config.model {
            request.model = model.clone();
        }
        if let Some(provider) = &config.provider {
            if !provider.is_empty() {
                request
                    .extra
                    .insert("provider".to_string(), serde_json::to_value(provider)?);
            }
        }
        request.stream = Some(false);
        let report = run_one_scenario(&client, &proxy_url, scenario, request).await;
        cases.push(report);
    }

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
        total: cases.len(),
        passed,
        failed,
        warnings,
        cases,
    };
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
) -> TraceHarnessCaseReport {
    let model = request.model.clone();
    let response = client
        .post(format!("{proxy_url}/v1/chat/completions"))
        .json(&request)
        .send()
        .await;

    let mut failures = Vec::new();
    let mut warnings = Vec::new();
    let mut final_content_len = 0usize;
    let mut final_content_preview = None;
    let mut final_tool_calls = Vec::new();

    match response {
        Ok(response) if response.status().is_success() => match response.text().await {
            Ok(body) => match serde_json::from_str::<Value>(&body) {
                Ok(value) => {
                    validate_proxy_response(&request, &value, &mut failures, &mut warnings);
                    if let Some(content) = value
                        .pointer("/choices/0/message/content")
                        .and_then(Value::as_str)
                    {
                        final_content_len = content.len();
                        final_content_preview = Some(preview(content));
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
                            final_tool_calls.push(format!("{name} {args}"));
                        }
                    }
                }
                Err(error) => {
                    failures.push(format!("proxy returned invalid JSON: {error}: {body}"))
                }
            },
            Err(error) => failures.push(format!("failed to read proxy response body: {error}")),
        },
        Ok(response) => {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            failures.push(format!("proxy returned HTTP {status}: {}", preview(&body)));
        }
        Err(error) => failures.push(format!("proxy request failed: {error}")),
    }

    TraceHarnessCaseReport {
        scenario_id: scenario.id,
        dataset: scenario.dataset,
        source_id: scenario.source_id,
        turn_index: scenario.turn_index,
        model,
        passed: failures.is_empty(),
        failures,
        warnings,
        final_content_len,
        final_content_preview,
        final_tool_calls,
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

async fn start_proxy(
    mut config: ProxyConfig,
    output_path: &Path,
) -> Result<(String, Option<JoinHandle<()>>)> {
    config.upstream.api_key = config
        .upstream
        .api_key
        .clone()
        .or_else(|| env_key("OPENROUTER_API_KEY"))
        .or_else(|| read_env_key("OPENROUTER_API_KEY"))
        .or_else(|| env_key("MCP_UPSTREAM_API_KEY"))
        .or_else(|| read_env_key("MCP_UPSTREAM_API_KEY"));
    if config.upstream.api_key.is_none() {
        return Err(anyhow!(
            "OPENROUTER_API_KEY or MCP_UPSTREAM_API_KEY is required for in-process live proxy runs"
        ));
    }
    if config.upstream.base_url.trim().is_empty() {
        config.upstream.base_url = "https://openrouter.ai/api/v1".to_string();
    }
    let stem = output_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("trace-harness-run");
    let parent = output_path.parent().unwrap_or_else(|| Path::new("."));
    tokio::fs::create_dir_all(parent).await?;
    config.trace.path = parent.join(format!("{stem}-proxy-traces.local.jsonl"));
    config.trace.correction_path = parent.join(format!("{stem}-corrections.local.jsonl"));

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
    fn preview_truncates_unicode_on_char_boundaries() {
        let text = "x ".repeat(260) + "🥺";
        let shortened = preview(&text);

        assert!(shortened.ends_with("..."));
        assert!(shortened.len() < text.len());
    }
}
