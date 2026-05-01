use std::{net::SocketAddr, sync::Arc, time::Instant};

use anyhow::Result;
use axum::{
    body::Body,
    extract::State,
    http::{
        header::{CACHE_CONTROL, CONNECTION, CONTENT_TYPE},
        HeaderMap, HeaderValue, StatusCode,
    },
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Value};
use tower_http::{
    cors::CorsLayer,
    trace::{DefaultMakeSpan, DefaultOnResponse, TraceLayer},
};

use crate::{
    agents::{CorrectionAgent, DsrsCorrectionAgent, NoopCorrectionAgent},
    config::ProxyConfig,
    model_profile::resolve_profile,
    normalizer::normalize_request,
    openai::{
        chat_completion_response_to_sse, redacted_headers_for_trace, ChatCompletionRequest,
        ChatCompletionResponse, OpenAiError, OpenAiErrorResponse,
    },
    prompt_adapter::adapt_request,
    repair::{repair_response, RepairAction},
    response_interpreter::{interpret_response, ToolIntent},
    trace::{TraceRecord, TraceStore},
    upstream::{InboundAuth, UpstreamClient},
};

#[derive(Clone)]
pub struct Gateway {
    state: Arc<AppState>,
}

struct AppState {
    config: ProxyConfig,
    upstream: UpstreamClient,
    trace_store: TraceStore,
    correction_agent: Arc<dyn CorrectionAgent>,
}

impl Gateway {
    pub fn new(config: ProxyConfig) -> Result<Self> {
        let upstream = UpstreamClient::new(config.upstream.clone())?;
        let trace_store = TraceStore::new(config.trace.path.clone(), config.trace.enabled);
        let correction_agent: Arc<dyn CorrectionAgent> = if config.correction.enabled {
            Arc::new(DsrsCorrectionAgent::new(
                config.upstream.clone(),
                config.correction.clone(),
            ))
        } else {
            Arc::new(NoopCorrectionAgent)
        };

        tracing::info!(
            upstream_base_url = %config.upstream.base_url,
            upstream_timeout_seconds = config.upstream.timeout_seconds,
            trace_enabled = config.trace.enabled,
            trace_path = %config.trace.path.display(),
            correction_enabled = config.correction.enabled,
            policy_mode = ?config.policy.mode,
            "gateway configured"
        );

        Ok(Self {
            state: Arc::new(AppState {
                config,
                upstream,
                trace_store,
                correction_agent,
            }),
        })
    }

    pub fn router(&self) -> Router {
        Router::new()
            .route("/health", get(health))
            .route("/v1/models", get(models))
            .route("/v1/chat/completions", post(chat_completions))
            .layer(CorsLayer::permissive())
            .layer(
                TraceLayer::new_for_http()
                    .make_span_with(DefaultMakeSpan::new().level(tracing::Level::INFO))
                    .on_response(DefaultOnResponse::new().level(tracing::Level::INFO)),
            )
            .with_state(Arc::clone(&self.state))
    }

    pub async fn serve(self, bind: SocketAddr) -> Result<()> {
        let listener = tokio::net::TcpListener::bind(bind).await?;
        tracing::info!("listening on http://{bind}");
        axum::serve(listener, self.router()).await?;
        Ok(())
    }
}

async fn health() -> Json<serde_json::Value> {
    tracing::debug!("health check received");
    Json(json!({"status":"ok"}))
}

async fn models(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let started = Instant::now();
    let auth = InboundAuth::from_headers(&headers);
    tracing::info!(client_auth = auth.source_label(), "models request received");

    match state.upstream.models(&auth).await {
        Ok(response) => {
            tracing::info!(
                elapsed_ms = started.elapsed().as_millis(),
                upstream_models = response
                    .get("data")
                    .and_then(|value| value.as_array())
                    .map_or(0, Vec::len),
                "models request completed"
            );
            json_response(StatusCode::OK, response, None)
        }
        Err(error) => {
            tracing::warn!(
                elapsed_ms = started.elapsed().as_millis(),
                error = %error,
                "models request failed"
            );
            openai_error(StatusCode::BAD_GATEWAY, &error.to_string(), None)
        }
    }
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    let request_started = Instant::now();
    let client_requested_stream = request.stream.unwrap_or(false);
    let stream_include_usage = request
        .extra
        .get("stream_options")
        .and_then(|value| value.get("include_usage"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let metadata = json!({ "headers": redacted_headers_for_trace(&headers) });
    let mut trace = TraceRecord::new(request.clone(), metadata);
    let trace_id = trace.trace_id.clone();

    tracing::info!(
        %trace_id,
        model = %request.model,
        messages = request.messages.len(),
        tools = request.tools.as_ref().map_or(0, Vec::len),
        stream_requested = client_requested_stream,
        stream_include_usage,
        "chat completion request received"
    );
    tracing::debug!(
        %trace_id,
        max_tokens = ?request.max_tokens,
        max_completion_tokens = ?request.max_completion_tokens,
        temperature = ?request.temperature,
        top_p = ?request.top_p,
        tool_choice = %tool_choice_label(request.tool_choice.as_ref()),
        extra_fields = request.extra.len(),
        headers = ?redacted_headers_for_trace(&headers),
        "chat completion request details"
    );
    tracing::trace!(
        %trace_id,
        message_roles = ?message_roles(&request.messages),
        tool_names = ?tool_names(request.tools.as_deref()),
        "chat completion request shape"
    );

    let normalized = match normalize_request(request) {
        Ok(normalized) => normalized,
        Err(error) => {
            tracing::warn!(
                %trace_id,
                error = %error,
                elapsed_ms = request_started.elapsed().as_millis(),
                "request normalization failed"
            );
            trace.error = Some(error.to_string());
            let trace_id = append_trace(&state, trace).await;
            return openai_error(StatusCode::BAD_REQUEST, &error.to_string(), Some(trace_id));
        }
    };
    trace.normalized = Some(normalized.clone());
    tracing::debug!(
        %trace_id,
        model = %normalized.model,
        messages = normalized.messages.len(),
        tools = normalized.tools.len(),
        parallel_tool_calls = normalized.parallel_tool_calls,
        tool_choice = %tool_choice_label(normalized.tool_choice.as_ref()),
        "request normalized"
    );

    let profile = resolve_profile(&normalized.model, &state.config.model_profiles);
    trace.profile = Some(profile.clone());
    tracing::info!(
        %trace_id,
        profile = %profile.name,
        tool_mode = ?profile.tool_mode,
        tool_format = ?profile.tool_format,
        supports_parallel_tool_calls = profile.supports_parallel_tool_calls,
        max_correction_passes = profile.max_correction_passes,
        "model profile selected"
    );

    let adapted = match adapt_request(&normalized, &profile) {
        Ok(adapted) => adapted,
        Err(error) => {
            tracing::warn!(
                %trace_id,
                error = %error,
                elapsed_ms = request_started.elapsed().as_millis(),
                "request adaptation failed"
            );
            trace.error = Some(error.to_string());
            let trace_id = append_trace(&state, trace).await;
            return openai_error(StatusCode::BAD_REQUEST, &error.to_string(), Some(trace_id));
        }
    };
    trace.adapted_request = Some(adapted.clone());
    tracing::info!(
        %trace_id,
        adapter_mode = ?adapted.mode,
        upstream_stream = ?adapted.upstream_request.stream,
        upstream_tools = adapted.upstream_request.tools.as_ref().map_or(0, Vec::len),
        instruction_injected = adapted.injected_instruction.is_some(),
        instruction_len = adapted.injected_instruction.as_ref().map_or(0, String::len),
        "request adapted for upstream"
    );
    tracing::trace!(
        %trace_id,
        upstream_message_roles = ?message_roles(&adapted.upstream_request.messages),
        upstream_tool_names = ?tool_names(adapted.upstream_request.tools.as_deref()),
        "adapted upstream request shape"
    );

    let auth = InboundAuth::from_headers(&headers);
    let upstream_started = Instant::now();
    tracing::info!(
        %trace_id,
        client_auth = auth.source_label(),
        "calling upstream chat completions"
    );
    let upstream_response = match state
        .upstream
        .chat_completions(&adapted.upstream_request, &auth)
        .await
    {
        Ok(response) => response,
        Err(error) => {
            tracing::warn!(
                %trace_id,
                error = %error,
                elapsed_ms = upstream_started.elapsed().as_millis(),
                "upstream chat completions failed"
            );
            trace.error = Some(error.to_string());
            let trace_id = append_trace(&state, trace).await;
            return openai_error(StatusCode::BAD_GATEWAY, &error.to_string(), Some(trace_id));
        }
    };
    trace.upstream_response = Some(upstream_response.clone());
    tracing::info!(
        %trace_id,
        elapsed_ms = upstream_started.elapsed().as_millis(),
        upstream_model = %upstream_response.model,
        choices = upstream_response.choices.len(),
        finish_reason = ?first_finish_reason(&upstream_response),
        usage_present = upstream_response.usage.is_some(),
        "upstream response received"
    );

    let interpreted = interpret_response(&upstream_response, &normalized.tools);
    trace.interpreted = Some(interpreted.clone());
    tracing::info!(
        %trace_id,
        finish_reason = ?interpreted.finish_reason,
        content_len = interpreted.content.as_ref().map_or(0, String::len),
        reasoning_present = interpreted.reasoning.is_some(),
        tool_intents = interpreted.tool_intents.len(),
        suspicious_stop = interpreted.suspicious_stop,
        parse_events = interpreted.parse_events.len(),
        "upstream response interpreted"
    );
    if interpreted.suspicious_stop {
        tracing::warn!(
            %trace_id,
            parse_events = ?interpreted.parse_events,
            "interpreter marked response as suspicious stop"
        );
    } else {
        tracing::debug!(
            %trace_id,
            parse_events = ?interpreted.parse_events,
            "interpreter parse events"
        );
    }
    tracing::trace!(
        %trace_id,
        tool_intents = ?tool_intent_summaries(&interpreted.tool_intents),
        "interpreted tool intents"
    );

    let repair = match repair_response(
        &state.config,
        &normalized,
        &profile,
        &upstream_response,
        &interpreted,
        state.correction_agent.as_ref(),
        auth.api_key_value(),
    )
    .await
    {
        Ok(outcome) => outcome,
        Err(error) => {
            tracing::error!(
                %trace_id,
                error = %error,
                elapsed_ms = request_started.elapsed().as_millis(),
                "repair pipeline failed"
            );
            trace.error = Some(error.to_string());
            let trace_id = append_trace(&state, trace).await;
            return openai_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                &error.to_string(),
                Some(trace_id),
            );
        }
    };
    tracing::info!(
        %trace_id,
        repair_actions = repair.actions.len(),
        final_finish_reason = ?first_finish_reason(&repair.final_response),
        final_tool_calls = first_tool_call_count(&repair.final_response),
        final_content_len = first_content_len(&repair.final_response),
        "repair pipeline completed"
    );
    for action in &repair.actions {
        log_repair_action(&trace_id, action);
    }

    trace.repair_actions = repair.actions.clone();
    trace.final_response = Some(repair.final_response.clone());
    let trace_id = append_trace(&state, trace).await;

    if client_requested_stream {
        tracing::info!(
            %trace_id,
            elapsed_ms = request_started.elapsed().as_millis(),
            response_mode = "sse",
            "chat completion response sent"
        );
        sse_response(&repair.final_response, stream_include_usage, Some(trace_id))
    } else {
        tracing::info!(
            %trace_id,
            elapsed_ms = request_started.elapsed().as_millis(),
            response_mode = "json",
            "chat completion response sent"
        );
        json_response(StatusCode::OK, repair.final_response, Some(trace_id))
    }
}

async fn append_trace(state: &AppState, trace: TraceRecord) -> String {
    let fallback = trace.trace_id.clone();
    match state.trace_store.append(trace).await {
        Ok(trace_id) => trace_id,
        Err(error) => {
            tracing::warn!(
                trace_id = %fallback,
                error = %error,
                "failed to append trace"
            );
            fallback
        }
    }
}

fn openai_error(status: StatusCode, message: &str, trace_id: Option<String>) -> Response {
    json_response(
        status,
        OpenAiErrorResponse {
            error: OpenAiError {
                message: message.to_string(),
                error_type: "model_correction_proxy_error".to_string(),
                param: None,
                code: None,
            },
        },
        trace_id,
    )
}

fn json_response<T: serde::Serialize>(
    status: StatusCode,
    body: T,
    trace_id: Option<String>,
) -> Response {
    let mut response = (status, Json(body)).into_response();
    if let Some(trace_id) = trace_id {
        if let Ok(value) = HeaderValue::from_str(&trace_id) {
            response
                .headers_mut()
                .insert("x-model-correction-trace-id", value);
        }
    }
    response
}

fn message_roles(messages: &[crate::openai::ChatMessage]) -> Vec<&str> {
    messages
        .iter()
        .map(|message| message.role.as_str())
        .collect()
}

fn tool_names(tools: Option<&[crate::openai::OpenAiTool]>) -> Vec<&str> {
    tools
        .unwrap_or_default()
        .iter()
        .map(|tool| tool.function.name.as_str())
        .collect()
}

fn tool_choice_label(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(value)) => value.clone(),
        Some(value) => value.to_string(),
        None => "auto".to_string(),
    }
}

fn first_finish_reason(response: &ChatCompletionResponse) -> Option<&str> {
    response
        .choices
        .first()
        .and_then(|choice| choice.finish_reason.as_deref())
}

fn first_tool_call_count(response: &ChatCompletionResponse) -> usize {
    response
        .choices
        .first()
        .and_then(|choice| choice.message.tool_calls.as_ref())
        .map_or(0, Vec::len)
}

fn first_content_len(response: &ChatCompletionResponse) -> usize {
    response
        .choices
        .first()
        .and_then(|choice| choice.message.content_text())
        .map_or(0, |content| content.len())
}

fn tool_intent_summaries(intents: &[ToolIntent]) -> Vec<String> {
    intents
        .iter()
        .map(|intent| {
            format!(
                "{}:{:?}:confidence={:.2}:args_present={}",
                intent.name,
                intent.source,
                intent.confidence,
                intent.arguments.is_some()
            )
        })
        .collect()
}

fn log_repair_action(trace_id: &str, action: &RepairAction) {
    if repair_action_deserves_warning(&action.action) {
        tracing::warn!(
            %trace_id,
            action = %action.action,
            confidence = action.confidence,
            reason = %action.reason,
            "repair action applied"
        );
    } else {
        tracing::info!(
            %trace_id,
            action = %action.action,
            confidence = action.confidence,
            reason = %action.reason,
            "repair action applied"
        );
    }
}

fn repair_action_deserves_warning(action: &str) -> bool {
    action.contains("failed")
        || action.contains("dropped")
        || action.contains("unrecoverable")
        || action.contains("suppressed")
        || action.contains("missing")
}

fn sse_response(
    body: &ChatCompletionResponse,
    include_usage: bool,
    trace_id: Option<String>,
) -> Response {
    let sse = match chat_completion_response_to_sse(body, include_usage) {
        Ok(sse) => sse,
        Err(error) => {
            return openai_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("failed to serialize streaming response: {error}"),
                trace_id,
            )
        }
    };

    let mut response = match Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, "text/event-stream; charset=utf-8")
        .header(CACHE_CONTROL, "no-cache")
        .header(CONNECTION, "keep-alive")
        .header("x-accel-buffering", "no")
        .body(Body::from(sse))
    {
        Ok(response) => response,
        Err(error) => {
            return openai_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                &format!("failed to build streaming response: {error}"),
                trace_id,
            )
        }
    };

    if let Some(trace_id) = trace_id {
        if let Ok(value) = HeaderValue::from_str(&trace_id) {
            response
                .headers_mut()
                .insert("x-model-correction-trace-id", value);
        }
    }
    response
}
