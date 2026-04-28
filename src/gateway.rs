use std::{net::SocketAddr, sync::Arc};

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
use serde_json::json;
use tower_http::{cors::CorsLayer, trace::TraceLayer};

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
    repair::repair_response,
    response_interpreter::interpret_response,
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
            .layer(TraceLayer::new_for_http())
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
    Json(json!({"status":"ok"}))
}

async fn models(State(state): State<Arc<AppState>>, headers: HeaderMap) -> Response {
    let auth = InboundAuth::from_headers(&headers);

    match state.upstream.models(&auth).await {
        Ok(response) => json_response(StatusCode::OK, response, None),
        Err(error) => openai_error(StatusCode::BAD_GATEWAY, &error.to_string(), None),
    }
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<ChatCompletionRequest>,
) -> Response {
    let client_requested_stream = request.stream.unwrap_or(false);
    let stream_include_usage = request
        .extra
        .get("stream_options")
        .and_then(|value| value.get("include_usage"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);
    let metadata = json!({ "headers": redacted_headers_for_trace(&headers) });
    let mut trace = TraceRecord::new(request.clone(), metadata);

    let normalized = match normalize_request(request) {
        Ok(normalized) => normalized,
        Err(error) => {
            trace.error = Some(error.to_string());
            let trace_id = append_trace(&state, trace).await;
            return openai_error(StatusCode::BAD_REQUEST, &error.to_string(), Some(trace_id));
        }
    };
    trace.normalized = Some(normalized.clone());

    let profile = resolve_profile(&normalized.model, &state.config.model_profiles);
    trace.profile = Some(profile.clone());

    let adapted = match adapt_request(&normalized, &profile) {
        Ok(adapted) => adapted,
        Err(error) => {
            trace.error = Some(error.to_string());
            let trace_id = append_trace(&state, trace).await;
            return openai_error(StatusCode::BAD_REQUEST, &error.to_string(), Some(trace_id));
        }
    };
    trace.adapted_request = Some(adapted.clone());

    let auth = InboundAuth::from_headers(&headers);
    let upstream_response = match state
        .upstream
        .chat_completions(&adapted.upstream_request, &auth)
        .await
    {
        Ok(response) => response,
        Err(error) => {
            trace.error = Some(error.to_string());
            let trace_id = append_trace(&state, trace).await;
            return openai_error(StatusCode::BAD_GATEWAY, &error.to_string(), Some(trace_id));
        }
    };
    trace.upstream_response = Some(upstream_response.clone());

    let interpreted = interpret_response(&upstream_response, &normalized.tools);
    trace.interpreted = Some(interpreted.clone());

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
            trace.error = Some(error.to_string());
            let trace_id = append_trace(&state, trace).await;
            return openai_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                &error.to_string(),
                Some(trace_id),
            );
        }
    };

    trace.repair_actions = repair.actions.clone();
    trace.final_response = Some(repair.final_response.clone());
    let trace_id = append_trace(&state, trace).await;

    if client_requested_stream {
        sse_response(&repair.final_response, stream_include_usage, Some(trace_id))
    } else {
        json_response(StatusCode::OK, repair.final_response, Some(trace_id))
    }
}

async fn append_trace(state: &AppState, trace: TraceRecord) -> String {
    let fallback = trace.trace_id.clone();
    match state.trace_store.append(trace).await {
        Ok(trace_id) => trace_id,
        Err(error) => {
            tracing::warn!(%error, "failed to append trace");
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
