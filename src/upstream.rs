use std::{collections::BTreeMap, fmt, time::Instant};

use anyhow::{Context, Result};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    config::UpstreamConfig,
    openai::{ChatCompletionRequest, ChatCompletionResponse, ChatMessage, OpenAiTool},
};

#[derive(Debug, Clone, Default)]
pub struct InboundAuth {
    authorization: Option<HeaderValue>,
    x_api_key: Option<HeaderValue>,
    api_key: Option<HeaderValue>,
}

impl InboundAuth {
    pub fn from_headers(headers: &HeaderMap) -> Self {
        Self {
            authorization: headers.get(AUTHORIZATION).cloned(),
            x_api_key: headers.get("x-api-key").cloned(),
            api_key: headers.get("api-key").cloned(),
        }
    }

    pub fn api_key_value(&self) -> Option<String> {
        self.authorization
            .as_ref()
            .and_then(header_api_key_value)
            .or_else(|| self.x_api_key.as_ref().and_then(header_api_key_value))
            .or_else(|| self.api_key.as_ref().and_then(header_api_key_value))
    }

    pub fn source_label(&self) -> &'static str {
        if self.authorization.as_ref().is_some_and(non_empty) {
            "authorization"
        } else if self.x_api_key.as_ref().is_some_and(non_empty) {
            "x-api-key"
        } else if self.api_key.as_ref().is_some_and(non_empty) {
            "api-key"
        } else {
            "none"
        }
    }

    #[cfg(test)]
    fn authorization(value: impl AsRef<str>) -> Self {
        let mut headers = HeaderMap::new();
        headers.insert(
            AUTHORIZATION,
            HeaderValue::from_str(value.as_ref()).expect("valid auth header"),
        );
        Self::from_headers(&headers)
    }

    #[cfg(test)]
    fn x_api_key(value: impl AsRef<str>) -> Self {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(value.as_ref()).expect("valid x-api-key header"),
        );
        Self::from_headers(&headers)
    }

    #[cfg(test)]
    fn api_key(value: impl AsRef<str>) -> Self {
        let mut headers = HeaderMap::new();
        headers.insert(
            "api-key",
            HeaderValue::from_str(value.as_ref()).expect("valid api-key header"),
        );
        Self::from_headers(&headers)
    }
}

#[derive(Clone)]
pub struct UpstreamClient {
    config: UpstreamConfig,
    http: reqwest::Client,
}

impl UpstreamClient {
    pub fn new(config: UpstreamConfig) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(config.timeout())
            .build()
            .context("failed to build upstream HTTP client")?;
        Ok(Self { config, http })
    }

    pub async fn chat_completions(
        &self,
        request: &ChatCompletionRequest,
        inbound_auth: &InboundAuth,
    ) -> Result<ChatCompletionResponse, UpstreamError> {
        let started = Instant::now();
        let url = self.endpoint_url("chat/completions");
        let auth_source = self.effective_auth_source(inbound_auth);
        tracing::info!(
            endpoint = "chat/completions",
            model = %request.model,
            messages = request.messages.len(),
            tools = request.tools.as_ref().map_or(0, Vec::len),
            stream = request.stream.unwrap_or(false),
            auth_source,
            "upstream request started"
        );
        tracing::debug!(
            endpoint = "chat/completions",
            url = %url,
            timeout_seconds = self.config.timeout_seconds,
            max_tokens = ?request.max_tokens,
            max_completion_tokens = ?request.max_completion_tokens,
            temperature = ?request.temperature,
            top_p = ?request.top_p,
            extra_fields = request.extra.len(),
            "upstream request details"
        );
        tracing::trace!(
            endpoint = "chat/completions",
            message_roles = ?message_roles(&request.messages),
            tool_names = ?tool_names(request.tools.as_deref()),
            "upstream request shape"
        );

        let mut builder = self
            .http
            .post(url)
            .header(CONTENT_TYPE, "application/json")
            .header("X-Title", "attune")
            .json(request);

        builder = self.apply_auth(builder, inbound_auth);

        let response = match builder.send().await {
            Ok(response) => response,
            Err(error) => {
                tracing::warn!(
                    endpoint = "chat/completions",
                    elapsed_ms = started.elapsed().as_millis(),
                    error = %error,
                    "upstream transport failed"
                );
                return Err(UpstreamError::Transport(error));
            }
        };
        let status = response.status();
        let diagnostics = UpstreamDiagnostics::from_headers(response.headers());
        let body = match response.text().await {
            Ok(body) => body,
            Err(error) => {
                tracing::warn!(
                    endpoint = "chat/completions",
                    status = status.as_u16(),
                    elapsed_ms = started.elapsed().as_millis(),
                    error = %error,
                    "failed to read upstream response body"
                );
                return Err(UpstreamError::Transport(error));
            }
        };
        if !status.is_success() {
            tracing::warn!(
                endpoint = "chat/completions",
                status = status.as_u16(),
                elapsed_ms = started.elapsed().as_millis(),
                body_len = body.len(),
                body_preview = %preview(&body),
                diagnostic_headers = ?diagnostics.headers,
                "upstream returned non-success status"
            );
            return Err(UpstreamError::Status {
                status: status.as_u16(),
                body,
                diagnostics,
            });
        }

        let decoded = match serde_json::from_str::<ChatCompletionResponse>(&body) {
            Ok(decoded) => decoded,
            Err(source) => {
                tracing::warn!(
                    endpoint = "chat/completions",
                    status = status.as_u16(),
                    elapsed_ms = started.elapsed().as_millis(),
                    body_len = body.len(),
                    body_preview = %preview(&body),
                    diagnostic_headers = ?diagnostics.headers,
                    error = %source,
                    "failed to decode upstream response"
                );
                return Err(UpstreamError::Decode {
                    source,
                    body,
                    diagnostics,
                });
            }
        };

        tracing::info!(
            endpoint = "chat/completions",
            status = status.as_u16(),
            elapsed_ms = started.elapsed().as_millis(),
            response_model = %decoded.model,
            choices = decoded.choices.len(),
            finish_reason = ?first_finish_reason(&decoded),
            usage_present = decoded.usage.is_some(),
            response_provider = ?response_provider(&decoded),
            diagnostic_headers = ?diagnostics.headers,
            "upstream request completed"
        );
        Ok(decoded)
    }

    pub async fn models(
        &self,
        inbound_auth: &InboundAuth,
    ) -> Result<serde_json::Value, UpstreamError> {
        let started = Instant::now();
        let url = self.endpoint_url("models");
        let auth_source = self.effective_auth_source(inbound_auth);
        tracing::info!(endpoint = "models", auth_source, "upstream request started");
        tracing::debug!(
            endpoint = "models",
            url = %url,
            timeout_seconds = self.config.timeout_seconds,
            "upstream request details"
        );

        let builder = self.http.get(url).header("X-Title", "attune");
        let builder = self.apply_auth(builder, inbound_auth);

        let response = match builder.send().await {
            Ok(response) => response,
            Err(error) => {
                tracing::warn!(
                    endpoint = "models",
                    elapsed_ms = started.elapsed().as_millis(),
                    error = %error,
                    "upstream transport failed"
                );
                return Err(UpstreamError::Transport(error));
            }
        };
        let status = response.status();
        let diagnostics = UpstreamDiagnostics::from_headers(response.headers());
        let body = match response.text().await {
            Ok(body) => body,
            Err(error) => {
                tracing::warn!(
                    endpoint = "models",
                    status = status.as_u16(),
                    elapsed_ms = started.elapsed().as_millis(),
                    error = %error,
                    "failed to read upstream response body"
                );
                return Err(UpstreamError::Transport(error));
            }
        };
        if !status.is_success() {
            tracing::warn!(
                endpoint = "models",
                status = status.as_u16(),
                elapsed_ms = started.elapsed().as_millis(),
                body_len = body.len(),
                body_preview = %preview(&body),
                diagnostic_headers = ?diagnostics.headers,
                "upstream returned non-success status"
            );
            return Err(UpstreamError::Status {
                status: status.as_u16(),
                body,
                diagnostics,
            });
        }

        let decoded = match serde_json::from_str::<serde_json::Value>(&body) {
            Ok(decoded) => decoded,
            Err(source) => {
                tracing::warn!(
                    endpoint = "models",
                    status = status.as_u16(),
                    elapsed_ms = started.elapsed().as_millis(),
                    body_len = body.len(),
                    body_preview = %preview(&body),
                    diagnostic_headers = ?diagnostics.headers,
                    error = %source,
                    "failed to decode upstream response"
                );
                return Err(UpstreamError::Decode {
                    source,
                    body,
                    diagnostics,
                });
            }
        };

        tracing::info!(
            endpoint = "models",
            status = status.as_u16(),
            elapsed_ms = started.elapsed().as_millis(),
            models = decoded
                .get("data")
                .and_then(|value| value.as_array())
                .map_or(0, Vec::len),
            "upstream request completed"
        );
        Ok(decoded)
    }

    fn endpoint_url(&self, endpoint: &str) -> String {
        let mut base = self.config.base_url.trim_end_matches('/').to_string();
        if let Some(root) = base.strip_suffix("/chat/completions") {
            base = root.to_string();
        }
        format!("{}/{}", base, endpoint.trim_start_matches('/'))
    }

    fn apply_auth(
        &self,
        mut builder: reqwest::RequestBuilder,
        inbound_auth: &InboundAuth,
    ) -> reqwest::RequestBuilder {
        if inbound_auth.has_client_auth() {
            if let Some(auth) = inbound_auth
                .authorization
                .as_ref()
                .filter(|value| non_empty(value))
            {
                builder = builder.header(AUTHORIZATION, auth.clone());
            } else if let Some(auth) = inbound_auth.bearer_from_api_key_header() {
                builder = builder.header(AUTHORIZATION, auth);
            }

            if let Some(api_key) = inbound_auth
                .x_api_key
                .as_ref()
                .filter(|value| non_empty(value))
            {
                builder = builder.header("x-api-key", api_key.clone());
            }
            if let Some(api_key) = inbound_auth
                .api_key
                .as_ref()
                .filter(|value| non_empty(value))
            {
                builder = builder.header("api-key", api_key.clone());
            }
        } else if let Some(api_key) = self
            .config
            .api_key
            .as_ref()
            .filter(|key| !key.trim().is_empty())
        {
            builder = builder.bearer_auth(api_key);
        }
        builder
    }

    fn effective_auth_source(&self, inbound_auth: &InboundAuth) -> &'static str {
        if inbound_auth.has_client_auth() {
            inbound_auth.source_label()
        } else if self
            .config
            .api_key
            .as_ref()
            .is_some_and(|key| !key.trim().is_empty())
        {
            "configured-api-key"
        } else {
            "none"
        }
    }
}

impl InboundAuth {
    fn has_client_auth(&self) -> bool {
        [&self.authorization, &self.x_api_key, &self.api_key]
            .into_iter()
            .flatten()
            .any(non_empty)
    }

    fn bearer_from_api_key_header(&self) -> Option<HeaderValue> {
        self.x_api_key
            .as_ref()
            .or(self.api_key.as_ref())
            .and_then(api_key_to_bearer)
    }
}

fn non_empty(value: &HeaderValue) -> bool {
    value
        .to_str()
        .map_or(true, |value| !value.trim().is_empty())
}

fn api_key_to_bearer(value: &HeaderValue) -> Option<HeaderValue> {
    let value = header_api_key_value(value)?;
    HeaderValue::from_str(&format!("Bearer {value}")).ok()
}

fn header_api_key_value(value: &HeaderValue) -> Option<String> {
    let value = value.to_str().ok()?.trim();
    if value.is_empty() {
        return None;
    }

    if value.to_ascii_lowercase().starts_with("bearer ") {
        let token = value["bearer ".len()..].trim();
        if token.is_empty() {
            None
        } else {
            Some(token.to_string())
        }
    } else {
        Some(value.to_string())
    }
}

fn message_roles(messages: &[ChatMessage]) -> Vec<&str> {
    messages
        .iter()
        .map(|message| message.role.as_str())
        .collect()
}

fn tool_names(tools: Option<&[OpenAiTool]>) -> Vec<&str> {
    tools
        .unwrap_or_default()
        .iter()
        .map(|tool| tool.function.name.as_str())
        .collect()
}

fn first_finish_reason(response: &ChatCompletionResponse) -> Option<&str> {
    response
        .choices
        .first()
        .and_then(|choice| choice.finish_reason.as_deref())
}

fn response_provider(response: &ChatCompletionResponse) -> Option<&str> {
    response
        .extra
        .get("provider")
        .and_then(|value| value.as_str())
}

fn diagnostic_headers(headers: &HeaderMap) -> BTreeMap<String, String> {
    headers
        .iter()
        .filter_map(|(name, value)| {
            let name = name.as_str().to_ascii_lowercase();
            if !is_diagnostic_header(&name) {
                return None;
            }
            let value = value
                .to_str()
                .map(preview)
                .unwrap_or_else(|_| "<non-utf8>".to_string());
            Some((name, value))
        })
        .collect()
}

fn is_diagnostic_header(name: &str) -> bool {
    name.starts_with("x-openrouter")
        || name.starts_with("openrouter")
        || name.contains("provider")
        || name.starts_with("x-ratelimit")
        || matches!(name, "retry-after" | "cf-ray" | "x-request-id")
}

fn preview(value: &str) -> String {
    const MAX: usize = 512;
    let mut preview = value.chars().take(MAX).collect::<String>();
    if value.chars().count() > MAX {
        preview.push_str("...");
    }
    preview
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct UpstreamDiagnostics {
    headers: BTreeMap<String, String>,
}

impl UpstreamDiagnostics {
    fn from_headers(headers: &HeaderMap) -> Self {
        Self {
            headers: diagnostic_headers(headers),
        }
    }
}

impl fmt::Display for UpstreamDiagnostics {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.headers.is_empty() {
            return Ok(());
        }
        write!(formatter, "; diagnostic_headers={:?}", self.headers)
    }
}

#[derive(Debug, Error)]
pub enum UpstreamError {
    #[error("upstream transport error: {0}")]
    Transport(#[from] reqwest::Error),
    #[error("upstream returned HTTP {status}: {body}{diagnostics}")]
    Status {
        status: u16,
        body: String,
        diagnostics: UpstreamDiagnostics,
    },
    #[error("failed to decode upstream response: {source}; body: {body}{diagnostics}")]
    Decode {
        source: serde_json::Error,
        body: String,
        diagnostics: UpstreamDiagnostics,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UpstreamFailure {
    pub message: String,
}

impl From<UpstreamError> for UpstreamFailure {
    fn from(value: UpstreamError) -> Self {
        Self {
            message: value.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use httpmock::prelude::*;
    use serde_json::json;

    use super::*;

    fn client(base_url: impl Into<String>, api_key: Option<String>) -> UpstreamClient {
        UpstreamClient::new(UpstreamConfig {
            base_url: base_url.into(),
            api_key,
            timeout_seconds: 5,
        })
        .expect("client should build")
    }

    #[test]
    fn endpoint_url_preserves_openai_compatible_api_root() {
        assert_eq!(
            client("https://openrouter.ai/api/v1", None).endpoint_url("models"),
            "https://openrouter.ai/api/v1/models"
        );
        assert_eq!(
            client("http://127.0.0.1:18081/v1/", None).endpoint_url("models"),
            "http://127.0.0.1:18081/v1/models"
        );
        assert_eq!(
            client("http://127.0.0.1:18081/v1/chat/completions", None).endpoint_url("models"),
            "http://127.0.0.1:18081/v1/models"
        );
    }

    #[test]
    fn api_key_value_extracts_client_key_for_correction_agent() {
        assert_eq!(
            InboundAuth::authorization("Bearer inbound-key")
                .api_key_value()
                .as_deref(),
            Some("inbound-key")
        );
        assert_eq!(
            InboundAuth::x_api_key("inbound-key")
                .api_key_value()
                .as_deref(),
            Some("inbound-key")
        );
    }

    #[test]
    fn diagnostic_headers_capture_openrouter_provider_context() {
        let mut headers = HeaderMap::new();
        headers.insert(
            "x-openrouter-provider",
            HeaderValue::from_static("deepinfra"),
        );
        headers.insert("x-ratelimit-remaining", HeaderValue::from_static("42"));
        headers.insert("authorization", HeaderValue::from_static("Bearer secret"));

        let diagnostics = UpstreamDiagnostics::from_headers(&headers);

        assert_eq!(
            diagnostics
                .headers
                .get("x-openrouter-provider")
                .map(String::as_str),
            Some("deepinfra")
        );
        assert_eq!(
            diagnostics
                .headers
                .get("x-ratelimit-remaining")
                .map(String::as_str),
            Some("42")
        );
        assert!(!diagnostics.headers.contains_key("authorization"));
    }

    #[tokio::test]
    async fn decode_error_includes_diagnostic_headers() {
        let server = MockServer::start_async().await;
        let chat_mock = server
            .mock_async(|when, then| {
                when.method(POST).path("/v1/chat/completions");
                then.status(200)
                    .header("x-openrouter-provider", "deepinfra")
                    .body("not-json");
            })
            .await;

        let error = client(format!("{}/v1", server.base_url()), None)
            .chat_completions(
                &ChatCompletionRequest {
                    model: "test-model".to_string(),
                    messages: Vec::new(),
                    tools: None,
                    tool_choice: None,
                    parallel_tool_calls: None,
                    stream: None,
                    temperature: None,
                    top_p: None,
                    max_tokens: None,
                    max_completion_tokens: None,
                    response_format: None,
                    extra: serde_json::Map::new(),
                },
                &InboundAuth::default(),
            )
            .await
            .expect_err("invalid upstream JSON should fail");

        chat_mock.assert_async().await;
        let message = error.to_string();
        assert!(message.contains("failed to decode upstream response"));
        assert!(message.contains("x-openrouter-provider"));
        assert!(message.contains("deepinfra"));
    }

    #[tokio::test]
    async fn models_returns_upstream_json_and_forwards_inbound_auth() {
        let server = MockServer::start_async().await;
        let models_mock = server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/v1/models")
                    .header("authorization", "Bearer inbound");
                then.status(200).json_body(json!({
                    "object": "list",
                    "data": [{"id": "local-model", "object": "model"}]
                }));
            })
            .await;

        let response = client(format!("{}/v1", server.base_url()), None)
            .models(&InboundAuth::authorization("Bearer inbound"))
            .await
            .expect("models request should succeed");

        models_mock.assert_async().await;
        assert_eq!(response["data"][0]["id"], "local-model");
    }

    #[tokio::test]
    async fn inbound_authorization_overrides_configured_api_key_for_models() {
        let server = MockServer::start_async().await;
        let models_mock = server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/v1/models")
                    .header("authorization", "Bearer inbound");
                then.status(200)
                    .json_body(json!({"object": "list", "data": []}));
            })
            .await;

        client(
            format!("{}/v1", server.base_url()),
            Some("configured".to_string()),
        )
        .models(&InboundAuth::authorization("Bearer inbound"))
        .await
        .expect("models request should succeed");

        models_mock.assert_async().await;
    }

    #[tokio::test]
    async fn configured_api_key_is_used_when_client_auth_is_absent() {
        let server = MockServer::start_async().await;
        let models_mock = server
            .mock_async(|when, then| {
                when.method(GET)
                    .path("/v1/models")
                    .header("authorization", "Bearer configured");
                then.status(200)
                    .json_body(json!({"object": "list", "data": []}));
            })
            .await;

        client(
            format!("{}/v1", server.base_url()),
            Some("configured".to_string()),
        )
        .models(&InboundAuth::default())
        .await
        .expect("models request should succeed");

        models_mock.assert_async().await;
    }

    #[tokio::test]
    async fn chat_completions_forwards_inbound_authorization() {
        let server = MockServer::start_async().await;
        let chat_mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/v1/chat/completions")
                    .header("authorization", "Bearer inbound");
                then.status(200).json_body(json!({
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test-model",
                    "choices": []
                }));
            })
            .await;

        client(format!("{}/v1", server.base_url()), None)
            .chat_completions(
                &ChatCompletionRequest {
                    model: "test-model".to_string(),
                    messages: Vec::new(),
                    tools: None,
                    tool_choice: None,
                    parallel_tool_calls: None,
                    stream: None,
                    temperature: None,
                    top_p: None,
                    max_tokens: None,
                    max_completion_tokens: None,
                    response_format: None,
                    extra: serde_json::Map::new(),
                },
                &InboundAuth::authorization("Bearer inbound"),
            )
            .await
            .expect("chat request should succeed");

        chat_mock.assert_async().await;
    }

    #[tokio::test]
    async fn x_api_key_is_accepted_as_client_api_key_for_openai_compatible_upstreams() {
        let server = MockServer::start_async().await;
        let chat_mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/v1/chat/completions")
                    .header("authorization", "Bearer inbound-key")
                    .header("x-api-key", "inbound-key");
                then.status(200).json_body(json!({
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test-model",
                    "choices": []
                }));
            })
            .await;

        client(format!("{}/v1", server.base_url()), None)
            .chat_completions(
                &ChatCompletionRequest {
                    model: "test-model".to_string(),
                    messages: Vec::new(),
                    tools: None,
                    tool_choice: None,
                    parallel_tool_calls: None,
                    stream: None,
                    temperature: None,
                    top_p: None,
                    max_tokens: None,
                    max_completion_tokens: None,
                    response_format: None,
                    extra: serde_json::Map::new(),
                },
                &InboundAuth::x_api_key("inbound-key"),
            )
            .await
            .expect("chat request should succeed");

        chat_mock.assert_async().await;
    }

    #[tokio::test]
    async fn inbound_x_api_key_overrides_configured_api_key() {
        let server = MockServer::start_async().await;
        let chat_mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/v1/chat/completions")
                    .header("authorization", "Bearer inbound-key")
                    .header("x-api-key", "inbound-key");
                then.status(200).json_body(json!({
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test-model",
                    "choices": []
                }));
            })
            .await;

        client(
            format!("{}/v1", server.base_url()),
            Some("configured".to_string()),
        )
        .chat_completions(
            &ChatCompletionRequest {
                model: "test-model".to_string(),
                messages: Vec::new(),
                tools: None,
                tool_choice: None,
                parallel_tool_calls: None,
                stream: None,
                temperature: None,
                top_p: None,
                max_tokens: None,
                max_completion_tokens: None,
                response_format: None,
                extra: serde_json::Map::new(),
            },
            &InboundAuth::x_api_key("inbound-key"),
        )
        .await
        .expect("chat request should succeed");

        chat_mock.assert_async().await;
    }

    #[tokio::test]
    async fn api_key_is_accepted_as_client_api_key_for_openai_compatible_upstreams() {
        let server = MockServer::start_async().await;
        let chat_mock = server
            .mock_async(|when, then| {
                when.method(POST)
                    .path("/v1/chat/completions")
                    .header("authorization", "Bearer inbound-key")
                    .header("api-key", "inbound-key");
                then.status(200).json_body(json!({
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "test-model",
                    "choices": []
                }));
            })
            .await;

        client(format!("{}/v1", server.base_url()), None)
            .chat_completions(
                &ChatCompletionRequest {
                    model: "test-model".to_string(),
                    messages: Vec::new(),
                    tools: None,
                    tool_choice: None,
                    parallel_tool_calls: None,
                    stream: None,
                    temperature: None,
                    top_p: None,
                    max_tokens: None,
                    max_completion_tokens: None,
                    response_format: None,
                    extra: serde_json::Map::new(),
                },
                &InboundAuth::api_key("inbound-key"),
            )
            .await
            .expect("chat request should succeed");

        chat_mock.assert_async().await;
    }
}
