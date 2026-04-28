use anyhow::{Context, Result};
use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    config::UpstreamConfig,
    openai::{ChatCompletionRequest, ChatCompletionResponse},
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
        let mut builder = self
            .http
            .post(self.endpoint_url("chat/completions"))
            .header(CONTENT_TYPE, "application/json")
            .header("X-Title", "model-correction-proxy")
            .json(request);

        builder = self.apply_auth(builder, inbound_auth);

        let response = builder.send().await.map_err(UpstreamError::Transport)?;
        let status = response.status();
        let body = response.text().await.map_err(UpstreamError::Transport)?;
        if !status.is_success() {
            return Err(UpstreamError::Status {
                status: status.as_u16(),
                body,
            });
        }

        serde_json::from_str(&body).map_err(|source| UpstreamError::Decode { source, body })
    }

    pub async fn models(
        &self,
        inbound_auth: &InboundAuth,
    ) -> Result<serde_json::Value, UpstreamError> {
        let builder = self
            .http
            .get(self.endpoint_url("models"))
            .header("X-Title", "model-correction-proxy");
        let builder = self.apply_auth(builder, inbound_auth);

        let response = builder.send().await.map_err(UpstreamError::Transport)?;
        let status = response.status();
        let body = response.text().await.map_err(UpstreamError::Transport)?;
        if !status.is_success() {
            return Err(UpstreamError::Status {
                status: status.as_u16(),
                body,
            });
        }

        serde_json::from_str(&body).map_err(|source| UpstreamError::Decode { source, body })
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

#[derive(Debug, Error)]
pub enum UpstreamError {
    #[error("upstream transport error: {0}")]
    Transport(#[from] reqwest::Error),
    #[error("upstream returned HTTP {status}: {body}")]
    Status { status: u16, body: String },
    #[error("failed to decode upstream response: {source}; body: {body}")]
    Decode {
        source: serde_json::Error,
        body: String,
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
