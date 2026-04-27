use anyhow::{Context, Result};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    config::UpstreamConfig,
    openai::{ChatCompletionRequest, ChatCompletionResponse},
};

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
        inbound_authorization: Option<&str>,
    ) -> Result<ChatCompletionResponse, UpstreamError> {
        let mut builder = self
            .http
            .post(self.endpoint_url("chat/completions"))
            .header(CONTENT_TYPE, "application/json")
            .header("X-Title", "model-correction-proxy")
            .json(request);

        builder = self.apply_auth(builder, inbound_authorization);

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
        inbound_authorization: Option<&str>,
    ) -> Result<serde_json::Value, UpstreamError> {
        let builder = self
            .http
            .get(self.endpoint_url("models"))
            .header("X-Title", "model-correction-proxy");
        let builder = self.apply_auth(builder, inbound_authorization);

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
        inbound_authorization: Option<&str>,
    ) -> reqwest::RequestBuilder {
        if let Some(api_key) = self
            .config
            .api_key
            .as_ref()
            .filter(|key| !key.trim().is_empty())
        {
            builder = builder.bearer_auth(api_key);
        } else if let Some(auth) = inbound_authorization {
            builder = builder.header(AUTHORIZATION, auth);
        }
        builder
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
            .models(Some("Bearer inbound"))
            .await
            .expect("models request should succeed");

        models_mock.assert_async().await;
        assert_eq!(response["data"][0]["id"], "local-model");
    }

    #[tokio::test]
    async fn configured_api_key_overrides_inbound_auth_for_models() {
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
        .models(Some("Bearer inbound"))
        .await
        .expect("models request should succeed");

        models_mock.assert_async().await;
    }
}
