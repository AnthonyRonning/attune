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
        let url = format!(
            "{}/chat/completions",
            self.config
                .base_url
                .trim_end_matches("/v1")
                .trim_end_matches('/')
        );
        let url = if self.config.base_url.ends_with("/chat/completions") {
            self.config.base_url.clone()
        } else if self.config.base_url.ends_with("/v1") {
            format!(
                "{}/chat/completions",
                self.config.base_url.trim_end_matches('/')
            )
        } else {
            url
        };

        let mut builder = self
            .http
            .post(url)
            .header(CONTENT_TYPE, "application/json")
            .header("X-Title", "model-correction-proxy")
            .json(request);

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
