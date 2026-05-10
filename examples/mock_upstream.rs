use std::net::SocketAddr;

use attune::openai::{ChatChoice, ChatCompletionRequest, ChatCompletionResponse, ChatMessage};
use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Map, Value};

#[derive(Clone)]
struct MockState {
    response: Option<Value>,
    content: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let bind: SocketAddr = std::env::var("MOCK_UPSTREAM_BIND")
        .unwrap_or_else(|_| "127.0.0.1:18081".to_string())
        .parse()?;
    let response = std::env::var("MOCK_UPSTREAM_RESPONSE")
        .ok()
        .map(|value| serde_json::from_str(&value))
        .transpose()?;
    let content = std::env::var("MOCK_UPSTREAM_CONTENT")
        .unwrap_or_else(|_| "mock upstream response".to_string());
    let state = MockState { response, content };

    let router = Router::new()
        .route("/v1/models", get(models))
        .route("/v1/chat/completions", post(chat))
        .with_state(state);
    let listener = tokio::net::TcpListener::bind(bind).await?;
    axum::serve(listener, router).await?;
    Ok(())
}

async fn models() -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [
            {
                "id": "mock-model",
                "object": "model",
                "created": 0,
                "owned_by": "attune"
            }
        ]
    }))
}

async fn chat(
    State(state): State<MockState>,
    Json(request): Json<ChatCompletionRequest>,
) -> Json<Value> {
    if let Some(response) = state.response {
        return Json(response);
    }

    let mut response = ChatCompletionResponse::empty_for_model(request.model);
    response.choices.push(ChatChoice {
        index: 0,
        message: ChatMessage {
            role: "assistant".to_string(),
            content: Some(json!(state.content)),
            name: None,
            tool_call_id: None,
            tool_calls: None,
            extra: Map::new(),
        },
        finish_reason: Some("stop".to_string()),
        logprobs: None,
        extra: Map::new(),
    });
    Json(serde_json::to_value(response).expect("response serializes"))
}
