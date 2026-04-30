use std::{net::SocketAddr, path::PathBuf};

use httpmock::{Method::POST, MockServer};
use model_correction_proxy::{config::ProxyConfig, Gateway};
use serde_json::{json, Value};
use tempfile::NamedTempFile;
use tokio::task::JoinHandle;

async fn spawn_proxy(upstream_base_url: String, trace_path: PathBuf) -> (String, JoinHandle<()>) {
    let mut config = ProxyConfig::default();
    config.upstream.base_url = upstream_base_url;
    config.trace.path = trace_path;

    let gateway = Gateway::new(config).expect("gateway starts");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("binds proxy");
    let addr = listener.local_addr().expect("local addr");
    let handle = tokio::spawn(async move {
        axum::serve(listener, gateway.router())
            .await
            .expect("proxy serve failed");
    });

    (format!("http://{addr}"), handle)
}

fn pi_like_request(model: &str, prompt: &str, stream: bool) -> Value {
    json!({
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are an expert coding assistant operating inside pi. Use tools for file and shell work. Keep answers concise."
            },
            {"role": "user", "content": prompt}
        ],
        "tools": pi_tools(),
        "parallel_tool_calls": true,
        "stream": stream,
        "max_tokens": 500
    })
}

fn pi_tools() -> Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "read",
                "description": "Read file contents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "offset": {"type": "number"},
                        "limit": {"type": "number"}
                    },
                    "required": ["path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a bash command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout": {"type": "number"}
                    },
                    "required": ["command"]
                }
            }
        }
    ])
}

fn chat_response(model: &str, message: Value, finish_reason: &str) -> Value {
    json!({
        "id": "chatcmpl-e2e",
        "object": "chat.completion",
        "created": 0,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason
            }
        ]
    })
}

async fn run_case(
    model: &str,
    prompt: &str,
    stream: bool,
    upstream_response: Value,
) -> (String, String) {
    let upstream = MockServer::start_async().await;
    let _mock = upstream
        .mock_async(|when, then| {
            when.method(POST).path("/v1/chat/completions");
            then.status(200).json_body(upstream_response);
        })
        .await;

    let trace_file = NamedTempFile::new().expect("trace file");
    let trace_path = trace_file.path().to_path_buf();
    let (proxy_url, handle) =
        spawn_proxy(format!("{}/v1", upstream.base_url()), trace_path.clone()).await;

    let response = reqwest::Client::new()
        .post(format!("{proxy_url}/v1/chat/completions"))
        .json(&pi_like_request(model, prompt, stream))
        .send()
        .await
        .expect("proxy response");
    assert!(
        response.status().is_success(),
        "proxy status {}: {}",
        response.status(),
        response.text().await.unwrap_or_default()
    );
    let body = response.text().await.expect("response text");
    let trace = tokio::fs::read_to_string(trace_path)
        .await
        .expect("trace file read");
    handle.abort();
    (body, trace)
}

#[tokio::test]
async fn proxy_e2e_matrix_for_pi_like_prompts() {
    let (body, trace) = run_case(
        "qwen/qwen3.5-9b",
        "hey",
        true,
        chat_response(
            "qwen/qwen3.5-9b",
            json!({
                "role": "assistant",
                "content": "[[ ## content ## ]]\nHello. How can I help?\n[[ ## tool_calls ## ]]\n[]\n[[ ## completed ## ]]",
                "reasoning": "No tool is needed for a greeting.",
                "reasoning_details": [{"type": "reasoning.text", "text": "No tool is needed."}]
            }),
            "stop",
        ),
    )
    .await;
    assert!(!body.contains("[[ ##"));
    assert!(!body.contains("assistant text"));
    assert!(body.contains("No tool is needed for a greeting."));
    assert!(body.find("reasoning").unwrap() < body.find("Hello. How can I help?").unwrap());
    assert!(!trace.contains("assistant text"));

    let (body, _trace) = run_case(
        "meta-llama/llama-3.1-8b-instruct",
        "List the repository files.",
        false,
        chat_response(
            "meta-llama/llama-3.1-8b-instruct",
            json!({
                "role": "assistant",
                "content": "[[ ## content ## ]]\n\n[[ ## tool_calls ## ]]\n[{\"name\":\"bash\",\"arguments\":{\"command\":\"ls -la\"}}]\n[[ ## completed ## ]]"
            }),
            "stop",
        ),
    )
    .await;
    assert!(body.contains("\"tool_calls\""));
    assert!(body.contains("\"name\":\"bash\""));
    assert!(body.contains("ls -la"));

    let (body, _trace) = run_case(
        "qwen/qwen3.5-9b",
        "List packages.",
        false,
        chat_response(
            "qwen/qwen3.5-9b",
            json!({
                "role": "assistant",
                "content": "<tool_call>\n<function=bash>\n<parameter=\"command\">ls -la packages/\n</parameter>\n</tool_call>"
            }),
            "stop",
        ),
    )
    .await;
    assert!(body.contains("malformed tool call"));
    assert!(!body.contains("<function=bash>"));

    let (body, _trace) = run_case(
        "google/gemma-3-12b-it",
        "Run pwd.",
        false,
        chat_response(
            "google/gemma-3-12b-it",
            json!({
                "role": "assistant",
                "content": null,
                "tool_calls": [
                    {
                        "id": "call_bad",
                        "type": "function",
                        "function": {"name": "bash", "arguments": "{}"}
                    }
                ]
            }),
            "tool_calls",
        ),
    )
    .await;
    assert!(body.contains("malformed tool call"));
    assert!(!body.contains("\"tool_calls\""));
}

#[tokio::test]
#[ignore = "requires a valid OPENROUTER_API_KEY and makes paid live requests"]
async fn live_openrouter_pi_prompt_matrix() {
    let api_key = std::env::var("OPENROUTER_API_KEY").expect("OPENROUTER_API_KEY is required");
    let models = std::env::var("OPENROUTER_LIVE_MODELS").unwrap_or_else(|_| {
        [
            "qwen/qwen3.5-9b",
            "meta-llama/llama-3.1-8b-instruct",
            "google/gemma-3-12b-it",
        ]
        .join(",")
    });
    let models: Vec<&str> = models
        .split(',')
        .map(str::trim)
        .filter(|model| !model.is_empty())
        .collect();

    let trace_file = NamedTempFile::new().expect("trace file");
    let mut config = ProxyConfig::default();
    config.upstream.base_url = "https://openrouter.ai/api/v1".to_string();
    config.upstream.api_key = Some(api_key);
    config.trace.path = trace_file.path().to_path_buf();

    let gateway = Gateway::new(config).expect("gateway starts");
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("binds proxy");
    let addr: SocketAddr = listener.local_addr().expect("local addr");
    let handle = tokio::spawn(async move {
        axum::serve(listener, gateway.router())
            .await
            .expect("proxy serve failed");
    });
    let client = reqwest::Client::new();
    let proxy_url = format!("http://{addr}");

    for model in models {
        for (prompt, expect_tool) in [
            ("hey", false),
            ("Use bash to list the repository files.", true),
            ("Use read to inspect Cargo.toml.", true),
        ] {
            let response = client
                .post(format!("{proxy_url}/v1/chat/completions"))
                .json(&pi_like_request(model, prompt, false))
                .send()
                .await
                .unwrap_or_else(|error| panic!("{model} {prompt}: request failed: {error}"));
            assert!(
                response.status().is_success(),
                "{model} {prompt}: status {} body {}",
                response.status(),
                response.text().await.unwrap_or_default()
            );
            let body = response.text().await.expect("response text");
            assert!(
                !body.contains("[[ ##"),
                "{model} leaked DSRs markers: {body}"
            );
            assert!(
                !body.contains("assistant text:"),
                "{model} leaked presentation label: {body}"
            );
            if expect_tool {
                assert!(
                    body.contains("\"tool_calls\"") || body.contains("malformed tool call"),
                    "{model} did not produce or safely suppress a tool call: {body}"
                );
            }
        }
    }

    handle.abort();
}
