use std::{net::SocketAddr, path::PathBuf};

use httpmock::{Method::POST, MockServer};
use model_correction_proxy::{config::ProxyConfig, model_profile::ModelProfile, Gateway};
use serde_json::{json, Value};
use tempfile::NamedTempFile;
use tokio::task::JoinHandle;

async fn spawn_proxy(upstream_base_url: String, trace_path: PathBuf) -> (String, JoinHandle<()>) {
    let mut config = ProxyConfig::default();
    config.upstream.base_url = upstream_base_url;
    config.upstream.api_key = Some("test-key".to_string());
    config.trace.path = trace_path;
    config.trace.correction_path = config.trace.path.with_extension("corrections.jsonl");

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

async fn spawn_proxy_with_config(mut config: ProxyConfig) -> (String, JoinHandle<()>) {
    if config.upstream.api_key.is_none() {
        config.upstream.api_key = Some("test-key".to_string());
    }
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
        "stream": stream
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
    run_request(pi_like_request(model, prompt, stream), upstream_response).await
}

async fn run_request(request: Value, upstream_response: Value) -> (String, String) {
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
        .json(&request)
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
                "content": "[[ ## content ## ]]\nI will list the repository files.\n\n[[ ## tool_calls ## ]]\n[{\"name\":\"bash\",\"arguments\":{\"command\":\"ls -la\"}}]\n[[ ## completed ## ]]"
            }),
            "stop",
        ),
    )
    .await;
    assert!(body.contains("\"tool_calls\""));
    assert!(body.contains("I will list the repository files."));
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
async fn proxy_e2e_translates_multiturn_tool_history_into_dsrs_context() {
    let request = json!({
        "model": "google/gemma-4-26b-a4b-it",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert coding assistant operating inside pi. Use tools for file work."
            },
            {"role": "user", "content": [{"type":"text","text":"read the readme"}]},
            {
                "role": "assistant",
                "content": null,
                "tool_calls": [
                    {
                        "id": "call_read_1",
                        "type": "function",
                        "function": {
                            "name": "read",
                            "arguments": "{\"path\":\"README.md\"}"
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call_read_1",
                "content": "# Project\nA proxy for correcting model responses."
            },
            {"role": "user", "content": [{"type":"text","text":"summarize that briefly"}]}
        ],
        "tools": pi_tools(),
        "parallel_tool_calls": true,
        "stream": false
    });

    let (body, trace) = run_request(
        request,
        chat_response(
            "google/gemma-4-26b-a4b-it",
            json!({
                "role": "assistant",
                "content": "[[ ## content ## ]]\nIt is a proxy for correcting model responses.\n[[ ## tool_calls ## ]]\n[]\n[[ ## completed ## ]]"
            }),
            "stop",
        ),
    )
    .await;

    assert!(body.contains("proxy for correcting model responses"));
    assert!(
        trace.contains("[[ ## tool_calls ## ]]"),
        "trace did not include DSRs assistant tool history: {trace}"
    );
    assert!(trace.contains("\\\"name\\\": \\\"read\\\""));
    assert!(trace.contains("[[ ## tool_result ## ]]"));
    assert!(trace.contains("A proxy for correcting model responses."));
    assert!(trace.contains("\"summary\""));
}

#[tokio::test]
async fn proxy_e2e_routes_adjacent_json_violation_through_correction_agent() {
    let upstream = MockServer::start_async().await;
    let _correction_mock = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/v1/chat/completions")
                .body_contains("possible")
                .body_contains("confidence")
                .body_contains("tool_calls");
            then.status(200).json_body(chat_response(
                "qwen/qwen3.5-9b",
                json!({
                    "role": "assistant",
                    "content": "[[ ## possible ## ]]\ntrue\n\n[[ ## confidence ## ]]\n0.95\n\n[[ ## explanation ## ]]\nrecovered adjacent JSON tool calls\n\n[[ ## content ## ]]\n\n[[ ## tool_calls ## ]]\n[{\"name\":\"read\",\"arguments\":{\"path\":\"packages/ai/README.md\"}},{\"name\":\"read\",\"arguments\":{\"path\":\"packages/agent/README.md\"}}]\n\n[[ ## completed ## ]]"
                }),
                "stop",
            ));
        })
        .await;
    let _upstream_mock = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/v1/chat/completions")
                .body_contains("parallel_tool_calls");
            then.status(200).json_body(chat_response(
                "qwen/qwen3.5-9b",
                json!({
                    "role": "assistant",
                    "content": "[]\n\n[\n  {\"name\":\"read\",\"arguments\":{\"path\":\"packages/ai/README.md\"}},\n  {\"name\":\"read\",\"arguments\":{\"path\":\"packages/agent/README.md\"}}\n]"
                }),
                "stop",
            ));
        })
        .await;

    let trace_file = NamedTempFile::new().expect("trace file");
    let trace_path = trace_file.path().to_path_buf();
    let (proxy_url, handle) =
        spawn_proxy(format!("{}/v1", upstream.base_url()), trace_path.clone()).await;

    let response = reqwest::Client::new()
        .post(format!("{proxy_url}/v1/chat/completions"))
        .json(&pi_like_request(
            "qwen/qwen3.5-9b",
            "do each of them have a readme that says a little more than that though?",
            false,
        ))
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
    let parsed: Value = serde_json::from_str(&body).expect("response JSON");
    let calls = parsed
        .pointer("/choices/0/message/tool_calls")
        .and_then(Value::as_array)
        .expect("tool calls");
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0]["function"]["name"], "read");
    assert_eq!(
        calls[0]["function"]["arguments"],
        "{\"path\":\"packages/ai/README.md\"}"
    );

    let trace = tokio::fs::read_to_string(trace_path)
        .await
        .expect("trace file read");
    let correction_trace =
        tokio::fs::read_to_string(trace_file.path().with_extension("corrections.jsonl"))
            .await
            .expect("correction trace file read");
    assert!(trace.contains("correction_agent_tool_recovery"));
    assert!(trace.contains("\"correction_attempts\""));
    assert!(trace.contains("\"policy_decisions\""));
    assert!(correction_trace.contains("\"parent_trace_id\""));
    assert!(correction_trace.contains("\"raw_output\""));
    assert!(correction_trace.contains("recovered adjacent JSON tool calls"));
    assert!(trace.contains("\"summary\""));
    handle.abort();
}

#[tokio::test]
async fn proxy_e2e_configured_profile_controls_unknown_model_and_trace_metadata() {
    let upstream = MockServer::start_async().await;
    let _upstream_mock = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/v1/chat/completions")
                .body_contains("REQUEST CONFIG INSTRUCTION")
                .body_contains("[[ ## available_tools ## ]]");
            then.status(200).json_body(chat_response(
                "new-provider/custom-model",
                json!({
                    "role": "assistant",
                    "content": "[[ ## content ## ]]\nconfigured profile response\n[[ ## tool_calls ## ]]\n[]\n[[ ## completed ## ]]"
                }),
                "stop",
            ));
        })
        .await;

    let trace_file = NamedTempFile::new().expect("trace file");
    let trace_path = trace_file.path().to_path_buf();
    let mut profile = ModelProfile::default_balanced();
    profile.name = "custom-config-dsrs".to_string();
    profile.model_patterns = vec!["new-provider/custom".to_string()];
    profile.revision = 42;
    profile.request_adapter_artifact = Some("profiles/custom/request-gepa.json".to_string());
    profile.tool_instruction = "REQUEST CONFIG INSTRUCTION".to_string();
    profile.mark_config_source();

    let mut config = ProxyConfig::default();
    config.upstream.base_url = format!("{}/v1", upstream.base_url());
    config.trace.path = trace_path.clone();
    config.trace.correction_path = trace_path.with_extension("corrections.jsonl");
    config.model_profiles = vec![profile];
    let (proxy_url, handle) = spawn_proxy_with_config(config).await;

    let response = reqwest::Client::new()
        .post(format!("{proxy_url}/v1/chat/completions"))
        .json(&pi_like_request(
            "new-provider/custom-model",
            "say hello without tools",
            false,
        ))
        .send()
        .await
        .expect("proxy response");
    assert!(response.status().is_success());
    let body = response.text().await.expect("response text");
    assert!(body.contains("configured profile response"));
    assert!(!body.contains("[[ ##"));

    let trace = tokio::fs::read_to_string(trace_file.path())
        .await
        .expect("trace file read");
    assert!(trace.contains("\"profile_id\":\"custom-config-dsrs\""));
    assert!(trace.contains("\"profile_revision\":42"));
    assert!(trace.contains("\"profile_source\":\"config\""));
    assert!(trace.contains("profiles/custom/request-gepa.json"));
    assert!(trace.contains("\"summary\""));
    handle.abort();
}

#[tokio::test]
async fn proxy_e2e_correction_agent_uses_configured_instruction_artifact() {
    let upstream = MockServer::start_async().await;
    let _correction_mock = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/v1/chat/completions")
                .body_contains("CORRECTION CONFIG INSTRUCTION")
                .body_contains("possible")
                .body_contains("tool_calls");
            then.status(200).json_body(chat_response(
                "new-provider/custom-model",
                json!({
                    "role": "assistant",
                    "content": "[[ ## possible ## ]]\ntrue\n\n[[ ## confidence ## ]]\n0.97\n\n[[ ## explanation ## ]]\nrecovered through configured correction artifact\n\n[[ ## content ## ]]\n\n[[ ## tool_calls ## ]]\n[{\"name\":\"read\",\"arguments\":{\"path\":\"packages/ai/README.md\"}}]\n\n[[ ## completed ## ]]"
                }),
                "stop",
            ));
        })
        .await;
    let _upstream_mock = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/v1/chat/completions")
                .body_contains("REQUEST CONFIG INSTRUCTION");
            then.status(200).json_body(chat_response(
                "new-provider/custom-model",
                json!({
                    "role": "assistant",
                    "content": "[]\n\n[{\"name\":\"read\",\"arguments\":{\"path\":\"packages/ai/README.md\"}}]"
                }),
                "stop",
            ));
        })
        .await;

    let trace_file = NamedTempFile::new().expect("trace file");
    let trace_path = trace_file.path().to_path_buf();
    let mut profile = ModelProfile::default_balanced();
    profile.name = "custom-correction-dsrs".to_string();
    profile.model_patterns = vec!["new-provider/custom".to_string()];
    profile.revision = 5;
    profile.request_adapter_artifact = Some("profiles/custom/request-gepa.json".to_string());
    profile.correction_agent_artifact =
        Some("profiles/custom/correction-agent-gepa.json".to_string());
    profile.tool_instruction = "REQUEST CONFIG INSTRUCTION".to_string();
    profile.correction_instruction = Some("CORRECTION CONFIG INSTRUCTION".to_string());
    profile.mark_config_source();

    let mut config = ProxyConfig::default();
    config.upstream.base_url = format!("{}/v1", upstream.base_url());
    config.trace.path = trace_path.clone();
    config.trace.correction_path = trace_path.with_extension("corrections.jsonl");
    config.model_profiles = vec![profile];
    let (proxy_url, handle) = spawn_proxy_with_config(config).await;

    let response = reqwest::Client::new()
        .post(format!("{proxy_url}/v1/chat/completions"))
        .json(&pi_like_request(
            "new-provider/custom-model",
            "read the package readme",
            false,
        ))
        .send()
        .await
        .expect("proxy response");
    assert!(response.status().is_success());
    let body = response.text().await.expect("response text");
    let parsed: Value = serde_json::from_str(&body).expect("response JSON");
    let calls = parsed
        .pointer("/choices/0/message/tool_calls")
        .and_then(Value::as_array)
        .expect("tool calls");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["function"]["name"], "read");

    let trace = tokio::fs::read_to_string(trace_file.path())
        .await
        .expect("trace file read");
    let correction_trace =
        tokio::fs::read_to_string(trace_file.path().with_extension("corrections.jsonl"))
            .await
            .expect("correction trace file read");
    assert!(trace.contains("correction_agent_tool_recovery"));
    assert!(trace.contains("profiles/custom/correction-agent-gepa.json"));
    assert!(correction_trace.contains("\"profile_revision\":5"));
    assert!(correction_trace.contains("\"profile_source\":\"config\""));
    assert!(correction_trace.contains("profiles/custom/correction-agent-gepa.json"));
    assert!(correction_trace.contains("recovered through configured correction artifact"));
    handle.abort();
}

#[tokio::test]
async fn proxy_e2e_routes_dsrs_contract_violation_through_correction_agent() {
    let upstream = MockServer::start_async().await;
    let _correction_mock = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/v1/chat/completions")
                .body_contains("possible")
                .body_contains("tool_calls")
                .body_contains("dsrs_content_outside_tagged_fields");
            then.status(200).json_body(chat_response(
                "qwen/qwen3.5-9b",
                json!({
                    "role": "assistant",
                    "content": "[[ ## possible ## ]]\ntrue\n\n[[ ## confidence ## ]]\n0.96\n\n[[ ## explanation ## ]]\nrecovered malformed DSRs contract output\n\n[[ ## content ## ]]\n\n[[ ## tool_calls ## ]]\n[{\"name\":\"read\",\"arguments\":{\"path\":\"packages/ai/README.md\"}}]\n\n[[ ## completed ## ]]"
                }),
                "stop",
            ));
        })
        .await;
    let _upstream_mock = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/v1/chat/completions")
                .body_contains("parallel_tool_calls");
            then.status(200).json_body(chat_response(
                "qwen/qwen3.5-9b",
                json!({
                    "role": "assistant",
                    "content": "I should inspect the package README files.\n\n[{\"name\":\"read\",\"arguments\":{\"path\":\"packages/ai/README.md\"}}]\n\n[[ ## system_context ## ]]\nDo not repeat this.\n\n[[ ## content ## ]]\n---\n[[ ## tool_calls ## ]]\n[]\n[[ ## completed ## ]]"
                }),
                "stop",
            ));
        })
        .await;

    let trace_file = NamedTempFile::new().expect("trace file");
    let trace_path = trace_file.path().to_path_buf();
    let (proxy_url, handle) =
        spawn_proxy(format!("{}/v1", upstream.base_url()), trace_path.clone()).await;

    let response = reqwest::Client::new()
        .post(format!("{proxy_url}/v1/chat/completions"))
        .json(&pi_like_request(
            "qwen/qwen3.5-9b",
            "very cool. can you dive into each package and let me know more information about each?",
            false,
        ))
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
    let parsed: Value = serde_json::from_str(&body).expect("response JSON");
    let calls = parsed
        .pointer("/choices/0/message/tool_calls")
        .and_then(Value::as_array)
        .expect("tool calls");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["function"]["name"], "read");
    assert_eq!(
        calls[0]["function"]["arguments"],
        "{\"path\":\"packages/ai/README.md\"}"
    );

    let trace = tokio::fs::read_to_string(trace_path)
        .await
        .expect("trace file read");
    assert!(trace.contains("DsrsContractViolation"));
    assert!(trace.contains("DsrsContentOutsideTaggedFields"));
    assert!(trace.contains("PromptEcho"));
    assert!(trace.contains("correction_agent_tool_recovery"));
    assert!(trace.contains("\"correction_attempts\""));
    assert!(trace.contains("\"policy_decisions\""));
    handle.abort();
}

#[tokio::test]
async fn proxy_e2e_routes_empty_dsrs_output_through_correction_agent() {
    let upstream = MockServer::start_async().await;
    let _correction_mock = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/v1/chat/completions")
                .body_contains("possible")
                .body_contains("empty_dsrs_output")
                .body_contains("can you tell me more about this project?");
            then.status(200).json_body(chat_response(
                "google/gemma-4-26b-a4b-it",
                json!({
                    "role": "assistant",
                    "content": "[[ ## possible ## ]]\ntrue\n\n[[ ## confidence ## ]]\n0.93\n\n[[ ## explanation ## ]]\nempty DSRs output should inspect README for project context\n\n[[ ## content ## ]]\n\n[[ ## tool_calls ## ]]\n[{\"name\":\"read\",\"arguments\":{\"path\":\"README.md\"}}]\n\n[[ ## completed ## ]]"
                }),
                "stop",
            ));
        })
        .await;
    let _upstream_mock = upstream
        .mock_async(|when, then| {
            when.method(POST)
                .path("/v1/chat/completions")
                .body_contains("[[ ## profile_guidance ## ]]")
                .body_contains("MANDATORY INSPECTION FIRST");
            then.status(200).json_body(chat_response(
                "google/gemma-4-26b-a4b-it",
                json!({
                    "role": "assistant",
                    "content": "[[ ## content ## ]]\n[[ ## tool_calls ## ]]\n[]\n[[ ## completed ## ]]"
                }),
                "stop",
            ));
        })
        .await;

    let trace_file = NamedTempFile::new().expect("trace file");
    let trace_path = trace_file.path().to_path_buf();
    let (proxy_url, handle) =
        spawn_proxy(format!("{}/v1", upstream.base_url()), trace_path.clone()).await;

    let response = reqwest::Client::new()
        .post(format!("{proxy_url}/v1/chat/completions"))
        .json(&pi_like_request(
            "google/gemma-4-26b-a4b-it",
            "can you tell me more about this project?",
            false,
        ))
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
    let parsed: Value = serde_json::from_str(&body).expect("response JSON");
    let calls = parsed
        .pointer("/choices/0/message/tool_calls")
        .and_then(Value::as_array)
        .expect("tool calls");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0]["function"]["name"], "read");
    assert_eq!(
        calls[0]["function"]["arguments"],
        "{\"path\":\"README.md\"}"
    );

    let trace = tokio::fs::read_to_string(trace_path)
        .await
        .expect("trace file read");
    let correction_trace =
        tokio::fs::read_to_string(trace_file.path().with_extension("corrections.jsonl"))
            .await
            .expect("correction trace file read");
    assert!(trace.contains("EmptyDsrsOutput"));
    assert!(trace.contains("correction_agent_tool_recovery"));
    assert!(trace.contains("attempt_correction_agent"));
    assert!(correction_trace.contains("empty_dsrs_output"));
    assert!(correction_trace.contains("empty DSRs output should inspect README"));
    handle.abort();
}

#[tokio::test]
async fn proxy_e2e_parses_tagged_dsrs_with_inner_field_labels() {
    let (body, _trace) = run_case(
        "qwen/qwen3.5-9b",
        "oh cool. can you dive into all of the subpackages and tell me about them too?",
        false,
        chat_response(
            "qwen/qwen3.5-9b",
            json!({
                "role": "assistant",
                "content": "[[ ## content ## ]]\ncontent:\n[[ ## tool_calls ## ]]\ntool_calls: [\n  {\"name\":\"read\",\"arguments\":{\"path\":\"packages/agent/README.md\"}},\n  {\"name\":\"read\",\"arguments\":{\"path\":\"packages/ai/README.md\"}}\n]\n[[ ## completed ## ]]"
            }),
            "stop",
        ),
    )
    .await;

    let parsed: Value = serde_json::from_str(&body).expect("response JSON");
    let calls = parsed
        .pointer("/choices/0/message/tool_calls")
        .and_then(Value::as_array)
        .expect("tool calls");
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0]["function"]["name"], "read");
    assert_eq!(
        calls[0]["function"]["arguments"],
        "{\"path\":\"packages/agent/README.md\"}"
    );
    assert_eq!(
        parsed.pointer("/choices/0/finish_reason"),
        Some(&Value::String("tool_calls".to_string()))
    );
}

#[tokio::test]
async fn proxy_e2e_parses_tagged_dsrs_with_adjacent_tool_call_arrays() {
    let (body, _trace) = run_case(
        "qwen/qwen3.5-9b",
        "oh cool. there's a lot of sub packages here. can you dive into them to get more information about them all?",
        false,
        chat_response(
            "qwen/qwen3.5-9b",
            json!({
                "role": "assistant",
                "content": "[[ ## content ## ]]\n\n[[ ## tool_calls ## ]]\n[{\"name\":\"bash\",\"arguments\":{\"command\":\"ls -la packages/\"}}]\n[{\"name\":\"read\",\"arguments\":{\"path\":\"packages/coding-agent/README.md\",\"limit\":100}}]\n[{\"name\":\"read\",\"arguments\":{\"path\":\"packages/ai/README.md\",\"limit\":100}}]\n[[ ## completed ## ]]"
            }),
            "stop",
        ),
    )
    .await;

    let parsed: Value = serde_json::from_str(&body).expect("response JSON");
    let calls = parsed
        .pointer("/choices/0/message/tool_calls")
        .and_then(Value::as_array)
        .expect("tool calls");
    assert_eq!(calls.len(), 3);
    assert_eq!(calls[0]["function"]["name"], "bash");
    assert_eq!(calls[1]["function"]["name"], "read");
    let args: Value =
        serde_json::from_str(calls[2]["function"]["arguments"].as_str().unwrap()).unwrap();
    assert_eq!(args["path"], "packages/ai/README.md");
    assert_eq!(args["limit"], 100);
}

#[tokio::test]
#[ignore = "requires a valid OPENROUTER_API_KEY and makes paid live requests"]
async fn live_openrouter_pi_prompt_matrix() {
    let api_key = live_openrouter_key().expect("OPENROUTER_API_KEY is required");
    let models = std::env::var("OPENROUTER_LIVE_MODELS").unwrap_or_else(|_| {
        [
            "qwen/qwen3.5-9b",
            "qwen/qwen3.5-flash-02-23",
            "qwen/qwen3-8b",
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
    config.trace.correction_path = config.trace.path.with_extension("corrections.jsonl");

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
            let parsed: Value = serde_json::from_str(&body)
                .unwrap_or_else(|error| panic!("{model} {prompt}: invalid JSON {error}: {body}"));
            let content = parsed
                .pointer("/choices/0/message/content")
                .and_then(Value::as_str)
                .unwrap_or_default();
            assert!(
                !content.contains("[[ ##"),
                "{model} leaked DSRs markers: {body}"
            );
            assert!(
                !content.contains("assistant text:"),
                "{model} leaked presentation label: {body}"
            );
            let trimmed_content = content.trim_start().to_ascii_lowercase();
            assert!(
                !trimmed_content.starts_with("content ")
                    && !trimmed_content.starts_with("content:"),
                "{model} leaked DSRs content label: {body}"
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

fn live_openrouter_key() -> Option<String> {
    read_env_key("OPENROUTER_API_KEY")
        .or_else(|| read_env_key("MCP_UPSTREAM_API_KEY"))
        .or_else(|| env_key("OPENROUTER_API_KEY"))
        .or_else(|| env_key("MCP_UPSTREAM_API_KEY"))
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
