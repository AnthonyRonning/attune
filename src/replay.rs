use std::path::PathBuf;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::{
    agents::NoopCorrectionAgent, config::ProxyConfig, repair::repair_response,
    response_interpreter::interpret_response, trace::read_trace_records,
};

#[derive(Debug, Clone)]
pub struct ReplayConfig {
    pub trace_path: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplayReport {
    pub traces_seen: usize,
    pub traces_replayed: usize,
    pub mismatches: usize,
    pub failures: Vec<String>,
}

pub async fn replay_traces(config: ReplayConfig) -> Result<()> {
    let report = replay_trace_file(config).await?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

pub async fn replay_trace_file(config: ReplayConfig) -> Result<ReplayReport> {
    let records = read_trace_records(&config.trace_path).await?;
    let proxy_config = ProxyConfig::default();
    let correction_agent = NoopCorrectionAgent;
    let mut report = ReplayReport {
        traces_seen: records.len(),
        traces_replayed: 0,
        mismatches: 0,
        failures: Vec::new(),
    };

    for record in records {
        let (Some(normalized), Some(profile), Some(upstream), Some(expected)) = (
            record.normalized,
            record.profile,
            record.upstream_response,
            record.final_response,
        ) else {
            continue;
        };

        let interpreted = interpret_response(&upstream, &normalized.tools);
        match repair_response(
            &proxy_config,
            &normalized,
            &profile,
            &upstream,
            &interpreted,
            &correction_agent,
        )
        .await
        {
            Ok(outcome) => {
                report.traces_replayed += 1;
                if !responses_semantically_equal(&outcome.final_response, &expected) {
                    report.mismatches += 1;
                    report.failures.push(format!(
                        "{} replay final response differed from recorded final response",
                        record.trace_id
                    ));
                }
            }
            Err(error) => report
                .failures
                .push(format!("{} replay failed: {error}", record.trace_id)),
        }
    }

    Ok(report)
}

fn responses_semantically_equal(
    left: &crate::openai::ChatCompletionResponse,
    right: &crate::openai::ChatCompletionResponse,
) -> bool {
    if left.model != right.model || left.choices.len() != right.choices.len() {
        return false;
    }

    left.choices
        .iter()
        .zip(&right.choices)
        .all(|(left, right)| {
            left.finish_reason == right.finish_reason
                && left.message.role == right.message.role
                && left.message.content_text() == right.message.content_text()
                && normalize_tool_calls(left.message.tool_calls.as_deref())
                    == normalize_tool_calls(right.message.tool_calls.as_deref())
        })
}

fn normalize_tool_calls(calls: Option<&[crate::openai::OpenAiToolCall]>) -> Vec<serde_json::Value> {
    calls
        .unwrap_or_default()
        .iter()
        .map(|call| {
            serde_json::json!({
                "type": call.call_type,
                "name": call.function.name,
                "arguments": serde_json::from_str::<serde_json::Value>(&call.function.arguments)
                    .unwrap_or_else(|_| serde_json::Value::String(call.function.arguments.clone()))
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Map};

    use super::*;
    use crate::openai::{
        ChatChoice, ChatCompletionResponse, ChatMessage, OpenAiFunctionCall, OpenAiToolCall,
    };

    #[test]
    fn replay_normalization_ignores_generated_ids() {
        let mut left = ChatCompletionResponse::empty_for_model("m");
        let mut right = ChatCompletionResponse::empty_for_model("m");
        left.choices.push(choice("call_a"));
        right.choices.push(choice("call_b"));

        assert!(responses_semantically_equal(&left, &right));
    }

    fn choice(id: &str) -> ChatChoice {
        ChatChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".to_string(),
                content: Some(json!(null)),
                name: None,
                tool_call_id: None,
                tool_calls: Some(vec![OpenAiToolCall {
                    id: id.to_string(),
                    call_type: "function".to_string(),
                    function: OpenAiFunctionCall {
                        name: "read_file".to_string(),
                        arguments: "{}".to_string(),
                    },
                }]),
                extra: Map::new(),
            },
            finish_reason: Some("tool_calls".to_string()),
            logprobs: None,
            extra: Map::new(),
        }
    }
}
