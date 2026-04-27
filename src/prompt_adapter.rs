use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    model_profile::{ModelProfile, ToolFormat, ToolMode},
    normalizer::NormalizedRequest,
    openai::{ChatCompletionRequest, ChatMessage, OpenAiTool},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptedRequest {
    pub upstream_request: ChatCompletionRequest,
    pub mode: ToolMode,
    pub injected_instruction: Option<String>,
}

pub fn adapt_request(
    normalized: &NormalizedRequest,
    profile: &ModelProfile,
) -> Result<AdaptedRequest> {
    let mut upstream_request = normalized.original.clone();
    upstream_request.stream = Some(false);

    if normalized.tools.is_empty() || profile.tool_mode == ToolMode::PassThrough {
        return Ok(AdaptedRequest {
            upstream_request,
            mode: ToolMode::PassThrough,
            injected_instruction: None,
        });
    }

    upstream_request.tools = None;
    upstream_request.tool_choice = None;
    upstream_request.parallel_tool_calls = None;
    let instruction = match profile.tool_format {
        ToolFormat::Xml => {
            render_xml_tool_instruction(profile, &normalized.tools, normalized.parallel_tool_calls)?
        }
        ToolFormat::TaggedJson => render_tagged_json_tool_instruction(
            profile,
            &normalized.tools,
            normalized.parallel_tool_calls,
        )?,
    };
    inject_system_instruction(&mut upstream_request.messages, &instruction);

    Ok(AdaptedRequest {
        upstream_request,
        mode: ToolMode::ProxyOwned,
        injected_instruction: Some(instruction),
    })
}

fn inject_system_instruction(messages: &mut Vec<ChatMessage>, instruction: &str) {
    if let Some(message) = messages
        .iter_mut()
        .find(|message| message.role == "system" || message.role == "developer")
    {
        let mut content = message.content_text().unwrap_or_default();
        if !content.is_empty() {
            content.push_str("\n\n");
        }
        content.push_str(instruction);
        message.set_content_text(content);
    } else {
        messages.insert(0, ChatMessage::new("system", instruction));
    }
}

pub fn render_xml_tool_instruction(
    profile: &ModelProfile,
    tools: &[OpenAiTool],
    parallel_allowed: bool,
) -> Result<String> {
    let mut out = String::new();
    out.push_str(&profile.tool_instruction);
    out.push_str("\n\nAvailable tools:\n<tools>\n");
    for tool in tools {
        out.push_str(&format!(
            r#"<tool name="{}">"#,
            escape_attr(&tool.function.name)
        ));
        out.push('\n');
        if let Some(description) = &tool.function.description {
            out.push_str("<description>");
            out.push_str(&escape_text(description));
            out.push_str("</description>\n");
        }
        out.push_str("<parameters>\n");
        out.push_str(&serde_json::to_string_pretty(&tool.function.parameters)?);
        out.push_str("\n</parameters>\n</tool>\n");
    }
    out.push_str("</tools>\n");
    if parallel_allowed {
        out.push_str("Parallel tool calls are allowed when useful.\n");
    } else {
        out.push_str(
            "The client disabled parallel tool calls. Emit at most one tool_call block.\n",
        );
    }
    Ok(out)
}

pub fn render_tagged_json_tool_instruction(
    profile: &ModelProfile,
    tools: &[OpenAiTool],
    parallel_allowed: bool,
) -> Result<String> {
    let payload = Value::Array(
        tools
            .iter()
            .map(|tool| serde_json::to_value(&tool.function))
            .collect::<serde_json::Result<Vec<_>>>()?,
    );
    Ok(format!(
        "{}\n\nAvailable tools are in <available_tools_json>{}</available_tools_json>.\nWhen using tools, emit <tool_calls_json>{{\"tool_calls\":[{{\"name\":\"tool_name\",\"arguments\":{{}}}}]}}</tool_calls_json>.\nParallel allowed: {}",
        profile.tool_instruction,
        serde_json::to_string_pretty(&payload)?,
        parallel_allowed
    ))
}

fn escape_attr(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

fn escape_text(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use serde_json::{json, Map};

    use super::*;
    use crate::openai::{OpenAiFunctionTool, OpenAiTool};

    #[test]
    fn proxy_owned_mode_removes_native_tools_and_injects_xml() {
        let tool = OpenAiTool {
            tool_type: "function".to_string(),
            function: OpenAiFunctionTool {
                name: "read_file".to_string(),
                description: Some("Read a file".to_string()),
                parameters: json!({"type":"object"}),
            },
        };
        let req = ChatCompletionRequest {
            model: "qwen".to_string(),
            messages: vec![ChatMessage::new("user", "read x")],
            tools: Some(vec![tool.clone()]),
            tool_choice: None,
            parallel_tool_calls: Some(false),
            stream: None,
            temperature: None,
            top_p: None,
            max_tokens: None,
            max_completion_tokens: None,
            response_format: None,
            extra: Map::new(),
        };
        let normalized = crate::normalizer::normalize_request(req).unwrap();
        let adapted = adapt_request(&normalized, &ModelProfile::qwen()).unwrap();

        assert!(adapted.upstream_request.tools.is_none());
        assert!(adapted.upstream_request.messages[0]
            .content_text()
            .unwrap()
            .contains(r#"<tool name="read_file">"#));
    }
}
