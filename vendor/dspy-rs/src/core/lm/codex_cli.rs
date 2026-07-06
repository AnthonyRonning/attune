use std::{env, path::Path, process::Stdio};

use anyhow::Result;
use rig::{
    OneOrMany,
    completion::{
        AssistantContent, CompletionError, CompletionRequest, CompletionResponse, Message, Usage,
    },
    message::{Text, UserContent},
};
use tokio::{io::AsyncWriteExt, process::Command};

#[derive(Clone, Debug)]
pub struct CodexCliCompletionModel {
    model: String,
    reasoning_effort: String,
    codex_bin: String,
}

impl CodexCliCompletionModel {
    pub fn new(model_spec: &str) -> Result<Self> {
        let (model, inline_effort) = model_spec
            .trim()
            .split_once('@')
            .map(|(model, effort)| (model.trim(), Some(effort.trim())))
            .unwrap_or((model_spec.trim(), None));
        if model.is_empty() {
            anyhow::bail!("codex model string must include a model, for example codex:gpt-5.5");
        }

        let reasoning_effort = inline_effort
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .or_else(|| env::var("ATTUNE_CODEX_REASONING_EFFORT").ok())
            .unwrap_or_else(|| "medium".to_string());
        ensure_reasoning_effort_supported(&reasoning_effort)?;

        Ok(Self {
            model: model.to_string(),
            reasoning_effort,
            codex_bin: env::var("ATTUNE_CODEX_BIN").unwrap_or_else(|_| "codex".to_string()),
        })
    }

    async fn run_codex(&self, prompt: &str) -> Result<String, CompletionError> {
        let mut command = Command::new(&self.codex_bin);
        command
            .arg("exec")
            .arg("--ephemeral")
            .arg("--ignore-user-config")
            .arg("--ignore-rules")
            .arg("--skip-git-repo-check")
            .arg("--sandbox")
            .arg("read-only")
            .arg("--model")
            .arg(&self.model)
            .arg("-c")
            .arg(format!(
                "model_reasoning_effort=\"{}\"",
                self.reasoning_effort
            ))
            .arg("-")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        if let Ok(cwd) = env::var("ATTUNE_CODEX_CWD") {
            if !cwd.trim().is_empty() {
                command.current_dir(Path::new(cwd.trim()));
            }
        }

        let mut child = command.spawn().map_err(|error| {
            CompletionError::ProviderError(format!("failed to start codex CLI: {error}"))
        })?;
        let mut stdin = child.stdin.take().ok_or_else(|| {
            CompletionError::ProviderError("failed to open codex CLI stdin".to_string())
        })?;
        let prompt = prompt.to_string();
        let writer = tokio::spawn(async move {
            stdin.write_all(prompt.as_bytes()).await?;
            stdin.shutdown().await
        });

        let output = child.wait_with_output().await.map_err(|error| {
            CompletionError::ProviderError(format!("failed to wait for codex CLI: {error}"))
        })?;
        let write_result = writer.await.map_err(|error| {
            CompletionError::ProviderError(format!("codex CLI stdin task failed: {error}"))
        })?;
        write_result.map_err(|error| {
            CompletionError::ProviderError(format!("failed to write codex CLI stdin: {error}"))
        })?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(CompletionError::ProviderError(format!(
                "codex CLI exited with status {}: {}",
                output.status,
                truncate_for_error(stderr.trim())
            )));
        }

        let stdout = String::from_utf8(output.stdout).map_err(|error| {
            CompletionError::ResponseError(format!("codex CLI stdout was not UTF-8: {error}"))
        })?;
        let response = stdout.trim().to_string();
        if response.is_empty() {
            return Err(CompletionError::ResponseError(
                "codex CLI returned an empty final message".to_string(),
            ));
        }
        Ok(response)
    }
}

impl super::client_registry::CompletionProvider for CodexCliCompletionModel {
    async fn completion(
        &self,
        request: CompletionRequest,
    ) -> Result<CompletionResponse<()>, CompletionError> {
        if !request.tools.is_empty() {
            return Err(CompletionError::ProviderError(
                "codex CLI GEPA backend does not support tool calls".to_string(),
            ));
        }

        let prompt = codex_prompt_for_completion(&request);
        let response = self.run_codex(&prompt).await?;
        Ok(CompletionResponse {
            choice: OneOrMany::one(AssistantContent::Text(Text { text: response })),
            usage: Usage::new(),
            raw_response: (),
        })
    }
}

fn codex_prompt_for_completion(request: &CompletionRequest) -> String {
    let mut prompt = String::new();
    prompt.push_str(
        "You are serving as a Codex-mediated text completion backend for Attune GEPA.\n\
Follow the embedded completion request. Do not inspect files, run shell commands, use tools, or add commentary about Codex.\n\
Return only the assistant response requested by the embedded request.\n\n",
    );

    if let Some(preamble) = request.preamble.as_deref() {
        prompt.push_str("<system>\n");
        prompt.push_str(preamble);
        prompt.push_str("\n</system>\n\n");
    }

    prompt.push_str("<conversation>\n");
    for message in request.chat_history.iter() {
        push_message(&mut prompt, message);
    }
    prompt.push_str("</conversation>\n");

    prompt
}

fn push_message(prompt: &mut String, message: &Message) {
    match message {
        Message::User { content } => {
            prompt.push_str("<user>\n");
            for item in content.iter() {
                push_user_content(prompt, item);
            }
            prompt.push_str("</user>\n");
        }
        Message::Assistant { content, .. } => {
            prompt.push_str("<assistant>\n");
            for item in content.iter() {
                push_assistant_content(prompt, item);
            }
            prompt.push_str("</assistant>\n");
        }
    }
}

fn push_user_content(prompt: &mut String, content: &UserContent) {
    match content {
        UserContent::Text(text) => {
            prompt.push_str(&text.text);
            prompt.push('\n');
        }
        other => {
            prompt.push_str(&format!("{other:?}\n"));
        }
    }
}

fn push_assistant_content(prompt: &mut String, content: &AssistantContent) {
    match content {
        AssistantContent::Text(text) => {
            prompt.push_str(&text.text);
            prompt.push('\n');
        }
        AssistantContent::Reasoning(reasoning) => {
            prompt.push_str(&reasoning.reasoning.join("\n"));
            prompt.push('\n');
        }
        other => {
            prompt.push_str(&format!("{other:?}\n"));
        }
    }
}

fn ensure_reasoning_effort_supported(value: &str) -> Result<()> {
    match value {
        "minimal" | "low" | "medium" | "high" | "xhigh" => Ok(()),
        other => anyhow::bail!(
            "unsupported Codex reasoning effort {other:?}; use minimal, low, medium, high, or xhigh"
        ),
    }
}

fn truncate_for_error(value: &str) -> String {
    const LIMIT: usize = 1000;
    if value.len() <= LIMIT {
        value.to_string()
    } else {
        format!("{}...", value.chars().take(LIMIT).collect::<String>())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_inline_reasoning_effort() {
        let model = CodexCliCompletionModel::new("gpt-5.5@high").expect("codex model");
        assert_eq!(model.model, "gpt-5.5");
        assert_eq!(model.reasoning_effort, "high");
    }

    #[test]
    fn rejects_unknown_reasoning_effort() {
        let error = CodexCliCompletionModel::new("gpt-5.5@turbo").expect_err("invalid effort");
        assert!(error.to_string().contains("unsupported Codex reasoning effort"));
    }
}
