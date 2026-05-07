use std::path::Path;

use anyhow::{Context, Result};
use serde_json::Value;

pub fn extract_instruction_from_path(path: &Path, content: &str) -> Result<String> {
    let label = path.display().to_string();
    extract_instruction(
        &label,
        content,
        path.extension().and_then(|extension| extension.to_str()) == Some("json"),
    )
}

pub fn extract_instruction_from_json(label: &str, content: &str) -> Result<String> {
    extract_instruction(label, content, true)
}

fn extract_instruction(label: &str, content: &str, is_json: bool) -> Result<String> {
    if is_json {
        let value: Value = serde_json::from_str(content)
            .with_context(|| format!("failed to parse JSON artifact {label}"))?;
        return json_instruction_field(&value)
            .map(str::to_string)
            .filter(|instruction| !instruction.trim().is_empty())
            .with_context(|| {
                format!(
                    "JSON artifact {label} did not contain best_instruction, instruction, prompt, or content"
                )
            });
    }

    Ok(content.trim().to_string())
}

fn json_instruction_field(value: &Value) -> Option<&str> {
    value
        .get("best_instruction")
        .or_else(|| value.get("instruction"))
        .or_else(|| value.get("prompt"))
        .or_else(|| value.get("content"))
        .and_then(Value::as_str)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_best_instruction_from_json_artifact() {
        let instruction = extract_instruction_from_json(
            "test-artifact",
            r#"{"best_instruction":"Use DSRs carefully"}"#,
        )
        .unwrap();

        assert_eq!(instruction, "Use DSRs carefully");
    }
}
