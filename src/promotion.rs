use std::{
    path::{Component, Path, PathBuf},
    str::FromStr,
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use toml_edit::{value, Array, ArrayOfTables, DocumentMut, Item, Table};

use crate::optimization::GepaOptimizationReport;

#[derive(Debug, Clone)]
pub struct ArtifactPromotionConfig {
    pub config_path: PathBuf,
    pub artifact_path: PathBuf,
    pub artifact_reference: Option<String>,
    pub profile: Option<String>,
    pub model_patterns: Vec<String>,
    pub dry_run: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactPromotionReport {
    pub config_path: PathBuf,
    pub artifact_path: PathBuf,
    pub artifact_reference: String,
    pub artifact_type: String,
    pub promoted_field: String,
    pub profile: String,
    pub target_model: Option<String>,
    pub model_patterns: Vec<String>,
    pub previous_revision: Option<u32>,
    pub new_revision: u32,
    pub previous_artifact: Option<String>,
    pub new_artifact: String,
    pub created_profile: bool,
    pub dry_run: bool,
}

pub async fn promote_artifact(config: ArtifactPromotionConfig) -> Result<ArtifactPromotionReport> {
    let artifact = read_artifact_report(&config.artifact_path).await?;
    let promoted_field = promoted_field(&artifact.artifact_type)?;
    if let (Some(requested_profile), Some(artifact_profile)) = (&config.profile, &artifact.profile)
    {
        if requested_profile != artifact_profile {
            anyhow::bail!(
                "artifact profile {artifact_profile:?} does not match requested profile {requested_profile:?}"
            );
        }
    }
    let profile_name = config
        .profile
        .clone()
        .or_else(|| artifact.profile.clone())
        .with_context(|| {
            format!(
                "artifact {} did not declare a profile; pass --profile",
                config.artifact_path.display()
            )
        })?;
    let artifact_reference = config.artifact_reference.clone().unwrap_or_else(|| {
        artifact_reference_for_config(&config.config_path, &config.artifact_path)
    });

    let content = match tokio::fs::read_to_string(&config.config_path).await {
        Ok(content) => content,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => String::new(),
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed to read {}", config.config_path.display()))
        }
    };
    let mut document = if content.trim().is_empty() {
        DocumentMut::new()
    } else {
        DocumentMut::from_str(&content)
            .with_context(|| format!("failed to parse {}", config.config_path.display()))?
    };

    let outcome = promote_profile_in_document(
        &mut document,
        &profile_name,
        &config.model_patterns,
        artifact.target_model.as_deref(),
        promoted_field,
        &artifact_reference,
    )?;

    if !config.dry_run {
        if let Some(parent) = config.config_path.parent() {
            tokio::fs::create_dir_all(parent)
                .await
                .with_context(|| format!("failed to create {}", parent.display()))?;
        }
        tokio::fs::write(&config.config_path, document.to_string())
            .await
            .with_context(|| format!("failed to write {}", config.config_path.display()))?;
    }

    let new_artifact = artifact_reference.clone();
    Ok(ArtifactPromotionReport {
        config_path: config.config_path,
        artifact_path: config.artifact_path,
        artifact_reference,
        artifact_type: artifact.artifact_type,
        promoted_field: promoted_field.to_string(),
        profile: profile_name,
        target_model: artifact.target_model,
        model_patterns: outcome.model_patterns,
        previous_revision: outcome.previous_revision,
        new_revision: outcome.new_revision,
        previous_artifact: outcome.previous_artifact,
        new_artifact,
        created_profile: outcome.created_profile,
        dry_run: config.dry_run,
    })
}

async fn read_artifact_report(path: &Path) -> Result<GepaOptimizationReport> {
    let content = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("failed to read artifact {}", path.display()))?;
    serde_json::from_str(&content)
        .with_context(|| format!("failed to parse artifact {}", path.display()))
}

fn promoted_field(artifact_type: &str) -> Result<&'static str> {
    match artifact_type {
        "request_adapter_instruction" => Ok("request_adapter_artifact"),
        "correction_agent_instruction" => Ok("correction_agent_artifact"),
        other => anyhow::bail!(
            "unsupported artifact_type {other:?}; expected request_adapter_instruction or correction_agent_instruction"
        ),
    }
}

fn promote_profile_in_document(
    document: &mut DocumentMut,
    profile_name: &str,
    requested_model_patterns: &[String],
    fallback_target_model: Option<&str>,
    promoted_field: &str,
    artifact_reference: &str,
) -> Result<PromotionOutcome> {
    ensure_profiles_array(document)?;
    let profiles = document["model_profiles"]
        .as_array_of_tables_mut()
        .context("model_profiles was not an array of tables")?;

    let existing_index = profiles
        .iter()
        .position(|profile| table_string(profile, "name") == Some(profile_name));
    let created_profile = existing_index.is_none();
    let index = if let Some(index) = existing_index {
        index
    } else {
        let mut table = Table::new();
        table.insert("name", value(profile_name));
        profiles.push(table);
        profiles.len().saturating_sub(1)
    };
    let profile = profiles
        .get_mut(index)
        .context("promoted profile index was invalid")?;

    let previous_revision = table_u32(profile, "revision");
    let previous_artifact = table_string(profile, promoted_field).map(str::to_string);
    let model_patterns = selected_model_patterns(
        profile,
        requested_model_patterns,
        fallback_target_model,
        profile_name,
    );
    let new_revision = previous_revision.unwrap_or(0).saturating_add(1);
    profile.insert("name", value(profile_name));
    profile.insert("model_patterns", value(string_array(&model_patterns)));
    profile.insert("revision", value(i64::from(new_revision)));
    profile.insert(promoted_field, value(artifact_reference));

    Ok(PromotionOutcome {
        previous_revision,
        new_revision,
        previous_artifact,
        model_patterns,
        created_profile,
    })
}

#[derive(Debug)]
struct PromotionOutcome {
    previous_revision: Option<u32>,
    new_revision: u32,
    previous_artifact: Option<String>,
    model_patterns: Vec<String>,
    created_profile: bool,
}

fn ensure_profiles_array(document: &mut DocumentMut) -> Result<()> {
    if document.get("model_profiles").is_none() {
        document["model_profiles"] = Item::ArrayOfTables(ArrayOfTables::new());
        return Ok(());
    }

    if document["model_profiles"].is_array_of_tables() {
        Ok(())
    } else {
        anyhow::bail!("model_profiles exists but is not an array of tables")
    }
}

fn table_string<'a>(table: &'a Table, key: &str) -> Option<&'a str> {
    table.get(key).and_then(|item| item.as_str())
}

fn table_u32(table: &Table, key: &str) -> Option<u32> {
    table
        .get(key)
        .and_then(|item| item.as_integer())
        .and_then(|value| u32::try_from(value).ok())
}

fn selected_model_patterns(
    profile: &Table,
    requested_model_patterns: &[String],
    fallback_target_model: Option<&str>,
    profile_name: &str,
) -> Vec<String> {
    if !requested_model_patterns.is_empty() {
        return requested_model_patterns.to_vec();
    }

    let existing = table_string_array(profile, "model_patterns");
    if !existing.is_empty() {
        return existing;
    }

    fallback_target_model
        .map(|model| vec![model.to_string()])
        .unwrap_or_else(|| vec![profile_name.to_string()])
}

fn table_string_array(table: &Table, key: &str) -> Vec<String> {
    table
        .get(key)
        .and_then(|item| item.as_array())
        .map(|array| {
            array
                .iter()
                .filter_map(|value| value.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default()
}

fn string_array(values: &[String]) -> Array {
    let mut array = Array::new();
    for value in values {
        array.push(value.clone());
    }
    array.fmt();
    array
}

fn artifact_reference_for_config(config_path: &Path, artifact_path: &Path) -> String {
    let config_dir = config_path.parent().unwrap_or_else(|| Path::new("."));
    lexical_relative_path(config_dir, artifact_path)
        .unwrap_or_else(|| artifact_path.to_path_buf())
        .to_string_lossy()
        .to_string()
}

fn lexical_relative_path(from_dir: &Path, to_path: &Path) -> Option<PathBuf> {
    if from_dir.is_absolute() != to_path.is_absolute() {
        return None;
    }

    let from = normalized_components(from_dir);
    let to = normalized_components(to_path);
    if from.is_empty() || to.is_empty() {
        return None;
    }

    let common = from.iter().zip(&to).take_while(|(a, b)| a == b).count();
    let mut out = PathBuf::new();
    for _ in common..from.len() {
        out.push("..");
    }
    for component in &to[common..] {
        out.push(component);
    }
    Some(if out.as_os_str().is_empty() {
        PathBuf::from(".")
    } else {
        out
    })
}

fn normalized_components(path: &Path) -> Vec<String> {
    path.components()
        .filter_map(|component| match component {
            Component::Normal(value) => Some(value.to_string_lossy().to_string()),
            Component::ParentDir => Some("..".to_string()),
            Component::RootDir => Some("/".to_string()),
            Component::Prefix(prefix) => Some(prefix.as_os_str().to_string_lossy().to_string()),
            Component::CurDir => None,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[tokio::test]
    async fn promotes_request_adapter_artifact_into_existing_profile() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("configs/proxy.toml");
        let artifact_path = dir.path().join("datasets/gemma-request.json");
        tokio::fs::create_dir_all(config_path.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::create_dir_all(artifact_path.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(
            &config_path,
            r#"
[[model_profiles]]
name = "gemma-dsrs-conservative"
model_patterns = ["gemma", "gemma-4"]
revision = 2
request_adapter_artifact = "../datasets/old.json"
"#,
        )
        .await
        .unwrap();
        tokio::fs::write(
            &artifact_path,
            serde_json::to_vec_pretty(&json!({
                "artifact_id": "request-adapter/gemma-dsrs-conservative",
                "artifact_type": "request_adapter_instruction",
                "signature": "openai_tool_use_contract_profile_guidance/v1",
                "target_model": "google/gemma-4-26b-a4b-it",
                "profile": "gemma-dsrs-conservative",
                "optimizer_model": "google/gemma-4-26b-a4b-it",
                "created_at": "2026-05-07T00:00:00Z",
                "examples_loaded": 2,
                "best_instruction": "optimized",
                "best_average_score": 1.0,
                "total_rollouts": 12,
                "total_lm_calls": 6,
                "output_path": artifact_path
            }))
            .unwrap(),
        )
        .await
        .unwrap();

        let report = promote_artifact(ArtifactPromotionConfig {
            config_path: config_path.clone(),
            artifact_path,
            artifact_reference: None,
            profile: None,
            model_patterns: Vec::new(),
            dry_run: false,
        })
        .await
        .unwrap();

        assert_eq!(report.promoted_field, "request_adapter_artifact");
        assert_eq!(report.previous_revision, Some(2));
        assert_eq!(report.new_revision, 3);
        assert_eq!(
            report.previous_artifact.as_deref(),
            Some("../datasets/old.json")
        );
        assert_eq!(report.new_artifact, "../datasets/gemma-request.json");
        assert_eq!(report.model_patterns, vec!["gemma", "gemma-4"]);
        let config = tokio::fs::read_to_string(config_path).await.unwrap();
        assert!(config.contains("revision = 3"));
        assert!(config.contains("request_adapter_artifact"));
        assert!(config.contains("../datasets/gemma-request.json"));
    }

    #[tokio::test]
    async fn promotes_correction_artifact_into_new_profile() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("profiles.toml");
        let artifact_path = dir.path().join("correction.json");
        tokio::fs::write(
            &artifact_path,
            serde_json::to_vec_pretty(&json!({
                "artifact_id": "correction-agent/test",
                "artifact_type": "correction_agent_instruction",
                "signature": "correct_malformed_tool_response/v1",
                "target_model": "provider/model",
                "profile": "provider-dsrs",
                "optimizer_model": "provider/model",
                "created_at": "2026-05-07T00:00:00Z",
                "examples_loaded": 2,
                "best_instruction": "optimized",
                "best_average_score": 1.0,
                "total_rollouts": 12,
                "total_lm_calls": 6,
                "output_path": artifact_path
            }))
            .unwrap(),
        )
        .await
        .unwrap();

        let report = promote_artifact(ArtifactPromotionConfig {
            config_path: config_path.clone(),
            artifact_path,
            artifact_reference: Some("correction.json".to_string()),
            profile: None,
            model_patterns: Vec::new(),
            dry_run: false,
        })
        .await
        .unwrap();

        assert!(report.created_profile);
        assert_eq!(report.promoted_field, "correction_agent_artifact");
        assert_eq!(report.model_patterns, vec!["provider/model"]);
        let config = tokio::fs::read_to_string(config_path).await.unwrap();
        assert!(config.contains("name = \"provider-dsrs\""));
        assert!(config.contains("correction_agent_artifact = \"correction.json\""));
    }

    #[tokio::test]
    async fn dry_run_reports_without_writing_config() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("configs/proxy.toml");
        let artifact_path = dir.path().join("datasets/gemma-request.json");
        tokio::fs::create_dir_all(config_path.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::create_dir_all(artifact_path.parent().unwrap())
            .await
            .unwrap();
        tokio::fs::write(
            &config_path,
            r#"
[[model_profiles]]
name = "gemma-dsrs-conservative"
model_patterns = ["gemma"]
revision = 2
"#,
        )
        .await
        .unwrap();
        tokio::fs::write(
            &artifact_path,
            serde_json::to_vec_pretty(&json!({
                "artifact_id": "request-adapter/gemma-dsrs-conservative",
                "artifact_type": "request_adapter_instruction",
                "signature": "openai_tool_use_contract_profile_guidance/v1",
                "target_model": "google/gemma-4-26b-a4b-it",
                "profile": "gemma-dsrs-conservative",
                "optimizer_model": "google/gemma-4-26b-a4b-it",
                "created_at": "2026-05-07T00:00:00Z",
                "examples_loaded": 2,
                "best_instruction": "optimized",
                "best_average_score": 1.0,
                "total_rollouts": 12,
                "total_lm_calls": 6,
                "output_path": artifact_path
            }))
            .unwrap(),
        )
        .await
        .unwrap();

        let before = tokio::fs::read_to_string(&config_path).await.unwrap();
        let report = promote_artifact(ArtifactPromotionConfig {
            config_path: config_path.clone(),
            artifact_path,
            artifact_reference: None,
            profile: None,
            model_patterns: Vec::new(),
            dry_run: true,
        })
        .await
        .unwrap();
        let after = tokio::fs::read_to_string(config_path).await.unwrap();

        assert!(report.dry_run);
        assert_eq!(report.new_revision, 3);
        assert_eq!(before, after);
    }

    #[tokio::test]
    async fn rejects_profile_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let config_path = dir.path().join("profiles.toml");
        let artifact_path = dir.path().join("correction.json");
        tokio::fs::write(
            &artifact_path,
            serde_json::to_vec_pretty(&json!({
                "artifact_id": "correction-agent/test",
                "artifact_type": "correction_agent_instruction",
                "signature": "correct_malformed_tool_response/v1",
                "target_model": "provider/model",
                "profile": "provider-dsrs",
                "optimizer_model": "provider/model",
                "created_at": "2026-05-07T00:00:00Z",
                "examples_loaded": 2,
                "best_instruction": "optimized",
                "best_average_score": 1.0,
                "total_rollouts": 12,
                "total_lm_calls": 6,
                "output_path": artifact_path
            }))
            .unwrap(),
        )
        .await
        .unwrap();

        let error = promote_artifact(ArtifactPromotionConfig {
            config_path,
            artifact_path,
            artifact_reference: None,
            profile: Some("other-profile".to_string()),
            model_patterns: Vec::new(),
            dry_run: false,
        })
        .await
        .unwrap_err();

        assert!(error
            .to_string()
            .contains("does not match requested profile"));
    }
}
