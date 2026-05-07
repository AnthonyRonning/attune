use std::{
    collections::{BTreeMap, BTreeSet},
    env,
    error::Error,
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
};

use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Default, Deserialize)]
struct BuiltinDefaultsManifest {
    #[serde(default)]
    artifacts: Vec<ManifestArtifact>,
    #[serde(default)]
    profiles: Vec<ManifestProfile>,
}

#[derive(Debug, Deserialize)]
struct ManifestArtifact {
    id: String,
    artifact_type: String,
    path: String,
    #[serde(default)]
    profile: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ManifestProfile {
    name: String,
    revision: u32,
    #[serde(default)]
    model_patterns: Vec<String>,
    #[serde(default)]
    dsrs_history_format: Option<String>,
    #[serde(default)]
    request_adapter_artifact: Option<String>,
    #[serde(default)]
    correction_agent_artifact: Option<String>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?);
    let manifest_path = root.join("profiles/builtin-defaults.toml");
    println!("cargo:rerun-if-changed={}", manifest_path.display());

    let manifest = if manifest_path.exists() {
        let content = fs::read_to_string(&manifest_path)?;
        toml::from_str::<BuiltinDefaultsManifest>(&content)?
    } else {
        BuiltinDefaultsManifest::default()
    };

    validate_manifest(&root, &manifest)?;

    let generated = generate_builtin_defaults(&manifest)?;
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    fs::write(out_dir.join("builtin_defaults.rs"), generated)?;
    Ok(())
}

fn validate_manifest(
    root: &Path,
    manifest: &BuiltinDefaultsManifest,
) -> Result<(), Box<dyn Error>> {
    let mut artifact_ids = BTreeSet::new();
    let mut artifacts_by_id = BTreeMap::new();
    for artifact in &manifest.artifacts {
        if artifact.id.trim().is_empty() {
            return Err("built-in artifact id must not be empty".into());
        }
        if !artifact_ids.insert(artifact.id.as_str()) {
            return Err(format!("duplicate built-in artifact id {}", artifact.id).into());
        }
        let artifact_path = root.join(&artifact.path);
        println!("cargo:rerun-if-changed={}", artifact_path.display());
        let content = fs::read_to_string(&artifact_path)
            .map_err(|error| format!("failed to read {}: {error}", artifact_path.display()))?;
        validate_artifact_json(&artifact_path, artifact, &content)?;
        artifacts_by_id.insert(artifact.id.as_str(), artifact);
    }

    let mut profile_names = BTreeSet::new();
    for profile in &manifest.profiles {
        if profile.name.trim().is_empty() {
            return Err("built-in profile name must not be empty".into());
        }
        if !profile_names.insert(profile.name.as_str()) {
            return Err(format!("duplicate built-in profile default {}", profile.name).into());
        }
        if let Some(format) = &profile.dsrs_history_format {
            match format.as_str() {
                "append_only" | "regenerated_context" => {}
                other => {
                    return Err(format!(
                        "profile {} has unsupported dsrs_history_format {other:?}",
                        profile.name
                    )
                    .into())
                }
            }
        }
        validate_profile_artifact_ref(
            &profile.name,
            "request_adapter_artifact",
            profile.request_adapter_artifact.as_deref(),
            "request_adapter_instruction",
            &artifacts_by_id,
        )?;
        validate_profile_artifact_ref(
            &profile.name,
            "correction_agent_artifact",
            profile.correction_agent_artifact.as_deref(),
            "correction_agent_instruction",
            &artifacts_by_id,
        )?;
    }

    Ok(())
}

fn validate_artifact_json(
    path: &Path,
    manifest_artifact: &ManifestArtifact,
    content: &str,
) -> Result<(), Box<dyn Error>> {
    let value: Value = serde_json::from_str(content)
        .map_err(|error| format!("failed to parse {} as JSON: {error}", path.display()))?;
    let json_id = value.get("artifact_id").and_then(Value::as_str);
    if json_id != Some(manifest_artifact.id.as_str()) {
        return Err(format!(
            "{} artifact_id {:?} does not match manifest id {:?}",
            path.display(),
            json_id,
            manifest_artifact.id
        )
        .into());
    }
    let json_type = value.get("artifact_type").and_then(Value::as_str);
    if json_type != Some(manifest_artifact.artifact_type.as_str()) {
        return Err(format!(
            "{} artifact_type {:?} does not match manifest artifact_type {:?}",
            path.display(),
            json_type,
            manifest_artifact.artifact_type
        )
        .into());
    }
    if let Some(profile) = &manifest_artifact.profile {
        let json_profile = value.get("profile").and_then(Value::as_str);
        if json_profile != Some(profile.as_str()) {
            return Err(format!(
                "{} profile {:?} does not match manifest profile {:?}",
                path.display(),
                json_profile,
                profile
            )
            .into());
        }
    }
    let best_instruction = value
        .get("best_instruction")
        .or_else(|| value.get("instruction"))
        .or_else(|| value.get("prompt"))
        .or_else(|| value.get("content"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|instruction| !instruction.is_empty());
    if best_instruction.is_none() {
        return Err(format!(
            "{} did not contain best_instruction, instruction, prompt, or content",
            path.display()
        )
        .into());
    }
    Ok(())
}

fn validate_profile_artifact_ref<'a>(
    profile_name: &str,
    field: &str,
    artifact_id: Option<&'a str>,
    expected_type: &str,
    artifacts_by_id: &BTreeMap<&'a str, &'a ManifestArtifact>,
) -> Result<(), Box<dyn Error>> {
    let Some(artifact_id) = artifact_id else {
        return Ok(());
    };
    let artifact = artifacts_by_id.get(artifact_id).ok_or_else(|| {
        format!("profile {profile_name} references unknown {field} {artifact_id:?}")
    })?;
    if artifact.artifact_type != expected_type {
        return Err(format!(
            "profile {profile_name} {field} {artifact_id:?} has artifact_type {:?}; expected {expected_type:?}",
            artifact.artifact_type
        )
        .into());
    }
    if let Some(artifact_profile) = &artifact.profile {
        if artifact_profile != profile_name {
            return Err(format!(
                "profile {profile_name} references artifact {artifact_id:?} owned by profile {artifact_profile:?}"
            )
            .into());
        }
    }
    Ok(())
}

fn generate_builtin_defaults(manifest: &BuiltinDefaultsManifest) -> Result<String, Box<dyn Error>> {
    let mut output = String::new();
    writeln!(
        output,
        "// @generated by build.rs from profiles/builtin-defaults.toml\n"
    )?;

    writeln!(
        output,
        "pub static GENERATED_BUILTIN_ARTIFACTS: &[BuiltinArtifact] = &["
    )?;
    for artifact in &manifest.artifacts {
        writeln!(output, "    BuiltinArtifact {{")?;
        writeln!(output, "        id: {},", rust_string(&artifact.id))?;
        writeln!(
            output,
            "        artifact_type: {},",
            rust_string(&artifact.artifact_type)
        )?;
        writeln!(output, "        path: {},", rust_string(&artifact.path))?;
        writeln!(
            output,
            "        profile: {},",
            rust_option_string(artifact.profile.as_deref())
        )?;
        writeln!(
            output,
            "        content: {},",
            include_str_expr(&artifact.path)
        )?;
        writeln!(output, "    }},")?;
    }
    writeln!(output, "];")?;

    writeln!(
        output,
        "pub static GENERATED_BUILTIN_PROFILE_DEFAULTS: &[BuiltinProfileDefault] = &["
    )?;
    for profile in &manifest.profiles {
        writeln!(output, "    BuiltinProfileDefault {{")?;
        writeln!(output, "        name: {},", rust_string(&profile.name))?;
        writeln!(output, "        revision: {},", profile.revision)?;
        writeln!(
            output,
            "        model_patterns: &{},",
            rust_string_slice(&profile.model_patterns)
        )?;
        writeln!(
            output,
            "        dsrs_history_format: {},",
            rust_history_format(profile.dsrs_history_format.as_deref())
        )?;
        writeln!(
            output,
            "        request_adapter_artifact: {},",
            rust_option_string(profile.request_adapter_artifact.as_deref())
        )?;
        writeln!(
            output,
            "        correction_agent_artifact: {},",
            rust_option_string(profile.correction_agent_artifact.as_deref())
        )?;
        writeln!(output, "    }},")?;
    }
    writeln!(output, "];")?;

    Ok(output)
}

fn rust_string(value: &str) -> String {
    format!("{value:?}")
}

fn rust_option_string(value: Option<&str>) -> String {
    value
        .map(|value| format!("Some({})", rust_string(value)))
        .unwrap_or_else(|| "None".to_string())
}

fn rust_string_slice(values: &[String]) -> String {
    let values = values
        .iter()
        .map(|value| rust_string(value))
        .collect::<Vec<_>>();
    format!("[{}]", values.join(", "))
}

fn rust_history_format(value: Option<&str>) -> String {
    match value {
        Some("append_only") => "Some(DsrsHistoryFormat::AppendOnly)".to_string(),
        Some("regenerated_context") => "Some(DsrsHistoryFormat::RegeneratedContext)".to_string(),
        Some(other) => panic!("unsupported DSRs history format {other:?}"),
        None => "None".to_string(),
    }
}

fn escape_include_path(path: &str) -> String {
    path.replace('\\', "\\\\").replace('"', "\\\"")
}

fn include_str_expr(path: &str) -> String {
    if Path::new(path).is_absolute() {
        format!("include_str!({})", rust_string(path))
    } else {
        format!(
            "include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/{}\"))",
            escape_include_path(path)
        )
    }
}
