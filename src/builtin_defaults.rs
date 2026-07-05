use anyhow::{Context, Result};

use crate::{artifacts::extract_instruction_from_json, model_profile::DsrsHistoryFormat};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuiltinArtifact {
    pub id: &'static str,
    pub artifact_type: &'static str,
    pub path: &'static str,
    pub profile: Option<&'static str>,
    pub content: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BuiltinProfileDefault {
    pub name: &'static str,
    pub revision: u32,
    pub model_patterns: &'static [&'static str],
    pub dsrs_history_format: Option<DsrsHistoryFormat>,
    pub request_adapter_artifact: Option<&'static str>,
    pub correction_agent_artifact: Option<&'static str>,
}

include!(concat!(env!("OUT_DIR"), "/builtin_defaults.rs"));

pub const BUILTIN_REFERENCE_PREFIX: &str = "builtin:";

pub fn builtin_artifacts() -> &'static [BuiltinArtifact] {
    GENERATED_BUILTIN_ARTIFACTS
}

pub fn builtin_profile_defaults() -> &'static [BuiltinProfileDefault] {
    GENERATED_BUILTIN_PROFILE_DEFAULTS
}

pub fn profile_default(name: &str) -> Option<&'static BuiltinProfileDefault> {
    builtin_profile_defaults()
        .iter()
        .find(|profile| profile.name == name)
}

pub fn artifact_by_id(id: &str) -> Option<&'static BuiltinArtifact> {
    builtin_artifacts()
        .iter()
        .find(|artifact| artifact.id == id)
}

pub fn artifact_reference(id: &str) -> String {
    format!("{BUILTIN_REFERENCE_PREFIX}{id}")
}

pub fn artifact_id_from_reference(reference: &str) -> Option<&str> {
    reference
        .strip_prefix(BUILTIN_REFERENCE_PREFIX)
        .filter(|id| !id.trim().is_empty())
}

pub fn instruction_for_id(id: &str) -> Result<String> {
    let artifact =
        artifact_by_id(id).with_context(|| format!("unknown built-in artifact id {id:?}"))?;
    extract_instruction_from_json(
        &format!("built-in artifact {}", artifact.id),
        artifact.content,
    )
}

pub fn instruction_for_reference(reference: &str) -> Result<Option<String>> {
    let Some(id) = artifact_id_from_reference(reference) else {
        return Ok(None);
    };
    instruction_for_id(id).map(Some)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma_request_adapter_artifact_is_embedded() {
        let profile = profile_default("gemma-dsrs-conservative").unwrap();
        assert_eq!(profile.revision, 9);
        assert_eq!(
            profile.request_adapter_artifact,
            Some("request-adapter/gemma-dsrs-conservative/sonnet5-shuffled-r1-append-only")
        );

        let instruction = instruction_for_id(profile.request_adapter_artifact.unwrap()).unwrap();
        assert!(instruction.contains("DSRs Structural Contract"));
        assert!(instruction.contains("MANDATORY INSPECTION FIRST"));
    }

    #[test]
    fn qwen_request_adapter_artifact_is_embedded() {
        let profile = profile_default("qwen-dsrs").unwrap();
        assert_eq!(profile.revision, 6);
        assert_eq!(
            profile.dsrs_history_format,
            Some(DsrsHistoryFormat::RegeneratedContext)
        );
        assert_eq!(
            profile.request_adapter_artifact,
            Some("request-adapter/qwen-dsrs/sonnet5-shuffled-r1-regenerated-context")
        );

        let instruction = instruction_for_id(profile.request_adapter_artifact.unwrap()).unwrap();
        assert!(instruction.contains("CRITICAL FORMATTING REQUIREMENT"));
        assert!(instruction.contains("CRITICAL JSON ESCAPING RULE"));
    }
}
