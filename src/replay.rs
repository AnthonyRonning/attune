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
                if serde_json::to_value(&outcome.final_response)?
                    != serde_json::to_value(&expected)?
                {
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
