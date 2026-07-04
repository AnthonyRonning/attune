use std::{
    net::SocketAddr,
    path::{Path, PathBuf},
};

use anyhow::Context;
use attune::{
    config::ProxyConfig,
    dataset::{
        export_dataset, export_request_adapter_dataset, DatasetExportConfig, DatasetExportFilter,
        RequestAdapterDatasetExportConfig,
    },
    eval::{run_regression_suite, RegressionConfig, RegressionFilter},
    gateway::Gateway,
    model_profile::{DsrsHistoryFormat, ProviderRouting},
    optimization::{
        optimize_correction_prompt, optimize_request_adapter_prompt, GepaOptimizationConfig,
        DEFAULT_GEPA_JUDGE_MODEL, DEFAULT_GEPA_LM_MAX_TOKENS, DEFAULT_GEPA_REFLECTION_MODEL,
        DEFAULT_GEPA_ROLE_BASE_URL,
    },
    promotion::{
        promote_artifact, promote_default_artifact, ArtifactPromotionConfig,
        DefaultArtifactPromotionConfig,
    },
    replay::{replay_traces, ReplayConfig},
    trace::{read_trace_records, trace_summaries, TraceSummary},
    trace_harness::{
        import_hermes_rows_scenarios, import_pi_trace_scenarios, inspect_trace_harness_scenarios,
        run_trace_harness, run_trace_harness_compare, TraceHarnessCompareConfig,
        TraceHarnessImportConfig, TraceHarnessRunConfig,
    },
};
use clap::{Parser, Subcommand};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

#[derive(Debug, Parser)]
#[command(
    version,
    about = "Attune agent contract runtime for OpenAI-compatible models"
)]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    #[arg(long, env = "ATTUNE_BIND_ADDR", default_value = "127.0.0.1:8080")]
    bind: SocketAddr,

    #[arg(long, env = "ATTUNE_CONFIG_PATH")]
    config: Option<String>,

    #[arg(long, env = "ATTUNE_UPSTREAM_BASE_URL")]
    upstream_base_url: Option<String>,

    #[arg(long, env = "ATTUNE_UPSTREAM_API_KEY")]
    upstream_api_key: Option<String>,

    #[arg(long, env = "ATTUNE_TRACE_PATH")]
    trace_path: Option<String>,
}

#[derive(Debug, Clone, Default)]
struct RuntimeOverrides {
    upstream_base_url: Option<String>,
    upstream_api_key: Option<String>,
    trace_path: Option<String>,
}

impl RuntimeOverrides {
    fn from_cli(cli: &Cli) -> Self {
        Self {
            upstream_base_url: cli.upstream_base_url.clone(),
            upstream_api_key: cli.upstream_api_key.clone(),
            trace_path: cli.trace_path.clone(),
        }
    }

    fn apply_to_proxy_config(&self, config: &mut ProxyConfig) {
        if let Some(base_url) = &self.upstream_base_url {
            config.upstream.base_url = base_url.clone();
        }
        config.upstream.api_key = first_non_empty([
            self.upstream_api_key.clone(),
            std::env::var("OPENROUTER_API_KEY").ok(),
            config.upstream.api_key.clone(),
        ]);
        if let Some(trace_path) = &self.trace_path {
            config.trace.path = trace_path.clone().into();
        }
    }
}

#[derive(Debug, Subcommand)]
enum Command {
    Serve,
    ExportDataset {
        #[arg(long, default_value = "traces/attune.jsonl")]
        trace_path: String,
        #[arg(long, default_value = "datasets/corrections.jsonl")]
        output_path: String,
        #[arg(long = "trace-id")]
        trace_ids: Vec<String>,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        profile: Option<String>,
        #[arg(long = "failure-kind")]
        failure_kinds: Vec<String>,
        #[arg(long = "repair-action")]
        repair_actions: Vec<String>,
        #[arg(long = "correction-result")]
        correction_results: Vec<String>,
        #[arg(long)]
        expected_repair_json: Option<String>,
        #[arg(long)]
        expected_repair_path: Option<String>,
        #[arg(long)]
        append: bool,
    },
    ExportRequestAdapterDataset {
        #[arg(long, default_value = "traces/attune.jsonl")]
        trace_path: String,
        #[arg(long, default_value = "datasets/request-adapter/request-adapter.jsonl")]
        output_path: String,
        #[arg(long = "trace-id")]
        trace_ids: Vec<String>,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        profile: Option<String>,
        #[arg(long = "failure-kind")]
        failure_kinds: Vec<String>,
        #[arg(long = "repair-action")]
        repair_actions: Vec<String>,
        #[arg(long = "correction-result")]
        correction_results: Vec<String>,
        #[arg(long)]
        expected_output_json: Option<String>,
        #[arg(long)]
        expected_output_path: Option<String>,
        #[arg(long)]
        use_final_response: bool,
        #[arg(long)]
        allow_unlabeled: bool,
        #[arg(long)]
        append: bool,
        #[arg(long)]
        observed_failure_kind: Option<String>,
        #[arg(long)]
        observed_problem: Option<String>,
        #[arg(long)]
        prompt_goal: Option<String>,
    },
    Replay {
        #[arg(long, default_value = "traces/attune.jsonl")]
        trace_path: String,
    },
    InspectTraces {
        #[arg(long, default_value = "traces/attune.jsonl")]
        trace_path: String,
        #[arg(long, default_value_t = 20)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    Eval {
        #[arg(long, default_value = "eval/regressions.jsonl")]
        suite_path: String,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        profile: Option<String>,
    },
    OptimizePrompts {
        #[arg(long, default_value = "datasets/corrections.jsonl")]
        dataset_path: String,
        #[arg(long, default_value = "datasets/gepa-correction-prompt.json")]
        output_path: String,
        #[arg(
            long,
            env = "ATTUNE_UPSTREAM_BASE_URL",
            default_value = "https://openrouter.ai/api/v1"
        )]
        base_url: String,
        #[arg(
            long,
            env = "ATTUNE_GEPA_REFLECTION_MODEL",
            default_value = DEFAULT_GEPA_REFLECTION_MODEL
        )]
        model: String,
        #[arg(long, env = "ATTUNE_GEPA_REFLECTION_BASE_URL")]
        reflection_base_url: Option<String>,
        #[arg(long, env = "ATTUNE_GEPA_REFLECTION_API_KEY")]
        reflection_api_key: Option<String>,
        #[arg(long, env = "ATTUNE_GEPA_JUDGE_MODEL", default_value = DEFAULT_GEPA_JUDGE_MODEL)]
        judge_model: String,
        #[arg(long, env = "ATTUNE_GEPA_JUDGE_BASE_URL")]
        judge_base_url: Option<String>,
        #[arg(long, env = "ATTUNE_GEPA_JUDGE_API_KEY")]
        judge_api_key: Option<String>,
        #[arg(long)]
        target_model: Option<String>,
        #[arg(long)]
        profile: Option<String>,
        #[arg(long)]
        profile_revision: Option<u32>,
        #[arg(long = "target-provider-ignore")]
        target_provider_ignore: Vec<String>,
        #[arg(long)]
        artifact_id: Option<String>,
        #[arg(long)]
        seed_artifact: Option<String>,
        #[arg(long, default_value_t = 3)]
        iterations: usize,
        #[arg(long, default_value_t = 12)]
        max_examples: usize,
        #[arg(long, default_value_t = DEFAULT_GEPA_LM_MAX_TOKENS)]
        lm_max_tokens: u32,
    },
    OptimizeRequestAdapterPrompt {
        #[arg(
            long,
            default_value = "datasets/request-adapter/gemma-dsrs-conservative.jsonl"
        )]
        dataset_path: String,
        #[arg(
            long,
            default_value = "datasets/request-adapter/gemma-dsrs-conservative-local-append-only-gepa.json"
        )]
        output_path: String,
        #[arg(
            long,
            env = "ATTUNE_UPSTREAM_BASE_URL",
            default_value = "https://openrouter.ai/api/v1"
        )]
        base_url: String,
        #[arg(
            long,
            env = "ATTUNE_GEPA_REFLECTION_MODEL",
            default_value = DEFAULT_GEPA_REFLECTION_MODEL
        )]
        model: String,
        #[arg(long, env = "ATTUNE_GEPA_REFLECTION_BASE_URL")]
        reflection_base_url: Option<String>,
        #[arg(long, env = "ATTUNE_GEPA_REFLECTION_API_KEY")]
        reflection_api_key: Option<String>,
        #[arg(long, env = "ATTUNE_GEPA_JUDGE_MODEL", default_value = DEFAULT_GEPA_JUDGE_MODEL)]
        judge_model: String,
        #[arg(long, env = "ATTUNE_GEPA_JUDGE_BASE_URL")]
        judge_base_url: Option<String>,
        #[arg(long, env = "ATTUNE_GEPA_JUDGE_API_KEY")]
        judge_api_key: Option<String>,
        #[arg(long)]
        target_model: Option<String>,
        #[arg(long)]
        profile: Option<String>,
        #[arg(long)]
        profile_revision: Option<u32>,
        #[arg(long)]
        dsrs_history_format: Option<DsrsHistoryFormat>,
        #[arg(long = "target-provider-ignore")]
        target_provider_ignore: Vec<String>,
        #[arg(long)]
        artifact_id: Option<String>,
        #[arg(long)]
        seed_artifact: Option<String>,
        #[arg(long, default_value_t = 3)]
        iterations: usize,
        #[arg(long, default_value_t = 12)]
        max_examples: usize,
        #[arg(long, default_value_t = DEFAULT_GEPA_LM_MAX_TOKENS)]
        lm_max_tokens: u32,
    },
    PromoteArtifact {
        #[arg(long)]
        config_path: Option<String>,
        #[arg(long)]
        artifact_path: String,
        #[arg(long)]
        artifact_reference: Option<String>,
        #[arg(long)]
        profile: Option<String>,
        #[arg(long = "model-pattern")]
        model_patterns: Vec<String>,
        #[arg(long)]
        dry_run: bool,
    },
    PromoteDefaultArtifact {
        #[arg(long)]
        manifest_path: Option<String>,
        #[arg(long)]
        artifact_path: String,
        #[arg(long)]
        profile: Option<String>,
        #[arg(long = "model-pattern")]
        model_patterns: Vec<String>,
        #[arg(long)]
        dry_run: bool,
    },
    TraceHarness {
        #[command(subcommand)]
        command: TraceHarnessCommand,
    },
}

#[derive(Debug, Subcommand)]
enum TraceHarnessCommand {
    ImportPi {
        #[arg(long)]
        input_path: String,
        #[arg(
            long,
            default_value = "eval/trace-harness/scenarios/pi-mono.local.jsonl"
        )]
        output_path: String,
        #[arg(long, default_value = "google/gemma-4-26b-a4b-it")]
        model: String,
        #[arg(long, default_value_t = 24)]
        max_scenarios: usize,
        #[arg(long, default_value_t = 13)]
        seed: u64,
        #[arg(long)]
        append: bool,
        #[arg(long, default_value_t = 24)]
        max_messages: usize,
        #[arg(long, default_value_t = 60000)]
        max_request_chars: usize,
    },
    ImportHermesRows {
        #[arg(long)]
        input_path: String,
        #[arg(
            long,
            default_value = "eval/trace-harness/scenarios/hermes.local.jsonl"
        )]
        output_path: String,
        #[arg(long, default_value = "google/gemma-4-26b-a4b-it")]
        model: String,
        #[arg(long, default_value_t = 24)]
        max_scenarios: usize,
        #[arg(long, default_value_t = 13)]
        seed: u64,
        #[arg(long)]
        append: bool,
        #[arg(long, default_value_t = 24)]
        max_messages: usize,
        #[arg(long, default_value_t = 60000)]
        max_request_chars: usize,
    },
    Inspect {
        #[arg(long)]
        scenarios_path: String,
        #[arg(long, default_value_t = 20)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    Run {
        #[arg(long)]
        scenarios_path: String,
        #[arg(long, default_value = "eval/trace-harness/results/run.local.json")]
        output_path: String,
        #[arg(long)]
        proxy_url: Option<String>,
        #[arg(long)]
        model: Option<String>,
        #[arg(long, default_value_t = 24)]
        limit: usize,
        #[arg(long, default_value_t = 180)]
        request_timeout_seconds: u64,
        #[arg(long, default_value_t = 3)]
        retries: usize,
        #[arg(long, default_value_t = 1000)]
        retry_backoff_ms: u64,
        #[arg(long, default_value_t = 1)]
        parallel: usize,
        #[arg(long = "provider-order", value_delimiter = ',')]
        provider_order: Vec<String>,
        #[arg(long = "provider-only", value_delimiter = ',')]
        provider_only: Vec<String>,
        #[arg(long = "provider-ignore", value_delimiter = ',')]
        provider_ignore: Vec<String>,
        #[arg(long)]
        disable_provider_fallbacks: bool,
        #[arg(long)]
        require_provider_parameters: bool,
    },
    Compare {
        #[arg(long)]
        scenarios_path: String,
        #[arg(long, default_value = "eval/trace-harness/results/compare.local.json")]
        output_path: String,
        #[arg(long)]
        proxy_url: Option<String>,
        #[arg(long)]
        baseline_base_url: Option<String>,
        #[arg(long)]
        model: Option<String>,
        #[arg(long, default_value_t = 24)]
        limit: usize,
        #[arg(long, default_value_t = 180)]
        request_timeout_seconds: u64,
        #[arg(long, default_value_t = 3)]
        retries: usize,
        #[arg(long, default_value_t = 1000)]
        retry_backoff_ms: u64,
        #[arg(long, default_value_t = 1)]
        parallel: usize,
        #[arg(long = "provider-order", value_delimiter = ',')]
        provider_order: Vec<String>,
        #[arg(long = "provider-only", value_delimiter = ',')]
        provider_only: Vec<String>,
        #[arg(long = "provider-ignore", value_delimiter = ',')]
        provider_ignore: Vec<String>,
        #[arg(long)]
        disable_provider_fallbacks: bool,
        #[arg(long)]
        require_provider_parameters: bool,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(tracing_subscriber::fmt::layer())
        .init();
    tracing::debug!(
        "logging initialized; set RUST_LOG=attune=debug,tower_http=debug for more detail or attune=trace for request-shape traces"
    );

    let cli = Cli::parse();
    let runtime_config_path = cli.config.as_deref().map(PathBuf::from);
    let runtime_overrides = RuntimeOverrides::from_cli(&cli);
    let bind = cli.bind;
    match cli.command.unwrap_or(Command::Serve) {
        Command::Serve => {
            let config =
                load_runtime_config(runtime_config_path.as_deref(), &runtime_overrides).await?;
            let gateway = Gateway::new(config)?;
            gateway.serve(bind).await.context("proxy server failed")
        }
        Command::ExportDataset {
            trace_path,
            output_path,
            trace_ids,
            model,
            profile,
            failure_kinds,
            repair_actions,
            correction_results,
            expected_repair_json,
            expected_repair_path,
            append,
        } => {
            let expected_repair =
                read_optional_json_value(expected_repair_json, expected_repair_path).await?;
            export_dataset(DatasetExportConfig {
                trace_path: trace_path.into(),
                output_path: output_path.into(),
                filter: DatasetExportFilter {
                    trace_ids,
                    model,
                    profile,
                    failure_kinds,
                    repair_actions,
                    correction_results,
                },
                expected_repair,
                append,
            })
            .await
        }
        Command::ExportRequestAdapterDataset {
            trace_path,
            output_path,
            trace_ids,
            model,
            profile,
            failure_kinds,
            repair_actions,
            correction_results,
            expected_output_json,
            expected_output_path,
            use_final_response,
            allow_unlabeled,
            append,
            observed_failure_kind,
            observed_problem,
            prompt_goal,
        } => {
            let expected_output =
                read_optional_json_value(expected_output_json, expected_output_path).await?;
            export_request_adapter_dataset(RequestAdapterDatasetExportConfig {
                trace_path: trace_path.into(),
                output_path: output_path.into(),
                filter: DatasetExportFilter {
                    trace_ids,
                    model,
                    profile,
                    failure_kinds,
                    repair_actions,
                    correction_results,
                },
                expected_output,
                use_final_response,
                allow_unlabeled,
                append,
                observed_failure_kind,
                observed_problem,
                prompt_goal,
            })
            .await
        }
        Command::Replay { trace_path } => {
            replay_traces(ReplayConfig {
                trace_path: trace_path.into(),
            })
            .await
        }
        Command::InspectTraces {
            trace_path,
            limit,
            json,
        } => {
            let records = read_trace_records(&trace_path.into()).await?;
            let summaries = trace_summaries(&records, limit);
            if json {
                println!("{}", serde_json::to_string_pretty(&summaries)?);
            } else {
                print_trace_summaries(&summaries);
            }
            Ok(())
        }
        Command::Eval {
            suite_path,
            model,
            profile,
        } => {
            let proxy_config =
                load_runtime_config(runtime_config_path.as_deref(), &runtime_overrides).await?;
            let report = run_regression_suite(RegressionConfig {
                suite_path: suite_path.into(),
                proxy_config,
                filter: RegressionFilter { model, profile },
            })
            .await?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Command::OptimizePrompts {
            dataset_path,
            output_path,
            base_url,
            model,
            reflection_base_url,
            reflection_api_key,
            judge_model,
            judge_base_url,
            judge_api_key,
            target_model,
            profile,
            profile_revision,
            target_provider_ignore,
            artifact_id,
            seed_artifact,
            iterations,
            max_examples,
            lm_max_tokens,
        } => {
            let reflection_base_url = gepa_role_base_url(&model, reflection_base_url.as_deref());
            let judge_base_url = gepa_role_base_url(&judge_model, judge_base_url.as_deref());
            let report = optimize_correction_prompt(GepaOptimizationConfig {
                dataset_path: dataset_path.into(),
                output_path: output_path.into(),
                base_url,
                api_key: std::env::var("OPENROUTER_API_KEY").ok(),
                reflection_api_key: gepa_role_api_key(
                    &model,
                    reflection_base_url.as_deref(),
                    reflection_api_key,
                ),
                reflection_base_url,
                model,
                judge_api_key: gepa_role_api_key(
                    &judge_model,
                    judge_base_url.as_deref(),
                    judge_api_key,
                ),
                judge_base_url,
                judge_model,
                target_model,
                profile,
                profile_revision,
                dsrs_history_format: None,
                target_provider: target_provider_from_ignore(target_provider_ignore),
                artifact_id,
                seed_artifact_path: seed_artifact.map(Into::into),
                iterations,
                max_examples,
                lm_max_tokens,
            })
            .await?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Command::OptimizeRequestAdapterPrompt {
            dataset_path,
            output_path,
            base_url,
            model,
            reflection_base_url,
            reflection_api_key,
            judge_model,
            judge_base_url,
            judge_api_key,
            target_model,
            profile,
            profile_revision,
            dsrs_history_format,
            target_provider_ignore,
            artifact_id,
            seed_artifact,
            iterations,
            max_examples,
            lm_max_tokens,
        } => {
            let reflection_base_url = gepa_role_base_url(&model, reflection_base_url.as_deref());
            let judge_base_url = gepa_role_base_url(&judge_model, judge_base_url.as_deref());
            let report = optimize_request_adapter_prompt(GepaOptimizationConfig {
                dataset_path: dataset_path.into(),
                output_path: output_path.into(),
                base_url,
                api_key: std::env::var("OPENROUTER_API_KEY").ok(),
                reflection_api_key: gepa_role_api_key(
                    &model,
                    reflection_base_url.as_deref(),
                    reflection_api_key,
                ),
                reflection_base_url,
                model,
                judge_api_key: gepa_role_api_key(
                    &judge_model,
                    judge_base_url.as_deref(),
                    judge_api_key,
                ),
                judge_base_url,
                judge_model,
                target_model,
                profile,
                profile_revision,
                dsrs_history_format,
                target_provider: target_provider_from_ignore(target_provider_ignore),
                artifact_id,
                seed_artifact_path: seed_artifact.map(Into::into),
                iterations,
                max_examples,
                lm_max_tokens,
            })
            .await?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Command::PromoteArtifact {
            config_path,
            artifact_path,
            artifact_reference,
            profile,
            model_patterns,
            dry_run,
        } => {
            let config_path = config_path
                .map(PathBuf::from)
                .or_else(|| runtime_config_path.clone())
                .unwrap_or_else(|| PathBuf::from("configs/model-profiles.toml"));
            let report = promote_artifact(ArtifactPromotionConfig {
                config_path,
                artifact_path: artifact_path.into(),
                artifact_reference,
                profile,
                model_patterns,
                dry_run,
            })
            .await?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Command::PromoteDefaultArtifact {
            manifest_path,
            artifact_path,
            profile,
            model_patterns,
            dry_run,
        } => {
            let manifest_path = manifest_path
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("profiles/builtin-defaults.toml"));
            let report = promote_default_artifact(DefaultArtifactPromotionConfig {
                manifest_path,
                artifact_path: artifact_path.into(),
                profile,
                model_patterns,
                dry_run,
            })
            .await?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
        Command::TraceHarness { command } => match command {
            TraceHarnessCommand::ImportPi {
                input_path,
                output_path,
                model,
                max_scenarios,
                seed,
                append,
                max_messages,
                max_request_chars,
            } => {
                let report = import_pi_trace_scenarios(TraceHarnessImportConfig {
                    input_path: input_path.into(),
                    output_path: output_path.into(),
                    model,
                    max_scenarios,
                    seed,
                    append,
                    max_messages,
                    max_request_chars,
                })
                .await?;
                println!("{}", serde_json::to_string_pretty(&report)?);
                Ok(())
            }
            TraceHarnessCommand::ImportHermesRows {
                input_path,
                output_path,
                model,
                max_scenarios,
                seed,
                append,
                max_messages,
                max_request_chars,
            } => {
                let report = import_hermes_rows_scenarios(TraceHarnessImportConfig {
                    input_path: input_path.into(),
                    output_path: output_path.into(),
                    model,
                    max_scenarios,
                    seed,
                    append,
                    max_messages,
                    max_request_chars,
                })
                .await?;
                println!("{}", serde_json::to_string_pretty(&report)?);
                Ok(())
            }
            TraceHarnessCommand::Inspect {
                scenarios_path,
                limit,
                json,
            } => {
                let previews =
                    inspect_trace_harness_scenarios(&PathBuf::from(scenarios_path), limit).await?;
                if json {
                    println!("{}", serde_json::to_string_pretty(&previews)?);
                } else {
                    for preview in previews {
                        println!(
                            "{} [{}] messages={} tools={} observed={} latest_user={}",
                            preview.id,
                            preview.dataset,
                            preview.messages,
                            preview.tools,
                            preview.observed_kind,
                            preview.latest_user.unwrap_or_default()
                        );
                    }
                }
                Ok(())
            }
            TraceHarnessCommand::Run {
                scenarios_path,
                output_path,
                proxy_url,
                model,
                limit,
                request_timeout_seconds,
                retries,
                retry_backoff_ms,
                parallel,
                provider_order,
                provider_only,
                provider_ignore,
                disable_provider_fallbacks,
                require_provider_parameters,
            } => {
                let proxy_config =
                    load_runtime_config(runtime_config_path.as_deref(), &runtime_overrides).await?;
                let report = run_trace_harness(TraceHarnessRunConfig {
                    scenarios_path: scenarios_path.into(),
                    output_path: output_path.into(),
                    proxy_url,
                    model,
                    limit,
                    request_timeout_seconds,
                    retries,
                    retry_backoff_ms,
                    parallel,
                    provider: provider_routing_from_flags(
                        provider_order,
                        provider_only,
                        provider_ignore,
                        disable_provider_fallbacks,
                        require_provider_parameters,
                    ),
                    proxy_config,
                })
                .await?;
                println!("{}", serde_json::to_string_pretty(&report)?);
                Ok(())
            }
            TraceHarnessCommand::Compare {
                scenarios_path,
                output_path,
                proxy_url,
                baseline_base_url,
                model,
                limit,
                request_timeout_seconds,
                retries,
                retry_backoff_ms,
                parallel,
                provider_order,
                provider_only,
                provider_ignore,
                disable_provider_fallbacks,
                require_provider_parameters,
            } => {
                let proxy_config =
                    load_runtime_config(runtime_config_path.as_deref(), &runtime_overrides).await?;
                let report = run_trace_harness_compare(TraceHarnessCompareConfig {
                    scenarios_path: scenarios_path.into(),
                    output_path: output_path.into(),
                    proxy_url,
                    baseline_base_url,
                    model,
                    limit,
                    request_timeout_seconds,
                    retries,
                    retry_backoff_ms,
                    parallel,
                    provider: provider_routing_from_flags(
                        provider_order,
                        provider_only,
                        provider_ignore,
                        disable_provider_fallbacks,
                        require_provider_parameters,
                    ),
                    proxy_config,
                })
                .await?;
                println!("{}", serde_json::to_string_pretty(&report)?);
                Ok(())
            }
        },
    }
}

async fn load_runtime_config(
    path: Option<&Path>,
    overrides: &RuntimeOverrides,
) -> anyhow::Result<ProxyConfig> {
    let mut config = ProxyConfig::from_optional_path(path).await?;
    overrides.apply_to_proxy_config(&mut config);
    Ok(config)
}

fn provider_routing_from_flags(
    order: Vec<String>,
    only: Vec<String>,
    ignore: Vec<String>,
    disable_fallbacks: bool,
    require_parameters: bool,
) -> Option<ProviderRouting> {
    let provider = ProviderRouting {
        order: normalize_provider_list(order),
        only: normalize_provider_list(only),
        ignore: normalize_provider_list(ignore),
        allow_fallbacks: disable_fallbacks.then_some(false),
        require_parameters: require_parameters.then_some(true),
    };

    if provider.is_empty() {
        None
    } else {
        Some(provider)
    }
}

fn normalize_provider_list(values: Vec<String>) -> Vec<String> {
    values
        .into_iter()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .collect()
}

fn print_trace_summaries(summaries: &[TraceSummary]) {
    for summary in summaries {
        println!("trace {}", summary.trace_id);
        println!(
            "  model={} profile={} rev={} source={} mode={} messages={} tools={}",
            summary.model,
            summary.profile.as_deref().unwrap_or("-"),
            summary
                .profile_revision
                .map(|revision| revision.to_string())
                .unwrap_or_else(|| "-".to_string()),
            summary.profile_source.as_deref().unwrap_or("-"),
            summary.adapter_mode.as_deref().unwrap_or("-"),
            summary.request_messages,
            summary.request_tools
        );
        if summary.request_adapter_artifact.is_some() || summary.correction_agent_artifact.is_some()
        {
            println!(
                "  artifacts: request={} correction={}",
                summary.request_adapter_artifact.as_deref().unwrap_or("-"),
                summary.correction_agent_artifact.as_deref().unwrap_or("-")
            );
        }
        if let Some(latest_user) = &summary.latest_user {
            println!("  user: {latest_user}");
        }
        println!(
            "  upstream: finish={:?} content_len={} reasoning_len={}",
            summary.upstream_finish_reason,
            summary.upstream_content_len,
            summary.upstream_reasoning_len
        );
        if let Some(content) = &summary.upstream_content_preview {
            println!("    content: {content}");
        }
        println!(
            "  interpreted: suspicious={:?} content_len={} intents={}",
            summary.suspicious_stop,
            summary.interpreted_content_len,
            summary.tool_intents.join(", ")
        );
        if !summary.parse_events.is_empty() {
            println!("    events: {}", summary.parse_events.join(" | "));
        }
        if !summary.failure_kinds.is_empty() {
            println!("    failures: {}", summary.failure_kinds.join(" | "));
        }
        if !summary.correction_attempts.is_empty() {
            println!("  correction: {}", summary.correction_attempts.join(" | "));
        }
        if !summary.policy_decisions.is_empty() {
            println!("  policy: {}", summary.policy_decisions.join(" | "));
        }
        if !summary.repair_actions.is_empty() {
            println!("  repair: {}", summary.repair_actions.join(" | "));
        }
        println!(
            "  final: finish={:?} content_len={} tool_calls={}",
            summary.final_finish_reason, summary.final_content_len, summary.final_tool_calls
        );
        if let Some(content) = &summary.final_content_preview {
            println!("    content: {content}");
        }
        if let Some(error) = &summary.error {
            println!("  error: {error}");
        }
        println!();
    }
}

async fn read_optional_json_value(
    inline_json: Option<String>,
    json_path: Option<String>,
) -> anyhow::Result<Option<serde_json::Value>> {
    match (inline_json, json_path) {
        (Some(_), Some(_)) => {
            anyhow::bail!("pass only one of --expected-output-json or --expected-output-path")
        }
        (Some(value), None) => serde_json::from_str(&value)
            .map(Some)
            .context("failed to parse --expected-output-json"),
        (None, Some(path)) => {
            let content = tokio::fs::read_to_string(&path)
                .await
                .with_context(|| format!("failed to read {path}"))?;
            serde_json::from_str(&content)
                .map(Some)
                .with_context(|| format!("failed to parse {path}"))
        }
        (None, None) => Ok(None),
    }
}

fn first_non_empty(values: impl IntoIterator<Item = Option<String>>) -> Option<String> {
    values
        .into_iter()
        .flatten()
        .find(|value| !value.trim().is_empty())
}

fn target_provider_from_ignore(ignore: Vec<String>) -> Option<ProviderRouting> {
    let ignore = ignore
        .into_iter()
        .map(|provider| provider.trim().to_string())
        .filter(|provider| !provider.is_empty())
        .collect::<Vec<_>>();
    (!ignore.is_empty()).then(|| ProviderRouting {
        ignore,
        ..ProviderRouting::default()
    })
}

fn gepa_role_api_key(
    model: &str,
    base_url: Option<&str>,
    explicit: Option<String>,
) -> Option<String> {
    let provider_model = model.trim().to_ascii_lowercase();
    let base_url = base_url.unwrap_or_default().to_ascii_lowercase();
    first_non_empty([
        explicit,
        (provider_model.starts_with("openrouter:") || base_url.contains("openrouter.ai"))
            .then(|| std::env::var("OPENROUTER_API_KEY").ok())
            .flatten(),
        (provider_model.starts_with("anthropic:"))
            .then(|| std::env::var("ANTHROPIC_API_KEY").ok())
            .flatten(),
        (provider_model.starts_with("openai:"))
            .then(|| std::env::var("OPENAI_API_KEY").ok())
            .flatten(),
        (provider_model.starts_with("gemini:"))
            .then(|| std::env::var("GEMINI_API_KEY").ok())
            .flatten(),
    ])
}

fn gepa_role_base_url(model: &str, explicit: Option<&str>) -> Option<String> {
    if let Some(explicit) = explicit {
        let explicit = explicit.trim();
        return (!explicit.is_empty()).then(|| explicit.to_string());
    }

    gepa_role_uses_openrouter_default(model).then(|| DEFAULT_GEPA_ROLE_BASE_URL.to_string())
}

fn gepa_role_uses_openrouter_default(model: &str) -> bool {
    let model = model.trim().to_ascii_lowercase();
    model.starts_with("openrouter:") || (!model.contains(':') && model.contains('/'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gepa_role_base_url_defaults_openrouter_for_slash_model_ids() {
        assert_eq!(
            gepa_role_base_url("anthropic/claude-sonnet-5", None).as_deref(),
            Some(DEFAULT_GEPA_ROLE_BASE_URL)
        );
    }

    #[test]
    fn gepa_role_base_url_leaves_native_anthropic_models_direct() {
        assert_eq!(gepa_role_base_url("anthropic:claude-sonnet-5", None), None);
    }

    #[test]
    fn gepa_role_base_url_empty_explicit_value_disables_default() {
        assert_eq!(
            gepa_role_base_url("anthropic/claude-sonnet-5", Some("")),
            None
        );
    }
}
