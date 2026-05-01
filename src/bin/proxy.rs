use std::net::SocketAddr;

use anyhow::Context;
use clap::{Parser, Subcommand};
use model_correction_proxy::{
    config::ProxyConfig,
    dataset::{export_dataset, DatasetExportConfig},
    eval::{run_regression_suite, RegressionConfig},
    gateway::Gateway,
    optimization::{optimize_correction_prompt, GepaOptimizationConfig},
    replay::{replay_traces, ReplayConfig},
    trace::{read_trace_records, trace_summaries, TraceSummary},
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

#[derive(Debug, Parser)]
#[command(version, about = "OpenAI-compatible model correction proxy")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    #[arg(long, env = "MCP_BIND_ADDR", default_value = "127.0.0.1:8080")]
    bind: SocketAddr,

    #[arg(
        long,
        env = "MCP_UPSTREAM_BASE_URL",
        default_value = "https://openrouter.ai/api/v1"
    )]
    upstream_base_url: String,

    #[arg(long, env = "MCP_UPSTREAM_API_KEY")]
    upstream_api_key: Option<String>,

    #[arg(
        long,
        env = "MCP_TRACE_PATH",
        default_value = "traces/model-correction-proxy.jsonl"
    )]
    trace_path: String,
}

#[derive(Debug, Subcommand)]
enum Command {
    Serve,
    ExportDataset {
        #[arg(long, default_value = "traces/model-correction-proxy.jsonl")]
        trace_path: String,
        #[arg(long, default_value = "datasets/corrections.jsonl")]
        output_path: String,
    },
    Replay {
        #[arg(long, default_value = "traces/model-correction-proxy.jsonl")]
        trace_path: String,
    },
    InspectTraces {
        #[arg(long, default_value = "traces/model-correction-proxy.jsonl")]
        trace_path: String,
        #[arg(long, default_value_t = 20)]
        limit: usize,
        #[arg(long)]
        json: bool,
    },
    Eval {
        #[arg(long, default_value = "eval/regressions.jsonl")]
        suite_path: String,
    },
    OptimizePrompts {
        #[arg(long, default_value = "datasets/corrections.jsonl")]
        dataset_path: String,
        #[arg(long, default_value = "datasets/gepa-correction-prompt.json")]
        output_path: String,
        #[arg(
            long,
            env = "MCP_UPSTREAM_BASE_URL",
            default_value = "https://openrouter.ai/api/v1"
        )]
        base_url: String,
        #[arg(long, env = "MCP_OPTIMIZATION_MODEL", default_value = "qwen/qwen3-8b")]
        model: String,
        #[arg(long, default_value_t = 3)]
        iterations: usize,
        #[arg(long, default_value_t = 12)]
        max_examples: usize,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(tracing_subscriber::fmt::layer())
        .init();
    tracing::debug!(
        "logging initialized; set RUST_LOG=model_correction_proxy=debug,tower_http=debug for more detail or model_correction_proxy=trace for request-shape traces"
    );

    let cli = Cli::parse();
    match cli.command.unwrap_or(Command::Serve) {
        Command::Serve => {
            let mut config = ProxyConfig::default();
            config.upstream.base_url = cli.upstream_base_url;
            config.upstream.api_key = first_non_empty([
                cli.upstream_api_key,
                std::env::var("OPENROUTER_API_KEY").ok(),
            ]);
            config.trace.path = cli.trace_path.into();
            let gateway = Gateway::new(config)?;
            gateway.serve(cli.bind).await.context("proxy server failed")
        }
        Command::ExportDataset {
            trace_path,
            output_path,
        } => {
            export_dataset(DatasetExportConfig {
                trace_path: trace_path.into(),
                output_path: output_path.into(),
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
        Command::Eval { suite_path } => {
            let report = run_regression_suite(RegressionConfig {
                suite_path: suite_path.into(),
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
            iterations,
            max_examples,
        } => {
            let report = optimize_correction_prompt(GepaOptimizationConfig {
                dataset_path: dataset_path.into(),
                output_path: output_path.into(),
                base_url,
                api_key: std::env::var("OPENROUTER_API_KEY").ok(),
                model,
                iterations,
                max_examples,
            })
            .await?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
    }
}

fn print_trace_summaries(summaries: &[TraceSummary]) {
    for summary in summaries {
        println!("trace {}", summary.trace_id);
        println!(
            "  model={} profile={} mode={} messages={} tools={}",
            summary.model,
            summary.profile.as_deref().unwrap_or("-"),
            summary.adapter_mode.as_deref().unwrap_or("-"),
            summary.request_messages,
            summary.request_tools
        );
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

fn first_non_empty(values: impl IntoIterator<Item = Option<String>>) -> Option<String> {
    values
        .into_iter()
        .flatten()
        .find(|value| !value.trim().is_empty())
}
