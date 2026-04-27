use std::net::SocketAddr;

use anyhow::Context;
use clap::{Parser, Subcommand};
use model_correction_proxy::{
    config::ProxyConfig,
    dataset::{export_dataset, DatasetExportConfig},
    eval::{run_regression_suite, RegressionConfig},
    gateway::Gateway,
    replay::{replay_traces, ReplayConfig},
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
    Eval {
        #[arg(long, default_value = "eval/regressions.jsonl")]
        suite_path: String,
    },
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();
    match cli.command.unwrap_or(Command::Serve) {
        Command::Serve => {
            let mut config = ProxyConfig::default();
            config.upstream.base_url = cli.upstream_base_url;
            config.upstream.api_key = cli
                .upstream_api_key
                .or_else(|| std::env::var("OPENROUTER_API_KEY").ok());
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
        Command::Eval { suite_path } => {
            let report = run_regression_suite(RegressionConfig {
                suite_path: suite_path.into(),
            })
            .await?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            Ok(())
        }
    }
}
