// pattern: Imperative Shell

mod run_output;

use std::env;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anyhow::Context;
use clap::{Parser, Subcommand};
use futures::StreamExt;
use halter::prelude::*;
use halter_config::{export_json_schema, generate_starter_config, load_path};
use halter_protocol::{AssistantMessage, SessionEvent, SessionEventPayload};
use run_output::{
    JsonResultTracker, RunOutputArgs, RunOutputMode, strip_signatures_from_assistant_message,
    strip_signatures_from_session_event,
};
use tokio::io::{AsyncBufReadExt, BufReader};
use tracing::{debug, info};
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

#[derive(Debug, Parser)]
#[command(name = "halter")]
#[command(about = "Lightweight Rust agent harness SDK and portable binary")]
struct Cli {
    #[arg(long, default_value = "halter.toml")]
    config: PathBuf,
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Init,
    Chat,
    Run {
        #[command(flatten)]
        output: RunOutputArgs,
        task: String,
    },
    Resources,
    Validate,
    Config {
        #[command(subcommand)]
        command: ConfigCommands,
    },
}

#[derive(Debug, Subcommand)]
enum ConfigCommands {
    Schema,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    init_logging()?;
    let cli = Cli::parse();
    debug!(config_path = %cli.config.display(), command = ?cli.command, "parsed cli arguments");

    match cli.command {
        Commands::Init => init_config(&cli.config).await,
        Commands::Chat => chat(&cli.config).await,
        Commands::Run { task, output } => run_once(&cli.config, &task, output.mode()).await,
        Commands::Resources => show_resources(&cli.config).await,
        Commands::Validate => validate(&cli.config).await,
        Commands::Config {
            command: ConfigCommands::Schema,
        } => {
            println!("{}", export_json_schema()?);
            Ok(())
        }
    }
}

async fn init_config(path: &Path) -> anyhow::Result<()> {
    info!(path = %path.display(), "initializing starter config");
    if path.exists() {
        anyhow::bail!(
            "failed to initialize config: {} already exists",
            path.display()
        );
    }
    tokio::fs::write(path, generate_starter_config())
        .await
        .with_context(|| format!("failed to write {}", path.display()))?;
    println!("wrote {}", path.display());
    Ok(())
}

async fn validate(path: &Path) -> anyhow::Result<()> {
    info!(path = %path.display(), "validating config");
    load_path(path).await?;
    println!("config valid");
    Ok(())
}

async fn show_resources(path: &Path) -> anyhow::Result<()> {
    info!(path = %path.display(), "compiling resources");
    let config = load_path(path).await?;
    let resources = ResourceCompiler::from_config(&config).compile().await?;
    println!("revision: {}", resources.snapshot.revision.0);
    println!("skills: {}", resources.snapshot.skills.len());
    println!("agents: {}", resources.snapshot.agents.len());
    println!("plugins: {}", resources.snapshot.plugins.len());
    Ok(())
}

async fn run_once(path: &Path, task: &str, output_mode: RunOutputMode) -> anyhow::Result<()> {
    info!(
        path = %path.display(),
        output_mode = ?output_mode,
        task_len = task.len(),
        "running single turn"
    );
    let harness = Halter::from_config_file(path).await?;
    let session = harness.new_session(SessionInit::default()).await?;
    let mut events = session.submit_turn(Turn::user(task)).await?;
    let mut stdout = io::stdout();

    match output_mode {
        RunOutputMode::StreamingJson => {
            while let Some(event) = events.next().await {
                write_json_event(&mut stdout, &event?)?;
            }
            Ok(())
        }
        RunOutputMode::JsonResult => {
            let mut tracker = JsonResultTracker::default();
            while let Some(event) = events.next().await {
                let event = event?;
                if let Some(result) = tracker
                    .observe(&event.payload)
                    .map_err(anyhow::Error::msg)?
                {
                    write_json_result(&mut stdout, result)?;
                    return Ok(());
                }
            }
            anyhow::bail!("failed to receive final assistant result")
        }
    }
}

async fn chat(path: &Path) -> anyhow::Result<()> {
    info!(path = %path.display(), "starting chat session");
    let harness = Halter::from_config_file(path).await?;
    let session = harness.new_session(SessionInit::default()).await?;
    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();
    let mut stdout = io::stdout();

    println!("halter chat; submit an empty line or press ctrl-d to exit");
    while let Some(line) = lines.next_line().await.context("failed to read stdin")? {
        if line.trim().is_empty() {
            break;
        }

        let mut events = session.submit_turn(Turn::user(line)).await?;
        while let Some(event) = events.next().await {
            match event?.payload {
                SessionEventPayload::DeltaItem { delta } => {
                    write!(stdout, "{}", delta.text).context("failed to write stdout")?;
                    stdout.flush().context("failed to flush stdout")?;
                }
                SessionEventPayload::ToolOutput { chunk, .. } => {
                    write!(stdout, "{}", chunk).context("failed to write stdout")?;
                    stdout.flush().context("failed to flush stdout")?;
                }
                SessionEventPayload::TurnCompleted { .. } => {
                    writeln!(stdout).context("failed to write newline")?;
                    break;
                }
                SessionEventPayload::TurnFailed { error, .. } => anyhow::bail!(error),
                _ => {}
            }
        }
    }
    Ok(())
}

fn write_json_event(stdout: &mut impl Write, event: &SessionEvent) -> anyhow::Result<()> {
    let event = strip_signatures_from_session_event(event);
    serde_json::to_writer(&mut *stdout, &event).context("failed to serialize session event")?;
    writeln!(stdout).context("failed to write stdout")?;
    stdout.flush().context("failed to flush stdout")
}

fn write_json_result(stdout: &mut impl Write, result: &AssistantMessage) -> anyhow::Result<()> {
    let result = strip_signatures_from_assistant_message(result);
    serde_json::to_writer(&mut *stdout, &result).context("failed to serialize assistant result")?;
    writeln!(stdout).context("failed to write stdout")?;
    stdout.flush().context("failed to flush stdout")
}

fn init_logging() -> anyhow::Result<()> {
    let filter = match env::var(EnvFilter::DEFAULT_ENV) {
        Ok(value) => EnvFilter::try_new(value).context("invalid RUST_LOG filter")?,
        Err(env::VarError::NotPresent) => EnvFilter::try_new("off")?,
        Err(env::VarError::NotUnicode(_)) => anyhow::bail!("invalid utf-8 in RUST_LOG"),
    };
    let subscriber = tracing_subscriber::registry().with(filter).with(
        fmt::layer()
            .with_writer(std::io::stderr)
            .with_target(true)
            .compact(),
    );
    tracing::subscriber::set_global_default(subscriber).context("failed to initialize logging")
}
