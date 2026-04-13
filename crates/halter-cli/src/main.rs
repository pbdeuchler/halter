// pattern: Imperative Shell

mod run_output;

use std::env;
use std::fs::File;
use std::io::{self, BufWriter, Write};
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
    #[arg(
        long,
        global = true,
        help = "Write CLI output to a file instead of standard output"
    )]
    output_file: Option<PathBuf>,
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
    let mut output = open_output_writer(cli.output_file.as_deref())?;

    match cli.command {
        Commands::Init => init_config(&cli.config, output.as_mut()).await,
        Commands::Chat => chat(&cli.config, output.as_mut()).await,
        Commands::Run {
            task,
            output: run_output,
        } => run_once(&cli.config, &task, run_output.mode(), output.as_mut()).await,
        Commands::Resources => show_resources(&cli.config, output.as_mut()).await,
        Commands::Validate => validate(&cli.config, output.as_mut()).await,
        Commands::Config {
            command: ConfigCommands::Schema,
        } => {
            write_output_line(output.as_mut(), export_json_schema()?)?;
            Ok(())
        }
    }?;

    output.flush().context("failed to flush output")
}

async fn init_config(path: &Path, output: &mut dyn Write) -> anyhow::Result<()> {
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
    write_output_line(output, format!("wrote {}", path.display()))
}

async fn validate(path: &Path, output: &mut dyn Write) -> anyhow::Result<()> {
    info!(path = %path.display(), "validating config");
    load_path(path).await?;
    write_output_line(output, "config valid")
}

async fn show_resources(path: &Path, output: &mut dyn Write) -> anyhow::Result<()> {
    info!(path = %path.display(), "compiling resources");
    let config = load_path(path).await?;
    let resources = ResourceCompiler::from_config(&config).compile().await?;
    write_output_line(
        output,
        format!("revision: {}", resources.snapshot.revision.0),
    )?;
    write_output_line(
        output,
        format!("skills: {}", resources.snapshot.skills.len()),
    )?;
    write_output_line(
        output,
        format!("agents: {}", resources.snapshot.agents.len()),
    )?;
    write_output_line(
        output,
        format!("plugins: {}", resources.snapshot.plugins.len()),
    )
}

async fn run_once(
    path: &Path,
    task: &str,
    output_mode: RunOutputMode,
    output: &mut dyn Write,
) -> anyhow::Result<()> {
    info!(
        path = %path.display(),
        output_mode = ?output_mode,
        task_len = task.len(),
        "running single turn"
    );
    let harness = Halter::from_config_file(path).await?;
    let session = harness.new_session(SessionInit::default()).await?;
    let mut events = session.submit_turn(Turn::user(task)).await?;

    match output_mode {
        RunOutputMode::StreamingJson => {
            while let Some(event) = events.next().await {
                write_json_event(output, &event?)?;
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
                    write_json_result(output, result)?;
                    return Ok(());
                }
            }
            anyhow::bail!("failed to receive final assistant result")
        }
    }
}

async fn chat(path: &Path, output: &mut dyn Write) -> anyhow::Result<()> {
    info!(path = %path.display(), "starting chat session");
    let harness = Halter::from_config_file(path).await?;
    let session = harness.new_session(SessionInit::default()).await?;
    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();

    write_output_line(
        output,
        "halter chat; submit an empty line or press ctrl-d to exit",
    )?;
    while let Some(line) = lines.next_line().await.context("failed to read stdin")? {
        if line.trim().is_empty() {
            break;
        }

        let mut events = session.submit_turn(Turn::user(line)).await?;
        while let Some(event) = events.next().await {
            match event?.payload {
                SessionEventPayload::DeltaItem { delta } => {
                    write!(output, "{}", delta.text).context("failed to write output")?;
                    output.flush().context("failed to flush output")?;
                }
                SessionEventPayload::ToolOutput { chunk, .. } => {
                    write!(output, "{}", chunk).context("failed to write output")?;
                    output.flush().context("failed to flush output")?;
                }
                SessionEventPayload::TurnCompleted { .. } => {
                    writeln!(output).context("failed to write output")?;
                    output.flush().context("failed to flush output")?;
                    break;
                }
                SessionEventPayload::TurnFailed { error, .. } => anyhow::bail!(error),
                _ => {}
            }
        }
    }
    Ok(())
}

fn write_json_event(output: &mut dyn Write, event: &SessionEvent) -> anyhow::Result<()> {
    let event = strip_signatures_from_session_event(event);
    serde_json::to_writer(&mut *output, &event).context("failed to serialize session event")?;
    writeln!(output).context("failed to write output")?;
    output.flush().context("failed to flush output")
}

fn write_json_result(output: &mut dyn Write, result: &AssistantMessage) -> anyhow::Result<()> {
    let result = strip_signatures_from_assistant_message(result);
    serde_json::to_writer(&mut *output, &result).context("failed to serialize assistant result")?;
    writeln!(output).context("failed to write output")?;
    output.flush().context("failed to flush output")
}

fn open_output_writer(path: Option<&Path>) -> anyhow::Result<Box<dyn Write>> {
    match path {
        Some(path) => {
            let file = File::create(path)
                .with_context(|| format!("failed to create output file {}", path.display()))?;
            Ok(Box::new(BufWriter::new(file)))
        }
        None => Ok(Box::new(io::stdout())),
    }
}

fn write_output_line(output: &mut dyn Write, line: impl std::fmt::Display) -> anyhow::Result<()> {
    writeln!(output, "{line}").context("failed to write output")
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn cli_accepts_output_file_before_subcommand() {
        let cli = Cli::try_parse_from(["halter", "--output-file", "out.jsonl", "run", "task"])
            .expect("parse");

        assert_eq!(cli.output_file, Some(PathBuf::from("out.jsonl")));
        assert!(matches!(cli.command, Commands::Run { .. }));
    }

    #[test]
    fn cli_accepts_output_file_after_subcommand() {
        let cli = Cli::try_parse_from(["halter", "run", "--output-file", "out.jsonl", "task"])
            .expect("parse");

        assert_eq!(cli.output_file, Some(PathBuf::from("out.jsonl")));
        assert!(matches!(cli.command, Commands::Run { .. }));
    }

    #[test]
    fn open_output_writer_redirects_output_to_file() {
        let temp = tempdir().expect("tempdir");
        let path = temp.path().join("output.txt");
        let mut output = open_output_writer(Some(&path)).expect("open output");

        write_output_line(output.as_mut(), "hello world").expect("write output");
        output.flush().expect("flush output");

        let contents = std::fs::read_to_string(&path).expect("read output");
        assert_eq!(contents, "hello world\n");
    }
}
