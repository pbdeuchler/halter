// pattern: Imperative Shell

mod run_output;

use std::env;
use std::fs::File;
use std::io::{self, LineWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

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
use tracing_subscriber::{
    EnvFilter,
    fmt::{self, MakeWriter},
    prelude::*,
};

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
    let cli = Cli::parse();
    let to_file = cli.output_file.is_some();
    let OutputHandles { mut output, trace } = open_output_handles(cli.output_file.as_deref())?;
    init_logging(trace, to_file)?;
    debug!(config_path = %cli.config.display(), command = ?cli.command, "parsed cli arguments");

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
    let mut line = serde_json::to_vec(&event).context("failed to serialize session event")?;
    line.push(b'\n');
    output.write_all(&line).context("failed to write output")?;
    output.flush().context("failed to flush output")
}

fn write_json_result(output: &mut dyn Write, result: &AssistantMessage) -> anyhow::Result<()> {
    let result = strip_signatures_from_assistant_message(result);
    let mut line = serde_json::to_vec(&result).context("failed to serialize assistant result")?;
    line.push(b'\n');
    output.write_all(&line).context("failed to write output")?;
    output.flush().context("failed to flush output")
}

struct OutputHandles {
    output: Box<dyn Write>,
    trace: TraceWriter,
}

#[derive(Clone)]
enum TraceWriter {
    Stderr,
    SharedFile(SharedFileWriter),
}

enum TraceWriteHandle {
    Stderr(io::Stderr),
    SharedFile(SharedFileWriter),
}

#[derive(Clone)]
struct SharedFileWriter {
    inner: Arc<Mutex<LineWriter<File>>>,
}

impl SharedFileWriter {
    fn create(path: &Path) -> anyhow::Result<Self> {
        let file = File::create(path)
            .with_context(|| format!("failed to create output file {}", path.display()))?;
        Ok(Self {
            inner: Arc::new(Mutex::new(LineWriter::new(file))),
        })
    }

    fn with_locked_writer<T>(
        &self,
        f: impl FnOnce(&mut LineWriter<File>) -> io::Result<T>,
    ) -> io::Result<T> {
        let mut writer = self.inner.lock().map_err(|_| {
            io::Error::new(io::ErrorKind::Other, "shared output writer mutex poisoned")
        })?;
        f(&mut writer)
    }
}

impl Write for SharedFileWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.with_locked_writer(|writer| writer.write(buf))
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        self.with_locked_writer(|writer| writer.write_all(buf))
    }

    fn flush(&mut self) -> io::Result<()> {
        self.with_locked_writer(|writer| writer.flush())
    }
}

impl Write for TraceWriteHandle {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Self::Stderr(writer) => writer.write(buf),
            Self::SharedFile(writer) => writer.write(buf),
        }
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        match self {
            Self::Stderr(writer) => writer.write_all(buf),
            Self::SharedFile(writer) => writer.write_all(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Self::Stderr(writer) => writer.flush(),
            Self::SharedFile(writer) => writer.flush(),
        }
    }
}

impl<'a> MakeWriter<'a> for TraceWriter {
    type Writer = TraceWriteHandle;

    fn make_writer(&'a self) -> Self::Writer {
        match self {
            Self::Stderr => TraceWriteHandle::Stderr(io::stderr()),
            Self::SharedFile(writer) => TraceWriteHandle::SharedFile(writer.clone()),
        }
    }
}

fn open_output_handles(path: Option<&Path>) -> anyhow::Result<OutputHandles> {
    match path {
        Some(path) => {
            let writer = SharedFileWriter::create(path)?;
            Ok(OutputHandles {
                output: Box::new(writer.clone()),
                trace: TraceWriter::SharedFile(writer),
            })
        }
        None => Ok(OutputHandles {
            output: Box::new(io::stdout()),
            trace: TraceWriter::Stderr,
        }),
    }
}

fn write_output_line(output: &mut dyn Write, line: impl std::fmt::Display) -> anyhow::Result<()> {
    writeln!(output, "{line}").context("failed to write output")
}

fn init_logging(writer: TraceWriter, json: bool) -> anyhow::Result<()> {
    let filter = match env::var(EnvFilter::DEFAULT_ENV) {
        Ok(value) => EnvFilter::try_new(value).context("invalid RUST_LOG filter")?,
        Err(env::VarError::NotPresent) => EnvFilter::try_new("off")?,
        Err(env::VarError::NotUnicode(_)) => anyhow::bail!("invalid utf-8 in RUST_LOG"),
    };
    let base = fmt::layer().with_writer(writer).with_target(true);
    let registry = tracing_subscriber::registry().with(filter);
    if json {
        tracing::subscriber::set_global_default(registry.with(base.json()))
    } else {
        tracing::subscriber::set_global_default(registry.with(base.compact()))
    }
    .context("failed to initialize logging")
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
    fn open_output_handles_redirect_output_and_traces_to_same_file() {
        let temp = tempdir().expect("tempdir");
        let path = temp.path().join("output.txt");
        let OutputHandles { mut output, trace } =
            open_output_handles(Some(&path)).expect("open output");

        write_output_line(output.as_mut(), "hello world").expect("write output");
        let mut trace_output = trace.make_writer();
        trace_output
            .write_all(b"trace line\n")
            .expect("write trace output");
        output.flush().expect("flush output");
        trace_output.flush().expect("flush trace output");

        let contents = std::fs::read_to_string(&path).expect("read output");
        assert_eq!(contents, "hello world\ntrace line\n");
    }
}
