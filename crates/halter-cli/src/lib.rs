// pattern: Imperative Shell

mod openai_oauth;
mod openai_oauth_core;
mod run_output;

use std::fs::File;
use std::io::{self, LineWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Context;
use clap::{Parser, Subcommand};
use futures::StreamExt;
use halter::prelude::*;
use halter::telemetry::{LogFormat, TelemetryConfig, tracing_subscriber::fmt::MakeWriter};
use halter_config::{export_json_schema, generate_starter_config, load_path};
use halter_protocol::{AssistantMessage, SessionEvent, SessionEventPayload};
use run_output::{
    ForegroundRun, RunOutputArgs, RunOutputMode, strip_signatures_from_assistant_message,
    strip_signatures_from_session_event,
};
use tokio::io::{AsyncBufReadExt, BufReader};
use tracing::{debug, info};

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
        #[arg(
            long,
            value_name = "PROMPT_FILE",
            conflicts_with = "task",
            help = "Read the run prompt from a file instead of a command-line string"
        )]
        prompt_file: Option<PathBuf>,
        #[arg(
            value_name = "TASK",
            required_unless_present = "prompt_file",
            conflicts_with = "prompt_file"
        )]
        task: Option<String>,
    },
    Resources,
    Validate,
    Auth {
        #[command(subcommand)]
        command: AuthCommands,
    },
    Config {
        #[command(subcommand)]
        command: ConfigCommands,
    },
}

impl Commands {
    /// Variant name for logging. Deliberately excludes payload fields —
    /// `run` carries the user's task text, which must not reach the logs.
    const fn name(&self) -> &'static str {
        match self {
            Self::Init => "init",
            Self::Chat => "chat",
            Self::Run { .. } => "run",
            Self::Resources => "resources",
            Self::Validate => "validate",
            Self::Auth { .. } => "auth",
            Self::Config { .. } => "config",
        }
    }
}

#[derive(Debug, Subcommand)]
enum AuthCommands {
    #[command(name = "openai-oauth")]
    OpenAiOauth(openai_oauth::OpenAiOAuthCommand),
}

#[derive(Debug, Subcommand)]
enum ConfigCommands {
    Schema,
}

pub async fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let to_file = cli.output_file.is_some();
    let OutputHandles { mut output, trace } = open_output_handles(cli.output_file.as_deref())?;
    let otel_endpoint = std::env::var("OTEL_EXPORTER_OTLP_ENDPOINT").ok();
    let (otel_layer, otel_guard) = maybe_init_otel(otel_endpoint.as_deref())?;
    init_logging(trace, to_file, otel_layer)?;
    debug!(config_path = %cli.config.display(), command = cli.command.name(), "parsed cli arguments");

    let result = match cli.command {
        Commands::Init => init_config(&cli.config, output.as_mut()).await,
        Commands::Chat => chat(&cli.config, output.as_mut()).await,
        Commands::Run {
            task,
            prompt_file,
            output: run_output,
        } => {
            async {
                let task = read_run_prompt(task, prompt_file).await?;
                run_once(&cli.config, &task, run_output.mode(), output.as_mut()).await
            }
            .await
        }
        Commands::Resources => show_resources(&cli.config, output.as_mut()).await,
        Commands::Validate => validate(&cli.config, output.as_mut()).await,
        Commands::Auth {
            command: AuthCommands::OpenAiOauth(command),
        } => openai_oauth::run(command, output.as_mut()).await,
        Commands::Config {
            command: ConfigCommands::Schema,
        } => {
            write_output_line(output.as_mut(), export_json_schema()?)?;
            Ok(())
        }
    };

    result?;
    output.flush().context("failed to flush output")?;
    // Explicit shutdown (rather than relying only on `OtelGuard`'s `Drop`)
    // makes the final flush deterministic relative to process exit.
    shutdown_otel(otel_guard)
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

async fn read_run_prompt(
    task: Option<String>,
    prompt_file: Option<PathBuf>,
) -> anyhow::Result<String> {
    match (task, prompt_file) {
        (Some(task), None) => Ok(task),
        (None, Some(path)) => tokio::fs::read_to_string(&path)
            .await
            .with_context(|| format!("failed to read prompt file {}", path.display())),
        (None, None) => {
            anyhow::bail!(
                "failed to resolve run prompt: pass <TASK> or --prompt-file <PROMPT_FILE>"
            )
        }
        (Some(_), Some(_)) => {
            anyhow::bail!(
                "failed to resolve run prompt: pass either <TASK> or --prompt-file <PROMPT_FILE>, not both"
            )
        }
    }
}

/// Bound on how long the runtime gets to drain in-flight turns after a
/// SIGINT/SIGTERM. The CLI sets this; embedders can pick their own.
const SHUTDOWN_DRAIN: Duration = Duration::from_secs(10);

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
    let (session, mut events) = harness.new_session(SessionInit::default()).await?;
    let (result, reason) = tokio::select! {
        biased;
        _ = tokio::signal::ctrl_c() => {
            info!("ctrl-c received, draining runtime before exit");
            (Err(anyhow::anyhow!("interrupted by signal")), "interrupted")
        }
        result = run_once_body(&session, &mut events, task, output_mode, output) => (result, "run_complete"),
    };
    drain_then_end_session(&harness, &session, result, reason).await
}

async fn run_once_body(
    session: &HalterSession,
    events: &mut SessionEventStream,
    task: &str,
    output_mode: RunOutputMode,
    output: &mut dyn Write,
) -> anyhow::Result<()> {
    let submission = session.submit(Message::user(task)).await?;
    let mut foreground = ForegroundRun::new(submission.message_id.clone());
    while let Some(event) = events.next().await {
        let event = event?;
        if output_mode == RunOutputMode::StreamingJson {
            write_json_event(output, &event)?;
        }
        if event.session_id != *session.id() || event.sequence() < submission.sequence {
            continue;
        }
        if foreground
            .observe(&event.payload)
            .map_err(anyhow::Error::msg)?
        {
            if output_mode == RunOutputMode::JsonResult {
                let result = foreground.final_result().map_err(anyhow::Error::msg)?;
                write_json_result(output, result)?;
            }
            return Ok(());
        }
    }
    anyhow::bail!("session closed before foreground execution stopped")
}

async fn chat(path: &Path, output: &mut dyn Write) -> anyhow::Result<()> {
    info!(path = %path.display(), "starting chat session");
    let harness = Halter::from_config_file(path).await?;
    let (session, mut events) = harness.new_session(SessionInit::default()).await?;

    let (result, reason) = tokio::select! {
        biased;
        _ = tokio::signal::ctrl_c() => {
            info!("ctrl-c received, draining runtime before exit");
            (Err(anyhow::anyhow!("interrupted by signal")), "interrupted")
        }
        result = chat_body(&session, &mut events, output) => (result, "chat_complete"),
    };
    drain_then_end_session(&harness, &session, result, reason).await
}

/// Close the session driver before stopping its runtime.
async fn drain_then_end_session(
    harness: &Halter,
    session: &HalterSession,
    result: anyhow::Result<()>,
    reason: &str,
) -> anyhow::Result<()> {
    let session_shutdown = session.shutdown(Some(SHUTDOWN_DRAIN)).await;
    let report = harness.shutdown(SHUTDOWN_DRAIN).await;
    info!(
        drained = report.turns_drained,
        aborted = report.turns_aborted,
        timed_out = report.timed_out,
        reason,
        "runtime drained"
    );
    result?;
    session_shutdown.map_err(Into::into)
}

async fn chat_body(
    session: &HalterSession,
    events: &mut SessionEventStream,
    output: &mut dyn Write,
) -> anyhow::Result<()> {
    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();

    write_output_line(output, "halter chat; press ctrl-d to exit")?;
    while let Some(line) = lines.next_line().await.context("failed to read stdin")? {
        if line.trim().is_empty() {
            continue;
        }

        let submission = session.submit(Message::user(line)).await?;
        let mut foreground = ForegroundRun::new(submission.message_id.clone());
        let mut completed = false;
        while let Some(event) = events.next().await {
            let event = event?;
            if event.session_id != *session.id() || event.sequence() < submission.sequence {
                continue;
            }
            if foreground
                .observe(&event.payload)
                .map_err(anyhow::Error::msg)?
            {
                completed = true;
                writeln!(output).context("failed to write output")?;
                output.flush().context("failed to flush output")?;
                break;
            }
            match event.payload {
                SessionEventPayload::DeltaItem { delta } => {
                    write!(output, "{}", delta.text).context("failed to write output")?;
                    output.flush().context("failed to flush output")?;
                }
                SessionEventPayload::ToolOutput { chunk, .. } => {
                    write!(output, "{}", chunk).context("failed to write output")?;
                    output.flush().context("failed to flush output")?;
                }
                _ => {}
            }
        }
        if !completed {
            anyhow::bail!("session closed before execution completed");
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
}

enum TraceWriteHandle {
    Stderr(io::Stderr),
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
        let mut writer = self
            .inner
            .lock()
            .map_err(|_| io::Error::other("shared output writer mutex poisoned"))?;
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
        }
    }

    fn write_all(&mut self, buf: &[u8]) -> io::Result<()> {
        match self {
            Self::Stderr(writer) => writer.write_all(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Self::Stderr(writer) => writer.flush(),
        }
    }
}

impl<'a> MakeWriter<'a> for TraceWriter {
    type Writer = TraceWriteHandle;

    fn make_writer(&'a self) -> Self::Writer {
        match self {
            Self::Stderr => TraceWriteHandle::Stderr(io::stderr()),
        }
    }
}

fn open_output_handles(path: Option<&Path>) -> anyhow::Result<OutputHandles> {
    match path {
        Some(path) => {
            let writer = SharedFileWriter::create(path)?;
            Ok(OutputHandles {
                output: Box::new(writer),
                trace: TraceWriter::Stderr,
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

#[cfg(feature = "otel")]
fn init_logging(
    writer: TraceWriter,
    json: bool,
    otel: Option<halter::telemetry::otel::OtelLayers>,
) -> anyhow::Result<()> {
    let format = if json {
        LogFormat::Json
    } else {
        LogFormat::Compact
    };
    let config = TelemetryConfig::new()
        .with_writer(writer)
        .with_format(format);
    match otel {
        // `try_init_with_otel` composes the console formatter and the OTel
        // layers as siblings, each under its own per-layer filter, so OTel
        // sees info-level halter spans even when the console (and
        // `RUST_LOG`) stays at the default `warn`. Plain `try_init_with`
        // would nest both under one shared filter and silently export
        // nothing by default — see `halter::telemetry::otel`'s module docs.
        Some(layers) => config.try_init_with_otel(layers),
        None => config.try_init(),
    }
}

#[cfg(not(feature = "otel"))]
fn init_logging(writer: TraceWriter, json: bool, _otel: Option<()>) -> anyhow::Result<()> {
    let format = if json {
        LogFormat::Json
    } else {
        LogFormat::Compact
    };
    TelemetryConfig::new()
        .with_writer(writer)
        .with_format(format)
        .try_init()
}

/// Builds the OTLP trace/metric layers and their shutdown guard, but only
/// when the `otel` feature is compiled in *and* `endpoint` names a non-empty
/// OTLP endpoint (normally read from `OTEL_EXPORTER_OTLP_ENDPOINT`). Takes
/// the endpoint as a parameter rather than reading the env var itself so
/// this is testable without mutating process-wide env state.
///
/// The CLI never exports telemetry by default: without `--features otel`,
/// or with the feature but no endpoint configured, this returns `(None,
/// None)` and no OTel exporter, provider, or layer is ever constructed.
#[cfg(feature = "otel")]
fn maybe_init_otel(
    endpoint: Option<&str>,
) -> anyhow::Result<(
    Option<halter::telemetry::otel::OtelLayers>,
    Option<halter::telemetry::otel::OtelGuard>,
)> {
    match endpoint {
        Some(endpoint) if !endpoint.is_empty() => {
            let (layers, guard) = halter::telemetry::otel::OtelConfig::new().build()?;
            Ok((Some(layers), Some(guard)))
        }
        _ => Ok((None, None)),
    }
}

#[cfg(not(feature = "otel"))]
fn maybe_init_otel(_endpoint: Option<&str>) -> anyhow::Result<(Option<()>, Option<()>)> {
    Ok((None, None))
}

#[cfg(feature = "otel")]
fn shutdown_otel(guard: Option<halter::telemetry::otel::OtelGuard>) -> anyhow::Result<()> {
    guard.map_or(Ok(()), halter::telemetry::otel::OtelGuard::shutdown)
}

#[cfg(not(feature = "otel"))]
fn shutdown_otel(_guard: Option<()>) -> anyhow::Result<()> {
    Ok(())
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

    /// `maybe_init_otel` must never construct an OTel exporter/provider/layer
    /// when no endpoint is configured, with or without the `otel` feature.
    /// Takes the endpoint as a parameter (rather than reading
    /// `OTEL_EXPORTER_OTLP_ENDPOINT` itself) specifically so this doesn't
    /// need to mutate process-wide env state.
    #[test]
    fn maybe_init_otel_is_noop_without_an_endpoint() {
        for endpoint in [None, Some("")] {
            let (layer, guard) = maybe_init_otel(endpoint).expect("must not error");
            assert!(layer.is_none(), "no layer without an endpoint");
            assert!(guard.is_none(), "no guard without an endpoint");
        }
    }

    #[test]
    fn cli_accepts_run_prompt_file() {
        let cli =
            Cli::try_parse_from(["halter", "run", "--prompt-file", "prompt.md"]).expect("parse");

        match cli.command {
            Commands::Run {
                prompt_file, task, ..
            } => {
                assert_eq!(prompt_file, Some(PathBuf::from("prompt.md")));
                assert_eq!(task, None);
            }
            _ => panic!("expected run command"),
        }
    }

    #[test]
    fn cli_rejects_missing_run_prompt_source() {
        let error = Cli::try_parse_from(["halter", "run"])
            .expect_err("run should require a task or prompt file");

        assert_eq!(
            error.kind(),
            clap::error::ErrorKind::MissingRequiredArgument
        );
    }

    #[test]
    fn cli_rejects_run_task_and_prompt_file() {
        let error = Cli::try_parse_from([
            "halter",
            "run",
            "--prompt-file",
            "prompt.md",
            "command-line task",
        ])
        .expect_err("run should reject multiple prompt sources");

        assert_eq!(error.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    #[test]
    fn cli_accepts_openai_oauth_auth_command() {
        let cli = Cli::try_parse_from([
            "halter",
            "auth",
            "openai-oauth",
            "--no-open-browser",
            "--format",
            "env",
        ])
        .expect("parse");

        assert!(matches!(
            cli.command,
            Commands::Auth {
                command: AuthCommands::OpenAiOauth(_)
            }
        ));
    }

    #[test]
    fn cli_rejects_conflicting_openai_oauth_api_key_exchange_flags() {
        let error = Cli::try_parse_from([
            "halter",
            "auth",
            "openai-oauth",
            "--skip-api-key-exchange",
            "--require-api-key-exchange",
        ])
        .expect_err("conflicting flags should fail");

        assert_eq!(error.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    #[tokio::test]
    async fn read_run_prompt_reads_prompt_file() {
        let temp = tempdir().expect("tempdir");
        let path = temp.path().join("prompt.md");
        tokio::fs::write(&path, "prompt from file\n")
            .await
            .expect("write prompt");

        let prompt = read_run_prompt(None, Some(path))
            .await
            .expect("read prompt");

        assert_eq!(prompt, "prompt from file\n");
    }

    #[test]
    fn open_output_handles_redirects_only_command_output_to_file() {
        let temp = tempdir().expect("tempdir");
        let path = temp.path().join("output.txt");
        let OutputHandles { mut output, trace } =
            open_output_handles(Some(&path)).expect("open output");

        write_output_line(output.as_mut(), "hello world").expect("write output");
        output.flush().expect("flush output");
        assert!(matches!(trace, TraceWriter::Stderr));

        let contents = std::fs::read_to_string(&path).expect("read output");
        assert_eq!(contents, "hello world\n");
    }
}
