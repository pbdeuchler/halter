// pattern: Imperative Shell

mod openai_oauth;
mod openai_oauth_core;
mod run_output;

use std::env;
use std::fs::File;
use std::io::{self, LineWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

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
    init_logging(trace, to_file)?;
    debug!(config_path = %cli.config.display(), command = ?cli.command, "parsed cli arguments");

    match cli.command {
        Commands::Init => init_config(&cli.config, output.as_mut()).await,
        Commands::Chat => chat(&cli.config, output.as_mut()).await,
        Commands::Run {
            task,
            prompt_file,
            output: run_output,
        } => {
            let task = read_run_prompt(task, prompt_file).await?;
            run_once(&cli.config, &task, run_output.mode(), output.as_mut()).await
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
    let session = harness.new_session(SessionInit::default()).await?;
    let result = tokio::select! {
        biased;
        _ = tokio::signal::ctrl_c() => {
            info!("ctrl-c received, draining runtime before exit");
            let _ = session.shutdown("interrupted").await;
            let report = harness.shutdown(SHUTDOWN_DRAIN).await;
            info!(
                drained = report.turns_drained,
                aborted = report.turns_aborted,
                timed_out = report.timed_out,
                "runtime drained on signal"
            );
            return Err(anyhow::anyhow!("interrupted by signal"));
        }
        result = run_once_body(&session, task, output_mode, output) => result,
    };
    let session_shutdown = session.shutdown("run_complete").await;
    let _ = harness.shutdown(SHUTDOWN_DRAIN).await;
    result?;
    session_shutdown
}

async fn run_once_body(
    session: &HalterSession,
    task: &str,
    output_mode: RunOutputMode,
    output: &mut dyn Write,
) -> anyhow::Result<()> {
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

    let result = tokio::select! {
        biased;
        _ = tokio::signal::ctrl_c() => {
            info!("ctrl-c received, draining runtime before exit");
            let _ = session.shutdown("interrupted").await;
            Err(anyhow::anyhow!("interrupted by signal"))
        }
        result = chat_body(&session, output) => result,
    };
    let session_shutdown = session.shutdown("chat_complete").await;
    let report = harness.shutdown(SHUTDOWN_DRAIN).await;
    info!(
        drained = report.turns_drained,
        aborted = report.turns_aborted,
        timed_out = report.timed_out,
        "runtime drained on chat exit"
    );
    result?;
    session_shutdown
}

async fn chat_body(session: &HalterSession, output: &mut dyn Write) -> anyhow::Result<()> {
    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();

    write_output_line(output, "halter chat; press ctrl-d to exit")?;
    while let Some(line) = lines.next_line().await.context("failed to read stdin")? {
        if line.trim().is_empty() {
            continue;
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

/// Third-party `tracing` targets that emit one DEBUG line per shell token,
/// HTTP connection, or pool event. Promoting them to WARN keeps the harness's
/// own DEBUG output readable when users opt into `RUST_LOG=debug`. Listed in
/// the directive string *before* the user's filter so an explicit per-target
/// user directive (e.g. `RUST_LOG=hyper=trace`) wins over the suppression
/// (per `EnvFilter` last-match-wins precedence), while the suppression still
/// overrides a global level like `debug` or `trace` for these noisy targets.
const NOISY_TARGET_SUPPRESSIONS: &str = "tokenize=warn,parse=warn,expansion=warn,commands=warn,pattern=warn,\
     completion=warn,jobs=warn,unimplemented=warn,\
     hyper_util=warn,hyper=warn,reqwest=warn,h2=warn,rustls=warn";

/// Compose the `EnvFilter` directive string so that user directives come last.
/// Per `EnvFilter` precedence, the last matching directive wins; putting user
/// directives after the suppressions lets explicit `RUST_LOG=hyper=trace`
/// overrides take effect while still quieting noisy targets for `RUST_LOG=debug`.
fn compose_logging_filter(user_directives: Option<&str>) -> String {
    let user = user_directives.map(str::trim).unwrap_or("");
    if user.is_empty() {
        format!("{NOISY_TARGET_SUPPRESSIONS},warn")
    } else {
        format!("{NOISY_TARGET_SUPPRESSIONS},{user}")
    }
}

fn init_logging(writer: TraceWriter, json: bool) -> anyhow::Result<()> {
    let user_directives = match env::var(EnvFilter::DEFAULT_ENV) {
        Ok(value) => value,
        Err(env::VarError::NotPresent) => "warn".to_owned(),
        Err(env::VarError::NotUnicode(_)) => anyhow::bail!("invalid utf-8 in RUST_LOG"),
    };
    // User directives must come last so they win over suppressions.
    let composed = compose_logging_filter(Some(&user_directives));
    let filter = EnvFilter::try_new(&composed).context("invalid RUST_LOG filter")?;
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

    // --- logging composition tests ---

    macro_rules! assert_target_level {
        ($composed:expr, $target:literal, $level:path, $expected:expr) => {{
            let filter = EnvFilter::try_new(&$composed).expect("valid filter");
            let subscriber = tracing_subscriber::registry().with(filter);
            let mut enabled = false;
            tracing::subscriber::with_default(subscriber, || {
                enabled = tracing::enabled!(target: $target, $level);
            });
            assert_eq!(
                enabled, $expected,
                "expected target={} level={:?} to be {}",
                $target, $level, $expected
            );
        }};
    }

    #[test]
    fn compose_filter_uses_warn_fallback_when_unset() {
        let composed = compose_logging_filter(None);
        assert!(composed.starts_with(NOISY_TARGET_SUPPRESSIONS));
        assert!(composed.ends_with(",warn"));
        EnvFilter::try_new(&composed).expect("valid filter");
    }

    #[test]
    fn compose_filter_parses_whitespace_only_rust_log() {
        let composed = compose_logging_filter(Some("   "));
        assert_eq!(composed, compose_logging_filter(None));
        EnvFilter::try_new(&composed).expect("valid filter");
    }

    #[test]
    fn compose_filter_honors_global_level() {
        let composed = compose_logging_filter(Some("info"));
        EnvFilter::try_new(&composed).expect("valid filter");
        assert_target_level!(composed, "halter", tracing::Level::INFO, true);
        assert_target_level!(composed, "halter", tracing::Level::DEBUG, false);
        assert_target_level!(composed, "tokenize", tracing::Level::DEBUG, false);
        assert_target_level!(composed, "hyper", tracing::Level::DEBUG, false);
    }

    #[test]
    fn compose_filter_honors_multi_directive_user_filter() {
        let composed = compose_logging_filter(Some("debug,halter=trace"));
        EnvFilter::try_new(&composed).expect("valid filter");
        assert_target_level!(composed, "halter", tracing::Level::TRACE, true);
        assert_target_level!(composed, "some_crate", tracing::Level::DEBUG, true);
        assert_target_level!(composed, "tokenize", tracing::Level::DEBUG, false);
    }

    #[test]
    fn regression_99_explicit_per_target_wins_over_suppression() {
        let composed = compose_logging_filter(Some("hyper=trace"));
        assert_target_level!(composed, "hyper", tracing::Level::TRACE, true);
    }

    #[test]
    fn regression_99_explicit_target_overrides_global_warn_and_suppression() {
        let composed = compose_logging_filter(Some("warn,hyper=trace"));
        assert_target_level!(composed, "some_other_crate", tracing::Level::INFO, false);
        assert_target_level!(composed, "hyper", tracing::Level::TRACE, true);
    }

    #[test]
    fn regression_99_multiple_suppressed_targets_can_be_overridden() {
        let composed = compose_logging_filter(Some("reqwest=debug,h2=info"));
        assert_target_level!(composed, "reqwest", tracing::Level::DEBUG, true);
        assert_target_level!(composed, "h2", tracing::Level::INFO, true);
        assert_target_level!(composed, "hyper", tracing::Level::DEBUG, false);
    }

    #[test]
    fn compose_filter_global_debug_still_suppresses_noisy_targets() {
        let composed = compose_logging_filter(Some("debug"));
        assert_target_level!(composed, "halter", tracing::Level::DEBUG, true);
        assert_target_level!(composed, "tokenize", tracing::Level::DEBUG, false);
        assert_target_level!(composed, "hyper", tracing::Level::DEBUG, false);
        assert_target_level!(composed, "reqwest", tracing::Level::DEBUG, false);
    }

    #[test]
    fn compose_filter_rejects_invalid_user_directive() {
        let composed = compose_logging_filter(Some("hyper=invalid_level"));
        EnvFilter::try_new(&composed).expect_err("invalid user directive should fail");
    }
}
