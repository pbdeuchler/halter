// pattern: Imperative Shell

use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use halter_protocol::{ToolCapabilities, ToolConcurrency, ToolName, ToolResult, ToolSpec};
use parking_lot::Mutex;
use portable_pty::{Child, ChildKiller, CommandBuilder, PtySize, native_pty_system};
use serde_json::{Value, json};

use crate::{Tool, ToolContext, ToolRuntimeEvent};

use super::common::{
    ToolScope, ensure_not_cancelled, optional_string, optional_u64, parse_env_map, required_string,
    resolve_path,
};
use super::process::{kill_process_group, kill_tree, process_group_id};

const TERM_SIGNAL: i32 = 15;
const KILL_SIGNAL: i32 = 9;
const CONTROL_POLL_INTERVAL: Duration = Duration::from_millis(25);

// Minimum env vars passed through to spawned PTYs. Keeps the surface small
// while preserving locale and shell ergonomics. Anything more should come
// from explicit caller-supplied env in `PtyConfig::env`.
const PTY_ENV_ALLOWLIST: &[&str] = &[
    "PATH", "HOME", "USER", "LOGNAME", "TERM", "LANG", "LC_ALL", "LC_CTYPE", "TZ", "PWD", "SHELL",
];

/// Args passed to `sh`. We use `-c` (not `-lc`) so the spawned shell does not
/// source login rc files (e.g. `~/.bash_profile`, `/etc/profile`) which can
/// run arbitrary user-controlled startup code that bypasses the policy.
fn pty_shell_args() -> &'static [&'static str] {
    &["-c"]
}

/// Build the env vector that will be set on the spawned PTY after a clear.
/// Order: allowlisted parent vars first, then caller-supplied overrides.
/// Takes the parent environment and caller-supplied overrides explicitly so
/// tests never need to mutate process-wide environment variables.
fn pty_scrubbed_env(
    parent: impl IntoIterator<Item = (std::ffi::OsString, std::ffi::OsString)>,
    overrides: Option<&HashMap<String, String>>,
) -> Vec<(String, std::ffi::OsString)> {
    let mut out: Vec<(String, std::ffi::OsString)> = Vec::new();
    for (key, value) in parent {
        if let Some(key) = key.to_str()
            && PTY_ENV_ALLOWLIST.contains(&key)
        {
            out.push((key.to_owned(), value));
        }
    }
    if let Some(overrides) = overrides {
        for (key, value) in overrides {
            out.push((key.clone(), std::ffi::OsString::from(value)));
        }
    }
    out
}

/// Reads a `u16` pty dimension from `input[key]`, falling back to `default`
/// when unset. Rejects values that do not fit in `u16` instead of silently
/// truncating via `as u16`. (finding L29)
fn checked_u16(input: &Value, key: &str, default: u16) -> anyhow::Result<u16> {
    let Some(raw) = optional_u64(input, key)? else {
        return Ok(default);
    };
    u16::try_from(raw).map_err(|_| {
        anyhow::anyhow!(
            "failed to execute pty tool: '{key}' must fit in u16 (<= {}), got {raw}",
            u16::MAX
        )
    })
}

/// Handle for an active PTY session stored per halter session.
pub struct PtySessionHandle {
    control_tx: mpsc::Sender<ControlMessage>,
    task: tokio::task::JoinHandle<anyhow::Result<()>>,
    generation: Arc<()>,
    termination: PtyTermination,
}

// A separate killer lets shutdown unblock a worker stuck writing PTY input.
// Taking the option ensures native process termination happens only once.
type PtyTermination = Arc<Mutex<PtyTerminationState>>;

#[derive(Default)]
struct PtyTerminationState {
    stopped: bool,
    killer: Option<PtyChildKiller>,
}

struct PtyChildKiller {
    killer: Box<dyn ChildKiller + Send + Sync>,
    child_pid: Option<i32>,
    process_group: Option<i32>,
}

#[derive(Debug)]
/// Built-in tool for interacting with a persistent pseudo-terminal.
pub struct PtyTool;

#[derive(Clone)]
struct PtyConfig {
    command: String,
    cwd: Option<String>,
    env: Option<HashMap<String, String>>,
    cols: u16,
    rows: u16,
    timeout: Option<Duration>,
}

enum ControlMessage {
    Input(String),
    Resize { cols: u16, rows: u16 },
    Kill,
}

enum ReaderEvent {
    Output(String),
    Closed,
}

#[async_trait]
impl Tool for PtyTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("pty"),
            description: "Manage an interactive PTY session".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["start", "write", "resize", "kill"] },
                    "command": { "type": "string" },
                    "cwd": { "type": "string" },
                    "env": { "type": "object", "additionalProperties": { "type": "string" } },
                    "timeout_ms": { "type": "integer", "minimum": 1 },
                    "cols": { "type": "integer", "minimum": 20 },
                    "rows": { "type": "integer", "minimum": 5 },
                    "input": { "type": "string" }
                },
                "required": ["action"],
            }),
            concurrency: ToolConcurrency::Exclusive,
            capabilities: ToolCapabilities {
                mutating: true,
                requires_approval: false,
                cancellable: false,
                long_running: true,
            },
            provider_aliases: Default::default(),
        }
    }

    async fn execute(&self, context: ToolContext, input: Value) -> anyhow::Result<ToolResult> {
        let _scope = ToolScope::new(&context, "pty");
        ensure_not_cancelled(&context.cancel)?;
        context.policy.check_shell_enabled().await?;

        let action = required_string(&input, "action")?;
        let session = context.tool_sessions.pty_session(&context.session_id);

        match action {
            "start" => {
                let config = PtyConfig {
                    command: required_string(&input, "command")?.to_owned(),
                    cwd: optional_string(&input, "cwd").map(ToOwned::to_owned),
                    env: parse_env_map(input.get("env"))?,
                    timeout: optional_u64(&input, "timeout_ms")?.map(Duration::from_millis),
                    cols: checked_u16(&input, "cols", 120)?,
                    rows: checked_u16(&input, "rows", 40)?,
                };
                let mode = context.policy.shell_mode();
                context
                    .policy
                    .check_shell_command_strict(&config.command, mode)
                    .await?;
                let cwd = config
                    .cwd
                    .as_deref()
                    .map(|cwd| resolve_path(&context.working_dir, cwd))
                    .unwrap_or_else(|| context.working_dir.clone());
                let cwd = context.policy.check_read_path(&cwd, 0).await?;
                let config = PtyConfig {
                    cwd: Some(cwd.path().to_string_lossy().into_owned()),
                    ..config
                };
                ensure_not_cancelled(&context.cancel)?;
                start_session(
                    session,
                    config,
                    context.emit.clone(),
                    context.tool_sessions.process_lifetime(&context.session_id),
                )
                .await?;
                Ok(ToolResult::Json {
                    value: json!({ "started": true }),
                })
            }
            "write" => {
                let input = required_string(&input, "input")?.to_owned();
                send_control(&session, ControlMessage::Input(input))?;
                Ok(ToolResult::Json {
                    value: json!({ "ok": true }),
                })
            }
            "resize" => {
                let cols = checked_u16(&input, "cols", 120)?;
                let rows = checked_u16(&input, "rows", 40)?;
                send_control(&session, ControlMessage::Resize { cols, rows })?;
                Ok(ToolResult::Json {
                    value: json!({ "ok": true }),
                })
            }
            "kill" => {
                stop_session(&session).await?;
                Ok(ToolResult::Json {
                    value: json!({ "ok": true }),
                })
            }
            _ => anyhow::bail!("failed to execute pty tool: unknown action '{action}'"),
        }
    }
}

async fn start_session(
    session: Arc<Mutex<Option<PtySessionHandle>>>,
    config: PtyConfig,
    emit: Arc<dyn crate::ToolEventSink>,
    lifetime: tokio_util::sync::CancellationToken,
) -> anyhow::Result<()> {
    let (control_tx, control_rx) = mpsc::channel();
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel();
    let termination = Arc::new(Mutex::new(PtyTerminationState::default()));
    let termination_for_task = termination.clone();
    let session_for_task = Arc::clone(&session);
    let generation = Arc::new(());
    let generation_for_task = generation.clone();
    {
        let mut guard = session.lock();
        anyhow::ensure!(
            !lifetime.is_cancelled(),
            "failed to execute pty tool: session is closed"
        );
        anyhow::ensure!(
            guard.is_none(),
            "failed to execute pty tool: PTY session already active"
        );
        let task = tokio::task::spawn_blocking(move || {
            let result = match prepare_pty(&config, termination_for_task, &lifetime) {
                Ok(state) => {
                    let _ = ready_tx.send(Ok(()));
                    run_pty_loop(state, config.timeout, control_rx, emit)
                }
                Err(error) => {
                    let _ = ready_tx.send(Err(error));
                    Ok(())
                }
            };
            let mut guard = session_for_task.lock();
            if guard
                .as_ref()
                .is_some_and(|handle| Arc::ptr_eq(&handle.generation, &generation_for_task))
            {
                *guard = None;
            }
            result
        });
        *guard = Some(PtySessionHandle {
            control_tx,
            task,
            generation,
            termination,
        });
    }
    ready_rx
        .await
        .map_err(|error| anyhow::anyhow!("failed to start PTY session: {error}"))?
}

/// Explicitly stop and join the worker rather than dropping its control slot.
/// The worker owns the child, reader thread and PTY until cleanup finishes.
pub(crate) async fn stop_session(
    session: &Arc<Mutex<Option<PtySessionHandle>>>,
) -> anyhow::Result<()> {
    let handle = session.lock().take();
    if let Some(handle) = handle {
        let _ = handle.control_tx.send(ControlMessage::Kill);
        let killed = tokio::task::spawn_blocking(move || terminate_pty(&handle.termination)).await;
        // Join even if the independent killer panicked; the worker owns the
        // child wait and reader thread, and both must finish before returning.
        let joined = handle.task.await;
        killed.map_err(|error| anyhow::anyhow!("failed to terminate PTY child: {error}"))?;
        joined.map_err(|error| anyhow::anyhow!("failed to stop PTY worker: {error}"))??;
    }
    Ok(())
}

pub(crate) fn request_stop_session(session: &Arc<Mutex<Option<PtySessionHandle>>>) {
    if let Some(handle) = session.lock().as_ref() {
        let _ = handle.control_tx.send(ControlMessage::Kill);
        // Native termination is independent of the worker's control loop,
        // which may currently be blocked writing to a full terminal buffer.
        terminate_pty(&handle.termination);
    }
}

fn send_control(
    session: &Arc<Mutex<Option<PtySessionHandle>>>,
    message: ControlMessage,
) -> anyhow::Result<()> {
    let guard = session.lock();
    let Some(handle) = guard.as_ref() else {
        anyhow::bail!("failed to execute pty tool: no active PTY session");
    };
    handle
        .control_tx
        .send(message)
        .map_err(|_| anyhow::anyhow!("failed to execute pty tool: PTY session is unavailable"))
}

struct PtyRunState {
    child: Box<dyn Child + Send + Sync>,
    termination: PtyTermination,
    reader: Box<dyn Read + Send>,
    writer: Box<dyn Write + Send>,
    master: Box<dyn portable_pty::MasterPty + Send>,
}

fn prepare_pty(
    config: &PtyConfig,
    termination: PtyTermination,
    lifetime: &tokio_util::sync::CancellationToken,
) -> anyhow::Result<PtyRunState> {
    let system = native_pty_system();
    let pair = system.openpty(PtySize {
        rows: config.rows,
        cols: config.cols,
        pixel_width: 0,
        pixel_height: 0,
    })?;

    #[cfg(unix)]
    let mut command = CommandBuilder::new("/bin/sh");
    #[cfg(not(unix))]
    let mut command = CommandBuilder::new("sh");
    for arg in pty_shell_args() {
        command.arg(arg);
    }
    command.arg(&config.command);
    if let Some(cwd) = config.cwd.as_ref() {
        command.cwd(cwd);
    }
    command.env_clear();
    for (key, value) in pty_scrubbed_env(std::env::vars_os(), config.env.as_ref()) {
        command.env(key, value);
    }

    // Allocate fallible IO handles before spawning so an allocation failure
    // cannot leave an unowned child process behind.
    let reader = pair.master.try_clone_reader()?;
    let writer = pair.master.take_writer()?;
    let child = pair.slave.spawn_command(command)?;
    let child_pid = child.process_id().map(|pid| pid as i32);
    let process_group = child_pid.and_then(process_group_id);
    let mut control = termination.lock();
    control.killer = Some(PtyChildKiller {
        killer: child.clone_killer(),
        child_pid,
        process_group,
    });
    let stopped = control.stopped || lifetime.is_cancelled();
    drop(control);
    if stopped {
        terminate_pty(&termination);
    }

    Ok(PtyRunState {
        child,
        termination,
        reader,
        writer,
        master: pair.master,
    })
}

fn run_pty_loop(
    state: PtyRunState,
    timeout: Option<Duration>,
    control_rx: mpsc::Receiver<ControlMessage>,
    emit: Arc<dyn crate::ToolEventSink>,
) -> anyhow::Result<()> {
    let PtyRunState {
        mut child,
        termination,
        reader,
        mut writer,
        master,
    } = state;
    let start = Instant::now();
    let (reader_tx, reader_rx) = mpsc::channel();
    let reader_thread = spawn_reader_thread(reader, reader_tx);
    let mut reader_closed = false;

    let outcome = (|| -> anyhow::Result<()> {
        loop {
            drain_reader_output(&reader_rx, &emit, &mut reader_closed);

            match control_rx.recv_timeout(CONTROL_POLL_INTERVAL) {
                Ok(message) => match message {
                    ControlMessage::Input(input) => {
                        let _ = writer.write_all(input.as_bytes());
                        let _ = writer.flush();
                    }
                    ControlMessage::Resize { cols, rows } => {
                        let _ = master.resize(PtySize {
                            rows,
                            cols,
                            pixel_width: 0,
                            pixel_height: 0,
                        });
                    }
                    ControlMessage::Kill => {
                        terminate_pty(&termination);
                        break;
                    }
                },
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    terminate_pty(&termination);
                    break;
                }
            }

            drain_reader_output(&reader_rx, &emit, &mut reader_closed);

            if timeout.is_some_and(|timeout| start.elapsed() >= timeout) {
                terminate_pty(&termination);
                break;
            }

            if reader_closed {
                let mut killer = termination.lock();
                if child.try_wait()?.is_some() {
                    // Retire the killer while reaping under the same lock,
                    // so concurrent shutdown cannot signal an already reaped PID.
                    killer.killer.take();
                    break;
                }
            }
        }
        Ok(())
    })();

    if outcome.is_err() {
        terminate_pty(&termination);
    }

    drop(writer);
    drop(master);
    let waited = child.wait();
    let joined = reader_thread.join();
    drain_reader_output(&reader_rx, &emit, &mut reader_closed);
    outcome?;
    waited?;
    joined.map_err(|_| anyhow::anyhow!("failed to stop PTY reader: worker panicked"))?;
    Ok(())
}

fn spawn_reader_thread(
    mut reader: Box<dyn Read + Send>,
    tx: mpsc::Sender<ReaderEvent>,
) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let mut buffer = [0u8; 8192];
        loop {
            match reader.read(&mut buffer) {
                Ok(0) => break,
                Ok(count) => {
                    let chunk = String::from_utf8_lossy(&buffer[..count]).into_owned();
                    if tx.send(ReaderEvent::Output(chunk)).is_err() {
                        return;
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(_) => break,
            }
        }
        let _ = tx.send(ReaderEvent::Closed);
    })
}

fn drain_reader_output(
    reader_rx: &mpsc::Receiver<ReaderEvent>,
    emit: &Arc<dyn crate::ToolEventSink>,
    reader_closed: &mut bool,
) {
    while let Ok(event) = reader_rx.try_recv() {
        match event {
            ReaderEvent::Output(chunk) => emit.emit(ToolRuntimeEvent::ToolOutput {
                tool_name: "pty".to_owned(),
                chunk,
            }),
            ReaderEvent::Closed => *reader_closed = true,
        }
    }
}

fn terminate_pty(termination: &PtyTermination) {
    let mut state = termination.lock();
    state.stopped = true;
    if let Some(mut target) = state.killer.take() {
        terminate_pty_child(&mut target.killer, target.child_pid, target.process_group);
    }
}

#[cfg(unix)]
fn terminate_pty_child(
    child: &mut Box<dyn ChildKiller + Send + Sync>,
    child_pid: Option<i32>,
    process_group: Option<i32>,
) {
    if let Some(process_group) = process_group {
        let _ = kill_process_group(process_group, TERM_SIGNAL);
    }
    if let Some(child_pid) = child_pid {
        let _ = kill_tree(child_pid, TERM_SIGNAL);
    }
    let _ = child.kill();
    if let Some(process_group) = process_group {
        let _ = kill_process_group(process_group, KILL_SIGNAL);
    }
    if let Some(child_pid) = child_pid {
        let _ = kill_tree(child_pid, KILL_SIGNAL);
    }
}

#[cfg(not(unix))]
fn terminate_pty_child(
    child: &mut Box<dyn ChildKiller + Send + Sync>,
    child_pid: Option<i32>,
    _process_group: Option<i32>,
) {
    if let Some(child_pid) = child_pid {
        let _ = kill_tree(child_pid, TERM_SIGNAL);
    }
    let _ = child.kill();
    if let Some(child_pid) = child_pid {
        let _ = kill_tree(child_pid, KILL_SIGNAL);
    }
}

// AC1.9 (PTY half): Adversarial pinning for the rc-file inheritance and env
// scrubbing rules. See `docs/design-plans/2026-04-17-review-remediation-core.md`
// §AC1.9 — `sh -lc` is rejected; the spawned shell uses `-c` only and runs
// with a scrubbed env.
#[cfg(test)]
mod security_tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    use crate::{
        DefaultToolPolicy, NoopToolEventSink, PathLockMap, PolicySettings, ToolContext, ToolPolicy,
        ToolSessionStore,
    };

    use super::*;

    fn tool_context_with(policy: Arc<dyn ToolPolicy>) -> ToolContext {
        ToolContext {
            session_id: halter_protocol::SessionId::new(),
            working_dir: std::env::temp_dir(),
            path_locks: Arc::new(PathLockMap::default()),
            tool_sessions: Arc::new(ToolSessionStore::default()),
            snapshot: Arc::new(halter_protocol::ResourceSnapshot::empty()),
            cancel: CancellationToken::new(),
            emit: Arc::new(NoopToolEventSink),
            policy,
            shell_timeout_secs: 30,
            subagent_parent: None,
        }
    }

    #[test]
    fn ac1_9_pty_uses_dash_c_not_dash_lc() {
        // Pinning: the shell args must never include `-l` (login shell, sources
        // ~/.bash_profile, /etc/profile, ...). If this assertion fails, a
        // contributor reintroduced rc-file inheritance — a well-known bypass.
        let args = pty_shell_args();
        assert_eq!(args, &["-c"], "PTY shell args must be `-c`, got {args:?}");
        assert!(
            !args.iter().any(|a| a.contains('l')),
            "args must not contain `-l` / `-lc` flag (would source rc files)"
        );
    }

    #[test]
    fn pty_tool_spec_marks_tool_as_mutating() {
        let spec = PtyTool.spec();

        assert!(spec.capabilities.mutating);
        assert_eq!(spec.concurrency, ToolConcurrency::Exclusive);
    }

    #[test]
    fn ac1_9_pty_env_is_clear_then_allowlist_then_overrides() {
        let mut overrides = HashMap::new();
        overrides.insert("CALLER_OVERRIDE".to_owned(), "yes".to_owned());

        let env = pty_scrubbed_env(
            [
                (
                    std::ffi::OsString::from("AWS_SECRET_ACCESS_KEY"),
                    std::ffi::OsString::from("leaked"),
                ),
                (
                    std::ffi::OsString::from("PATH"),
                    std::ffi::OsString::from("/usr/bin"),
                ),
            ],
            Some(&overrides),
        );
        let keys: Vec<&str> = env.iter().map(|(k, _)| k.as_str()).collect();

        assert!(
            !keys.contains(&"AWS_SECRET_ACCESS_KEY"),
            "scrub must drop AWS_SECRET_ACCESS_KEY, got {keys:?}"
        );
        assert!(keys.contains(&"PATH"), "PATH must be preserved");
        assert!(
            keys.contains(&"CALLER_OVERRIDE"),
            "caller-supplied overrides must survive"
        );
    }

    #[tokio::test]
    async fn ac1_9_pty_start_is_denied_when_shell_disabled() {
        let policy: Arc<dyn ToolPolicy> = Arc::new(DefaultToolPolicy::new(PolicySettings {
            shell_enabled: false,
            ..PolicySettings::default()
        }));
        let context = tool_context_with(policy);

        let err = PtyTool
            .execute(
                context,
                json!({
                    "action": "start",
                    "command": "echo hi"
                }),
            )
            .await
            .expect_err("PTY start must be denied when shell is disabled");
        assert!(
            err.to_string().contains("disabled"),
            "expected ShellDisabled, got: {err}"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn session_shutdown_unblocks_pty_input_and_waits_for_process_and_worker() {
        let root = tempfile::tempdir().unwrap();
        let policy: Arc<dyn ToolPolicy> = Arc::new(DefaultToolPolicy::new(PolicySettings {
            allowed_read_roots: vec![root.path().to_owned()],
            allowed_shell_commands: vec![
                "printf".to_owned(),
                "sleep".to_owned(),
                "stty".to_owned(),
            ],
            ..PolicySettings::default()
        }));
        let mut context = tool_context_with(policy);
        context.working_dir = root.path().to_owned();
        PtyTool
            .execute(
                context.clone(),
                json!({"action": "start", "command": "stty -icanon -echo; printf '%s' \"$$\" > pid; sleep 30"}),
            )
            .await
            .unwrap();
        let pid = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Ok(pid) = tokio::fs::read_to_string(root.path().join("pid")).await
                    && let Ok(pid) = pid.parse::<i32>()
                {
                    break pid;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("PTY command publishes pid");
        assert!(
            PtyTool
                .execute(
                    context.clone(),
                    json!({"action": "start", "command": "sleep 30"})
                )
                .await
                .unwrap_err()
                .to_string()
                .contains("already active")
        );
        // This child does not read input. Raw mode stops canonical input
        // discarding from hiding the full buffer; FIFO control ordering puts
        // this write before Kill, so worker-only shutdown would block here.
        PtyTool
            .execute(
                context.clone(),
                json!({"action": "write", "input": "x".repeat(1024 * 1024)}),
            )
            .await
            .unwrap();
        context.cancel.cancel();
        assert!(
            context
                .tool_sessions
                .pty_session(&context.session_id)
                .lock()
                .is_some()
        );
        tokio::time::timeout(
            Duration::from_secs(5),
            context.tool_sessions.shutdown_session(&context.session_id),
        )
        .await
        .expect("shutdown unblocks the PTY writer")
        .unwrap();
        assert!(!context.tool_sessions.has_process_state(&context.session_id));
        // SAFETY: signal zero only probes the process created by this test.
        let exists = unsafe { libc::kill(pid, 0) };
        assert_eq!(exists, -1, "PTY process still alive after shutdown");
        assert_eq!(
            std::io::Error::last_os_error().raw_os_error(),
            Some(libc::ESRCH)
        );
    }
}
