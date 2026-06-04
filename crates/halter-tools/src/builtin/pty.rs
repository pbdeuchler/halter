// pattern: Imperative Shell

use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::{Arc, mpsc};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use halter_protocol::{ToolCapabilities, ToolConcurrency, ToolName, ToolResult, ToolSpec};
use parking_lot::Mutex;
use portable_pty::{Child, CommandBuilder, PtySize, native_pty_system};
use serde_json::{Value, json};

use crate::{Tool, ToolContext, ToolRuntimeEvent};

use super::common::{
    ToolScope, ensure_not_cancelled, optional_string, optional_u64, parse_env_map, required_string,
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
/// Pure: takes its inputs (parent env via `std::env::var_os` and the
/// caller-supplied map) and returns a Vec — no side effects, easy to test.
fn pty_scrubbed_env(
    overrides: Option<&HashMap<String, String>>,
) -> Vec<(String, std::ffi::OsString)> {
    let mut out: Vec<(String, std::ffi::OsString)> = Vec::new();
    for var in PTY_ENV_ALLOWLIST {
        if let Some(value) = std::env::var_os(var) {
            out.push(((*var).to_owned(), value));
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
                mutating: false,
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
                start_session(session, config, context.emit.clone()).await?;
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
                send_control(&session, ControlMessage::Kill)?;
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
) -> anyhow::Result<()> {
    let (control_tx, control_rx) = mpsc::channel();
    // Use a one-shot blocking channel to surface the spawn result
    // synchronously back to the async caller (finding M40). The blocking
    // task first opens the pty + spawns the child; only on success does it
    // publish the session handle and enter the event loop.
    let (ready_tx, ready_rx) = std::sync::mpsc::sync_channel::<anyhow::Result<()>>(1);
    let session_for_task = Arc::clone(&session);
    tokio::task::spawn_blocking(move || {
        let state = match prepare_pty(&config) {
            Ok(state) => state,
            Err(error) => {
                let _ = ready_tx.send(Err(error));
                return;
            }
        };
        {
            let mut guard = session_for_task.lock();
            *guard = Some(PtySessionHandle { control_tx });
        }
        let _ = ready_tx.send(Ok(()));
        let _ = run_pty_loop(state, config.timeout, control_rx, emit);
        *session_for_task.lock() = None;
    });

    tokio::task::spawn_blocking(move || {
        ready_rx.recv().unwrap_or_else(|_| {
            Err(anyhow::anyhow!(
                "pty tool: spawn task dropped ready channel"
            ))
        })
    })
    .await
    .map_err(|err| anyhow::anyhow!("failed to execute pty tool: spawn await failed: {err}"))?
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
    child_pid: Option<i32>,
    process_group: Option<i32>,
    reader: Box<dyn Read + Send>,
    writer: Box<dyn Write + Send>,
    master: Box<dyn portable_pty::MasterPty + Send>,
}

fn prepare_pty(config: &PtyConfig) -> anyhow::Result<PtyRunState> {
    let system = native_pty_system();
    let pair = system.openpty(PtySize {
        rows: config.rows,
        cols: config.cols,
        pixel_width: 0,
        pixel_height: 0,
    })?;

    let mut command = CommandBuilder::new("sh");
    for arg in pty_shell_args() {
        command.arg(arg);
    }
    command.arg(&config.command);
    if let Some(cwd) = config.cwd.as_ref() {
        command.cwd(cwd);
    }
    command.env_clear();
    for (key, value) in pty_scrubbed_env(config.env.as_ref()) {
        command.env(key, value);
    }

    let child = pair.slave.spawn_command(command)?;
    let child_pid = child.process_id().map(|pid| pid as i32);
    let process_group = child_pid.and_then(process_group_id);

    let reader = pair.master.try_clone_reader()?;
    let writer = pair.master.take_writer()?;
    Ok(PtyRunState {
        child,
        child_pid,
        process_group,
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
        child_pid,
        process_group,
        reader,
        mut writer,
        master,
    } = state;
    let start = Instant::now();
    let (reader_tx, reader_rx) = mpsc::channel();
    let reader_thread = spawn_reader_thread(reader, reader_tx);
    let mut reader_closed = false;

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
                    terminate_pty(&mut child, child_pid, process_group);
                    break;
                }
            },
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                if child.try_wait()?.is_some() {
                    break;
                }
            }
        }

        drain_reader_output(&reader_rx, &emit, &mut reader_closed);

        if timeout.is_some_and(|timeout| start.elapsed() >= timeout) {
            terminate_pty(&mut child, child_pid, process_group);
            break;
        }

        if reader_closed && child.try_wait()?.is_some() {
            break;
        }
    }

    drop(writer);
    drop(master);
    let _ = child.wait();
    let _ = reader_thread.join();
    drain_reader_output(&reader_rx, &emit, &mut reader_closed);
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

#[cfg(unix)]
fn terminate_pty(
    child: &mut Box<dyn Child + Send + Sync>,
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
fn terminate_pty(
    child: &mut Box<dyn Child + Send + Sync>,
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
    fn ac1_9_pty_env_is_clear_then_allowlist_then_overrides() {
        // SAFETY: a deliberately leaky env var that should not survive the scrub.
        unsafe { std::env::set_var("AWS_SECRET_ACCESS_KEY", "leaked") };
        unsafe { std::env::set_var("PATH", "/usr/bin") };

        let mut overrides = HashMap::new();
        overrides.insert("CALLER_OVERRIDE".to_owned(), "yes".to_owned());

        let env = pty_scrubbed_env(Some(&overrides));
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

        unsafe { std::env::remove_var("AWS_SECRET_ACCESS_KEY") };
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
}
