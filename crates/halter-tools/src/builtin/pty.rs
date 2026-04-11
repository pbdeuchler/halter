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
    ToolScope, ensure_not_cancelled, optional_string, optional_u64, required_string,
};
use super::process::{kill_process_group, kill_tree, process_group_id};

const TERM_SIGNAL: i32 = 15;
const KILL_SIGNAL: i32 = 9;
const CONTROL_POLL_INTERVAL: Duration = Duration::from_millis(25);

pub struct PtySessionHandle {
    control_tx: mpsc::Sender<ControlMessage>,
}

#[derive(Debug)]
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
                requires_approval: true,
                cancellable: false,
                long_running: true,
            },
            provider_aliases: Default::default(),
        }
    }

    async fn execute(&self, context: ToolContext, input: Value) -> anyhow::Result<ToolResult> {
        let _scope = ToolScope::new(&context, "pty");
        ensure_not_cancelled(&context.cancel)?;
        context.policy.check_shell("shell").await?;

        let action = required_string(&input, "action")?;
        let session = context.tool_sessions.pty_session(&context.session_id);

        match action {
            "start" => {
                let config = PtyConfig {
                    command: required_string(&input, "command")?.to_owned(),
                    cwd: optional_string(&input, "cwd").map(ToOwned::to_owned),
                    env: parse_env_map(input.get("env"))?,
                    timeout: optional_u64(&input, "timeout_ms")?.map(Duration::from_millis),
                    cols: optional_u64(&input, "cols")?.unwrap_or(120) as u16,
                    rows: optional_u64(&input, "rows")?.unwrap_or(40) as u16,
                };
                start_session(session, config, context.emit.clone());
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
                let cols = optional_u64(&input, "cols")?.unwrap_or(120) as u16;
                let rows = optional_u64(&input, "rows")?.unwrap_or(40) as u16;
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

fn parse_env_map(value: Option<&Value>) -> anyhow::Result<Option<HashMap<String, String>>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let object = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("invalid tool input: env must be an object"))?;
    let mut env = HashMap::with_capacity(object.len());
    for (key, value) in object {
        let value = value
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("invalid tool input: env values must be strings"))?;
        env.insert(key.clone(), value.to_owned());
    }
    Ok(Some(env))
}

fn start_session(
    session: Arc<Mutex<Option<PtySessionHandle>>>,
    config: PtyConfig,
    emit: Arc<dyn crate::ToolEventSink>,
) {
    let (control_tx, control_rx) = mpsc::channel();
    {
        let mut guard = session.lock();
        *guard = Some(PtySessionHandle { control_tx });
    }

    tokio::task::spawn_blocking(move || {
        let _ = run_pty(config, control_rx, emit);
        *session.lock() = None;
    });
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

fn run_pty(
    config: PtyConfig,
    control_rx: mpsc::Receiver<ControlMessage>,
    emit: Arc<dyn crate::ToolEventSink>,
) -> anyhow::Result<()> {
    let system = native_pty_system();
    let pair = system.openpty(PtySize {
        rows: config.rows,
        cols: config.cols,
        pixel_width: 0,
        pixel_height: 0,
    })?;

    let mut command = CommandBuilder::new("sh");
    command.arg("-lc");
    command.arg(&config.command);
    if let Some(cwd) = config.cwd.as_ref() {
        command.cwd(cwd);
    }
    if let Some(env) = config.env.as_ref() {
        for (key, value) in env {
            command.env(key, value);
        }
    }

    let mut child = pair.slave.spawn_command(command)?;
    let child_pid = child.process_id().map(|pid| pid as i32);
    let process_group = child_pid.and_then(process_group_id);

    let reader = pair.master.try_clone_reader()?;
    let mut writer = pair.master.take_writer()?;
    let master = pair.master;
    let start = Instant::now();
    let timeout = config.timeout;
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
