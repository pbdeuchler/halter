// pattern: Imperative Shell

use std::collections::HashMap;
use std::time::Duration;

use brush_builtins::{BuiltinSet, default_builtins};
use brush_core::{
    CreateOptions, ExecutionControlFlow, ExecutionExitCode, ExecutionResult,
    ProcessGroupPolicy, Shell as BrushShell, ShellValue, ShellVariable, env::EnvironmentScope,
    openfiles::{self, OpenFile, OpenFiles},
};
use tokio::sync::Mutex as TokioMutex;
use tokio::sync::mpsc;
use tokio::time::{self, timeout};
use tokio_util::sync::CancellationToken;

use crate::ToolEventSink;

use super::streaming::{collect_output, pipe_to_files};
use crate::builtin::process::{kill_process_group, kill_tree};

const TERM_SIGNAL: i32 = 15;
const KILL_SIGNAL: i32 = 9;
const POST_EXIT_IDLE: Duration = Duration::from_millis(250);
const POST_EXIT_MAX: Duration = Duration::from_secs(2);
const READER_SHUTDOWN_TIMEOUT: Duration = Duration::from_millis(250);

pub struct ShellSessionCore {
    pub shell: BrushShell,
}

pub struct ShellRunOptions {
    pub command: String,
    pub cwd: Option<String>,
    pub env: Option<HashMap<String, String>>,
    pub timeout: Option<Duration>,
}

pub struct ShellRunResult {
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub timed_out: bool,
    pub cancelled: bool,
}

struct ShellCommandOutput {
    result: ExecutionResult,
    stdout: String,
    stderr: String,
}

pub async fn run_persistent_shell(
    session: std::sync::Arc<TokioMutex<Option<ShellSessionCore>>>,
    options: ShellRunOptions,
    emit: std::sync::Arc<dyn ToolEventSink>,
    cancel: CancellationToken,
) -> anyhow::Result<ShellRunResult> {
    let run_cancel = CancellationToken::new();
    let task = tokio::spawn({
        let session = session.clone();
        let emit = emit.clone();
        let options = ShellRunOptions {
            command: options.command.clone(),
            cwd: options.cwd.clone(),
            env: options.env.clone(),
            timeout: options.timeout,
        };
        let run_cancel = run_cancel.clone();
        async move {
            let mut guard = session.lock().await;
            let shell = match &mut *guard {
                Some(shell) => shell,
                None => guard.insert(create_session().await?),
            };

            let result = run_shell_command(shell, &options, emit, run_cancel).await;
            if !result.as_ref().is_ok_and(|output| session_keepalive(&output.result)) {
                *guard = None;
            }
            result
        }
    });
    tokio::pin!(task);

    let outcome = tokio::select! {
        joined = &mut task => joined.map_err(|error| anyhow::anyhow!("failed to execute shell session task: {error}"))??,
        _ = cancel.cancelled() => {
            run_cancel.cancel();
            let _ = timeout(Duration::from_secs(2), &mut task).await;
            return Ok(ShellRunResult {
                exit_code: None,
                stdout: String::new(),
                stderr: String::new(),
                timed_out: false,
                cancelled: true,
            });
        }
        _ = async {
            if let Some(timeout) = options.timeout {
                time::sleep(timeout).await;
            } else {
                std::future::pending::<()>().await;
            }
        } => {
            run_cancel.cancel();
            let _ = timeout(Duration::from_secs(2), &mut task).await;
            return Ok(ShellRunResult {
                exit_code: None,
                stdout: String::new(),
                stderr: String::new(),
                timed_out: true,
                cancelled: false,
            });
        }
    };

    Ok(ShellRunResult {
        exit_code: Some(exit_code(&outcome.result)),
        stdout: outcome.stdout,
        stderr: outcome.stderr,
        timed_out: false,
        cancelled: false,
    })
}

async fn create_session() -> anyhow::Result<ShellSessionCore> {
    let create_options = CreateOptions {
        interactive: false,
        login: false,
        no_profile: true,
        no_rc: true,
        do_not_inherit_env: true,
        builtins: default_builtins(BuiltinSet::BashMode),
        ..Default::default()
    };

    let mut shell = BrushShell::new(create_options)
        .await
        .map_err(|error| anyhow::anyhow!("failed to initialize shell: {error}"))?;

    let mut merged_path: Option<String> = None;
    for (key, value) in std::env::vars() {
        let normalized_key = normalize_env_key(&key);
        if should_skip_env_var(normalized_key) {
            continue;
        }
        if normalized_key == "PATH" {
            merged_path = Some(match merged_path {
                Some(existing) => merge_path_values(&existing, &value),
                None => value,
            });
            continue;
        }

        let mut variable = ShellVariable::new(ShellValue::String(value));
        variable.export();
        shell.env.set_global(normalized_key, variable)?;
    }

    if let Some(path_value) = merged_path {
        let mut variable = ShellVariable::new(ShellValue::String(path_value));
        variable.export();
        shell.env.set_global("PATH", variable)?;
    }

    Ok(ShellSessionCore { shell })
}

async fn run_shell_command(
    session: &mut ShellSessionCore,
    options: &ShellRunOptions,
    emit: std::sync::Arc<dyn ToolEventSink>,
    cancel: CancellationToken,
) -> anyhow::Result<ShellCommandOutput> {
    if let Some(cwd) = options.cwd.as_deref() {
        session
            .shell
            .set_working_dir(cwd)
            .map_err(|error| anyhow::anyhow!("failed to set shell cwd: {error}"))?;
    }

    let (stdout_reader, stdout_writer) = pipe_to_files("stdout")?;
    let (stderr_reader, stderr_writer) = pipe_to_files("stderr")?;

    let mut params = session.shell.default_exec_params();
    params.set_fd(OpenFiles::STDIN_FD, null_file()?);
    params.set_fd(OpenFiles::STDOUT_FD, OpenFile::from(stdout_writer));
    params.set_fd(OpenFiles::STDERR_FD, OpenFile::from(stderr_writer));
    params.process_group_policy = ProcessGroupPolicy::NewProcessGroup;
    params.set_cancel_token(cancel.clone());

    let mut env_scope_pushed = false;
    if let Some(env) = options.env.as_ref() {
        session.shell.env.push_scope(EnvironmentScope::Command);
        env_scope_pushed = true;
        for (key, value) in env {
            let normalized_key = normalize_env_key(key);
            if should_skip_env_var(normalized_key) {
                continue;
            }
            let mut variable = ShellVariable::new(ShellValue::String(value.clone()));
            variable.export();
            session
                .shell
                .env
                .add(normalized_key, variable, EnvironmentScope::Command)?;
        }
    }

    let reader_cancel = CancellationToken::new();
    let (activity_tx, mut activity_rx) = mpsc::channel::<()>(4);
    let mut stdout_handle = tokio::spawn(collect_output(
        stdout_reader,
        emit.clone(),
        "shell",
        reader_cancel.clone(),
        activity_tx.clone(),
    ));
    let mut stderr_handle = tokio::spawn(collect_output(
        stderr_reader,
        emit,
        "shell",
        reader_cancel.clone(),
        activity_tx,
    ));

    let result = session.shell.run_string(options.command.clone(), &params).await;

    if cancel.is_cancelled() {
        terminate_background_jobs(&session.shell);
    }

    if env_scope_pushed {
        session.shell.env.pop_scope(EnvironmentScope::Command)?;
    }

    let mut stdout_finished = false;
    let mut stderr_finished = false;
    let mut idle_timer = Box::pin(time::sleep(POST_EXIT_IDLE));
    let mut max_timer = Box::pin(time::sleep(POST_EXIT_MAX));

    loop {
        tokio::select! {
            output = &mut stdout_handle, if !stdout_finished => {
                stdout_finished = true;
                if stderr_finished {
                    break;
                }
                let _ = output;
            }
            output = &mut stderr_handle, if !stderr_finished => {
                stderr_finished = true;
                if stdout_finished {
                    break;
                }
                let _ = output;
            }
            message = activity_rx.recv() => {
                if message.is_none() {
                    break;
                }
                idle_timer.as_mut().reset(time::Instant::now() + POST_EXIT_IDLE);
            }
            () = &mut idle_timer => break,
            () = &mut max_timer => break,
        }
    }

    if !stdout_finished || !stderr_finished {
        reader_cancel.cancel();
    }

    let stdout = await_reader(&mut stdout_handle).await;
    let stderr = await_reader(&mut stderr_handle).await;

    let result = result.map_err(|error| anyhow::anyhow!("failed to execute shell command: {error}"))?;
    Ok(ShellCommandOutput {
        result,
        stdout,
        stderr,
    })
}

async fn await_reader(
    handle: &mut tokio::task::JoinHandle<anyhow::Result<String>>,
) -> String {
    match timeout(READER_SHUTDOWN_TIMEOUT, &mut *handle).await {
        Ok(Ok(Ok(output))) => output,
        Ok(Ok(Err(_))) | Ok(Err(_)) | Err(_) => {
            handle.abort();
            String::new()
        }
    }
}

fn null_file() -> anyhow::Result<OpenFile> {
    openfiles::null().map_err(|error| anyhow::anyhow!("failed to create null file: {error}"))
}

fn exit_code(result: &ExecutionResult) -> i32 {
    match result.exit_code {
        ExecutionExitCode::Success => 0,
        ExecutionExitCode::GeneralError => 1,
        ExecutionExitCode::InvalidUsage => 2,
        ExecutionExitCode::Unimplemented => 99,
        ExecutionExitCode::CannotExecute => 126,
        ExecutionExitCode::NotFound => 127,
        ExecutionExitCode::Interrupted => 130,
        ExecutionExitCode::Custom(code) => code as i32,
    }
}

#[cfg(windows)]
const fn normalize_env_key(key: &str) -> &str {
    if key.eq_ignore_ascii_case("PATH") {
        "PATH"
    } else {
        key
    }
}

#[cfg(not(windows))]
const fn normalize_env_key(key: &str) -> &str {
    key
}

#[cfg(windows)]
fn merge_path_values(existing: &str, incoming: &str) -> String {
    let mut merged = Vec::new();
    let mut seen = HashSet::new();
    push_unique_paths(&mut merged, &mut seen, existing);
    push_unique_paths(&mut merged, &mut seen, incoming);

    std::env::join_paths(merged.iter())
        .map_or_else(|_| merged.join(";"), |paths| paths.to_string_lossy().into_owned())
}

#[cfg(windows)]
fn push_unique_paths(merged: &mut Vec<String>, seen: &mut HashSet<String>, value: &str) {
    for segment in std::env::split_paths(value) {
        let segment_str = segment.to_string_lossy().into_owned();
        let normalized = segment_str.trim().trim_matches('"').to_ascii_lowercase();
        if normalized.is_empty() {
            continue;
        }
        if seen.insert(normalized) {
            merged.push(segment_str);
        }
    }
}

#[cfg(not(windows))]
fn merge_path_values(_existing: &str, incoming: &str) -> String {
    incoming.to_owned()
}

fn should_skip_env_var(key: &str) -> bool {
    if key.starts_with("BASH_FUNC_") && key.ends_with("%%") {
        return true;
    }

    matches!(
        key,
        "BASH_ENV"
            | "ENV"
            | "HISTFILE"
            | "HISTTIMEFORMAT"
            | "HISTCMD"
            | "PS0"
            | "PS1"
            | "PS2"
            | "PS4"
            | "BRUSH_PS_ALT"
            | "READLINE_LINE"
            | "READLINE_POINT"
            | "BRUSH_VERSION"
            | "BASH"
            | "BASHOPTS"
            | "BASH_ALIASES"
            | "BASH_ARGV0"
            | "BASH_CMDS"
            | "BASH_SOURCE"
            | "BASH_SUBSHELL"
            | "BASH_VERSINFO"
            | "BASH_VERSION"
            | "SHELLOPTS"
            | "SHLVL"
            | "SHELL"
            | "COMP_WORDBREAKS"
            | "DIRSTACK"
            | "EPOCHREALTIME"
            | "EPOCHSECONDS"
            | "FUNCNAME"
            | "GROUPS"
            | "IFS"
            | "LINENO"
            | "MACHTYPE"
            | "OSTYPE"
            | "OPTERR"
            | "OPTIND"
            | "PIPESTATUS"
            | "PPID"
            | "PWD"
            | "OLDPWD"
            | "RANDOM"
            | "SRANDOM"
            | "SECONDS"
            | "UID"
            | "EUID"
            | "HOSTNAME"
            | "HOSTTYPE"
    )
}

fn session_keepalive(result: &ExecutionResult) -> bool {
    matches!(result.next_control_flow, ExecutionControlFlow::Normal)
}

#[cfg(unix)]
fn terminate_background_jobs(shell: &BrushShell) {
    if shell.jobs.jobs.is_empty() {
        return;
    }

    let mut pgids = Vec::new();
    let mut pids = Vec::new();
    for job in &shell.jobs.jobs {
        if let Some(pgid) = job.process_group_id() {
            if !pgids.contains(&pgid) {
                pgids.push(pgid);
            }
        }
        if let Some(pid) = job.representative_pid() {
            if !pids.contains(&pid) {
                pids.push(pid);
            }
        }
    }

    for &pgid in &pgids {
        let _ = kill_process_group(pgid, TERM_SIGNAL);
    }
    for &pid in &pids {
        let _ = kill_tree(pid, TERM_SIGNAL);
    }

    tokio::spawn(async move {
        time::sleep(Duration::from_millis(500)).await;
        for pgid in pgids {
            let _ = kill_process_group(pgid, KILL_SIGNAL);
        }
        for pid in pids {
            let _ = kill_tree(pid, KILL_SIGNAL);
        }
    });
}

#[cfg(not(unix))]
fn terminate_background_jobs(_shell: &BrushShell) {}
