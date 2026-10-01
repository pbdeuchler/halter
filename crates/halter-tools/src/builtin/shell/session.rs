// pattern: Imperative Shell

use std::collections::HashMap;
#[cfg(windows)]
use std::collections::HashSet;
use std::time::Duration;

use brush_builtins::{BuiltinSet, default_builtins};
use brush_core::{
    ExecutionControlFlow, ExecutionExitCode, ExecutionResult, ProcessGroupPolicy,
    ProfileLoadBehavior, RcLoadBehavior, Shell as BrushShell, ShellValue, ShellVariable,
    SourceInfo,
    env::EnvironmentScope,
    openfiles::{self, OpenFile, OpenFiles},
};
use tokio::sync::Mutex as TokioMutex;
use tokio::sync::mpsc;
use tokio::time::{self, timeout};
use tokio_util::sync::CancellationToken;

use crate::ToolEventSink;

use super::streaming::{collect_output, pipe_to_files};
use crate::builtin::process::{kill_process_group, signal_process};

const TERM_SIGNAL: i32 = 15;
const KILL_SIGNAL: i32 = 9;
const POST_EXIT_IDLE: Duration = Duration::from_millis(250);
const POST_EXIT_MAX: Duration = Duration::from_secs(2);
const READER_SHUTDOWN_TIMEOUT: Duration = Duration::from_millis(250);

pub struct ShellSessionCore {
    pub shell: BrushShell,
    lifetime: CancellationToken,
    processes: brush_core::processes::ProcessTracker,
}

/// Process controls stay accessible while Brush holds its execution lock.
#[derive(Default)]
pub struct ShellSession {
    core: TokioMutex<Option<ShellSessionCore>>,
    lifetime: CancellationToken,
    processes: brush_core::processes::ProcessTracker,
}

impl std::ops::Deref for ShellSession {
    type Target = TokioMutex<Option<ShellSessionCore>>;
    fn deref(&self) -> &Self::Target {
        &self.core
    }
}

impl ShellSession {
    pub(crate) fn new(
        lifetime: CancellationToken,
        activity: tokio::sync::watch::Sender<()>,
    ) -> Self {
        Self {
            lifetime,
            processes: brush_core::processes::ProcessTracker::with_activity(activity),
            ..Self::default()
        }
    }
    pub(crate) fn has_running(&self) -> bool {
        self.processes.has_running()
    }
    pub(crate) fn request_stop(&self, force: bool) {
        self.lifetime.cancel();
        if force {
            self.processes.force_stop();
        }
    }
}

impl ShellSessionCore {
    async fn shutdown(mut self) -> anyhow::Result<()> {
        self.lifetime.cancel();
        terminate_background_jobs(&self.shell).await;
        let mut errors = Vec::new();
        for job in &mut self.shell.jobs_mut().jobs {
            if let Err(error) = job.wait().await
                && !matches!(error.kind(), brush_core::error::ErrorKind::Interrupted)
            {
                errors.push(error.to_string());
            }
        }
        anyhow::ensure!(
            errors.is_empty(),
            "failed to reap persistent shell jobs: {}",
            errors.join("; ")
        );
        Ok(())
    }
}

impl Drop for ShellSessionCore {
    fn drop(&mut self) {
        self.lifetime.cancel();
        self.processes.force_stop();
    }
}

pub struct ShellRunOptions {
    pub command: String,
    pub cwd: Option<String>,
    pub default_cwd: Option<String>,
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
    session: std::sync::Arc<ShellSession>,
    options: ShellRunOptions,
    emit: std::sync::Arc<dyn ToolEventSink>,
    cancel: CancellationToken,
) -> anyhow::Result<ShellRunResult> {
    let run_cancel = CancellationToken::new();
    struct CancelOnDrop {
        cancel: CancellationToken,
        processes: brush_core::processes::ProcessTracker,
        armed: bool,
    }
    impl Drop for CancelOnDrop {
        fn drop(&mut self) {
            if self.armed {
                self.cancel.cancel();
                self.processes.force_stop();
            }
        }
    }
    // A force-aborted tool future must signal its separately spawned worker.
    // Otherwise dropping that worker's handle would detach running children.
    let mut abort_guard = CancelOnDrop {
        cancel: run_cancel.clone(),
        processes: session.processes.clone(),
        armed: true,
    };
    let mut task = tokio::spawn({
        let session = session.clone();
        let emit = emit.clone();
        let options = ShellRunOptions {
            command: options.command.clone(),
            cwd: options.cwd.clone(),
            default_cwd: options.default_cwd.clone(),
            env: options.env.clone(),
            timeout: options.timeout,
        };
        let run_cancel = run_cancel.clone();
        async move {
            let mut guard = session.lock().await;
            anyhow::ensure!(
                !session.lifetime.is_cancelled(),
                "failed to execute shell: session is closed"
            );
            let shell = match &mut *guard {
                Some(shell) => shell,
                None => {
                    let mut shell =
                        create_session(session.lifetime.child_token(), session.processes.clone())
                            .await?;
                    if let Some(cwd) = options.default_cwd.as_deref() {
                        shell.shell.set_working_dir(cwd).map_err(|error| {
                            anyhow::anyhow!("failed to set default shell cwd: {error}")
                        })?;
                    }
                    guard.insert(shell)
                }
            };

            let result = run_shell_command(shell, &options, emit, run_cancel).await;
            if !result
                .as_ref()
                .is_ok_and(|output| session_keepalive(&output.result))
                && let Some(shell) = guard.take()
            {
                shell.shutdown().await?;
            }
            result
        }
    });

    let outcome = tokio::select! {
        joined = &mut task => joined.map_err(|error| anyhow::anyhow!("failed to execute shell session task: {error}"))??,
        _ = cancel.cancelled() => {
            cancel_shell_task(&session, &run_cancel, &mut task).await;
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
            cancel_shell_task(&session, &run_cancel, &mut task).await;
            return Ok(ShellRunResult {
                exit_code: None,
                stdout: String::new(),
                stderr: String::new(),
                timed_out: true,
                cancelled: false,
            });
        }
    };

    abort_guard.armed = false;

    Ok(ShellRunResult {
        exit_code: Some(exit_code(&outcome.result)),
        stdout: outcome.stdout,
        stderr: outcome.stderr,
        timed_out: false,
        cancelled: false,
    })
}

async fn cancel_shell_task(
    session: &std::sync::Arc<ShellSession>,
    run_cancel: &CancellationToken,
    task: &mut tokio::task::JoinHandle<anyhow::Result<ShellCommandOutput>>,
) {
    run_cancel.cancel();
    if timeout(Duration::from_secs(2), &mut *task).await.is_err() {
        task.abort();
        let _ = task.await;
    }
    reset_shell_session(session).await;
}

async fn reset_shell_session(session: &std::sync::Arc<ShellSession>) {
    if let Err(error) = shutdown_shell_session(session).await {
        tracing::warn!(%error, "failed to reap persistent shell jobs");
    }
}

pub(crate) async fn shutdown_shell_session(
    session: &std::sync::Arc<ShellSession>,
) -> anyhow::Result<()> {
    let mut guard = session.lock().await;
    if let Some(shell) = guard.take() {
        shell.shutdown().await?;
    }
    Ok(())
}

async fn create_session(
    lifetime: CancellationToken,
    processes: brush_core::processes::ProcessTracker,
) -> anyhow::Result<ShellSessionCore> {
    let mut shell = BrushShell::builder()
        .interactive(false)
        .login(false)
        .profile(ProfileLoadBehavior::Skip)
        .rc(RcLoadBehavior::Skip)
        .do_not_inherit_env(true)
        .builtins(default_builtins(BuiltinSet::BashMode))
        .build()
        .await
        .map_err(|error| anyhow::anyhow!("failed to initialize shell: {error}"))?;

    let mut merged_path: Option<String> = None;
    for (key, value) in std::env::vars() {
        let normalized_key = normalize_env_key(&key);
        if should_skip_env_var(normalized_key) || !should_inherit_env_var(normalized_key) {
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
        shell.env_mut().set_global(normalized_key, variable)?;
    }

    if let Some(path_value) = merged_path {
        let mut variable = ShellVariable::new(ShellValue::String(path_value));
        variable.export();
        shell.env_mut().set_global("PATH", variable)?;
    }

    Ok(ShellSessionCore {
        shell,
        lifetime,
        processes,
    })
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
    let scoped_cancel = session.lifetime.child_token();
    params.set_cancel_token(scoped_cancel.clone());
    params.set_process_tracker(session.processes.clone());

    let mut env_scope_pushed = false;
    if let Some(env) = options.env.as_ref() {
        session
            .shell
            .env_mut()
            .push_scope(EnvironmentScope::Command);
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
                .env_mut()
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

    let source = SourceInfo::default();
    let result = {
        let execution = session
            .shell
            .run_string(options.command.clone(), &source, &params);
        tokio::pin!(execution);
        tokio::select! {
            result = &mut execution => result,
            _ = cancel.cancelled() => {
                scoped_cancel.cancel();
                execution.await
            }
        }
    };

    if cancel.is_cancelled() {
        terminate_background_jobs(&session.shell).await;
    }

    if env_scope_pushed {
        session
            .shell
            .env_mut()
            .pop_scope(EnvironmentScope::Command)?;
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

    let result =
        result.map_err(|error| anyhow::anyhow!("failed to execute shell command: {error}"))?;
    Ok(ShellCommandOutput {
        result,
        stdout,
        stderr,
    })
}

async fn await_reader(handle: &mut tokio::task::JoinHandle<anyhow::Result<String>>) -> String {
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
        // 141 = 128 + SIGPIPE.
        ExecutionExitCode::BrokenPipe => 141,
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

    std::env::join_paths(merged.iter()).map_or_else(
        |_| merged.join(";"),
        |paths| paths.to_string_lossy().into_owned(),
    )
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

fn should_inherit_env_var(key: &str) -> bool {
    if key == "PATH" {
        return true;
    }

    if matches!(
        key,
        "HOME"
            | "USER"
            | "LOGNAME"
            | "TERM"
            | "COLORTERM"
            | "LANG"
            | "LC_ALL"
            | "TMPDIR"
            | "TEMP"
            | "TMP"
            | "USERNAME"
            | "APPDATA"
            | "LOCALAPPDATA"
            | "ProgramData"
            | "SystemRoot"
            | "WINDIR"
            | "ComSpec"
            | "PATHEXT"
            | "NUMBER_OF_PROCESSORS"
    ) {
        return true;
    }

    key.starts_with("LC_") || key.starts_with("XDG_")
}

fn session_keepalive(result: &ExecutionResult) -> bool {
    matches!(result.next_control_flow, ExecutionControlFlow::Normal)
}

/// Delay between TERM and the follow-up KILL sweep. Matches the pre-review
/// behavior; kept inline (not a detached `tokio::spawn`) so callers observe a
/// fully drained process group before returning (finding M36).
#[cfg(unix)]
const POST_EXIT_KILL_DELAY: Duration = Duration::from_millis(500);

#[cfg(unix)]
async fn terminate_background_jobs(shell: &BrushShell) {
    if shell.jobs().jobs.is_empty() {
        return;
    }

    let mut pgids = Vec::new();
    let mut pids = Vec::new();
    for job in &shell.jobs().jobs {
        if let Some(pgid) = job.process_group_id()
            && !pgids.contains(&pgid)
        {
            pgids.push(pgid);
        }
        if let Some(pid) = job.representative_pid()
            && !pids.contains(&pid)
        {
            pids.push(pid);
        }
    }

    let mut signalled = false;
    for &pgid in &pgids {
        signalled |= kill_process_group(pgid, TERM_SIGNAL);
    }
    for &pid in &pids {
        signalled |= signal_process(pid, TERM_SIGNAL);
    }
    if !signalled {
        return;
    }

    time::sleep(POST_EXIT_KILL_DELAY).await;
    for pgid in pgids {
        let _ = kill_process_group(pgid, KILL_SIGNAL);
    }
    for pid in pids {
        let _ = signal_process(pid, KILL_SIGNAL);
    }
}

#[cfg(not(unix))]
async fn terminate_background_jobs(shell: &BrushShell) {
    // Brush's process tracker uses native tree termination on Windows.
    for job in &shell.jobs().jobs {
        if let Some(pid) = job.representative_pid() {
            signal_process(pid, KILL_SIGNAL);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(unix)]
    #[tokio::test]
    async fn shell_background_activity_finishes_without_relocking_shell() {
        let store = crate::ToolSessionStore::default();
        let id = halter_protocol::SessionId::new();
        store.open_session(&id);
        let mut activity = store.subscribe_activity();
        let session = store.shell_session(&id);
        run_persistent_shell(
            session.clone(),
            ShellRunOptions {
                command: "sleep 30 &".to_owned(),
                cwd: None,
                default_cwd: None,
                env: None,
                timeout: None,
            },
            std::sync::Arc::new(crate::NoopToolEventSink),
            CancellationToken::new(),
        )
        .await
        .unwrap();
        assert!(
            store.has_running_jobs(&id),
            "the asynchronous shell job survives command completion"
        );
        // Holding the shell execution lock cannot prevent activity inspection
        // or final completion notification from the owned job task.
        let shell = session.lock().await;
        activity.borrow_and_update();
        session.processes.force_stop();
        time::timeout(Duration::from_secs(5), async {
            while store.has_running_jobs(&id) {
                activity.changed().await.unwrap();
            }
        })
        .await
        .expect("the shell job publishes completion without a new command");
        assert!(shell.is_some(), "the persistent shell slot remains open");
        drop(shell);
        store.shutdown_session(&id).await.unwrap();
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancellation_reaps_empty_group_without_waiting_full_term_grace() {
        let mut command = tokio::process::Command::new("/bin/sleep");
        command.arg("30").process_group(0).kill_on_drop(true);
        let child = command.spawn().unwrap();
        let pid = child.id().unwrap() as i32;
        let mut child = brush_core::processes::ChildProcess::new(child, Some(pid), Some(pid));
        let cancel = CancellationToken::new();
        cancel.cancel();
        let result = tokio::time::timeout(Duration::from_millis(450), child.wait(Some(cancel)))
            .await
            .expect("an empty group needs no remaining TERM grace")
            .unwrap();
        assert!(matches!(
            result,
            brush_core::processes::ProcessWaitResult::Cancelled
        ));
    }

    #[test]
    fn inherit_env_var_uses_allowlist() {
        assert!(should_inherit_env_var("PATH"));
        assert!(should_inherit_env_var("HOME"));
        assert!(!should_inherit_env_var("OPENAI_API_KEY"));
        assert!(!should_inherit_env_var("GITHUB_TOKEN"));
    }
}
