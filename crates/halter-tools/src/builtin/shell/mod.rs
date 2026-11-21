// pattern: Imperative Shell

pub mod detect;
pub mod streaming;

use std::process::{ExitStatus, Stdio};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use halter_protocol::{ToolCapabilities, ToolConcurrency, ToolName, ToolSpec};
use serde::Serialize;
use serde_json::{Value, json};
use tokio::process::Command;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::policy::check_tool_timeout;
use crate::{ToolContext, ToolEventSink, ToolOutput, TypedTool};

use self::streaming::{collect_output, pipe_to_files};
use super::common::{
    ToolScope, ensure_not_cancelled, optional_string, optional_u64, parse_env_map, required_string,
    resolve_path,
};

/// How long a terminated process group gets to exit before it is killed.
#[cfg(unix)]
const TERM_GRACE: Duration = Duration::from_millis(500);
/// How long output readers may run after the process group is gone. Only a
/// descendant that escaped the group (e.g. via `setsid`) can hold pipes open.
const READER_DRAIN: Duration = Duration::from_secs(2);

/// Environment variables a spawned shell inherits: essentials only, never
/// service credentials. Matched ASCII-case-insensitively for Windows.
const INHERITED_ENV: &[&str] = &[
    "PATH",
    "HOME",
    "USER",
    "LOGNAME",
    "SHELL",
    "TERM",
    "COLORTERM",
    "LANG",
    "TZ",
    "TMPDIR",
    "TEMP",
    "TMP",
    "USERNAME",
    "APPDATA",
    "LOCALAPPDATA",
    "ProgramData",
    "SystemRoot",
    "WINDIR",
    "ComSpec",
    "PATHEXT",
    "NUMBER_OF_PROCESSORS",
];

fn inherits_env(key: &str) -> bool {
    INHERITED_ENV
        .iter()
        .any(|name| name.eq_ignore_ascii_case(key))
        || key.starts_with("LC_")
        || key.starts_with("XDG_")
}

/// This process's environment entries that spawned shells inherit.
pub(crate) fn inherited_env() -> impl Iterator<Item = (std::ffi::OsString, std::ffi::OsString)> {
    std::env::vars_os().filter(|(key, _)| key.to_str().is_some_and(inherits_env))
}

/// A one-shot `command` under the detected shell, with null stdin, a scrubbed
/// environment, and (on Unix) its own process group so it can be swept as a unit.
pub(crate) fn shell_command(command: &str) -> Command {
    let shell = detect::detected();
    let mut process = Command::new(&shell.path);
    process
        .args(shell.kind.argv(command))
        .env_clear()
        .envs(inherited_env())
        .stdin(Stdio::null())
        .kill_on_drop(true);
    #[cfg(unix)]
    process.process_group(0);
    process
}

#[derive(Debug)]
/// Built-in tool that runs each command in a fresh shell process.
pub struct ShellTool;

#[async_trait]
impl TypedTool for ShellTool {
    type Output = ShellOutput;

    fn describe(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("shell"),
            description: detect::detected().description(std::env::consts::OS),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Command line to run."
                    },
                    "cwd": {
                        "type": "string",
                        "description": "Directory to run in; relative paths resolve against the session working directory."
                    },
                    "env": {
                        "type": "object",
                        "additionalProperties": { "type": "string" },
                        "description": "Extra environment variables for this command."
                    },
                    "timeout_ms": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Kill the command after this many milliseconds."
                    }
                },
                "required": ["command"],
            }),
            concurrency: ToolConcurrency::Exclusive,
            capabilities: ToolCapabilities {
                mutating: true,
                requires_approval: true,
                cancellable: true,
                long_running: true,
            },
            provider_aliases: Default::default(),
        }
    }

    async fn run(&self, context: ToolContext, input: Value) -> anyhow::Result<Self::Output> {
        let _scope = ToolScope::new(&context, "shell");
        ensure_not_cancelled(&context.cancel)?;
        context.policy.check_shell_enabled().await?;
        // Session shutdown cancels this lifetime and with it any running command.
        let lifetime = context.tool_sessions.process_lifetime(&context.session_id);
        anyhow::ensure!(
            !lifetime.is_cancelled(),
            "failed to execute shell: session is closed"
        );
        let timeout = check_tool_timeout(
            context.policy.as_ref(),
            "shell",
            optional_u64(&input, "timeout_ms")?,
        )?
        .unwrap_or_else(|| Duration::from_secs(context.shell_timeout_secs.max(1)));
        let cwd = optional_string(&input, "cwd").map_or_else(
            || context.working_dir.clone(),
            |cwd| resolve_path(&context.working_dir, cwd),
        );
        let mut command = shell_command(required_string(&input, "command")?);
        command
            .current_dir(cwd)
            .envs(parse_env_map(input.get("env"))?.unwrap_or_default());
        run_command(
            command,
            context.emit.clone(),
            &context.cancel,
            &lifetime,
            timeout,
        )
        .await
    }
}

enum Outcome {
    Exited,
    Cancelled,
    TimedOut,
}

/// Run `command` to completion, cancellation, or timeout, streaming its output
/// as tool events. Whatever ends the command, its whole process group is swept
/// before returning, so nothing it started outlives the call.
async fn run_command(
    mut command: Command,
    emit: Arc<dyn ToolEventSink>,
    cancel: &CancellationToken,
    lifetime: &CancellationToken,
    timeout: Duration,
) -> anyhow::Result<ShellOutput> {
    let (stdout_reader, stdout_writer) = pipe_to_files("stdout")?;
    let (stderr_reader, stderr_writer) = pipe_to_files("stderr")?;
    let mut child = command
        .stdout(stdout_writer)
        .stderr(stderr_writer)
        .spawn()
        .map_err(|error| anyhow::anyhow!("failed to spawn shell: {error}"))?;
    // The parent's write ends live in `command`; readers see EOF only once they close.
    drop(command);
    let pid = child
        .id()
        .and_then(|pid| i32::try_from(pid).ok())
        .ok_or_else(|| anyhow::anyhow!("failed to spawn shell: missing child pid"))?;
    let guard = KillGroupOnDrop(Some(pid));
    let readers_cancel = CancellationToken::new();
    let stdout = tokio::spawn(collect_output(
        stdout_reader,
        emit.clone(),
        "shell",
        readers_cancel.clone(),
    ));
    let stderr = tokio::spawn(collect_output(
        stderr_reader,
        emit,
        "shell",
        readers_cancel.clone(),
    ));

    let outcome = tokio::select! {
        status = child.wait() => {
            status.map_err(|error| anyhow::anyhow!("failed to wait for shell: {error}"))?;
            Outcome::Exited
        }
        () = cancel.cancelled() => Outcome::Cancelled,
        () = lifetime.cancelled() => Outcome::Cancelled,
        () = tokio::time::sleep(timeout) => Outcome::TimedOut,
    };
    // Reap concurrently: an unreaped shell would keep the group alive as a zombie.
    let (_, status) = tokio::join!(terminate_group(pid), child.wait());
    guard.disarm();

    let readers = async { tokio::join!(join_output(stdout), join_output(stderr)) };
    tokio::pin!(readers);
    let (stdout, stderr) = tokio::select! {
        output = &mut readers => output,
        () = tokio::time::sleep(READER_DRAIN) => {
            readers_cancel.cancel();
            readers.await
        }
    };

    Ok(ShellOutput {
        exit_code: match outcome {
            Outcome::Exited => status.ok().and_then(exit_code),
            Outcome::Cancelled | Outcome::TimedOut => None,
        },
        stdout,
        stderr,
        timed_out: matches!(outcome, Outcome::TimedOut),
        cancelled: matches!(outcome, Outcome::Cancelled),
    })
}

async fn join_output(reader: JoinHandle<anyhow::Result<String>>) -> String {
    reader.await.ok().and_then(Result::ok).unwrap_or_default()
}

/// The exit code, or `128 + signal` for a signalled process as shells report it.
fn exit_code(status: ExitStatus) -> Option<i32> {
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        status
            .code()
            .or_else(|| status.signal().map(|signal| 128 + signal))
    }
    #[cfg(not(unix))]
    status.code()
}

/// Kills the group if the tool future is dropped before it swept the group itself.
struct KillGroupOnDrop(Option<i32>);

impl KillGroupOnDrop {
    fn disarm(mut self) {
        self.0 = None;
    }
}

impl Drop for KillGroupOnDrop {
    fn drop(&mut self) {
        if let Some(pid) = self.0 {
            #[cfg(unix)]
            crate::builtin::process::kill_process_group(pid, libc::SIGKILL);
            #[cfg(not(unix))]
            crate::builtin::process::kill_tree(pid, 9);
        }
    }
}

/// TERM the group, wait up to [`TERM_GRACE`] for it to empty, then KILL the rest.
/// Returns at once when the group is already gone.
#[cfg(unix)]
async fn terminate_group(pgid: i32) {
    use crate::builtin::process::kill_process_group;
    if !kill_process_group(pgid, libc::SIGTERM) {
        return;
    }
    let deadline = tokio::time::Instant::now() + TERM_GRACE;
    while kill_process_group(pgid, 0) && tokio::time::Instant::now() < deadline {
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    kill_process_group(pgid, libc::SIGKILL);
}

/// Without process groups, kill the shell's process tree outright.
#[cfg(not(unix))]
async fn terminate_group(pid: i32) {
    crate::builtin::process::kill_tree(pid, 9);
}

#[derive(Debug, Serialize)]
/// Result of a `shell` call.
pub struct ShellOutput {
    exit_code: Option<i32>,
    stdout: String,
    stderr: String,
    timed_out: bool,
    cancelled: bool,
}

impl ToolOutput for ShellOutput {
    /// Stdout verbatim, stderr under a `[stderr]` line, then a status trailer
    /// only when the command did not simply exit 0.
    fn to_model_string(&self) -> String {
        let status = match (self.timed_out, self.cancelled, self.exit_code) {
            (true, _, _) => "[timed out]".to_owned(),
            (_, true, _) => "[cancelled]".to_owned(),
            (_, _, Some(0)) => String::new(),
            (_, _, Some(code)) => format!("[exit code {code}]"),
            (_, _, None) => "[no exit code]".to_owned(),
        };
        let stderr = match self.stderr.as_str() {
            "" => String::new(),
            stderr => format!("[stderr]\n{stderr}"),
        };
        let text = [self.stdout.as_str(), &stderr, &status]
            .into_iter()
            .filter(|section| !section.is_empty())
            .fold(String::new(), |mut text, section| {
                if !(text.is_empty() || text.ends_with('\n')) {
                    text.push('\n');
                }
                text + section
            });
        if text.is_empty() {
            "[no output]".to_owned()
        } else {
            text
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::Tool;
    use halter_protocol::ToolResult;
    use serde_json::{Value, json};
    use tokio_util::sync::CancellationToken;

    use crate::{
        DefaultToolPolicy, NoopToolEventSink, PathLockMap, PolicySettings, ToolContext, ToolPolicy,
        ToolSessionStore,
    };

    use super::*;

    fn tool_context_with(root: &std::path::Path, settings: PolicySettings) -> ToolContext {
        let session_id = halter_protocol::SessionId::new();
        let tool_sessions = Arc::new(ToolSessionStore::default());
        tool_sessions.open_session(&session_id);
        ToolContext {
            session_id,
            working_dir: root.to_path_buf(),
            path_locks: Arc::new(PathLockMap::default()),
            tool_sessions,
            snapshot: Arc::new(halter_protocol::ResourceSnapshot::empty()),
            cancel: CancellationToken::new(),
            emit: Arc::new(NoopToolEventSink),
            policy: Arc::new(DefaultToolPolicy::new(settings)) as Arc<dyn ToolPolicy>,
            shell_timeout_secs: 30,
            subagent_parent: None,
        }
    }

    fn tool_context(root: &std::path::Path) -> ToolContext {
        tool_context_with(
            root,
            PolicySettings {
                allowed_write_roots: vec![root.to_path_buf()],
                ..PolicySettings::default()
            },
        )
    }

    async fn run(context: ToolContext, input: Value) -> anyhow::Result<Value> {
        ShellTool
            .execute(context, input)
            .await
            .map(|result| match result {
                ToolResult::Rendered { value, .. } => value,
                other => panic!("expected rendered result, got {other:?}"),
            })
    }

    #[tokio::test]
    async fn shell_runs_in_the_working_dir_or_a_relative_cwd() {
        // Asserted through a file the command creates, not `pwd` output,
        // which Git Bash prints as `/c/...` on Windows.
        let root = tempfile::tempdir().unwrap();
        std::fs::create_dir(root.path().join("nested")).unwrap();
        for (cwd, expected) in [
            (None, root.path().to_owned()),
            (Some("nested"), root.path().join("nested")),
        ] {
            let value = run(
                tool_context(root.path()),
                json!({"command": "echo here > marker", "cwd": cwd}),
            )
            .await
            .unwrap();
            assert_eq!(value["exit_code"], 0, "{value}");
            assert!(expected.join("marker").is_file(), "{cwd:?}");
        }
    }

    #[tokio::test]
    async fn each_call_is_a_fresh_shell() {
        let root = tempfile::tempdir().unwrap();
        let context = tool_context(root.path());
        std::fs::create_dir(root.path().join("nested")).unwrap();
        run(
            context.clone(),
            json!({"command": "cd nested; export HALTER_X=1"}),
        )
        .await
        .unwrap();
        let value = run(
            context,
            json!({"command": "test -d nested && printf 'at-root '; printf %s \"${HALTER_X:-unset}\""}),
        )
        .await
        .unwrap();
        assert_eq!(value["stdout"], "at-root unset");
    }

    #[tokio::test]
    async fn shell_reports_streams_exit_codes_and_env() {
        let root = tempfile::tempdir().unwrap();
        let value = run(
            tool_context(root.path()),
            json!({
                "command": "printf %s \"$GREETING\"; printf oops >&2; exit 3",
                "env": {"GREETING": "hi"}
            }),
        )
        .await
        .unwrap();
        assert_eq!(value["exit_code"], 3);
        assert_eq!(value["stdout"], "hi");
        assert_eq!(value["stderr"], "oops");
        assert_eq!(value["timed_out"], false);
        assert_eq!(value["cancelled"], false);
    }

    #[test]
    fn shell_does_not_inherit_credentials_from_the_host_environment() {
        assert!(inherits_env("PATH") && inherits_env("Path") && inherits_env("LC_CTYPE"));
        assert!(!inherits_env("OPENAI_API_KEY") && !inherits_env("GITHUB_TOKEN"));
        assert!(inherited_env().all(|(key, _)| inherits_env(key.to_str().unwrap())));
    }

    #[tokio::test]
    async fn shell_rejects_disabled_policy_closed_sessions_and_bad_input() {
        let root = tempfile::tempdir().unwrap();
        let disabled = tool_context_with(
            root.path(),
            PolicySettings {
                shell_enabled: false,
                ..PolicySettings::default()
            },
        );
        assert!(run(disabled, json!({"command": "true"})).await.is_err());

        let closed = tool_context(root.path());
        closed
            .tool_sessions
            .shutdown_session(&closed.session_id)
            .await
            .unwrap();
        let error = run(closed, json!({"command": "true"})).await.unwrap_err();
        assert!(error.to_string().contains("session is closed"), "{error}");

        for input in [
            json!({}),
            json!({"command": "true", "env": {"A": 1}}),
            json!({"command": "true", "timeout_ms": "soon"}),
        ] {
            assert!(run(tool_context(root.path()), input).await.is_err());
        }
    }

    #[cfg(unix)]
    /// Finding 8: a model-supplied `timeout_ms` had no ceiling, so a huge one
    /// left a command effectively unbounded. Rows run `shell` under a 1s
    /// ceiling: omitted and in-range values run; the ceiling itself runs;
    /// one past it is rejected before spawning, naming the maximum. Mutation
    /// killed: `check_tool_timeout` passing every value through (row
    /// "over the ceiling" runs instead of failing).
    #[tokio::test]
    async fn model_timeout_is_checked_against_the_policy_ceiling() {
        let root = tempfile::tempdir().unwrap();
        for (name, input, rejected) in [
            ("omitted", json!({"command": "true"}), false),
            (
                "in range",
                json!({"command": "true", "timeout_ms": 500}),
                false,
            ),
            (
                "at the ceiling",
                json!({"command": "true", "timeout_ms": 1_000}),
                false,
            ),
            (
                "over the ceiling",
                json!({"command": "true", "timeout_ms": 1_001}),
                true,
            ),
        ] {
            let context = tool_context_with(
                root.path(),
                PolicySettings {
                    allowed_write_roots: vec![root.path().to_path_buf()],
                    max_tool_timeout_secs: 1,
                    ..PolicySettings::default()
                },
            );
            match (run(context, input).await, rejected) {
                (Ok(value), false) => assert_eq!(value["exit_code"], 0, "{name}"),
                (Err(error), true) => assert!(
                    error
                        .to_string()
                        .contains("timeout_ms 1001 exceeds the maximum of 1000"),
                    "{name}: {error:#}"
                ),
                (other, _) => panic!("{name}: unexpected {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn timeout_returns_partial_output_and_kills_the_command() {
        let root = tempfile::tempdir().unwrap();
        let started = std::time::Instant::now();
        let value = run(
            tool_context(root.path()),
            json!({"command": "echo partial; sleep 30", "timeout_ms": 300}),
        )
        .await
        .unwrap();
        assert!(started.elapsed() < Duration::from_secs(5));
        assert_eq!(value["timed_out"], true);
        assert_eq!(value["exit_code"], Value::Null);
        assert_eq!(value["stdout"], "partial\n");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn background_children_are_killed_when_the_command_exits() {
        let root = tempfile::tempdir().unwrap();
        let started = std::time::Instant::now();
        let value = run(
            tool_context(root.path()),
            json!({"command": "sleep 30 & printf %s $! > pid; echo done"}),
        )
        .await
        .unwrap();
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "pipes were held open"
        );
        assert_eq!(value["stdout"], "done\n");
        let pid: i32 = std::fs::read_to_string(root.path().join("pid"))
            .unwrap()
            .parse()
            .unwrap();
        // SAFETY: signal zero only probes the process created by this test.
        assert_eq!(
            unsafe { libc::kill(pid, 0) },
            -1,
            "background child survived"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn cancellation_waits_for_process_group_cleanup() {
        for stop in ["cancel", "lifetime"] {
            let root = tempfile::tempdir().unwrap();
            let context = tool_context(root.path());
            let command_context = context.clone();
            let command = tokio::spawn(async move {
                run(
                    command_context,
                    json!({"command": "sh -c 'trap \"\" TERM; printf %s \"$$\" > pid; sleep 30'"}),
                )
                .await
            });
            let pid: i32 = tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    if let Ok(pid) = tokio::fs::read_to_string(root.path().join("pid")).await
                        && let Ok(pid) = pid.parse()
                    {
                        break pid;
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .expect("command publishes its pid");
            match stop {
                "cancel" => context.cancel.cancel(),
                _ => {
                    context
                        .tool_sessions
                        .request_stop_session(&context.session_id)
                        .await
                }
            }
            let value = tokio::time::timeout(Duration::from_secs(5), command)
                .await
                .unwrap()
                .unwrap()
                .unwrap();
            assert_eq!(value["cancelled"], true, "{stop}");
            // SAFETY: signal zero only probes the process created by this test.
            assert_eq!(
                unsafe { libc::kill(pid, 0) },
                -1,
                "{stop}: process survived"
            );
            assert_eq!(
                std::io::Error::last_os_error().raw_os_error(),
                Some(libc::ESRCH)
            );
        }
    }

    #[test]
    fn shell_output_renders_streams_verbatim_with_an_informative_trailer() {
        let output = |exit_code, stdout: &str, stderr: &str, timed_out, cancelled| ShellOutput {
            exit_code,
            stdout: stdout.to_owned(),
            stderr: stderr.to_owned(),
            timed_out,
            cancelled,
        };
        for (output, expected) in [
            (output(Some(0), "a \"b\"\n", "", false, false), "a \"b\"\n"),
            (
                output(Some(2), "out", "warn: é\n", false, false),
                "out\n[stderr]\nwarn: é\n[exit code 2]",
            ),
            (output(Some(0), "", "", false, false), "[no output]"),
            (
                output(None, "partial\n", "", true, false),
                "partial\n[timed out]",
            ),
            (output(None, "", "", false, true), "[cancelled]"),
            (output(None, "", "", false, false), "[no exit code]"),
        ] {
            assert_eq!(output.to_model_string(), expected);
        }
        assert_eq!(
            serde_json::to_value(output(Some(0), "o", "e", false, false)).expect("serialize"),
            json!({
                "exit_code": 0,
                "stdout": "o",
                "stderr": "e",
                "timed_out": false,
                "cancelled": false,
            })
        );
    }

    /// The `-EncodedCommand` path end to end, against the PowerShell every
    /// Windows host ships, whichever shell detection picked.
    #[cfg(windows)]
    #[test]
    fn powershell_runs_encoded_commands_verbatim() {
        let output = std::process::Command::new("powershell.exe")
            .args(detect::ShellKind::PowerShell.argv(
                "$x = 'a \"quoted\" é'; Write-Output \"$x|$($x.Length) $env:HALTER_PS\"; exit 4",
            ))
            .env("HALTER_PS", "env")
            .output()
            .unwrap();
        assert_eq!(output.status.code(), Some(4));
        // The length proves `é` survived decoding even if the console code
        // page mangles how it prints.
        let stdout = String::from_utf8_lossy(&output.stdout);
        assert!(stdout.starts_with("a \"quoted\" "), "{stdout}");
        assert!(stdout.trim_end().ends_with("|12 env"), "{stdout}");
    }

    #[cfg(unix)]
    #[test]
    fn exit_code_maps_signals_like_a_shell() {
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(exit_code(ExitStatus::from_raw(3 << 8)), Some(3));
        assert_eq!(exit_code(ExitStatus::from_raw(libc::SIGKILL)), Some(137));
    }
}
