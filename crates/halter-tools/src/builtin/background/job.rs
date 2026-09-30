// pattern: Imperative Shell

use std::collections::{HashMap, VecDeque};
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use serde::Serialize;
use serde_json::{Value, json};
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex as AsyncMutex, watch};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::CanonicalPath;
use crate::builtin::process::{kill_process_group, kill_tree};

const TERM_GRACE: Duration = Duration::from_millis(500);
const READER_DRAIN_TIMEOUT: Duration = Duration::from_secs(2);

/// Child ownership stays in the monitor task. Handles signal it and await
/// its completion, including output drains and child reaping.
pub(super) struct BackgroundJob {
    pub(super) id: String,
    command: String,
    cwd: std::path::PathBuf,
    pid: u32,
    started_at_ms: u128,
    output: Arc<Mutex<OutputBuffer>>,
    stop: CancellationToken,
    status: watch::Receiver<JobStatus>,
    task: AsyncMutex<Option<JoinHandle<()>>>,
}

#[derive(Clone, Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
enum JobStatus {
    Running,
    Exited {
        exit_code: Option<i32>,
        signal: Option<i32>,
    },
    Failed {
        error: String,
    },
}

struct OutputBuffer {
    bytes: VecDeque<u8>,
    end: u64,
    capacity: usize,
}

impl OutputBuffer {
    fn append(&mut self, bytes: &[u8]) {
        self.end = self.end.saturating_add(bytes.len() as u64);
        if bytes.len() >= self.capacity {
            self.bytes.clear();
            self.bytes.extend(&bytes[bytes.len() - self.capacity..]);
        } else {
            let excess = (self.bytes.len() + bytes.len()).saturating_sub(self.capacity);
            self.bytes.drain(..excess);
            self.bytes.extend(bytes);
        }
    }
}

impl BackgroundJob {
    pub(super) fn spawn(
        id: String,
        command: String,
        cwd: CanonicalPath,
        env: Option<HashMap<String, String>>,
        capacity: usize,
    ) -> anyhow::Result<Arc<Self>> {
        let cwd_path = cwd.path().to_owned();
        let mut process = shell_command(&command)?;
        process.current_dir(&cwd_path).env_clear();
        // Deliberately inherit only shell essentials, never service tokens or
        // startup-hook variables. Explicit environment entries are caller-owned.
        for key in [
            "PATH",
            "HOME",
            "USER",
            "LOGNAME",
            "TERM",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "TZ",
            "SystemRoot",
            "WINDIR",
            "ComSpec",
            "PATHEXT",
        ] {
            if let Some(value) = std::env::var_os(key) {
                process.env(key, value);
            }
        }
        if let Some(env) = env {
            process.envs(env);
        }
        process
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            // Open through the authorized parent fd. fchdir binds spawn to
            // this exact directory even if a symlink is changed meanwhile.
            let directory = cwd.open_read_blocking()?;
            anyhow::ensure!(
                directory.metadata()?.is_dir(),
                "failed to spawn background job: cwd is not a directory"
            );
            process.process_group(0);
            // SAFETY: fchdir is async-signal-safe. The owned fd lives in the
            // closure through spawn and no allocation/locking occurs here.
            unsafe {
                process.pre_exec(move || {
                    if libc::fchdir(directory.as_raw_fd()) == 0 {
                        Ok(())
                    } else {
                        Err(std::io::Error::last_os_error())
                    }
                });
            }
        }
        #[cfg(not(unix))]
        anyhow::ensure!(
            cwd_path.is_dir(),
            "failed to spawn background job: cwd is not a directory"
        );
        let mut child = process.spawn()?;
        let pid = child
            .id()
            .ok_or_else(|| anyhow::anyhow!("failed to spawn background job: missing child pid"))?;
        let output = Arc::new(Mutex::new(OutputBuffer {
            bytes: VecDeque::new(),
            end: 0,
            capacity,
        }));
        let stdout = child.stdout.take().expect("spawn configured piped stdout");
        let stderr = child.stderr.take().expect("spawn configured piped stderr");
        let stop = CancellationToken::new();
        let (status_tx, status) = watch::channel(JobStatus::Running);
        let task = tokio::spawn({
            let output = output.clone();
            let stop = stop.clone();
            async move {
                let stdout = tokio::spawn(capture(stdout, output.clone()));
                let stderr = tokio::spawn(capture(stderr, output));
                let outcome = monitor(&mut child, pid, &stop).await;
                let stdout_result = drain_reader(stdout).await;
                let stderr_result = drain_reader(stderr).await;
                let status = match outcome
                    .and_then(|status| stdout_result.and(stderr_result).map(|()| status))
                {
                    Ok(status) => status,
                    Err(error) => JobStatus::Failed {
                        error: error.to_string(),
                    },
                };
                status_tx.send_replace(status);
            }
        });
        Ok(Arc::new(Self {
            id,
            command,
            cwd: cwd_path,
            pid,
            started_at_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_millis(),
            output,
            stop,
            status,
            task: AsyncMutex::new(Some(task)),
        }))
    }

    pub(super) fn summary(&self) -> Value {
        json!({"id": self.id, "command": self.command, "cwd": self.cwd, "pid": self.pid, "started_at_ms": self.started_at_ms, "status": self.status.borrow().clone()})
    }

    pub(super) fn output(&self, cursor: u64) -> anyhow::Result<Value> {
        let buffer = self.output.lock();
        anyhow::ensure!(
            cursor <= buffer.end,
            "invalid tool input: output cursor is beyond available output"
        );
        let oldest = buffer.end.saturating_sub(buffer.bytes.len() as u64);
        let start = cursor.max(oldest);
        let bytes: Vec<_> = buffer
            .bytes
            .iter()
            .skip((start - oldest) as usize)
            .copied()
            .collect();
        Ok(
            json!({"id": self.id, "output": String::from_utf8_lossy(&bytes), "start_cursor": start, "next_cursor": buffer.end, "truncated": cursor < oldest, "status": self.status.borrow().clone()}),
        )
    }

    pub(super) fn request_stop(&self) {
        self.stop.cancel();
    }

    pub(super) async fn wait(&self) -> anyhow::Result<()> {
        let mut status = self.status.clone();
        while matches!(*status.borrow_and_update(), JobStatus::Running) {
            status.changed().await.map_err(|_| {
                anyhow::anyhow!("background process monitor stopped without recording completion")
            })?;
        }
        // Only one caller joins, while the mutex makes other waiters await
        // that join too. Thus every successful caller observes task completion.
        if let Some(task) = self.task.lock().await.take() {
            task.await
                .map_err(|error| anyhow::anyhow!("background process monitor failed: {error}"))?;
        }
        if let JobStatus::Failed { error } = &*status.borrow() {
            anyhow::bail!("background process failed: {error}");
        }
        Ok(())
    }
}

impl Drop for BackgroundJob {
    fn drop(&mut self) {
        // Fallback for callers dropping the store without explicit shutdown.
        // The monitor remains alive long enough to reap the process.
        self.stop.cancel();
    }
}

#[cfg(unix)]
fn shell_command(command: &str) -> anyhow::Result<Command> {
    let mut process = Command::new("/bin/sh");
    process.args(["-c", command]);
    Ok(process)
}

#[cfg(not(unix))]
fn shell_command(_command: &str) -> anyhow::Result<Command> {
    // Policy parses shell commands as Bash. cmd.exe uses different quoting
    // rules, so executing the same string there would bypass authorization.
    anyhow::bail!(
        "background spawn is unsupported on this platform: shell command policy requires a Unix shell"
    )
}

async fn capture(
    mut reader: impl AsyncRead + Unpin,
    output: Arc<Mutex<OutputBuffer>>,
) -> anyhow::Result<()> {
    let mut bytes = [0; 8192];
    loop {
        let count = reader.read(&mut bytes).await?;
        if count == 0 {
            return Ok(());
        }
        output.lock().append(&bytes[..count]);
    }
}

async fn drain_reader(mut reader: JoinHandle<anyhow::Result<()>>) -> anyhow::Result<()> {
    match tokio::time::timeout(READER_DRAIN_TIMEOUT, &mut reader).await {
        Ok(joined) => {
            joined.map_err(|error| anyhow::anyhow!("background output reader failed: {error}"))?
        }
        Err(_) => {
            reader.abort();
            let _ = reader.await;
            anyhow::bail!("background output pipe remained open after process cleanup");
        }
    }
}

async fn monitor(
    child: &mut Child,
    pid: u32,
    stop: &CancellationToken,
) -> anyhow::Result<JobStatus> {
    let outcome = tokio::select! {
        result = child.wait() => result,
        _ = stop.cancelled() => {
            signal_job(pid, 15);
            // Give every member of the owned process group a TERM grace
            // period, even if the group leader exits sooner.
            tokio::time::sleep(TERM_GRACE).await;
            signal_job(pid, 9);
            child.wait().await
        }
    };
    // A command using '&' can leave descendants behind after its shell exits.
    // Those are still this job's resources, never implicitly detached jobs.
    #[cfg(unix)]
    if kill_process_group(pid as i32, 15) {
        tokio::time::sleep(TERM_GRACE).await;
        kill_process_group(pid as i32, 9);
    }
    #[cfg(not(unix))]
    kill_tree(pid as i32, 9);
    let status = outcome?;
    #[cfg(unix)]
    let signal = {
        use std::os::unix::process::ExitStatusExt;
        status.signal()
    };
    #[cfg(not(unix))]
    let signal = None;
    Ok(JobStatus::Exited {
        exit_code: status.code(),
        signal,
    })
}

fn signal_job(pid: u32, signal: i32) {
    #[cfg(unix)]
    kill_process_group(pid as i32, signal);
    // Also catch currently discoverable descendants that changed process
    // groups. Daemons escaping ancestry/session ownership remain unsupported.
    kill_tree(pid as i32, signal);
}

#[cfg(all(test, not(unix)))]
mod tests {
    use super::*;

    #[test]
    fn spawn_rejects_shells_with_unvalidated_command_grammar() {
        assert!(shell_command("echo 'safe & del sensitive.txt & echo tail'").is_err());
    }
}
