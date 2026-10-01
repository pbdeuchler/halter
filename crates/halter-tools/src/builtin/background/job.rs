// pattern: Imperative Shell

use std::collections::{HashMap, VecDeque};
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
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
use crate::builtin::process::{kill_process_group, list_descendants, signal_process};

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
    force: CancellationToken,
    owned_process: Arc<Mutex<Option<OwnedProcessIds>>>,
    status: watch::Receiver<JobStatus>,
    task: AsyncMutex<Option<JoinHandle<()>>>,
}

struct OwnedProcessIds {
    pid: Option<u32>,
    group: u32,
}

struct OwnedProcessGuard(Arc<Mutex<Option<OwnedProcessIds>>>);

// One lease covers process settlement and both output drains. Its Drop also
// publishes completion if the monitor panics, without a separate watcher task.
struct RunningJob {
    count: Arc<AtomicUsize>,
    activity: watch::Sender<()>,
}

impl RunningJob {
    fn new(count: Arc<AtomicUsize>, activity: watch::Sender<()>) -> Self {
        count.fetch_add(1, Ordering::AcqRel);
        activity.send_replace(());
        Self { count, activity }
    }
}

impl Drop for RunningJob {
    fn drop(&mut self) {
        self.count.fetch_sub(1, Ordering::AcqRel);
        self.activity.send_replace(());
    }
}

impl Drop for OwnedProcessGuard {
    fn drop(&mut self) {
        if let Some(owned) = self.0.lock().take() {
            signal_owned(&owned, 9);
        }
    }
}

fn signal_owned(owned: &OwnedProcessIds, signal: i32) {
    #[cfg(unix)]
    kill_process_group(owned.group as i32, signal);
    if let Some(pid) = owned.pid {
        signal_process(pid as i32, signal);
    }
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
        running: Arc<AtomicUsize>,
        activity: watch::Sender<()>,
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
        let force = CancellationToken::new();
        let (status_tx, status) = watch::channel(JobStatus::Running);
        let owned_process = Arc::new(Mutex::new(Some(OwnedProcessIds {
            pid: Some(pid),
            group: pid,
        })));
        let ownership = OwnedProcessGuard(owned_process.clone());
        let running = RunningJob::new(running, activity);
        let task = tokio::spawn({
            let output = output.clone();
            let stop = stop.clone();
            let force = force.clone();
            let owned_process = owned_process.clone();
            async move {
                let _running = running;
                // A local child drops before the activity lease on unwind,
                // after the group guard has signalled owned descendants.
                let mut child = child;
                let _ownership = ownership;
                let stdout = tokio::spawn(capture(stdout, output.clone()));
                let stderr = tokio::spawn(capture(stderr, output));
                let outcome = monitor(&mut child, pid, &stop, &force, &owned_process).await;
                // Disable signalling before output drains; a completed PID
                // must never be targeted after the OS reuses it.
                owned_process.lock().take();
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
            force,
            owned_process,
            status,
            task: AsyncMutex::new(Some(task)),
        }))
    }

    pub(super) fn summary(&self) -> Value {
        json!({"id": self.id, "command": self.command, "cwd": self.cwd, "pid": self.pid, "started_at_ms": self.started_at_ms, "status": self.status.borrow().clone()})
    }

    pub(super) fn is_finished(&self) -> bool {
        !matches!(*self.status.borrow(), JobStatus::Running)
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

    pub(super) fn force_stop(&self) {
        self.force.cancel();
        self.stop.cancel();
        // A native cleanup scan may own the PID lock. Never make the actor
        // wait for that scan; the monitor observes the force token afterward.
        if let Some(control) = self.owned_process.try_lock()
            && let Some(owned) = control.as_ref()
        {
            signal_owned(owned, 9);
        }
    }

    pub(super) async fn wait(&self) -> anyhow::Result<()> {
        let mut status = self.status.clone();
        let mut monitor_closed = false;
        while matches!(*status.borrow_and_update(), JobStatus::Running) {
            if status.changed().await.is_err() {
                monitor_closed = true;
                break;
            }
        }
        // Only one caller joins, while the mutex makes other waiters await
        // that join too. Thus every successful caller observes task completion.
        let mut task = self.task.lock().await;
        let joined = match task.as_mut() {
            Some(task) => Some(task.await),
            None => None,
        };
        task.take();
        if let Some(joined) = joined {
            joined
                .map_err(|error| anyhow::anyhow!("background process monitor failed: {error}"))?;
        }
        anyhow::ensure!(
            !monitor_closed,
            "background process monitor stopped without recording completion"
        );
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
    force: &CancellationToken,
    owned: &Arc<Mutex<Option<OwnedProcessIds>>>,
) -> anyhow::Result<JobStatus> {
    let outcome = tokio::select! {
        result = wait_owned_child(child, owned) => result,
        _ = stop.cancelled() => {
            signal_job(owned.clone(), if force.is_cancelled() { 9 } else { 15 }).await?;
            // Give every member of the owned process group a TERM grace
            // period, even if the group leader exits sooner.
            tokio::select! {
                _ = tokio::time::sleep(TERM_GRACE), if !force.is_cancelled() => {},
                _ = force.cancelled() => {},
            }
            signal_job(owned.clone(), 9).await?;
            wait_owned_child(child, owned).await
        }
    };
    // A command using '&' can leave descendants behind after its shell exits.
    // Those are still this job's resources, never implicitly detached jobs.
    #[cfg(unix)]
    if kill_process_group(pid as i32, 15) {
        tokio::select! {
            _ = tokio::time::sleep(TERM_GRACE) => {},
            _ = force.cancelled() => {},
        }
        kill_process_group(pid as i32, 9);
    }
    #[cfg(not(unix))]
    signal_process(pid as i32, 9);
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

async fn wait_owned_child(
    child: &mut Child,
    owned: &Mutex<Option<OwnedProcessIds>>,
) -> std::io::Result<std::process::ExitStatus> {
    let wait = child.wait();
    tokio::pin!(wait);
    std::future::poll_fn(|cx| {
        // Reaping and external force-stop use the same lock. A reaped PID
        // is retired before another caller could signal its replacement.
        let mut control = owned.lock();
        let result = std::future::Future::poll(wait.as_mut(), cx);
        if result.is_ready()
            && let Some(owned) = control.as_mut()
        {
            owned.pid = None;
        }
        result
    })
    .await
}

async fn signal_job(owned: Arc<Mutex<Option<OwnedProcessIds>>>, signal: i32) -> anyhow::Result<()> {
    // The monitor owns and joins this native operation before reaping. Stop
    // requests only set tokens, avoiding /proc scans on the actor thread.
    tokio::task::spawn_blocking(move || {
        let root = owned.lock().as_ref().and_then(|control| control.pid);
        let descendants = root
            .map(|pid| list_descendants(pid as i32))
            .unwrap_or_default();
        // Scanning holds no shared lock. Verify the root is still owned before
        // signalling the snapshot; reaping and signalling share this short lock.
        let control = owned.lock();
        if let Some(control) = control.as_ref() {
            if control.pid == root && root.is_some() {
                for pid in descendants.into_iter().rev() {
                    signal_process(pid, signal);
                }
            }
            signal_owned(control, signal);
        }
    })
    .await
    .map_err(|error| anyhow::anyhow!("background process termination failed: {error}"))
}

#[cfg(test)]
mod ownership_tests {
    use super::*;

    #[tokio::test]
    async fn missing_status_still_joins_monitor_and_retains_cancelled_waiter() {
        let (send_status, status) = watch::channel(JobStatus::Running);
        drop(send_status);
        let (finish, finishing) = tokio::sync::oneshot::channel();
        let settled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let task = tokio::spawn({
            let settled = settled.clone();
            async move {
                finishing.await.unwrap();
                settled.store(true, Ordering::Release);
            }
        });
        let job = BackgroundJob {
            id: "failed-monitor".to_owned(),
            command: "test".to_owned(),
            cwd: std::path::PathBuf::new(),
            pid: 0,
            started_at_ms: 0,
            output: Arc::new(Mutex::new(OutputBuffer {
                bytes: VecDeque::new(),
                end: 0,
                capacity: 1,
            })),
            stop: CancellationToken::new(),
            force: CancellationToken::new(),
            owned_process: Arc::new(Mutex::new(None)),
            status,
            task: AsyncMutex::new(Some(task)),
        };
        assert!(
            tokio::time::timeout(Duration::from_millis(25), job.wait())
                .await
                .is_err(),
            "status closure must not abandon a settling monitor"
        );
        assert!(!settled.load(Ordering::Acquire));
        finish.send(()).unwrap();
        let error = job.wait().await.unwrap_err();
        assert!(error.to_string().contains("without recording completion"));
        assert!(settled.load(Ordering::Acquire));
    }
}

#[cfg(all(test, not(unix)))]
mod tests {
    use super::*;

    #[test]
    fn spawn_rejects_shells_with_unvalidated_command_grammar() {
        assert!(shell_command("echo 'safe & del sensitive.txt & echo tail'").is_err());
    }
}
