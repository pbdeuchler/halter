//! Process management

use futures::FutureExt;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

use crate::{error, sys};

/// Tracks live, owned children independently of their execution futures.
/// Hosts can force-stop them without acquiring the shell's execution lock.
#[derive(Clone, Default)]
pub struct ProcessTracker {
    children: Arc<Mutex<HashMap<sys::process::ProcessId, Option<sys::process::ProcessId>>>>,
    tasks: Arc<AtomicUsize>,
    activity: Option<tokio::sync::watch::Sender<()>>,
}

impl ProcessTracker {
    /// Attach a coalescing notification channel for owned work changes.
    pub fn with_activity(activity: tokio::sync::watch::Sender<()>) -> Self {
        Self {
            activity: Some(activity),
            ..Self::default()
        }
    }

    /// Whether native children or asynchronous shell jobs remain active.
    pub fn has_running(&self) -> bool {
        self.tasks.load(Ordering::Acquire) != 0
            || !self
                .children
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .is_empty()
    }

    fn notify(&self) {
        if let Some(activity) = &self.activity {
            activity.send_replace(());
        }
    }

    pub(crate) fn start_task(&self) -> TrackedTask {
        self.tasks.fetch_add(1, Ordering::AcqRel);
        self.notify();
        TrackedTask(self.clone())
    }

    /// Immediately stop all currently owned children and process groups.
    pub fn force_stop(&self) {
        let children = self
            .children
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        for (&pid, &pgid) in children.iter() {
            force_stop_process(Some(pid), pgid);
        }
    }
}

pub(crate) struct TrackedTask(ProcessTracker);

impl Drop for TrackedTask {
    fn drop(&mut self) {
        self.0.tasks.fetch_sub(1, Ordering::AcqRel);
        self.0.notify();
    }
}

/// A waitable future that will yield the results of a child process's execution.
pub(crate) type WaitableChildProcess = std::pin::Pin<
    Box<dyn futures::Future<Output = Result<std::process::Output, std::io::Error>> + Send + Sync>,
>;

/// Tracks a child process being awaited.
pub struct ChildProcess {
    /// A waitable future that will yield the results of a child process's execution.
    exec_future: WaitableChildProcess,
    /// If available, the process ID of the child.
    pid: Option<sys::process::ProcessId>,
    /// If available, the process group ID of the child.
    pgid: Option<sys::process::ProcessId>,
    owned: bool,
    tracker: Option<ProcessTracker>,
}

impl ChildProcess {
    /// Wraps a child process and its future.
    pub fn new(
        child: sys::process::Child,
        pid: Option<sys::process::ProcessId>,
        pgid: Option<sys::process::ProcessId>,
    ) -> Self {
        Self {
            exec_future: Box::pin(child.wait_with_output()),
            pid,
            pgid,
            owned: true,
            tracker: None,
        }
    }

    pub(crate) fn track(&mut self, tracker: Option<ProcessTracker>) {
        if let (Some(pid), Some(tracker)) = (self.pid, tracker) {
            tracker
                .children
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .insert(pid, self.pgid);
            tracker.notify();
            let tracked = tracker.clone();
            let mut execution =
                std::mem::replace(&mut self.exec_future, Box::pin(std::future::pending()));
            self.exec_future = Box::pin(futures::future::poll_fn(move |cx| {
                // Serialize reaping with signalling, retiring the PID before
                // releasing the lock when the OS wait completes.
                let mut children = tracked
                    .children
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                let result = execution.as_mut().poll(cx);
                let removed = result.is_ready() && children.remove(&pid).is_some();
                drop(children);
                if removed {
                    tracked.notify();
                }
                result
            }));
            self.tracker = Some(tracker);
        }
    }

    fn retire(&mut self) {
        self.owned = false;
        if let (Some(pid), Some(tracker)) = (self.pid, self.tracker.take()) {
            let removed = tracker
                .children
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .remove(&pid);
            if removed.is_some() {
                tracker.notify();
            }
        }
    }

    /// Returns the process's ID.
    pub const fn pid(&self) -> Option<sys::process::ProcessId> {
        self.pid
    }

    /// Returns the process's group ID.
    pub const fn pgid(&self) -> Option<sys::process::ProcessId> {
        self.pgid
    }

    /// Waits for the process to exit.
    ///
    /// # Arguments
    ///
    /// * `cancel_token` - Optionally provides a cancellation token; if the token is
    ///   triggered before the process exits, the owned process is terminated
    ///   and reaped before
    ///   [`ProcessWaitResult::Cancelled`] is returned.
    pub async fn wait(
        &mut self,
        cancel_token: Option<CancellationToken>,
    ) -> Result<ProcessWaitResult, error::Error> {
        #[allow(unused_mut, reason = "only mutated on some platforms")]
        let mut sigtstp = sys::signal::tstp_signal_listener()?;
        #[allow(unused_mut, reason = "only mutated on some platforms")]
        let mut sigchld = sys::signal::chld_signal_listener()?;

        let cancelled = async {
            match cancel_token.as_ref() {
                Some(token) => token.cancelled().await,
                None => std::future::pending().await,
            }
        };
        tokio::pin!(cancelled);

        #[allow(clippy::ignored_unit_patterns)]
        loop {
            tokio::select! {
                output = &mut self.exec_future => {
                    self.retire();
                    break Ok(ProcessWaitResult::Completed(output?))
                },
                _ = &mut cancelled => {
                    self.cancel_and_reap().await?;
                    break Ok(ProcessWaitResult::Cancelled)
                },
                _ = sigtstp.recv() => {
                    break Ok(ProcessWaitResult::Stopped)
                },
                _ = sigchld.recv() => {
                    if sys::signal::poll_for_stopped_children()? {
                        break Ok(ProcessWaitResult::Stopped);
                    }
                },
                _ = sys::signal::await_ctrl_c() => {
                    // SIGINT got thrown. Handle it and continue looping. The child should
                    // have received it as well, and either handled it or ended up getting
                    // terminated (in which case we'll see the child exit).
                },
            }
        }
    }

    /// A cancelled wait must not abandon children. On Unix, terminate the
    /// owned pipeline group (never the harness's group), then force remaining
    /// members down after a grace period. Native Windows taskkill terminates
    /// the child tree without Unix grace semantics.
    async fn cancel_and_reap(&mut self) -> Result<(), error::Error> {
        #[cfg(unix)]
        // SAFETY: getpgrp has no arguments or memory ownership requirements.
        let harness_group = unsafe { libc::getpgrp() };
        #[cfg(unix)]
        if let Some(target) = self
            .pgid
            .filter(|pgid| *pgid > 0 && *pgid != harness_group)
            .map(|pgid| -pgid)
            .or(self.pid)
        {
            use nix::sys::signal::{Signal, kill};
            use nix::unistd::Pid;
            let signal = |signal| match kill(Pid::from_raw(target), signal) {
                Ok(()) => Ok(true),
                Err(nix::errno::Errno::ESRCH) => Ok(false),
                // Darwin can report EPERM after TERM has removed the group.
                // Confirm the group is empty or contains only zombies with
                // libproc; never suppress errors for executing processes.
                Err(nix::errno::Errno::EPERM) if target < 0 && process_group_is_gone(-target) => {
                    Ok(false)
                }
                Err(error) => Err(error),
            };
            if signal(Some(Signal::SIGTERM))? {
                let grace = tokio::time::sleep(std::time::Duration::from_millis(500));
                tokio::pin!(grace);
                let mut output = None;
                loop {
                    // Reap promptly, but preserve the group's grace while
                    // surviving descendants still exist. A leader exiting
                    // is not evidence that its process group has finished.
                    if output.is_some() && (target > 0 || !signal(None)?) {
                        break;
                    }
                    tokio::select! {
                        result = &mut self.exec_future, if output.is_none() => output = Some(result),
                        () = &mut grace => { signal(Some(Signal::SIGKILL))?; break; },
                        () = tokio::time::sleep(std::time::Duration::from_millis(25)), if output.is_some() => {},
                    }
                }
                if let Some(output) = output {
                    self.retire();
                    output?;
                    return Ok(());
                }
            }
        }
        #[cfg(windows)]
        if let Some(pid) = self.pid {
            let executable = std::env::var_os("SystemRoot")
                .map(std::path::PathBuf::from)
                .unwrap_or_else(|| std::path::PathBuf::from("C:\\Windows"))
                .join("System32")
                .join("taskkill.exe");
            let status = tokio::process::Command::new(executable)
                .args(["/PID", &pid.to_string(), "/T", "/F"])
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .await?;
            if !status.success() {
                tracing::warn!(pid, "native child-tree termination reported failure");
            }
        }
        (&mut self.exec_future).await?;
        self.retire();
        Ok(())
    }

    pub(crate) fn poll(&mut self) -> Option<Result<std::process::Output, error::Error>> {
        let checkable_future = &mut self.exec_future;
        let result = checkable_future
            .now_or_never()
            .map(|result| result.map_err(Into::into));
        if result.is_some() {
            self.retire();
        }
        result
    }
}

impl Drop for ChildProcess {
    fn drop(&mut self) {
        if self.owned {
            // Dropping an aborted execution future bypasses async TERM/reap
            // handling. Kill the owned group before dropping its child future.
            force_stop_process(self.pid, self.pgid);
            self.retire();
        }
    }
}

fn force_stop_process(pid: Option<sys::process::ProcessId>, pgid: Option<sys::process::ProcessId>) {
    #[cfg(unix)]
    {
        use nix::sys::signal::{Signal, kill};
        use nix::unistd::Pid;
        // SAFETY: getpgrp has no arguments or ownership requirements.
        let harness_group = unsafe { libc::getpgrp() };
        if let Some(target) = pgid
            .filter(|group| *group > 0 && *group != harness_group)
            .map(|group| -group)
            .or_else(|| pid.filter(|pid| *pid > 0))
        {
            let _ = kill(Pid::from_raw(target), Signal::SIGKILL);
        }
    }
    #[cfg(windows)]
    if let Some(pid) = pid {
        let executable = std::env::var_os("SystemRoot")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from("C:\\Windows"))
            .join("System32")
            .join("taskkill.exe");
        // Start native tree termination without blocking the executor/drop.
        let _ = std::process::Command::new(executable)
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn();
    }
    #[cfg(not(unix))]
    let _ = pgid;
    #[cfg(not(any(unix, windows)))]
    let _ = pid;
}

#[cfg(target_os = "macos")]
fn process_group_is_gone(group: i32) -> bool {
    #[link(name = "proc", kind = "dylib")]
    unsafe extern "C" {
        fn proc_listpids(
            kind: u32,
            typeinfo: u32,
            buffer: *mut std::ffi::c_void,
            buffersize: i32,
        ) -> i32;
    }
    const PROC_PGRP_ONLY: u32 = 2;
    let mut members = [0i32; 128];
    let Ok(buffer_size) = i32::try_from(size_of_val(&members)) else {
        return false;
    };
    // SAFETY: members is a live, correctly sized output buffer. proc_listpids
    // writes at most the exact byte size supplied.
    let bytes = unsafe {
        proc_listpids(
            PROC_PGRP_ONLY,
            group.cast_unsigned(),
            members.as_mut_ptr().cast(),
            buffer_size,
        )
    };
    if bytes == 0 {
        return true;
    }
    if bytes < 0 || bytes >= buffer_size {
        return false;
    }
    let Ok(bytes) = usize::try_from(bytes) else {
        return false;
    };
    let count = bytes / size_of::<i32>();
    members[..count].iter().all(|pid| {
        let mut info = std::mem::MaybeUninit::<libc::proc_bsdinfo>::uninit();
        let Ok(info_size) = i32::try_from(size_of::<libc::proc_bsdinfo>()) else {
            return false;
        };
        // SAFETY: info is a correctly aligned buffer for proc_bsdinfo; the
        // exact byte size is supplied. Only a complete write is assumed initialized.
        let read = unsafe {
            libc::proc_pidinfo(
                *pid,
                libc::PROC_PIDTBSDINFO,
                0,
                info.as_mut_ptr().cast(),
                info_size,
            )
        };
        if read == info_size {
            // SAFETY: proc_pidinfo wrote the complete proc_bsdinfo above.
            let info = unsafe { info.assume_init() };
            info.pbi_status == libc::SZOMB
        } else {
            // Darwin lists zombies in proc_listpids but proc_pidinfo returns
            // ESRCH for them. Permission failures remain distinguishable.
            std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
        }
    })
}

#[cfg(all(unix, not(target_os = "macos")))]
const fn process_group_is_gone(_group: i32) -> bool {
    false
}

/// Represents the result of waiting for an executing process.
pub enum ProcessWaitResult {
    /// The process completed.
    Completed(std::process::Output),
    /// The process stopped and has not yet completed.
    Stopped,
    /// The wait was abandoned because cancellation was requested.
    Cancelled,
}
