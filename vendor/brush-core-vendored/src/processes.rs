//! Process management

use futures::FutureExt;
use tokio_util::sync::CancellationToken;

use crate::{error, sys};

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
                Ok(()) | Err(nix::errno::Errno::ESRCH) => Ok(()),
                // Darwin can report EPERM after TERM has removed the group.
                // Confirm the group is empty or contains only zombies with
                // libproc; never suppress errors for executing processes.
                Err(nix::errno::Errno::EPERM) if target < 0 && process_group_is_gone(-target) => {
                    Ok(())
                }
                Err(error) => Err(error),
            };
            signal(Signal::SIGTERM)?;
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            signal(Signal::SIGKILL)?;
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
        Ok(())
    }

    pub(crate) fn poll(&mut self) -> Option<Result<std::process::Output, error::Error>> {
        let checkable_future = &mut self.exec_future;
        checkable_future
            .now_or_never()
            .map(|result| result.map_err(Into::into))
    }
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
fn process_group_is_gone(_group: i32) -> bool {
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
