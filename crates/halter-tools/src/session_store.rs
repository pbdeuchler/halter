// pattern: Imperative Shell

use std::sync::Arc;

use dashmap::DashMap;
use halter_protocol::SessionId;
use parking_lot::Mutex;
#[cfg(feature = "browser-tools")]
use tokio::sync::Mutex as TokioMutex;
use tokio_util::sync::CancellationToken;

use crate::builtin::background::BackgroundRegistry;
#[cfg(feature = "browser-tools")]
use crate::builtin::browser::session::BrowserSession;
#[cfg(feature = "pty")]
use crate::builtin::pty::PtySessionHandle;
use crate::builtin::shell::session::ShellSession;
use crate::builtin::task::TaskList;

#[derive(Default)]
/// Per-session storage for stateful tools.
pub struct ToolSessionStore {
    shell_sessions: DashMap<String, Arc<ShellSession>>,
    process_lifetimes: DashMap<String, CancellationToken>,
    task_sessions: DashMap<String, Arc<Mutex<TaskList>>>,
    background_sessions: DashMap<String, Arc<BackgroundRegistry>>,
    #[cfg(feature = "pty")]
    pty_sessions: DashMap<String, Arc<Mutex<Option<PtySessionHandle>>>>,
    #[cfg(feature = "browser-tools")]
    browser_sessions: DashMap<String, Arc<TokioMutex<Option<BrowserSession>>>>,
}

impl ToolSessionStore {
    pub(crate) fn process_lifetime(&self, session_id: &SessionId) -> CancellationToken {
        self.process_lifetimes
            .entry(session_id.0.clone())
            .or_default()
            .clone()
    }

    /// Close process admission and signal resources without waiting for reaping.
    pub async fn request_stop_session(&self, session_id: &SessionId) {
        self.stop_session_resources(session_id, false).await;
    }

    /// Force-stop owned processes while retaining their cleanup slots.
    pub async fn force_stop_session(&self, session_id: &SessionId) {
        self.stop_session_resources(session_id, true).await;
    }

    async fn stop_session_resources(&self, session_id: &SessionId, force: bool) {
        if force {
            // A cleanup deadline can fire after some resources have already
            // settled. Never recreate closed slots that would poison reopen.
            if let Some(lifetime) = self.process_lifetimes.get(&session_id.0) {
                lifetime.cancel();
            }
            let jobs = self
                .background_sessions
                .get(&session_id.0)
                .map(|entry| entry.clone());
            if let Some(jobs) = jobs {
                jobs.request_stop(true).await;
            }
        } else {
            self.process_lifetime(session_id).cancel();
            self.background_session(session_id)
                .request_stop(false)
                .await;
        }
        if let Some(shell) = self.shell_sessions.get(&session_id.0) {
            shell.request_stop(force);
        }
        #[cfg(feature = "pty")]
        if let Some(pty) = self.pty_sessions.get(&session_id.0) {
            crate::builtin::pty::request_stop_session(&pty);
        }
    }

    pub async fn request_stop_all(&self) {
        for id in self.process_session_ids() {
            self.request_stop_session(&id).await;
        }
    }

    pub async fn force_stop_all(&self) {
        for id in self.process_session_ids() {
            self.force_stop_session(&id).await;
        }
    }

    fn process_session_ids(&self) -> Vec<SessionId> {
        let mut ids = std::collections::HashSet::new();
        ids.extend(
            self.process_lifetimes
                .iter()
                .map(|entry| entry.key().clone()),
        );
        ids.extend(
            self.background_sessions
                .iter()
                .map(|entry| entry.key().clone()),
        );
        ids.extend(self.shell_sessions.iter().map(|entry| entry.key().clone()));
        #[cfg(feature = "pty")]
        ids.extend(self.pty_sessions.iter().map(|entry| entry.key().clone()));
        #[cfg(feature = "browser-tools")]
        ids.extend(
            self.browser_sessions
                .iter()
                .map(|entry| entry.key().clone()),
        );
        ids.into_iter().map(SessionId::from).collect()
    }
    pub(crate) fn background_session(&self, session_id: &SessionId) -> Arc<BackgroundRegistry> {
        self.background_sessions
            .entry(session_id.0.clone())
            .or_insert_with(|| Arc::new(BackgroundRegistry::new(self.process_lifetime(session_id))))
            .clone()
    }

    /// Terminate and await live tool resources after agent execution has
    /// settled. Runtime admission must remain closed during this operation.
    /// Task lists are retained; process resources receive fresh slots on reopen.
    pub async fn shutdown_session(&self, session_id: &SessionId) -> anyhow::Result<()> {
        self.request_stop_session(session_id).await;
        let mut errors = Vec::new();
        let jobs = self
            .background_sessions
            .get(&session_id.0)
            .map(|entry| entry.clone());
        if let Some(jobs) = jobs
            && let Err(error) = jobs.shutdown().await
        {
            errors.push(error.to_string());
        }
        let shell = self
            .shell_sessions
            .get(&session_id.0)
            .map(|entry| entry.clone());
        if let Some(shell) = shell
            && let Err(error) = crate::builtin::shell::session::shutdown_shell_session(&shell).await
        {
            errors.push(error.to_string());
        }
        #[cfg(feature = "pty")]
        let pty = self
            .pty_sessions
            .get(&session_id.0)
            .map(|entry| entry.clone());
        #[cfg(feature = "pty")]
        if let Some(pty) = pty
            && let Err(error) = crate::builtin::pty::stop_session(&pty).await
        {
            errors.push(error.to_string());
        }
        #[cfg(feature = "browser-tools")]
        if let Some(browser) = self
            .browser_sessions
            .get(&session_id.0)
            .map(|entry| entry.clone())
        {
            let browser = browser.lock().await.take();
            if let Some(browser) = browser
                && let Err(error) = browser.close().await
            {
                errors.push(error.to_string());
            }
        }
        self.background_sessions.remove(&session_id.0);
        self.shell_sessions.remove(&session_id.0);
        #[cfg(feature = "pty")]
        self.pty_sessions.remove(&session_id.0);
        #[cfg(feature = "browser-tools")]
        self.browser_sessions.remove(&session_id.0);
        self.process_lifetimes.remove(&session_id.0);
        anyhow::ensure!(
            errors.is_empty(),
            "failed to shut down session tool resources: {}",
            errors.join("; ")
        );
        Ok(())
    }

    /// Return the persistent shell session slot for a halter session.
    #[must_use]
    pub fn shell_session(&self, session_id: &SessionId) -> Arc<ShellSession> {
        self.shell_sessions
            .entry(session_id.0.clone())
            .or_insert_with(|| Arc::new(ShellSession::new(self.process_lifetime(session_id))))
            .clone()
    }

    /// Returns the in-memory task list bound to this session, creating it on
    /// first access. Storage is process-local: the session log is the durable
    /// record, and the runtime rebuilds the list from it on resume via
    /// [`TaskList::from_results`] and [`Self::restore_task_session`].
    #[must_use]
    pub fn task_session(&self, session_id: &SessionId) -> Arc<Mutex<TaskList>> {
        self.task_sessions
            .entry(session_id.0.clone())
            .or_insert_with(|| Arc::new(Mutex::new(TaskList::default())))
            .clone()
    }

    /// Whether this process holds a shell, background, pty or browser session for
    /// `session_id`. Slots are created on first use, so `false` means that
    /// state, if the session ever had it, died with an earlier process.
    #[must_use]
    pub fn has_process_state(&self, session_id: &SessionId) -> bool {
        let held = self.shell_sessions.contains_key(&session_id.0)
            || self.background_sessions.contains_key(&session_id.0);
        #[cfg(feature = "pty")]
        let held = held || self.pty_sessions.contains_key(&session_id.0);
        #[cfg(feature = "browser-tools")]
        let held = held || self.browser_sessions.contains_key(&session_id.0);
        held
    }

    /// Install `list` as this session's task list unless the process already
    /// holds one, which is newer than anything rebuilt from the log.
    pub fn restore_task_session(&self, session_id: &SessionId, list: TaskList) {
        self.task_sessions
            .entry(session_id.0.clone())
            .or_insert_with(|| Arc::new(Mutex::new(list)));
    }

    #[cfg(feature = "pty")]
    /// Return the PTY session slot for a halter session.
    #[must_use]
    pub fn pty_session(&self, session_id: &SessionId) -> Arc<Mutex<Option<PtySessionHandle>>> {
        self.pty_sessions
            .entry(session_id.0.clone())
            .or_insert_with(|| Arc::new(Mutex::new(None)))
            .clone()
    }

    #[cfg(feature = "browser-tools")]
    /// Return the browser session slot for a halter session.
    #[must_use]
    pub fn browser_session(
        &self,
        session_id: &SessionId,
    ) -> Arc<TokioMutex<Option<BrowserSession>>> {
        self.browser_sessions
            .entry(session_id.0.clone())
            .or_insert_with(|| Arc::new(TokioMutex::new(None)))
            .clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn process_state_is_held_once_a_stateful_slot_exists() {
        let session = SessionId::from("held");
        // (slot first used, whether that counts as process state)
        #[allow(unused_mut)]
        let mut cases = vec![("none", false), ("task", false), ("shell", true)];
        #[cfg(feature = "pty")]
        cases.push(("pty", true));
        #[cfg(feature = "browser-tools")]
        cases.push(("browser", true));
        for (slot, held) in cases {
            let store = ToolSessionStore::default();
            match slot {
                "task" => drop(store.task_session(&session)),
                "shell" => drop(store.shell_session(&session)),
                #[cfg(feature = "pty")]
                "pty" => drop(store.pty_session(&session)),
                #[cfg(feature = "browser-tools")]
                "browser" => drop(store.browser_session(&session)),
                _ => {}
            }
            assert_eq!(store.has_process_state(&session), held, "{slot}");
            assert!(
                !store.has_process_state(&SessionId::from("other")),
                "{slot}: other sessions hold nothing"
            );
        }
    }
}
