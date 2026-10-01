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
    activity: tokio::sync::watch::Sender<()>,
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
    /// Observe changes to live tool work. Subscribe before inspecting jobs to
    /// avoid missing the final completion between inspection and waiting.
    pub fn subscribe_activity(&self) -> tokio::sync::watch::Receiver<()> {
        self.activity.subscribe()
    }

    #[cfg(feature = "pty")]
    pub(crate) fn activity_sender(&self) -> tokio::sync::watch::Sender<()> {
        self.activity.clone()
    }

    /// Whether native processes or shell background tasks still own work.
    /// Empty persistent tool slots and completed job records do not count.
    pub fn has_running_jobs(&self, session_id: &SessionId) -> bool {
        if self
            .background_sessions
            .get(&session_id.0)
            .is_some_and(|jobs| jobs.has_running())
            || self
                .shell_sessions
                .get(&session_id.0)
                .is_some_and(|shell| shell.has_running())
        {
            return true;
        }
        #[cfg(feature = "pty")]
        if self.pty_sessions.get(&session_id.0).is_some_and(|pty| {
            pty.lock()
                .as_ref()
                .is_some_and(PtySessionHandle::is_running)
        }) {
            return true;
        }
        false
    }
    /// Admit tools for a fresh incarnation after the previous owner has settled.
    pub fn open_session(&self, session_id: &SessionId) {
        let mut lifetime = self
            .process_lifetimes
            .entry(session_id.0.clone())
            .or_default();
        if lifetime.is_cancelled() {
            *lifetime = CancellationToken::new();
        }
    }

    #[cfg(any(test, feature = "pty", feature = "browser-tools"))]
    pub(crate) fn process_lifetime(&self, session_id: &SessionId) -> CancellationToken {
        self.process_lifetimes
            .get(&session_id.0)
            .map_or_else(Self::closed_lifetime, |lifetime| lifetime.clone())
    }

    fn closed_lifetime() -> CancellationToken {
        let closed = CancellationToken::new();
        closed.cancel();
        closed
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
        // Resource getters retain a shared admission guard through insertion.
        // Exclusive cancellation fences those inserts before cleanup scans.
        if let Some(lifetime) = self.process_lifetimes.get_mut(&session_id.0) {
            lifetime.cancel();
        }
        let jobs = self
            .background_sessions
            .get(&session_id.0)
            .map(|entry| entry.clone());
        if let Some(jobs) = jobs {
            jobs.request_stop(force).await;
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
        let Some(lifetime) = self
            .process_lifetimes
            .get(&session_id.0)
            .filter(|lifetime| !lifetime.is_cancelled())
        else {
            return Arc::new(BackgroundRegistry::new(
                Self::closed_lifetime(),
                self.activity.clone(),
            ));
        };
        let registry = self
            .background_sessions
            .entry(session_id.0.clone())
            .or_insert_with(|| {
                let registry = Arc::new(BackgroundRegistry::new(
                    lifetime.clone(),
                    self.activity.clone(),
                ));
                self.activity.send_replace(());
                registry
            })
            .clone();
        drop(lifetime);
        registry
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
        let Some(lifetime) = self
            .process_lifetimes
            .get(&session_id.0)
            .filter(|lifetime| !lifetime.is_cancelled())
        else {
            return Arc::new(ShellSession::new(
                Self::closed_lifetime(),
                self.activity.clone(),
            ));
        };
        let shell = self
            .shell_sessions
            .entry(session_id.0.clone())
            .or_insert_with(|| {
                let shell = Arc::new(ShellSession::new(lifetime.clone(), self.activity.clone()));
                self.activity.send_replace(());
                shell
            })
            .clone();
        drop(lifetime);
        shell
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
        let Some(lifetime) = self
            .process_lifetimes
            .get(&session_id.0)
            .filter(|lifetime| !lifetime.is_cancelled())
        else {
            return Arc::new(Mutex::new(None));
        };
        let pty = self
            .pty_sessions
            .entry(session_id.0.clone())
            .or_insert_with(|| {
                self.activity.send_replace(());
                Arc::new(Mutex::new(None))
            })
            .clone();
        drop(lifetime);
        pty
    }

    #[cfg(feature = "browser-tools")]
    /// Return the browser session slot for a halter session.
    #[must_use]
    pub fn browser_session(
        &self,
        session_id: &SessionId,
    ) -> Arc<TokioMutex<Option<BrowserSession>>> {
        let Some(lifetime) = self
            .process_lifetimes
            .get(&session_id.0)
            .filter(|lifetime| !lifetime.is_cancelled())
        else {
            return Arc::new(TokioMutex::new(None));
        };
        let browser = self
            .browser_sessions
            .entry(session_id.0.clone())
            .or_insert_with(|| {
                self.activity.send_replace(());
                Arc::new(TokioMutex::new(None))
            })
            .clone();
        drop(lifetime);
        browser
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opening_existing_admission_preserves_its_live_lifetime() {
        let store = ToolSessionStore::default();
        let session = SessionId::from("live-child");
        store.open_session(&session);
        let lifetime = store.process_lifetime(&session);
        store.open_session(&session);
        lifetime.cancel();
        assert!(store.process_lifetime(&session).is_cancelled());
    }

    #[tokio::test]
    async fn repeated_stop_does_not_recreate_resources_and_reopen_restores_admission() {
        let store = ToolSessionStore::default();
        let session = SessionId::from("reopen");
        store.open_session(&session);
        let original = store.process_lifetime(&session);
        store.shutdown_session(&session).await.unwrap();
        for force in [false, true, false] {
            store.stop_session_resources(&session, force).await;
            assert!(!store.has_process_state(&session));
            assert!(!store.process_lifetimes.contains_key(&session.0));
        }
        assert!(original.is_cancelled());
        assert!(
            store.process_lifetime(&session).is_cancelled(),
            "late resource creation remains closed"
        );
        store.open_session(&session);
        assert!(!store.process_lifetime(&session).is_cancelled());
    }

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
            store.open_session(&session);
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
                !store.has_running_jobs(&session),
                "{slot}: an empty slot holds no work"
            );
            assert!(
                !store.has_process_state(&SessionId::from("other")),
                "{slot}: other sessions hold nothing"
            );
        }
    }

    #[tokio::test]
    async fn closed_session_ids_and_stale_getters_leave_no_process_entries() {
        let store = ToolSessionStore::default();
        for index in 0..128 {
            let id = SessionId::from(format!("closed-{index}"));
            store.open_session(&id);
            drop(store.shell_session(&id));
            drop(store.background_session(&id));
            #[cfg(feature = "pty")]
            drop(store.pty_session(&id));
            #[cfg(feature = "browser-tools")]
            drop(store.browser_session(&id));
            store.shutdown_session(&id).await.unwrap();
            for force in [false, true] {
                store.stop_session_resources(&id, force).await;
            }
            assert!(store.process_lifetime(&id).is_cancelled());
            drop(store.shell_session(&id));
            drop(store.background_session(&id));
            #[cfg(feature = "pty")]
            drop(store.pty_session(&id));
            #[cfg(feature = "browser-tools")]
            drop(store.browser_session(&id));
            assert!(
                !store.has_process_state(&id),
                "late getters must not retain closed slots"
            );
        }
        assert!(store.process_lifetimes.is_empty());
        assert!(store.shell_sessions.is_empty());
        assert!(store.background_sessions.is_empty());
        #[cfg(feature = "pty")]
        assert!(store.pty_sessions.is_empty());
        #[cfg(feature = "browser-tools")]
        assert!(store.browser_sessions.is_empty());
    }
}
