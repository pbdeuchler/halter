// pattern: Imperative Shell

use std::sync::Arc;

use dashmap::DashMap;
use halter_protocol::SessionId;
use parking_lot::Mutex;
use tokio::sync::Mutex as TokioMutex;

#[cfg(feature = "browser-tools")]
use crate::builtin::browser::session::BrowserSession;
#[cfg(feature = "pty")]
use crate::builtin::pty::PtySessionHandle;
use crate::builtin::shell::session::ShellSessionCore;
use crate::builtin::task::TaskList;

#[derive(Default)]
/// Per-session storage for stateful tools.
pub struct ToolSessionStore {
    shell_sessions: DashMap<String, Arc<TokioMutex<Option<ShellSessionCore>>>>,
    task_sessions: DashMap<String, Arc<Mutex<TaskList>>>,
    #[cfg(feature = "pty")]
    pty_sessions: DashMap<String, Arc<Mutex<Option<PtySessionHandle>>>>,
    #[cfg(feature = "browser-tools")]
    browser_sessions: DashMap<String, Arc<TokioMutex<Option<BrowserSession>>>>,
}

impl ToolSessionStore {
    /// Return the persistent shell session slot for a halter session.
    #[must_use]
    pub fn shell_session(
        &self,
        session_id: &SessionId,
    ) -> Arc<TokioMutex<Option<ShellSessionCore>>> {
        self.shell_sessions
            .entry(session_id.0.clone())
            .or_insert_with(|| Arc::new(TokioMutex::new(None)))
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
