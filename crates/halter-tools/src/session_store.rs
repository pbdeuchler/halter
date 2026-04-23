// pattern: Imperative Shell

use std::sync::Arc;

use dashmap::DashMap;
use halter_protocol::SessionId;
#[cfg(feature = "pty")]
use parking_lot::Mutex;
use tokio::sync::Mutex as TokioMutex;

#[cfg(feature = "browser-tools")]
use crate::builtin::browser::session::BrowserSession;
#[cfg(feature = "pty")]
use crate::builtin::pty::PtySessionHandle;
use crate::builtin::shell::session::ShellSessionCore;

#[derive(Default)]
pub struct ToolSessionStore {
    shell_sessions: DashMap<String, Arc<TokioMutex<Option<ShellSessionCore>>>>,
    #[cfg(feature = "pty")]
    pty_sessions: DashMap<String, Arc<Mutex<Option<PtySessionHandle>>>>,
    #[cfg(feature = "browser-tools")]
    browser_sessions: DashMap<String, Arc<TokioMutex<Option<BrowserSession>>>>,
}

impl ToolSessionStore {
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

    #[cfg(feature = "pty")]
    #[must_use]
    pub fn pty_session(&self, session_id: &SessionId) -> Arc<Mutex<Option<PtySessionHandle>>> {
        self.pty_sessions
            .entry(session_id.0.clone())
            .or_insert_with(|| Arc::new(Mutex::new(None)))
            .clone()
    }

    #[cfg(feature = "browser-tools")]
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
