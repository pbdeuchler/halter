// pattern: Imperative Shell

use std::sync::Arc;

use dashmap::DashMap;
use halter_protocol::SessionId;
#[cfg(feature = "pty")]
use parking_lot::Mutex;
use tokio::sync::Mutex as TokioMutex;

#[cfg(feature = "pty")]
use crate::builtin::pty::PtySessionHandle;
use crate::builtin::shell::session::ShellSessionCore;

#[derive(Default)]
pub struct ToolSessionStore {
    shell_sessions: DashMap<String, Arc<TokioMutex<Option<ShellSessionCore>>>>,
    #[cfg(feature = "pty")]
    pty_sessions: DashMap<String, Arc<Mutex<Option<PtySessionHandle>>>>,
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
}
