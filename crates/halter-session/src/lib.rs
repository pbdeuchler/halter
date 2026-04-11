// pattern: Functional Core

mod memory;
#[cfg(feature = "sqlite")]
mod sqlite;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use halter_protocol::{ResourceSnapshot, SessionBlueprint, SessionEvent, SessionId, SessionState};

pub use memory::InMemorySessionStore;
#[cfg(feature = "sqlite")]
pub use sqlite::SqliteSessionStore;

#[derive(Debug, Clone)]
pub struct StoredSession {
    pub blueprint: SessionBlueprint,
    pub state: SessionState,
    pub snapshot: Arc<ResourceSnapshot>,
}

#[async_trait]
pub trait SessionStore: Send + Sync {
    async fn create_session(&self, session: StoredSession) -> Result<()>;
    async fn load_session(&self, session_id: &SessionId) -> Result<Option<StoredSession>>;
    async fn commit(
        &self,
        session_id: &SessionId,
        snapshot: Option<Arc<ResourceSnapshot>>,
        state: Option<SessionState>,
        events: Vec<SessionEvent>,
    ) -> Result<Vec<SessionEvent>>;
    async fn replay(&self, session_id: &SessionId) -> Result<Vec<SessionEvent>>;
    async fn list_sessions(&self) -> Result<Vec<SessionBlueprint>>;

    fn transcript_path(&self, _session_id: &SessionId) -> Option<PathBuf> {
        None
    }
}
