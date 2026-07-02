//! Session persistence abstraction and built-in stores.
//!
//! The runtime persists immutable session blueprints, mutable session state,
//! resource snapshots, and committed event streams through [`SessionStore`].
//! The in-memory store is always available; the sqlite store is enabled by
//! the `sqlite` feature.
// pattern: Functional Core

mod memory;
#[cfg(feature = "sqlite")]
mod sqlite;

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use halter_protocol::{
    PendingEvent, ResourceSnapshot, SessionBlueprint, SessionEvent, SessionId, SessionState,
};
use thiserror::Error;

pub use memory::InMemorySessionStore;
#[cfg(feature = "sqlite")]
pub use sqlite::SqliteSessionStore;

#[derive(Debug, Clone)]
/// Persisted session record loaded by a [`SessionStore`].
pub struct StoredSession {
    pub blueprint: SessionBlueprint,
    pub state: SessionState,
    pub snapshot: Arc<ResourceSnapshot>,
}

#[derive(Debug, Clone, Error)]
#[error("failed to commit session '{session_id}': session state changed concurrently")]
/// Optimistic-concurrency failure raised when a commit sees stale state.
pub struct SessionCommitConflict {
    pub session_id: SessionId,
}

#[async_trait]
/// Storage contract used by the session runtime.
pub trait SessionStore: Send + Sync {
    /// Create a new persisted session.
    async fn create_session(&self, session: StoredSession) -> Result<()>;
    /// Load one session by id.
    async fn load_session(&self, session_id: &SessionId) -> Result<Option<StoredSession>>;
    /// Atomically update snapshot/state and commit pending events.
    ///
    /// When `expected_state` is supplied, the store must reject the commit with
    /// [`SessionCommitConflict`] if the currently persisted state differs.
    /// "Differs" is defined structurally — `SessionState`'s `PartialEq`, which
    /// is order-insensitive for its map fields — never by comparing serialized
    /// representations. All backends must implement the same semantics; the
    /// shared conformance suite in `tests/store_conformance.rs` locks this in.
    async fn commit(
        &self,
        session_id: &SessionId,
        snapshot: Option<Arc<ResourceSnapshot>>,
        expected_state: Option<SessionState>,
        state: Option<SessionState>,
        events: Vec<PendingEvent>,
    ) -> Result<Vec<SessionEvent>>;
    /// Replay committed events for a session in sequence order.
    async fn replay(&self, session_id: &SessionId) -> Result<Vec<SessionEvent>>;
    /// List stored session blueprints.
    async fn list_sessions(&self) -> Result<Vec<SessionBlueprint>>;

    /// Filesystem path to a persisted transcript, when the backend has one.
    fn transcript_path(&self, _session_id: &SessionId) -> Option<PathBuf> {
        None
    }
}
