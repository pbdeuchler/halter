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
///
/// The append-only event log is the source of truth; `state` is a checkpoint
/// of that log. `state_sequence` records the log position the checkpoint
/// reflects and `head_sequence` the highest committed sequence, so a loader
/// can reproduce the current state as
/// `fold(state, events in (state_sequence, head_sequence])` via
/// `halter_protocol::fold`.
pub struct StoredSession {
    pub blueprint: SessionBlueprint,
    pub state: SessionState,
    pub snapshot: Arc<ResourceSnapshot>,
    /// Highest event sequence the `state` checkpoint reflects. `0` for a
    /// freshly created session.
    pub state_sequence: u64,
    /// Highest committed event sequence at load time. `0` for a freshly
    /// created session; always `>= state_sequence`.
    pub head_sequence: u64,
}

impl StoredSession {
    /// Build a record for a session that has no committed events yet — the
    /// only shape [`SessionStore::create_session`] accepts.
    #[must_use]
    pub fn new(
        blueprint: SessionBlueprint,
        state: SessionState,
        snapshot: Arc<ResourceSnapshot>,
    ) -> Self {
        Self {
            blueprint,
            state,
            snapshot,
            state_sequence: 0,
            head_sequence: 0,
        }
    }
}

#[derive(Debug, Clone, Error)]
#[error(
    "failed to commit session '{session_id}': event log advanced concurrently \
     (expected head {expected_head_sequence}, found {actual_head_sequence})"
)]
/// Optimistic-concurrency failure raised when a commit expected a different
/// event-log head than the store currently holds.
pub struct SessionCommitConflict {
    pub session_id: SessionId,
    pub expected_head_sequence: u64,
    pub actual_head_sequence: u64,
}

#[async_trait]
/// Storage contract used by the session runtime.
pub trait SessionStore: Send + Sync {
    /// Create a new persisted session. Rejects records whose `state_sequence`
    /// or `head_sequence` is non-zero — new sessions start with an empty log.
    async fn create_session(&self, session: StoredSession) -> Result<()>;
    /// Load one session by id, including its checkpoint and head sequences.
    async fn load_session(&self, session_id: &SessionId) -> Result<Option<StoredSession>>;
    /// Atomically append events and update the snapshot/state checkpoint.
    ///
    /// When `expected_head_sequence` is supplied, the store must reject the
    /// commit with [`SessionCommitConflict`] unless the highest committed
    /// sequence equals it — a concurrent writer that appended anything since
    /// the caller loaded loses the race. Events receive gap-free monotonic
    /// sequences starting at the head + 1. When `state` is supplied it is
    /// written as a checkpoint stamped with the post-append head. All
    /// backends must implement the same semantics; the shared conformance
    /// suite in `tests/store_conformance.rs` locks this in.
    async fn commit(
        &self,
        session_id: &SessionId,
        snapshot: Option<Arc<ResourceSnapshot>>,
        expected_head_sequence: Option<u64>,
        state: Option<SessionState>,
        events: Vec<PendingEvent>,
    ) -> Result<Vec<SessionEvent>>;
    /// Replay committed events for a session in sequence order.
    async fn replay(&self, session_id: &SessionId) -> Result<Vec<SessionEvent>>;
    /// Replay committed events with sequence greater than `after_sequence`,
    /// in sequence order. Backends should override the default full-replay
    /// filter when they can push the bound into the query.
    async fn replay_after(
        &self,
        session_id: &SessionId,
        after_sequence: u64,
    ) -> Result<Vec<SessionEvent>> {
        Ok(self
            .replay(session_id)
            .await?
            .into_iter()
            .filter(|event| event.sequence() > after_sequence)
            .collect())
    }
    /// List stored session blueprints.
    async fn list_sessions(&self) -> Result<Vec<SessionBlueprint>>;

    /// Filesystem path to a persisted transcript, when the backend has one.
    fn transcript_path(&self, _session_id: &SessionId) -> Option<PathBuf> {
        None
    }
}
