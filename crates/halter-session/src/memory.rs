// pattern: Imperative Shell

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Context;
use async_trait::async_trait;
use halter_protocol::{PendingEvent, SessionBlueprint, SessionEvent, SessionId, SessionState};
use tokio::sync::RwLock;
use tracing::debug;

use crate::{SessionCommitConflict, SessionStore, StoredSession};

#[derive(Debug, Default)]
struct MemoryStoreState {
    sessions: HashMap<String, StoredSession>,
    events: HashMap<String, Vec<SessionEvent>>,
}

#[derive(Debug, Default, Clone)]
/// In-process session store used by default and in tests.
pub struct InMemorySessionStore {
    state: Arc<RwLock<MemoryStoreState>>,
}

#[async_trait]
impl SessionStore for InMemorySessionStore {
    async fn create_session(&self, session: StoredSession) -> anyhow::Result<()> {
        let session_id = session.blueprint.session_id.0.clone();
        debug!(session_id = %session_id, "creating in-memory session");
        if session.state_sequence != 0 || session.head_sequence != 0 {
            anyhow::bail!(
                "failed to create session '{}': new sessions must start at sequence 0 \
                 (state_sequence {}, head_sequence {})",
                session_id,
                session.state_sequence,
                session.head_sequence
            );
        }
        let mut state = self.state.write().await;
        if state.sessions.contains_key(&session_id) {
            anyhow::bail!(
                "failed to create session: session '{}' already exists",
                session_id
            );
        }
        state.sessions.insert(session_id, session);
        Ok(())
    }

    async fn load_session(&self, session_id: &SessionId) -> anyhow::Result<Option<StoredSession>> {
        let state = self.state.read().await;
        let loaded = state.sessions.get(&session_id.0).cloned();
        debug!(session_id = %session_id, found = loaded.is_some(), "loading in-memory session");
        Ok(loaded)
    }

    async fn commit(
        &self,
        session_id: &SessionId,
        snapshot: Option<Arc<halter_protocol::ResourceSnapshot>>,
        expected_head_sequence: Option<u64>,
        state: Option<SessionState>,
        events: Vec<PendingEvent>,
    ) -> anyhow::Result<Vec<SessionEvent>> {
        debug!(
            session_id = %session_id,
            event_count = events.len(),
            replace_snapshot = snapshot.is_some(),
            expected_head_sequence = ?expected_head_sequence,
            replace_state = state.is_some(),
            "committing in-memory session state"
        );
        let mut store = self.state.write().await;
        let session = store.sessions.get_mut(&session_id.0).with_context(|| {
            format!(
                "failed to commit session: unknown session '{}'",
                session_id.0
            )
        })?;

        // The session record tracks the head directly (rather than deriving
        // it from the events map) so gap-free monotonicity survives even if
        // a future feature prunes committed events from the in-memory store.
        let current_head = session.head_sequence;
        if let Some(expected) = expected_head_sequence
            && expected != current_head
        {
            return Err(SessionCommitConflict {
                session_id: session_id.clone(),
                expected_head_sequence: expected,
                actual_head_sequence: current_head,
            }
            .into());
        }

        if let Some(snapshot) = snapshot {
            session.blueprint.snapshot_revision = snapshot.revision.clone();
            session.snapshot = snapshot;
        }

        let committed: Vec<SessionEvent> = events
            .into_iter()
            .enumerate()
            .map(|(offset, pending)| {
                PendingEvent {
                    session_id: session_id.clone(),
                    ..pending
                }
                .into_committed(current_head + 1 + offset as u64)
            })
            .collect();
        let new_head = committed
            .last()
            .map_or(current_head, SessionEvent::sequence);
        session.head_sequence = new_head;

        if let Some(state) = state {
            session.state = state;
            session.state_sequence = new_head;
        }

        store
            .events
            .entry(session_id.0.clone())
            .or_default()
            .extend(committed.iter().cloned());
        Ok(committed)
    }

    async fn replay(&self, session_id: &SessionId) -> anyhow::Result<Vec<SessionEvent>> {
        let state = self.state.read().await;
        let events = state.events.get(&session_id.0).cloned().unwrap_or_default();
        debug!(session_id = %session_id, event_count = events.len(), "replaying in-memory session");
        Ok(events)
    }

    async fn list_sessions(&self) -> anyhow::Result<Vec<SessionBlueprint>> {
        let state = self.state.read().await;
        let sessions = state
            .sessions
            .values()
            .map(|session| session.blueprint.clone())
            .collect::<Vec<_>>();
        debug!(session_count = sessions.len(), "listing in-memory sessions");
        Ok(sessions)
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use halter_protocol::{
        ModelId, ResourceSnapshot, Revision, SessionBlueprint, SubagentEventForwarding,
    };

    use super::*;

    #[tokio::test]
    async fn memory_store_roundtrips_session() {
        let store = InMemorySessionStore::default();
        let blueprint = SessionBlueprint {
            session_id: SessionId::new(),
            parent_session_id: None,
            default_model: ModelId::from("default"),
            subagent_model: ModelId::from("subagent"),
            subagent_event_forwarding: SubagentEventForwarding::Off,
            snapshot_revision: Revision::from("revision-1"),
            working_dir: PathBuf::from("."),
            system_prompt_seed: Vec::new(),
            max_turns: None,
            subagent_depth: 0,
        };

        store
            .create_session(StoredSession::new(
                blueprint.clone(),
                SessionState::default(),
                Arc::new(ResourceSnapshot::empty()),
            ))
            .await
            .expect("create session");

        let loaded = store
            .load_session(&blueprint.session_id)
            .await
            .expect("load session")
            .expect("session exists");

        assert_eq!(loaded.blueprint, blueprint);
        assert_eq!(loaded.state_sequence, 0);
        assert_eq!(loaded.head_sequence, 0);
    }

    #[tokio::test]
    async fn memory_store_rejects_nonzero_sequences_on_create() {
        let store = InMemorySessionStore::default();
        let blueprint = SessionBlueprint {
            session_id: SessionId::new(),
            parent_session_id: None,
            default_model: ModelId::from("default"),
            subagent_model: ModelId::from("subagent"),
            subagent_event_forwarding: SubagentEventForwarding::Off,
            snapshot_revision: Revision::from("revision-1"),
            working_dir: PathBuf::from("."),
            system_prompt_seed: Vec::new(),
            max_turns: None,
            subagent_depth: 0,
        };
        let mut session = StoredSession::new(
            blueprint,
            SessionState::default(),
            Arc::new(ResourceSnapshot::empty()),
        );
        session.head_sequence = 3;

        let error = store
            .create_session(session)
            .await
            .expect_err("nonzero sequences must be rejected");
        assert!(
            error.to_string().contains("must start at sequence 0"),
            "unexpected error: {error}"
        );
    }

    #[tokio::test]
    async fn memory_store_rejects_stale_head_commit() {
        let store = InMemorySessionStore::default();
        let blueprint = SessionBlueprint {
            session_id: SessionId::new(),
            parent_session_id: None,
            default_model: ModelId::from("default"),
            subagent_model: ModelId::from("subagent"),
            subagent_event_forwarding: SubagentEventForwarding::Off,
            snapshot_revision: Revision::from("revision-1"),
            working_dir: PathBuf::from("."),
            system_prompt_seed: Vec::new(),
            max_turns: None,
            subagent_depth: 0,
        };

        store
            .create_session(StoredSession::new(
                blueprint.clone(),
                SessionState::default(),
                Arc::new(ResourceSnapshot::empty()),
            ))
            .await
            .expect("create session");

        let updated_state = SessionState {
            pending_warning_messages: vec![halter_protocol::HookWarning {
                category: "test".to_owned(),
                message: "first".to_owned(),
                ..halter_protocol::HookWarning::default()
            }],
            ..SessionState::default()
        };
        store
            .commit(
                &blueprint.session_id,
                None,
                Some(0),
                Some(updated_state.clone()),
                vec![PendingEvent::new(
                    blueprint.session_id.clone(),
                    halter_protocol::Delivery::Lossless,
                    halter_protocol::SessionEventPayload::SessionStarted,
                )],
            )
            .await
            .expect("commit updated state");

        let error = store
            .commit(
                &blueprint.session_id,
                None,
                Some(0),
                Some(SessionState::default()),
                Vec::new(),
            )
            .await
            .expect_err("stale commit should fail");
        let conflict = error
            .downcast_ref::<SessionCommitConflict>()
            .expect("conflict error");
        assert_eq!(conflict.expected_head_sequence, 0);
        assert_eq!(conflict.actual_head_sequence, 1);

        let reloaded = store
            .load_session(&blueprint.session_id)
            .await
            .expect("load session")
            .expect("session exists");
        assert_eq!(reloaded.state, updated_state);
        assert_eq!(reloaded.state_sequence, 1);
        assert_eq!(reloaded.head_sequence, 1);
    }
}
