// pattern: Imperative Shell

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Context;
use async_trait::async_trait;
use halter_protocol::{SessionBlueprint, SessionEvent, SessionId, SessionState};
use tokio::sync::RwLock;
use tracing::debug;

use crate::{SessionStore, StoredSession};

#[derive(Debug, Default)]
struct MemoryStoreState {
    sessions: HashMap<String, StoredSession>,
    events: HashMap<String, Vec<SessionEvent>>,
}

#[derive(Debug, Default, Clone)]
pub struct InMemorySessionStore {
    state: Arc<RwLock<MemoryStoreState>>,
}

#[async_trait]
impl SessionStore for InMemorySessionStore {
    async fn create_session(&self, session: StoredSession) -> anyhow::Result<()> {
        let session_id = session.blueprint.session_id.0.clone();
        debug!(session_id = %session_id, "creating in-memory session");
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
        state: Option<SessionState>,
        mut events: Vec<SessionEvent>,
    ) -> anyhow::Result<Vec<SessionEvent>> {
        debug!(
            session_id = %session_id,
            event_count = events.len(),
            replace_snapshot = snapshot.is_some(),
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

        if let Some(snapshot) = snapshot {
            session.blueprint.snapshot_revision = snapshot.revision.clone();
            session.snapshot = snapshot;
        }

        if let Some(state) = state {
            session.state = state;
        }

        let existing = store.events.entry(session_id.0.clone()).or_default();
        let base_sequence = existing.len() as u64;
        for (index, event) in events.iter_mut().enumerate() {
            event.sequence = base_sequence + index as u64 + 1;
            event.session_id = session_id.clone();
        }
        existing.extend(events.iter().cloned());
        Ok(events)
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

    use halter_protocol::{ModelId, ResourceSnapshot, Revision, SessionBlueprint};

    use super::*;

    #[tokio::test]
    async fn memory_store_roundtrips_session() {
        let store = InMemorySessionStore::default();
        let blueprint = SessionBlueprint {
            session_id: SessionId::new(),
            parent_session_id: None,
            default_model: ModelId::from("default"),
            subagent_model: ModelId::from("subagent"),
            snapshot_revision: Revision::from("revision-1"),
            working_dir: PathBuf::from("."),
            system_prompt_seed: Vec::new(),
            max_turns: None,
            // max_tool_calls_per_turn: 8,
            subagent_depth: 0,
        };

        store
            .create_session(StoredSession {
                blueprint: blueprint.clone(),
                state: SessionState::default(),
                snapshot: Arc::new(ResourceSnapshot::empty()),
            })
            .await
            .expect("create session");

        let loaded = store
            .load_session(&blueprint.session_id)
            .await
            .expect("load session")
            .expect("session exists");

        assert_eq!(loaded.blueprint, blueprint);
    }
}
