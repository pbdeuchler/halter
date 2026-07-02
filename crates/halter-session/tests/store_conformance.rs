//! Shared conformance suite for every `SessionStore` backend.
//!
//! Each check runs against all available backends so commit/conflict
//! semantics cannot silently diverge again (the sqlite store previously
//! compared serialized JSON while the memory store compared structurally,
//! which disagreed on `IndexMap` ordering).

use std::path::PathBuf;
use std::sync::Arc;

use halter_protocol::{
    Delivery, ModelId, PendingEvent, PendingToolCall, ResourceSnapshot, SessionBlueprint,
    SessionEvent, SessionEventPayload, SessionId, SessionState, SubagentEventForwarding, Timestamp,
    ToolCall, ToolCallId, ToolName,
};
use halter_session::{InMemorySessionStore, SessionCommitConflict, SessionStore, StoredSession};

fn backends() -> Vec<(&'static str, Box<dyn SessionStore>)> {
    #[cfg_attr(not(feature = "sqlite"), allow(unused_mut))]
    let mut backends: Vec<(&'static str, Box<dyn SessionStore>)> =
        vec![("memory", Box::new(InMemorySessionStore::default()))];
    #[cfg(feature = "sqlite")]
    backends.push((
        "sqlite",
        Box::new(halter_session::SqliteSessionStore::open(":memory:").expect("open sqlite store")),
    ));
    backends
}

#[tokio::test]
async fn commit_succeeds_with_matching_expected_state() {
    for (backend, store) in backends() {
        let session = test_session("matching");
        let original_state = session.state.clone();
        store.create_session(session.clone()).await.expect(backend);

        let updated_state = state_with_summary("updated");
        store
            .commit(
                &session.blueprint.session_id,
                None,
                Some(original_state),
                Some(updated_state.clone()),
                Vec::new(),
            )
            .await
            .unwrap_or_else(|error| panic!("{backend}: {error}"));

        let reloaded = store
            .load_session(&session.blueprint.session_id)
            .await
            .expect(backend)
            .expect(backend);
        assert_eq!(reloaded.state, updated_state, "{backend}");
    }
}

#[tokio::test]
async fn commit_rejects_stale_expected_state() {
    for (backend, store) in backends() {
        let session = test_session("stale");
        let original_state = session.state.clone();
        store.create_session(session.clone()).await.expect(backend);

        store
            .commit(
                &session.blueprint.session_id,
                None,
                Some(original_state.clone()),
                Some(state_with_summary("first-writer")),
                Vec::new(),
            )
            .await
            .unwrap_or_else(|error| panic!("{backend}: {error}"));

        let error = store
            .commit(
                &session.blueprint.session_id,
                None,
                Some(original_state),
                Some(state_with_summary("second-writer")),
                Vec::new(),
            )
            .await
            .expect_err("stale commit must fail");
        assert!(
            error.downcast_ref::<SessionCommitConflict>().is_some(),
            "{backend}: expected SessionCommitConflict, got {error:#}"
        );

        // The losing writer must not have clobbered the state.
        let reloaded = store
            .load_session(&session.blueprint.session_id)
            .await
            .expect(backend)
            .expect(backend);
        assert_eq!(
            reloaded.state,
            state_with_summary("first-writer"),
            "{backend}"
        );
    }
}

/// The case the two backends previously disagreed on: two logically equal
/// states whose `IndexMap` fields serialize in different key orders must not
/// be treated as a conflict by any backend.
#[tokio::test]
async fn commit_treats_map_key_order_as_equal() {
    let submitted_at = fixed_timestamp();
    for (backend, store) in backends() {
        let mut stored_state = SessionState::default();
        for name in ["alpha", "beta"] {
            insert_pending_call(&mut stored_state, name, submitted_at);
        }
        let mut expected_state = SessionState::default();
        for name in ["beta", "alpha"] {
            insert_pending_call(&mut expected_state, name, submitted_at);
        }
        assert_eq!(stored_state, expected_state, "premise: structurally equal");
        assert_ne!(
            serde_json::to_string(&stored_state).expect("serialize"),
            serde_json::to_string(&expected_state).expect("serialize"),
            "premise: serialization order differs"
        );

        let session = StoredSession {
            state: stored_state,
            ..test_session("map-order")
        };
        store.create_session(session.clone()).await.expect(backend);

        store
            .commit(
                &session.blueprint.session_id,
                None,
                Some(expected_state),
                Some(state_with_summary("after-map-order")),
                Vec::new(),
            )
            .await
            .unwrap_or_else(|error| {
                panic!("{backend}: reordered-but-equal state must not conflict: {error:#}")
            });
    }
}

#[tokio::test]
async fn commit_without_expected_state_skips_conflict_detection() {
    for (backend, store) in backends() {
        let session = test_session("unchecked");
        store.create_session(session.clone()).await.expect(backend);

        // Two commits with no expected state both succeed regardless of the
        // interleaving writer.
        for summary in ["one", "two"] {
            store
                .commit(
                    &session.blueprint.session_id,
                    None,
                    None,
                    Some(state_with_summary(summary)),
                    Vec::new(),
                )
                .await
                .unwrap_or_else(|error| panic!("{backend}: {error}"));
        }
    }
}

#[tokio::test]
async fn commit_rejects_unknown_session() {
    for (backend, store) in backends() {
        let error = store
            .commit(
                &SessionId::from("missing-session"),
                None,
                None,
                None,
                vec![test_event("orphan")],
            )
            .await
            .expect_err("commit against unknown session must fail");
        assert!(
            error.to_string().contains("unknown session"),
            "{backend}: unexpected error: {error:#}"
        );
    }
}

#[tokio::test]
async fn commit_assigns_gap_free_sequences_across_commits() {
    for (backend, store) in backends() {
        let session = test_session("sequences");
        store.create_session(session.clone()).await.expect(backend);

        let first = store
            .commit(
                &session.blueprint.session_id,
                None,
                None,
                None,
                vec![test_event("one"), test_event("two")],
            )
            .await
            .expect(backend);
        let second = store
            .commit(
                &session.blueprint.session_id,
                None,
                None,
                None,
                vec![test_event("three")],
            )
            .await
            .expect(backend);

        let sequences: Vec<u64> = first
            .iter()
            .chain(second.iter())
            .map(SessionEvent::sequence)
            .collect();
        assert_eq!(sequences, vec![1, 2, 3], "{backend}");

        let replayed: Vec<u64> = store
            .replay(&session.blueprint.session_id)
            .await
            .expect(backend)
            .iter()
            .map(SessionEvent::sequence)
            .collect();
        assert_eq!(replayed, sequences, "{backend}");
    }
}

fn test_session(name: &str) -> StoredSession {
    let snapshot = Arc::new(ResourceSnapshot::empty());
    let blueprint = SessionBlueprint {
        session_id: SessionId::from(format!("session-{name}")),
        parent_session_id: None,
        default_model: ModelId::from("default"),
        subagent_model: ModelId::from("subagent"),
        subagent_event_forwarding: SubagentEventForwarding::Off,
        snapshot_revision: snapshot.revision.clone(),
        working_dir: PathBuf::from("."),
        system_prompt_seed: Vec::new(),
        max_turns: None,
        subagent_depth: 0,
    };
    StoredSession {
        blueprint,
        state: SessionState::default(),
        snapshot,
    }
}

fn state_with_summary(text: &str) -> SessionState {
    SessionState {
        summaries: vec![halter_protocol::SummarySlice {
            id: format!("summary-{text}"),
            text: text.to_owned(),
        }],
        ..SessionState::default()
    }
}

fn insert_pending_call(state: &mut SessionState, name: &str, submitted_at: Timestamp) {
    state.pending_tool_calls.insert(
        ToolCallId::from(format!("call-{name}")),
        PendingToolCall {
            call: ToolCall {
                id: ToolCallId::from(format!("call-{name}")),
                name: ToolName::from(name),
                arguments: serde_json::json!({}),
            },
            submitted_at,
        },
    );
}

fn fixed_timestamp() -> Timestamp {
    serde_json::from_value(serde_json::json!("2026-01-01T00:00:00Z")).expect("parse timestamp")
}

fn test_event(summary: &str) -> PendingEvent {
    PendingEvent::new(
        SessionId::from("overwritten-by-store"),
        Delivery::Lossless,
        SessionEventPayload::ContextCompacted {
            summary: summary.to_owned(),
        },
    )
}
