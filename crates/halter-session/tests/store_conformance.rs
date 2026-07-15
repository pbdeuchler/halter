//! Shared conformance suite for every `SessionStore` backend.
//!
//! Each check runs against all available backends so commit/conflict
//! semantics cannot silently diverge. The suite also locks in the
//! log/checkpoint invariant: the state checkpoint a store returns must agree
//! with folding the committed event log (`halter_protocol::fold`) on every
//! fold-covered field.

use std::path::PathBuf;
use std::sync::Arc;

use halter_protocol::{
    CompactionEventEffects, Delivery, Message, ModelId, PendingEvent, ResourceSnapshot,
    SessionBlueprint, SessionEvent, SessionEventPayload, SessionId, SessionState,
    SubagentEventForwarding, UserMessage, fold,
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
async fn commit_succeeds_with_matching_expected_head() {
    for (backend, store) in backends() {
        let session = test_session("matching");
        store.create_session(session.clone()).await.expect(backend);

        let updated_state = state_with_summary("updated");
        store
            .commit(
                &session.blueprint.session_id,
                None,
                Some(0),
                Some(updated_state.clone()),
                vec![test_event("advance")],
            )
            .await
            .unwrap_or_else(|error| panic!("{backend}: {error}"));

        let reloaded = store
            .load_session(&session.blueprint.session_id)
            .await
            .expect(backend)
            .expect(backend);
        assert_eq!(reloaded.state, updated_state, "{backend}");
        assert_eq!(reloaded.state_sequence, 1, "{backend}");
        assert_eq!(reloaded.head_sequence, 1, "{backend}");
    }
}

#[tokio::test]
async fn commit_rejects_stale_expected_head() {
    for (backend, store) in backends() {
        let session = test_session("stale");
        store.create_session(session.clone()).await.expect(backend);

        store
            .commit(
                &session.blueprint.session_id,
                None,
                Some(0),
                Some(state_with_summary("first-writer")),
                vec![test_event("first")],
            )
            .await
            .unwrap_or_else(|error| panic!("{backend}: {error}"));

        let error = store
            .commit(
                &session.blueprint.session_id,
                None,
                Some(0),
                Some(state_with_summary("second-writer")),
                vec![test_event("second")],
            )
            .await
            .expect_err("stale commit must fail");
        let conflict = error
            .downcast_ref::<SessionCommitConflict>()
            .unwrap_or_else(|| panic!("{backend}: expected SessionCommitConflict, got {error:#}"));
        assert_eq!(conflict.expected_head_sequence, 0, "{backend}");
        assert_eq!(conflict.actual_head_sequence, 1, "{backend}");

        // The losing writer must not have clobbered the state or the log.
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
        assert_eq!(reloaded.head_sequence, 1, "{backend}");
        let replayed = store
            .replay(&session.blueprint.session_id)
            .await
            .expect(backend);
        assert_eq!(replayed.len(), 1, "{backend}");
    }
}

#[tokio::test]
async fn commit_without_expected_head_skips_conflict_detection() {
    for (backend, store) in backends() {
        let session = test_session("unchecked");
        store.create_session(session.clone()).await.expect(backend);

        // Two commits with no expected head both succeed regardless of the
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
async fn create_session_rejects_nonzero_sequences() {
    for (backend, store) in backends() {
        let mut session = test_session("nonzero");
        session.head_sequence = 2;
        session.state_sequence = 2;

        let error = store
            .create_session(session)
            .await
            .expect_err("nonzero sequences must be rejected");
        assert!(
            error.to_string().contains("must start at sequence 0"),
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

/// Event-only commits advance the head while the checkpoint stays behind —
/// the exact gap a loader closes by folding `replay_after(state_sequence)`.
#[tokio::test]
async fn checkpoint_lags_head_after_event_only_commit() {
    for (backend, store) in backends() {
        let session = test_session("lagging");
        store.create_session(session.clone()).await.expect(backend);

        store
            .commit(
                &session.blueprint.session_id,
                None,
                Some(0),
                Some(state_with_summary("checkpointed")),
                vec![test_event("one")],
            )
            .await
            .expect(backend);
        store
            .commit(
                &session.blueprint.session_id,
                None,
                Some(1),
                None,
                vec![test_event("two"), test_event("three")],
            )
            .await
            .expect(backend);

        let reloaded = store
            .load_session(&session.blueprint.session_id)
            .await
            .expect(backend)
            .expect(backend);
        assert_eq!(reloaded.state_sequence, 1, "{backend}");
        assert_eq!(reloaded.head_sequence, 3, "{backend}");

        let tail: Vec<u64> = store
            .replay_after(&session.blueprint.session_id, reloaded.state_sequence)
            .await
            .expect(backend)
            .iter()
            .map(SessionEvent::sequence)
            .collect();
        assert_eq!(tail, vec![2, 3], "{backend}");

        let empty_tail = store
            .replay_after(&session.blueprint.session_id, reloaded.head_sequence)
            .await
            .expect(backend);
        assert!(empty_tail.is_empty(), "{backend}");
    }
}

/// The log/checkpoint invariant: folding the full committed log over a
/// default state agrees with the persisted checkpoint on every fold-covered
/// field, through message appends and a state-rewriting compaction.
#[tokio::test]
async fn fold_of_full_log_matches_state_checkpoint() {
    for (backend, store) in backends() {
        let session = test_session("folds");
        store.create_session(session.clone()).await.expect(backend);
        let session_id = &session.blueprint.session_id;
        let mut state = session.state.clone();
        let mut head = 0u64;

        // Turn one: user + assistant messages.
        let user = Message::User(UserMessage::text("hello"));
        let assistant = assistant_message("hi there", 12, 4);
        state.messages.push(user.clone());
        state.messages.push(assistant.clone());
        state
            .usage_so_far
            .saturating_accumulate(&assistant_usage(12, 4));
        head = commit_events(
            store.as_ref(),
            session_id,
            head,
            &state,
            vec![
                message_event(user),
                message_event(assistant),
                PendingEvent::new(
                    session_id.clone(),
                    Delivery::Lossless,
                    SessionEventPayload::TurnCompleted {
                        turn_id: halter_protocol::TurnId::new(),
                        usage: assistant_usage(12, 4),
                    },
                ),
            ],
        )
        .await;

        // Compaction rewrites the window; the event carries the effects.
        let window = vec![Message::User(UserMessage::text("carried forward"))];
        let prefix = vec![serde_json::json!({"kind": "compacted"})];
        state.messages = window.clone();
        state.compacted_prefix = prefix.clone();
        head = commit_events(
            store.as_ref(),
            session_id,
            head,
            &state,
            vec![PendingEvent::new(
                session_id.clone(),
                Delivery::Lossless,
                SessionEventPayload::ContextCompacted {
                    summary: "squashed".to_owned(),
                    effects: Some(Box::new(CompactionEventEffects {
                        messages: window,
                        compacted_prefix: prefix,
                    })),
                },
            )],
        )
        .await;

        // Turn two: another assistant message after compaction.
        let assistant = assistant_message("post-compaction", 3, 9);
        state.messages.push(assistant.clone());
        state
            .usage_so_far
            .saturating_accumulate(&assistant_usage(3, 9));
        commit_events(
            store.as_ref(),
            session_id,
            head,
            &state,
            vec![message_event(assistant)],
        )
        .await;

        let reloaded = store
            .load_session(session_id)
            .await
            .expect(backend)
            .expect(backend);
        let replayed = store.replay(session_id).await.expect(backend);
        let folded = fold::fold_events(SessionState::default(), &replayed);

        assert!(
            fold::covered_state_matches(&folded, &reloaded.state),
            "{backend}: folded log diverged from checkpoint\nfolded: {folded:?}\ncheckpoint: {:?}",
            reloaded.state
        );
        assert_eq!(reloaded.state_sequence, reloaded.head_sequence, "{backend}");
    }
}

async fn commit_events(
    store: &dyn SessionStore,
    session_id: &SessionId,
    expected_head: u64,
    state: &SessionState,
    events: Vec<PendingEvent>,
) -> u64 {
    let committed = store
        .commit(
            session_id,
            None,
            Some(expected_head),
            Some(state.clone()),
            events,
        )
        .await
        .expect("commit");
    committed
        .last()
        .map_or(expected_head, SessionEvent::sequence)
}

fn assistant_usage(input: u64, output: u64) -> halter_protocol::Usage {
    halter_protocol::Usage {
        input_tokens: input,
        output_tokens: output,
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: 0,
    }
}

fn assistant_message(text: &str, input_tokens: u64, output_tokens: u64) -> Message {
    let created_at: halter_protocol::Timestamp =
        serde_json::from_value(serde_json::json!("2026-01-01T00:00:00Z")).expect("timestamp");
    Message::Assistant(halter_protocol::AssistantMessage {
        id: halter_protocol::MessageId::new(),
        created_at,
        parts: vec![halter_protocol::AssistantPart::Text {
            text: text.to_owned(),
        }],
        stop_reason: Some(halter_protocol::StopReason::EndTurn),
        usage: Some(assistant_usage(input_tokens, output_tokens)),
        replay_meta: halter_protocol::ReplayMeta::default(),
    })
}

fn message_event(message: Message) -> PendingEvent {
    PendingEvent::new(
        SessionId::from("overwritten-by-store"),
        Delivery::Lossless,
        SessionEventPayload::MessageItem { message },
    )
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
    StoredSession::new(blueprint, SessionState::default(), snapshot)
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

fn test_event(summary: &str) -> PendingEvent {
    PendingEvent::new(
        SessionId::from("overwritten-by-store"),
        Delivery::Lossless,
        SessionEventPayload::ContextCompacted {
            summary: summary.to_owned(),
            effects: None,
        },
    )
}
