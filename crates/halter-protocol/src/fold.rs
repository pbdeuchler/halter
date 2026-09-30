//! Pure fold from committed session events onto session state.
//!
//! The session store persists two representations of a session: an
//! append-only event log and a [`SessionState`] checkpoint stamped with the
//! log position it reflects. This module is the bridge between them: applying
//! the events after a checkpoint to that checkpoint's state reproduces the
//! current state, which makes the log the source of truth and the checkpoint
//! a cache.
//!
//! # Covered fields
//!
//! The fold reproduces the *domain* fields of [`SessionState`] — the ones
//! that define conversational context and telemetry:
//!
//! - `messages` — appended by [`SessionEventPayload::MessageItem`] (never by
//!   [`SessionEventPayload::MessageRecorded`]), replaced
//!   by [`SessionEventPayload::ContextCompacted`] when it carries
//!   [`CompactionEventEffects`] and by [`SessionEventPayload::ContextRestored`].
//! - `compacted_prefix` — replaced by `ContextCompacted` and `ContextRestored`
//!   effects.
//! - `usage_so_far` — accumulated (saturating) from the `usage` stamped on
//!   assistant messages, appended or only recorded.
//! - `token_ledger` — advanced by every `MessageItem` through
//!   [`SessionState::append`], rebuilt from `ContextCompacted` effects, and
//!   put back by `ContextRestored`, exactly as the runtime does. A
//!   checkpoint from before ledger accounting re-estimates its ledger at the
//!   next request, so it matches a whole-log fold again only from the next
//!   provider report on.
//! - `context_window` — advanced by each state-rewriting compaction.
//! - `subagents` — upserted by [`SessionEventPayload::SubagentUpdated`],
//!   ignoring a record whose generation is older than the one held.
//! - `compaction_notifications` — inserted by
//!   [`SessionEventPayload::CompactionNotified`], cleared by each
//!   state-rewriting compaction.
//! - `pending_inputs` — queued by [`SessionEventPayload::InputAccepted`],
//!   removed by matching user `MessageItem` delivery or `InputRejected`.
//!   Context rewrites leave this inbox intact.
//! - `session_status` — replaced by
//!   [`SessionEventPayload::SessionStatusChanged`].
//!
//! Runtime bookkeeping fields (`pending_tool_calls`, `fired_hook_ids`,
//! `appended_prompt_segments`, `lineage`, hook latches, and
//! provider-chaining fields) are deliberately generally carried by the
//! checkpoint, which the runtime writes on every state-changing commit.
//! Compaction and rollover events reset `last_response_id` and
//! `messages_seen_by_provider`. Rollover additionally clears
//! `appended_prompt_segments`. These event-covered
//! resets keep replay from retaining bookkeeping from a previous window;
//! ordinary updates to those fields still depend on the checkpoint.
//!
//! The store conformance suite locks the invariant in: after any sequence of
//! commits, folding the full log over a default state must agree with the
//! persisted checkpoint on every covered field ([`covered_state_matches`]).

use crate::{Message, SessionEvent, SessionEventPayload, SessionState};

/// Apply one committed event payload to `state`, mutating only the
/// fold-covered fields (see the module docs for the exact list). Events that
/// carry no state transition — lifecycle markers, hook run summaries, deltas,
/// tool output chunks — are no-ops.
pub fn apply_event(state: &mut SessionState, payload: &SessionEventPayload) {
    match payload {
        SessionEventPayload::InputAccepted { message } => {
            state.pending_inputs.push(message.clone());
        }
        SessionEventPayload::InputRejected { message_id, .. } => {
            state
                .pending_inputs
                .retain(|message| &message.id != message_id);
        }
        SessionEventPayload::SessionStatusChanged { status } => {
            state.session_status = *status;
        }
        SessionEventPayload::MessageItem { message }
        | SessionEventPayload::MessageRecorded { message } => {
            if let Message::Assistant(assistant) = message
                && let Some(usage) = &assistant.usage
            {
                state.usage_so_far.saturating_accumulate(usage);
            }
            if matches!(payload, SessionEventPayload::MessageItem { .. }) {
                state.append(message.clone());
            }
        }
        SessionEventPayload::ContextProjectionUpdated { request_tokens } => {
            let compacted_prefix = &state.compacted_prefix;
            let messages = &state.messages;
            state
                .token_ledger
                .prepare_request(*request_tokens, compacted_prefix, messages);
        }
        SessionEventPayload::ContextRestored { effects, .. } => {
            state.messages = effects.messages.clone();
            state.compacted_prefix = effects.compacted_prefix.clone();
            state.token_ledger = effects.token_ledger;
        }
        SessionEventPayload::ContextCompacted {
            effects: Some(effects),
            ..
        }
        | SessionEventPayload::ContextWindowRolledOver { effects, .. } => {
            if matches!(payload, SessionEventPayload::ContextWindowRolledOver { .. }) {
                state.appended_prompt_segments.clear();
            }
            state.usage_so_far.saturating_accumulate(&effects.usage);
            state.messages = effects.messages.clone();
            state.compacted_prefix = effects.compacted_prefix.clone();
            // Compaction breaks the provider response chain; mirror the
            // runtime so replayed views never chain onto pre-rewrite context.
            state.last_response_id = None;
            state.messages_seen_by_provider = 0;
            // Usage reported by the preserved tail predates this rewrite.
            state
                .token_ledger
                .rebuild_context(&state.compacted_prefix, &state.messages);
            state.context_window = state.context_window.saturating_add(1);
            state.compaction_notifications.clear();
        }
        SessionEventPayload::ContextCompacted { effects: None, .. }
        | SessionEventPayload::SessionStarted
        | SessionEventPayload::SessionResumed
        | SessionEventPayload::Warning { .. }
        | SessionEventPayload::TurnStarted { .. }
        | SessionEventPayload::DeltaItem { .. }
        | SessionEventPayload::ProviderMetadata { .. }
        | SessionEventPayload::ToolExecutionStarted { .. }
        | SessionEventPayload::ToolOutput { .. }
        | SessionEventPayload::HookStarted { .. }
        | SessionEventPayload::HookCompleted { .. }
        | SessionEventPayload::ToolExecutionCompleted { .. }
        | SessionEventPayload::ApprovalRequested { .. }
        | SessionEventPayload::TurnCompleted { .. }
        | SessionEventPayload::TurnFailed { .. }
        | SessionEventPayload::Lagged { .. }
        | SessionEventPayload::SessionShutdownComplete
        | SessionEventPayload::Unknown => {}
        SessionEventPayload::CompactionNotified { id } => {
            state.compaction_notifications.insert(id.clone());
        }
        SessionEventPayload::SubagentUpdated { record } => {
            let id = &record.status.agent_id;
            if state
                .subagents
                .get(id)
                .is_none_or(|held| held.generation <= record.generation)
            {
                state.subagents.insert(id.clone(), record.clone());
            }
        }
    }
}

/// Fold committed events onto a checkpoint state, returning the resulting
/// state. Events must be supplied in sequence order (as returned by
/// `SessionStore::replay`).
#[must_use]
pub fn fold_events(mut state: SessionState, events: &[SessionEvent]) -> SessionState {
    for event in events {
        apply_event(&mut state, &event.payload);
    }
    state
}

/// Whether two states agree on every fold-covered field. This is the
/// conformance predicate for the log/checkpoint invariant; bookkeeping fields
/// outside the fold's coverage are intentionally ignored.
#[must_use]
pub fn covered_state_matches(a: &SessionState, b: &SessionState) -> bool {
    a.messages == b.messages
        && a.pending_inputs == b.pending_inputs
        && a.session_status == b.session_status
        && a.compacted_prefix == b.compacted_prefix
        && a.usage_so_far == b.usage_so_far
        && a.token_ledger == b.token_ledger
        && a.context_window == b.context_window
        && a.subagents == b.subagents
        && a.compaction_notifications == b.compaction_notifications
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use proptest::prelude::*;
    use serde_json::json;

    use super::*;
    use crate::{
        AssistantMessage, CompactionEventEffects, Delivery, MessageId, PendingEvent, ReplayMeta,
        SessionId, SessionStatus, StopReason, TokenLedger, Usage, UserMessage,
    };

    fn assistant_message(text: &str, usage: Option<Usage>) -> Message {
        Message::Assistant(AssistantMessage {
            id: MessageId::new(),
            created_at: Utc::now(),
            parts: vec![crate::AssistantPart::Text {
                text: text.to_owned(),
            }],
            stop_reason: Some(StopReason::EndTurn),
            usage,
            replay_meta: ReplayMeta::default(),
        })
    }

    fn usage(input: u64, output: u64) -> Usage {
        Usage {
            input_tokens: input,
            output_tokens: output,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        }
    }

    fn committed(sequence: u64, payload: SessionEventPayload) -> SessionEvent {
        PendingEvent::new(SessionId::from("session"), Delivery::Lossless, payload)
            .into_committed(sequence)
    }

    #[test]
    fn message_item_appends_and_accumulates_assistant_usage() {
        let mut state = SessionState::default();
        apply_event(
            &mut state,
            &SessionEventPayload::MessageItem {
                message: Message::User(UserMessage::text("hi")),
            },
        );
        apply_event(
            &mut state,
            &SessionEventPayload::MessageItem {
                message: assistant_message("hello", Some(usage(10, 5))),
            },
        );

        assert_eq!(state.messages.len(), 2);
        assert_eq!(state.usage_so_far, usage(10, 5));
        // The completed report anchors the ledger, just as the runtime's
        // append does.
        assert_eq!(
            state.token_ledger,
            TokenLedger {
                authoritative_tokens: 15,
                inferred_tokens: 0,
                ..TokenLedger::default()
            }
        );
    }

    #[test]
    fn accepted_input_enters_history_only_on_delivery() {
        let first = UserMessage::text("first");
        let second = UserMessage::text("second");
        let mut accepted = SessionState::default();
        for message in [&first, &second] {
            apply_event(
                &mut accepted,
                &SessionEventPayload::InputAccepted {
                    message: message.clone(),
                },
            );
        }
        assert_eq!(accepted.pending_inputs, [first.clone(), second.clone()]);
        assert!(accepted.messages.is_empty());
        assert_eq!(accepted.token_ledger, TokenLedger::default());

        let cases = [
            (
                "delivered",
                SessionEventPayload::MessageItem {
                    message: Message::User(first.clone()),
                },
                vec![second.clone()],
                vec![Message::User(first.clone())],
            ),
            (
                "recorded only",
                SessionEventPayload::MessageRecorded {
                    message: Message::User(first.clone()),
                },
                vec![first.clone(), second.clone()],
                vec![],
            ),
            (
                "rejected",
                SessionEventPayload::InputRejected {
                    message_id: first.id.clone(),
                    reason: "invalid input".to_owned(),
                },
                vec![second.clone()],
                vec![],
            ),
            (
                "unrelated rejection",
                SessionEventPayload::InputRejected {
                    message_id: MessageId::from("not-pending"),
                    reason: "invalid input".to_owned(),
                },
                vec![first.clone(), second.clone()],
                vec![],
            ),
        ];
        for (name, payload, pending_inputs, messages) in cases {
            let mut state = accepted.clone();
            apply_event(&mut state, &payload);
            assert_eq!(state.pending_inputs, pending_inputs, "{name}");
            assert_eq!(state.messages, messages, "{name}");
            assert_eq!(
                state.token_ledger,
                TokenLedger::inferred_from(&[], &state.messages),
                "{name}"
            );
        }
    }

    #[test]
    fn context_rewrites_preserve_pending_input_and_session_status() {
        let pending = UserMessage::text("keep the public API");
        let window = vec![Message::user("rewritten context")];
        let effects = CompactionEventEffects {
            messages: window.clone(),
            compacted_prefix: vec![],
            usage: Usage::default(),
        };
        let cases = [
            SessionEventPayload::ContextCompacted {
                summary: "compacted".to_owned(),
                effects: Some(Box::new(effects.clone())),
            },
            SessionEventPayload::ContextWindowRolledOver {
                summary: "rolled over".to_owned(),
                effects: Box::new(effects),
            },
            SessionEventPayload::ContextRestored {
                reason: "failed compaction".to_owned(),
                effects: Box::new(crate::RestoredContext {
                    messages: window.clone(),
                    compacted_prefix: vec![],
                    token_ledger: TokenLedger::inferred_from(&[], &window),
                }),
            },
        ];
        for payload in cases {
            let mut state = SessionState {
                pending_inputs: vec![pending.clone()],
                session_status: SessionStatus::Running,
                ..SessionState::default()
            };
            state.append(Message::user("old context"));
            apply_event(&mut state, &payload);
            assert_eq!(state.messages, window, "{payload:?}");
            assert_eq!(
                state.pending_inputs,
                std::slice::from_ref(&pending),
                "{payload:?}"
            );
            assert_eq!(state.session_status, SessionStatus::Running, "{payload:?}");
        }
    }

    #[test]
    fn session_lifecycle_status_replays_without_discarding_the_inbox() {
        let pending = UserMessage::text("continue later");
        let mut state = SessionState {
            pending_inputs: vec![pending.clone()],
            ..SessionState::default()
        };
        for status in [
            SessionStatus::Running,
            SessionStatus::Closed,
            SessionStatus::Idle,
        ] {
            apply_event(
                &mut state,
                &SessionEventPayload::SessionStatusChanged { status },
            );
            assert_eq!(state.session_status, status);
            assert_eq!(state.pending_inputs, std::slice::from_ref(&pending));
        }
    }

    #[test]
    fn unknown_events_leave_state_untouched() {
        let mut state = SessionState::default();
        apply_event(
            &mut state,
            &SessionEventPayload::MessageItem {
                message: assistant_message("kept", Some(usage(3, 2))),
            },
        );
        let before = state.clone();
        apply_event(&mut state, &SessionEventPayload::Unknown);
        assert_eq!(state, before);
    }

    #[test]
    fn recorded_messages_count_usage_without_entering_the_transcript() {
        let cases = [
            (
                "user",
                Message::User(UserMessage::text("summarize")),
                usage(0, 0),
            ),
            (
                "assistant with usage",
                assistant_message("gist", Some(usage(10, 5))),
                usage(10, 5),
            ),
            (
                "assistant without usage",
                assistant_message("gist", None),
                usage(0, 0),
            ),
        ];
        for (name, message, expected_usage) in cases {
            let mut state = SessionState::default();
            apply_event(
                &mut state,
                &SessionEventPayload::MessageRecorded { message },
            );
            assert!(state.messages.is_empty(), "{name}");
            assert_eq!(state.token_ledger, TokenLedger::default(), "{name}");
            assert_eq!(state.usage_so_far, expected_usage, "{name}");
        }
    }

    #[test]
    fn assistant_message_without_usage_appends_without_accumulating() {
        let mut state = SessionState::default();
        apply_event(
            &mut state,
            &SessionEventPayload::MessageItem {
                message: assistant_message("hello", None),
            },
        );

        assert_eq!(state.messages.len(), 1);
        assert_eq!(state.usage_so_far, Usage::default());
        assert_eq!(state.token_ledger.authoritative_tokens, 0);
        assert_eq!(
            state.token_ledger.inferred_tokens,
            crate::estimate_messages_tokens(&state.messages)
        );
    }

    #[test]
    fn usage_accumulation_saturates_instead_of_overflowing() {
        let mut state = SessionState {
            usage_so_far: usage(u64::MAX - 1, 0),
            ..SessionState::default()
        };
        apply_event(
            &mut state,
            &SessionEventPayload::MessageItem {
                message: assistant_message("hello", Some(usage(10, 3))),
            },
        );

        assert_eq!(state.usage_so_far.input_tokens, u64::MAX);
        assert_eq!(state.usage_so_far.output_tokens, 3);
    }

    #[test]
    fn compaction_with_effects_replaces_window_and_resets_chain() {
        let mut state = SessionState {
            messages: vec![
                Message::User(UserMessage::text("old-1")),
                Message::User(UserMessage::text("old-2")),
            ],
            last_response_id: Some("resp-1".to_owned()),
            messages_seen_by_provider: 2,
            token_ledger: TokenLedger {
                authoritative_tokens: 80_000,
                inferred_tokens: 0,
                ..TokenLedger::default()
            },
            ..SessionState::default()
        };
        let window = vec![Message::User(UserMessage::text("kept"))];
        apply_event(
            &mut state,
            &SessionEventPayload::ContextCompacted {
                summary: "compacted".to_owned(),
                effects: Some(Box::new(CompactionEventEffects {
                    messages: window.clone(),
                    compacted_prefix: vec![json!({"kind": "prefix"})],
                    usage: Usage::default(),
                })),
            },
        );

        assert_eq!(state.messages, window);
        assert_eq!(state.compacted_prefix, vec![json!({"kind": "prefix"})]);
        assert_eq!(state.last_response_id, None);
        assert_eq!(state.messages_seen_by_provider, 0);
        // Replay must rebuild the same ledger the runtime does, or a resumed
        // session would keep the stale pre-compaction anchor.
        assert_eq!(
            state.token_ledger,
            TokenLedger::inferred_from(&state.compacted_prefix, &window)
        );

        assert_eq!(state.token_ledger.authoritative_tokens, 0);
    }

    /// A pass that appended and then failed leaves its exchange in the log;
    /// the restore is the transition that takes the window back, ledger
    /// included, without touching the usage those messages accumulated.
    #[test]
    fn context_restored_puts_the_window_and_ledger_back() {
        let mut state = SessionState::default();
        state.append(Message::User(UserMessage::text("kept")));
        let before = state.clone();
        apply_event(
            &mut state,
            &SessionEventPayload::MessageItem {
                message: Message::User(UserMessage::text("compaction nudge")),
            },
        );
        apply_event(
            &mut state,
            &SessionEventPayload::MessageItem {
                message: assistant_message("reply", Some(usage(10, 5))),
            },
        );
        assert_ne!(state.messages, before.messages);

        apply_event(
            &mut state,
            &SessionEventPayload::ContextRestored {
                reason: "checkpoint inference failed".to_owned(),
                effects: Box::new(crate::RestoredContext {
                    messages: before.messages.clone(),
                    compacted_prefix: before.compacted_prefix.clone(),
                    token_ledger: before.token_ledger,
                }),
            },
        );

        assert_eq!(state.messages, before.messages);
        assert_eq!(state.compacted_prefix, before.compacted_prefix);
        assert_eq!(state.token_ledger, before.token_ledger);
        assert_eq!(state.usage_so_far, usage(10, 5), "billed usage is kept");
    }

    /// A checkpoint from before ledger accounting (≤ v0.5) re-estimates its
    /// ledger at the next request, while a fold of the whole log keeps the
    /// last provider report. The two agree again at the next report.
    #[test]
    fn legacy_ledgers_rejoin_the_fold_at_the_next_provider_report() {
        let history = [
            SessionEventPayload::MessageItem {
                message: Message::User(UserMessage::text("before the upgrade")),
            },
            SessionEventPayload::MessageItem {
                message: assistant_message("reply", Some(usage(900, 100))),
            },
        ];
        let mut from_log = SessionState::default();
        for payload in &history {
            apply_event(&mut from_log, payload);
        }
        let mut from_checkpoint = SessionState {
            token_ledger: TokenLedger {
                accounting_version: 0,
                ..from_log.token_ledger
            },
            ..from_log.clone()
        };

        // (tail event, whether the two states agree after it)
        for (payload, agree) in [
            (
                SessionEventPayload::ContextProjectionUpdated {
                    request_tokens: 250,
                },
                false,
            ),
            (
                SessionEventPayload::MessageItem {
                    message: Message::User(UserMessage::text("after the upgrade")),
                },
                false,
            ),
            (
                SessionEventPayload::MessageItem {
                    message: assistant_message("reply", Some(usage(1_200, 50))),
                },
                true,
            ),
        ] {
            apply_event(&mut from_log, &payload);
            apply_event(&mut from_checkpoint, &payload);
            assert_eq!(
                covered_state_matches(&from_log, &from_checkpoint),
                agree,
                "{payload:?}\nlog: {:?}\ncheckpoint: {:?}",
                from_log.token_ledger,
                from_checkpoint.token_ledger
            );
        }
    }

    #[test]
    fn subagent_records_older_than_the_held_generation_are_ignored() {
        let record = |state, generation| crate::SubagentRecord {
            status: crate::SubagentStatus {
                agent_id: crate::AgentId::from("agent"),
                session_id: SessionId::from("child"),
                agent_type: None,
                task: "task".to_owned(),
                state,
                last_message: None,
                usage: None,
                error: None,
            },
            generation,
        };
        use crate::SubagentState::{Closed, Completed, Running};
        let cases = [
            ("first record", None, record(Running, 1), record(Running, 1)),
            (
                "same generation",
                Some(record(Running, 1)),
                record(Completed, 1),
                record(Completed, 1),
            ),
            (
                "newer generation",
                Some(record(Completed, 1)),
                record(Running, 2),
                record(Running, 2),
            ),
            (
                "older generation",
                Some(record(Closed, 2)),
                record(Completed, 1),
                record(Closed, 2),
            ),
        ];
        for (name, held, incoming, expected) in cases {
            let mut state = SessionState::default();
            if let Some(held) = held {
                state.subagents.insert(held.status.agent_id.clone(), held);
            }
            apply_event(
                &mut state,
                &SessionEventPayload::SubagentUpdated { record: incoming },
            );
            assert_eq!(
                state.subagents.values().collect::<Vec<_>>(),
                [&expected],
                "{name}"
            );
        }
    }

    #[test]
    fn compaction_without_effects_is_a_noop() {
        let original = SessionState {
            messages: vec![Message::User(UserMessage::text("kept"))],
            last_response_id: Some("resp-1".to_owned()),
            ..SessionState::default()
        };
        let mut state = original.clone();
        apply_event(
            &mut state,
            &SessionEventPayload::ContextCompacted {
                summary: "No compaction needed.".to_owned(),
                effects: None,
            },
        );

        assert_eq!(state, original);
    }

    #[test]
    fn non_covered_events_do_not_change_state() {
        let original = SessionState {
            messages: vec![Message::User(UserMessage::text("kept"))],
            usage_so_far: usage(7, 7),
            ..SessionState::default()
        };
        let payloads = [
            SessionEventPayload::SessionStarted,
            SessionEventPayload::SessionResumed,
            SessionEventPayload::TurnStarted {
                turn_id: crate::TurnId::new(),
                default_model: None,
                subagent_model: None,
            },
            SessionEventPayload::TurnCompleted {
                turn_id: crate::TurnId::new(),
                usage: usage(100, 100),
            },
            SessionEventPayload::TurnFailed {
                turn_id: crate::TurnId::new(),
                error: "boom".to_owned(),
                cancelled: false,
                retryable: false,
            },
            SessionEventPayload::DeltaItem {
                delta: crate::DeltaItem {
                    text: "chunk".to_owned(),
                },
            },
            SessionEventPayload::SessionShutdownComplete,
        ];
        for payload in &payloads {
            let mut state = original.clone();
            apply_event(&mut state, payload);
            assert_eq!(state, original, "payload {payload:?} must be a no-op");
        }
    }

    #[test]
    fn fold_events_replays_in_order() {
        let events = vec![
            committed(
                1,
                SessionEventPayload::MessageItem {
                    message: Message::User(UserMessage::text("one")),
                },
            ),
            committed(
                2,
                SessionEventPayload::ContextCompacted {
                    summary: "squash".to_owned(),
                    effects: Some(Box::new(CompactionEventEffects {
                        messages: Vec::new(),
                        compacted_prefix: vec![json!("p")],
                        usage: Usage::default(),
                    })),
                },
            ),
            committed(
                3,
                SessionEventPayload::MessageItem {
                    message: assistant_message("after", Some(usage(1, 2))),
                },
            ),
        ];

        let state = fold_events(SessionState::default(), &events);

        assert_eq!(state.messages.len(), 1);
        assert_eq!(state.compacted_prefix, vec![json!("p")]);
        assert_eq!(state.usage_so_far, usage(1, 2));
    }

    #[test]
    fn covered_state_matches_ignores_bookkeeping_but_not_domain_fields() {
        let base = SessionState {
            messages: vec![Message::User(UserMessage::text("m"))],
            usage_so_far: usage(1, 1),
            ..SessionState::default()
        };
        let bookkeeping_differs = SessionState {
            fired_hook_ids: vec!["hook".to_owned()],
            messages_seen_by_provider: 9,
            ..base.clone()
        };
        assert!(covered_state_matches(&base, &bookkeeping_differs));

        let usage_differs = SessionState {
            usage_so_far: usage(2, 1),
            ..base.clone()
        };
        assert!(!covered_state_matches(&base, &usage_differs));

        let messages_differ = SessionState {
            messages: Vec::new(),
            ..base.clone()
        };
        assert!(!covered_state_matches(&base, &messages_differ));

        let ledger_differs = SessionState {
            token_ledger: TokenLedger {
                authoritative_tokens: 1,
                inferred_tokens: 0,
                ..TokenLedger::default()
            },
            ..base.clone()
        };
        assert!(!covered_state_matches(&base, &ledger_differs));

        let window_differs = SessionState {
            context_window: 1,
            ..base.clone()
        };
        assert!(!covered_state_matches(&base, &window_differs));

        let inbox_differs = SessionState {
            pending_inputs: vec![UserMessage::text("pending")],
            ..base.clone()
        };
        assert!(!covered_state_matches(&base, &inbox_differs));

        let status_differs = SessionState {
            session_status: SessionStatus::Closed,
            ..base.clone()
        };
        assert!(!covered_state_matches(&base, &status_differs));
    }

    fn fold_payload_strategy() -> impl Strategy<Value = SessionEventPayload> {
        let user_message = "[a-zA-Z0-9 ]{0,32}".prop_map(|text| SessionEventPayload::MessageItem {
            message: Message::User(UserMessage::text(text)),
        });
        let assistant_message = (
            "[a-zA-Z0-9 ]{0,32}",
            any::<u64>(),
            any::<u64>(),
            any::<u64>(),
            any::<u64>(),
        )
            .prop_map(|(text, input, output, cache_creation, cache_read)| {
                SessionEventPayload::MessageItem {
                    message: assistant_message(
                        &text,
                        Some(Usage {
                            input_tokens: input,
                            output_tokens: output,
                            cache_creation_input_tokens: cache_creation,
                            cache_read_input_tokens: cache_read,
                        }),
                    ),
                }
            });
        let compaction = (
            prop::collection::vec("[a-zA-Z0-9 ]{0,24}", 0..6),
            prop::collection::vec(any::<u8>(), 0..6),
        )
            .prop_map(|(messages, prefix)| SessionEventPayload::ContextCompacted {
                summary: "generated compaction".to_owned(),
                effects: Some(Box::new(CompactionEventEffects {
                    messages: messages
                        .into_iter()
                        .map(|text| Message::User(UserMessage::text(text)))
                        .collect(),
                    compacted_prefix: prefix.into_iter().map(|value| json!(value)).collect(),
                    usage: Usage::default(),
                })),
            });
        let input = "[a-zA-Z0-9 ]{0,32}".prop_map(|text| SessionEventPayload::InputAccepted {
            message: UserMessage::text(text),
        });
        let status = prop_oneof![
            Just(SessionStatus::Idle),
            Just(SessionStatus::Running),
            Just(SessionStatus::Closed),
        ]
        .prop_map(|status| SessionEventPayload::SessionStatusChanged { status });

        prop_oneof![
            4 => user_message,
            4 => assistant_message,
            2 => compaction,
            2 => input,
            1 => status,
            1 => Just(SessionEventPayload::SessionStarted),
            1 => Just(SessionEventPayload::SessionResumed),
        ]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn refolding_from_any_checkpoint_matches_full_replay(
            payloads in prop::collection::vec(fold_payload_strategy(), 0..40),
            split_seed in any::<usize>(),
        ) {
            let events: Vec<_> = payloads
                .into_iter()
                .enumerate()
                .map(|(offset, payload)| committed(offset as u64 + 1, payload))
                .collect();
            let split = split_seed % (events.len() + 1);

            let full = fold_events(SessionState::default(), &events);
            let checkpoint = fold_events(SessionState::default(), &events[..split]);
            let resumed = fold_events(checkpoint, &events[split..]);

            prop_assert_eq!(resumed, full);
        }

        #[test]
        fn message_replay_preserves_supplied_order(
            texts in prop::collection::vec("[a-zA-Z0-9 ]{0,32}", 0..40),
        ) {
            let expected: Vec<_> = texts
                .into_iter()
                .map(|text| Message::User(UserMessage::text(text)))
                .collect();
            let events: Vec<_> = expected
                .iter()
                .cloned()
                .enumerate()
                .map(|(offset, message)| {
                    committed(
                        offset as u64 + 1,
                        SessionEventPayload::MessageItem { message },
                    )
                })
                .collect();

            let folded = fold_events(SessionState::default(), &events);

            prop_assert_eq!(folded.messages, expected);
        }

        #[test]
        fn accepted_inputs_are_partitioned_into_delivered_rejected_and_pending(
            inputs in prop::collection::vec(("[a-zA-Z0-9 ]{0,32}", 0u8..3), 0..40),
            split_seed in any::<usize>(),
        ) {
            let inputs: Vec<_> = inputs.into_iter()
                .map(|(text, disposition)| (UserMessage::text(text), disposition))
                .collect();
            let mut events = vec![];
            for (message, _) in &inputs {
                events.push(committed(events.len() as u64 + 1, SessionEventPayload::InputAccepted {
                    message: message.clone(),
                }));
            }
            for (message, disposition) in &inputs {
                let payload = match disposition {
                    0 => continue,
                    1 => SessionEventPayload::MessageItem { message: Message::User(message.clone()) },
                    _ => SessionEventPayload::InputRejected {
                        message_id: message.id.clone(),
                        reason: "rejected".to_owned(),
                    },
                };
                events.push(committed(events.len() as u64 + 1, payload));
            }

            let expected_pending: Vec<_> = inputs.iter()
                .filter(|(_, disposition)| *disposition == 0)
                .map(|(message, _)| message.clone())
                .collect();
            let expected_delivered: Vec<_> = inputs.iter()
                .filter(|(_, disposition)| *disposition == 1)
                .map(|(message, _)| Message::User(message.clone()))
                .collect();
            let split = split_seed % (events.len() + 1);
            let checkpoint = fold_events(SessionState::default(), &events[..split]);
            // Resume from the serialized checkpoint, as persistent stores do.
            let checkpoint = serde_json::from_slice(&serde_json::to_vec(&checkpoint).unwrap()).unwrap();
            let resumed = fold_events(checkpoint, &events[split..]);

            prop_assert_eq!(&resumed.pending_inputs, &expected_pending);
            prop_assert_eq!(&resumed.messages, &expected_delivered);
            prop_assert_eq!(resumed.token_ledger, TokenLedger::inferred_from(&[], &expected_delivered));
            prop_assert_eq!(resumed, fold_events(SessionState::default(), &events));
        }
    }
}
