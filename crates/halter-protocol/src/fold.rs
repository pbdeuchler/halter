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
//! - `messages` — appended by [`SessionEventPayload::MessageItem`], replaced
//!   by [`SessionEventPayload::ContextCompacted`] when it carries
//!   [`CompactionEventEffects`].
//! - `compacted_prefix` — replaced by `ContextCompacted` effects.
//! - `usage_so_far` — accumulated (saturating) from the `usage` stamped on
//!   assistant messages.
//!
//! Runtime bookkeeping fields (`file_view_cache`, `pending_tool_calls`,
//! `fired_hook_ids`, `appended_prompt_segments`, `lineage`, hook latches, and
//! provider-chaining fields) are deliberately *not* event-covered: they are
//! carried by the checkpoint, which the runtime writes on every
//! state-changing commit. The one exception is that `ContextCompacted`
//! effects also reset `last_response_id` / `messages_seen_by_provider`,
//! mirroring the runtime's compaction rules so a mid-replay view is not left
//! pointing at a provider response chain that predates the rewrite.
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
        SessionEventPayload::MessageItem { message } => {
            if let Message::Assistant(assistant) = message
                && let Some(usage) = &assistant.usage
            {
                state.usage_so_far.saturating_accumulate(usage);
            }
            state.messages.push(message.clone());
        }
        SessionEventPayload::ContextCompacted {
            effects: Some(effects),
            ..
        } => {
            state.messages = effects.messages.clone();
            state.compacted_prefix = effects.compacted_prefix.clone();
            // Compaction breaks the provider response chain; mirror the
            // runtime so replayed views never chain onto pre-rewrite context.
            state.last_response_id = None;
            state.messages_seen_by_provider = 0;
        }
        SessionEventPayload::ContextCompacted { effects: None, .. }
        | SessionEventPayload::SessionStarted
        | SessionEventPayload::SessionResumed
        | SessionEventPayload::Warning { .. }
        | SessionEventPayload::TurnStarted { .. }
        | SessionEventPayload::DeltaItem { .. }
        | SessionEventPayload::ToolExecutionStarted { .. }
        | SessionEventPayload::ToolOutput { .. }
        | SessionEventPayload::HookStarted { .. }
        | SessionEventPayload::HookCompleted { .. }
        | SessionEventPayload::ToolExecutionCompleted { .. }
        | SessionEventPayload::ApprovalRequested { .. }
        | SessionEventPayload::TurnCompleted { .. }
        | SessionEventPayload::TurnFailed { .. }
        | SessionEventPayload::Lagged { .. }
        | SessionEventPayload::SessionShutdownComplete => {}
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
        && a.compacted_prefix == b.compacted_prefix
        && a.usage_so_far == b.usage_so_far
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use serde_json::json;

    use super::*;
    use crate::{
        AssistantMessage, CompactionEventEffects, Delivery, MessageId, PendingEvent, ReplayMeta,
        SessionId, StopReason, Usage, UserMessage,
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
                })),
            },
        );

        assert_eq!(state.messages, window);
        assert_eq!(state.compacted_prefix, vec![json!({"kind": "prefix"})]);
        assert_eq!(state.last_response_id, None);
        assert_eq!(state.messages_seen_by_provider, 0);
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
    }
}
