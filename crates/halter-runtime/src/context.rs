// pattern: Imperative Shell

use async_trait::async_trait;
use halter_protocol::{
    CompactedContext, CompactionEventEffects, CompactionResult, ContextPlan, FileViewSlice,
    Message, ObservedState, PromptSegment, ResourceSnapshot, SessionBlueprint, SessionEventPayload,
    SessionState, ToolSpec, TranscriptWindow,
};
use sha2::{Digest, Sha256};

use crate::prompt::skill_prompt_segment;

/// Build one prompt segment per skill loaded into the resource snapshot,
/// in skill-name order so the resulting prefix is stable across rebuilds.
/// Snapshot order is already deterministic (`IndexMap`), but we still sort
/// by name to be defensive against future loader changes.
fn skill_prompt_segments(snapshot: &ResourceSnapshot) -> Vec<PromptSegment> {
    let mut entries: Vec<(&str, &str)> = snapshot
        .skills
        .values()
        .map(|skill| (skill.name.as_str(), skill.body.as_str()))
        .collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    entries
        .into_iter()
        .map(|(name, body)| skill_prompt_segment(name, body))
        .collect()
}

/// Prompt segments the next request carries: the session's system-prompt
/// seed, loaded skills, and hook-appended context, in that order.
#[must_use]
pub fn prompt_segments(
    blueprint: &SessionBlueprint,
    state: &SessionState,
    snapshot: &ResourceSnapshot,
) -> Vec<PromptSegment> {
    let mut segments = blueprint.system_prompt_seed.clone();
    segments.extend(skill_prompt_segments(snapshot));
    segments.extend(state.appended_prompt_segments.clone());
    segments
}

#[derive(Debug, Clone)]
/// State rewrite produced by a [`CompactionStrategy`](crate::CompactionStrategy).
pub struct CompactionEffects {
    /// Message window that replaces `SessionState::messages`.
    pub messages: Vec<Message>,
    /// Provider-native items that replace `SessionState::compacted_prefix`.
    pub compacted_context: CompactedContext,
    /// What happened, for the `ContextCompacted` event and PostCompact hooks.
    pub result: CompactionResult,
}

impl CompactionEffects {
    /// Apply the rewrite to session state and return the result together
    /// with the `ContextCompacted` payload that records it.
    ///
    /// The transition itself is [`halter_protocol::fold::apply_event`], the
    /// same function replay uses, so a live session and a session rebuilt
    /// from its event log land on identical state: the window and prefix are
    /// replaced, the `previous_response_id` chain is broken (the provider has
    /// no record of the synthetic prefix), and the token ledger is rebuilt
    /// from the compacted context because every usage report in the
    /// preserved tail describes the pre-compaction transcript.
    pub fn apply(self, state: &mut SessionState) -> (CompactionResult, SessionEventPayload) {
        let CompactionEffects {
            messages,
            compacted_context,
            result,
        } = self;
        let payload = SessionEventPayload::ContextCompacted {
            summary: result.summary.clone(),
            effects: Some(Box::new(CompactionEventEffects {
                messages,
                compacted_prefix: compacted_context.into_items(),
            })),
        };
        halter_protocol::fold::apply_event(state, &payload);
        (result, payload)
    }
}

#[async_trait]
/// Builds the context plan for the next provider request.
///
/// Planning is read-only. Compaction is decided by the runtime's token
/// ledger and executed through the configured
/// [`CompactionStrategy`](crate::CompactionStrategy) *before* `plan` runs, so
/// the plan describes the state it is given.
pub trait ContextManager: Send + Sync {
    /// Plan the next provider request.
    async fn plan(
        &self,
        blueprint: &SessionBlueprint,
        state: &SessionState,
        observed: &ObservedState,
        snapshot: &ResourceSnapshot,
        tool_specs: &[ToolSpec],
    ) -> anyhow::Result<ContextPlan>;
}

#[derive(Debug, Default, Clone, Copy)]
/// Default context manager: the full transcript window, the carried
/// compacted prefix, and the ledger's effective count as the size estimate.
pub struct DefaultContextManager;

#[async_trait]
impl ContextManager for DefaultContextManager {
    async fn plan(
        &self,
        blueprint: &SessionBlueprint,
        state: &SessionState,
        observed: &ObservedState,
        snapshot: &ResourceSnapshot,
        tool_specs: &[ToolSpec],
    ) -> anyhow::Result<ContextPlan> {
        let prompt_segments = prompt_segments(blueprint, state, snapshot);

        let file_views = state
            .file_view_cache
            .values()
            .cloned()
            .map(|entry| FileViewSlice {
                path: entry.path,
                full_hash: entry.full_hash,
                viewed_ranges: entry.viewed_ranges,
                last_shown_turn: entry.last_shown_turn,
            })
            .collect::<Vec<_>>();

        let estimated_tokens = state.token_ledger.effective_tokens();
        let (previous_response_id, new_messages_start) = resolve_response_chain(
            state.last_response_id.as_deref(),
            state.messages_seen_by_provider,
            state.messages.len(),
            !state.compacted_prefix.is_empty(),
        );
        let previous_response_id = previous_response_id.map(|s| s.to_owned());

        Ok(ContextPlan {
            prompt_segments,
            transcript_window: TranscriptWindow {
                messages: state.messages.clone(),
                elided_message_count: 0,
            },
            compacted_prefix: state.compacted_prefix.clone(),
            file_views,
            carried_summaries: state.summaries.clone(),
            elided_tool_results: Vec::new(),
            memory_items: Vec::new(),
            tool_specs: tool_specs.to_vec(),
            observed_state: observed.clone(),
            projected_input_tokens: estimated_tokens,
            cache_boundary_hash: cache_boundary_hash(),
            messages: state.messages.clone(),
            estimated_tokens,
            previous_response_id,
            new_messages_start,
        })
    }
}

fn cache_boundary_hash() -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"transcript_boundary_v2");
    format!("{:x}", hasher.finalize())
}

/// Determines whether a request should reuse the provider's
/// `previous_response_id` chain, and if so, from which message in the
/// transcript the "new since the provider last saw us" slice begins.
///
/// Chaining requires that no compacted prefix is in play: the provider has no
/// record of it, so the request must replay everything. (Compaction itself
/// clears `last_response_id`, so a rewrite in the current turn never chains
/// either.) When chaining is allowed, `new_messages_start` is the number of
/// messages at the head of the transcript the provider has already observed;
/// the caller sends the suffix only.
///
/// ```
/// # use halter_runtime::resolve_response_chain;
/// // No prior response: no chaining.
/// assert_eq!(resolve_response_chain(None, 0, 0, false), (None, 0));
///
/// // Clean turn, 6 messages, provider saw 4 → resume at 4.
/// assert_eq!(resolve_response_chain(Some("resp_1"), 4, 6, false), (Some("resp_1"), 4));
///
/// // The provider cannot have seen more than the transcript holds.
/// assert_eq!(resolve_response_chain(Some("resp_1"), 9, 6, false), (Some("resp_1"), 6));
///
/// // A compacted prefix is carried — must not chain.
/// assert_eq!(resolve_response_chain(Some("resp_1"), 4, 6, true), (None, 0));
/// ```
#[must_use]
pub fn resolve_response_chain(
    last_response_id: Option<&str>,
    messages_seen_by_provider: usize,
    total_messages: usize,
    has_compacted_prefix: bool,
) -> (Option<&str>, usize) {
    if has_compacted_prefix || messages_seen_by_provider == 0 || last_response_id.is_none() {
        return (None, 0);
    }
    (
        last_response_id,
        messages_seen_by_provider.min(total_messages),
    )
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use halter_protocol::{
        SessionId, SubagentEventForwarding, SummarySlice, TokenLedger, UserMessage,
    };

    use super::*;

    fn sample_blueprint() -> SessionBlueprint {
        SessionBlueprint {
            session_id: SessionId::new(),
            parent_session_id: None,
            default_model: "default".into(),
            subagent_model: "subagent".into(),
            subagent_event_forwarding: SubagentEventForwarding::Off,
            snapshot_revision: "r1".into(),
            working_dir: ".".into(),
            system_prompt_seed: Vec::new(),
            max_turns: None,
            subagent_depth: 0,
        }
    }

    fn sample_observed() -> ObservedState {
        ObservedState {
            cwd: ".".into(),
            git_branch: None,
            git_dirty: None,
            now_utc: Utc::now(),
            env_facts: Default::default(),
        }
    }

    async fn plan_with(state: &SessionState) -> ContextPlan {
        DefaultContextManager
            .plan(
                &sample_blueprint(),
                state,
                &sample_observed(),
                &ResourceSnapshot::empty(),
                &[],
            )
            .await
            .expect("plan")
    }

    #[tokio::test]
    async fn plan_disables_previous_response_chaining_when_compacted_prefix_exists() {
        let plan = plan_with(&SessionState {
            compacted_prefix: vec![serde_json::json!({
                "type": "compaction",
                "id": "cmp_1",
                "encrypted_content": "x",
            })],
            summaries: vec![SummarySlice {
                id: "summary-1".to_owned(),
                text: "summary".to_owned(),
            }],
            messages: vec![Message::User(UserMessage::text("hello"))],
            last_response_id: Some("resp_1".to_owned()),
            messages_seen_by_provider: 1,
            ..SessionState::default()
        })
        .await;

        assert!(plan.previous_response_id.is_none());
        assert_eq!(plan.new_messages_start, 0);
    }

    #[tokio::test]
    async fn plan_chains_and_carries_the_full_window_and_ledger_estimate() {
        let mut state = SessionState {
            last_response_id: Some("resp_1".to_owned()),
            messages_seen_by_provider: 1,
            ..SessionState::default()
        };
        state.append(Message::User(UserMessage::text("hello")));
        state.append(Message::User(UserMessage::text("and again")));

        let plan = plan_with(&state).await;

        assert_eq!(plan.previous_response_id.as_deref(), Some("resp_1"));
        assert_eq!(plan.new_messages_start, 1);
        assert_eq!(plan.messages, state.messages);
        assert_eq!(plan.transcript_window.messages, state.messages);
        assert_eq!(plan.transcript_window.elided_message_count, 0);
        assert_eq!(plan.estimated_tokens, state.token_ledger.effective_tokens());
        assert_eq!(plan.projected_input_tokens, plan.estimated_tokens);
    }

    #[test]
    fn apply_rewrites_state_and_returns_matching_event_payload() {
        let mut state = SessionState {
            last_response_id: Some("resp-1".to_owned()),
            messages_seen_by_provider: 2,
            token_ledger: TokenLedger {
                authoritative_tokens: 80_000,
                inferred_tokens: 0,
            },
            ..SessionState::default()
        };
        state.append(Message::User(UserMessage::text("old-1")));
        state.append(Message::User(UserMessage::text("old-2")));
        let window = vec![Message::User(UserMessage::text("kept"))];
        let prefix = vec![serde_json::json!({"kind": "compacted"})];
        let effects = CompactionEffects {
            messages: window.clone(),
            compacted_context: CompactedContext::from(prefix.clone()),
            result: CompactionResult {
                compacted_count: 2,
                summary: "squashed".to_owned(),
            },
        };

        let (result, payload) = effects.apply(&mut state);

        assert_eq!(result.summary, "squashed");
        assert_eq!(result.compacted_count, 2);
        assert_eq!(state.messages, window);
        assert_eq!(state.compacted_prefix, prefix);
        assert_eq!(state.last_response_id, None);
        assert_eq!(state.messages_seen_by_provider, 0);
        // The stale 80_000 anchor is gone; the ledger describes the compacted
        // state only.
        assert_eq!(
            state.token_ledger,
            TokenLedger::inferred_from(&prefix, &window, &[])
        );
        match payload {
            SessionEventPayload::ContextCompacted {
                summary,
                effects: Some(effects),
            } => {
                assert_eq!(summary, "squashed");
                assert_eq!(effects.messages, state.messages);
                assert_eq!(effects.compacted_prefix, state.compacted_prefix);
            }
            other => panic!("expected a state-complete ContextCompacted payload, got {other:?}"),
        }
    }

    /// Replaying the payload `apply` returns onto the pre-compaction state
    /// must land on the applied state: that is what makes the event log
    /// state-complete.
    #[test]
    fn apply_payload_replays_to_the_same_state() {
        let mut live = SessionState::default();
        live.append(Message::User(UserMessage::text("old")));
        let mut replayed = live.clone();
        let effects = CompactionEffects {
            messages: Vec::new(),
            compacted_context: CompactedContext::from(vec![serde_json::json!("p")]),
            result: CompactionResult {
                compacted_count: 1,
                summary: "squashed".to_owned(),
            },
        };

        let (_, payload) = effects.apply(&mut live);
        halter_protocol::fold::apply_event(&mut replayed, &payload);

        assert_eq!(replayed, live);
    }
}
