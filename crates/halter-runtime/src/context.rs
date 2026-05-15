// pattern: Imperative Shell

use async_trait::async_trait;
use halter_protocol::{
    CompactionResult, ContextPlan, FileViewSlice, Message, ObservedState, PromptSegment,
    ProviderCompactionRequest, ResolvedModel, ResourceSnapshot, SessionBlueprint, SessionState,
    ToolSpec, TranscriptWindow,
};
use halter_providers::Provider;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tracing::info;

use crate::compaction::{
    ContextSettings, estimate_context_tokens, prepare_compaction, render_compaction_event_summary,
    should_trigger_compaction,
};
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

const DEFAULT_COMPACTION_PROMPT_MARKDOWN: &str = include_str!("../prompts/default-compaction.md");

#[derive(Debug, Clone)]
pub struct CompactionOutcome {
    pub messages: Vec<Message>,
    pub compacted_prefix: Vec<Value>,
    pub compaction: Option<CompactionResult>,
}

impl CompactionOutcome {
    /// Apply the outcome to a `SessionState` in place. Used by both the
    /// turn loop and the manual `compact()` entry point so the rules for
    /// "what changes when compaction lands" live in one place rather than
    /// being copy-pasted into every caller.
    ///
    /// Returns the inner `CompactionResult` when compaction actually fired
    /// (so callers can publish the event), or `None` when there was
    /// nothing to compact and the state was left untouched.
    pub fn apply(self, state: &mut SessionState) -> Option<CompactionResult> {
        let CompactionOutcome {
            messages,
            compacted_prefix,
            compaction,
        } = self;
        let result = compaction?;
        state.compacted_prefix = compacted_prefix;
        state.messages = messages;
        // Compaction breaks the previous_response_id chain: the provider
        // has no record of the synthetic `compacted_prefix` we just
        // injected, so the next request must replay everything.
        state.last_response_id = None;
        state.messages_seen_by_provider = 0;
        Some(result)
    }
}

#[async_trait]
#[allow(clippy::too_many_arguments)]
pub trait ContextManager: Send + Sync {
    async fn plan(
        &self,
        blueprint: &SessionBlueprint,
        state: &SessionState,
        observed: &ObservedState,
        snapshot: &ResourceSnapshot,
        tool_specs: &[ToolSpec],
        compaction_model: &ResolvedModel,
        compaction_provider: &(dyn Provider + Send + Sync),
    ) -> anyhow::Result<ContextPlan>;

    async fn compact_now(
        &self,
        blueprint: &SessionBlueprint,
        state: &SessionState,
        observed: &ObservedState,
        snapshot: &ResourceSnapshot,
        tool_specs: &[ToolSpec],
        compaction_model: &ResolvedModel,
        compaction_provider: &(dyn Provider + Send + Sync),
        custom_instructions: Option<&str>,
    ) -> anyhow::Result<CompactionOutcome>;
}

#[derive(Debug, Default)]
pub struct DefaultContextManager {
    settings: ContextSettings,
}

impl DefaultContextManager {
    #[must_use]
    pub fn new(
        compaction_threshold: u64,
        pre_compaction_target: u64,
        prune_signal_threshold: halter_protocol::PruneSignalThreshold,
    ) -> Self {
        Self {
            settings: ContextSettings {
                compaction_threshold,
                pre_compaction_target,
                prune_signal_threshold,
            },
        }
    }

    #[must_use]
    pub fn from_settings(settings: ContextSettings) -> Self {
        Self { settings }
    }

    #[must_use]
    pub fn settings(&self) -> ContextSettings {
        self.settings
    }

    #[allow(clippy::too_many_arguments)]
    async fn execute_compaction(
        &self,
        blueprint: &SessionBlueprint,
        state: &SessionState,
        prompt_segments: &[PromptSegment],
        tool_specs: &[ToolSpec],
        compaction_model: &ResolvedModel,
        compaction_provider: &(dyn Provider + Send + Sync),
        force: bool,
        custom_instructions: Option<&str>,
    ) -> anyhow::Result<CompactionOutcome> {
        let estimated_tokens = estimate_context_tokens(
            prompt_segments,
            &state.summaries,
            &state.compacted_prefix,
            &state.messages,
        );
        if !force && !should_trigger_compaction(estimated_tokens, &self.settings) {
            return Ok(CompactionOutcome {
                messages: state.messages.clone(),
                compacted_prefix: state.compacted_prefix.clone(),
                compaction: None,
            });
        }

        let capabilities = compaction_provider.capabilities();
        if !capabilities.supports_compaction {
            anyhow::bail!(
                "failed to compact session: provider '{}' does not support compaction",
                compaction_model.provider
            );
        }

        let preparation = prepare_compaction(
            &self.settings,
            &state.compacted_prefix,
            &state.messages,
            capabilities.compaction_strategy,
        );
        if state.compacted_prefix.is_empty() && preparation.compact_messages.is_empty() {
            return Ok(CompactionOutcome {
                messages: state.messages.clone(),
                compacted_prefix: state.compacted_prefix.clone(),
                compaction: None,
            });
        }

        let response = compaction_provider
            .compact(
                ProviderCompactionRequest {
                    session_id: blueprint.session_id.clone(),
                    model: compaction_model.clone(),
                    compacted_prefix: state.compacted_prefix.clone(),
                    messages: preparation.compact_messages.clone(),
                    tools: tool_specs.to_vec(),
                    instructions: compaction_instructions(custom_instructions),
                },
                tokio_util::sync::CancellationToken::new(),
            )
            .await?;
        let summary = render_compaction_event_summary(
            preparation.compacted_message_count,
            response.output.len(),
            preparation.evicted_unit_count,
            preparation.reserved_response_block,
        );

        Ok(CompactionOutcome {
            messages: preparation.reserved_suffix,
            compacted_prefix: response.output,
            compaction: Some(CompactionResult {
                compacted_count: preparation.compacted_message_count,
                summary,
            }),
        })
    }
}

#[async_trait]
impl ContextManager for DefaultContextManager {
    async fn plan(
        &self,
        blueprint: &SessionBlueprint,
        state: &SessionState,
        observed: &ObservedState,
        snapshot: &ResourceSnapshot,
        tool_specs: &[ToolSpec],
        compaction_model: &ResolvedModel,
        compaction_provider: &(dyn Provider + Send + Sync),
    ) -> anyhow::Result<ContextPlan> {
        let mut prompt_segments = blueprint.system_prompt_seed.clone();
        prompt_segments.extend(skill_prompt_segments(snapshot));
        prompt_segments.extend(state.appended_prompt_segments.clone());

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

        let outcome = self
            .execute_compaction(
                blueprint,
                state,
                &prompt_segments,
                tool_specs,
                compaction_model,
                compaction_provider,
                false,
                None,
            )
            .await?;
        let estimated_tokens = estimate_context_tokens(
            &prompt_segments,
            &state.summaries,
            &outcome.compacted_prefix,
            &outcome.messages,
        );

        if let Some(compaction) = outcome.compaction.as_ref() {
            info!(
                compacted_messages = compaction.compacted_count,
                remaining_messages = outcome.messages.len(),
                compacted_prefix_items = outcome.compacted_prefix.len(),
                estimated_tokens,
                compaction_threshold = self.settings.compaction_threshold,
                "context manager compacted session state"
            );
        }

        let (previous_response_id, new_messages_start) = resolve_response_chain(
            state.last_response_id.as_deref(),
            state.messages_seen_by_provider,
            state.messages.len(),
            outcome.messages.len(),
            outcome.compaction.is_some(),
            !outcome.compacted_prefix.is_empty(),
        );
        let previous_response_id = previous_response_id.map(|s| s.to_owned());

        Ok(ContextPlan {
            prompt_segments,
            transcript_window: TranscriptWindow {
                messages: outcome.messages.clone(),
                elided_message_count: state.messages.len().saturating_sub(outcome.messages.len())
                    as u64,
            },
            compacted_prefix: outcome.compacted_prefix.clone(),
            file_views,
            carried_summaries: state.summaries.clone(),
            elided_tool_results: Vec::new(),
            memory_items: Vec::new(),
            tool_specs: tool_specs.to_vec(),
            observed_state: observed.clone(),
            projected_input_tokens: estimated_tokens,
            cache_boundary_hash: cache_boundary_hash(),
            messages: outcome.messages,
            estimated_tokens,
            compaction: outcome.compaction,
            previous_response_id,
            new_messages_start,
        })
    }

    async fn compact_now(
        &self,
        blueprint: &SessionBlueprint,
        state: &SessionState,
        _observed: &ObservedState,
        snapshot: &ResourceSnapshot,
        tool_specs: &[ToolSpec],
        compaction_model: &ResolvedModel,
        compaction_provider: &(dyn Provider + Send + Sync),
        custom_instructions: Option<&str>,
    ) -> anyhow::Result<CompactionOutcome> {
        let mut prompt_segments = blueprint.system_prompt_seed.clone();
        prompt_segments.extend(skill_prompt_segments(snapshot));
        prompt_segments.extend(state.appended_prompt_segments.clone());
        self.execute_compaction(
            blueprint,
            state,
            &prompt_segments,
            tool_specs,
            compaction_model,
            compaction_provider,
            true,
            custom_instructions,
        )
        .await
    }
}

fn cache_boundary_hash() -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"transcript_boundary_v2");
    format!("{:x}", hasher.finalize())
}

/// Determines whether a request should reuse the provider's
/// `previous_response_id` chain, and if so, from which message in the pruned
/// transcript window the "new since the provider last saw us" slice begins.
///
/// Chaining requires that no compaction happened *this* turn and that no prior
/// compacted prefix is in play — both conditions force a clean replay. When
/// chaining is allowed, `new_messages_start` is the number of messages at the
/// head of the window the provider has already observed; the caller sends the
/// suffix only.
///
/// ```
/// # use halter_runtime::resolve_response_chain;
/// // No prior response: no chaining.
/// assert_eq!(resolve_response_chain(None, 0, 0, 0, false, false), (None, 0));
///
/// // Clean turn, 6 total messages, provider saw 4, window has 6 → resume at 4.
/// let (id, start) = resolve_response_chain(Some("resp_1"), 4, 6, 6, false, false);
/// assert_eq!(id, Some("resp_1"));
/// assert_eq!(start, 4);
///
/// // A 2-message head was pruned: window has 4, provider saw 4 of the original
/// // 6 → the first 2 seen messages fell outside the window, resume at 2.
/// let (_, start) = resolve_response_chain(Some("resp_1"), 4, 6, 4, false, false);
/// assert_eq!(start, 2);
///
/// // Compaction fired this turn — must not chain.
/// assert_eq!(resolve_response_chain(Some("resp_1"), 4, 6, 6, true, false), (None, 0));
///
/// // A compacted prefix is already carried — must not chain.
/// assert_eq!(resolve_response_chain(Some("resp_1"), 4, 6, 6, false, true), (None, 0));
/// ```
#[must_use]
pub fn resolve_response_chain(
    last_response_id: Option<&str>,
    messages_seen_by_provider: usize,
    total_messages: usize,
    window_messages: usize,
    compacted_this_turn: bool,
    has_compacted_prefix: bool,
) -> (Option<&str>, usize) {
    if compacted_this_turn
        || has_compacted_prefix
        || messages_seen_by_provider == 0
        || last_response_id.is_none()
    {
        return (None, 0);
    }
    let window_offset = total_messages.saturating_sub(window_messages);
    let new_start = messages_seen_by_provider
        .saturating_sub(window_offset)
        .min(window_messages);
    (last_response_id, new_start)
}

fn compaction_instructions(custom_instructions: Option<&str>) -> String {
    let base = DEFAULT_COMPACTION_PROMPT_MARKDOWN.trim();
    if let Some(custom_instructions) =
        custom_instructions.filter(|instructions| !instructions.trim().is_empty())
    {
        format!("{base}\n\n{custom_instructions}")
    } else {
        base.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use halter_protocol::{
        ModelId, ModelRole, ProviderCapabilities, ProviderKind, ProviderName, ResolvedModel,
        SessionId, SubagentEventForwarding, SummarySlice, ToolCallIdPolicy, Usage, UserMessage,
    };

    use super::*;

    #[tokio::test]
    async fn plan_disables_previous_response_chaining_when_compacted_prefix_exists() {
        let manager = DefaultContextManager::default();
        let outcome = manager
            .plan(
                &SessionBlueprint {
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
                },
                &SessionState {
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
                },
                &ObservedState {
                    cwd: ".".into(),
                    git_branch: None,
                    git_dirty: None,
                    now_utc: Utc::now(),
                    env_facts: Default::default(),
                },
                &ResourceSnapshot::empty(),
                &[],
                &ResolvedModel {
                    role: ModelRole::default(),
                    id: ModelId::from("default"),
                    provider: ProviderName::from("fake"),
                    provider_kind: ProviderKind::Fake,
                    api_kind: halter_protocol::ApiKind::Fake,
                    model: "fake".to_owned(),
                    max_input_tokens: None,
                    max_output_tokens: None,
                    reasoning: None,
                    tokens_per_minute: None,
                },
                &NoopProvider,
            )
            .await
            .expect("plan");

        assert!(outcome.previous_response_id.is_none());
    }

    struct NoopProvider;

    #[async_trait]
    impl Provider for NoopProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                supports_compaction: true,
                tool_call_id_policy: ToolCallIdPolicy::ProviderSupplied,
                ..ProviderCapabilities::default()
            }
        }

        async fn stream(
            &self,
            _request: halter_protocol::ProviderRequest,
            _cancel: tokio_util::sync::CancellationToken,
        ) -> anyhow::Result<
            futures::stream::BoxStream<
                'static,
                Result<halter_protocol::StreamEvent, halter_protocol::ProviderError>,
            >,
        > {
            anyhow::bail!("stream should not be called in this test");
        }

        async fn compact(
            &self,
            _request: ProviderCompactionRequest,
            _cancel: tokio_util::sync::CancellationToken,
        ) -> anyhow::Result<halter_protocol::ProviderCompactionResponse> {
            Ok(halter_protocol::ProviderCompactionResponse {
                output: vec![serde_json::json!({
                    "type": "compaction",
                    "id": "cmp_1",
                    "encrypted_content": "summary",
                })],
                usage: Usage::default(),
            })
        }
    }

    #[test]
    fn compaction_instructions_append_custom_text() {
        let instructions = compaction_instructions(Some("Focus on decisions."));
        assert!(instructions.contains("Compress the conversation"));
        assert!(instructions.contains("Focus on decisions."));
    }

    #[test]
    fn compaction_instructions_ignore_blank_custom_text() {
        assert_eq!(
            compaction_instructions(Some("   ")),
            DEFAULT_COMPACTION_PROMPT_MARKDOWN.trim()
        );
    }
}
