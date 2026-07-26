// pattern: Imperative Shell

use async_trait::async_trait;
use halter_protocol::{
    CompactedContext, CompactionEventEffects, CompactionResult, ContextPlan, FileViewSlice,
    HookSessionStartSource, Message, ObservedState, PromptSegment, ProviderCompactionRequest,
    ResolvedModel, ResourceSnapshot, SessionBlueprint, SessionState, ToolSpec, TranscriptWindow,
};
use halter_providers::Provider;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tracing::{info, warn};

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

#[derive(Debug, Clone)]
/// Result of a compaction pass before it is applied to session state.
pub struct CompactionOutcome {
    pub messages: Vec<Message>,
    pub compacted_prefix: Vec<Value>,
    pub compaction: Option<CompactionResult>,
    /// Why automatic compaction did not run, when it was due but failed.
    /// Always `None` for manual compaction, which propagates its errors
    /// instead.
    pub compaction_error: Option<String>,
    pub session_start_latch: Option<HookSessionStartSource>,
}

#[derive(Debug, Clone)]
/// State mutation produced by compaction.
pub struct CompactionEffects {
    pub messages: Vec<Message>,
    pub compacted_context: CompactedContext,
    pub result: Option<CompactionResult>,
    pub session_start_latch: Option<HookSessionStartSource>,
}

impl CompactionEffects {
    /// Apply compaction side effects to session state.
    pub fn apply(self, state: &mut SessionState) -> Option<CompactionResult> {
        let CompactionEffects {
            messages,
            compacted_context,
            result,
            session_start_latch,
        } = self;
        if result.is_some() {
            state.compacted_prefix = compacted_context.into_items();
            state.messages = messages;
            // Compaction breaks the previous_response_id chain: the provider
            // has no record of the synthetic `compacted_prefix` we just
            // injected, so the next request must replay everything.
            state.last_response_id = None;
            state.messages_seen_by_provider = 0;
            // Every usage report in the preserved tail describes the
            // pre-compaction context and would re-trigger compaction on the
            // very next turn.
            state.usage_anchor_floor = state.messages.len();
        }
        if let Some(source) = session_start_latch {
            state.pending_session_start_source = Some(source);
        }
        result
    }
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
        self.into_effects().apply(state)
    }

    /// Apply the outcome and, when compaction fired, also return the
    /// state-complete [`CompactionEventEffects`] the `ContextCompacted`
    /// event must carry so the event log alone can reproduce the rewrite.
    pub fn apply_with_effects(
        self,
        state: &mut SessionState,
    ) -> Option<(CompactionResult, CompactionEventEffects)> {
        let effects = self.into_effects();
        let record = CompactionEventEffects {
            messages: effects.messages.clone(),
            compacted_prefix: effects.compacted_context.items().to_vec(),
        };
        effects.apply(state).map(|result| (result, record))
    }

    /// Usage-anchor floor this outcome implies. Compaction invalidates every
    /// report in the tail it preserved; otherwise the session's existing
    /// floor still stands.
    fn usage_anchor_floor(&self, state: &SessionState) -> usize {
        if self.compaction.is_some() {
            self.messages.len()
        } else {
            state.usage_anchor_floor
        }
    }

    fn into_effects(self) -> CompactionEffects {
        let CompactionOutcome {
            messages,
            compacted_prefix,
            compaction,
            compaction_error: _,
            session_start_latch,
        } = self;
        CompactionEffects {
            messages,
            compacted_context: CompactedContext::from(compacted_prefix),
            result: compaction,
            session_start_latch,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum CompactionMode<'a> {
    AutoThreshold,
    Manual {
        custom_instructions: Option<&'a str>,
    },
}

impl<'a> CompactionMode<'a> {
    fn is_forced(self) -> bool {
        matches!(self, Self::Manual { .. })
    }

    fn custom_instructions(self) -> Option<&'a str> {
        match self {
            Self::AutoThreshold => None,
            Self::Manual {
                custom_instructions,
            } => custom_instructions,
        }
    }

    fn session_start_latch(self) -> Option<HookSessionStartSource> {
        match self {
            Self::AutoThreshold => None,
            Self::Manual { .. } => Some(HookSessionStartSource::Compact),
        }
    }
}

#[async_trait]
#[allow(clippy::too_many_arguments)]
/// Builds context plans and performs compaction.
pub trait ContextManager: Send + Sync {
    /// Plan the next provider request.
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

    /// Force a compaction pass, optionally with additional instructions.
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
/// Default context manager using heuristic token estimates and signal pruning.
pub struct DefaultContextManager {
    settings: ContextSettings,
}

impl DefaultContextManager {
    /// Construct from explicit compaction settings.
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

    /// Construct from a settings struct.
    #[must_use]
    pub fn from_settings(settings: ContextSettings) -> Self {
        Self { settings }
    }

    /// Current context settings.
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
        mode: CompactionMode<'_>,
    ) -> anyhow::Result<CompactionOutcome> {
        let estimated_tokens = estimate_context_tokens(
            prompt_segments,
            &state.summaries,
            &state.compacted_prefix,
            &state.messages,
            state.usage_anchor_floor,
        );
        if !mode.is_forced() && !should_trigger_compaction(estimated_tokens, &self.settings) {
            return Ok(uncompacted_outcome(state, mode, None));
        }

        let capabilities = compaction_provider.capabilities();
        if !capabilities.supports_compaction {
            return degrade_or_fail(
                state,
                mode,
                format!(
                    "failed to compact session: provider '{}' does not support compaction",
                    compaction_model.provider
                ),
            );
        }

        let Some(window) = compaction_provider.compaction_window(&state.messages) else {
            return degrade_or_fail(
                state,
                mode,
                format!(
                    "failed to compact session: provider '{}' did not provide a compaction window",
                    compaction_model.provider
                ),
            );
        };
        let compacted_context = CompactedContext::from(state.compacted_prefix.clone());
        let preparation = prepare_compaction(&self.settings, &compacted_context, window);
        if compacted_context.is_empty() && preparation.compact_messages.is_empty() {
            return Ok(uncompacted_outcome(state, mode, None));
        }

        let response = match compaction_provider
            .compact(
                ProviderCompactionRequest {
                    session_id: blueprint.session_id.clone(),
                    model: compaction_model.clone(),
                    compacted_prefix: state.compacted_prefix.clone(),
                    messages: preparation.compact_messages.clone(),
                    tools: tool_specs.to_vec(),
                    instructions: compaction_instructions(mode.custom_instructions()),
                },
                tokio_util::sync::CancellationToken::new(),
            )
            .await
        {
            Ok(response) => response,
            Err(error) => return degrade_or_fail(state, mode, format!("{error:#}")),
        };
        let summary = render_compaction_event_summary(
            preparation.compacted_message_count,
            response.output.len(),
            preparation.evicted_unit_count,
            preparation.reserved_response_block,
        );

        Ok(CompactionOutcome {
            messages: preparation.preserved_messages,
            compacted_prefix: response.output,
            compaction: Some(CompactionResult {
                compacted_count: preparation.compacted_message_count,
                summary,
            }),
            compaction_error: None,
            session_start_latch: mode.session_start_latch(),
        })
    }
}

/// Automatic compaction is best-effort: a provider that cannot compact must
/// degrade the turn to an uncompacted context rather than fail it, because
/// the alternative is that every turn past the threshold becomes unrecoverable.
/// Manual compaction propagates instead — the caller asked for compaction
/// explicitly and needs to know it did not happen.
fn degrade_or_fail(
    state: &SessionState,
    mode: CompactionMode<'_>,
    error: String,
) -> anyhow::Result<CompactionOutcome> {
    if mode.is_forced() {
        anyhow::bail!(error);
    }
    warn!(error, "automatic compaction failed; continuing uncompacted");
    Ok(uncompacted_outcome(state, mode, Some(error)))
}

/// The turn's state left exactly as it was, carrying any reason compaction
/// did not run.
fn uncompacted_outcome(
    state: &SessionState,
    mode: CompactionMode<'_>,
    compaction_error: Option<String>,
) -> CompactionOutcome {
    CompactionOutcome {
        messages: state.messages.clone(),
        compacted_prefix: state.compacted_prefix.clone(),
        compaction: None,
        compaction_error,
        session_start_latch: mode.session_start_latch(),
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
                CompactionMode::AutoThreshold,
            )
            .await?;
        let estimated_tokens = estimate_context_tokens(
            &prompt_segments,
            &state.summaries,
            &outcome.compacted_prefix,
            &outcome.messages,
            outcome.usage_anchor_floor(state),
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
            compaction_warning: outcome.compaction_error,
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
            CompactionMode::Manual {
                custom_instructions,
            },
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
    let base = crate::prompt::default_compaction_prompt();
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

    fn sample_model() -> ResolvedModel {
        ResolvedModel {
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
        }
    }

    /// A manager whose threshold is low enough that every plan compacts.
    fn always_compacting_manager() -> DefaultContextManager {
        DefaultContextManager::new(1, 0, halter_protocol::PruneSignalThreshold::Normal)
    }

    async fn plan_with(
        manager: &DefaultContextManager,
        state: &SessionState,
        provider: &StubProvider,
    ) -> anyhow::Result<ContextPlan> {
        manager
            .plan(
                &sample_blueprint(),
                state,
                &sample_observed(),
                &ResourceSnapshot::empty(),
                &[],
                &sample_model(),
                provider,
            )
            .await
    }

    #[tokio::test]
    async fn plan_disables_previous_response_chaining_when_compacted_prefix_exists() {
        let outcome = plan_with(
            &DefaultContextManager::default(),
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
            &StubProvider::working(),
        )
        .await
        .expect("plan");

        assert!(outcome.previous_response_id.is_none());
    }

    /// A provider that cannot compact must not take the turn down with it.
    /// Before this, `plan()` propagated the error and every turn past the
    /// compaction threshold failed with no way to recover.
    #[tokio::test]
    async fn plan_degrades_to_uncompacted_context_when_compaction_fails() {
        let state = SessionState {
            messages: vec![Message::User(UserMessage::text("hello"))],
            ..SessionState::default()
        };

        let plan = plan_with(
            &always_compacting_manager(),
            &state,
            &StubProvider::failing("compaction endpoint exploded"),
        )
        .await
        .expect("plan must survive a failed compaction");

        assert!(plan.compaction.is_none());
        assert_eq!(plan.messages, state.messages);
        assert!(
            plan.compaction_warning
                .as_deref()
                .is_some_and(|warning| warning.contains("compaction endpoint exploded")),
            "expected the provider error to be carried out, got {:?}",
            plan.compaction_warning
        );
    }

    #[tokio::test]
    async fn plan_degrades_when_provider_does_not_support_compaction() {
        let plan = plan_with(
            &always_compacting_manager(),
            &SessionState {
                messages: vec![Message::User(UserMessage::text("hello"))],
                ..SessionState::default()
            },
            &StubProvider::without_compaction(),
        )
        .await
        .expect("plan must survive a provider that cannot compact");

        assert!(plan.compaction.is_none());
        assert!(
            plan.compaction_warning
                .as_deref()
                .is_some_and(|warning| warning.contains("does not support compaction")),
            "got {:?}",
            plan.compaction_warning
        );
    }

    /// A provider advertising compaction but yielding no window used to
    /// disable compaction silently on the auto path. It is a misconfiguration,
    /// so it must be reported rather than swallowed.
    #[tokio::test]
    async fn plan_degrades_when_provider_yields_no_compaction_window() {
        let plan = plan_with(
            &always_compacting_manager(),
            &SessionState {
                messages: vec![Message::User(UserMessage::text("hello"))],
                ..SessionState::default()
            },
            &StubProvider::without_window(),
        )
        .await
        .expect("plan must survive a provider that yields no window");

        assert!(plan.compaction.is_none());
        assert!(
            plan.compaction_warning
                .as_deref()
                .is_some_and(|warning| warning.contains("did not provide a compaction window")),
            "got {:?}",
            plan.compaction_warning
        );
    }

    /// Regression guard for a compaction loop. Compaction preserves a tail of
    /// real messages, and the assistant message in that tail still reports the
    /// pre-compaction context size. If the estimator kept anchoring on it,
    /// every subsequent turn would see the old (large) figure and compact
    /// again — forever. The floor advance is what stops that.
    #[tokio::test]
    async fn compaction_does_not_retrigger_on_the_following_turn() {
        let manager = DefaultContextManager::new(
            /*compaction_threshold*/ 1_000,
            /*pre_compaction_target*/ 500,
            halter_protocol::PruneSignalThreshold::Normal,
        );
        let mut state = SessionState {
            messages: vec![
                Message::User(UserMessage::text("hello")),
                Message::Assistant(halter_protocol::AssistantMessage {
                    id: halter_protocol::MessageId::new(),
                    created_at: Utc::now(),
                    parts: vec![halter_protocol::AssistantPart::Text {
                        text: "done".to_owned(),
                    }],
                    stop_reason: Some(halter_protocol::StopReason::EndTurn),
                    usage: Some(Usage {
                        input_tokens: 80_000,
                        output_tokens: 500,
                        cache_creation_input_tokens: 0,
                        cache_read_input_tokens: 0,
                    }),
                    replay_meta: Default::default(),
                }),
            ],
            ..SessionState::default()
        };

        let first = plan_with(&manager, &state, &StubProvider::working())
            .await
            .expect("first plan");
        assert!(
            first.compaction.is_some(),
            "80_500 reported tokens must exceed the 1_000 threshold"
        );

        let outcome = CompactionOutcome {
            messages: first.messages.clone(),
            compacted_prefix: first.compacted_prefix.clone(),
            compaction: first.compaction.clone(),
            compaction_error: None,
            session_start_latch: None,
        };
        outcome.apply(&mut state);
        assert_eq!(state.usage_anchor_floor, state.messages.len());

        let second = plan_with(&manager, &state, &StubProvider::working())
            .await
            .expect("second plan");

        assert!(
            second.compaction.is_none(),
            "stale pre-compaction usage must not re-trigger compaction"
        );
    }

    #[tokio::test]
    async fn plan_reports_no_warning_when_compaction_succeeds() {
        let plan = plan_with(
            &always_compacting_manager(),
            &SessionState {
                messages: vec![Message::User(UserMessage::text("hello"))],
                ..SessionState::default()
            },
            &StubProvider::working(),
        )
        .await
        .expect("plan");

        assert!(plan.compaction.is_some());
        assert_eq!(plan.compaction_warning, None);
    }

    /// Manual compaction keeps propagating: the caller asked for it
    /// explicitly and a silent no-op would be a lie.
    #[tokio::test]
    async fn compact_now_propagates_provider_failures() {
        let error = always_compacting_manager()
            .compact_now(
                &sample_blueprint(),
                &SessionState {
                    messages: vec![Message::User(UserMessage::text("hello"))],
                    ..SessionState::default()
                },
                &sample_observed(),
                &ResourceSnapshot::empty(),
                &[],
                &sample_model(),
                &StubProvider::failing("compaction endpoint exploded"),
                None,
            )
            .await
            .expect_err("manual compaction must surface provider failures");

        assert!(error.to_string().contains("compaction endpoint exploded"));
    }

    #[tokio::test]
    async fn compact_now_propagates_unsupported_compaction() {
        let error = always_compacting_manager()
            .compact_now(
                &sample_blueprint(),
                &SessionState {
                    messages: vec![Message::User(UserMessage::text("hello"))],
                    ..SessionState::default()
                },
                &sample_observed(),
                &ResourceSnapshot::empty(),
                &[],
                &sample_model(),
                &StubProvider::without_compaction(),
                None,
            )
            .await
            .expect_err("manual compaction must surface unsupported providers");

        assert!(error.to_string().contains("does not support compaction"));
    }

    /// Compaction provider stub. `supports_compaction` drives the advertised
    /// capability, `offers_window` whether a compaction window is produced,
    /// and `compact_error` makes the compaction call fail.
    struct StubProvider {
        supports_compaction: bool,
        offers_window: bool,
        compact_error: Option<&'static str>,
    }

    impl StubProvider {
        fn working() -> Self {
            Self {
                supports_compaction: true,
                offers_window: true,
                compact_error: None,
            }
        }

        fn failing(error: &'static str) -> Self {
            Self {
                compact_error: Some(error),
                ..Self::working()
            }
        }

        fn without_compaction() -> Self {
            Self {
                supports_compaction: false,
                ..Self::working()
            }
        }

        /// Advertises compaction but never yields a window — the shape a
        /// provider takes when it forgets to override `compaction_window`.
        fn without_window() -> Self {
            Self {
                offers_window: false,
                ..Self::working()
            }
        }
    }

    #[async_trait]
    impl Provider for StubProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                supports_compaction: self.supports_compaction,
                tool_call_id_policy: ToolCallIdPolicy::ProviderSupplied,
                ..ProviderCapabilities::default()
            }
        }

        fn compaction_window(
            &self,
            messages: &[Message],
        ) -> Option<halter_protocol::CompactionWindow> {
            self.offers_window.then(|| {
                halter_protocol::CompactionWindow::preserve_latest_assistant_response_block(
                    messages,
                )
            })
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
            if let Some(error) = self.compact_error {
                anyhow::bail!(error);
            }
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
    fn apply_with_effects_returns_event_record_matching_applied_state() {
        let mut state = SessionState {
            messages: vec![
                Message::User(halter_protocol::UserMessage::text("old-1")),
                Message::User(halter_protocol::UserMessage::text("old-2")),
            ],
            last_response_id: Some("resp-1".to_owned()),
            messages_seen_by_provider: 2,
            ..SessionState::default()
        };
        let window = vec![Message::User(halter_protocol::UserMessage::text("kept"))];
        let prefix = vec![serde_json::json!({"kind": "compacted"})];
        let outcome = CompactionOutcome {
            messages: window.clone(),
            compacted_prefix: prefix.clone(),
            compaction: Some(CompactionResult {
                compacted_count: 2,
                summary: "squashed".to_owned(),
            }),
            compaction_error: None,
            session_start_latch: None,
        };

        let (result, effects) = outcome
            .apply_with_effects(&mut state)
            .expect("compaction fired");

        assert_eq!(result.summary, "squashed");
        assert_eq!(effects.messages, state.messages);
        assert_eq!(effects.compacted_prefix, state.compacted_prefix);
        assert_eq!(state.messages, window);
        assert_eq!(state.compacted_prefix, prefix);
        assert_eq!(state.last_response_id, None);
        assert_eq!(state.messages_seen_by_provider, 0);
    }

    #[test]
    fn apply_with_effects_without_compaction_leaves_state_untouched() {
        let original = SessionState {
            messages: vec![Message::User(halter_protocol::UserMessage::text("kept"))],
            last_response_id: Some("resp-1".to_owned()),
            ..SessionState::default()
        };
        let mut state = original.clone();
        let outcome = CompactionOutcome {
            messages: Vec::new(),
            compacted_prefix: Vec::new(),
            compaction: None,
            compaction_error: None,
            session_start_latch: None,
        };

        assert!(outcome.apply_with_effects(&mut state).is_none());
        assert_eq!(state, original);
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
            crate::prompt::default_compaction_prompt()
        );
    }
}
