//! The compaction strategy seam. The runtime owns *when* — the session
//! token ledger, the two trigger points, and the consistent boundaries
//! compaction may run at — and a [`CompactionStrategy`] owns *what happens*.
// pattern: Imperative Shell

use std::collections::BTreeSet;
use std::sync::Arc;

use async_trait::async_trait;
use halter_protocol::{
    AssistantMessage, Message, ObservedState, PendingEvent, PromptSegment, ProviderRequest,
    ResolvedModel, ResourceSnapshot, SessionBlueprint, SessionEventPayload, SessionId,
    SessionState, ToolCall, ToolSpec, TurnId, Usage,
};
use halter_providers::Provider;
use halter_tools::Tool;
use tokio_util::sync::CancellationToken;

use crate::context::CompactionEffects;
use crate::session::SessionHandle;

/// Why a compaction pass is running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionTrigger<'a> {
    /// The boundary policy requested ordinary compaction. A failure degrades
    /// the turn to an uncompacted context with a warning event.
    Automatic,
    /// A boundary policy requested a clean logical-window rollover. This is
    /// distinct from threshold compaction so strategies such as CleanWindow
    /// can wipe context instead of summarizing it.
    Rollover,
    /// `HalterSession::compact` was called. A failure propagates to the
    /// caller, who asked for compaction explicitly and needs to know it did
    /// not happen.
    Manual {
        /// Extra instructions the caller supplied for this pass.
        custom_instructions: Option<&'a str>,
    },
}

impl<'a> CompactionTrigger<'a> {
    /// Caller-supplied instructions, when this is a manual pass.
    #[must_use]
    pub fn custom_instructions(self) -> Option<&'a str> {
        match self {
            Self::Automatic | Self::Rollover => None,
            Self::Manual {
                custom_instructions,
            } => custom_instructions,
        }
    }
}

/// Runtime action requested by a strategy at a consistent context boundary.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum CompactionDirective {
    /// Keep accumulating context.
    #[default]
    Continue,
    /// Run ordinary automatic compaction.
    Compact,
    /// Force a clean logical-window rollover.
    Rollover,
}

/// One exactly-once message a strategy wants delivered in this window.
#[derive(Debug, Clone)]
pub struct CompactionNotification {
    pub id: String,
    pub message: Message,
}

/// Result of evaluating one consistent context boundary.
#[derive(Debug, Clone, Default)]
pub struct CompactionBoundaryResult {
    pub notifications: Vec<CompactionNotification>,
    pub directive: CompactionDirective,
}

impl CompactionBoundaryResult {
    #[must_use]
    pub const fn compact() -> Self {
        Self {
            notifications: Vec::new(),
            directive: CompactionDirective::Compact,
        }
    }

    #[must_use]
    pub const fn rollover() -> Self {
        Self {
            notifications: Vec::new(),
            directive: CompactionDirective::Rollover,
        }
    }

    #[must_use]
    pub fn with_notification(mut self, id: impl Into<String>, message: Message) -> Self {
        self.notifications.push(CompactionNotification {
            id: id.into(),
            message,
        });
        self
    }
}

/// Per-session facts supplied when the runtime reaches a consistent context
/// boundary. Notification ids are persisted by the runtime and scoped to
/// `window`, so a stateless strategy can provide exactly-once milestones.
pub struct CompactionBoundary<'a> {
    session_id: &'a SessionId,
    window: u64,
    previous_tokens: u64,
    current_tokens: u64,
    compaction_threshold: u64,
    delivered_notifications: &'a BTreeSet<String>,
}

impl<'a> CompactionBoundary<'a> {
    pub(crate) fn new(
        session_id: &'a SessionId,
        window: u64,
        previous_tokens: u64,
        current_tokens: u64,
        compaction_threshold: u64,
        delivered_notifications: &'a BTreeSet<String>,
    ) -> Self {
        Self {
            session_id,
            window,
            previous_tokens,
            current_tokens,
            compaction_threshold,
            delivered_notifications,
        }
    }

    #[must_use]
    pub fn session_id(&self) -> &SessionId {
        self.session_id
    }

    #[must_use]
    pub const fn window(&self) -> u64 {
        self.window
    }

    #[must_use]
    pub const fn previous_tokens(&self) -> u64 {
        self.previous_tokens
    }

    #[must_use]
    pub const fn current_tokens(&self) -> u64 {
        self.current_tokens
    }

    #[must_use]
    pub const fn compaction_threshold(&self) -> u64 {
        self.compaction_threshold
    }

    #[must_use]
    pub fn notification_was_delivered(&self, id: &str) -> bool {
        self.delivered_notifications.contains(id)
    }
}

/// The built-in checkpoint instructions, with a manual pass's custom
/// instructions appended. What [`ModelSummary`](crate::ModelSummary) asks
/// the session's model and what [`ProviderDefault`](crate::ProviderDefault)
/// hands the provider.
#[must_use]
pub fn compaction_instructions(custom_instructions: Option<&str>) -> String {
    let base = crate::prompt::default_compaction_prompt();
    match custom_instructions.filter(|instructions| !instructions.trim().is_empty()) {
        Some(custom_instructions) => format!("{base}\n\n{custom_instructions}"),
        None => base.to_owned(),
    }
}

/// Everything a strategy may read and do while compacting one session.
///
/// Reads see the live session: its state (including the token ledger), the
/// prompt segments and tool specs the next request would carry, the default
/// model and its provider. Actions run through the runtime's own machinery
/// so a strategy that talks to the model or executes tools gets the same
/// planning, prompt caching, hooks, policy, and event log as a turn:
///
/// - [`append`](Self::append) puts a message in the transcript and the log,
/// - [`infer`](Self::infer) runs one inference over the transcript,
/// - [`execute_tool_calls`](Self::execute_tool_calls) runs tool calls.
///
/// Every message appended this way is replaced when the returned
/// [`CompactionEffects`] apply; the event log keeps the whole exchange.
pub struct CompactionContext<'a> {
    session: &'a SessionHandle,
    blueprint: &'a SessionBlueprint,
    snapshot: Arc<ResourceSnapshot>,
    state: &'a mut SessionState,
    events: &'a mut Vec<PendingEvent>,
    fired_hook_ids: &'a mut BTreeSet<String>,
    turn_id: &'a TurnId,
    observed: ObservedState,
    model: ResolvedModel,
    provider: Arc<dyn Provider>,
    trigger: CompactionTrigger<'a>,
    cancel: CancellationToken,
    usage: &'a mut Usage,
}

impl<'a> CompactionContext<'a> {
    #[expect(clippy::too_many_arguments)]
    pub(crate) fn new(
        session: &'a SessionHandle,
        blueprint: &'a SessionBlueprint,
        snapshot: Arc<ResourceSnapshot>,
        state: &'a mut SessionState,
        events: &'a mut Vec<PendingEvent>,
        fired_hook_ids: &'a mut BTreeSet<String>,
        turn_id: &'a TurnId,
        observed: ObservedState,
        model: ResolvedModel,
        provider: Arc<dyn Provider>,
        trigger: CompactionTrigger<'a>,
        cancel: CancellationToken,
        usage: &'a mut Usage,
    ) -> Self {
        Self {
            session,
            blueprint,
            snapshot,
            state,
            events,
            fired_hook_ids,
            turn_id,
            observed,
            model,
            provider,
            trigger,
            cancel,
            usage,
        }
    }

    /// The session's immutable blueprint.
    #[must_use]
    pub fn blueprint(&self) -> &SessionBlueprint {
        self.blueprint
    }

    /// Read view of the session, including its `token_ledger`.
    #[must_use]
    pub fn state(&self) -> &SessionState {
        self.state
    }

    /// The session's default model, which compaction runs against.
    #[must_use]
    pub fn model(&self) -> &ResolvedModel {
        &self.model
    }

    /// Provider serving [`model`](Self::model).
    #[must_use]
    pub fn provider(&self) -> &dyn Provider {
        self.provider.as_ref()
    }

    /// Why this pass is running.
    #[must_use]
    pub fn trigger(&self) -> CompactionTrigger<'a> {
        self.trigger
    }

    /// Cancelled when the surrounding turn is cancelled.
    #[must_use]
    pub fn cancel(&self) -> &CancellationToken {
        &self.cancel
    }

    /// Prompt segments the next request carries: seed, skills, hook context.
    #[must_use]
    pub fn prompt_segments(&self) -> Vec<PromptSegment> {
        crate::context::prompt_segments(self.blueprint, self.state, &self.snapshot)
    }

    /// Specs of every tool registered on the runtime.
    #[must_use]
    pub fn tool_specs(&self) -> Vec<ToolSpec> {
        self.session.services().tools.specs()
    }

    /// Append a message to the transcript and record it in the event log.
    pub fn append(&mut self, message: Message) {
        self.state.append(message.clone());
        self.record(message);
    }

    /// Append a message to the transcript without recording it. For a reply
    /// the log already holds in full ([`infer`](Self::infer) records every
    /// reply) when only part of it should reach the model next.
    pub fn append_unlogged(&mut self, message: Message) {
        self.state.append(message);
    }

    /// Record a message in the event log without changing the transcript.
    pub fn record(&mut self, message: Message) {
        self.session
            .push_event(self.events, SessionEventPayload::MessageItem { message });
    }

    /// Run one inference over the current transcript with the session's
    /// static prefix and tools, exactly as a turn would build it, so the
    /// request shares the turn's prompt cache. The reply is recorded in the
    /// event log and its usage accounted, but it is *not* appended: the
    /// strategy decides what, if anything, of it reaches the transcript.
    pub async fn infer(&mut self) -> anyhow::Result<AssistantMessage> {
        let prompt_segments = self.prompt_segments();
        let tool_specs = self.tool_specs();
        let request_tokens =
            halter_protocol::estimate_request_tokens(&prompt_segments, &tool_specs);
        if self
            .state
            .token_ledger
            .needs_request_preparation(request_tokens)
        {
            let compacted_prefix = &self.state.compacted_prefix;
            let messages = &self.state.messages;
            self.state
                .token_ledger
                .prepare_request(request_tokens, compacted_prefix, messages);
            self.session.push_event(
                self.events,
                SessionEventPayload::ContextProjectionUpdated { request_tokens },
            );
        }
        let (plan, prompt) = self
            .session
            .plan_and_assemble(self.blueprint, &self.snapshot, self.state, &self.observed)
            .await?;
        let request = ProviderRequest {
            session_id: self.session.session_id().clone(),
            turn_id: self.turn_id.clone(),
            model: self.model.clone(),
            prompt,
            compacted_prefix: plan.compacted_prefix,
            messages: plan.messages,
            tools: plan.tool_specs,
            previous_response_id: plan.previous_response_id,
            new_messages_start: plan.new_messages_start,
        };
        let materialized = self
            .session
            .run_provider_request(
                &self.model,
                self.provider.as_ref(),
                request,
                self.cancel.child_token(),
            )
            .await?;
        self.state
            .usage_so_far
            .saturating_accumulate(&materialized.usage);
        self.usage.saturating_accumulate(&materialized.usage);
        for payload in materialized.events {
            self.session.push_event(self.events, payload);
        }
        self.record(Message::Assistant(materialized.message.clone()));
        Ok(materialized.message)
    }

    /// Execute tool calls through the turn's tool path: hooks, policy,
    /// pending-call accounting, events. Every result is appended to the
    /// transcript and recorded; the appended messages are returned.
    pub async fn execute_tool_calls(
        &mut self,
        calls: Vec<ToolCall>,
    ) -> anyhow::Result<Vec<Message>> {
        let before = self.state.messages.len();
        let events = self
            .session
            .execute_tool_calls(
                self.blueprint,
                self.snapshot.clone(),
                self.cancel.child_token(),
                &self.blueprint.default_model,
                &self.blueprint.subagent_model,
                self.turn_id,
                self.fired_hook_ids,
                self.state,
                calls,
            )
            .await?;
        self.events.extend(events);
        Ok(self.state.messages[before..].to_vec())
    }
}

#[async_trait]
/// Rewrites session context when the runtime decides it is time.
///
/// # Halter owns the trigger
///
/// The runtime alone decides when compaction runs. Every append updates the
/// session's [`TokenLedger`](halter_protocol::TokenLedger); at the next
/// consistent boundary — no assistant tool call still awaiting its result —
/// [`context_boundary`](Self::context_boundary) chooses whether to invoke
/// [`compact`](Self::compact). Its default does so at or past
/// `context.compaction_threshold`. **Server-side or provider-automatic
/// compaction is never used.** A strategy must not enable a provider's own
/// auto-compaction: the ledger, the `previous_response_id` chain, and the
/// committed event log would silently diverge from what the provider holds.
/// Strategies are told *when*; they decide *what*.
///
/// A strategy is shared by every session of a harness, so it holds
/// configuration rather than per-session state; per-session facts and
/// actions go through [`CompactionContext`].
pub trait CompactionStrategy: Send + Sync {
    /// Tools the strategy needs on every session. `HalterBuilder::build`
    /// registers them after the built-ins and before `with_tool` entries, so
    /// an explicitly supplied tool of the same name wins.
    fn tools(&self) -> Vec<Arc<dyn Tool>> {
        Vec::new()
    }

    /// System-prompt segments appended to every new session's seed.
    fn prompt_segments(&self) -> Vec<PromptSegment> {
        Vec::new()
    }

    /// Decide what happens at a consistent per-session boundary. The default
    /// requests ordinary compaction exactly at the configured threshold.
    /// Notification ids returned here are deduplicated and persisted by the
    /// runtime for the current logical window.
    fn context_boundary(&self, boundary: CompactionBoundary<'_>) -> CompactionBoundaryResult {
        if boundary.current_tokens() >= boundary.compaction_threshold() {
            CompactionBoundaryResult::compact()
        } else {
            CompactionBoundaryResult::default()
        }
    }

    /// Compact the session. `Ok(None)` means there was nothing to compact and
    /// the transcript is left as the strategy found it; `Err` means
    /// compaction could not run, which the runtime degrades to a warning for
    /// [`CompactionTrigger::Automatic`] and propagates for
    /// [`CompactionTrigger::Manual`]. Either way, anything the strategy
    /// appended through the context is committed only when effects are
    /// returned successfully. `Err` and `Ok(None)` roll state and audit
    /// events back to the pre-strategy checkpoint.
    async fn compact(
        &self,
        ctx: CompactionContext<'_>,
    ) -> anyhow::Result<Option<CompactionEffects>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoopStrategy;

    #[async_trait]
    impl CompactionStrategy for NoopStrategy {
        async fn compact(
            &self,
            _ctx: CompactionContext<'_>,
        ) -> anyhow::Result<Option<CompactionEffects>> {
            Ok(None)
        }
    }

    #[test]
    fn default_boundary_policy_compacts_at_the_exact_threshold() {
        let session_id = SessionId::from("boundary-test");
        let notifications = BTreeSet::new();
        for (tokens, expected) in [
            (999, CompactionDirective::Continue),
            (1_000, CompactionDirective::Compact),
        ] {
            let boundary =
                CompactionBoundary::new(&session_id, 0, 0, tokens, 1_000, &notifications);
            assert_eq!(NoopStrategy.context_boundary(boundary).directive, expected);
        }
    }

    #[test]
    fn trigger_exposes_custom_instructions_only_for_manual_passes() {
        assert_eq!(CompactionTrigger::Automatic.custom_instructions(), None);
        assert_eq!(
            CompactionTrigger::Manual {
                custom_instructions: Some("x")
            }
            .custom_instructions(),
            Some("x")
        );
    }

    #[test]
    fn compaction_instructions_append_custom_text() {
        let instructions = compaction_instructions(Some("Focus on decisions."));
        assert!(instructions.starts_with(crate::prompt::default_compaction_prompt()));
        assert!(instructions.ends_with("Focus on decisions."));
    }

    #[test]
    fn compaction_instructions_ignore_blank_custom_text() {
        assert_eq!(
            compaction_instructions(Some("   ")),
            crate::prompt::default_compaction_prompt()
        );
        assert_eq!(
            compaction_instructions(None),
            crate::prompt::default_compaction_prompt()
        );
    }
}
