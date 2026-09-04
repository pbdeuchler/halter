//! The compaction strategy seam. The runtime owns *when* — the session
//! token ledger, the two trigger points, and the consistent boundaries
//! compaction may run at — and a [`CompactionStrategy`] owns *what happens*.
// pattern: Imperative Shell

use std::sync::Arc;

use async_trait::async_trait;
use halter_protocol::{
    CompactedContext, CompactionResult, Message, PromptSegment, ProviderCompactionRequest,
    ResolvedModel, SessionBlueprint, SessionState, ToolSpec,
};
use halter_providers::Provider;
use halter_tools::Tool;
use tokio_util::sync::CancellationToken;

use crate::compaction::{ContextSettings, prepare_compaction, render_compaction_event_summary};
use crate::context::CompactionEffects;

/// Why a compaction pass is running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionTrigger<'a> {
    /// The token ledger reached `context.compaction_threshold`. A failure
    /// degrades the turn to an uncompacted context with a warning event.
    Automatic,
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
            Self::Automatic => None,
            Self::Manual {
                custom_instructions,
            } => custom_instructions,
        }
    }
}

/// Everything a strategy may read while compacting one session.
pub struct CompactionContext<'a> {
    pub blueprint: &'a SessionBlueprint,
    /// Read view of the session, including its `token_ledger`.
    pub state: &'a SessionState,
    /// Prompt segments the next request carries: seed, skills, hook context.
    pub prompt_segments: &'a [PromptSegment],
    /// Specs of every tool registered on the runtime.
    pub tool_specs: &'a [ToolSpec],
    /// The session's default model, which compaction runs against.
    pub model: &'a ResolvedModel,
    /// Provider serving `model`.
    pub provider: &'a (dyn Provider + Send + Sync),
    pub trigger: CompactionTrigger<'a>,
    /// Cancelled when the surrounding turn is cancelled.
    pub cancel: CancellationToken,
}

#[async_trait]
/// Rewrites session context when the runtime decides it is time.
///
/// # Halter owns the trigger
///
/// The runtime alone decides when compaction runs. Every append updates the
/// session's [`TokenLedger`](halter_protocol::TokenLedger); at the next
/// consistent boundary — no assistant tool call still awaiting its result —
/// a ledger at or past `context.compaction_threshold` invokes
/// [`compact`](Self::compact). **Server-side or provider-automatic compaction
/// is never used.** A strategy must not enable a provider's own
/// auto-compaction: the ledger, the `previous_response_id` chain, and the
/// committed event log would silently diverge from what the provider holds.
/// Strategies are told *when*; they decide *what*.
///
/// A strategy is shared by every session of a harness, so it holds
/// configuration rather than per-session state; per-session facts arrive
/// through [`CompactionContext`].
pub trait CompactionStrategy: Send + Sync {
    /// Tools the strategy needs on every session. `HalterBuilder::build`
    /// registers them after the built-ins; a same-named tool replaces the
    /// built-in entry.
    fn tools(&self) -> Vec<Arc<dyn Tool>> {
        Vec::new()
    }

    /// System-prompt segments appended to every new session's seed.
    fn prompt_segments(&self) -> Vec<PromptSegment> {
        Vec::new()
    }

    /// Invoked at each consistent boundary with the ledger's effective count
    /// at the previous boundary and now, against the configured threshold.
    /// Return messages to append — reminders as the context fills — and the
    /// runtime appends and emits them like any other message.
    fn threshold_notifications(
        &self,
        _previous_tokens: u64,
        _current_tokens: u64,
        _compaction_threshold: u64,
    ) -> Vec<Message> {
        Vec::new()
    }

    /// Compact the session. `Ok(None)` means there was nothing to compact and
    /// state is left untouched; `Err` means compaction could not run, which
    /// the runtime degrades to a warning for [`CompactionTrigger::Automatic`]
    /// and propagates for [`CompactionTrigger::Manual`].
    async fn compact(
        &self,
        ctx: CompactionContext<'_>,
    ) -> anyhow::Result<Option<CompactionEffects>>;
}

#[derive(Debug, Clone, Copy)]
/// The provider-delegated strategy: prune low-signal units toward
/// `pre_compaction_target`, then hand the provider's own compaction window
/// to [`Provider::compact`] and carry its output as the compacted prefix.
pub struct ProviderCompaction {
    settings: ContextSettings,
}

impl ProviderCompaction {
    /// Construct from the context settings that govern pruning.
    #[must_use]
    pub fn new(settings: ContextSettings) -> Self {
        Self { settings }
    }

    /// Settings this strategy prunes with.
    #[must_use]
    pub fn settings(&self) -> ContextSettings {
        self.settings
    }
}

#[async_trait]
impl CompactionStrategy for ProviderCompaction {
    async fn compact(
        &self,
        ctx: CompactionContext<'_>,
    ) -> anyhow::Result<Option<CompactionEffects>> {
        if !ctx.provider.capabilities().supports_compaction {
            anyhow::bail!(
                "failed to compact session: provider '{}' does not support compaction",
                ctx.model.provider
            );
        }
        let Some(window) = ctx.provider.compaction_window(&ctx.state.messages) else {
            anyhow::bail!(
                "failed to compact session: provider '{}' did not provide a compaction window",
                ctx.model.provider
            );
        };
        let compacted_context = CompactedContext::from(ctx.state.compacted_prefix.clone());
        let preparation = prepare_compaction(&self.settings, &compacted_context, window);
        if compacted_context.is_empty() && preparation.compact_messages.is_empty() {
            return Ok(None);
        }

        let response = ctx
            .provider
            .compact(
                ProviderCompactionRequest {
                    session_id: ctx.blueprint.session_id.clone(),
                    model: ctx.model.clone(),
                    compacted_prefix: ctx.state.compacted_prefix.clone(),
                    messages: preparation.compact_messages.clone(),
                    tools: ctx.tool_specs.to_vec(),
                    instructions: compaction_instructions(ctx.trigger.custom_instructions()),
                },
                ctx.cancel,
            )
            .await?;
        let summary = render_compaction_event_summary(
            preparation.compacted_message_count,
            response.output.len(),
            preparation.evicted_unit_count,
            preparation.reserved_response_block,
        );

        Ok(Some(CompactionEffects {
            messages: preparation.preserved_messages,
            compacted_context: CompactedContext::from(response.output),
            result: CompactionResult {
                compacted_count: preparation.compacted_message_count,
                summary,
            },
        }))
    }
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
    use std::sync::Mutex;

    use halter_protocol::{
        ModelId, ModelRole, ProviderCapabilities, ProviderKind, ProviderName, ResolvedModel,
        SessionId, SubagentEventForwarding, ToolCallIdPolicy, Usage, UserMessage,
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

    fn one_message_state() -> SessionState {
        let mut state = SessionState::default();
        state.append(Message::User(UserMessage::text("hello")));
        state
    }

    async fn compact_with(
        provider: &StubProvider,
        state: &SessionState,
        trigger: CompactionTrigger<'_>,
    ) -> anyhow::Result<Option<CompactionEffects>> {
        ProviderCompaction::new(ContextSettings {
            compaction_threshold: 1,
            pre_compaction_target: 0,
            ..ContextSettings::default()
        })
        .compact(CompactionContext {
            blueprint: &sample_blueprint(),
            state,
            prompt_segments: &[],
            tool_specs: &[],
            model: &sample_model(),
            provider,
            trigger,
            cancel: CancellationToken::new(),
        })
        .await
    }

    #[tokio::test]
    async fn working_provider_yields_effects_for_the_compacted_window() {
        let provider = StubProvider::working();
        let mut state = SessionState::default();
        state.append(Message::User(UserMessage::text("old")));
        state.append(halter_protocol::Message::Assistant(
            halter_protocol::AssistantMessage {
                id: halter_protocol::MessageId::new(),
                created_at: chrono::Utc::now(),
                parts: vec![halter_protocol::AssistantPart::Text {
                    text: "latest".to_owned(),
                }],
                stop_reason: None,
                usage: None,
                replay_meta: Default::default(),
            },
        ));

        let effects = compact_with(&provider, &state, CompactionTrigger::Automatic)
            .await
            .expect("compaction succeeds")
            .expect("something to compact");

        // The stub preserves the latest assistant block and compacts the rest.
        assert_eq!(effects.messages.len(), 1);
        assert!(matches!(effects.messages[0], Message::Assistant(_)));
        assert_eq!(effects.compacted_context.len(), 1);
        assert_eq!(effects.result.compacted_count, 1);
        assert!(
            effects
                .result
                .summary
                .contains("Compacted 1 older messages")
        );
        let request = provider.last_request().expect("provider was called");
        assert_eq!(request.messages.len(), 1);
        assert_eq!(
            request.instructions,
            crate::prompt::default_compaction_prompt()
        );
    }

    #[tokio::test]
    async fn manual_trigger_forwards_custom_instructions() {
        let provider = StubProvider::working();

        compact_with(
            &provider,
            &one_message_state(),
            CompactionTrigger::Manual {
                custom_instructions: Some("Focus on decisions."),
            },
        )
        .await
        .expect("compaction succeeds")
        .expect("something to compact");

        let request = provider.last_request().expect("provider was called");
        assert!(
            request
                .instructions
                .starts_with(crate::prompt::default_compaction_prompt())
        );
        assert!(request.instructions.ends_with("Focus on decisions."));
    }

    #[tokio::test]
    async fn empty_context_has_nothing_to_compact() {
        let provider = StubProvider::working();

        let effects = compact_with(
            &provider,
            &SessionState::default(),
            CompactionTrigger::Automatic,
        )
        .await
        .expect("compaction succeeds");

        assert!(effects.is_none());
        assert!(
            provider.last_request().is_none(),
            "provider must not be called"
        );
    }

    /// The three ways the provider path cannot run. The strategy reports
    /// them; the runtime decides whether that degrades (automatic) or
    /// propagates (manual).
    #[tokio::test]
    async fn provider_problems_are_reported_as_errors() {
        struct Case {
            name: &'static str,
            provider: StubProvider,
            expected: &'static str,
        }
        let cases = [
            Case {
                name: "unsupported",
                provider: StubProvider::without_compaction(),
                expected: "provider 'fake' does not support compaction",
            },
            Case {
                name: "no window",
                provider: StubProvider::without_window(),
                expected: "provider 'fake' did not provide a compaction window",
            },
            Case {
                name: "failing endpoint",
                provider: StubProvider::failing("compaction endpoint exploded"),
                expected: "compaction endpoint exploded",
            },
        ];

        for case in cases {
            let error = compact_with(
                &case.provider,
                &one_message_state(),
                CompactionTrigger::Automatic,
            )
            .await
            .expect_err(case.name);
            assert!(
                error.to_string().contains(case.expected),
                "{}: expected {:?} in {error:#}",
                case.name,
                case.expected
            );
        }
    }

    #[test]
    fn default_contributions_are_empty() {
        let strategy = ProviderCompaction::new(ContextSettings::default());
        assert!(strategy.tools().is_empty());
        assert!(strategy.prompt_segments().is_empty());
        assert!(strategy.threshold_notifications(0, 1_000, 1_000).is_empty());
        assert_eq!(strategy.settings().compaction_threshold, 80_000);
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

    /// Compaction provider stub. `supports_compaction` drives the advertised
    /// capability, `offers_window` whether a compaction window is produced,
    /// and `compact_error` makes the compaction call fail.
    struct StubProvider {
        supports_compaction: bool,
        offers_window: bool,
        compact_error: Option<&'static str>,
        last_request: Mutex<Option<ProviderCompactionRequest>>,
    }

    impl StubProvider {
        fn working() -> Self {
            Self {
                supports_compaction: true,
                offers_window: true,
                compact_error: None,
                last_request: Mutex::new(None),
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

        fn last_request(&self) -> Option<ProviderCompactionRequest> {
            self.last_request.lock().expect("lock").clone()
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
            _cancel: CancellationToken,
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
            request: ProviderCompactionRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<halter_protocol::ProviderCompactionResponse> {
            *self.last_request.lock().expect("lock") = Some(request);
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
}
