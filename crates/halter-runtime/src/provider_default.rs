//! Provider-delegated compaction: the provider's native compaction rewrites
//! the context, on Halter's trigger.
// pattern: Imperative Shell

use async_trait::async_trait;
use halter_protocol::{
    CompactedContext, CompactionResult, Message, ProviderCapabilities, ProviderCompactionRejected,
    ProviderCompactionRequest, ProviderCompactionStrategy, SessionState,
};

use crate::compaction_strategy::{
    CompactionContext, CompactionStrategy, CompactionTrigger, compaction_instructions,
};
use crate::context::CompactionEffects;

#[derive(Debug, Clone, Copy, Default)]
/// Delegate the rewrite to [`Provider::compact`](halter_providers::Provider::compact):
/// OpenAI's dedicated `/v1/responses/compact` endpoint, or an inline
/// summarization request for Anthropic and OpenRouter. The window a rewrite
/// may replace follows the provider's
/// [`compaction_strategy`](halter_protocol::ProviderCapabilities::compaction_strategy);
/// a provider without one cannot be selected, which `HalterBuilder::build`
/// enforces before any compaction runs.
///
/// Triggered exclusively by the runtime's token ledger; no provider-side
/// auto-compaction is ever enabled.
pub struct ProviderDefault;

#[async_trait]
impl CompactionStrategy for ProviderDefault {
    /// Whether any message would leave the verbatim window. When none would,
    /// a rewrite could only re-summarize the existing prefix: worth a
    /// provider call when a caller asks for it, never on the ledger's
    /// trigger. A provider that cannot compact answers `true`, so
    /// [`compact`](Self::compact) reports why.
    fn needs_compaction(
        &self,
        state: &SessionState,
        capabilities: &ProviderCapabilities,
        trigger: CompactionTrigger<'_>,
    ) -> bool {
        capabilities
            .compaction_strategy
            .filter(|_| capabilities.supports_compaction)
            .is_none_or(|strategy| {
                let (_, preserved) = split_window(strategy, &state.messages);
                state.messages.len() > preserved.len()
                    || (!state.compacted_prefix.is_empty()
                        && matches!(trigger, CompactionTrigger::Manual { .. }))
            })
    }

    async fn compact(
        &self,
        ctx: CompactionContext<'_>,
    ) -> anyhow::Result<Option<CompactionEffects>> {
        let capabilities = ctx.provider().capabilities();
        let Some(strategy) = capabilities
            .compaction_strategy
            .filter(|_| capabilities.supports_compaction)
        else {
            anyhow::bail!(
                "failed to compact session: provider '{}' does not support compaction",
                ctx.model().provider
            );
        };
        let state = ctx.state();
        if !self.needs_compaction(state, &capabilities, ctx.trigger()) {
            return Ok(None);
        }
        let (eligible, preserved) = split_window(strategy, &state.messages);
        let compacted_count = state.messages.len() - preserved.len();

        let response = ctx
            .provider()
            .compact(
                ProviderCompactionRequest {
                    session_id: ctx.blueprint().session_id.clone(),
                    model: ctx.model().clone(),
                    compacted_prefix: state.compacted_prefix.clone(),
                    messages: eligible.clone(),
                    tools: ctx.tool_specs(),
                    instructions: compaction_instructions(ctx.trigger().custom_instructions()),
                },
                ctx.cancel().clone(),
            )
            .await?;
        if response.output.is_empty() {
            // Billed all the same: the runtime counts a rejection's usage.
            return Err(ProviderCompactionRejected {
                reason: format!(
                    "failed to compact session: provider '{}' returned no compacted context",
                    ctx.model().provider
                ),
                text: String::new(),
                usage: response.usage,
            }
            .into());
        }
        let summary = format!(
            "Compacted {compacted_count} older messages into {} provider-native items; kept {} verbatim.",
            response.output.len(),
            preserved.len()
        );

        Ok(Some(CompactionEffects {
            messages: preserved,
            compacted_context: CompactedContext::from(response.output),
            result: CompactionResult {
                compacted_count,
                summary,
            },
            usage: response.usage,
        }))
    }
}

/// Which messages a rewrite may replace, as `(eligible, preserved)`.
///
/// A dedicated endpoint restores compacted context as provider-native items,
/// so everything before the latest assistant block is eligible and that
/// block stays verbatim. Both shapes preserve chronology because
/// provider-native compacted items are encoded before `messages` in the
/// next request.
///
/// An inline summary is re-injected as system text, so the window must
/// still open with the latest user message: it stays verbatim, followed by
/// the latest assistant block of its turn and that block's tool results.
/// Everything before that block is eligible — including the user message,
/// so the summary knows what the work it describes was for — which lets a
/// long tool-using turn shed its older exchanges without splitting a call
/// from its result. A turn with no assistant block yet keeps the user-led
/// suffix and offers only the prefix before it.
pub(crate) fn split_window(
    strategy: ProviderCompactionStrategy,
    messages: &[Message],
) -> (Vec<Message>, Vec<Message>) {
    let last = |pick: fn(&Message) -> bool| messages.iter().rposition(pick);
    match strategy {
        ProviderCompactionStrategy::Dedicated => {
            let pivot =
                last(|message| matches!(message, Message::Assistant(_))).unwrap_or(messages.len());
            (messages[..pivot].to_vec(), messages[pivot..].to_vec())
        }
        ProviderCompactionStrategy::Inline => {
            let Some(user) = last(|message| matches!(message, Message::User(_))) else {
                return (messages.to_vec(), Vec::new());
            };
            match last(|message| matches!(message, Message::Assistant(_))) {
                Some(block) if block > user => (
                    messages[..block].to_vec(),
                    std::iter::once(&messages[user])
                        .chain(&messages[block..])
                        .cloned()
                        .collect(),
                ),
                _ => (messages[..user].to_vec(), messages[user..].to_vec()),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use halter_protocol::{
        AssistantMessage, AssistantPart, MessageId, ToolCallId, ToolResult, ToolResultMessage,
        UserMessage,
    };

    use super::*;

    fn user(text: &str) -> Message {
        Message::User(UserMessage::text(text))
    }

    fn assistant(text: &str) -> Message {
        Message::Assistant(AssistantMessage {
            id: MessageId::new(),
            created_at: Utc::now(),
            parts: vec![AssistantPart::Text {
                text: text.to_owned(),
            }],
            stop_reason: None,
            usage: None,
            replay_meta: Default::default(),
        })
    }

    fn tool(call_id: &str) -> Message {
        Message::Tool(ToolResultMessage {
            id: MessageId::new(),
            call_id: ToolCallId::from(call_id),
            content: ToolResult::Empty,
            error: None,
            created_at: Utc::now(),
        })
    }

    fn labels(messages: &[Message]) -> Vec<String> {
        messages
            .iter()
            .map(|message| match message {
                Message::User(user) => user.plain_text(),
                Message::Assistant(assistant) => match &assistant.parts[0] {
                    AssistantPart::Text { text } => text.clone(),
                    _ => unreachable!(),
                },
                Message::Tool(tool) => tool.call_id.to_string(),
                Message::System(_) => unreachable!(),
            })
            .collect()
    }

    #[test]
    fn split_window_follows_the_provider_strategy() {
        // (case, strategy, transcript, eligible, preserved)
        type Case = (
            &'static str,
            ProviderCompactionStrategy,
            Vec<Message>,
            &'static [&'static str],
            &'static [&'static str],
        );
        use ProviderCompactionStrategy::{Dedicated, Inline};
        let cases: [Case; 9] = [
            (
                "dedicated keeps the latest assistant block",
                Dedicated,
                vec![user("first"), assistant("answer"), user("follow up")],
                &["first"],
                &["answer", "follow up"],
            ),
            (
                "dedicated without an assistant compacts everything",
                Dedicated,
                vec![user("first"), user("second")],
                &["first", "second"],
                &[],
            ),
            (
                // The shape of every refused or no-op pass in the eval runs:
                // one user instruction, then a long run of tool exchanges.
                "inline sheds the older exchanges of a long tool turn",
                Inline,
                vec![
                    user("task"),
                    assistant("a1"),
                    tool("t1"),
                    assistant("a2"),
                    tool("t2"),
                    assistant("a3"),
                    tool("t3"),
                ],
                &["task", "a1", "t1", "a2", "t2"],
                &["task", "a3", "t3"],
            ),
            (
                "inline folds prior turns and the current turn's older exchanges",
                Inline,
                vec![
                    user("first"),
                    assistant("answer"),
                    user("latest"),
                    assistant("a1"),
                    tool("t1"),
                    assistant("tail"),
                ],
                &["first", "answer", "latest", "a1", "t1"],
                &["latest", "tail"],
            ),
            (
                "inline at the start of a turn keeps the user-led suffix",
                Inline,
                vec![user("first"), assistant("answer"), user("latest")],
                &["first", "answer"],
                &["latest"],
            ),
            (
                "inline with one exchange in the turn has nothing to shed",
                Inline,
                vec![user("only user"), assistant("large"), tool("t1")],
                &["only user"],
                &["only user", "large", "t1"],
            ),
            (
                "inline without a user compacts everything",
                Inline,
                vec![assistant("only")],
                &["only"],
                &[],
            ),
            (
                "dedicated empty transcript",
                Dedicated,
                Vec::new(),
                &[],
                &[],
            ),
            ("inline empty transcript", Inline, Vec::new(), &[], &[]),
        ];

        for (case, strategy, messages, eligible, preserved) in cases {
            let (got_eligible, got_preserved) = split_window(strategy, &messages);
            assert_eq!(labels(&got_eligible), eligible, "{case}: eligible");
            assert_eq!(labels(&got_preserved), preserved, "{case}: preserved");
            // The verbatim window replays on its own after the summary, so it
            // must open with a user message when one exists and never hold a
            // tool result whose call is not in front of it.
            if strategy == Inline && messages.iter().any(|m| matches!(m, Message::User(_))) {
                assert!(
                    matches!(got_preserved.first(), Some(Message::User(_))),
                    "{case}: window opens with the user"
                );
            }
            if let Some(Message::Tool(_)) = got_preserved.get(1) {
                panic!("{case}: a tool result follows the user without its call");
            }
        }
    }

    /// The runtime asks this before any hook fires, so a `false` here is
    /// what keeps `PreCompact`/`PostCompact` quiet at a busy boundary.
    /// Kills dropping either clause of the `||` (the shedding and manual
    /// prefix cases flip), dropping the `Manual` match (the automatic
    /// prefix case flips), and answering `false` for a provider that cannot
    /// compact (its error would turn into a silent skip).
    #[test]
    fn needs_compaction_when_a_message_leaves_the_window_or_a_caller_asks() {
        use ProviderCompactionStrategy::{Dedicated, Inline};
        let manual = CompactionTrigger::Manual {
            custom_instructions: None,
        };
        let one_exchange = || vec![user("only"), assistant("large"), tool("t1")];
        let long_turn = || vec![user("task"), assistant("a1"), tool("t1"), assistant("a2")];
        // (case, provider strategy, transcript, has prefix, trigger, needs)
        let cases: [(&str, _, Vec<Message>, bool, CompactionTrigger<'_>, bool); 9] = [
            (
                "provider cannot compact",
                None,
                one_exchange(),
                false,
                CompactionTrigger::Automatic,
                true,
            ),
            (
                "inline long turn sheds",
                Some(Inline),
                long_turn(),
                false,
                CompactionTrigger::Automatic,
                true,
            ),
            (
                "inline lone exchange, no prefix",
                Some(Inline),
                one_exchange(),
                false,
                CompactionTrigger::Automatic,
                false,
            ),
            (
                "inline lone exchange, automatic prefix-only rewrite",
                Some(Inline),
                one_exchange(),
                true,
                CompactionTrigger::Automatic,
                false,
            ),
            (
                "inline lone exchange, manual prefix-only rewrite",
                Some(Inline),
                one_exchange(),
                true,
                manual,
                true,
            ),
            (
                "inline lone exchange, manual without a prefix",
                Some(Inline),
                one_exchange(),
                false,
                manual,
                false,
            ),
            (
                "dedicated sheds before the latest assistant",
                Some(Dedicated),
                one_exchange(),
                false,
                CompactionTrigger::Automatic,
                true,
            ),
            (
                "dedicated empty transcript",
                Some(Dedicated),
                Vec::new(),
                false,
                manual,
                false,
            ),
            (
                "rollover prefix-only rewrite",
                Some(Inline),
                one_exchange(),
                true,
                CompactionTrigger::Rollover { requested: true },
                false,
            ),
        ];
        for (case, strategy, messages, has_prefix, trigger, needs) in cases {
            let capabilities = ProviderCapabilities {
                supports_compaction: strategy.is_some(),
                compaction_strategy: strategy,
                ..ProviderCapabilities::default()
            };
            let state = SessionState {
                messages,
                compacted_prefix: if has_prefix {
                    vec![serde_json::json!({"type": "compaction", "text": "gist"})]
                } else {
                    Vec::new()
                },
                ..SessionState::default()
            };
            assert_eq!(
                ProviderDefault.needs_compaction(&state, &capabilities, trigger),
                needs,
                "{case}"
            );
        }
    }
}
