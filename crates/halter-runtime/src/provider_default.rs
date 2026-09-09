//! Provider-delegated compaction: the provider's native compaction rewrites
//! the context, on Halter's trigger.
// pattern: Imperative Shell

use async_trait::async_trait;
use halter_protocol::{
    CompactedContext, CompactionResult, Message, ProviderCompactionRequest,
    ProviderCompactionStrategy,
};

use crate::compaction_strategy::{CompactionContext, CompactionStrategy, compaction_instructions};
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
        let (eligible, preserved) = split_window(strategy, &state.messages);
        if eligible.is_empty() && state.compacted_prefix.is_empty() {
            return Ok(None);
        }

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
            anyhow::bail!(
                "failed to compact session: provider '{}' returned no compacted context",
                ctx.model().provider
            );
        }
        let summary = format!(
            "Compacted {} older messages into {} provider-native items; kept {} verbatim.",
            eligible.len(),
            response.output.len(),
            preserved.len()
        );

        Ok(Some(CompactionEffects {
            messages: preserved,
            compacted_context: CompactedContext::from(response.output),
            result: CompactionResult {
                compacted_count: eligible.len(),
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
/// block stays verbatim. An inline request summarizes the prefix before the
/// latest user message and keeps that user-led suffix verbatim. Both shapes
/// preserve chronology because provider-native compacted items are encoded
/// before `messages` in the next request.
pub(crate) fn split_window(
    strategy: ProviderCompactionStrategy,
    messages: &[Message],
) -> (Vec<Message>, Vec<Message>) {
    match strategy {
        ProviderCompactionStrategy::Dedicated => {
            let pivot = messages
                .iter()
                .rposition(|message| matches!(message, Message::Assistant(_)))
                .unwrap_or(messages.len());
            (messages[..pivot].to_vec(), messages[pivot..].to_vec())
        }
        ProviderCompactionStrategy::Inline => {
            let pivot = messages
                .iter()
                .rposition(|message| matches!(message, Message::User(_)))
                .unwrap_or(messages.len());

            (messages[..pivot].to_vec(), messages[pivot..].to_vec())
        }
    }
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use halter_protocol::{AssistantMessage, AssistantPart, MessageId, UserMessage};

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

    #[test]
    fn split_window_follows_the_provider_strategy() {
        struct Case {
            name: &'static str,
            strategy: ProviderCompactionStrategy,
            messages: Vec<Message>,
            eligible: usize,
            preserved_first: Option<&'static str>,
        }
        let cases = [
            Case {
                name: "dedicated keeps the latest assistant block",
                strategy: ProviderCompactionStrategy::Dedicated,
                messages: vec![user("first"), assistant("answer"), user("follow up")],
                eligible: 1,
                preserved_first: Some("answer"),
            },
            Case {
                name: "dedicated without an assistant compacts everything",
                strategy: ProviderCompactionStrategy::Dedicated,
                messages: vec![user("first"), user("second")],
                eligible: 2,
                preserved_first: None,
            },
            Case {
                name: "inline compacts the prefix before the latest user",
                strategy: ProviderCompactionStrategy::Inline,
                messages: vec![
                    user("first"),
                    assistant("answer"),
                    user("latest"),
                    assistant("tail"),
                ],
                eligible: 2,
                preserved_first: Some("latest"),
            },
            Case {
                name: "inline without a user compacts everything",
                strategy: ProviderCompactionStrategy::Inline,
                messages: vec![assistant("only")],
                eligible: 1,
                preserved_first: None,
            },
            Case {
                name: "empty transcript",
                strategy: ProviderCompactionStrategy::Inline,
                messages: Vec::new(),
                eligible: 0,
                preserved_first: None,
            },
        ];

        for case in cases {
            let (eligible, preserved) = split_window(case.strategy, &case.messages);
            assert_eq!(eligible.len(), case.eligible, "{}", case.name);
            assert_eq!(
                eligible.len() + preserved.len(),
                case.messages.len(),
                "{}: nothing is lost",
                case.name
            );
            let first = preserved.first().map(|message| match message {
                Message::User(user) => user.plain_text(),
                Message::Assistant(assistant) => match &assistant.parts[0] {
                    AssistantPart::Text { text } => text.clone(),
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            });
            assert_eq!(first.as_deref(), case.preserved_first, "{}", case.name);
        }
    }
}
