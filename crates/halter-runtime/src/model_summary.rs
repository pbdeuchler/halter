//! The default compaction strategy: ask the session's own model for a context
//! checkpoint and start the next window from that summary alone.
// pattern: Imperative Shell

use async_trait::async_trait;
use halter_protocol::{
    AssistantMessage, AssistantPart, CompactedContext, CompactionResult, Message, ToolCall,
    UserMessage,
};

use crate::compaction_strategy::{CompactionContext, CompactionStrategy, compaction_instructions};
use crate::context::CompactionEffects;

/// Name of the built-in todo tool the nudge targets.
pub const TODO_TOOL_NAME: &str = "task";

/// Sent before the checkpoint when the todo tool is registered, so the model
/// persists its remaining work somewhere the summary cannot lose it.
pub const TODO_NUDGE: &str = "Compaction is imminent. Use the todo tool if not already to ensure that a high fidelity checklist of what else you need to accomplish is persisted. Keep your todos concise and unambiguous, do not include broader context and background.";

/// Framing that opens the user-role message the next window starts from.
pub const CHECKPOINT_PREFIX: &str = "Context checkpoint from your previous window:\n\n";

/// Appended to the checkpoint when the nudge persisted todos.
pub const TODO_REMINDER: &str =
    "\n\nYour todo list (the task tool) holds the remaining next steps; list it before continuing.";

#[derive(Debug, Clone, Copy, Default)]
/// Model-written context checkpoint. Three steps against the session's
/// default model, each sharing the turn's static prefix so they hit the
/// prompt cache:
///
/// 1. When the todo tool is registered, a nudge to persist the remaining
///    work. Any `task` calls in the reply run; its text and other tool
///    calls never reach the next request (the event log keeps the whole
///    reply).
/// 2. A checkpoint request: the built-in instructions plus a manual pass's
///    custom instructions.
/// 3. The reply becomes the next window: the static prefix plus the summary
///    as a single user-role message — the one first-message shape every
///    provider accepts, the strongest attention position, and byte-stable
///    for the system prompt's cache — with a todo reminder when step 1 ran
///    calls. No tail of old messages is kept; the summary carries
///    everything, which is also the only client-side shape that survives
///    preserved-thinking checks.
pub struct ModelSummary;

#[async_trait]
impl CompactionStrategy for ModelSummary {
    async fn compact(
        &self,
        mut ctx: CompactionContext<'_>,
    ) -> anyhow::Result<Option<CompactionEffects>> {
        let compacted_count = ctx.state().messages.len();
        if compacted_count == 0 && ctx.state().compacted_prefix.is_empty() {
            return Ok(None);
        }

        let mut todo_persisted = false;
        if ctx
            .tool_specs()
            .iter()
            .any(|spec| spec.name.0 == TODO_TOOL_NAME)
        {
            ctx.append(Message::User(UserMessage::text(TODO_NUDGE)));
            let reply = ctx.infer().await?;
            let (kept, task_calls) = keep_task_calls(reply);
            if !task_calls.is_empty() {
                ctx.append_unlogged(Message::Assistant(kept));
                ctx.execute_tool_calls(task_calls).await?;
                todo_persisted = true;
            }
        }

        ctx.append(Message::User(UserMessage::text(compaction_instructions(
            ctx.trigger().custom_instructions(),
        ))));
        let summary = assistant_text(&ctx.infer().await?);
        if summary.trim().is_empty() {
            anyhow::bail!("failed to compact session: the model returned no checkpoint summary");
        }

        let mut checkpoint = format!("{CHECKPOINT_PREFIX}{summary}");
        if todo_persisted {
            checkpoint.push_str(TODO_REMINDER);
        }
        Ok(Some(CompactionEffects {
            messages: vec![Message::User(UserMessage::text(checkpoint))],
            compacted_context: CompactedContext::default(),
            result: CompactionResult {
                compacted_count,
                summary: format!(
                    "Checkpointed {compacted_count} messages into a context summary{}.",
                    if todo_persisted {
                        " after persisting todos"
                    } else {
                        ""
                    }
                ),
            },
            usage: Default::default(),
        }))
    }
}

/// The nudge reply reduced to what may enter the transcript: its `task`
/// calls, plus any thinking blocks (providers require those to accompany
/// the calls they reasoned about). Text and other tool calls are dropped,
/// so no unanswered call is ever sent back to the provider.
fn keep_task_calls(mut reply: AssistantMessage) -> (AssistantMessage, Vec<ToolCall>) {
    let mut task_calls = Vec::new();
    reply.parts.retain(|part| match part {
        AssistantPart::ToolCall(call) if call.name.0 == TODO_TOOL_NAME => {
            task_calls.push(call.clone());
            true
        }
        AssistantPart::ToolCall(_) | AssistantPart::Text { .. } => false,
        AssistantPart::Thinking(_) => true,
    });
    (reply, task_calls)
}

fn assistant_text(message: &AssistantMessage) -> String {
    message
        .parts
        .iter()
        .filter_map(|part| match part {
            AssistantPart::Text { text } => Some(text.as_str()),
            AssistantPart::Thinking(_) | AssistantPart::ToolCall(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use halter_protocol::{MessageId, ThinkingBlock, ToolCallId};
    use serde_json::json;

    use super::*;

    fn call(name: &str) -> ToolCall {
        ToolCall {
            id: ToolCallId::from(format!("call_{name}")),
            name: name.into(),
            arguments: json!({"action": "list"}),
        }
    }

    fn reply(parts: Vec<AssistantPart>) -> AssistantMessage {
        AssistantMessage {
            id: MessageId::new(),
            created_at: Utc::now(),
            parts,
            stop_reason: None,
            usage: None,
            replay_meta: Default::default(),
        }
    }

    #[test]
    fn keep_task_calls_drops_text_and_foreign_calls_but_keeps_thinking() {
        let (kept, task_calls) = keep_task_calls(reply(vec![
            AssistantPart::Thinking(ThinkingBlock {
                text: "plan".to_owned(),
                signature: None,
            }),
            AssistantPart::Text {
                text: "noting todos".to_owned(),
            },
            AssistantPart::ToolCall(call("task")),
            AssistantPart::ToolCall(call("write")),
            AssistantPart::ToolCall(call("task")),
        ]));

        assert_eq!(task_calls.len(), 2);
        assert!(task_calls.iter().all(|call| call.name.0 == "task"));
        assert_eq!(kept.parts.len(), 3);
        assert!(matches!(kept.parts[0], AssistantPart::Thinking(_)));
        assert!(kept.parts[1..].iter().all(|part| matches!(
            part,
            AssistantPart::ToolCall(call) if call.name.0 == "task"
        )));
    }

    #[test]
    fn keep_task_calls_reports_nothing_for_a_text_only_reply() {
        let (kept, task_calls) = keep_task_calls(reply(vec![AssistantPart::Text {
            text: "nothing to persist".to_owned(),
        }]));

        assert!(task_calls.is_empty());
        assert!(kept.parts.is_empty());
    }

    #[test]
    fn assistant_text_joins_text_parts_only() {
        let text = assistant_text(&reply(vec![
            AssistantPart::Text {
                text: "one".to_owned(),
            },
            AssistantPart::ToolCall(call("task")),
            AssistantPart::Text {
                text: "two".to_owned(),
            },
        ]));
        assert_eq!(text, "one\ntwo");
        assert_eq!(assistant_text(&reply(Vec::new())), "");
    }
}
