//! The default compaction strategy: ask the session's own model for a context
//! checkpoint and start the next window from that summary alone.
// pattern: Imperative Shell

use async_trait::async_trait;
use halter_protocol::{
    AssistantMessage, AssistantPart, CompactedContext, CompactionResult, Message,
    ProviderCapabilities, SessionState, StopReason, ToolCall, UserMessage, UserPart,
};

use crate::compaction_strategy::{
    CompactionContext, CompactionStrategy, CompactionTrigger, compaction_instructions,
};
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

/// Labels the user's first request where a checkpoint carries it verbatim.
pub const FIRST_REQUEST_LABEL: &str = "The user's first request, verbatim:\n\n";

/// Labels the user's latest request, when it is not the first.
pub const LATEST_REQUEST_LABEL: &str = "The user's latest request, verbatim:\n\n";

/// Bound, in bytes, on each request a checkpoint carries, marker included.
pub const CARRIED_REQUEST_MAX_BYTES: usize = 4_000;

/// Ends a carried request clipped to [`CARRIED_REQUEST_MAX_BYTES`].
const CLIPPED_MARKER: &str = "\n[clipped]";

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
///    preserved-thinking checks. The same message carries the user's first
///    request and, when it differs, the latest one, verbatim up to
///    [`CARRIED_REQUEST_MAX_BYTES`] each: what the work is for should not
///    depend on a summary of a summary. The first request is copied from
///    the previous checkpoint, so it stays byte-identical however many
///    times the session compacts.
pub struct ModelSummary;

#[async_trait]
impl CompactionStrategy for ModelSummary {
    /// Anything at all to checkpoint: a transcript or a legacy prefix.
    fn needs_compaction(
        &self,
        state: &SessionState,
        _capabilities: &ProviderCapabilities,
        _trigger: CompactionTrigger<'_>,
    ) -> bool {
        !state.messages.is_empty() || !state.compacted_prefix.is_empty()
    }

    async fn compact(
        &self,
        mut ctx: CompactionContext<'_>,
    ) -> anyhow::Result<Option<CompactionEffects>> {
        if !self.needs_compaction(ctx.state(), &ctx.provider().capabilities(), ctx.trigger()) {
            return Ok(None);
        }
        let compacted_count = ctx.state().messages.len();
        // Before the pass appends its own requests to the window.
        let carried = carried_requests(&ctx.state().messages);

        let mut todo_persisted = false;
        if ctx
            .tool_specs()
            .iter()
            .any(|spec| spec.name.0 == TODO_TOOL_NAME)
        {
            ctx.append(Message::User(UserMessage::harness(TODO_NUDGE)));
            let reply = ctx.infer().await?;
            let (kept, task_calls) = keep_task_calls(reply);
            if !task_calls.is_empty() {
                ctx.append_unlogged(Message::Assistant(kept));
                ctx.execute_tool_calls(task_calls).await?;
                todo_persisted = true;
            }
        }

        ctx.append(Message::User(UserMessage::harness(
            compaction_instructions(ctx.trigger().custom_instructions()),
        )));
        let summary = checkpoint_summary(&ctx.infer().await?)?;

        let mut checkpoint = format!("{CHECKPOINT_PREFIX}{summary}");
        if todo_persisted {
            checkpoint.push_str(TODO_REMINDER);
        }
        let checkpoint = UserMessage::harness(checkpoint);
        Ok(Some(CompactionEffects {
            messages: vec![Message::User(UserMessage {
                parts: checkpoint.parts.into_iter().chain(carried).collect(),
                ..checkpoint
            })],
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

/// The user's first and, when it differs, latest request in `messages`, as
/// labelled parts for the next checkpoint. Requests are the window's user
/// messages; an earlier checkpoint contributes the ones it carried, so its
/// first request stays first, byte for byte.
fn carried_requests(messages: &[Message]) -> Vec<UserPart> {
    let labels = [FIRST_REQUEST_LABEL, LATEST_REQUEST_LABEL];
    let requests: Vec<String> = messages
        .iter()
        .filter_map(|message| match message {
            Message::User(user) => Some(user),
            _ => None,
        })
        .flat_map(|user| {
            let text = user.plain_text();
            if text.starts_with(CHECKPOINT_PREFIX) {
                user.parts
                    .iter()
                    .filter_map(|part| match part {
                        UserPart::Text { text } => labels
                            .iter()
                            .find_map(|label| text.strip_prefix(label))
                            .map(str::to_owned),
                        UserPart::Image { .. } | UserPart::Document { .. } => None,
                    })
                    .collect()
            } else if user.is_harness() || text.trim().is_empty() {
                // A harness-authored message is not a request.
                Vec::new()
            } else {
                vec![clip(text)]
            }
        })
        .collect();
    let first = requests.first();
    let latest = requests.last().filter(|latest| Some(*latest) != first);
    labels
        .iter()
        .zip([first, latest])
        .filter_map(|(label, request)| {
            request.map(|request| UserPart::Text {
                text: format!("{label}{request}"),
            })
        })
        .collect()
}

/// `text` within [`CARRIED_REQUEST_MAX_BYTES`], cut on a char boundary and
/// marked. Idempotent, so a carried request survives being carried again.
fn clip(text: String) -> String {
    if text.len() <= CARRIED_REQUEST_MAX_BYTES {
        return text;
    }
    let budget = CARRIED_REQUEST_MAX_BYTES - CLIPPED_MARKER.len();
    let end = (0..=budget)
        .rev()
        .find(|&index| text.is_char_boundary(index))
        .unwrap_or_default();
    format!("{}{CLIPPED_MARKER}", &text[..end])
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

/// The checkpoint reply's text, when it is a summary the model finished. A
/// reply cut by `max_tokens`, refused (`Error`), or paused (`Interrupted`)
/// still carries text, and checkpointing it would replace the window with a
/// fragment. Its usage is already counted either way.
fn checkpoint_summary(reply: &AssistantMessage) -> anyhow::Result<String> {
    if let Some(reason @ (StopReason::MaxTokens | StopReason::Error | StopReason::Interrupted)) =
        reply.stop_reason
    {
        anyhow::bail!("failed to compact session: the checkpoint reply stopped early ({reason:?})");
    }
    let summary = assistant_text(reply);
    if summary.trim().is_empty() {
        anyhow::bail!("failed to compact session: the model returned no checkpoint summary");
    }
    Ok(summary)
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

    /// Only a finished, non-empty reply is a checkpoint. Kills dropping the
    /// stop-reason guard (the three early stops pass), widening it to
    /// `EndTurn` or `None` (the finished rows fail), and dropping the empty
    /// check (the blank row passes).
    #[test]
    fn checkpoint_summary_accepts_only_a_finished_reply() {
        // (case, stop reason, text, Ok(summary) or Err(message fragment))
        type Case = (
            &'static str,
            Option<StopReason>,
            &'static str,
            Result<&'static str, &'static str>,
        );
        let cases: [Case; 7] = [
            ("end turn", Some(StopReason::EndTurn), "gist", Ok("gist")),
            ("tool use", Some(StopReason::ToolUse), "gist", Ok("gist")),
            ("unreported", None, "gist", Ok("gist")),
            (
                "max tokens",
                Some(StopReason::MaxTokens),
                "gi",
                Err("stopped early (MaxTokens)"),
            ),
            (
                "refusal",
                Some(StopReason::Error),
                "I can't",
                Err("stopped early (Error)"),
            ),
            (
                "paused",
                Some(StopReason::Interrupted),
                "gi",
                Err("stopped early (Interrupted)"),
            ),
            (
                "blank",
                Some(StopReason::EndTurn),
                "  ",
                Err("returned no checkpoint summary"),
            ),
        ];
        for (case, stop_reason, text, expected) in cases {
            let reply = AssistantMessage {
                stop_reason,
                ..reply(vec![AssistantPart::Text {
                    text: text.to_owned(),
                }])
            };
            match (checkpoint_summary(&reply), expected) {
                (Ok(summary), Ok(want)) => assert_eq!(summary, want, "{case}"),
                (Err(error), Err(fragment)) => {
                    assert!(error.to_string().contains(fragment), "{case}: {error}");
                }
                (got, want) => panic!("{case}: got {got:?}, want {want:?}"),
            }
        }
    }

    fn user(text: &str) -> Message {
        Message::User(UserMessage::text(text))
    }

    /// A checkpoint as a previous pass built it: summary first, then the
    /// carried requests.
    fn checkpoint(carried: &[(&str, &str)]) -> Message {
        let parts = std::iter::once(format!("{CHECKPOINT_PREFIX}older gist"))
            .chain(carried.iter().map(|(label, text)| format!("{label}{text}")))
            .map(|text| UserPart::Text { text })
            .collect();
        Message::User(UserMessage {
            parts,
            ..UserMessage::text("")
        })
    }

    /// Kills: dropping the checkpoint branch (a re-carried first request is
    /// lost or carried with its label doubled), dropping the distinctness
    /// filter ("repeat" carries the request twice), dropping the blank
    /// filter ("image only" carries an empty request), and `clip` cutting
    /// off a char boundary or over the bound ("multibyte").
    #[test]
    fn carried_requests_keeps_the_first_and_latest_verbatim() {
        let first = (FIRST_REQUEST_LABEL, "port the parser");
        let image_only = Message::User(UserMessage {
            parts: vec![UserPart::Image {
                media_type: "image/png".into(),
                data: Vec::new().into(),
            }],
            ..UserMessage::text("")
        });
        let long = "é".repeat(2_500);
        let clipped = clip(long.clone());
        // (case, window, carried (label, text) pairs)
        type Case = (&'static str, Vec<Message>, Vec<(&'static str, String)>);
        let cases: Vec<Case> = vec![
            ("no request", vec![image_only.clone()], vec![]),
            (
                "one request",
                vec![user("port the parser")],
                vec![(first.0, first.1.to_owned())],
            ),
            (
                "two requests",
                vec![user("port the parser"), user("now the lexer")],
                vec![
                    (first.0, first.1.to_owned()),
                    (LATEST_REQUEST_LABEL, "now the lexer".to_owned()),
                ],
            ),
            (
                "repeat",
                vec![user("port the parser"), user("port the parser")],
                vec![(first.0, first.1.to_owned())],
            ),
            // Kills dropping the `is_harness` filter.
            (
                "a harness message is not a request",
                vec![
                    user("port the parser"),
                    Message::User(UserMessage::harness("Continue.")),
                ],
                vec![(first.0, first.1.to_owned())],
            ),
            (
                "first carried from a checkpoint",
                vec![checkpoint(&[first]), image_only, user("now the lexer")],
                vec![
                    (first.0, first.1.to_owned()),
                    (LATEST_REQUEST_LABEL, "now the lexer".to_owned()),
                ],
            ),
            (
                "latest carried when nothing new arrived",
                vec![checkpoint(&[
                    first,
                    (LATEST_REQUEST_LABEL, "now the lexer"),
                ])],
                vec![
                    (first.0, first.1.to_owned()),
                    (LATEST_REQUEST_LABEL, "now the lexer".to_owned()),
                ],
            ),
            (
                "a new latest replaces the carried one",
                vec![
                    checkpoint(&[first, (LATEST_REQUEST_LABEL, "now the lexer")]),
                    user("ship it"),
                ],
                vec![
                    (first.0, first.1.to_owned()),
                    (LATEST_REQUEST_LABEL, "ship it".to_owned()),
                ],
            ),
            (
                "multibyte",
                vec![user(&long)],
                vec![(first.0, clipped.clone())],
            ),
            (
                "a clipped request survives being carried again",
                vec![checkpoint(&[(FIRST_REQUEST_LABEL, &clipped)])],
                vec![(first.0, clipped.clone())],
            ),
        ];
        for (case, window, expected) in cases {
            let expected: Vec<UserPart> = expected
                .into_iter()
                .map(|(label, text)| UserPart::Text {
                    text: format!("{label}{text}"),
                })
                .collect();
            assert_eq!(carried_requests(&window), expected, "{case}");
        }
        assert!(
            clipped.len() <= CARRIED_REQUEST_MAX_BYTES,
            "{}",
            clipped.len()
        );
        assert!(clipped.ends_with(CLIPPED_MARKER));
        assert!(long.starts_with(clipped.trim_end_matches(CLIPPED_MARKER)));
        assert_eq!(clip("short".to_owned()), "short");
    }

    /// Kills dropping either side of the `||`: an empty session must skip
    /// before any hook fires, and a legacy prefix alone is still work.
    #[test]
    fn needs_compaction_unless_the_session_is_empty() {
        // (case, has a message, has a prefix, needs)
        let cases = [
            ("empty", false, false, false),
            ("transcript", true, false, true),
            ("legacy prefix only", false, true, true),
        ];
        for (case, has_message, has_prefix, needs) in cases {
            let state = SessionState {
                messages: if has_message {
                    vec![Message::User(UserMessage::text("hi"))]
                } else {
                    Vec::new()
                },
                compacted_prefix: if has_prefix {
                    vec![json!({"type": "compaction"})]
                } else {
                    Vec::new()
                },
                ..SessionState::default()
            };
            assert_eq!(
                ModelSummary.needs_compaction(
                    &state,
                    &ProviderCapabilities::default(),
                    CompactionTrigger::Automatic
                ),
                needs,
                "{case}"
            );
        }
    }
}
