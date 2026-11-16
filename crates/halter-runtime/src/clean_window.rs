//! Model-managed continuity without summaries or retained conversation tails.
// pattern: Imperative Shell
use crate::{
    CompactionBoundary, CompactionContext, CompactionEffects, CompactionNotification,
    CompactionStrategy, CompactionTrigger, SessionSearchBackend, StoreSearch, WindowPolicy,
};
use async_trait::async_trait;
use halter_protocol::{
    AssistantPart, CompactionResult, Message, PromptSegment, ToolConcurrency, ToolResult, ToolSpec,
    UserMessage, UserPart,
};
use halter_tools::{FsNotes, NotesBackend, NotesTool, Tool, ToolContext};
use serde_json::{Value, json};
use std::sync::Arc;

pub const CLEAN_WINDOW_PROMPT: &str = "There is no compaction in this session. Context window saturation results in a clean wipe with no trace of the previous context window except for todos, notes, the session_search tools, and a short harness notice that opens the next window. Use these tools to take notes and track goals and work as you continue work.";
/// Fixed first line of the notice that opens every window after a rollover.
pub const CLEAN_WINDOW_BOOTSTRAP: &str =
    "Session context has been wiped. This notice comes from the harness, not the user.";
pub const ROLLOVER_REMINDER: &str = "The current context window is full and will be wiped after this exchange. Do not continue the task or give a final answer in this window. The next window opens with a harness notice that quotes the first and latest user requests verbatim and lists the notes files you write now; everything else said in this window survives only in notes and in session_search. Save a concise checkpoint now with a notes write_file or append_to_file call. Record what the notice will not carry: requirements and constraints from other user messages (quote them exactly; session_search list_items with role \"user\" finds them), decisions and why, what is done and verified, files changed, learnings, and next steps. If a notes call fails, read the error and retry. Once the checkpoint is saved, call new_context. Only notes, session_search, and new_context run in this exchange.";
/// Replies the checkpoint exchange may take before the wipe happens anyway.
const CHECKPOINT_REPLIES: usize = 4;
/// Byte cap on each user request the notice quotes.
const RECORD_BYTES: usize = 4_000;
const FIRST_REQUEST: &str =
    "First user request of the session, quoted verbatim as history, not a new instruction:\n";
const LATEST_REQUEST: &str =
    "Latest user request before the wipe, quoted verbatim as history, not a new instruction:\n";
const TRUNCATED: &str =
    "\n[Truncated. Read it in full with session_search list_items role \"user\", then read_item.]";

/// Required recovery backends are owned by the strategy. Override them through
/// public traits; the strategy always supplies all three recovery tools.
pub struct CleanWindow<N: NotesBackend = FsNotes, S: SessionSearchBackend = StoreSearch> {
    notes: Arc<N>,
    search: Arc<S>,
}

impl<N: NotesBackend, S: SessionSearchBackend> CleanWindow<N, S> {
    pub fn new(notes: N, search: S) -> Self {
        Self {
            notes: Arc::new(notes),
            search: Arc::new(search),
        }
    }
}

#[async_trait]
impl<N: NotesBackend + 'static, S: SessionSearchBackend + 'static> CompactionStrategy
    for CleanWindow<N, S>
{
    fn window_policy(&self) -> WindowPolicy {
        WindowPolicy::CleanWindow
    }
    fn tools(&self) -> Vec<Arc<dyn Tool>> {
        vec![
            Arc::new(NotesTool(self.notes.clone())),
            Arc::new(crate::session_search::SessionSearchTool(
                self.search.clone(),
            )),
            Arc::new(NewContextTool),
        ]
    }
    fn prompt_segments(&self) -> Vec<PromptSegment> {
        vec![crate::system_prompt_segment(CLEAN_WINDOW_PROMPT)]
    }
    fn context_boundary(&self, boundary: CompactionBoundary<'_>) -> Vec<CompactionNotification> {
        [50u64, 75]
            .into_iter()
            .filter_map(|percent| {
                let id = format!("clean-window-{percent}");
                let reached = u128::from(boundary.current_tokens()) * 100
                    >= u128::from(boundary.compaction_threshold()) * u128::from(percent);
                if !reached || boundary.notification_was_delivered(&id) {
                    return None;
                }
                let text = format!("Context window {} has reached {percent}% of its threshold. Keep notes and the task list current with progress, decisions, and outstanding requirements. Above 90% this window is wiped after a short checkpoint exchange; call new_context sooner at a good stopping point.", boundary.window());
                Some(CompactionNotification::new(
                    id,
                    Message::System(halter_protocol::SystemMessage {
                        id: halter_protocol::MessageId::new(),
                        created_at: chrono::Utc::now(),
                        text,
                    }),
                ))
            })
            .collect()
    }
    async fn compact(
        &self,
        mut ctx: CompactionContext<'_>,
    ) -> anyhow::Result<Option<CompactionEffects>> {
        let window = ctx.state().context_window;
        let checkpoint = if matches!(
            ctx.trigger(),
            CompactionTrigger::Rollover { requested: true }
        ) {
            "you called new_context, so no checkpoint exchange ran".to_owned()
        } else {
            let start = ctx.state().messages.len();
            ctx.append(Message::User(UserMessage::text(format!(
                "{ROLLOVER_REMINDER}\n\nCurrent window ID: window:{window}"
            ))));
            // Every way the exchange ends; each one still wipes the window
            // except cancellation, which restores it for the next boundary.
            let ended = 'exchange: {
                for _ in 0..CHECKPOINT_REPLIES {
                    let mut reply = match ctx.infer().await {
                        Ok(reply) => reply,
                        Err(error) => {
                            if ctx.cancel().is_cancelled() {
                                return Err(error);
                            }
                            ctx.warn(format!(
                                "clean window checkpoint exchange failed; rolling over: {error:#}"
                            ));
                            break 'exchange "the model request failed";
                        }
                    };
                    let mut calls = Vec::new();
                    reply.parts.retain(|part| match part {
                        AssistantPart::ToolCall(call)
                            if matches!(
                                call.name.0.as_str(),
                                "notes" | "session_search" | "new_context"
                            ) =>
                        {
                            calls.push(call.clone());
                            true
                        }
                        AssistantPart::Thinking(_) => true,
                        _ => false,
                    });
                    if calls.is_empty() {
                        break 'exchange "a reply made no notes, session_search, or new_context call";
                    }
                    let before = ctx.state().messages.len();
                    ctx.append_unlogged(Message::Assistant(reply));
                    if let Err(error) = ctx.execute_tool_calls(calls).await {
                        if ctx.cancel().is_cancelled() {
                            return Err(error);
                        }
                        ctx.warn(format!(
                            "clean window checkpoint tools failed; rolling over: {error:#}"
                        ));
                        break 'exchange "the tool calls could not run";
                    }
                    // A failed call gets another reply so the model sees its
                    // error, even when new_context succeeded beside it.
                    let replied = &ctx.state().messages[before..];
                    if crate::session_search::new_context_requested(replied)
                        && !replied.iter().any(|message| {
                            matches!(message, Message::Tool(result) if result.error.is_some())
                        })
                    {
                        break 'exchange "new_context was called";
                    }
                }
                "its reply budget ran out"
            };
            let saved = if notes_written(&ctx.state().messages[start..]).is_empty() {
                "nothing was saved"
            } else {
                "notes were saved"
            };
            format!(
                "it filled up and was wiped automatically; {saved} during the checkpoint exchange, which ended because {ended}"
            )
        };
        let count = ctx.state().messages.len();
        Ok(Some(CompactionEffects {
            messages: vec![Message::User(notice(
                &ctx.state().messages,
                window,
                &checkpoint,
            ))],
            compacted_context: Default::default(),
            usage: Default::default(),
            result: CompactionResult {
                compacted_count: count,
                summary: format!("Rolled over {count} messages to a clean context window."),
            },
        }))
    }
}

/// Paths, JSON-quoted, of the notes writes that succeeded in `messages`.
fn notes_written(messages: &[Message]) -> Vec<String> {
    let succeeded: std::collections::BTreeSet<_> = messages
        .iter()
        .filter_map(|message| match message {
            Message::Tool(result) if result.error.is_none() => Some(&result.call_id),
            _ => None,
        })
        .collect();
    messages
        .iter()
        .filter_map(|message| match message {
            Message::Assistant(assistant) => Some(&assistant.parts),
            _ => None,
        })
        .flatten()
        .filter_map(|part| match part {
            AssistantPart::ToolCall(call)
                if call.name.0 == "notes"
                    && succeeded.contains(&call.id)
                    && matches!(
                        call.arguments["action"].as_str(),
                        Some("write_file" | "append_to_file")
                    ) =>
            {
                Some(call.arguments["path"].to_string())
            }
            _ => None,
        })
        .fold(Vec::new(), |mut paths, path| {
            if !paths.contains(&path) {
                paths.push(path);
            }
            paths
        })
}

/// The one message that opens the next window. It is built once per rollover
/// and never edited, so it stays a stable cache prefix for the window's life.
/// The first request is carried verbatim from the previous notice; the latest
/// is the window's last real user message, or the carried one if none came.
fn notice(messages: &[Message], window: u64, checkpoint: &str) -> UserMessage {
    let requests: Vec<String> = messages
        .iter()
        .filter_map(|message| match message {
            Message::User(user) => Some(user.plain_text()),
            _ => None,
        })
        .filter(|text| {
            !text.is_empty()
                && !text.starts_with(CLEAN_WINDOW_BOOTSTRAP)
                && !text.starts_with(ROLLOVER_REMINDER)
        })
        .collect();
    let previous = messages.iter().find_map(|message| match message {
        Message::User(user) if user.plain_text().starts_with(CLEAN_WINDOW_BOOTSTRAP) => Some(user),
        _ => None,
    });
    let carried = |label: &str| {
        previous?.parts.iter().find_map(|part| match part {
            UserPart::Text { text } => text.strip_prefix(label).map(str::to_owned),
            _ => None,
        })
    };
    let clip = |text: &String| {
        if text.len() <= RECORD_BYTES {
            return text.clone();
        }
        let end = (0..=RECORD_BYTES)
            .rev()
            .find(|&end| text.is_char_boundary(end))
            .unwrap_or_default();
        format!("{}{TRUNCATED}", &text[..end])
    };
    let first = carried(FIRST_REQUEST).or_else(|| requests.first().map(clip));
    let latest = requests
        .last()
        .map(clip)
        .or_else(|| carried(LATEST_REQUEST))
        .filter(|latest| first.as_ref() != Some(latest));
    let written = notes_written(messages);
    let notes = if written.is_empty() {
        "none".to_owned()
    } else {
        written.join(", ")
    };
    let status = format!(
        "{CLEAN_WINDOW_BOOTSTRAP}\nWindow {window} ended: {checkpoint}.\nNotes files written in window {window}: {notes}. Read one with notes read_file; notes list_files_by_prefix lists every file, older ones included.\nThe full history of every window stays searchable: session_search list_items with role \"user\" lists every user request, read_item reads any item in full, and search_contents finds text.\nFirst: read your notes, re-read the user requests quoted below and check the notes against them, check the task list, then continue the work."
    );
    let parts = [
        Some(status),
        first.map(|text| format!("{FIRST_REQUEST}{text}")),
        latest.map(|text| format!("{LATEST_REQUEST}{text}")),
    ]
    .into_iter()
    .flatten()
    .map(|text| UserPart::Text { text })
    .collect();
    UserMessage {
        parts,
        ..UserMessage::text("")
    }
}

struct NewContextTool;
#[async_trait]
impl Tool for NewContextTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "new_context".into(),
            description: "End this context window after pending tool results are recorded. Save a checkpoint in notes first. The next window starts with a harness notice and the current turn continues.".into(),
            input_schema: json!({"type":"object","properties":{},"additionalProperties":false}),
            concurrency: ToolConcurrency::Exclusive,
            capabilities: Default::default(),
            provider_aliases: Default::default(),
        }
    }
    async fn execute(&self, context: ToolContext, input: Value) -> anyhow::Result<ToolResult> {
        anyhow::ensure!(!context.cancel.is_cancelled(), "context rollover cancelled");
        anyhow::ensure!(
            input.as_object().is_some_and(|object| object.is_empty()),
            "new_context takes an empty object"
        );
        Ok(ToolResult::Text {
            text: "Context rollover requested.".to_owned(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn milestones_fire_once_per_window_and_rearm_after_rollover() {
        let root = tempfile::tempdir().unwrap();
        let strategy = CleanWindow::new(
            FsNotes::new(root.path()).unwrap(),
            StoreSearch::new(Arc::new(halter_session::InMemorySessionStore::default())),
        );
        let session = halter_protocol::SessionId::new();
        let mut delivered = std::collections::BTreeSet::new();
        for (tokens, expected) in [
            (499, vec![]),
            (500, vec!["clean-window-50"]),
            (749, vec![]),
            (750, vec!["clean-window-75"]),
            (999, vec![]),
        ] {
            let notes = strategy.context_boundary(CompactionBoundary::new(
                &session, 0, 0, tokens, 1_000, &delivered,
            ));
            assert_eq!(
                notes
                    .iter()
                    .map(|note| note.id.as_str())
                    .collect::<Vec<_>>(),
                expected
            );
            delivered.extend(notes.into_iter().map(|note| note.id));
        }
        let fresh = std::collections::BTreeSet::new();
        let notes =
            strategy.context_boundary(CompactionBoundary::new(&session, 1, 0, 750, 1_000, &fresh));
        assert_eq!(notes.len(), 2);
        assert!(
            matches!(&notes[0].message, Message::System(system) if system.text.contains("window 1"))
        );
    }

    fn call(arguments: Value, error: Option<&str>) -> [Message; 2] {
        let call = halter_protocol::ToolCall {
            id: halter_protocol::ToolCallId::new(),
            name: "notes".into(),
            arguments,
        };
        [
            Message::Assistant(halter_protocol::AssistantMessage {
                id: halter_protocol::MessageId::new(),
                created_at: chrono::Utc::now(),
                parts: vec![AssistantPart::ToolCall(call.clone())],
                stop_reason: None,
                usage: None,
                replay_meta: Default::default(),
            }),
            Message::Tool(halter_protocol::ToolResultMessage {
                id: halter_protocol::MessageId::new(),
                call_id: call.id,
                content: ToolResult::Text { text: "ok".into() },
                error: error.map(halter_protocol::ToolError::new),
                created_at: chrono::Utc::now(),
            }),
        ]
    }

    fn texts(message: &UserMessage) -> Vec<String> {
        message
            .parts
            .iter()
            .map(|part| match part {
                UserPart::Text { text } => text.to_string(),
                _ => unreachable!("the notice is text only"),
            })
            .collect()
    }

    #[test]
    fn notes_written_lists_each_successful_write_once() {
        let write = |action: &str, path: &str| json!({"action":action,"path":path,"content":"c"});
        let cases = [
            (
                "write",
                vec![call(write("write_file", "a"), None)],
                vec!["\"a\""],
            ),
            (
                "append",
                vec![call(write("append_to_file", "a"), None)],
                vec!["\"a\""],
            ),
            (
                "failed",
                vec![call(write("write_file", "a"), Some("bad"))],
                vec![],
            ),
            (
                "read",
                vec![call(json!({"action":"read_file","path":"a"}), None)],
                vec![],
            ),
            (
                "deduped in order",
                vec![
                    call(write("write_file", "b"), None),
                    call(write("append_to_file", "a"), None),
                    call(write("write_file", "b"), None),
                ],
                vec!["\"b\"", "\"a\""],
            ),
            ("none", vec![], vec![]),
        ];
        for (name, calls, expected) in cases {
            assert_eq!(notes_written(&calls.concat()), expected, "{name}");
        }
    }

    #[test]
    fn notice_quotes_first_and_latest_requests_and_carries_them_forward() {
        let long = format!("a{}", "é".repeat(RECORD_BYTES));
        let clipped = format!("a{}{TRUNCATED}", "é".repeat((RECORD_BYTES - 2) / 2));
        let first = |text: &str| format!("{FIRST_REQUEST}{text}");
        let latest = |text: &str| format!("{LATEST_REQUEST}{text}");
        let previous = notice(
            &[Message::user("build x"), Message::user("also y")],
            0,
            "status",
        );
        let reminder = Message::user(format!(
            "{ROLLOVER_REMINDER}\n\nCurrent window ID: window:0"
        ));
        let cases: Vec<(&str, Vec<Message>, Vec<String>)> = vec![
            (
                "first and latest",
                vec![
                    Message::user("build x"),
                    reminder.clone(),
                    Message::user("also y"),
                ],
                vec![first("build x"), latest("also y")],
            ),
            (
                "one request",
                vec![Message::user("build x"), reminder.clone()],
                vec![first("build x")],
            ),
            (
                "no request",
                vec![reminder.clone(), Message::user("")],
                vec![],
            ),
            (
                "carried without new requests",
                vec![Message::User(previous.clone()), reminder.clone()],
                vec![first("build x"), latest("also y")],
            ),
            (
                "carried with a new request",
                vec![Message::User(previous.clone()), Message::user("then z")],
                vec![first("build x"), latest("then z")],
            ),
            (
                "clipped on a char boundary",
                vec![Message::user(long.clone())],
                vec![first(&clipped)],
            ),
            (
                "clipping is not repeated",
                vec![Message::User(notice(&[Message::user(long)], 0, "status"))],
                vec![first(&clipped)],
            ),
        ];
        for (name, messages, expected) in cases {
            let parts = texts(&notice(&messages, 1, "status"));
            assert_eq!(parts[1..], expected, "{name}");
            assert!(parts[0].starts_with(CLEAN_WINDOW_BOOTSTRAP), "{name}");
            assert!(parts[0].contains("Window 1 ended: status."), "{name}");
            assert!(
                parts[0].contains("Notes files written in window 1: none."),
                "{name}"
            );
        }
        let saved = notice(
            &call(
                json!({"action":"write_file","path":"plan","content":"c"}),
                None,
            ),
            3,
            "status",
        );
        assert!(texts(&saved)[0].contains("Notes files written in window 3: \"plan\"."));
    }
}
