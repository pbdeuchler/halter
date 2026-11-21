//! Model-managed continuity without summaries or retained conversation tails.
// pattern: Imperative Shell
use crate::{
    CompactionBoundary, CompactionContext, CompactionEffects, CompactionNotification,
    CompactionStrategy, CompactionTrigger, SessionSearchBackend, StoreSearch, WindowPolicy,
};
use async_trait::async_trait;
use halter_protocol::{
    AssistantPart, CompactionResult, Message, MessageId, PromptSegment, SubagentState,
    ToolConcurrency, ToolError, ToolResult, ToolResultMessage, ToolSpec, UserMessage, UserPart,
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
/// The only tools that run during the checkpoint exchange.
const CHECKPOINT_TOOLS: [&str; 3] = ["notes", "session_search", "new_context"];
/// Byte caps the notice tries on each user request it quotes, largest first,
/// until it fits its budget. At 0 it quotes none.
const QUOTE_CAPS: [usize; 4] = [4_000, 1_000, 250, 0];
/// The notice lists one notes path or running item per this many bytes of
/// quote cap: 10 at the full cap, none once quotes shrink to 250 bytes.
const LIST_BYTES: usize = 400;
/// Byte cap on each running item the notice lists.
const ENTRY_BYTES: usize = 200;
const FIRST_REQUEST: &str =
    "First user request of the session, quoted verbatim as history, not a new instruction:\n";
/// Opens the first quote when window 0 had no request to carry forward.
const EARLIEST_REQUEST: &str = "Earliest user request this notice can quote, from window ";
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
            ctx.append(Message::User(UserMessage::harness(format!(
                "{ROLLOVER_REMINDER}\n\nCurrent window ID: window:{window}"
            ))));
            // Every way the exchange ends; each one still wipes the window
            // except cancellation, which restores it for the next boundary.
            let ended = 'exchange: {
                for _ in 0..CHECKPOINT_REPLIES {
                    let reply = match ctx.infer().await {
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
                    // Every call is answered, so the wire stays valid and the
                    // model learns why a refused call did nothing. The offered
                    // tools stay as they are: narrowing them would make this
                    // request, the largest of the window, miss the cache.
                    let (allowed, refused): (Vec<_>, Vec<_>) = reply
                        .parts
                        .iter()
                        .filter_map(|part| match part {
                            AssistantPart::ToolCall(call) => Some(call.clone()),
                            _ => None,
                        })
                        .partition(|call| CHECKPOINT_TOOLS.contains(&call.name.0.as_str()));
                    if allowed.is_empty() && refused.is_empty() {
                        break 'exchange "a reply made no tool call";
                    }
                    let before = ctx.state().messages.len();
                    ctx.append_unlogged(Message::Assistant(reply));
                    for call in refused {
                        ctx.append(Message::Tool(ToolResultMessage {
                            id: MessageId::new(),
                            call_id: call.id,
                            content: ToolResult::Empty,
                            error: Some(ToolError::new(format!(
                                "{} did not run: only notes, session_search, and new_context run during the checkpoint exchange.",
                                call.name
                            ))),
                            created_at: chrono::Utc::now(),
                        }));
                    }
                    if let Err(error) = ctx.execute_tool_calls(allowed).await {
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
        let state = ctx.state();
        let count = state.messages.len();
        let services = ctx.services();
        let session = &ctx.blueprint().session_id;
        let specs = ctx.tool_specs();
        let tools: Vec<String> = specs.iter().map(|spec| spec.name.0.clone()).collect();
        let running: Vec<String> = services
            .tool_sessions
            .running_background_jobs(session)
            .await
            .into_iter()
            .map(|(id, command)| format!("background job {id}: $ {command}"))
            .chain(
                state
                    .subagents
                    .values()
                    .filter(|record| record.status.state == SubagentState::Running)
                    .map(|record| {
                        format!(
                            "subagent {}: {}",
                            record.status.agent_id, record.status.task
                        )
                    }),
            )
            .chain(
                services
                    .tool_sessions
                    .pty_running(session)
                    .then(|| "the pty session (the pty tool manages it)".to_owned()),
            )
            .map(|line| line.replace('\n', " "))
            .map(|line| match line.len() > ENTRY_BYTES {
                true => format!("{}...", cut(&line, ENTRY_BYTES)),
                false => line,
            })
            .collect();
        // The rollover check fails the turn when the new window starts above
        // 90% of the threshold. Fit the notice into half of what the static
        // prefix leaves, shrinking the quotes and lists before failing.
        let appended = &state.appended_prompt_segments;
        let segments: Vec<_> = ctx
            .prompt_segments()
            .into_iter()
            .filter(|segment| !appended.iter().any(|added| added.id == segment.id))
            .collect();
        let budget = (services.context.compaction_threshold.saturating_mul(9) / 10)
            .saturating_sub(halter_protocol::estimate_request_tokens(&segments, &specs))
            / 2;
        let build = |cap| {
            Message::User(notice(
                &state.messages,
                window,
                &checkpoint,
                &tools,
                &running,
                cap,
            ))
        };
        let opening = QUOTE_CAPS
            .into_iter()
            .map(build)
            .find(|notice| halter_protocol::estimate_message_tokens(notice) <= budget)
            .unwrap_or_else(|| build(0));
        Ok(Some(CompactionEffects {
            messages: vec![opening],
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

/// The longest prefix of `text` within `cap` bytes that ends on a char boundary.
fn cut(text: &str, cap: usize) -> &str {
    let end = (0..=cap.min(text.len()))
        .rev()
        .find(|&end| text.is_char_boundary(end))
        .unwrap_or_default();
    &text[..end]
}

/// A quote of `text` within `cap` bytes, with a pointer to the full text
/// when cut, or `None` at a cap of 0. A pointer already present is kept and
/// not counted, so a quote clipped once is carried verbatim.
fn clip(text: &str, cap: usize) -> Option<String> {
    let body = text.strip_suffix(TRUNCATED).unwrap_or(text);
    match cap {
        0 => None,
        _ if body.len() <= cap => Some(text.to_owned()),
        _ => Some(format!("{}{TRUNCATED}", cut(body, cap))),
    }
}

/// The first `shown` of `items` and how many more there are.
fn listed(items: &[String], shown: usize) -> (&[String], usize) {
    let shown = shown.min(items.len());
    (&items[..shown], items.len() - shown)
}

/// The one message that opens the next window. It is built once per rollover
/// and never edited, so it stays a stable cache prefix for the window's life.
///
/// It names only the tools in `tools`. The first request is carried verbatim
/// from the previous notice; the latest is the window's last user request, or
/// the carried one if none came. Messages the harness wrote are never quoted.
/// `cap` bounds each quote and, through [`LIST_BYTES`], the notes paths and
/// the `running` items listed; the wipe statement and the recovery
/// instructions never shrink.
fn notice(
    messages: &[Message],
    window: u64,
    checkpoint: &str,
    tools: &[String],
    running: &[String],
    cap: usize,
) -> UserMessage {
    let has = |tool: &str| tools.iter().any(|name| name == tool);
    let requests: Vec<String> = messages
        .iter()
        .filter_map(|message| match message {
            Message::User(user) if !user.is_harness() => Some(user.plain_text()),
            _ => None,
        })
        // Notices and reminders logged before the harness mark existed.
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
    // A quote the previous notice carried, as its label line and its text.
    let carried = |labels: &[&str]| {
        previous?.parts.iter().find_map(|part| match part {
            UserPart::Text { text } if labels.iter().any(|label| text.starts_with(label)) => {
                text.split_once('\n')
            }
            _ => None,
        })
    };
    // Window 0 holds the session's first request. A later window whose
    // previous notice quoted none holds only the earliest one left to quote.
    let first = carried(&[FIRST_REQUEST, EARLIEST_REQUEST])
        .map(|(label, text)| (label.to_owned(), text.to_owned()))
        .or_else(|| {
            let label = match window {
                0 => FIRST_REQUEST.trim_end().to_owned(),
                _ => format!("{EARLIEST_REQUEST}{window} (session_search list_items with role \"user\" finds any earlier one), quoted verbatim as history, not a new instruction:"),
            };
            Some((label, requests.first()?.clone()))
        });
    let latest = requests
        .last()
        .cloned()
        .or_else(|| carried(&[LATEST_REQUEST]).map(|(_, text)| text.to_owned()))
        .filter(|latest| first.as_ref().is_none_or(|(_, first)| first != latest));
    let quotes: Vec<String> = [
        first,
        latest.map(|text| (LATEST_REQUEST.trim_end().to_owned(), text)),
    ]
    .into_iter()
    .flatten()
    .filter_map(|(label, text)| Some(format!("{label}\n{}", clip(&text, cap)?)))
    .collect();

    let shown = cap / LIST_BYTES;
    let notes = match listed(&notes_written(messages), shown) {
        ([], 0) => "none".to_owned(),
        ([], more) => format!("{more} files"),
        (paths, 0) => paths.join(", "),
        (paths, more) => format!("{} and {more} more", paths.join(", ")),
    };
    let still_running = (!running.is_empty()).then(|| {
        let (items, more) = listed(running, shown);
        let more = (more > 0).then(|| format!("{more} more not listed here; session_search finds their ids in the calls that started them"));
        let lines: String = items
            .iter()
            .cloned()
            .chain(more)
            .map(|line| format!("\n- {line}"))
            .collect();
        format!("Still running across the wipe; manage them by these ids:{lines}")
    });
    let reread = match (quotes.is_empty(), has("session_search")) {
        (false, _) => Some("re-read the user requests quoted below"),
        (true, true) => Some("re-read the user requests with session_search"),
        (true, false) => None,
    }
    .map(|step| match has("notes") {
        true => format!("{step} and check the notes against them"),
        false => step.to_owned(),
    });
    let steps: Vec<String> = [
        has("notes").then(|| "read your notes".to_owned()),
        reread,
        has("task").then(|| "check the task list".to_owned()),
    ]
    .into_iter()
    .flatten()
    .collect();
    let then = if steps.is_empty() { "" } else { "then " };
    let status = [
        Some(format!("{CLEAN_WINDOW_BOOTSTRAP}\nWindow {window} ended: {checkpoint}.")),
        has("notes").then(|| format!("Notes files written in window {window}: {notes}. Read one with notes read_file; notes list_files_by_prefix lists every file, older ones included.")),
        has("session_search").then(|| "The full history of every window stays searchable: session_search list_items with role \"user\" lists every user request, read_item reads any item in full, and search_contents finds text.".to_owned()),
        still_running,
        Some(format!(
            "First: {}{then}continue the work.",
            steps.iter().map(|step| format!("{step}, ")).collect::<String>()
        )),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join("\n");
    UserMessage {
        parts: std::iter::once(status)
            .chain(quotes)
            .map(|text| UserPart::Text { text })
            .collect(),
        ..UserMessage::harness("")
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
    /// The input is ignored: a model that adds a field it was not asked for
    /// still gets its rollover.
    async fn execute(&self, context: ToolContext, _input: Value) -> anyhow::Result<ToolResult> {
        anyhow::ensure!(!context.cancel.is_cancelled(), "context rollover cancelled");
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

    const WIPED: &str =
        "Session context has been wiped. This notice comes from the harness, not the user.";
    const NOTES_TAIL: &str = "Read one with notes read_file; notes list_files_by_prefix lists every file, older ones included.";
    const SEARCH_LINE: &str = "The full history of every window stays searchable: session_search list_items with role \"user\" lists every user request, read_item reads any item in full, and search_contents finds text.";

    fn names(tools: &[&str]) -> Vec<String> {
        tools.iter().map(|tool| (*tool).to_owned()).collect()
    }

    fn all_tools() -> Vec<String> {
        names(&["notes", "session_search", "task"])
    }

    fn quotes(messages: &[Message], window: u64) -> Vec<String> {
        texts(&notice(
            messages,
            window,
            "status",
            &all_tools(),
            &[],
            QUOTE_CAPS[0],
        ))[1..]
            .to_vec()
    }

    // Kills: dropping the `!user.is_harness()` filter, the carried lookup,
    // the EARLIEST label, the first != latest filter, or clip's suffix strip.
    #[test]
    fn notice_quotes_first_and_latest_requests_and_carries_them_forward() {
        let cap = QUOTE_CAPS[0];
        let long = format!("a{}", "é".repeat(cap));
        let clipped = format!("a{}{TRUNCATED}", "é".repeat((cap - 2) / 2));
        let first = |text: &str| format!("{FIRST_REQUEST}{text}");
        let latest = |text: &str| format!("{LATEST_REQUEST}{text}");
        let earliest = |window: u64, text: &str| {
            format!(
                "{EARLIEST_REQUEST}{window} (session_search list_items with role \"user\" finds any earlier one), quoted verbatim as history, not a new instruction:\n{text}"
            )
        };
        let previous = notice(
            &[Message::user("build x"), Message::user("also y")],
            0,
            "status",
            &all_tools(),
            &[],
            cap,
        );
        let empty = notice(&[], 0, "status", &all_tools(), &[], cap);
        let harness = |text: &str| Message::User(UserMessage::harness(text));
        // Logged before the harness mark existed: recognised by prefix.
        let legacy = Message::user(format!(
            "{ROLLOVER_REMINDER}\n\nCurrent window ID: window:0"
        ));
        let cases: Vec<(&str, Vec<Message>, u64, Vec<String>)> = vec![
            (
                "first and latest",
                vec![
                    Message::user("build x"),
                    legacy.clone(),
                    Message::user("also y"),
                ],
                0,
                vec![first("build x"), latest("also y")],
            ),
            (
                "harness continuation is never the latest",
                vec![
                    Message::user("build x"),
                    Message::user("also y"),
                    harness("Continue."),
                ],
                0,
                vec![first("build x"), latest("also y")],
            ),
            (
                "one request",
                vec![Message::user("build x"), legacy.clone()],
                0,
                vec![first("build x")],
            ),
            (
                "no request",
                vec![legacy.clone(), Message::user("")],
                0,
                vec![],
            ),
            (
                "carried without new requests",
                vec![Message::User(previous.clone()), harness("Continue.")],
                1,
                vec![first("build x"), latest("also y")],
            ),
            (
                "carried with a new request",
                vec![Message::User(previous.clone()), Message::user("then z")],
                1,
                vec![first("build x"), latest("then z")],
            ),
            (
                "window 0 had no request",
                vec![Message::User(empty.clone()), Message::user("then z")],
                1,
                vec![earliest(1, "then z")],
            ),
            (
                "earliest label is carried, not relabelled",
                vec![
                    Message::User(notice(
                        &[Message::User(empty), Message::user("then z")],
                        1,
                        "status",
                        &all_tools(),
                        &[],
                        cap,
                    )),
                    Message::user("and w"),
                ],
                2,
                vec![earliest(1, "then z"), latest("and w")],
            ),
            (
                "clipped on a char boundary",
                vec![Message::user(long.clone())],
                0,
                vec![first(&clipped)],
            ),
            (
                "clipping is not repeated",
                vec![Message::User(notice(
                    &[Message::user(long)],
                    0,
                    "status",
                    &all_tools(),
                    &[],
                    cap,
                ))],
                1,
                vec![first(&clipped)],
            ),
        ];
        for (name, messages, window, expected) in cases {
            assert_eq!(quotes(&messages, window), expected, "{name}");
        }
    }

    // Kills: an unconditional notes, session_search, or task mention; a
    // "quoted below" step with no quotes; a dangling "then" with no steps.
    #[test]
    fn notice_names_only_the_tools_that_exist() {
        let request = [Message::user("build x")];
        let notes_line = format!("Notes files written in window 1: none. {NOTES_TAIL}");
        let head = format!("{WIPED}\nWindow 1 ended: status.");
        let cases: Vec<(&str, &[Message], Vec<String>, String)> = vec![
            (
                "every tool",
                &request,
                all_tools(),
                format!(
                    "{head}\n{notes_line}\n{SEARCH_LINE}\nFirst: read your notes, re-read the user requests quoted below and check the notes against them, check the task list, then continue the work."
                ),
            ),
            (
                "no task tool",
                &request,
                names(&["notes", "session_search"]),
                format!(
                    "{head}\n{notes_line}\n{SEARCH_LINE}\nFirst: read your notes, re-read the user requests quoted below and check the notes against them, then continue the work."
                ),
            ),
            (
                "search only, nothing to quote",
                &[],
                names(&["session_search"]),
                format!(
                    "{head}\n{SEARCH_LINE}\nFirst: re-read the user requests with session_search, then continue the work."
                ),
            ),
            (
                "no recovery tools, nothing to quote",
                &[],
                vec![],
                format!("{head}\nFirst: continue the work."),
            ),
        ];
        for (name, messages, tools, expected) in cases {
            let parts = texts(&notice(messages, 1, "status", &tools, &[], QUOTE_CAPS[0]));
            assert_eq!(parts[0], expected, "{name}");
        }
    }

    // Kills: listing more than cap / LIST_BYTES entries, dropping the "more"
    // count, or dropping the wipe statement or search line as the cap shrinks.
    #[test]
    fn notice_lists_shrink_with_the_cap_but_the_recovery_lines_stay() {
        let writes: Vec<Message> = (0..12)
            .flat_map(|index| {
                call(
                    json!({"action":"write_file","path":format!("n{index:02}"),"content":"c"}),
                    None,
                )
            })
            .chain([Message::user("build x")])
            .collect();
        let running: Vec<String> = (0..11).map(|index| format!("job {index:02}")).collect();
        let paths = |count: usize| {
            (0..count)
                .map(|index| format!("\"n{index:02}\""))
                .collect::<Vec<_>>()
                .join(", ")
        };
        let jobs = |count: usize, more: usize| {
            let lines: String = (0..count)
                .map(|index| format!("\n- job {index:02}"))
                .collect();
            format!(
                "Still running across the wipe; manage them by these ids:{lines}\n- {more} more not listed here; session_search finds their ids in the calls that started them"
            )
        };
        let first = format!("{FIRST_REQUEST}build x");
        let head = format!("{WIPED}\nWindow 0 ended: status.");
        let tail = "First: read your notes, re-read the user requests quoted below and check the notes against them, check the task list, then continue the work.";
        let bare = "First: read your notes, re-read the user requests with session_search and check the notes against them, check the task list, then continue the work.";
        let cases = [
            (
                QUOTE_CAPS[0],
                format!(
                    "{head}\nNotes files written in window 0: {} and 2 more. {NOTES_TAIL}\n{SEARCH_LINE}\n{}\n{tail}",
                    paths(10),
                    jobs(10, 1)
                ),
                vec![first.clone()],
            ),
            (
                QUOTE_CAPS[1],
                format!(
                    "{head}\nNotes files written in window 0: {} and 10 more. {NOTES_TAIL}\n{SEARCH_LINE}\n{}\n{tail}",
                    paths(2),
                    jobs(2, 9)
                ),
                vec![first.clone()],
            ),
            (
                QUOTE_CAPS[2],
                format!(
                    "{head}\nNotes files written in window 0: 12 files. {NOTES_TAIL}\n{SEARCH_LINE}\n{}\n{tail}",
                    jobs(0, 11)
                ),
                vec![first],
            ),
            (
                QUOTE_CAPS[3],
                format!(
                    "{head}\nNotes files written in window 0: 12 files. {NOTES_TAIL}\n{SEARCH_LINE}\n{}\n{bare}",
                    jobs(0, 11)
                ),
                vec![],
            ),
        ];
        for (cap, status, quoted) in cases {
            let parts = texts(&notice(&writes, 0, "status", &all_tools(), &running, cap));
            assert_eq!(parts[0], status, "cap {cap}");
            assert_eq!(parts[1..], quoted, "cap {cap}");
        }
        let few = texts(&notice(
            &[],
            1,
            "status",
            &all_tools(),
            &running[..1],
            QUOTE_CAPS[0],
        ));
        assert!(
            few[0].contains(
                "\nStill running across the wipe; manage them by these ids:\n- job 00\nFirst:"
            ),
            "a short list has no overflow line"
        );
    }

    // Kills: clip appending the pointer to text that fits, clip quoting at
    // cap 0, or cut splitting a multi-byte char.
    #[test]
    fn cut_and_clip_respect_char_boundaries_and_the_cap() {
        let cut_cases = [
            ("abc", 2, "ab"),
            ("é", 1, ""),
            ("aé", 2, "a"),
            ("ab", 9, "ab"),
            ("ab", 0, ""),
        ];
        for (text, cap, expected) in cut_cases {
            assert_eq!(cut(text, cap), expected, "cut {text:?} at {cap}");
        }
        let clip_cases = [
            ("abc".to_owned(), 0, None),
            ("abc".to_owned(), 3, Some("abc".to_owned())),
            ("abcd".to_owned(), 3, Some(format!("abc{TRUNCATED}"))),
            (
                format!("abc{TRUNCATED}"),
                3,
                Some(format!("abc{TRUNCATED}")),
            ),
        ];
        for (text, cap, expected) in clip_cases {
            assert_eq!(clip(&text, cap), expected, "clip {text:?} at {cap}");
        }
    }

    #[test]
    fn notice_reports_saved_notes_by_path() {
        let saved = notice(
            &call(
                json!({"action":"write_file","path":"plan","content":"c"}),
                None,
            ),
            3,
            "status",
            &all_tools(),
            &[],
            QUOTE_CAPS[0],
        );
        assert!(texts(&saved)[0].contains(&format!(
            "Notes files written in window 3: \"plan\". {NOTES_TAIL}"
        )));
        assert!(saved.is_harness());
    }
}
