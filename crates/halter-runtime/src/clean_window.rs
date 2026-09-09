//! Model-managed continuity without summaries or retained conversation tails.
// pattern: Imperative Shell
use crate::{
    CompactionBoundary, CompactionContext, CompactionEffects, CompactionNotification,
    CompactionStrategy, CompactionTrigger, SessionSearchBackend, StoreSearch, WindowPolicy,
};
use async_trait::async_trait;
use halter_protocol::{
    AssistantPart, CompactionResult, Message, PromptSegment, ToolConcurrency, ToolResult, ToolSpec,
    UserMessage,
};
use halter_tools::{FsNotes, NotesBackend, NotesTool, Tool, ToolContext};
use serde_json::{Value, json};
use std::sync::Arc;

pub const CLEAN_WINDOW_PROMPT: &str = "There is no compaction in this session. Context window saturation results in a clean wipe with no trace of the previous context window except for todos, notes, and the session_search tools. Use these tools to take notes and track goals and work as you continue work.";
pub const CLEAN_WINDOW_BOOTSTRAP: &str = "Session context has been wiped. Use the todo (task), notes, and session_search tools to regain context and continue your tasks.";
pub const ROLLOVER_REMINDER: &str = "The current context window is exhausted. Do not continue the task or give a final answer in this window. The next window will not automatically include this conversation. Make exactly one write_file or append_to_file call to notes now to save a concise checkpoint with the goal, decisions, progress, learnings, next steps, and the window ID and item ID of every relevant user request still being solved, as well as important actions/tool calls for future reference. After the notes result returns, call new_context; do not use any tools other than notes and new_context.";

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
        [50u64, 75].into_iter().filter_map(|percent| {
            let id = format!("clean-window-{percent}");
            if u128::from(boundary.current_tokens()) * 100 >= u128::from(boundary.compaction_threshold()) * u128::from(percent) && !boundary.notification_was_delivered(&id) {
                let text = format!("Context window {} has reached {percent}% of its threshold. Save progress and outstanding requests with notes and task; use session_search for stable history IDs. Above 90% this window will be wiped. Call new_context when ready.", boundary.window());
                Some(CompactionNotification::new(id, Message::System(halter_protocol::SystemMessage { id:halter_protocol::MessageId::new(), created_at:chrono::Utc::now(), text })))
            } else { None }
        }).collect()
    }
    async fn compact(
        &self,
        mut ctx: CompactionContext<'_>,
    ) -> anyhow::Result<Option<CompactionEffects>> {
        let count = ctx.state().messages.len();
        if !matches!(
            ctx.trigger(),
            CompactionTrigger::Rollover { requested: true }
        ) {
            ctx.append(Message::User(UserMessage::text(format!(
                "{ROLLOVER_REMINDER}\n\nCurrent window ID: window:{}",
                ctx.state().context_window
            ))));
            // At most two replies: checkpoint, then new_context. A refusal, bad
            // call, provider failure, or an over-cap exchange cannot veto the wipe.
            for _ in 0..2 {
                let mut reply = match ctx.infer().await {
                    Ok(reply) => reply,
                    Err(error) => {
                        if ctx.cancel().is_cancelled() {
                            return Err(error);
                        }
                        ctx.warn(format!(
                            "clean window checkpoint exchange failed; rolling over: {error:#}"
                        ));
                        break;
                    }
                };
                let mut calls = Vec::new();
                reply.parts.retain(|part| match part {
                    AssistantPart::ToolCall(call)
                        if matches!(call.name.0.as_str(), "notes" | "new_context") =>
                    {
                        calls.push(call.clone());
                        true
                    }
                    AssistantPart::Thinking(_) => true,
                    _ => false,
                });
                if calls.is_empty() {
                    break;
                }
                ctx.append_unlogged(Message::Assistant(reply));
                if let Err(error) = ctx.execute_tool_calls(calls).await {
                    if ctx.cancel().is_cancelled() {
                        return Err(error);
                    }
                    ctx.warn(format!(
                        "clean window checkpoint tools failed; rolling over: {error:#}"
                    ));
                    break;
                }
                if crate::session_search::new_context_requested(&ctx.state().messages) {
                    break;
                }
            }
        }
        Ok(Some(CompactionEffects {
            messages: vec![Message::User(UserMessage::text(CLEAN_WINDOW_BOOTSTRAP))],
            compacted_context: Default::default(),
            usage: Default::default(),
            result: CompactionResult {
                compacted_count: count,
                summary: format!("Rolled over {count} messages to a clean context window."),
            },
        }))
    }
}

struct NewContextTool;
#[async_trait]
impl Tool for NewContextTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
        name:"new_context".into(),description:"End this context window after pending tool results are recorded. Save a checkpoint in notes first. The next window starts with a recovery reminder and the current turn continues.".into(),
        input_schema:json!({"type":"object","properties":{},"additionalProperties":false}), concurrency:ToolConcurrency::Exclusive,capabilities:Default::default(),provider_aliases:Default::default(),
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
            StoreSearch(Arc::new(halter_session::InMemorySessionStore::default())),
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
}
