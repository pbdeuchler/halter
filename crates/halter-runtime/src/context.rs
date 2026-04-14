// pattern: Functional Core

use async_trait::async_trait;
use halter_protocol::{
    AssistantPart, CompactionResult, ContextPlan, FileViewSlice, Message, MessageId, MessageSignal,
    ObservedState, ResourceSnapshot, SessionBlueprint, SessionState, SummarySlice, ToolCallId,
    ToolResult, TranscriptWindow,
};
use sha2::{Digest, Sha256};
use tracing::info;

#[derive(Debug, Clone, Copy)]
pub struct ContextSettings {
    pub max_context_messages: usize,
    pub token_budget: u64,
}

impl Default for ContextSettings {
    fn default() -> Self {
        Self {
            max_context_messages: 24,
            token_budget: 80_000,
        }
    }
}

#[async_trait]
pub trait ContextManager: Send + Sync {
    async fn plan(
        &self,
        blueprint: &SessionBlueprint,
        state: &SessionState,
        observed: &ObservedState,
        snapshot: &ResourceSnapshot,
        tool_specs: &[halter_protocol::ToolSpec],
    ) -> anyhow::Result<ContextPlan>;
}

#[derive(Debug, Default)]
pub struct DefaultContextManager {
    settings: ContextSettings,
}

impl DefaultContextManager {
    #[must_use]
    pub fn new(max_context_messages: usize, token_budget: u64) -> Self {
        Self {
            settings: ContextSettings {
                max_context_messages,
                token_budget,
            },
        }
    }

    #[must_use]
    pub fn settings(&self) -> ContextSettings {
        self.settings
    }
}

#[async_trait]
impl ContextManager for DefaultContextManager {
    async fn plan(
        &self,
        blueprint: &SessionBlueprint,
        state: &SessionState,
        observed: &ObservedState,
        _snapshot: &ResourceSnapshot,
        tool_specs: &[halter_protocol::ToolSpec],
    ) -> anyhow::Result<ContextPlan> {
        let mut prompt_segments = blueprint.system_prompt_seed.clone();
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

        let (messages, compaction) = plan_context(
            &self.settings,
            &prompt_segments,
            &state.messages,
            &state.summaries,
        );

        let estimated_tokens = estimate_tokens(&prompt_segments, &messages);

        if let Some(ref c) = compaction {
            info!(
                compacted_messages = c.compacted_count,
                remaining_messages = messages.len(),
                estimated_tokens,
                token_budget = self.settings.token_budget,
                "context planner compacted messages into summary"
            );
        }

        let mut carried_summaries = state.summaries.clone();
        if let Some(ref c) = compaction {
            carried_summaries.push(c.summary.clone());
        }

        // Determine previous_response_id chaining eligibility.
        // Chain when: (1) we have a response ID, (2) no compaction occurred this turn
        // (compaction changes what the model sees), and (3) the model has already seen
        // some messages via a prior turn.
        let (previous_response_id, new_messages_start) =
            if compaction.is_none()
                && state.last_response_id.is_some()
                && state.messages_seen_by_provider > 0
            {
                // new_messages_start is relative to the messages vec we're sending.
                // messages_seen_by_provider is an absolute index into state.messages.
                // After compaction, state.messages may have been trimmed, so the provider
                // window starts at 0. We need the offset within `messages`.
                let seen = state.messages_seen_by_provider;
                let total = state.messages.len();
                let window_offset = total.saturating_sub(messages.len());
                let new_start = seen.saturating_sub(window_offset).min(messages.len());
                (state.last_response_id.clone(), new_start)
            } else {
                (None, 0)
            };

        if previous_response_id.is_some() {
            info!(
                new_messages = messages.len() - new_messages_start,
                total_messages = messages.len(),
                "chaining via previous_response_id"
            );
        }

        Ok(ContextPlan {
            transcript_window: TranscriptWindow {
                messages: messages.clone(),
                elided_message_count: state.messages.len().saturating_sub(messages.len()) as u64,
            },
            file_views,
            carried_summaries,
            elided_tool_results: Vec::new(),
            memory_items: Vec::new(),
            prompt_segments,
            tool_specs: tool_specs.to_vec(),
            observed_state: observed.clone(),
            projected_input_tokens: estimated_tokens,
            cache_boundary_hash: cache_boundary_hash(),
            messages,
            estimated_tokens,
            compaction,
            previous_response_id,
            new_messages_start,
        })
    }
}

// ---------------------------------------------------------------------------
// Context planning
// ---------------------------------------------------------------------------

fn plan_context(
    settings: &ContextSettings,
    prompt_segments: &[halter_protocol::PromptSegment],
    messages: &[Message],
    summaries: &[SummarySlice],
) -> (Vec<Message>, Option<CompactionResult>) {
    if messages.is_empty() {
        return (Vec::new(), None);
    }

    let prefix_tokens =
        estimate_segment_tokens(prompt_segments) + estimate_summary_tokens(summaries);
    let message_tokens: Vec<u64> = messages.iter().map(estimate_message_tokens).collect();
    let total_message_tokens: u64 = message_tokens.iter().sum();

    let within_budget = prefix_tokens + total_message_tokens <= settings.token_budget;
    let within_message_cap = messages.len() <= settings.max_context_messages;

    if within_budget && within_message_cap {
        return (messages.to_vec(), None);
    }

    // Signal-based eviction: sort candidate messages by (signal score, position) so
    // we preferentially compact low-value messages first. Anchor messages (user) and
    // messages at the tail of the conversation are kept as long as possible.

    // Build scored indices: (index, signal, tokens).
    let mut scored: Vec<(usize, MessageSignal, u64)> = messages
        .iter()
        .enumerate()
        .map(|(i, m)| (i, score_message(m), message_tokens[i]))
        .collect();

    // Sort by signal ascending (low signal first), then by position ascending (older first).
    // This gives us eviction priority: VeryLow+oldest evicted first.
    scored.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

    // Mark messages to evict until we're within budget and message cap.
    let mut evicted = vec![false; messages.len()];
    let mut remaining_tokens = total_message_tokens;
    let mut remaining_count = messages.len();

    for &(idx, signal, tokens) in &scored {
        if remaining_tokens + prefix_tokens <= settings.token_budget
            && remaining_count <= settings.max_context_messages
        {
            break;
        }
        // Never evict Anchor messages (user messages).
        if signal == MessageSignal::Anchor {
            continue;
        }
        evicted[idx] = true;
        remaining_tokens = remaining_tokens.saturating_sub(tokens);
        remaining_count -= 1;
    }

    // Ensure tool call pair integrity: if we evict a tool result, also evict the
    // matching assistant; if we keep a tool result, ensure its assistant is kept.
    repair_tool_call_pairs(messages, &mut evicted);

    let kept: Vec<Message> = messages
        .iter()
        .enumerate()
        .filter(|(i, _)| !evicted[*i])
        .map(|(_, m)| m.clone())
        .collect();
    let compacted: Vec<&Message> = messages
        .iter()
        .enumerate()
        .filter(|(i, _)| evicted[*i])
        .map(|(_, m)| m)
        .collect();

    if compacted.is_empty() {
        return (messages.to_vec(), None);
    }

    let summary_text = render_compaction_summary(&compacted);
    let compaction = CompactionResult {
        compacted_count: compacted.len(),
        summary: SummarySlice {
            id: MessageId::new().0,
            text: summary_text,
        },
    };

    (kept, Some(compaction))
}

// ---------------------------------------------------------------------------
// Message signal scoring
// ---------------------------------------------------------------------------

/// Score a single message for signal value. Higher signal = more worth keeping.
#[must_use]
pub fn score_message(message: &Message) -> MessageSignal {
    match message {
        Message::User(_) => MessageSignal::Anchor,
        Message::System(_) => MessageSignal::High,
        Message::Assistant(msg) => {
            let has_text = msg
                .parts
                .iter()
                .any(|p| matches!(p, AssistantPart::Text { text } if !text.is_empty()));
            if has_text {
                MessageSignal::High
            } else {
                MessageSignal::Normal
            }
        }
        Message::Tool(msg) => {
            if msg.error.is_some() {
                return MessageSignal::Low;
            }
            match &msg.content {
                ToolResult::Empty => MessageSignal::VeryLow,
                _ => MessageSignal::Normal,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tool call pair integrity for signal-based eviction
// ---------------------------------------------------------------------------

/// Enforce tool call pair integrity after signal-based eviction.
///
/// Invariant: an assistant message and all its tool results are either ALL kept
/// or ALL evicted. If the eviction loop marked the assistant for removal, its
/// tool results go too. If the assistant was kept, any accidentally-evicted
/// tool results are restored.
fn repair_tool_call_pairs(messages: &[Message], evicted: &mut [bool]) {
    for i in 0..messages.len() {
        let Message::Assistant(assistant) = &messages[i] else {
            continue;
        };
        let tool_call_ids: Vec<&ToolCallId> = assistant
            .parts
            .iter()
            .filter_map(|p| match p {
                AssistantPart::ToolCall(call) => Some(&call.id),
                _ => None,
            })
            .collect();
        if tool_call_ids.is_empty() {
            continue;
        }
        // Propagate the assistant's eviction state to all its tool results.
        let assistant_evicted = evicted[i];
        for j in (i + 1)..messages.len() {
            if let Message::Tool(tool) = &messages[j]
                && tool_call_ids.contains(&&tool.call_id)
            {
                evicted[j] = assistant_evicted;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Compaction summary rendering
// ---------------------------------------------------------------------------

fn render_compaction_summary(messages: &[&Message]) -> String {
    let mut lines = Vec::new();
    // Track tool calls and file reads for grouping.
    let mut tool_call_groups: Vec<String> = Vec::new();
    let mut error_count = 0u32;

    for message in messages {
        match message {
            Message::User(msg) => {
                flush_tool_group(&mut tool_call_groups, &mut lines, error_count);
                error_count = 0;
                lines.push(format!("- User: {}", truncate(&msg.plain_text(), 200)));
            }
            Message::Assistant(msg) => {
                let text_parts: Vec<&str> = msg
                    .parts
                    .iter()
                    .filter_map(|p| match p {
                        AssistantPart::Text { text } if !text.is_empty() => Some(text.as_str()),
                        _ => None,
                    })
                    .collect();
                let tool_names: Vec<&str> = msg
                    .parts
                    .iter()
                    .filter_map(|p| match p {
                        AssistantPart::ToolCall(call) => Some(call.name.0.as_str()),
                        _ => None,
                    })
                    .collect();
                if !text_parts.is_empty() {
                    flush_tool_group(&mut tool_call_groups, &mut lines, error_count);
                    error_count = 0;
                    lines.push(format!(
                        "- Assistant: {}",
                        truncate(&text_parts.join(" "), 200)
                    ));
                }
                if !tool_names.is_empty() {
                    // Accumulate tool calls for grouping rather than emitting per-message.
                    tool_call_groups.extend(tool_names.iter().map(|n| (*n).to_owned()));
                }
            }
            Message::Tool(msg) => {
                if msg.error.is_some() {
                    error_count += 1;
                }
            }
            Message::System(msg) => {
                flush_tool_group(&mut tool_call_groups, &mut lines, error_count);
                error_count = 0;
                lines.push(format!("- System: {}", truncate(&msg.text, 100)));
            }
        }
    }

    flush_tool_group(&mut tool_call_groups, &mut lines, error_count);

    if lines.is_empty() {
        "Earlier conversation context was compacted.".to_owned()
    } else {
        format!(
            "Summary of {} earlier messages:\n{}",
            messages.len(),
            lines.join("\n")
        )
    }
}

/// Flush accumulated tool calls into a single grouped summary line.
fn flush_tool_group(group: &mut Vec<String>, lines: &mut Vec<String>, error_count: u32) {
    if group.is_empty() {
        return;
    }
    // Deduplicate and count: "read x3, shell x2"
    let mut counts = std::collections::BTreeMap::<&str, usize>::new();
    for name in group.iter() {
        *counts.entry(name.as_str()).or_default() += 1;
    }
    let summary: Vec<String> = counts
        .into_iter()
        .map(|(name, count)| {
            if count > 1 {
                format!("{name} x{count}")
            } else {
                name.to_owned()
            }
        })
        .collect();
    let mut line = format!("- Tools: {}", summary.join(", "));
    if error_count > 0 {
        line.push_str(&format!(" ({error_count} failed)"));
    }
    lines.push(line);
    group.clear();
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.len() <= max_chars {
        s.to_owned()
    } else {
        let boundary = s.char_indices().nth(max_chars).map_or(s.len(), |(i, _)| i);
        format!("{}...", &s[..boundary])
    }
}


// ---------------------------------------------------------------------------
// Token estimation
// ---------------------------------------------------------------------------

fn estimate_segment_tokens(segments: &[halter_protocol::PromptSegment]) -> u64 {
    segments.iter().map(|s| estimate_text_tokens(&s.text)).sum()
}

fn estimate_summary_tokens(summaries: &[SummarySlice]) -> u64 {
    summaries
        .iter()
        .map(|s| estimate_text_tokens(&s.text))
        .sum()
}

fn estimate_message_tokens(message: &Message) -> u64 {
    match message {
        Message::System(m) => estimate_text_tokens(&m.text),
        Message::User(m) => estimate_text_tokens(&m.plain_text()),
        Message::Assistant(m) => m
            .parts
            .iter()
            .map(|part| match part {
                AssistantPart::Text { text } => estimate_text_tokens(text),
                AssistantPart::Thinking(block) => estimate_text_tokens(&block.text),
                AssistantPart::ToolCall(_) => 8,
            })
            .sum(),
        Message::Tool(m) => match &m.content {
            ToolResult::Empty => 0,
            ToolResult::Text { text } => estimate_text_tokens(text),
            ToolResult::Json { value } => estimate_text_tokens(&value.to_string()),
        },
    }
}

/// Cheap heuristic: ~1.3 tokens per whitespace-delimited word.
fn estimate_text_tokens(text: &str) -> u64 {
    let words = text.split_whitespace().count() as u64;
    words * 13 / 10
}

fn estimate_tokens(
    prompt_segments: &[halter_protocol::PromptSegment],
    messages: &[Message],
) -> u64 {
    let prompt_cost: u64 = prompt_segments
        .iter()
        .map(|s| estimate_text_tokens(&s.text))
        .sum();
    let message_cost: u64 = messages.iter().map(estimate_message_tokens).sum();
    prompt_cost + message_cost
}

fn cache_boundary_hash() -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"transcript_boundary_v1");
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use halter_protocol::{
        AssistantMessage, MessageId, ToolCall, ToolError, ToolResult, ToolResultMessage,
        UserMessage,
    };
    use serde_json::json;

    use super::*;

    // -- Tool call pair preservation tests (existing) -------------------------

    #[test]
    fn signal_eviction_evicts_tool_call_pairs_together() {
        let tool_call_id = ToolCallId::from("call_1");
        let messages = vec![
            Message::User(UserMessage::text("earlier")),
            Message::Assistant(AssistantMessage {
                id: MessageId::new(),
                created_at: Utc::now(),
                parts: vec![AssistantPart::ToolCall(ToolCall {
                    id: tool_call_id.clone(),
                    name: "read".into(),
                    arguments: json!({"path": "README.md"}),
                })],
                stop_reason: None,
                usage: None,
                replay_meta: Default::default(),
            }),
            Message::Tool(ToolResultMessage {
                id: MessageId::new(),
                call_id: tool_call_id.clone(),
                content: ToolResult::Json {
                    value: json!({"ok": true}),
                },
                error: None,
                created_at: Utc::now(),
            }),
            Message::User(UserMessage::text("next")),
            Message::Assistant(AssistantMessage {
                id: MessageId::new(),
                created_at: Utc::now(),
                parts: vec![AssistantPart::Text {
                    text: "done".to_owned(),
                }],
                stop_reason: None,
                usage: None,
                replay_meta: Default::default(),
            }),
        ];

        // max_context_messages=3 forces eviction. Signal-based eviction removes the
        // Normal-scored tool call assistant + tool result pair together, keeping both
        // Anchor (user) messages and the High-scored text assistant.
        let settings = ContextSettings {
            max_context_messages: 3,
            token_budget: 1_000_000,
        };
        let (window, compaction) = plan_context(&settings, &[], &messages, &[]);

        assert_eq!(window.len(), 3);
        assert!(compaction.is_some());
        // Tool call pair was evicted together — no orphaned tool results.
        assert!(matches!(window[0], Message::User(_)));
        assert!(matches!(window[1], Message::User(_)));
        assert!(matches!(window[2], Message::Assistant(_)));
    }

    #[test]
    fn signal_eviction_evicts_multi_tool_result_blocks_together() {
        let first = ToolCallId::from("call_1");
        let second = ToolCallId::from("call_2");
        let messages = vec![
            Message::User(UserMessage::text("loop")),
            Message::Assistant(AssistantMessage {
                id: MessageId::new(),
                created_at: Utc::now(),
                parts: vec![
                    AssistantPart::ToolCall(ToolCall {
                        id: first.clone(),
                        name: "read".into(),
                        arguments: json!({"path": "a.txt"}),
                    }),
                    AssistantPart::ToolCall(ToolCall {
                        id: second.clone(),
                        name: "read".into(),
                        arguments: json!({"path": "b.txt"}),
                    }),
                ],
                stop_reason: None,
                usage: None,
                replay_meta: Default::default(),
            }),
            Message::Tool(ToolResultMessage {
                id: MessageId::new(),
                call_id: first,
                content: ToolResult::Text {
                    text: "a".to_owned(),
                },
                error: None,
                created_at: Utc::now(),
            }),
            Message::Tool(ToolResultMessage {
                id: MessageId::new(),
                call_id: second.clone(),
                content: ToolResult::Text {
                    text: "b".to_owned(),
                },
                error: None,
                created_at: Utc::now(),
            }),
            Message::Assistant(AssistantMessage {
                id: MessageId::new(),
                created_at: Utc::now(),
                parts: vec![AssistantPart::Text {
                    text: "complete".to_owned(),
                }],
                stop_reason: None,
                usage: None,
                replay_meta: Default::default(),
            }),
        ];

        // max_context_messages=3 forces eviction. The tool-call-only assistant (Normal)
        // and both tool results (Normal) are evicted together by repair_tool_call_pairs,
        // leaving the user message and the text assistant.
        let settings = ContextSettings {
            max_context_messages: 3,
            token_budget: 1_000_000,
        };
        let (window, compaction) = plan_context(&settings, &[], &messages, &[]);

        assert_eq!(window.len(), 2);
        assert!(compaction.is_some());
        assert!(matches!(window[0], Message::User(_)));
        assert!(matches!(window[1], Message::Assistant(_)));
    }

    // -- Token budget tests ---------------------------------------------------

    #[test]
    fn plan_context_sends_all_when_under_budget() {
        let messages = vec![
            Message::User(UserMessage::text("hello")),
            Message::Assistant(AssistantMessage {
                id: MessageId::new(),
                created_at: Utc::now(),
                parts: vec![AssistantPart::Text {
                    text: "hi".to_owned(),
                }],
                stop_reason: None,
                usage: None,
                replay_meta: Default::default(),
            }),
        ];

        let settings = ContextSettings {
            max_context_messages: 100,
            token_budget: 100_000,
        };
        let (window, compaction) = plan_context(&settings, &[], &messages, &[]);

        assert_eq!(window.len(), 2);
        assert!(compaction.is_none());
    }

    #[test]
    fn plan_context_compacts_when_over_budget() {
        // Create messages that exceed a tight budget.
        let mut messages = Vec::new();
        for i in 0..20 {
            messages.push(Message::User(UserMessage::text(format!(
                "message {} with enough words to eat some token budget for testing",
                i
            ))));
            messages.push(Message::Assistant(AssistantMessage {
                id: MessageId::new(),
                created_at: Utc::now(),
                parts: vec![AssistantPart::Text {
                    text: format!("reply {} with some filler text to consume tokens", i),
                }],
                stop_reason: None,
                usage: None,
                replay_meta: Default::default(),
            }));
        }

        let settings = ContextSettings {
            max_context_messages: 100,
            token_budget: 100, // Very tight budget forces compaction
        };
        let (window, compaction) = plan_context(&settings, &[], &messages, &[]);

        assert!(compaction.is_some());
        let c = compaction.unwrap();
        assert!(c.compacted_count > 0);
        assert!(window.len() < messages.len());
        assert!(!c.summary.text.is_empty());
    }

    #[test]
    fn plan_context_compacts_when_over_message_cap() {
        // 4 user messages (Anchor) and 26 assistant messages (evictable, High).
        // Cap=10 requires evicting at least 20 messages.
        let mut messages = Vec::new();
        for i in 0..4 {
            messages.push(Message::User(UserMessage::text(format!("msg {i}"))));
            for j in 0..6 {
                messages.push(Message::Assistant(AssistantMessage {
                    id: MessageId::new(),
                    created_at: Utc::now(),
                    parts: vec![AssistantPart::Text {
                        text: format!("reply {i}_{j}"),
                    }],
                    stop_reason: None,
                    usage: None,
                    replay_meta: Default::default(),
                }));
            }
        }

        let settings = ContextSettings {
            max_context_messages: 10,
            token_budget: 1_000_000,
        };
        let (window, compaction) = plan_context(&settings, &[], &messages, &[]);

        assert!(compaction.is_some());
        assert!(window.len() <= 10);
    }

    #[test]
    fn plan_context_preserves_all_anchors_when_only_anchors() {
        // When all messages are Anchor (user), none can be evicted even if over cap.
        let mut messages = Vec::new();
        for i in 0..15 {
            messages.push(Message::User(UserMessage::text(format!("msg {i}"))));
        }

        let settings = ContextSettings {
            max_context_messages: 10,
            token_budget: 1_000_000,
        };
        let (window, compaction) = plan_context(&settings, &[], &messages, &[]);

        // No compaction possible — all messages are anchors.
        assert!(compaction.is_none());
        assert_eq!(window.len(), 15);
    }

    // -- Message signal scoring tests -----------------------------------------

    #[test]
    fn score_user_message_is_anchor() {
        let msg = Message::User(UserMessage::text("hello"));
        assert_eq!(score_message(&msg), MessageSignal::Anchor);
    }

    #[test]
    fn score_failed_tool_result_is_low() {
        let msg = Message::Tool(ToolResultMessage {
            id: MessageId::new(),
            call_id: ToolCallId::from("call_1"),
            content: ToolResult::Text {
                text: "error output".to_owned(),
            },
            error: Some(ToolError::new("failed")),
            created_at: Utc::now(),
        });
        assert_eq!(score_message(&msg), MessageSignal::Low);
    }

    #[test]
    fn score_empty_tool_result_is_very_low() {
        let msg = Message::Tool(ToolResultMessage {
            id: MessageId::new(),
            call_id: ToolCallId::from("call_1"),
            content: ToolResult::Empty,
            error: None,
            created_at: Utc::now(),
        });
        assert_eq!(score_message(&msg), MessageSignal::VeryLow);
    }

    #[test]
    fn score_assistant_with_text_is_high() {
        let msg = Message::Assistant(AssistantMessage {
            id: MessageId::new(),
            created_at: Utc::now(),
            parts: vec![AssistantPart::Text {
                text: "reasoning".to_owned(),
            }],
            stop_reason: None,
            usage: None,
            replay_meta: Default::default(),
        });
        assert_eq!(score_message(&msg), MessageSignal::High);
    }

    #[test]
    fn score_tool_only_assistant_is_normal() {
        let msg = Message::Assistant(AssistantMessage {
            id: MessageId::new(),
            created_at: Utc::now(),
            parts: vec![AssistantPart::ToolCall(ToolCall {
                id: ToolCallId::from("call_1"),
                name: "read".into(),
                arguments: json!({}),
            })],
            stop_reason: None,
            usage: None,
            replay_meta: Default::default(),
        });
        assert_eq!(score_message(&msg), MessageSignal::Normal);
    }

    // -- Summary rendering tests ----------------------------------------------

    #[test]
    fn render_compaction_summary_includes_user_and_tool_calls() {
        let messages = vec![
            Message::User(UserMessage::text("read the file")),
            Message::Assistant(AssistantMessage {
                id: MessageId::new(),
                created_at: Utc::now(),
                parts: vec![AssistantPart::ToolCall(ToolCall {
                    id: ToolCallId::from("call_1"),
                    name: "read".into(),
                    arguments: json!({"path": "README.md"}),
                })],
                stop_reason: None,
                usage: None,
                replay_meta: Default::default(),
            }),
            Message::Tool(ToolResultMessage {
                id: MessageId::new(),
                call_id: ToolCallId::from("call_1"),
                content: ToolResult::Text {
                    text: "file contents".to_owned(),
                },
                error: None,
                created_at: Utc::now(),
            }),
        ];

        let refs: Vec<&Message> = messages.iter().collect();
        let summary = render_compaction_summary(&refs);
        assert!(summary.contains("User: read the file"));
        assert!(summary.contains("Tools: read"));
        assert!(summary.contains("3 earlier messages"));
    }

    #[test]
    fn render_compaction_summary_notes_tool_errors() {
        let messages = vec![
            Message::Assistant(AssistantMessage {
                id: MessageId::new(),
                created_at: Utc::now(),
                parts: vec![AssistantPart::ToolCall(ToolCall {
                    id: ToolCallId::from("call_1"),
                    name: "read".into(),
                    arguments: json!({}),
                })],
                stop_reason: None,
                usage: None,
                replay_meta: Default::default(),
            }),
            Message::Tool(ToolResultMessage {
                id: MessageId::new(),
                call_id: ToolCallId::from("call_1"),
                content: ToolResult::Empty,
                error: Some(ToolError::new("not found")),
                created_at: Utc::now(),
            }),
        ];

        let refs: Vec<&Message> = messages.iter().collect();
        let summary = render_compaction_summary(&refs);
        assert!(summary.contains("1 failed"));
    }

    // -- Token estimation tests -----------------------------------------------

    #[test]
    fn estimate_text_tokens_scales_with_words() {
        let short = estimate_text_tokens("hello world");
        let long = estimate_text_tokens("the quick brown fox jumps over the lazy dog");
        assert!(long > short);
        // ~1.3x word count
        assert_eq!(short, 2); // 2 words * 1.3 = 2.6 → 2
        assert_eq!(long, 11); // 9 words * 1.3 = 11.7 → 11
    }
}
