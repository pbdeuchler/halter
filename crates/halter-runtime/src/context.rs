// pattern: Functional Core

use async_trait::async_trait;
use halter_protocol::{
    AssistantPart, ContextPlan, FileViewSlice, Message, ObservedState, ResourceSnapshot,
    SessionBlueprint, SessionState, ToolCallId, TranscriptWindow,
};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, Copy)]
pub struct ContextSettings {
    pub max_context_messages: usize,
}

impl Default for ContextSettings {
    fn default() -> Self {
        Self {
            max_context_messages: 24,
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
    pub fn new(max_context_messages: usize) -> Self {
        Self {
            settings: ContextSettings {
                max_context_messages,
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

        let messages = slice_recent_messages(&state.messages, self.settings.max_context_messages);
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
        let estimated_tokens = estimate_tokens(&prompt_segments, &messages);

        Ok(ContextPlan {
            transcript_window: TranscriptWindow {
                messages: messages.clone(),
                elided_message_count: state.messages.len().saturating_sub(messages.len()) as u64,
            },
            file_views,
            carried_summaries: state.summaries.clone(),
            elided_tool_results: Vec::new(),
            memory_items: Vec::new(),
            prompt_segments,
            tool_specs: tool_specs.to_vec(),
            observed_state: observed.clone(),
            projected_input_tokens: estimated_tokens,
            cache_boundary_hash: cache_boundary_hash(),
            messages,
            estimated_tokens,
        })
    }
}

fn slice_recent_messages(messages: &[Message], max_messages: usize) -> Vec<Message> {
    if max_messages == 0 || messages.is_empty() {
        return Vec::new();
    }
    if messages.len() <= max_messages {
        messages.to_vec()
    } else {
        let start = adjusted_slice_start(messages, messages.len() - max_messages);
        messages[start..].to_vec()
    }
}

fn adjusted_slice_start(messages: &[Message], start: usize) -> usize {
    let mut adjusted_start = start;

    for message in &messages[start..] {
        let Message::Tool(tool) = message else {
            continue;
        };

        if window_contains_tool_call(messages, adjusted_start, &tool.call_id) {
            continue;
        }

        if let Some(required_start) =
            find_matching_tool_call_start(messages, adjusted_start, &tool.call_id)
        {
            adjusted_start = adjusted_start.min(required_start);
        }
    }

    adjusted_start
}

fn window_contains_tool_call(messages: &[Message], start: usize, call_id: &ToolCallId) -> bool {
    messages[start..]
        .iter()
        .any(|message| message_contains_tool_call(message, call_id))
}

fn find_matching_tool_call_start(
    messages: &[Message],
    start: usize,
    call_id: &ToolCallId,
) -> Option<usize> {
    messages[..start]
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, message)| message_contains_tool_call(message, call_id).then_some(index))
}

fn message_contains_tool_call(message: &Message, call_id: &ToolCallId) -> bool {
    match message {
        Message::Assistant(assistant) => assistant
            .parts
            .iter()
            .any(|part| matches!(part, AssistantPart::ToolCall(call) if call.id == *call_id)),
        Message::System(_) | Message::User(_) | Message::Tool(_) => false,
    }
}

fn estimate_tokens(
    prompt_segments: &[halter_protocol::PromptSegment],
    messages: &[Message],
) -> u64 {
    // This remains a cheap heuristic until provider-specific tokenizers land.
    let prompt_cost = prompt_segments
        .iter()
        .map(|segment| segment.text.split_whitespace().count() as u64)
        .sum::<u64>();
    let message_cost = messages
        .iter()
        .map(|message| match message {
            Message::System(message) => message.text.split_whitespace().count() as u64,
            Message::User(message) => message.plain_text().split_whitespace().count() as u64,
            Message::Assistant(message) => message
                .parts
                .iter()
                .map(|part| match part {
                    halter_protocol::AssistantPart::Text { text } => {
                        text.split_whitespace().count() as u64
                    }
                    halter_protocol::AssistantPart::Thinking(block) => {
                        block.text.split_whitespace().count() as u64
                    }
                    halter_protocol::AssistantPart::ToolCall(_) => 8,
                })
                .sum(),
            Message::Tool(message) => match &message.content {
                halter_protocol::ToolResult::Empty => 0,
                halter_protocol::ToolResult::Text { text } => {
                    text.split_whitespace().count() as u64
                }
                halter_protocol::ToolResult::Json { value } => {
                    value.to_string().split_whitespace().count() as u64
                }
            },
        })
        .sum::<u64>();
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
        AssistantMessage, MessageId, ToolCall, ToolResult, ToolResultMessage, UserMessage,
    };
    use serde_json::json;

    use super::*;

    #[test]
    fn slice_recent_messages_preserves_matching_tool_calls() {
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

        let window = slice_recent_messages(&messages, 3);

        assert_eq!(window.len(), 4);
        assert!(matches!(window.first(), Some(Message::Assistant(_))));
        assert!(matches!(window.get(1), Some(Message::Tool(tool)) if tool.call_id == tool_call_id));
    }

    #[test]
    fn slice_recent_messages_preserves_multi_result_assistant_blocks() {
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

        let window = slice_recent_messages(&messages, 3);

        assert_eq!(window.len(), 4);
        assert!(matches!(window.first(), Some(Message::Assistant(_))));
        assert!(matches!(window.get(2), Some(Message::Tool(tool)) if tool.call_id == second));
    }
}
