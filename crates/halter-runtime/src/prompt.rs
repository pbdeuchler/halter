// pattern: Functional Core

use async_trait::async_trait;
use halter_protocol::{
    AssembledPrompt, CacheScope, ContentHash, ContextPlan, Message, PromptSegment, PromptSegmentId,
    Volatility,
};
use sha2::{Digest, Sha256};

const DEFAULT_SYSTEM_PROMPT_MARKDOWN: &str = include_str!("../prompts/default-system.md");

#[async_trait]
pub trait PromptAssembler: Send + Sync {
    async fn assemble(&self, plan: &ContextPlan) -> anyhow::Result<AssembledPrompt>;
}

#[derive(Debug, Default)]
pub struct DefaultPromptAssembler;

#[must_use]
pub(crate) fn default_system_prompt_text() -> &'static str {
    DEFAULT_SYSTEM_PROMPT_MARKDOWN.trim()
}

#[must_use]
pub(crate) fn default_system_prompt_segment() -> PromptSegment {
    let text = default_system_prompt_text().to_owned();
    PromptSegment {
        id: PromptSegmentId::new(),
        text: text.clone(),
        volatility: Volatility::Static,
        cache_scope: CacheScope::PrefixCacheable,
        content_hash: hash_prompt_text(&text),
    }
}

#[async_trait]
impl PromptAssembler for DefaultPromptAssembler {
    async fn assemble(&self, plan: &ContextPlan) -> anyhow::Result<AssembledPrompt> {
        let mut hasher = Sha256::new();
        let rendered_prefix = plan
            .prompt_segments
            .iter()
            .map(|segment| {
                if matches!(segment.cache_scope, CacheScope::PrefixCacheable) {
                    hasher.update(segment.content_hash.as_bytes());
                    hasher.update(segment.text.as_bytes());
                }
                segment.text.clone()
            })
            .collect::<Vec<_>>()
            .join("\n\n");
        hasher.update(plan.cache_boundary_hash.as_bytes());
        let rendered_transcript = plan
            .transcript_window
            .messages
            .iter()
            .map(render_message)
            .collect::<Vec<_>>()
            .join("\n");
        let rendered = if rendered_transcript.is_empty() {
            rendered_prefix.clone()
        } else if rendered_prefix.is_empty() {
            rendered_transcript.clone()
        } else {
            format!("{rendered_prefix}\n\n{rendered_transcript}")
        };

        Ok(AssembledPrompt {
            segments: plan.prompt_segments.clone(),
            transcript: plan.transcript_window.messages.clone(),
            ordered_segments: plan.prompt_segments.clone(),
            prefix_cache_key: format!("{:x}", hasher.finalize()),
            rendered_prefix,
            rendered_transcript,
            rendered,
        })
    }
}

fn render_message(message: &Message) -> String {
    match message {
        Message::System(message) => format!("system: {}", message.text),
        Message::User(message) => format!("user: {}", message.plain_text()),
        Message::Assistant(message) => {
            let body = message
                .parts
                .iter()
                .map(|part| match part {
                    halter_protocol::AssistantPart::Text { text } => text.clone(),
                    halter_protocol::AssistantPart::Thinking(block) => block.text.clone(),
                    halter_protocol::AssistantPart::ToolCall(call) => {
                        format!("tool_call {} {}", call.name, call.arguments)
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            format!("assistant: {body}")
        }
        Message::Tool(message) => match &message.content {
            halter_protocol::ToolResult::Empty => "tool: <empty>".to_owned(),
            halter_protocol::ToolResult::Text { text } => format!("tool: {text}"),
            halter_protocol::ToolResult::Json { value } => format!("tool: {value}"),
        },
    }
}

fn hash_prompt_text(text: &str) -> ContentHash {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use halter_protocol::{
        CacheScope, ContentHash, ContextPlan, Message, ObservedState, PromptSegment,
        PromptSegmentId, SummarySlice, TranscriptWindow, UserMessage, Volatility,
    };

    use super::*;

    #[test]
    fn default_system_prompt_segment_loads_embedded_markdown() {
        let segment = default_system_prompt_segment();

        assert_eq!(segment.text, default_system_prompt_text());
        assert!(!segment.text.is_empty());
        assert_eq!(segment.volatility, Volatility::Static);
        assert_eq!(segment.cache_scope, CacheScope::PrefixCacheable);
    }

    #[tokio::test]
    async fn prefix_cache_key_ignores_transcript_changes() {
        let assembler = DefaultPromptAssembler;
        let prompt_segments = vec![PromptSegment {
            id: PromptSegmentId::new(),
            text: "system instructions".to_owned(),
            volatility: Volatility::Static,
            cache_scope: CacheScope::PrefixCacheable,
            content_hash: ContentHash::from("segment-1"),
        }];
        let base_plan = ContextPlan {
            prompt_segments: prompt_segments.clone(),
            transcript_window: TranscriptWindow {
                messages: vec![Message::User(UserMessage::text("hello"))],
                elided_message_count: 0,
            },
            file_views: Vec::new(),
            carried_summaries: vec![SummarySlice {
                id: "summary".to_owned(),
                text: "kept summary".to_owned(),
            }],
            elided_tool_results: Vec::new(),
            memory_items: Vec::new(),
            tool_specs: Vec::new(),
            observed_state: ObservedState {
                cwd: ".".into(),
                git_branch: None,
                git_dirty: None,
                now_utc: Utc::now(),
                env_facts: Default::default(),
            },
            projected_input_tokens: 10,
            cache_boundary_hash: "boundary".to_owned(),
            messages: vec![Message::User(UserMessage::text("hello"))],
            estimated_tokens: 10,
        };

        let assembled_a = assembler.assemble(&base_plan).await.expect("assemble");

        let mut changed_plan = base_plan.clone();
        changed_plan.transcript_window = TranscriptWindow {
            messages: vec![Message::User(UserMessage::text("different"))],
            elided_message_count: 0,
        };
        changed_plan.messages = changed_plan.transcript_window.messages.clone();
        let assembled_b = assembler
            .assemble(&changed_plan)
            .await
            .expect("assemble changed");

        assert_eq!(assembled_a.prefix_cache_key, assembled_b.prefix_cache_key);
        assert_ne!(
            assembled_a.rendered_transcript,
            assembled_b.rendered_transcript
        );
    }
}
