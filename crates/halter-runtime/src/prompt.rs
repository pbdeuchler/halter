// pattern: Functional Core

use async_trait::async_trait;
use halter_protocol::{
    AssembledPrompt, CacheScope, ContentHash, ContextPlan, Message, PromptSegment, PromptSegmentId,
    Volatility,
};
use sha2::{Digest, Sha256};

use crate::compaction::stable_json;

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

        // Layer 1: system prompt segments (static prefix).
        let mut prefix_parts: Vec<String> = plan
            .prompt_segments
            .iter()
            .map(|segment| {
                if matches!(segment.cache_scope, CacheScope::PrefixCacheable) {
                    hasher.update(segment.content_hash.as_bytes());
                    hasher.update(segment.text.as_bytes());
                }
                segment.text.clone()
            })
            .collect();

        // Layer 2: accumulated summaries (append-only, changes only on compaction).
        // Including these in the prefix keeps the cache key stable across turns
        // while giving the model visibility into compacted earlier conversation.
        for summary in &plan.carried_summaries {
            hasher.update(summary.id.as_bytes());
            hasher.update(summary.text.as_bytes());
            prefix_parts.push(summary.text.clone());
        }

        // Layer 3: raw compacted prefix items returned by /v1/responses/compact.
        for item in &plan.compacted_prefix {
            let serialized = stable_json(item);
            hasher.update(serialized.as_bytes());
            prefix_parts.push(serialized);
        }

        hasher.update(plan.cache_boundary_hash.as_bytes());
        let rendered_prefix = prefix_parts.join("\n\n");

        // Layer 4: active conversation tail (mutable, uncached).
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
    // Length-prefixed framing keeps rendered output unambiguous even when
    // payloads contain literal role labels like "assistant:". Hashes or
    // prompt-cache keys derived from rendered text cannot be spoofed by an
    // attacker-controlled body that embeds a delimiter.
    let (role, body) = match message {
        Message::System(message) => ("system", message.text.clone()),
        Message::User(message) => ("user", message.plain_text()),
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
            ("assistant", body)
        }
        Message::Tool(message) => (
            "tool",
            match &message.content {
                halter_protocol::ToolResult::Empty => "<empty>".to_owned(),
                halter_protocol::ToolResult::Text { text } => text.clone(),
                halter_protocol::ToolResult::Json { value } => value.to_string(),
            },
        ),
    };
    format!("[{role}:{} bytes]\n{body}", body.len())
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
            compacted_prefix: vec![],
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
            compaction: None,
            previous_response_id: None,
            new_messages_start: 0,
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

    #[tokio::test]
    async fn prefix_cache_key_changes_when_summaries_change() {
        let assembler = DefaultPromptAssembler;
        let segments = vec![PromptSegment {
            id: PromptSegmentId::new(),
            text: "system".to_owned(),
            volatility: Volatility::Static,
            cache_scope: CacheScope::PrefixCacheable,
            content_hash: ContentHash::from("seg-1"),
        }];
        let base_plan = ContextPlan {
            prompt_segments: segments.clone(),
            transcript_window: TranscriptWindow {
                messages: vec![Message::User(UserMessage::text("hello"))],
                elided_message_count: 0,
            },
            compacted_prefix: vec![],
            file_views: Vec::new(),
            carried_summaries: vec![],
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
            compaction: None,
            previous_response_id: None,
            new_messages_start: 0,
        };

        let without_summary = assembler.assemble(&base_plan).await.expect("assemble");

        let mut with_summary = base_plan.clone();
        with_summary.carried_summaries = vec![SummarySlice {
            id: "s1".to_owned(),
            text: "earlier context was compacted".to_owned(),
        }];
        let with_summary = assembler.assemble(&with_summary).await.expect("assemble");

        // Adding a summary changes the prefix cache key.
        assert_ne!(
            without_summary.prefix_cache_key,
            with_summary.prefix_cache_key
        );
        // The summary text appears in the rendered prefix.
        assert!(
            with_summary
                .rendered_prefix
                .contains("earlier context was compacted")
        );
    }
}
