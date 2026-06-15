// pattern: Functional Core

use async_trait::async_trait;
use halter_protocol::{
    AssembledPrompt, CacheBreakpoints, CacheScope, ContentHash, ContextPlan, Message,
    PromptSegment, PromptSegmentId, PromptSegmentKind, Volatility,
};
use sha2::{Digest, Sha256};

use crate::compaction::stable_json;

const DEFAULT_SYSTEM_PROMPT_MARKDOWN: &str = include_str!("../prompts/default-system.md");

#[async_trait]
/// Turns a [`ContextPlan`] into provider-ready prompt material.
pub trait PromptAssembler: Send + Sync {
    /// Assemble prompt segments and transcript into an [`AssembledPrompt`].
    async fn assemble(&self, plan: &ContextPlan) -> anyhow::Result<AssembledPrompt>;
}

#[derive(Debug, Default)]
/// Default prompt assembler used by the runtime.
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
        kind: PromptSegmentKind::System,
    }
}

/// Build a single skill segment from a SkillDef. Skill segments live
/// between the system prompt and the conversation, and the assembler
/// places a cache breakpoint after the last one so subsequent turns
/// re-hit the cache while skills remain unchanged.
#[must_use]
pub fn skill_prompt_segment(name: &str, body: &str) -> PromptSegment {
    let text = format!("# Skill: {name}\n\n{body}");
    let hash = hash_prompt_text(&text);
    PromptSegment {
        id: PromptSegmentId::new(),
        text,
        volatility: Volatility::SessionStable,
        cache_scope: CacheScope::PrefixCacheable,
        content_hash: hash,
        kind: PromptSegmentKind::Skill,
    }
}

#[async_trait]
impl PromptAssembler for DefaultPromptAssembler {
    async fn assemble(&self, plan: &ContextPlan) -> anyhow::Result<AssembledPrompt> {
        let mut hasher = Sha256::new();

        // Group segments by kind so the on-wire layout is independent of
        // insertion order and so cache breakpoints land on stable section
        // boundaries rather than wherever an `Append` happened to be added.
        let (system_segments, skill_segments, append_segments) =
            group_segments(&plan.prompt_segments);
        let system_segment_count = system_segments.len();
        let skill_segment_count = skill_segments.len();

        let mut ordered_segments: Vec<PromptSegment> =
            Vec::with_capacity(plan.prompt_segments.len());
        ordered_segments.extend(system_segments.iter().cloned());
        ordered_segments.extend(skill_segments.iter().cloned());
        ordered_segments.extend(append_segments.iter().cloned());

        let mut prefix_parts: Vec<String> = Vec::with_capacity(ordered_segments.len() + 8);
        for segment in &ordered_segments {
            if matches!(segment.cache_scope, CacheScope::PrefixCacheable) {
                hasher.update(segment.content_hash.as_bytes());
                hasher.update(segment.text.as_bytes());
            }
            prefix_parts.push(segment.text.clone());
        }

        // Accumulated summaries (append-only, changes only on compaction).
        // Including these in the prefix keeps the cache key stable across
        // turns while giving the model visibility into compacted earlier
        // conversation.
        for summary in &plan.carried_summaries {
            hasher.update(summary.id.as_bytes());
            hasher.update(summary.text.as_bytes());
            prefix_parts.push(summary.text.clone());
        }

        // Raw compacted prefix items returned by the provider compaction
        // path. These sit after the last breakpoint, immediately before
        // the active transcript window.
        for item in &plan.compacted_prefix {
            let serialized = stable_json(item);
            hasher.update(serialized.as_bytes());
            prefix_parts.push(serialized);
        }

        hasher.update(plan.cache_boundary_hash.as_bytes());
        let rendered_prefix = prefix_parts.join("\n\n");

        // Active conversation tail (mutable, uncached).
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

        let cache_breakpoints = build_cache_breakpoints(
            system_segment_count,
            skill_segment_count,
            &plan.transcript_window.messages,
            !plan.tool_specs.is_empty(),
        );

        Ok(AssembledPrompt {
            segments: plan.prompt_segments.clone(),
            transcript: plan.transcript_window.messages.clone(),
            ordered_segments,
            prefix_cache_key: format!("{:x}", hasher.finalize()),
            rendered_prefix,
            rendered_transcript,
            rendered,
            cache_breakpoints,
            system_segment_count,
            skill_segment_count,
        })
    }
}

fn group_segments(
    segments: &[PromptSegment],
) -> (Vec<PromptSegment>, Vec<PromptSegment>, Vec<PromptSegment>) {
    let mut system = Vec::new();
    let mut skills = Vec::new();
    let mut append = Vec::new();
    for segment in segments {
        match segment.kind {
            PromptSegmentKind::System => system.push(segment.clone()),
            PromptSegmentKind::Skill => skills.push(segment.clone()),
            PromptSegmentKind::Append => append.push(segment.clone()),
        }
    }
    (system, skills, append)
}

fn build_cache_breakpoints(
    system_segment_count: usize,
    skill_segment_count: usize,
    transcript_messages: &[Message],
    has_tools: bool,
) -> CacheBreakpoints {
    CacheBreakpoints {
        after_system: system_segment_count > 0,
        after_tools: has_tools,
        after_skills: skill_segment_count > 0,
        after_user_prompt: transcript_messages
            .iter()
            .any(|message| matches!(message, Message::User(_))),
    }
}

fn render_message(message: &Message) -> String {
    // Length-prefixed framing keeps rendered output unambiguous even when
    // payloads contain literal role labels like "assistant:". Hashes or
    // prompt-cache keys derived from rendered text cannot be spoofed by an
    // attacker-controlled body that embeds a delimiter.
    let (role, body) = match message {
        Message::System(message) => ("system", message.text.clone()),
        Message::Meta(message) => ("meta", message.text.clone()),
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
        PromptSegmentId, PromptSegmentKind, SummarySlice, TranscriptWindow, UserMessage,
        Volatility,
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
            kind: PromptSegmentKind::System,
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
            kind: PromptSegmentKind::System,
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

    #[tokio::test]
    async fn assembler_groups_segments_and_records_breakpoints() {
        let assembler = DefaultPromptAssembler;
        let segments = vec![
            PromptSegment {
                id: PromptSegmentId::new(),
                text: "system".to_owned(),
                volatility: Volatility::Static,
                cache_scope: CacheScope::PrefixCacheable,
                content_hash: ContentHash::from("sys"),
                kind: PromptSegmentKind::System,
            },
            // Append injected before the skill: must end up after the skill
            // so the skills cache breakpoint sits between them.
            PromptSegment {
                id: PromptSegmentId::new(),
                text: "appended runtime hint".to_owned(),
                volatility: Volatility::TurnDynamic,
                cache_scope: CacheScope::Dynamic,
                content_hash: ContentHash::from("app"),
                kind: PromptSegmentKind::Append,
            },
            PromptSegment {
                id: PromptSegmentId::new(),
                text: "# Skill: pairs\n\nplay nicely".to_owned(),
                volatility: Volatility::SessionStable,
                cache_scope: CacheScope::PrefixCacheable,
                content_hash: ContentHash::from("skill"),
                kind: PromptSegmentKind::Skill,
            },
        ];
        let plan = ContextPlan {
            prompt_segments: segments,
            transcript_window: TranscriptWindow {
                messages: vec![Message::User(UserMessage::text("hi"))],
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
            projected_input_tokens: 0,
            cache_boundary_hash: "boundary".to_owned(),
            messages: vec![Message::User(UserMessage::text("hi"))],
            estimated_tokens: 0,
            compaction: None,
            previous_response_id: None,
            new_messages_start: 0,
        };

        let assembled = assembler.assemble(&plan).await.expect("assemble");
        let kinds: Vec<PromptSegmentKind> = assembled
            .ordered_segments
            .iter()
            .map(|seg| seg.kind)
            .collect();
        assert_eq!(
            kinds,
            vec![
                PromptSegmentKind::System,
                PromptSegmentKind::Skill,
                PromptSegmentKind::Append,
            ],
            "system → skill → append, regardless of insertion order"
        );
        assert_eq!(assembled.system_segment_count, 1);
        assert_eq!(assembled.skill_segment_count, 1);
        assert!(assembled.cache_breakpoints.after_system);
        assert!(assembled.cache_breakpoints.after_skills);
        assert!(assembled.cache_breakpoints.after_user_prompt);
        assert!(
            !assembled.cache_breakpoints.after_tools,
            "no tools provided → no tools breakpoint"
        );
    }
}
