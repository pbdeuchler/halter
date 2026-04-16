// pattern: Imperative Shell

use async_trait::async_trait;
use halter_protocol::{
    CompactionResult, ContextPlan, FileViewSlice, Message, ObservedState, PromptSegment,
    ProviderCompactionRequest, ResolvedModel, ResourceSnapshot, SessionBlueprint, SessionState,
    ToolSpec, TranscriptWindow,
};
use halter_providers::Provider;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tracing::info;

use crate::compaction::{
    ContextSettings, estimate_context_tokens, prepare_compaction, render_compaction_event_summary,
    should_trigger_compaction,
};

const DEFAULT_COMPACTION_PROMPT_MARKDOWN: &str = include_str!("../prompts/default-compaction.md");

#[derive(Debug, Clone)]
pub struct CompactionOutcome {
    pub messages: Vec<Message>,
    pub compacted_prefix: Vec<Value>,
    pub compaction: Option<CompactionResult>,
}

#[async_trait]
pub trait ContextManager: Send + Sync {
    async fn plan(
        &self,
        blueprint: &SessionBlueprint,
        state: &SessionState,
        observed: &ObservedState,
        snapshot: &ResourceSnapshot,
        tool_specs: &[ToolSpec],
        compaction_model: &ResolvedModel,
        compaction_provider: &(dyn Provider + Send + Sync),
    ) -> anyhow::Result<ContextPlan>;

    async fn compact_now(
        &self,
        blueprint: &SessionBlueprint,
        state: &SessionState,
        observed: &ObservedState,
        snapshot: &ResourceSnapshot,
        tool_specs: &[ToolSpec],
        compaction_model: &ResolvedModel,
        compaction_provider: &(dyn Provider + Send + Sync),
        custom_instructions: Option<&str>,
    ) -> anyhow::Result<CompactionOutcome>;
}

#[derive(Debug, Default)]
pub struct DefaultContextManager {
    settings: ContextSettings,
}

impl DefaultContextManager {
    #[must_use]
    pub fn new(
        compaction_threshold: u64,
        pre_compaction_target: u64,
        prune_signal_threshold: halter_protocol::PruneSignalThreshold,
    ) -> Self {
        Self {
            settings: ContextSettings {
                compaction_threshold,
                pre_compaction_target,
                prune_signal_threshold,
            },
        }
    }

    #[must_use]
    pub fn from_settings(settings: ContextSettings) -> Self {
        Self { settings }
    }

    #[must_use]
    pub fn settings(&self) -> ContextSettings {
        self.settings
    }

    async fn execute_compaction(
        &self,
        blueprint: &SessionBlueprint,
        state: &SessionState,
        prompt_segments: &[PromptSegment],
        tool_specs: &[ToolSpec],
        compaction_model: &ResolvedModel,
        compaction_provider: &(dyn Provider + Send + Sync),
        force: bool,
        custom_instructions: Option<&str>,
    ) -> anyhow::Result<CompactionOutcome> {
        let estimated_tokens = estimate_context_tokens(
            prompt_segments,
            &state.summaries,
            &state.compacted_prefix,
            &state.messages,
        );
        if !force && !should_trigger_compaction(estimated_tokens, &self.settings) {
            return Ok(CompactionOutcome {
                messages: state.messages.clone(),
                compacted_prefix: state.compacted_prefix.clone(),
                compaction: None,
            });
        }

        if !compaction_provider.capabilities().supports_compaction {
            anyhow::bail!(
                "failed to compact session: provider '{}' does not support compaction",
                compaction_model.provider
            );
        }

        let preparation =
            prepare_compaction(&self.settings, &state.compacted_prefix, &state.messages);
        if state.compacted_prefix.is_empty() && preparation.compact_messages.is_empty() {
            return Ok(CompactionOutcome {
                messages: state.messages.clone(),
                compacted_prefix: state.compacted_prefix.clone(),
                compaction: None,
            });
        }

        let response = compaction_provider
            .compact(
                ProviderCompactionRequest {
                    session_id: blueprint.session_id.clone(),
                    model: compaction_model.clone(),
                    compacted_prefix: state.compacted_prefix.clone(),
                    messages: preparation.compact_messages.clone(),
                    tools: tool_specs.to_vec(),
                    instructions: compaction_instructions(custom_instructions),
                },
                tokio_util::sync::CancellationToken::new(),
            )
            .await?;
        let summary = render_compaction_event_summary(
            preparation.compacted_message_count,
            response.output.len(),
            preparation.evicted_unit_count,
            preparation.reserved_response_block,
        );

        Ok(CompactionOutcome {
            messages: preparation.reserved_suffix,
            compacted_prefix: response.output,
            compaction: Some(CompactionResult {
                compacted_count: preparation.compacted_message_count,
                summary,
            }),
        })
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
        tool_specs: &[ToolSpec],
        compaction_model: &ResolvedModel,
        compaction_provider: &(dyn Provider + Send + Sync),
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

        let outcome = self
            .execute_compaction(
                blueprint,
                state,
                &prompt_segments,
                tool_specs,
                compaction_model,
                compaction_provider,
                false,
                None,
            )
            .await?;
        let estimated_tokens = estimate_context_tokens(
            &prompt_segments,
            &state.summaries,
            &outcome.compacted_prefix,
            &outcome.messages,
        );

        if let Some(compaction) = outcome.compaction.as_ref() {
            info!(
                compacted_messages = compaction.compacted_count,
                remaining_messages = outcome.messages.len(),
                compacted_prefix_items = outcome.compacted_prefix.len(),
                estimated_tokens,
                compaction_threshold = self.settings.compaction_threshold,
                "context manager compacted session state"
            );
        }

        let (previous_response_id, new_messages_start) = if outcome.compaction.is_none()
            && outcome.compacted_prefix.is_empty()
            && state.last_response_id.is_some()
            && state.messages_seen_by_provider > 0
        {
            let seen = state.messages_seen_by_provider;
            let total = state.messages.len();
            let window_offset = total.saturating_sub(outcome.messages.len());
            let new_start = seen
                .saturating_sub(window_offset)
                .min(outcome.messages.len());
            (state.last_response_id.clone(), new_start)
        } else {
            (None, 0)
        };

        Ok(ContextPlan {
            prompt_segments,
            transcript_window: TranscriptWindow {
                messages: outcome.messages.clone(),
                elided_message_count: state.messages.len().saturating_sub(outcome.messages.len())
                    as u64,
            },
            compacted_prefix: outcome.compacted_prefix.clone(),
            file_views,
            carried_summaries: state.summaries.clone(),
            elided_tool_results: Vec::new(),
            memory_items: Vec::new(),
            tool_specs: tool_specs.to_vec(),
            observed_state: observed.clone(),
            projected_input_tokens: estimated_tokens,
            cache_boundary_hash: cache_boundary_hash(),
            messages: outcome.messages,
            estimated_tokens,
            compaction: outcome.compaction,
            previous_response_id,
            new_messages_start,
        })
    }

    async fn compact_now(
        &self,
        blueprint: &SessionBlueprint,
        state: &SessionState,
        _observed: &ObservedState,
        _snapshot: &ResourceSnapshot,
        tool_specs: &[ToolSpec],
        compaction_model: &ResolvedModel,
        compaction_provider: &(dyn Provider + Send + Sync),
        custom_instructions: Option<&str>,
    ) -> anyhow::Result<CompactionOutcome> {
        let mut prompt_segments = blueprint.system_prompt_seed.clone();
        prompt_segments.extend(state.appended_prompt_segments.clone());
        self.execute_compaction(
            blueprint,
            state,
            &prompt_segments,
            tool_specs,
            compaction_model,
            compaction_provider,
            true,
            custom_instructions,
        )
        .await
    }
}

fn cache_boundary_hash() -> String {
    let mut hasher = Sha256::new();
    hasher.update(b"transcript_boundary_v2");
    format!("{:x}", hasher.finalize())
}

fn compaction_instructions(custom_instructions: Option<&str>) -> String {
    let base = DEFAULT_COMPACTION_PROMPT_MARKDOWN.trim();
    if let Some(custom_instructions) =
        custom_instructions.filter(|instructions| !instructions.trim().is_empty())
    {
        format!("{base}\n\n{custom_instructions}")
    } else {
        base.to_owned()
    }
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use halter_protocol::{
        ModelId, ModelRole, ProviderCapabilities, ProviderKind, ProviderName, ResolvedModel,
        SessionId, SummarySlice, ToolCallIdPolicy, Usage, UserMessage,
    };

    use super::*;

    #[tokio::test]
    async fn plan_disables_previous_response_chaining_when_compacted_prefix_exists() {
        let manager = DefaultContextManager::default();
        let outcome = manager
            .plan(
                &SessionBlueprint {
                    session_id: SessionId::new(),
                    parent_session_id: None,
                    default_model: "default".into(),
                    subagent_model: "subagent".into(),
                    snapshot_revision: "r1".into(),
                    working_dir: ".".into(),
                    system_prompt_seed: Vec::new(),
                    max_turns: None,
                    subagent_depth: 0,
                },
                &SessionState {
                    compacted_prefix: vec![serde_json::json!({
                        "type": "compaction",
                        "id": "cmp_1",
                        "encrypted_content": "x",
                    })],
                    summaries: vec![SummarySlice {
                        id: "summary-1".to_owned(),
                        text: "summary".to_owned(),
                    }],
                    messages: vec![Message::User(UserMessage::text("hello"))],
                    last_response_id: Some("resp_1".to_owned()),
                    messages_seen_by_provider: 1,
                    ..SessionState::default()
                },
                &ObservedState {
                    cwd: ".".into(),
                    git_branch: None,
                    git_dirty: None,
                    now_utc: Utc::now(),
                    env_facts: Default::default(),
                },
                &ResourceSnapshot::empty(),
                &[],
                &ResolvedModel {
                    role: ModelRole::default(),
                    id: ModelId::from("default"),
                    provider: ProviderName::from("fake"),
                    provider_kind: ProviderKind::Fake,
                    model: "fake".to_owned(),
                    max_input_tokens: None,
                    max_output_tokens: None,
                    reasoning: None,
                    tokens_per_minute: None,
                },
                &NoopProvider,
            )
            .await
            .expect("plan");

        assert!(outcome.previous_response_id.is_none());
    }

    struct NoopProvider;

    #[async_trait]
    impl Provider for NoopProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                supports_compaction: true,
                tool_call_id_policy: ToolCallIdPolicy::ProviderSupplied,
                ..ProviderCapabilities::default()
            }
        }

        async fn stream(
            &self,
            _request: halter_protocol::ProviderRequest,
            _cancel: tokio_util::sync::CancellationToken,
        ) -> anyhow::Result<
            futures::stream::BoxStream<
                'static,
                Result<halter_protocol::StreamEvent, halter_protocol::ProviderError>,
            >,
        > {
            anyhow::bail!("stream should not be called in this test");
        }

        async fn compact(
            &self,
            _request: ProviderCompactionRequest,
            _cancel: tokio_util::sync::CancellationToken,
        ) -> anyhow::Result<halter_protocol::ProviderCompactionResponse> {
            Ok(halter_protocol::ProviderCompactionResponse {
                output: vec![serde_json::json!({
                    "type": "compaction",
                    "id": "cmp_1",
                    "encrypted_content": "summary",
                })],
                usage: Usage::default(),
            })
        }
    }

    #[test]
    fn compaction_instructions_append_custom_text() {
        let instructions = compaction_instructions(Some("Focus on decisions."));
        assert!(instructions.contains("Compress the conversation"));
        assert!(instructions.contains("Focus on decisions."));
    }

    #[test]
    fn compaction_instructions_ignore_blank_custom_text() {
        assert_eq!(
            compaction_instructions(Some("   ")),
            DEFAULT_COMPACTION_PROMPT_MARKDOWN.trim()
        );
    }
}
