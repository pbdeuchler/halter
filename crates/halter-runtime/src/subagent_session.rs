// pattern: Functional Core

use halter_protocol::{
    AgentName, AssistantMessage, AssistantPart, CacheScope, ContentHash, Message, PromptSegment,
    PromptSegmentId, PromptSegmentKind, SessionEvent, SessionEventPayload, SessionId, SessionState,
    SpawnSubagentRequest, SubagentRef, Usage, Volatility,
};
use halter_tools::SubagentParentContext;
use sha2::{Digest, Sha256};

use crate::SessionInit;

pub fn build_subagent_session_init(
    parent: &SubagentParentContext,
    child_session_id: &SessionId,
    request: &SpawnSubagentRequest,
) -> anyhow::Result<SessionInit> {
    let mut system_prompt_seed = parent.blueprint.system_prompt_seed.clone();
    if let Some(agent_type) = request.agent_type.as_ref() {
        system_prompt_seed.push(build_agent_prompt_segment(
            parent.snapshot.as_ref(),
            agent_type,
        )?);
    }

    Ok(SessionInit {
        session_id: Some(child_session_id.clone()),
        parent_session_id: Some(parent.blueprint.session_id.clone()),
        working_dir: parent.blueprint.working_dir.clone(),
        system_prompt_seed,
        max_turns: parent.blueprint.max_turns,
        default_model: Some(
            request
                .model
                .clone()
                .unwrap_or_else(|| parent.subagent_model.clone()),
        ),
        subagent_model: Some(parent.subagent_model.clone()),
        subagent_depth: parent.blueprint.subagent_depth + 1,
    })
}

pub fn build_subagent_state(
    parent: &SubagentParentContext,
    child_session_id: &SessionId,
    task: &str,
    fork_context: bool,
) -> SessionState {
    let lineage = build_lineage(&parent.state, child_session_id, task);
    if !fork_context {
        return SessionState {
            lineage,
            ..SessionState::default()
        };
    }

    SessionState {
        messages: parent.state.messages.clone(),
        compacted_prefix: parent.state.compacted_prefix.clone(),
        file_view_cache: parent.state.file_view_cache.clone(),
        appended_prompt_segments: parent.state.appended_prompt_segments.clone(),
        pending_tool_calls: Default::default(),
        usage_so_far: Usage::default(),
        summaries: parent.state.summaries.clone(),
        lineage,
        fired_hook_ids: parent.state.fired_hook_ids.clone(),
        pending_session_start_source: None,
        pending_warning_messages: parent.state.pending_warning_messages.clone(),
        last_response_id: None,
        messages_seen_by_provider: 0,
    }
}

pub fn extract_subagent_output(events: &[SessionEvent]) -> Option<String> {
    events.iter().rev().find_map(|event| match &event.payload {
        SessionEventPayload::MessageItem {
            message: Message::Assistant(message),
        } => Some(render_assistant_output(message)),
        _ => None,
    })
}

pub fn extract_subagent_usage(events: &[SessionEvent]) -> Option<Usage> {
    events.iter().rev().find_map(|event| match &event.payload {
        SessionEventPayload::TurnCompleted { usage, .. } => Some(usage.clone()),
        _ => None,
    })
}

fn build_agent_prompt_segment(
    snapshot: &halter_protocol::ResourceSnapshot,
    agent_type: &AgentName,
) -> anyhow::Result<PromptSegment> {
    let agent = snapshot.agents.get(agent_type).ok_or_else(|| {
        let available_agent_types = snapshot
            .agents
            .keys()
            .map(|name| name.0.as_str())
            .collect::<Vec<_>>();
        if available_agent_types.is_empty() {
            anyhow::anyhow!(
                "failed to execute spawn_agent tool: unknown agent_type '{}'; no named agent roles are loaded, omit agent_type to use the default child session",
                agent_type.0
            )
        } else {
            anyhow::anyhow!(
                "failed to execute spawn_agent tool: unknown agent_type '{}'; available agent_type values: {}",
                agent_type.0,
                available_agent_types.join(", ")
            )
        }
    })?;
    Ok(PromptSegment {
        id: PromptSegmentId::new(),
        text: agent.prompt.clone(),
        volatility: Volatility::SessionStable,
        cache_scope: CacheScope::PrefixCacheable,
        content_hash: hash_text(&agent.prompt),
        kind: PromptSegmentKind::System,
    })
}

fn build_lineage(
    parent: &SessionState,
    child_session_id: &SessionId,
    task: &str,
) -> Vec<SubagentRef> {
    let mut lineage = parent.lineage.clone();
    lineage.push(SubagentRef {
        session_id: child_session_id.clone(),
        task: task.to_owned(),
    });
    lineage
}

fn render_assistant_output(message: &AssistantMessage) -> String {
    message
        .parts
        .iter()
        .filter_map(|part| match part {
            AssistantPart::Text { text } => Some(text.clone()),
            AssistantPart::Thinking(_) | AssistantPart::ToolCall(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn hash_text(text: &str) -> ContentHash {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use halter_protocol::{
        AgentDef, AgentId, MessageId, PromptSegment, Revision, SessionBlueprint, SessionEvent,
        SessionEventPayload,
    };

    #[test]
    fn build_subagent_state_clears_pending_and_usage() {
        let parent = SubagentParentContext {
            blueprint: SessionBlueprint {
                session_id: SessionId::from("parent"),
                parent_session_id: None,
                default_model: "default".into(),
                subagent_model: "subagent".into(),
                snapshot_revision: Revision::from("revision"),
                working_dir: ".".into(),
                system_prompt_seed: Vec::new(),
                max_turns: None,
                subagent_depth: 0,
            },
            state: SessionState {
                usage_so_far: Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 0,
                },
                pending_tool_calls: Default::default(),
                ..SessionState::default()
            },
            snapshot: Arc::new(halter_protocol::ResourceSnapshot::empty()),
            subagent_model: "subagent".into(),
        };

        let child = build_subagent_state(&parent, &SessionId::from("child"), "task", true);

        assert_eq!(child.usage_so_far, Usage::default());
        assert!(child.pending_tool_calls.is_empty());
        assert_eq!(child.lineage.len(), 1);
        assert_eq!(child.lineage[0].session_id, SessionId::from("child"));
    }

    #[test]
    fn build_subagent_session_init_appends_agent_prompt() {
        let mut snapshot = halter_protocol::ResourceSnapshot::empty();
        snapshot.agents.insert(
            AgentName::from("helper"),
            AgentDef {
                id: AgentId::new(),
                name: "helper".to_owned(),
                prompt: "specialized helper prompt".to_owned(),
            },
        );
        let parent = SubagentParentContext {
            blueprint: SessionBlueprint {
                session_id: SessionId::from("parent"),
                parent_session_id: None,
                default_model: "default".into(),
                subagent_model: "subagent".into(),
                snapshot_revision: Revision::from("revision"),
                working_dir: ".".into(),
                system_prompt_seed: vec![PromptSegment {
                    id: PromptSegmentId::new(),
                    text: "base prompt".to_owned(),
                    volatility: Volatility::Static,
                    cache_scope: CacheScope::PrefixCacheable,
                    content_hash: "base".into(),
                    kind: PromptSegmentKind::System,
                }],
                max_turns: Some(4),
                subagent_depth: 1,
            },
            state: SessionState::default(),
            snapshot: Arc::new(snapshot),
            subagent_model: "subagent".into(),
        };

        let init = build_subagent_session_init(
            &parent,
            &SessionId::from("child"),
            &SpawnSubagentRequest {
                message: "task".to_owned(),
                agent_type: Some(AgentName::from("helper")),
                fork_context: true,
                model: Some("custom".into()),
            },
        )
        .expect("init");

        assert_eq!(init.default_model, Some("custom".into()));
        assert_eq!(init.subagent_model, Some("subagent".into()));
        assert_eq!(init.subagent_depth, 2);
        assert_eq!(init.system_prompt_seed.len(), 2);
        assert_eq!(init.system_prompt_seed[1].text, "specialized helper prompt");
    }

    #[test]
    fn build_subagent_session_init_reports_available_roles_for_unknown_agent_type() {
        let mut snapshot = halter_protocol::ResourceSnapshot::empty();
        snapshot.agents.insert(
            AgentName::from("helper"),
            AgentDef {
                id: AgentId::new(),
                name: "helper".to_owned(),
                prompt: "specialized helper prompt".to_owned(),
            },
        );
        let parent = SubagentParentContext {
            blueprint: SessionBlueprint {
                session_id: SessionId::from("parent"),
                parent_session_id: None,
                default_model: "default".into(),
                subagent_model: "subagent".into(),
                snapshot_revision: Revision::from("revision"),
                working_dir: ".".into(),
                system_prompt_seed: Vec::new(),
                max_turns: None,
                subagent_depth: 0,
            },
            state: SessionState::default(),
            snapshot: Arc::new(snapshot),
            subagent_model: "subagent".into(),
        };

        let error = build_subagent_session_init(
            &parent,
            &SessionId::from("child"),
            &SpawnSubagentRequest {
                message: "task".to_owned(),
                agent_type: Some(AgentName::from("general")),
                fork_context: false,
                model: None,
            },
        )
        .expect_err("unknown agent type should fail");

        assert!(
            error
                .to_string()
                .contains("available agent_type values: helper")
        );
    }

    #[test]
    fn extract_subagent_output_reads_last_assistant_text() {
        let events = vec![SessionEvent::new_committed(
            SessionId::from("child"),
            1,
            halter_protocol::Delivery::Lossless,
            SessionEventPayload::MessageItem {
                message: Message::Assistant(AssistantMessage {
                    id: MessageId::new(),
                    created_at: chrono::Utc::now(),
                    parts: vec![
                        AssistantPart::Thinking(halter_protocol::ThinkingBlock {
                            text: "internal".to_owned(),
                            signature: None,
                        }),
                        AssistantPart::Text {
                            text: "done".to_owned(),
                        },
                    ],
                    stop_reason: Some(halter_protocol::StopReason::EndTurn),
                    usage: None,
                    replay_meta: halter_protocol::ReplayMeta::default(),
                }),
            },
        )];

        assert_eq!(extract_subagent_output(&events), Some("done".to_owned()));
    }
}
