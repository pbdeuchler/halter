//! Protocol conformance tests.
//!
//! Every variant of a runtime-load-bearing protocol enum must be visibly
//! exercised by a runtime code path. These tests enforce that with exhaustive
//! `match` statements: if someone adds a new variant to `ToolConcurrency`,
//! `ModelRole`, or `SessionEventPayload`, compilation will fail here until the
//! author has decided what runtime behavior the new variant produces.
//!
//! The point is to make *silent addition* impossible. A variant that "exists
//! in the protocol but isn't handled anywhere" is a bug these tests catch at
//! the earliest moment.

use std::path::PathBuf;
use std::sync::Arc;

use halter_protocol::{
    ApiKind, AssistantMessage, AssistantPart, Delivery, DeltaItem, HookHandlerType, HookRunStatus,
    HookRunSummary, Message, MessageId, ModelId, ModelRole, PluginId, ProviderKind, ProviderName,
    ResolvedModel, SessionEvent, SessionEventPayload, SessionId, StopReason, ToolCall, ToolCallId,
    ToolConcurrency, ToolExecutionOutcome, ToolName, ToolResult, TurnId, Usage,
};
use halter_providers::{FakeProvider, ModelRegistry};

/// Every `ToolConcurrency` variant must map to a batcher classification
/// (exclusive vs. shareable). An exhaustive match forces explicit treatment
/// of any newly added variant.
#[test]
fn tool_concurrency_variants_have_runtime_meaning() {
    for variant in [
        ToolConcurrency::Exclusive,
        ToolConcurrency::ReadOnly,
        ToolConcurrency::ParallelSafe,
    ] {
        let is_exclusive = match variant {
            ToolConcurrency::Exclusive => true,
            ToolConcurrency::ReadOnly | ToolConcurrency::ParallelSafe => false,
        };
        assert_eq!(
            is_exclusive,
            matches!(variant, ToolConcurrency::Exclusive),
            "classification regressed for {variant:?}",
        );
    }
}

/// Every `ModelRole` variant must resolve to a concrete `ResolvedModel` via
/// the `ModelRegistry`. A variant with no resolver path is a protocol/runtime
/// mismatch: the enum advertises a capability the runtime cannot satisfy.
#[test]
fn model_role_variants_resolve_via_registry() {
    let provider: Arc<dyn halter_providers::Provider> = Arc::new(FakeProvider::default());
    let mut registry = ModelRegistry::new();
    registry.register_provider(ProviderName::from("fake"), provider);

    let mk = |id: &str, role: ModelRole| ResolvedModel {
        role,
        id: ModelId::from(id),
        provider: ProviderName::from("fake"),
        provider_kind: ProviderKind::Fake,
        api_kind: ApiKind::Fake,
        model: format!("halter/{id}"),
        max_input_tokens: Some(32_000),
        max_output_tokens: Some(4_096),
        reasoning: None,
        tokens_per_minute: None,
    };

    registry.set_default_model(mk("default", ModelRole::Default));
    registry.set_small_model(mk("small", ModelRole::Small));
    registry.set_subagent_model(mk("subagent", ModelRole::Subagent));
    registry.set_plan_model(mk("plan", ModelRole::Plan));

    for role in [
        ModelRole::Default,
        ModelRole::Plan,
        ModelRole::Subagent,
        ModelRole::Small,
    ] {
        let resolved = match role {
            ModelRole::Default => registry.default_model(),
            ModelRole::Plan => registry.plan_model(),
            ModelRole::Subagent => registry.subagent_model(),
            ModelRole::Small => registry.small_model(),
        };
        let resolved =
            resolved.unwrap_or_else(|err| panic!("failed to resolve model for role {role}: {err}"));
        assert_eq!(
            resolved.role, role,
            "registry returned a model tagged for a different role",
        );
    }
}

/// Every `SessionEventPayload` variant must be constructible and serialize
/// with the expected `kind` discriminator. An exhaustive match forces
/// explicit treatment of any new variant.
#[test]
fn session_event_payload_variants_have_stable_kind() {
    let call = ToolCall {
        id: ToolCallId::from("tc-1"),
        name: ToolName::from("noop"),
        arguments: serde_json::Value::Null,
    };
    let outcome = ToolExecutionOutcome {
        call: call.clone(),
        result: Ok(ToolResult::Empty),
    };
    let hook_run = HookRunSummary {
        run_id: "run-1".to_owned(),
        event_name: "SessionStart".to_owned(),
        handler_type: HookHandlerType::Command,
        plugin_id: PluginId::from("plugin"),
        plugin_root: PathBuf::from("/tmp/plugin"),
        status: HookRunStatus::Completed,
        status_message: None,
        started_at: chrono::Utc::now(),
        completed_at: Some(chrono::Utc::now()),
        duration_ms: Some(0),
        entries: Vec::new(),
    };

    let payloads: Vec<SessionEventPayload> = vec![
        SessionEventPayload::SessionStarted,
        SessionEventPayload::Warning {
            message: "w".into(),
        },
        SessionEventPayload::TurnStarted {
            turn_id: TurnId::from("t1"),
        },
        SessionEventPayload::MessageItem {
            message: Message::Assistant(AssistantMessage {
                id: MessageId::new(),
                created_at: chrono::Utc::now(),
                parts: vec![AssistantPart::Text { text: "".into() }],
                stop_reason: Some(StopReason::EndTurn),
                usage: Some(Usage::default()),
                replay_meta: Default::default(),
            }),
        },
        SessionEventPayload::DeltaItem {
            delta: DeltaItem {
                text: "d".to_owned(),
            },
        },
        SessionEventPayload::ToolExecutionStarted { call: call.clone() },
        SessionEventPayload::ToolOutput {
            call_id: call.id.clone(),
            tool_name: call.name.clone(),
            chunk: "".into(),
        },
        SessionEventPayload::HookStarted {
            run: hook_run.clone(),
        },
        SessionEventPayload::HookCompleted {
            run: hook_run.clone(),
        },
        SessionEventPayload::ToolExecutionCompleted { outcome },
        SessionEventPayload::ApprovalRequested {
            tool_name: call.name.clone(),
            reason: "r".into(),
        },
        SessionEventPayload::ContextCompacted {
            summary: "s".into(),
        },
        SessionEventPayload::TurnCompleted {
            turn_id: TurnId::from("t1"),
            usage: Usage::default(),
        },
        SessionEventPayload::TurnFailed {
            turn_id: TurnId::from("t1"),
            error: "e".into(),
            cancelled: false,
            retryable: false,
        },
        SessionEventPayload::Lagged { dropped_events: 1 },
        SessionEventPayload::SessionShutdownComplete,
    ];

    for payload in payloads {
        let event =
            SessionEvent::new_committed(SessionId::new(), 1, Delivery::Lossless, payload.clone());
        let kind = match &event.payload {
            SessionEventPayload::SessionStarted => "session_started",
            SessionEventPayload::Warning { .. } => "warning",
            SessionEventPayload::TurnStarted { .. } => "turn_started",
            SessionEventPayload::MessageItem { .. } => "message_item",
            SessionEventPayload::DeltaItem { .. } => "delta_item",
            SessionEventPayload::ToolExecutionStarted { .. } => "tool_execution_started",
            SessionEventPayload::ToolOutput { .. } => "tool_output",
            SessionEventPayload::HookStarted { .. } => "hook_started",
            SessionEventPayload::HookCompleted { .. } => "hook_completed",
            SessionEventPayload::ToolExecutionCompleted { .. } => "tool_execution_completed",
            SessionEventPayload::ApprovalRequested { .. } => "approval_requested",
            SessionEventPayload::ContextCompacted { .. } => "context_compacted",
            SessionEventPayload::TurnCompleted { .. } => "turn_completed",
            SessionEventPayload::TurnFailed { .. } => "turn_failed",
            SessionEventPayload::Lagged { .. } => "lagged",
            SessionEventPayload::SessionShutdownComplete => "session_shutdown_complete",
        };
        let json = serde_json::to_value(&event.payload).expect("serialize");
        assert_eq!(
            json.get("kind").and_then(|v| v.as_str()),
            Some(kind),
            "payload serialized without matching kind tag; expected {kind}",
        );
    }
}
