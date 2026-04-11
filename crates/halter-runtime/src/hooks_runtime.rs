// pattern: Imperative Shell

use std::collections::BTreeSet;
use std::path::Path;

use halter_hooks::{HookDispatchRequest, HookEventName, HookMergedOutcome};
use halter_protocol::{
    AssistantMessage, AssistantPart, HookRunSummary, HookSessionStartSource, ModelId, SessionState,
    ToolCall, ToolError, ToolResult, TurnId,
};
use serde_json::{Value, json};

use crate::HalterSession;

#[derive(Clone, Copy)]
pub struct HookInvocationContext<'a> {
    pub turn_id: &'a TurnId,
    pub model: &'a ModelId,
    pub working_dir: &'a Path,
}

pub struct ExecutedHookDispatch {
    pub preview_runs: Vec<HookRunSummary>,
    pub completed_runs: Vec<HookRunSummary>,
    pub merged: HookMergedOutcome,
    pub fired_hook_ids: Vec<String>,
}

pub async fn run_session_start(
    sess: &HalterSession,
    state: &SessionState,
    ctx: HookInvocationContext<'_>,
    source: HookSessionStartSource,
) -> anyhow::Result<ExecutedHookDispatch> {
    execute_hooks(
        sess,
        state,
        HookDispatchRequest {
            event_name: HookEventName::SessionStart,
            matcher_value: Some(session_start_source_name(source).to_owned()),
            payload: base_payload(sess, &ctx, HookEventName::SessionStart, json!({
                "source": session_start_source_name(source),
            })),
            fired_hook_ids: fired_hook_ids(state),
        },
    )
    .await
}

pub async fn run_user_prompt_submit(
    sess: &HalterSession,
    state: &SessionState,
    ctx: HookInvocationContext<'_>,
    prompt: &str,
) -> anyhow::Result<ExecutedHookDispatch> {
    execute_hooks(
        sess,
        state,
        HookDispatchRequest {
            event_name: HookEventName::UserPromptSubmit,
            matcher_value: None,
            payload: base_payload(sess, &ctx, HookEventName::UserPromptSubmit, json!({
                "prompt": prompt,
            })),
            fired_hook_ids: fired_hook_ids(state),
        },
    )
    .await
}

pub async fn run_pre_tool_use(
    sess: &HalterSession,
    state: &SessionState,
    ctx: HookInvocationContext<'_>,
    call: &ToolCall,
) -> anyhow::Result<ExecutedHookDispatch> {
    execute_hooks(
        sess,
        state,
        HookDispatchRequest {
            event_name: HookEventName::PreToolUse,
            matcher_value: Some(call.name.0.clone()),
            payload: base_payload(sess, &ctx, HookEventName::PreToolUse, json!({
                "tool_name": call.name.0,
                "tool_input": call.arguments,
                "tool_use_id": call.id,
            })),
            fired_hook_ids: fired_hook_ids(state),
        },
    )
    .await
}

pub async fn run_post_tool_use(
    sess: &HalterSession,
    state: &SessionState,
    ctx: HookInvocationContext<'_>,
    call: &ToolCall,
    result: &ToolResult,
) -> anyhow::Result<ExecutedHookDispatch> {
    execute_hooks(
        sess,
        state,
        HookDispatchRequest {
            event_name: HookEventName::PostToolUse,
            matcher_value: Some(call.name.0.clone()),
            payload: base_payload(sess, &ctx, HookEventName::PostToolUse, json!({
                "tool_name": call.name.0,
                "tool_input": call.arguments,
                "tool_use_id": call.id,
                "tool_response": result,
            })),
            fired_hook_ids: fired_hook_ids(state),
        },
    )
    .await
}

pub async fn run_post_tool_use_failure(
    sess: &HalterSession,
    state: &SessionState,
    ctx: HookInvocationContext<'_>,
    call: &ToolCall,
    error: &ToolError,
) -> anyhow::Result<ExecutedHookDispatch> {
    execute_hooks(
        sess,
        state,
        HookDispatchRequest {
            event_name: HookEventName::PostToolUseFailure,
            matcher_value: Some(call.name.0.clone()),
            payload: base_payload(sess, &ctx, HookEventName::PostToolUseFailure, json!({
                "tool_name": call.name.0,
                "tool_input": call.arguments,
                "tool_use_id": call.id,
                "error": error.message,
            })),
            fired_hook_ids: fired_hook_ids(state),
        },
    )
    .await
}

pub async fn run_stop(
    sess: &HalterSession,
    state: &SessionState,
    ctx: HookInvocationContext<'_>,
    last_message: Option<&AssistantMessage>,
    stop_hook_active: bool,
) -> anyhow::Result<ExecutedHookDispatch> {
    execute_hooks(
        sess,
        state,
        HookDispatchRequest {
            event_name: HookEventName::Stop,
            matcher_value: None,
            payload: base_payload(sess, &ctx, HookEventName::Stop, json!({
                "stop_hook_active": stop_hook_active,
                "last_assistant_message": last_message.map(render_assistant_text),
            })),
            fired_hook_ids: fired_hook_ids(state),
        },
    )
    .await
}

async fn execute_hooks(
    sess: &HalterSession,
    _state: &SessionState,
    request: HookDispatchRequest,
) -> anyhow::Result<ExecutedHookDispatch> {
    let hooks = sess.services().resources.hooks();
    let prepared = hooks.prepare(request);
    let preview_runs = prepared.preview_runs().to_vec();
    let outcome = prepared.run().await;
    Ok(ExecutedHookDispatch {
        preview_runs,
        completed_runs: outcome.runs,
        merged: outcome.merged,
        fired_hook_ids: outcome.fired_hook_ids,
    })
}

fn base_payload(
    sess: &HalterSession,
    ctx: &HookInvocationContext<'_>,
    event_name: HookEventName,
    extra: Value,
) -> Value {
    let mut payload = serde_json::Map::new();
    payload.insert(
        "hook_event_name".to_owned(),
        Value::String(event_name.canonical_name().to_owned()),
    );
    payload.insert("halter_version".to_owned(), Value::from(1));
    payload.insert(
        "session_id".to_owned(),
        Value::String(sess.session_id().0.clone()),
    );
    payload.insert("turn_id".to_owned(), Value::String(ctx.turn_id.0.clone()));
    payload.insert(
        "cwd".to_owned(),
        Value::String(ctx.working_dir.display().to_string()),
    );
    payload.insert("model".to_owned(), Value::String(ctx.model.0.clone()));

    if let Value::Object(extra) = extra {
        payload.extend(extra);
    }

    Value::Object(payload)
}

fn fired_hook_ids(state: &SessionState) -> BTreeSet<String> {
    state.fired_hook_ids.iter().cloned().collect()
}

fn session_start_source_name(source: HookSessionStartSource) -> &'static str {
    match source {
        HookSessionStartSource::Startup => "startup",
        HookSessionStartSource::Resume => "resume",
        HookSessionStartSource::Clear => "clear",
        HookSessionStartSource::Compact => "compact",
    }
}

fn render_assistant_text(message: &AssistantMessage) -> String {
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
