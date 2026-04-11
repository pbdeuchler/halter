// pattern: Imperative Shell

use std::collections::BTreeSet;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;

use anyhow::Context;
use futures::stream::{FuturesUnordered, StreamExt};
use halter_hooks::{
    AgentHookConfig, CommandHookConfig, ConfiguredHandler, HOOK_PROTOCOL_VERSION,
    HookDispatchRequest, HookEventName, HookHandlerConfig, HookMergedOutcome, HookOutput,
    HttpHookConfig, MergeInput, PromptHookConfig, merge_outputs, summary_entries,
};
use halter_protocol::{
    AssembledPrompt, AssistantMessage, AssistantPart, CacheScope, HookOutputEntry, HookOutputKind,
    HookRunStatus, HookRunSummary, HookSessionStartSource, Message, ModelId, PromptSegment,
    PromptSegmentId, SessionId, SessionState, ToolCall, ToolError, ToolResult, Turn, TurnId,
    UserMessage, Volatility,
};
use halter_session::InMemorySessionStore;
use reqwest::Url;
use serde_json::{Value, json};
use tokio::io::AsyncWriteExt;
use tokio::net::lookup_host;
use tokio::process::Command;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;
use tracing::warn;

use crate::session::{
    MaterializedAssistantMessage, create_session_seeded, materialize_assistant_message,
};
use crate::{EventBus, HalterSession, ResourceHandle, RuntimeServices, SessionInit};

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
            payload: base_payload(
                sess,
                &ctx,
                HookEventName::SessionStart,
                json!({
                    "source": session_start_source_name(source),
                    "working_dir": ctx.working_dir.display().to_string(),
                }),
            ),
            fired_hook_ids: fired_hook_ids(state),
        },
    )
    .await
}

pub async fn run_session_end(
    sess: &HalterSession,
    state: &SessionState,
    ctx: HookInvocationContext<'_>,
    reason: &str,
) -> anyhow::Result<ExecutedHookDispatch> {
    execute_hooks(
        sess,
        state,
        HookDispatchRequest {
            event_name: HookEventName::SessionEnd,
            matcher_value: Some(reason.to_owned()),
            payload: base_payload(
                sess,
                &ctx,
                HookEventName::SessionEnd,
                json!({
                    "reason": reason,
                }),
            ),
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
            payload: base_payload(
                sess,
                &ctx,
                HookEventName::UserPromptSubmit,
                json!({
                    "prompt": prompt,
                }),
            ),
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
            payload: base_payload(
                sess,
                &ctx,
                HookEventName::PreToolUse,
                json!({
                    "tool_name": call.name.0,
                    "tool_input": call.arguments,
                    "tool_use_id": call.id,
                }),
            ),
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
            payload: base_payload(
                sess,
                &ctx,
                HookEventName::PostToolUse,
                json!({
                    "tool_name": call.name.0,
                    "tool_input": call.arguments,
                    "tool_use_id": call.id,
                    "tool_response": result,
                }),
            ),
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
            payload: base_payload(
                sess,
                &ctx,
                HookEventName::PostToolUseFailure,
                json!({
                    "tool_name": call.name.0,
                    "tool_input": call.arguments,
                    "tool_use_id": call.id,
                    "error": error.message,
                }),
            ),
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
            payload: base_payload(
                sess,
                &ctx,
                HookEventName::Stop,
                json!({
                    "stop_hook_active": stop_hook_active,
                    "last_assistant_message": last_message.map(render_assistant_text),
                }),
            ),
            fired_hook_ids: fired_hook_ids(state),
        },
    )
    .await
}

pub async fn run_subagent_start(
    sess: &HalterSession,
    state: &SessionState,
    ctx: HookInvocationContext<'_>,
    agent_id: &halter_protocol::AgentId,
    agent_type: &str,
    parent_session_id: &halter_protocol::SessionId,
) -> anyhow::Result<ExecutedHookDispatch> {
    execute_hooks(
        sess,
        state,
        HookDispatchRequest {
            event_name: HookEventName::SubagentStart,
            matcher_value: Some(agent_type.to_owned()),
            payload: base_payload(
                sess,
                &ctx,
                HookEventName::SubagentStart,
                json!({
                    "agent_id": agent_id,
                    "agent_type": agent_type,
                    "parent_session_id": parent_session_id,
                }),
            ),
            fired_hook_ids: fired_hook_ids(state),
        },
    )
    .await
}

pub async fn run_subagent_stop(
    sess: &HalterSession,
    state: &SessionState,
    ctx: HookInvocationContext<'_>,
    agent_id: &halter_protocol::AgentId,
    agent_type: &str,
    transcript_path: Option<&Path>,
) -> anyhow::Result<ExecutedHookDispatch> {
    let mut extra = serde_json::Map::new();
    extra.insert("agent_id".to_owned(), serde_json::to_value(agent_id)?);
    extra.insert(
        "agent_type".to_owned(),
        Value::String(agent_type.to_owned()),
    );
    if let Some(transcript_path) = transcript_path {
        extra.insert(
            "agent_transcript_path".to_owned(),
            Value::String(transcript_path.display().to_string()),
        );
    }

    execute_hooks(
        sess,
        state,
        HookDispatchRequest {
            event_name: HookEventName::SubagentStop,
            matcher_value: Some(agent_type.to_owned()),
            payload: base_payload(
                sess,
                &ctx,
                HookEventName::SubagentStop,
                Value::Object(extra),
            ),
            fired_hook_ids: fired_hook_ids(state),
        },
    )
    .await
}

pub async fn run_pre_compact(
    sess: &HalterSession,
    state: &SessionState,
    ctx: HookInvocationContext<'_>,
    trigger: &str,
    custom_instructions: Option<&str>,
) -> anyhow::Result<ExecutedHookDispatch> {
    execute_hooks(
        sess,
        state,
        HookDispatchRequest {
            event_name: HookEventName::PreCompact,
            matcher_value: Some(trigger.to_owned()),
            payload: base_payload(
                sess,
                &ctx,
                HookEventName::PreCompact,
                json!({
                    "trigger": trigger,
                    "custom_instructions": custom_instructions,
                }),
            ),
            fired_hook_ids: fired_hook_ids(state),
        },
    )
    .await
}

pub async fn run_post_compact(
    sess: &HalterSession,
    state: &SessionState,
    ctx: HookInvocationContext<'_>,
    trigger: &str,
    summary: &str,
) -> anyhow::Result<ExecutedHookDispatch> {
    execute_hooks(
        sess,
        state,
        HookDispatchRequest {
            event_name: HookEventName::PostCompact,
            matcher_value: Some(trigger.to_owned()),
            payload: base_payload(
                sess,
                &ctx,
                HookEventName::PostCompact,
                json!({
                    "trigger": trigger,
                    "compact_summary": summary,
                }),
            ),
            fired_hook_ids: fired_hook_ids(state),
        },
    )
    .await
}

pub async fn run_notification(
    sess: &HalterSession,
    state: &SessionState,
    ctx: HookInvocationContext<'_>,
    notification_type: &str,
    message: &str,
) -> anyhow::Result<ExecutedHookDispatch> {
    execute_hooks(
        sess,
        state,
        HookDispatchRequest {
            event_name: HookEventName::Notification,
            matcher_value: Some(notification_type.to_owned()),
            payload: base_payload(
                sess,
                &ctx,
                HookEventName::Notification,
                json!({
                    "notification_type": notification_type,
                    "message": message,
                }),
            ),
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
    let matched_handlers = prepared.matched_handlers().to_vec();
    let request = prepared.request().clone();

    let mut completed_runs = Vec::with_capacity(matched_handlers.len());
    let mut merge_inputs = Vec::new();
    let mut fired_hook_ids = Vec::new();
    let cancel = CancellationToken::new();
    let mut running = FuturesUnordered::new();

    for (handler, preview) in matched_handlers
        .iter()
        .cloned()
        .zip(preview_runs.iter().cloned())
    {
        let token = cancel.child_token();
        let request = request.clone();
        running.push(async move { run_handler(sess, &request, handler, preview, token).await });
    }

    while let Some(result) = running.next().await {
        if result.handler.once {
            fired_hook_ids.push(result.handler.handler_id.clone());
        }
        if let Some(output) = result.output.clone() {
            if should_cancel_siblings(&output) && !cancel.is_cancelled() {
                cancel.cancel();
            }
            merge_inputs.push(MergeInput {
                handler_id: result.handler.handler_id.clone(),
                priority: result.handler.priority.clone(),
                output,
            });
        }
        completed_runs.push(result.summary);
    }

    let (merged, conflicts) = merge_outputs(&merge_inputs);
    for conflict in conflicts {
        warn!(
            field = conflict.field,
            winner = %conflict.winner,
            loser = %conflict.loser,
            "hooks.merge_conflict"
        );
    }

    Ok(ExecutedHookDispatch {
        preview_runs,
        completed_runs,
        merged,
        fired_hook_ids,
    })
}

struct HandlerRunResult {
    handler: ConfiguredHandler,
    summary: HookRunSummary,
    output: Option<HookOutput>,
}

async fn run_handler(
    sess: &HalterSession,
    request: &HookDispatchRequest,
    handler: ConfiguredHandler,
    preview: HookRunSummary,
    cancel: CancellationToken,
) -> HandlerRunResult {
    let started_at = preview.started_at;
    let execution = execute_handler(sess, request, &handler, cancel).await;

    match execution {
        Ok(HandlerExecution::Completed(output)) => {
            let completed_at = chrono::Utc::now();
            let duration_ms = completed_at
                .signed_duration_since(started_at)
                .num_milliseconds()
                .max(0) as u64;
            let status = if matches!(output.continue_execution, Some(false)) {
                HookRunStatus::Stopped
            } else if matches!(output.decision, Some(halter_hooks::HookDecision::Block)) {
                HookRunStatus::Blocked
            } else {
                HookRunStatus::Completed
            };
            HandlerRunResult {
                handler,
                summary: HookRunSummary {
                    status,
                    completed_at: Some(completed_at),
                    duration_ms: Some(duration_ms),
                    entries: summary_entries(&output),
                    ..preview
                },
                output: Some(output),
            }
        }
        Ok(HandlerExecution::Cancelled) => {
            let completed_at = chrono::Utc::now();
            let duration_ms = completed_at
                .signed_duration_since(started_at)
                .num_milliseconds()
                .max(0) as u64;
            HandlerRunResult {
                handler,
                summary: HookRunSummary {
                    status: HookRunStatus::Cancelled,
                    completed_at: Some(completed_at),
                    duration_ms: Some(duration_ms),
                    entries: vec![HookOutputEntry {
                        kind: HookOutputKind::Warning,
                        text: "hook cancelled after higher-priority stop or block".to_owned(),
                    }],
                    ..preview
                },
                output: None,
            }
        }
        Err(error) => {
            let completed_at = chrono::Utc::now();
            let duration_ms = completed_at
                .signed_duration_since(started_at)
                .num_milliseconds()
                .max(0) as u64;
            HandlerRunResult {
                handler,
                summary: HookRunSummary {
                    status: HookRunStatus::Failed,
                    completed_at: Some(completed_at),
                    duration_ms: Some(duration_ms),
                    entries: vec![HookOutputEntry {
                        kind: HookOutputKind::Error,
                        text: error.to_string(),
                    }],
                    ..preview
                },
                output: None,
            }
        }
    }
}

enum HandlerExecution {
    Completed(HookOutput),
    Cancelled,
}

async fn execute_handler(
    sess: &HalterSession,
    request: &HookDispatchRequest,
    handler: &ConfiguredHandler,
    cancel: CancellationToken,
) -> anyhow::Result<HandlerExecution> {
    if cancel.is_cancelled() {
        return Ok(HandlerExecution::Cancelled);
    }

    match &handler.config {
        HookHandlerConfig::Command(command) => run_command(handler, command, request, cancel).await,
        HookHandlerConfig::Http(http) => run_http(handler, http, request, cancel).await,
        HookHandlerConfig::Prompt(prompt) => {
            run_prompt(sess, handler.timeout, prompt, request, cancel).await
        }
        HookHandlerConfig::Agent(agent) => {
            run_agent(sess, handler.timeout, agent, request, cancel).await
        }
    }
}

async fn run_command(
    handler: &ConfiguredHandler,
    command: &CommandHookConfig,
    request: &HookDispatchRequest,
    cancel: CancellationToken,
) -> anyhow::Result<HandlerExecution> {
    let expanded_command = expand_placeholders(&command.command, &handler.plugin_root);
    let mut child = build_command(handler, command, &expanded_command, request)?;
    let payload = build_payload(handler, request)?;
    let mut body = serde_json::to_vec(&payload)?;
    body.push(b'\n');

    let mut stdin = child
        .stdin
        .take()
        .context("failed to run hook command: missing stdin")?;
    stdin
        .write_all(&body)
        .await
        .context("failed to write hook stdin")?;
    stdin
        .shutdown()
        .await
        .context("failed to close hook stdin")?;

    let mut wait = tokio::spawn(async move { child.wait_with_output().await });
    let output = tokio::select! {
        _ = cancel.cancelled() => {
            wait.abort();
            return Ok(HandlerExecution::Cancelled);
        }
        output = timeout(handler.timeout, &mut wait) => {
            output
                .context("hook timed out")?
                .context("failed to join hook command task")?
                .context("failed to wait for hook command")?
        }
    };
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();

    match output.status.code() {
        Some(0) => Ok(HandlerExecution::Completed(parse_command_output(&stdout)?)),
        Some(2) => Ok(HandlerExecution::Completed(HookOutput {
            decision: Some(halter_hooks::HookDecision::Block),
            reason: (!stderr.is_empty()).then_some(stderr),
            ..HookOutput::default()
        })),
        Some(code) => {
            let detail = if stderr.is_empty() {
                format!("hook command exited with status {code}")
            } else {
                format!("hook command exited with status {code}: {stderr}")
            };
            anyhow::bail!(detail)
        }
        None => anyhow::bail!("hook command terminated by signal"),
    }
}

async fn run_http(
    handler: &ConfiguredHandler,
    config: &HttpHookConfig,
    request: &HookDispatchRequest,
    cancel: CancellationToken,
) -> anyhow::Result<HandlerExecution> {
    if matches!(
        handler.event_name,
        HookEventName::SessionStart | HookEventName::SessionEnd | HookEventName::Setup
    ) {
        anyhow::bail!(
            "http hooks are not allowed for {}",
            handler.event_name.canonical_name()
        );
    }

    let url = Url::parse(&config.url).context("failed to parse hook url")?;
    validate_http_url(handler, &url).await?;
    let payload = build_payload(handler, request)?;
    let headers = build_http_headers(handler, config)?;
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .context("failed to construct http hook client")?;

    let response = tokio::select! {
        _ = cancel.cancelled() => return Ok(HandlerExecution::Cancelled),
        result = timeout(handler.timeout, client.post(url).headers(headers).json(&payload).send()) => {
            result.context("hook timed out")?.context("failed to execute http hook")?
        }
    };

    if !response.status().is_success() {
        anyhow::bail!("http hook returned {}", response.status());
    }

    let body = response
        .text()
        .await
        .context("failed to read http hook response body")?;
    Ok(HandlerExecution::Completed(parse_json_hook_output(&body)?))
}

async fn run_prompt(
    sess: &HalterSession,
    timeout_limit: std::time::Duration,
    config: &PromptHookConfig,
    request: &HookDispatchRequest,
    cancel: CancellationToken,
) -> anyhow::Result<HandlerExecution> {
    let model = resolve_prompt_model(sess, config.model.as_deref())?;
    let provider = sess.services().models.provider(&model.provider)?;
    let payload_json = serde_json::to_string(&request.payload)?;
    let prompt_text = config.prompt.replace("$ARGUMENTS", &payload_json);
    let user_message = UserMessage::text(prompt_text.clone());
    let request = halter_protocol::ProviderRequest {
        session_id: SessionId::new(),
        turn_id: TurnId::new(),
        model: model.clone(),
        prompt: AssembledPrompt {
            segments: Vec::new(),
            transcript: vec![Message::User(user_message.clone())],
            ordered_segments: Vec::new(),
            prefix_cache_key: String::new(),
            rendered_prefix: String::new(),
            rendered_transcript: prompt_text.clone(),
            rendered: prompt_text,
        },
        messages: vec![Message::User(user_message)],
        tools: Vec::new(),
    };

    let materialized = tokio::select! {
        _ = cancel.cancelled() => return Ok(HandlerExecution::Cancelled),
        result = timeout(timeout_limit, async {
            let stream = provider.stream(request, cancel.child_token()).await?;
            materialize_assistant_message(stream, &model).await
        }) => result.context("hook timed out")??,
    };
    let MaterializedAssistantMessage { message, .. } = materialized;
    let text = render_assistant_text(&message);
    Ok(HandlerExecution::Completed(parse_json_hook_output(&text)?))
}

async fn run_agent(
    sess: &HalterSession,
    timeout_limit: std::time::Duration,
    config: &AgentHookConfig,
    request: &HookDispatchRequest,
    cancel: CancellationToken,
) -> anyhow::Result<HandlerExecution> {
    let resources = sess.services().resources.snapshot();
    let tools = Arc::new(sess.services().tools.clone_filtered(&config.allowed_tools));
    let services = Arc::new(RuntimeServices {
        resources: Arc::new(ResourceHandle::new(
            resources.as_ref().clone(),
            Arc::new(halter_hooks::Hooks::default()),
            Vec::new(),
        )),
        models: sess.services().models.clone(),
        tools,
        path_locks: sess.services().path_locks.clone(),
        tool_sessions: Arc::new(halter_tools::ToolSessionStore::default()),
        sessions: Arc::new(InMemorySessionStore::default()),
        policy: sess.services().policy.clone(),
        prompt_assembler: sess.services().prompt_assembler.clone(),
        context_manager: sess.services().context_manager.clone(),
        event_bus: Arc::new(EventBus::default()),
        max_tool_output_bytes: sess.services().max_tool_output_bytes,
        shell_timeout_secs: sess.services().shell_timeout_secs,
    });
    let model = resolve_agent_model(sess, config.model.as_deref())?;
    let working_dir = request
        .payload
        .get("cwd")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let init = SessionInit {
        session_id: Some(SessionId::new()),
        parent_session_id: Some(sess.session_id().clone()),
        working_dir,
        system_prompt_seed: vec![hook_prompt_segment(&config.prompt)],
        max_turns: config.max_turns,
        default_model: Some(model.id.clone()),
        subagent_model: Some(sess.services().models.subagent_model()?.id),
        subagent_depth: 0,
    };
    let initial_state = SessionState::default();
    let agent_session = create_session_seeded(services, init, initial_state, resources).await?;
    let payload_json = serde_json::to_string_pretty(&request.payload)?;
    let agent_cancel = cancel.child_token();
    let events = tokio::select! {
        _ = cancel.cancelled() => return Ok(HandlerExecution::Cancelled),
        result = timeout(
            timeout_limit,
            tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<halter_protocol::SessionEvent>> {
                let runtime = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .context("failed to build hook agent runtime")?;
                runtime.block_on(async move {
                    let stream = agent_session
                        .submit_turn_with_cancel(Turn::user(payload_json), agent_cancel)
                        .await?;
                    stream.collect::<Vec<_>>().await.into_iter().collect::<anyhow::Result<Vec<_>>>()
                })
            })
        ) => result.context("hook agent timed out")???,
    };
    let output = crate::subagent_session::extract_subagent_output(&events)
        .context("hook agent did not produce a final assistant message")?;
    Ok(HandlerExecution::Completed(parse_json_hook_output(
        &output,
    )?))
}

fn resolve_prompt_model(
    sess: &HalterSession,
    override_model: Option<&str>,
) -> anyhow::Result<halter_protocol::ResolvedModel> {
    match override_model {
        Some(model) => sess
            .services()
            .models
            .model(&ModelId::from(model.to_owned())),
        None => sess.services().models.small_model(),
    }
}

fn resolve_agent_model(
    sess: &HalterSession,
    override_model: Option<&str>,
) -> anyhow::Result<halter_protocol::ResolvedModel> {
    match override_model {
        Some(model) => sess
            .services()
            .models
            .model(&ModelId::from(model.to_owned())),
        None => sess.services().models.default_model(),
    }
}

fn parse_command_output(stdout: &str) -> anyhow::Result<HookOutput> {
    if stdout.is_empty() {
        return Ok(HookOutput::default());
    }
    serde_json::from_str(stdout).or(Ok(HookOutput::default()))
}

fn parse_json_hook_output(body: &str) -> anyhow::Result<HookOutput> {
    if body.trim().is_empty() {
        return Ok(HookOutput::default());
    }
    serde_json::from_str(body).context("failed to decode hook output json")
}

fn should_cancel_siblings(output: &HookOutput) -> bool {
    matches!(output.continue_execution, Some(false))
        || matches!(output.decision, Some(halter_hooks::HookDecision::Block))
}

fn build_command(
    handler: &ConfiguredHandler,
    command: &CommandHookConfig,
    expanded_command: &str,
    request: &HookDispatchRequest,
) -> anyhow::Result<tokio::process::Child> {
    let (program, args) = shell_invocation(command.shell, expanded_command);
    let mut child = Command::new(program);
    child
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    let cwd = request
        .payload
        .get("cwd")
        .and_then(Value::as_str)
        .unwrap_or(".");
    child.current_dir(cwd);
    child.env("PLUGIN_ROOT", &handler.plugin_root);
    child.env("CLAUDE_PLUGIN_ROOT", &handler.plugin_root);
    child.env("PLUGIN_ID", handler.plugin_id.0.as_str());
    child.env("HOOK_EVENT", handler.event_name.canonical_name());
    child.env("HOOK_VERSION", HOOK_PROTOCOL_VERSION.to_string());
    child.env("HOOK_SOURCE_PATH", &handler.source_path);
    if let Some(session_id) = request.payload.get("session_id").and_then(Value::as_str) {
        child.env("SESSION_ID", session_id);
    }
    if let Some(turn_id) = request.payload.get("turn_id").and_then(Value::as_str) {
        child.env("TURN_ID", turn_id);
    }
    if let Some(project_dir) = request.payload.get("cwd").and_then(Value::as_str) {
        child.env("PROJECT_DIR", project_dir);
        child.env("CLAUDE_PROJECT_DIR", project_dir);
    }

    for (key, value) in &command.env {
        child.env(key, expand_placeholders(value, &handler.plugin_root));
    }

    child
        .spawn()
        .context("failed to spawn hook command process")
}

fn build_payload(
    handler: &ConfiguredHandler,
    request: &HookDispatchRequest,
) -> anyhow::Result<Value> {
    let mut payload = request
        .payload
        .as_object()
        .cloned()
        .context("failed to build hook payload: expected object")?;
    payload.insert(
        "plugin_id".to_owned(),
        Value::String(handler.plugin_id.0.clone()),
    );
    payload.insert(
        "plugin_root".to_owned(),
        Value::String(handler.plugin_root.display().to_string()),
    );
    Ok(Value::Object(payload))
}

fn shell_invocation(shell: halter_hooks::HookShell, command: &str) -> (String, Vec<String>) {
    match shell {
        halter_hooks::HookShell::Bash => (
            std::env::var("SHELL")
                .ok()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| "/bin/sh".to_owned()),
            vec!["-lc".to_owned(), command.to_owned()],
        ),
        halter_hooks::HookShell::Pwsh => (
            "pwsh".to_owned(),
            vec!["-Command".to_owned(), command.to_owned()],
        ),
    }
}

fn expand_placeholders(value: &str, plugin_root: &Path) -> String {
    let plugin_root = plugin_root.display().to_string();
    let plugin_data = plugin_root.clone() + "/.data";
    value
        .replace("${PLUGIN_ROOT}", &plugin_root)
        .replace("${CLAUDE_PLUGIN_ROOT}", &plugin_root)
        .replace("${HALTER_PLUGIN_ROOT}", &plugin_root)
        .replace("${PLUGIN_DATA}", &plugin_data)
        .replace("${CLAUDE_PLUGIN_DATA}", &plugin_data)
        .replace("${HALTER_PLUGIN_DATA}", &plugin_data)
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
    payload.insert(
        "permission_mode".to_owned(),
        Value::String("default".to_owned()),
    );
    if let Some(path) = sess.services().sessions.transcript_path(sess.session_id()) {
        payload.insert(
            "transcript_path".to_owned(),
            Value::String(path.display().to_string()),
        );
    }

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

fn build_http_headers(
    handler: &ConfiguredHandler,
    config: &HttpHookConfig,
) -> anyhow::Result<reqwest::header::HeaderMap> {
    let mut headers = reqwest::header::HeaderMap::new();
    let allowed = config
        .allowed_env_vars
        .iter()
        .filter(|name| {
            handler
                .allowed_env_vars
                .iter()
                .any(|allowed| allowed == *name)
        })
        .cloned()
        .collect::<BTreeSet<_>>();

    for (key, raw_value) in &config.headers {
        let value = sanitize_header_value(&expand_env_placeholders(raw_value, &allowed));
        let header_name = reqwest::header::HeaderName::from_bytes(key.as_bytes())
            .context("failed to build hook header name")?;
        let header_value = reqwest::header::HeaderValue::from_str(&value)
            .context("failed to build hook header value")?;
        headers.insert(header_name, header_value);
    }

    Ok(headers)
}

fn expand_env_placeholders(value: &str, allowed: &BTreeSet<String>) -> String {
    let mut expanded = String::with_capacity(value.len());
    let chars = value.as_bytes();
    let mut index = 0usize;

    while index < chars.len() {
        if chars[index] != b'$' {
            expanded.push(chars[index] as char);
            index += 1;
            continue;
        }

        if index + 1 < chars.len() && chars[index + 1] == b'{' {
            if let Some(close) = value[index + 2..].find('}') {
                let name = &value[index + 2..index + 2 + close];
                expanded.push_str(&expanded_env(name, allowed));
                index += close + 3;
                continue;
            }
        }

        let start = index + 1;
        let mut end = start;
        while end < chars.len()
            && ((chars[end] as char).is_ascii_alphanumeric() || chars[end] == b'_')
        {
            end += 1;
        }
        let name = &value[start..end];
        expanded.push_str(&expanded_env(name, allowed));
        index = end;
    }

    expanded
}

fn expanded_env(name: &str, allowed: &BTreeSet<String>) -> String {
    if !allowed.contains(name) {
        return String::new();
    }
    std::env::var(name).unwrap_or_default()
}

fn sanitize_header_value(value: &str) -> String {
    value.replace(['\r', '\n', '\0'], "")
}

async fn validate_http_url(handler: &ConfiguredHandler, url: &Url) -> anyhow::Result<()> {
    let host = url
        .host_str()
        .context("http hook url must include a host")?;
    if !matches_allowed_host(host, &handler.allowed_http_hosts) {
        anyhow::bail!("http hook host '{host}' is not allowed by plugin manifest");
    }

    let port = url.port_or_known_default().unwrap_or(443);
    let addrs = lookup_host((host, port))
        .await
        .with_context(|| format!("failed to resolve http hook host '{host}'"))?
        .collect::<Vec<_>>();
    if addrs.is_empty() {
        anyhow::bail!("failed to resolve http hook host '{host}'");
    }
    for addr in addrs {
        if !is_allowed_http_ip(addr.ip()) {
            anyhow::bail!(
                "http hook host '{host}' resolved to disallowed address '{}'",
                addr.ip()
            );
        }
    }

    Ok(())
}

fn matches_allowed_host(host: &str, patterns: &[String]) -> bool {
    patterns
        .iter()
        .any(|pattern| matches_host_pattern(host, pattern))
}

fn matches_host_pattern(host: &str, pattern: &str) -> bool {
    let host_segments = host.split('.').collect::<Vec<_>>();
    let pattern_segments = pattern.split('.').collect::<Vec<_>>();
    if host_segments.len() != pattern_segments.len() {
        return false;
    }

    pattern_segments
        .iter()
        .zip(host_segments.iter())
        .all(|(pattern_segment, host_segment)| {
            *pattern_segment == "*" || pattern_segment.eq_ignore_ascii_case(host_segment)
        })
}

fn is_allowed_http_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            if ip.octets()[0] == 127 {
                return true;
            }
            !matches!(
                ip,
                ip if ip_in_v4_cidr(ip, Ipv4Addr::new(0, 0, 0, 0), 8)
                    || ip_in_v4_cidr(ip, Ipv4Addr::new(10, 0, 0, 0), 8)
                    || ip_in_v4_cidr(ip, Ipv4Addr::new(100, 64, 0, 0), 10)
                    || ip_in_v4_cidr(ip, Ipv4Addr::new(169, 254, 0, 0), 16)
                    || ip_in_v4_cidr(ip, Ipv4Addr::new(172, 16, 0, 0), 12)
                    || ip_in_v4_cidr(ip, Ipv4Addr::new(192, 168, 0, 0), 16)
            )
        }
        IpAddr::V6(ip) => {
            if ip == Ipv6Addr::LOCALHOST {
                return true;
            }
            let segments = ip.segments();
            !(ip == Ipv6Addr::UNSPECIFIED
                || (segments[0] & 0xfe00) == 0xfc00
                || (segments[0] & 0xffc0) == 0xfe80
                || (segments[0] == 0
                    && segments[1] == 0
                    && segments[2] == 0
                    && segments[3] == 0
                    && segments[4] == 0
                    && segments[5] == 0xffff))
        }
    }
}

fn ip_in_v4_cidr(ip: Ipv4Addr, network: Ipv4Addr, prefix: u8) -> bool {
    let mask = if prefix == 0 {
        0
    } else {
        u32::MAX << (32 - u32::from(prefix))
    };
    (u32::from(ip) & mask) == (u32::from(network) & mask)
}

fn hook_prompt_segment(text: &str) -> PromptSegment {
    PromptSegment {
        id: PromptSegmentId::new(),
        text: text.to_owned(),
        volatility: Volatility::SessionStable,
        cache_scope: CacheScope::PrefixCacheable,
        content_hash: hash_text(text),
    }
}

fn hash_text(text: &str) -> String {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}
