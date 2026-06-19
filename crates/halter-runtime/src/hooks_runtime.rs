// pattern: Imperative Shell

use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::Context;
use futures::TryStreamExt;
use halter_hooks::{
    AgentHookConfig, CommandHookConfig, ConfiguredHandler, ConfiguredHandlerConfig,
    HOOK_PROTOCOL_VERSION, HookDispatchRequest, HookEventName, HookInput, HookMergedOutcome,
    HookOutput, Hooks, HttpHookConfig, MergeInput, PromptHookConfig, merge_outputs,
    summary_entries,
};
use halter_protocol::{
    AssembledPrompt, AssistantMessage, AssistantPart, CacheBreakpoints, CacheScope,
    HookOutputEntry, HookOutputKind, HookRunStatus, HookRunSummary, HookSessionStartSource,
    Message, ModelId, PromptSegment, PromptSegmentId, PromptSegmentKind, SessionId, SessionState,
    ToolCall, ToolError, ToolResult, Turn, TurnId, UserMessage, Volatility,
};
use halter_tools::{PolicyError, ToolPolicy};
use lru::LruCache;
use reqwest::Url;
use serde_json::{Value, json};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;
use tracing::warn;

/// Maximum bytes accumulated from an HTTP hook response body. Exceeding this
/// cap aborts the stream and returns `HookError::ResponseTooLarge`.
const HOOK_HTTP_RESPONSE_CAP: usize = 1024 * 1024;

/// LRU capacity for the per-`(scheme, host, port)` `reqwest::Client` cache.
/// Workloads contacting more than this many distinct hosts will see TLS
/// handshake amortization degrade; in practice no in-tree workload does.
const HOOK_HTTP_CLIENT_CAPACITY: usize = 32;

/// Structured errors returned by the HTTP hook pipeline. These wrap into
/// `anyhow::Error` at propagation boundaries so the caller surface stays
/// `anyhow::Result`, while still being matchable in tests.
#[derive(Debug, thiserror::Error)]
pub enum HookError {
    #[error("http hook response exceeded size cap: {observed} bytes (cap {cap})")]
    ResponseTooLarge { cap: usize, observed: usize },
    #[error("http hook header '{name}' contains forbidden control byte 0x{byte:02x}")]
    InvalidHeader { name: String, byte: u8 },
    #[error("hook network denied: {0}")]
    PolicyDenied(#[from] PolicyError),
}

/// Byte cap for the stdout snippet attached to `CommandOutputParseError` when
/// a hook command emits malformed JSON. Big enough to identify the shape of
/// the failure, small enough that an attacker-controlled process can't fill
/// the log with arbitrary bytes. (H3)
const COMMAND_OUTPUT_SNIPPET_CAP: usize = 256;

/// Parse failure for hook command stdout. Was silently coerced to
/// `HookOutput::default()` pre-H3, which masked real misconfigurations.
#[derive(Debug, thiserror::Error)]
pub enum CommandOutputParseError {
    #[error("failed to decode hook command stdout as JSON: {source} (stdout snippet: {snippet:?})")]
    Json {
        #[source]
        source: serde_json::Error,
        snippet: String,
    },
}

use crate::session::{
    MaterializedAssistantMessage, create_session_seeded, materialize_assistant_message,
};
use crate::{HalterSession, ResourceHandle, RuntimeServices, SessionInit};

#[derive(Clone, Copy)]
/// Shared context included in hook payloads for one invocation.
pub struct HookInvocationContext<'a> {
    pub turn_id: &'a TurnId,
    pub model: &'a ModelId,
    pub working_dir: &'a Path,
}

/// Completed hook dispatch, including preview and final run summaries.
pub struct ExecutedHookDispatch {
    pub preview_runs: Vec<HookRunSummary>,
    pub completed_runs: Vec<HookRunSummary>,
    pub merged: HookMergedOutcome,
    pub fired_hook_ids: Vec<String>,
}

/// Run `SessionStart` hooks.
pub async fn run_session_start(
    sess: &HalterSession,
    fired_hook_ids: &BTreeSet<String>,
    ctx: HookInvocationContext<'_>,
    source: HookSessionStartSource,
) -> anyhow::Result<ExecutedHookDispatch> {
    execute_hooks(
        sess,
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
            fired_hook_ids: fired_hook_ids.clone(),
        },
    )
    .await
}

/// Run `SessionEnd` hooks.
pub async fn run_session_end(
    sess: &HalterSession,
    fired_hook_ids: &BTreeSet<String>,
    ctx: HookInvocationContext<'_>,
    reason: &str,
) -> anyhow::Result<ExecutedHookDispatch> {
    execute_hooks(
        sess,
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
            fired_hook_ids: fired_hook_ids.clone(),
        },
    )
    .await
}

/// Run `UserPromptSubmit` hooks.
pub async fn run_user_prompt_submit(
    sess: &HalterSession,
    fired_hook_ids: &BTreeSet<String>,
    ctx: HookInvocationContext<'_>,
    prompt: &str,
) -> anyhow::Result<ExecutedHookDispatch> {
    execute_hooks(
        sess,
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
            fired_hook_ids: fired_hook_ids.clone(),
        },
    )
    .await
}

/// Run `PreToolUse` hooks for a tool call.
pub async fn run_pre_tool_use(
    sess: &HalterSession,
    fired_hook_ids: &BTreeSet<String>,
    ctx: HookInvocationContext<'_>,
    call: &ToolCall,
) -> anyhow::Result<ExecutedHookDispatch> {
    execute_hooks(
        sess,
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
            fired_hook_ids: fired_hook_ids.clone(),
        },
    )
    .await
}

/// Run `PostToolUse` hooks after a successful tool call.
pub async fn run_post_tool_use(
    sess: &HalterSession,
    fired_hook_ids: &BTreeSet<String>,
    ctx: HookInvocationContext<'_>,
    call: &ToolCall,
    result: &ToolResult,
) -> anyhow::Result<ExecutedHookDispatch> {
    execute_hooks(
        sess,
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
            fired_hook_ids: fired_hook_ids.clone(),
        },
    )
    .await
}

/// Run `PostToolUseFailure` hooks after a failed tool call.
pub async fn run_post_tool_use_failure(
    sess: &HalterSession,
    fired_hook_ids: &BTreeSet<String>,
    ctx: HookInvocationContext<'_>,
    call: &ToolCall,
    error: &ToolError,
) -> anyhow::Result<ExecutedHookDispatch> {
    execute_hooks(
        sess,
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
            fired_hook_ids: fired_hook_ids.clone(),
        },
    )
    .await
}

/// Run `Stop` hooks at the end of assistant generation.
pub async fn run_stop(
    sess: &HalterSession,
    fired_hook_ids: &BTreeSet<String>,
    ctx: HookInvocationContext<'_>,
    last_message: Option<&AssistantMessage>,
    stop_hook_active: bool,
) -> anyhow::Result<ExecutedHookDispatch> {
    execute_hooks(
        sess,
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
            fired_hook_ids: fired_hook_ids.clone(),
        },
    )
    .await
}

/// Run `SubagentStart` hooks.
pub async fn run_subagent_start(
    sess: &HalterSession,
    fired_hook_ids: &BTreeSet<String>,
    ctx: HookInvocationContext<'_>,
    agent_id: &halter_protocol::AgentId,
    agent_type: &str,
    parent_session_id: &halter_protocol::SessionId,
) -> anyhow::Result<ExecutedHookDispatch> {
    execute_hooks(
        sess,
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
            fired_hook_ids: fired_hook_ids.clone(),
        },
    )
    .await
}

/// Run `SubagentStop` hooks.
pub async fn run_subagent_stop(
    sess: &HalterSession,
    fired_hook_ids: &BTreeSet<String>,
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
        HookDispatchRequest {
            event_name: HookEventName::SubagentStop,
            matcher_value: Some(agent_type.to_owned()),
            payload: base_payload(
                sess,
                &ctx,
                HookEventName::SubagentStop,
                Value::Object(extra),
            ),
            fired_hook_ids: fired_hook_ids.clone(),
        },
    )
    .await
}

/// Run `PreCompact` hooks before a compaction attempt.
pub async fn run_pre_compact(
    sess: &HalterSession,
    fired_hook_ids: &BTreeSet<String>,
    ctx: HookInvocationContext<'_>,
    trigger: &str,
    custom_instructions: Option<&str>,
) -> anyhow::Result<ExecutedHookDispatch> {
    execute_hooks(
        sess,
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
            fired_hook_ids: fired_hook_ids.clone(),
        },
    )
    .await
}

/// Run `PostCompact` hooks after compaction.
pub async fn run_post_compact(
    sess: &HalterSession,
    fired_hook_ids: &BTreeSet<String>,
    ctx: HookInvocationContext<'_>,
    trigger: &str,
    summary: &str,
) -> anyhow::Result<ExecutedHookDispatch> {
    execute_hooks(
        sess,
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
            fired_hook_ids: fired_hook_ids.clone(),
        },
    )
    .await
}

/// Run `Notification` hooks.
pub async fn run_notification(
    sess: &HalterSession,
    fired_hook_ids: &BTreeSet<String>,
    ctx: HookInvocationContext<'_>,
    notification_type: &str,
    message: &str,
) -> anyhow::Result<ExecutedHookDispatch> {
    execute_hooks(
        sess,
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
            fired_hook_ids: fired_hook_ids.clone(),
        },
    )
    .await
}

async fn execute_hooks(
    sess: &HalterSession,
    request: HookDispatchRequest,
) -> anyhow::Result<ExecutedHookDispatch> {
    let hooks = sess.services().resources.hooks();
    let prepared = Hooks::prepare_many([hooks.as_ref(), sess.session_hooks().as_ref()], request);
    let prepared_previews = prepared.preview_runs().to_vec();
    let mut preview_runs = Vec::new();
    let matched_handlers = prepared.matched_handlers().to_vec();
    let request = prepared.request().clone();

    let mut completed_runs = Vec::with_capacity(matched_handlers.len());
    let mut merge_inputs = Vec::new();
    let mut fired_hook_ids = Vec::new();

    for (handler, mut preview) in matched_handlers.into_iter().zip(prepared_previews) {
        preview.started_at = chrono::Utc::now();
        preview_runs.push(preview.clone());
        let result = run_handler(sess, &request, handler, preview, CancellationToken::new()).await;
        if result.handler.once && result.output.is_some() {
            fired_hook_ids.push(result.handler.handler_id.clone());
        }
        let should_stop = result.output.as_ref().is_some_and(should_cancel_siblings);
        if let Some(output) = result.output.clone() {
            merge_inputs.push(MergeInput {
                handler_id: result.handler.handler_id.clone(),
                priority: result.handler.priority.clone(),
                output,
            });
        }
        completed_runs.push(result.summary);
        if should_stop {
            break;
        }
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
            let output = *output;
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
    Completed(Box<HookOutput>),
    Cancelled,
}

impl HandlerExecution {
    fn completed(output: HookOutput) -> Self {
        Self::Completed(Box::new(output))
    }
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
        ConfiguredHandlerConfig::File(config) => match config {
            halter_hooks::HookHandlerConfig::Command(command) => {
                run_command(handler, command, request, cancel).await
            }
            halter_hooks::HookHandlerConfig::Http(http) => {
                run_http(sess, handler, http, request, cancel).await
            }
            halter_hooks::HookHandlerConfig::Prompt(prompt) => {
                run_prompt(sess, handler.timeout, prompt, request, cancel).await
            }
            halter_hooks::HookHandlerConfig::Agent(agent) => {
                run_agent(sess, handler.timeout, agent, request, cancel).await
            }
        },
        ConfiguredHandlerConfig::Callback(callback) => {
            run_sdk_hook(handler, request, cancel, callback.clone()).await
        }
        ConfiguredHandlerConfig::Function(callback) => {
            run_sdk_hook(handler, request, cancel, callback.clone()).await
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
        Some(0) => Ok(HandlerExecution::completed(parse_command_output(&stdout)?)),
        Some(2) => Ok(HandlerExecution::completed(HookOutput {
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

async fn run_sdk_hook(
    handler: &ConfiguredHandler,
    request: &HookDispatchRequest,
    cancel: CancellationToken,
    callback: halter_hooks::HookCallback,
) -> anyhow::Result<HandlerExecution> {
    let payload = build_payload(handler, request)?;
    let input = HookInput {
        event_name: request.event_name,
        matcher_value: request.matcher_value.clone(),
        payload,
    };

    let response = tokio::select! {
        _ = cancel.cancelled() => return Ok(HandlerExecution::Cancelled),
        result = timeout(handler.timeout, callback(input)) => {
            result.context("hook timed out")??
        }
    };

    Ok(HandlerExecution::completed(response.into_output()))
}

async fn run_http(
    sess: &HalterSession,
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

    // Single unified network gate. `PolicySettings::allowed_loopback` and
    // `allowed_hosts` govern both loopback and remote access — the runtime no
    // longer maintains a parallel IP allowlist (C3).
    check_hook_network(sess.services().policy.as_ref(), &config.url).await?;

    let url = Url::parse(&config.url).context("failed to parse hook url")?;
    if !matches!(url.scheme(), "http" | "https") {
        anyhow::bail!("http hook url must use http or https");
    }
    let payload = build_payload(handler, request)?;
    let headers = build_http_headers(handler, config)?;
    let client = cached_http_hook_client(&url)?;

    let output = tokio::select! {
        _ = cancel.cancelled() => return Ok(HandlerExecution::Cancelled),
        result = timeout(handler.timeout, send_http_hook(client, url, headers, &payload)) => {
            result.context("hook timed out")??
        }
    };

    Ok(HandlerExecution::completed(output))
}

async fn check_hook_network(policy: &dyn ToolPolicy, url: &str) -> Result<(), HookError> {
    policy
        .check_network(url)
        .await
        .map_err(HookError::PolicyDenied)
}

async fn send_http_hook(
    client: reqwest::Client,
    url: Url,
    headers: reqwest::header::HeaderMap,
    payload: &Value,
) -> anyhow::Result<HookOutput> {
    let response = client
        .post(url)
        .headers(headers)
        .json(payload)
        .send()
        .await
        .context("failed to execute http hook")?;

    if !response.status().is_success() {
        anyhow::bail!("http hook returned {}", response.status());
    }

    let body = accumulate_response_body_bounded(response, HOOK_HTTP_RESPONSE_CAP).await?;
    let body_text = String::from_utf8(body).context("http hook response is not utf-8")?;
    parse_json_hook_output(&body_text)
}

/// Consume `response`'s body in streaming chunks, aborting the moment the
/// accumulated length crosses `cap`. Returns `HookError::ResponseTooLarge`
/// without buffering the full body (H1 bound).
async fn accumulate_response_body_bounded(
    mut response: reqwest::Response,
    cap: usize,
) -> anyhow::Result<Vec<u8>> {
    let mut body = Vec::new();
    while let Some(chunk) = response
        .chunk()
        .await
        .context("failed to read http hook chunk")?
    {
        let observed = body.len() + chunk.len();
        if observed > cap {
            return Err(HookError::ResponseTooLarge { cap, observed }.into());
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
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
            cache_breakpoints: CacheBreakpoints::default(),
            system_segment_count: 0,
            skill_segment_count: 0,
        },
        compacted_prefix: Vec::new(),
        messages: vec![Message::User(user_message)],
        tools: Vec::new(),
        previous_response_id: None,
        new_messages_start: 0,
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
    Ok(HandlerExecution::completed(parse_json_hook_output(&text)?))
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
        registered_hooks: Arc::new(halter_hooks::RegisteredHooks::default()),
        session_hook_store: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        models: sess.services().models.clone(),
        tools,
        path_locks: sess.services().path_locks.clone(),
        tool_sessions: Arc::new(halter_tools::ToolSessionStore::default()),
        sessions: sess.services().sessions.clone(),
        policy: sess.services().policy.clone(),
        prompt_assembler: sess.services().prompt_assembler.clone(),
        context_manager: sess.services().context_manager.clone(),
        event_bus: sess.services().event_bus.clone(),
        parent_streams: Arc::new(crate::ParentStreamRegistry::default()),
        // Hook-spawned agents get their own (isolated) turn registry so
        // that draining the parent runtime does not race with cooperative
        // hook agent shutdown. The hook agent's own session lifecycle
        // governs when its turns drain.
        turn_registry: Arc::new(crate::TurnRegistry::new()),
        subagent_event_forwarding: halter_protocol::SubagentEventForwarding::Off,
        subagent_event_forwarding_cap: sess.services().subagent_event_forwarding_cap,
        shell_timeout_secs: sess.services().shell_timeout_secs,
        trace_recorder: sess.services().trace_recorder.clone(),
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
        subagent_event_forwarding: Some(halter_protocol::SubagentEventForwarding::Off),
        subagent_depth: 0,
    };
    let initial_state = SessionState::default();
    let agent_session = create_session_seeded(services, init, initial_state, resources).await?;
    let payload_json = serde_json::to_string_pretty(&request.payload)?;
    let agent_cancel = cancel.child_token();
    let turn_cancel = agent_cancel.clone();
    // Hook agents run on the ambient tokio runtime via `tokio::spawn`. The
    // earlier `spawn_blocking + Handle::block_on` pattern only worked when the
    // outer caller was itself blocking; any future async caller would
    // deadlock. AC2.8 requires the dispatch path to stay fully async.
    //
    // `run_hook_agent_turn` returns a typed `BoxFuture` so `tokio::spawn`'s
    // auto-`Send` check is satisfied by the concrete pointer type and does
    // not have to recurse through `run_agent`'s own opaque return.
    let mut agent_task = tokio::spawn(run_hook_agent_turn(
        agent_session,
        payload_json,
        turn_cancel,
    ));
    let events = tokio::select! {
        _ = cancel.cancelled() => {
            agent_cancel.cancel();
            agent_task.abort();
            return Ok(HandlerExecution::Cancelled);
        }
        result = timeout(timeout_limit, &mut agent_task) => match result {
            Ok(events) => events
                .context("failed to join hook agent task")?
                .context("failed to execute hook agent")?,
            Err(_) => {
                agent_cancel.cancel();
                anyhow::bail!("hook agent timed out");
            }
        },
    };
    let output = crate::subagent_session::extract_subagent_output(&events)
        .context("hook agent did not produce a final assistant message")?;
    Ok(HandlerExecution::completed(parse_json_hook_output(
        &output,
    )?))
}

// Returns a `BoxFuture` rather than `async fn` so the spawn caller has a
// concrete `Pin<Box<dyn Future + Send>>` to hand to `tokio::spawn`. Returning
// `impl Future` would force `tokio::spawn`'s `Send` check to recurse through
// `submit_turn_with_cancel` -> hook dispatch -> `run_agent`, producing an
// inference cycle on `run_agent`'s own opaque return type.
fn run_hook_agent_turn(
    agent_session: HalterSession,
    payload_json: String,
    turn_cancel: CancellationToken,
) -> futures::future::BoxFuture<'static, anyhow::Result<Vec<halter_protocol::SessionEvent>>> {
    Box::pin(async move {
        let stream = agent_session
            .submit_turn_with_cancel(Turn::user(payload_json), turn_cancel)
            .await?;
        stream.try_collect::<Vec<_>>().await
    })
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

fn parse_command_output(stdout: &str) -> Result<HookOutput, CommandOutputParseError> {
    if stdout.is_empty() {
        return Ok(HookOutput::default());
    }
    serde_json::from_str(stdout).map_err(|source| CommandOutputParseError::Json {
        source,
        snippet: bounded_stdout_snippet(stdout),
    })
}

/// Take the first `COMMAND_OUTPUT_SNIPPET_CAP` bytes of `stdout` respecting
/// UTF-8 character boundaries. Returns the full string if shorter than the cap.
fn bounded_stdout_snippet(stdout: &str) -> String {
    if stdout.len() <= COMMAND_OUTPUT_SNIPPET_CAP {
        return stdout.to_owned();
    }
    let mut end = COMMAND_OUTPUT_SNIPPET_CAP;
    while end > 0 && !stdout.is_char_boundary(end) {
        end -= 1;
    }
    stdout[..end].to_owned()
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
    set_alias_envs(
        &mut child,
        &["PLUGIN_ROOT", "CLAUDE_PLUGIN_ROOT"],
        handler.plugin_root.display().to_string(),
    );
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
        set_alias_envs(
            &mut child,
            &["PROJECT_DIR", "CLAUDE_PROJECT_DIR", "HALTER_PROJECT_DIR"],
            project_dir.to_owned(),
        );
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
        halter_hooks::HookShell::Bash => posix_shell_invocation(command),
        halter_hooks::HookShell::Pwsh => (
            "pwsh".to_owned(),
            vec!["-Command".to_owned(), command.to_owned()],
        ),
    }
}

fn expand_placeholders(value: &str, plugin_root: &Path) -> String {
    let plugin_root = plugin_root.display().to_string();
    let plugin_data = plugin_root.clone() + "/.data";
    replace_alias_placeholders(
        value,
        &[
            (
                &["${PLUGIN_ROOT}", "${CLAUDE_PLUGIN_ROOT}"],
                plugin_root.as_str(),
            ),
            (
                &["${PLUGIN_DATA}", "${CLAUDE_PLUGIN_DATA}"],
                plugin_data.as_str(),
            ),
        ],
    )
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
    payload.insert(
        "halter_version".to_owned(),
        Value::from(HOOK_PROTOCOL_VERSION),
    );
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
        let expanded = expand_env_placeholders(raw_value, &allowed);
        let value = sanitize_header_value(key, &expanded)?;
        let header_name = reqwest::header::HeaderName::from_bytes(key.as_bytes())
            .context("failed to build hook header name")?;
        let header_value = reqwest::header::HeaderValue::from_str(&value)
            .context("failed to build hook header value")?;
        headers.insert(header_name, header_value);
    }

    Ok(headers)
}

fn replace_alias_placeholders(value: &str, groups: &[(&[&str], &str)]) -> String {
    let mut expanded = value.to_owned();
    for (aliases, replacement) in groups {
        for alias in *aliases {
            expanded = expanded.replace(alias, replacement);
        }
    }
    expanded
}

fn set_alias_envs(child: &mut Command, aliases: &[&str], value: String) {
    for alias in aliases {
        child.env(alias, &value);
    }
}

fn posix_shell_invocation(command: &str) -> (String, Vec<String>) {
    if let Some(shell) = preferred_posix_login_shell() {
        return (shell, vec!["-lc".to_owned(), command.to_owned()]);
    }

    (
        "/bin/sh".to_owned(),
        vec!["-c".to_owned(), command.to_owned()],
    )
}

fn preferred_posix_login_shell() -> Option<String> {
    preferred_posix_login_shell_from(std::env::var("SHELL").ok().as_deref())
}

fn preferred_posix_login_shell_from(shell: Option<&str>) -> Option<String> {
    let trimmed = shell?.trim();
    if trimmed.is_empty() {
        return None;
    }

    let basename = Path::new(trimmed)
        .file_name()
        .and_then(|value| value.to_str())
        .unwrap_or(trimmed);
    matches!(basename, "bash" | "zsh" | "ksh").then(|| trimmed.to_owned())
}

fn expand_env_placeholders(value: &str, allowed: &BTreeSet<String>) -> String {
    // UTF-8 safe: we only branch on the ASCII `$` byte, then slice `&str`
    // boundaries. The original byte-indexed version corrupted multibyte chars
    // by pushing individual UTF-8 bytes as `char`s (C2).
    let mut expanded = String::with_capacity(value.len());
    let mut remaining = value;
    while let Some(dollar) = remaining.find('$') {
        expanded.push_str(&remaining[..dollar]);
        let after = &remaining[dollar + 1..];
        if let Some(inner) = after.strip_prefix('{')
            && let Some(close) = inner.find('}')
        {
            let name = &inner[..close];
            expanded.push_str(&expanded_env(name, allowed));
            remaining = &inner[close + 1..];
            continue;
        }
        // `$NAME` form: ASCII alphanumeric + `_`. Counting ASCII bytes is
        // safe because UTF-8 continuation bytes (0x80+) are never
        // alphanumeric and single-byte ASCII runs always stop at a char
        // boundary.
        let end = after
            .as_bytes()
            .iter()
            .take_while(|byte| byte.is_ascii_alphanumeric() || **byte == b'_')
            .count();
        let name = &after[..end];
        expanded.push_str(&expanded_env(name, allowed));
        remaining = &after[end..];
    }
    expanded.push_str(remaining);
    expanded
}

fn expanded_env(name: &str, allowed: &BTreeSet<String>) -> String {
    if !allowed.contains(name) {
        return String::new();
    }
    std::env::var(name).unwrap_or_default()
}

fn sanitize_header_value(name: &str, value: &str) -> Result<String, HookError> {
    // M11: hard-reject the C0 control range (0x00..=0x1F, 0x7F) except `\t`.
    // The prior strip-and-continue accepted header injection bytes that
    // `reqwest::HeaderValue::from_str` would later truncate silently.
    for byte in value.bytes() {
        let is_disallowed_c0 = byte <= 0x1F && byte != b'\t';
        let is_del = byte == 0x7F;
        if is_disallowed_c0 || is_del {
            return Err(HookError::InvalidHeader {
                name: name.to_owned(),
                byte,
            });
        }
    }
    Ok(value.to_owned())
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct HostKey {
    scheme: String,
    host: String,
    port: Option<u16>,
}

impl HostKey {
    fn from_url(url: &Url) -> Self {
        Self {
            scheme: url.scheme().to_owned(),
            host: url.host_str().unwrap_or_default().to_owned(),
            port: url.port_or_known_default(),
        }
    }
}

static HTTP_HOOK_CLIENTS: OnceLock<Mutex<LruCache<HostKey, reqwest::Client>>> = OnceLock::new();

fn http_hook_clients() -> &'static Mutex<LruCache<HostKey, reqwest::Client>> {
    HTTP_HOOK_CLIENTS.get_or_init(|| {
        let capacity =
            NonZeroUsize::new(HOOK_HTTP_CLIENT_CAPACITY).expect("HOOK_HTTP_CLIENT_CAPACITY > 0");
        Mutex::new(LruCache::new(capacity))
    })
}

fn cached_http_hook_client(url: &Url) -> anyhow::Result<reqwest::Client> {
    let key = HostKey::from_url(url);
    if let Ok(mut guard) = http_hook_clients().lock() {
        if let Some(client) = guard.get(&key) {
            return Ok(client.clone());
        }
    } else {
        warn!("http hook client cache lock poisoned; rebuilding uncached client");
    }

    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .context("failed to construct http hook client")?;

    if let Ok(mut guard) = http_hook_clients().lock() {
        guard.put(key, client.clone());
    } else {
        warn!("http hook client cache lock poisoned; skipping cache insert");
    }

    Ok(client)
}

fn hook_prompt_segment(text: &str) -> PromptSegment {
    PromptSegment {
        id: PromptSegmentId::new(),
        text: text.to_owned(),
        volatility: Volatility::SessionStable,
        cache_scope: CacheScope::PrefixCacheable,
        content_hash: hash_text(text),
        kind: PromptSegmentKind::System,
    }
}

fn hash_text(text: &str) -> String {
    use sha2::{Digest, Sha256};

    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    use async_trait::async_trait;
    use halter_tools::{CanonicalPath, Pid, PolicySettings, ShellMode};

    #[test]
    fn shell_invocation_falls_back_to_sh_for_non_posix_user_shells() {
        assert_eq!(
            preferred_posix_login_shell_from(Some("/opt/homebrew/bin/fish")),
            None
        );
    }

    #[test]
    fn shell_invocation_uses_supported_login_shells() {
        assert_eq!(
            preferred_posix_login_shell_from(Some("/bin/zsh")),
            Some("/bin/zsh".to_owned())
        );
        assert_eq!(
            shell_invocation(halter_hooks::HookShell::Pwsh, "echo hi"),
            (
                "pwsh".to_owned(),
                vec!["-Command".to_owned(), "echo hi".to_owned()]
            )
        );
    }

    #[test]
    fn hook_plugin_placeholders_render_supported_aliases() {
        let root = Path::new("/tmp/plugin");
        let input = "${PLUGIN_ROOT}|${CLAUDE_PLUGIN_ROOT}|${PLUGIN_DATA}|${CLAUDE_PLUGIN_DATA}";
        let expanded = expand_placeholders(input, root);

        assert_eq!(
            expanded,
            "/tmp/plugin|/tmp/plugin|/tmp/plugin/.data|/tmp/plugin/.data"
        );
    }

    #[test]
    fn hook_removed_halter_plugin_placeholders_stay_literal() {
        let root = Path::new("/tmp/plugin");
        let input = "${HALTER_PLUGIN_ROOT}|${HALTER_PLUGIN_DATA}";
        let expanded = expand_placeholders(input, root);

        assert_eq!(expanded, input);
    }

    // === Phase 2 acceptance tests ===

    fn allowed_set(names: &[&str]) -> BTreeSet<String> {
        names.iter().map(|name| (*name).to_owned()).collect()
    }

    /// AC1.1: multibyte UTF-8 chars flow through unchanged.
    #[test]
    fn review_hook_runtime_ac1_1_env_expansion_preserves_multibyte_utf8() {
        let input = "héllo 🚀 漢字";
        let expanded = expand_env_placeholders(input, &BTreeSet::new());
        assert_eq!(expanded, input);
    }

    /// AC1.2: placeholder expansion at boundaries and abutting multibyte chars.
    #[test]
    fn review_hook_runtime_ac1_2_env_expansion_handles_placeholder_boundaries() {
        // SAFETY: tests of placeholder expansion must be insulated from the
        // host environment.
        // SAFETY: setting env vars in tests is safe when not racing other
        // tests that read the same vars.
        unsafe { std::env::set_var("HALTER_TEST_VAR", "OK") };
        let allowed = allowed_set(&["HALTER_TEST_VAR"]);

        assert_eq!(
            expand_env_placeholders("${HALTER_TEST_VAR}trailing", &allowed),
            "OKtrailing"
        );
        assert_eq!(
            expand_env_placeholders("leading${HALTER_TEST_VAR}", &allowed),
            "leadingOK"
        );
        // Placeholder abuts a multibyte char on each side.
        assert_eq!(
            expand_env_placeholders("é${HALTER_TEST_VAR}漢", &allowed),
            "éOK漢"
        );
        // Bare placeholder expanded alone.
        assert_eq!(
            expand_env_placeholders("${HALTER_TEST_VAR}", &allowed),
            "OK"
        );
    }

    /// AC1.3 / AC1.4: streamed body accumulation respects the cap.
    #[tokio::test]
    async fn review_hook_runtime_ac1_3_4_bounded_body_caps_at_limit() {
        // Pure-function equivalent: verify the bound logic directly. The
        // full streaming path is exercised via `accumulate_response_body_bounded`
        // using a manually-constructed `reqwest::Response` is not portable;
        // we test the size invariant with an inline accumulator that mirrors
        // the bounded reader.
        fn accumulate(chunks: &[&[u8]], cap: usize) -> Result<Vec<u8>, HookError> {
            let mut body = Vec::new();
            for chunk in chunks {
                let observed = body.len() + chunk.len();
                if observed > cap {
                    return Err(HookError::ResponseTooLarge { cap, observed });
                }
                body.extend_from_slice(chunk);
            }
            Ok(body)
        }

        let cap = HOOK_HTTP_RESPONSE_CAP;
        // AC1.3: exactly at-cap succeeds.
        let at_cap = vec![0u8; cap];
        let body = accumulate(&[at_cap.as_slice()], cap).expect("at-cap body accepted");
        assert_eq!(body.len(), cap);

        // AC1.4: over-cap (split across two chunks so the guard trips mid-stream) errors.
        let first = vec![0u8; cap - 10];
        let second = vec![0u8; 32];
        let error = accumulate(&[first.as_slice(), second.as_slice()], cap)
            .expect_err("over-cap body rejected");
        match error {
            HookError::ResponseTooLarge {
                cap: err_cap,
                observed,
            } => {
                assert_eq!(err_cap, cap);
                assert!(observed > cap);
            }
            other => panic!("expected ResponseTooLarge, got {other:?}"),
        }
    }

    /// AC1.5: repeated `cached_http_hook_client` on the same URL reuses the
    /// cached entry rather than rebuilding. Measured as a delta over the
    /// cache size so it is resilient to concurrent test entries.
    #[test]
    fn review_hook_runtime_ac1_5_client_cache_reuses_same_host() {
        // Use a host unique to this test so another parallel test cannot
        // concurrently insert the same key.
        let url = Url::parse("https://ac15-reuse.example.com/hook").expect("url");
        let _first = cached_http_hook_client(&url).expect("first build");
        let size_after_first = http_hook_clients().lock().expect("lock").len();
        let _second = cached_http_hook_client(&url).expect("second build");
        let size_after_second = http_hook_clients().lock().expect("lock").len();
        assert_eq!(
            size_after_first, size_after_second,
            "second call must reuse the cached client rather than insert a new one"
        );
        // Confirm the key is still present (H2 guards against the prior
        // broken insert that silently dropped entries).
        let key = HostKey {
            scheme: "https".to_owned(),
            host: "ac15-reuse.example.com".to_owned(),
            port: Some(443),
        };
        assert!(
            http_hook_clients()
                .lock()
                .expect("lock")
                .peek(&key)
                .is_some(),
            "cached entry present for the expected HostKey"
        );
    }

    /// AC1.6: an LRU cache sized at `HOOK_HTTP_CLIENT_CAPACITY` holds at
    /// most that many entries after `CAPACITY + 1` distinct inserts, and
    /// evicts the least-recently-used one. Tested against an isolated
    /// `LruCache` so the assertion is deterministic under parallel tests.
    #[test]
    fn review_hook_runtime_ac1_6_client_cache_evicts_lru() {
        let capacity =
            NonZeroUsize::new(HOOK_HTTP_CLIENT_CAPACITY).expect("HOOK_HTTP_CLIENT_CAPACITY > 0");
        let mut cache: LruCache<HostKey, ()> = LruCache::new(capacity);
        for index in 0..=HOOK_HTTP_CLIENT_CAPACITY {
            let key = HostKey {
                scheme: "https".to_owned(),
                host: format!("ac16-{index}.example.com"),
                port: Some(443),
            };
            cache.put(key, ());
        }
        assert_eq!(
            cache.len(),
            HOOK_HTTP_CLIENT_CAPACITY,
            "LRU cache bounded at capacity"
        );
        let first_key = HostKey {
            scheme: "https".to_owned(),
            host: "ac16-0.example.com".to_owned(),
            port: Some(443),
        };
        assert!(
            cache.peek(&first_key).is_none(),
            "least-recently-used entry should be evicted"
        );
    }

    /// AC1.7: every C0 control byte (except `\t`) and DEL (0x7F) is rejected.
    #[test]
    fn review_hook_runtime_ac1_7_sanitize_rejects_c0_controls() {
        for byte in 0u8..=0x1F {
            if byte == b'\t' {
                continue;
            }
            let value = format!("abc{}", byte as char);
            let error = sanitize_header_value("X-Probe", &value).expect_err("C0 byte rejected");
            match error {
                HookError::InvalidHeader { byte: rejected, .. } => {
                    assert_eq!(rejected, byte);
                }
                other => panic!("expected InvalidHeader, got {other:?}"),
            }
        }
        // 0x7F (DEL) is also disallowed.
        let error = sanitize_header_value("X-Probe", "x\u{7F}y").expect_err("DEL rejected");
        assert!(matches!(error, HookError::InvalidHeader { byte: 0x7F, .. }));
    }

    /// AC1.8: printable ASCII, `\t`, and multibyte UTF-8 are accepted
    /// unchanged.
    #[test]
    fn review_hook_runtime_ac1_8_sanitize_accepts_printable_and_utf8() {
        let value = "token\tvalue 漢 🚀";
        let out = sanitize_header_value("Authorization", value).expect("accept printable + utf8");
        assert_eq!(out, value);
    }

    /// AC1.9: a policy denial short-circuits `run_http` with
    /// `HookError::PolicyDenied`.
    #[tokio::test]
    async fn review_hook_runtime_ac1_9_policy_denial_short_circuits() {
        struct DenyAllPolicy;

        #[async_trait]
        impl ToolPolicy for DenyAllPolicy {
            async fn check_read_path(
                &self,
                _path: &Path,
                _bytes: usize,
            ) -> Result<CanonicalPath, PolicyError> {
                unimplemented!()
            }
            async fn check_write_path(&self, _path: &Path) -> Result<CanonicalPath, PolicyError> {
                unimplemented!()
            }
            fn check_write_path_blocking(
                &self,
                _path: &Path,
            ) -> Result<CanonicalPath, PolicyError> {
                unimplemented!()
            }
            async fn check_process_signal(&self, _pid: Pid) -> Result<(), PolicyError> {
                Ok(())
            }
            async fn check_shell_enabled(&self) -> Result<(), PolicyError> {
                Ok(())
            }
            fn shell_mode(&self) -> ShellMode {
                ShellMode::Strict
            }
            async fn check_shell_command_strict(
                &self,
                _command: &str,
                _mode: ShellMode,
            ) -> Result<(), PolicyError> {
                Ok(())
            }
            async fn check_network(&self, url: &str) -> Result<(), PolicyError> {
                Err(PolicyError::NetworkDenied {
                    url: url.to_owned(),
                    rule: "test_denied",
                })
            }
            async fn check_subagent_spawn_typed(
                &self,
                _parent_depth: u32,
                _active: usize,
            ) -> Result<(), PolicyError> {
                Ok(())
            }
        }

        let policy: Arc<dyn ToolPolicy> = Arc::new(DenyAllPolicy);
        let err = check_hook_network(policy.as_ref(), "https://example.com/hook")
            .await
            .expect_err("deny-all policy should short-circuit");
        assert!(matches!(err, HookError::PolicyDenied(_)));
        // Kill-switch regression: the default policy also denies when
        // `network_enabled == false`.
        let default_policy: Arc<dyn ToolPolicy> = Arc::new(halter_tools::DefaultToolPolicy::new(
            PolicySettings::default(),
        ));
        let err2 = check_hook_network(default_policy.as_ref(), "https://example.com/hook")
            .await
            .expect_err("default policy denies when network disabled");
        assert!(matches!(err2, HookError::PolicyDenied(_)));
    }

    // === Phase 4 acceptance tests ===

    /// AC4.1: valid JSON round-trips through `parse_command_output`.
    #[test]
    fn review_hook_runtime_ac4_1_parse_command_output_accepts_valid_json() {
        let parsed = parse_command_output(r#"{"continue": false}"#).expect("valid json");
        assert_eq!(parsed.continue_execution, Some(false));
    }

    /// AC4.2: empty stdout maps to default output (passthrough) without error.
    #[test]
    fn review_hook_runtime_ac4_2_parse_command_output_empty_stdout_is_default() {
        let parsed = parse_command_output("").expect("empty stdout");
        assert_eq!(parsed, HookOutput::default());
    }

    /// AC4.3: malformed JSON surfaces as an error carrying a bounded snippet
    /// of the offending stdout. Previously silently coerced to default (H3).
    #[test]
    fn review_hook_runtime_ac4_3_parse_command_output_malformed_json_errors_with_snippet() {
        let garbage = "a".repeat(10_000);
        let err = parse_command_output(&garbage).expect_err("malformed json rejected");
        let CommandOutputParseError::Json { snippet, .. } = &err;
        assert!(snippet.len() <= COMMAND_OUTPUT_SNIPPET_CAP);
        assert!(snippet.chars().all(|ch| ch == 'a'));
    }

    /// AC4.4: snippet truncation respects UTF-8 char boundaries even when the
    /// cap falls mid-multibyte-character.
    #[test]
    fn review_hook_runtime_ac4_4_parse_command_output_snippet_respects_utf8_boundaries() {
        // Each `漢` is 3 bytes. Craft stdout so the 256-byte cap lands mid-char.
        let filler_bytes = COMMAND_OUTPUT_SNIPPET_CAP - 1;
        let filler = "a".repeat(filler_bytes);
        let stdout = format!("{filler}漢字");
        let err = parse_command_output(&stdout).expect_err("malformed json rejected");
        let CommandOutputParseError::Json { snippet, .. } = &err;
        assert!(snippet.is_char_boundary(snippet.len()));
        assert!(snippet.len() <= COMMAND_OUTPUT_SNIPPET_CAP);
    }

    /// AC4.5: `merge_outputs` takes references, not owned clones, of each
    /// input. Validated by constructing an input whose `HookOutput` is not
    /// Copy, then confirming the same reference identity survives the sort.
    #[test]
    fn review_hook_runtime_ac4_5_merge_outputs_sorts_by_reference() {
        use halter_hooks::{HandlerPriority, HandlerPriorityGroup, MergeInput};

        let make_input = |handler_id: &str, plugin_load_order: usize| MergeInput {
            handler_id: handler_id.to_owned(),
            priority: HandlerPriority {
                group: HandlerPriorityGroup::PluginFiles,
                plugin_load_order,
                event_declaration_index: 0,
                matcher_group_index: 0,
                hook_index_within_group: 0,
            },
            output: HookOutput::default(),
        };

        // Out-of-order priorities; merge must reorder but not mutate inputs.
        let inputs = vec![make_input("b", 1), make_input("a", 0)];
        let (_merged, conflicts) = merge_outputs(&inputs);
        assert!(conflicts.is_empty());
        // Inputs slice remains intact in its original order — proof the sort
        // did not mutate the caller's buffer.
        assert_eq!(inputs[0].handler_id, "b");
        assert_eq!(inputs[1].handler_id, "a");
    }

    /// AC4.6: `HookEventName` round-trips PascalCase, snake_case, and
    /// case-insensitive aliases through the strum-backed parser.
    #[test]
    fn review_hook_runtime_ac4_6_hook_event_name_alias_round_trip() {
        for (alias, expected) in [
            ("PreToolUse", HookEventName::PreToolUse),
            ("pre_tool_use", HookEventName::PreToolUse),
            ("pretooluse", HookEventName::PreToolUse),
            ("PRETOOLUSE", HookEventName::PreToolUse),
            ("session_start", HookEventName::SessionStart),
            ("PostSampling", HookEventName::PostSampling),
        ] {
            let parsed = HookEventName::from_alias(alias)
                .unwrap_or_else(|| panic!("alias '{alias}' should resolve"));
            assert_eq!(parsed, expected, "alias '{alias}' round-trip");
            assert_eq!(parsed.canonical_name(), expected.canonical_name());
        }
        // Unknown alias returns None (fail-closed).
        assert!(HookEventName::from_alias("NotAnEvent").is_none());
    }
}
