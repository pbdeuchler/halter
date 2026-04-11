// pattern: Imperative Shell

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::Stdio;

use anyhow::Context;
use chrono::Utc;
use futures::future::join_all;
use halter_protocol::{
    HookHandlerType, HookOutputEntry, HookOutputKind, HookRunStatus, HookRunSummary, PluginId,
};
use regex::Regex;
use serde_json::Value;
use tokio::io::AsyncWriteExt;
use tokio::process::Command;
use tokio::time::timeout;
use tracing::warn;

use crate::config::{CommandHookConfig, HookEventName, HookHandlerConfig, HookShell, HooksFile};
use crate::merge::{
    HookDecision, HookMergedOutcome, HookOutput, HandlerPriority, MergeConflict, MergeInput,
    merge_outputs, summary_entries,
};

pub const HOOK_PROTOCOL_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct HookRegistrySource {
    pub plugin_id: PluginId,
    pub plugin_root: PathBuf,
    pub source_path: PathBuf,
    pub file: HooksFile,
}

#[derive(Debug, Clone, Default)]
pub struct Hooks {
    handlers_by_event: std::collections::BTreeMap<HookEventName, Vec<ConfiguredHandler>>,
}

impl Hooks {
    #[must_use]
    pub fn from_sources(sources: impl IntoIterator<Item = HookRegistrySource>) -> Self {
        let mut handlers_by_event = std::collections::BTreeMap::new();

        for (plugin_index, source) in sources.into_iter().enumerate() {
            for (event_index, (event_name, matcher_groups)) in source.file.hooks.iter().enumerate() {
                for (matcher_index, matcher_group) in matcher_groups.iter().enumerate() {
                    for (hook_index, hook) in matcher_group.hooks.iter().enumerate() {
                        let matcher = matcher_group
                            .matcher
                            .as_deref()
                            .map(Regex::new)
                            .transpose()
                            .ok()
                            .flatten();
                        let handler_id = format!(
                            "{}:{}:{}:{}:{}",
                            source.plugin_id,
                            event_name.canonical_name(),
                            event_index,
                            matcher_index,
                            hook_index
                        );
                        handlers_by_event
                            .entry(*event_name)
                            .or_insert_with(Vec::new)
                            .push(ConfiguredHandler {
                                handler_id,
                                plugin_id: source.plugin_id.clone(),
                                plugin_root: source.plugin_root.clone(),
                                source_path: source.source_path.clone(),
                                event_name: *event_name,
                                handler_type: hook.handler_type,
                                timeout: hook.timeout,
                                status_message: hook.status_message.clone(),
                                once: hook.once,
                                matcher,
                                config: hook.config.clone(),
                                priority: HandlerPriority {
                                    plugin_load_order: plugin_index,
                                    event_declaration_index: event_index,
                                    matcher_group_index: matcher_index,
                                    hook_index_within_group: hook_index,
                                },
                            });
                    }
                }
            }
        }

        Self { handlers_by_event }
    }

    #[must_use]
    pub fn prepare(&self, request: HookDispatchRequest) -> PreparedHookDispatch {
        let mut previews = Vec::new();
        let mut matched_handlers = Vec::new();

        for handler in self
            .handlers_by_event
            .get(&request.event_name)
            .into_iter()
            .flatten()
        {
            if handler.once && request.fired_hook_ids.contains(&handler.handler_id) {
                continue;
            }
            if !handler.matches(request.matcher_value.as_deref()) {
                continue;
            }

            previews.push(HookRunSummary {
                run_id: format!("{}:{}", handler.handler_id, Utc::now().timestamp_nanos_opt().unwrap_or_default()),
                event_name: request.event_name.canonical_name().to_owned(),
                handler_type: handler.handler_type,
                plugin_id: handler.plugin_id.clone(),
                plugin_root: handler.plugin_root.clone(),
                status: HookRunStatus::Running,
                status_message: handler.status_message.clone(),
                started_at: Utc::now(),
                completed_at: None,
                duration_ms: None,
                entries: Vec::new(),
            });
            matched_handlers.push(handler.clone());
        }

        PreparedHookDispatch {
            request,
            previews,
            matched_handlers,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HookDispatchRequest {
    pub event_name: HookEventName,
    pub matcher_value: Option<String>,
    pub payload: Value,
    pub fired_hook_ids: BTreeSet<String>,
}

#[derive(Debug, Clone)]
pub struct PreparedHookDispatch {
    request: HookDispatchRequest,
    previews: Vec<HookRunSummary>,
    matched_handlers: Vec<ConfiguredHandler>,
}

impl PreparedHookDispatch {
    #[must_use]
    pub fn preview_runs(&self) -> &[HookRunSummary] {
        &self.previews
    }

    pub async fn run(self) -> HookDispatchOutcome {
        let results = join_all(
            self.matched_handlers
                .iter()
                .zip(self.previews.iter())
                .map(|(handler, preview)| run_handler(handler, preview, &self.request)),
        )
        .await;

        let mut completed_runs = Vec::with_capacity(results.len());
        let mut merge_inputs = Vec::new();
        let mut fired_hook_ids = Vec::new();

        for (handler, result) in self.matched_handlers.iter().zip(results) {
            if handler.once {
                fired_hook_ids.push(handler.handler_id.clone());
            }
            completed_runs.push(result.summary);
            if let Some(output) = result.output {
                merge_inputs.push(MergeInput {
                    handler_id: handler.handler_id.clone(),
                    priority: handler.priority.clone(),
                    output,
                });
            }
        }

        let (merged, conflicts) = merge_outputs(&merge_inputs);
        log_conflicts(&conflicts);

        HookDispatchOutcome {
            merged,
            runs: completed_runs,
            fired_hook_ids,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HookDispatchOutcome {
    pub merged: HookMergedOutcome,
    pub runs: Vec<HookRunSummary>,
    pub fired_hook_ids: Vec<String>,
}

#[derive(Debug, Clone)]
struct ConfiguredHandler {
    handler_id: String,
    plugin_id: PluginId,
    plugin_root: PathBuf,
    source_path: PathBuf,
    event_name: HookEventName,
    handler_type: HookHandlerType,
    timeout: std::time::Duration,
    status_message: Option<String>,
    once: bool,
    matcher: Option<Regex>,
    config: HookHandlerConfig,
    priority: HandlerPriority,
}

impl ConfiguredHandler {
    fn matches(&self, candidate: Option<&str>) -> bool {
        match (&self.matcher, self.event_name.matcher_field()) {
            (Some(regex), Some(_)) => candidate.is_some_and(|value| regex.is_match(value)),
            (Some(_), None) => true,
            (None, _) => true,
        }
    }
}

struct HandlerRunResult {
    summary: HookRunSummary,
    output: Option<HookOutput>,
}

async fn run_handler(
    handler: &ConfiguredHandler,
    preview: &HookRunSummary,
    request: &HookDispatchRequest,
) -> HandlerRunResult {
    let started_at = preview.started_at;
    let execution = match &handler.config {
        HookHandlerConfig::Command(command) => run_command(handler, command, request).await,
    };

    match execution {
        Ok(output) => {
            let completed_at = Utc::now();
            let duration_ms = completed_at
                .signed_duration_since(started_at)
                .num_milliseconds()
                .max(0) as u64;
            let status = if matches!(output.continue_execution, Some(false)) {
                HookRunStatus::Stopped
            } else if matches!(output.decision, Some(HookDecision::Block))
                || output
                    .hook_specific_output
                    .as_ref()
                    .and_then(|hook| hook.permission_decision)
                    .is_some_and(|decision| {
                        matches!(
                            decision,
                            crate::merge::PermissionDecision::Deny
                                | crate::merge::PermissionDecision::Ask
                        )
                    })
            {
                HookRunStatus::Blocked
            } else {
                HookRunStatus::Completed
            };
            HandlerRunResult {
                summary: HookRunSummary {
                    status,
                    completed_at: Some(completed_at),
                    duration_ms: Some(duration_ms),
                    entries: summary_entries(&output),
                    ..preview.clone()
                },
                output: Some(output),
            }
        }
        Err(error) => {
            let completed_at = Utc::now();
            let duration_ms = completed_at
                .signed_duration_since(started_at)
                .num_milliseconds()
                .max(0) as u64;
            HandlerRunResult {
                summary: HookRunSummary {
                    status: HookRunStatus::Failed,
                    completed_at: Some(completed_at),
                    duration_ms: Some(duration_ms),
                    entries: vec![HookOutputEntry {
                        kind: HookOutputKind::Error,
                        text: error.to_string(),
                    }],
                    ..preview.clone()
                },
                output: None,
            }
        }
    }
}

async fn run_command(
    handler: &ConfiguredHandler,
    command: &CommandHookConfig,
    request: &HookDispatchRequest,
) -> anyhow::Result<HookOutput> {
    let expanded_command = expand_placeholders(&command.command, &handler.plugin_root);
    let mut child = build_command(handler, command.shell, &expanded_command, request)?;
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

    let output = timeout(handler.timeout, child.wait_with_output())
        .await
        .context("hook timed out")?
        .context("failed to wait for hook command")?;
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();

    match output.status.code() {
        Some(0) => {
            if stdout.is_empty() {
                return Ok(HookOutput::default());
            }
            serde_json::from_slice(&output.stdout)
                .or_else(|_| Ok::<HookOutput, serde_json::Error>(HookOutput::default()))
                .context("failed to decode hook stdout")
        }
        Some(2) => Ok(HookOutput {
            decision: Some(HookDecision::Block),
            reason: (!stderr.is_empty()).then_some(stderr),
            ..HookOutput::default()
        }),
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

fn build_command(
    handler: &ConfiguredHandler,
    shell: HookShell,
    expanded_command: &str,
    request: &HookDispatchRequest,
) -> anyhow::Result<tokio::process::Child> {
    let (program, args) = shell_invocation(shell, expanded_command);
    let mut command = Command::new(program);
    command
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
    command.current_dir(cwd);
    command.env("PLUGIN_ROOT", &handler.plugin_root);
    command.env("CLAUDE_PLUGIN_ROOT", &handler.plugin_root);
    command.env("PLUGIN_ID", handler.plugin_id.0.as_str());
    command.env("HOOK_EVENT", handler.event_name.canonical_name());
    command.env("HOOK_VERSION", HOOK_PROTOCOL_VERSION.to_string());
    command.env("HOOK_SOURCE_PATH", &handler.source_path);
    if let Some(session_id) = request.payload.get("session_id").and_then(Value::as_str) {
        command.env("SESSION_ID", session_id);
    }
    if let Some(turn_id) = request.payload.get("turn_id").and_then(Value::as_str) {
        command.env("TURN_ID", turn_id);
    }
    if let Some(project_dir) = request.payload.get("cwd").and_then(Value::as_str) {
        command.env("PROJECT_DIR", project_dir);
        command.env("CLAUDE_PROJECT_DIR", project_dir);
    }

    let HookHandlerConfig::Command(command_config) = &handler.config;
    for (key, value) in &command_config.env {
        command.env(key, expand_placeholders(value, &handler.plugin_root));
    }

    command
        .spawn()
        .context("failed to spawn hook command process")
}

fn build_payload(handler: &ConfiguredHandler, request: &HookDispatchRequest) -> anyhow::Result<Value> {
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

fn shell_invocation(shell: HookShell, command: &str) -> (String, Vec<String>) {
    match shell {
        HookShell::Bash => (
            std::env::var("SHELL")
                .ok()
                .filter(|value| !value.trim().is_empty())
                .unwrap_or_else(|| "/bin/sh".to_owned()),
            vec!["-lc".to_owned(), command.to_owned()],
        ),
        HookShell::Pwsh => (
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

fn log_conflicts(conflicts: &[MergeConflict]) {
    for conflict in conflicts {
        warn!(
            field = conflict.field,
            winner = %conflict.winner,
            loser = %conflict.loser,
            "hooks.merge_conflict"
        );
    }
}
