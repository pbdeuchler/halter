// pattern: Functional Core

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Context;
use chrono::Utc;
use halter_protocol::{HookHandlerType, HookRunStatus, HookRunSummary, PluginId};
use serde_json::Value;

use crate::config::{HookEventName, HookHandlerConfig as FileHookHandlerConfig, HooksFile};
use crate::matcher::CompiledMatcher;
use crate::merge::{HandlerPriority, HandlerPriorityGroup, HookMergedOutcome};
use crate::sdk::{HookCallback, HookKind, RegisteredHook, RegisteredHookPriority};

pub const HOOK_PROTOCOL_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct HookRegistrySource {
    pub plugin_id: PluginId,
    pub plugin_root: PathBuf,
    pub source_path: PathBuf,
    pub allowed_http_hosts: Vec<String>,
    pub allowed_env_vars: Vec<String>,
    pub file: HooksFile,
}

#[derive(Debug, Clone, Default)]
pub struct Hooks {
    handlers_by_event: BTreeMap<HookEventName, Vec<ConfiguredHandler>>,
}

impl Hooks {
    #[must_use]
    pub fn from_sources(sources: impl IntoIterator<Item = HookRegistrySource>) -> Self {
        let mut handlers_by_event = BTreeMap::new();

        for (plugin_index, source) in sources.into_iter().enumerate() {
            for (event_index, (event_name, matcher_groups)) in source.file.hooks.iter().enumerate()
            {
                for (matcher_index, matcher_group) in matcher_groups.iter().enumerate() {
                    for (hook_index, hook) in matcher_group.hooks.iter().enumerate() {
                        let matcher = matcher_group.matcher.clone();
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
                                allowed_http_hosts: source.allowed_http_hosts.clone(),
                                allowed_env_vars: source.allowed_env_vars.clone(),
                                event_name: *event_name,
                                handler_type: hook.handler_type,
                                timeout: hook.timeout,
                                status_message: hook.status_message.clone(),
                                if_condition: hook.if_condition.clone(),
                                once: hook.once,
                                matcher,
                                config: ConfiguredHandlerConfig::File(hook.config.clone()),
                                priority: HandlerPriority {
                                    group: HandlerPriorityGroup::PluginFiles,
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

    pub fn from_registered(
        hooks: impl IntoIterator<Item = RegisteredHook>,
    ) -> anyhow::Result<Self> {
        let mut handlers_by_event = BTreeMap::new();

        for (hook_index, registered) in hooks.into_iter().enumerate() {
            let matcher = registered
                .hook
                .matcher
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(|pattern| {
                    if registered.hook.event.matcher_field().is_none() {
                        anyhow::bail!(
                            "hook event '{}' does not support matcher",
                            registered.hook.event.canonical_name()
                        );
                    }
                    Ok(CompiledMatcher::compile_regex(pattern)?)
                })
                .transpose()
                .with_context(|| {
                    format!(
                        "failed to compile sdk hook matcher for plugin '{}' event '{}'",
                        registered.plugin_id,
                        registered.hook.event.canonical_name()
                    )
                })?;
            let priority_group = match registered.priority {
                RegisteredHookPriority::BeforePlugins => HandlerPriorityGroup::SdkBeforePlugins,
                RegisteredHookPriority::AfterPlugins => HandlerPriorityGroup::SdkAfterPlugins,
            };
            let handler_type = registered.hook.kind.handler_type();
            let config = match registered.hook.kind {
                HookKind::Callback(callback) => ConfiguredHandlerConfig::Callback(callback),
                HookKind::Function(factory) => ConfiguredHandlerConfig::Function(factory()),
            };
            handlers_by_event
                .entry(registered.hook.event)
                .or_insert_with(Vec::new)
                .push(ConfiguredHandler {
                    handler_id: format!(
                        "{}:{}:sdk:{}",
                        registered.plugin_id,
                        registered.hook.event.canonical_name(),
                        hook_index
                    ),
                    plugin_id: registered.plugin_id.clone(),
                    plugin_root: registered.plugin_root.clone(),
                    source_path: PathBuf::from(format!(
                        "<sdk-hook:{}:{}>",
                        registered.plugin_id, hook_index
                    )),
                    allowed_http_hosts: Vec::new(),
                    allowed_env_vars: Vec::new(),
                    event_name: registered.hook.event,
                    handler_type,
                    timeout: registered.hook.timeout,
                    status_message: registered.hook.status_message.clone(),
                    if_condition: registered.hook.if_condition.clone(),
                    once: registered.hook.once,
                    matcher,
                    config,
                    priority: HandlerPriority {
                        group: priority_group,
                        plugin_load_order: hook_index,
                        event_declaration_index: 0,
                        matcher_group_index: 0,
                        hook_index_within_group: 0,
                    },
                });
        }

        Ok(Self { handlers_by_event })
    }

    #[must_use]
    pub fn prepare(&self, request: HookDispatchRequest) -> PreparedHookDispatch {
        Self::prepare_many([self], request)
    }

    #[must_use]
    pub fn prepare_many<'a>(
        hook_sets: impl IntoIterator<Item = &'a Hooks>,
        request: HookDispatchRequest,
    ) -> PreparedHookDispatch {
        let mut matched_handlers = Vec::new();

        for hooks in hook_sets {
            for handler in hooks
                .handlers_by_event
                .get(&request.event_name)
                .into_iter()
                .flatten()
            {
                if handler.once && request.fired_hook_ids.contains(&handler.handler_id) {
                    continue;
                }
                if !handler.matches(&request) {
                    continue;
                }

                matched_handlers.push(handler.clone());
            }
        }

        matched_handlers.sort_by(|left, right| left.priority.cmp(&right.priority));
        let previews = matched_handlers.iter().map(build_preview_run).collect();

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
    pub fn request(&self) -> &HookDispatchRequest {
        &self.request
    }

    #[must_use]
    pub fn preview_runs(&self) -> &[HookRunSummary] {
        &self.previews
    }

    #[must_use]
    pub fn matched_handlers(&self) -> &[ConfiguredHandler] {
        &self.matched_handlers
    }
}

#[derive(Debug, Clone)]
pub struct HookDispatchOutcome {
    pub merged: HookMergedOutcome,
    pub runs: Vec<HookRunSummary>,
    pub fired_hook_ids: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct ConfiguredHandler {
    pub handler_id: String,
    pub plugin_id: PluginId,
    pub plugin_root: PathBuf,
    pub source_path: PathBuf,
    pub allowed_http_hosts: Vec<String>,
    pub allowed_env_vars: Vec<String>,
    pub event_name: HookEventName,
    pub handler_type: HookHandlerType,
    pub timeout: Duration,
    pub status_message: Option<String>,
    pub if_condition: Option<String>,
    pub once: bool,
    pub matcher: Option<CompiledMatcher>,
    pub config: ConfiguredHandlerConfig,
    pub priority: HandlerPriority,
}

impl ConfiguredHandler {
    /// Single-pass match: the regex matcher must hit (or be absent) and the
    /// `if` expression must evaluate true (or be absent). Collapsed from two
    /// methods to one chained expression (finding L19).
    fn matches(&self, request: &HookDispatchRequest) -> bool {
        let matcher_hit = match (&self.matcher, self.event_name.matcher_field()) {
            (Some(matcher), Some(_)) => request
                .matcher_value
                .as_deref()
                .is_some_and(|value| matcher.is_match(value)),
            (Some(_), None) => false,
            (None, _) => true,
        };
        matcher_hit
            && self
                .if_condition
                .as_deref()
                .is_none_or(|condition| matches_if_condition(condition, request))
    }
}

#[derive(Clone)]
pub enum ConfiguredHandlerConfig {
    File(FileHookHandlerConfig),
    Callback(HookCallback),
    Function(HookCallback),
}

impl fmt::Debug for ConfiguredHandlerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::File(config) => f.debug_tuple("File").field(config).finish(),
            Self::Callback(_) => f.write_str("Callback(..)"),
            Self::Function(_) => f.write_str("Function(..)"),
        }
    }
}

fn build_preview_run(handler: &ConfiguredHandler) -> HookRunSummary {
    let started_at = Utc::now();
    HookRunSummary {
        run_id: format!(
            "{}:{}",
            handler.handler_id,
            started_at.timestamp_nanos_opt().unwrap_or_default()
        ),
        event_name: handler.event_name.canonical_name().to_owned(),
        handler_type: handler.handler_type,
        plugin_id: handler.plugin_id.clone(),
        plugin_root: handler.plugin_root.clone(),
        status: HookRunStatus::Running,
        status_message: handler.status_message.clone(),
        started_at,
        completed_at: None,
        duration_ms: None,
        entries: Vec::new(),
    }
}

fn matches_if_condition(condition: &str, request: &HookDispatchRequest) -> bool {
    let trimmed = condition.trim();
    if trimmed.is_empty() || trimmed == "*" {
        return true;
    }

    let Some(tool_name) = request.payload.get("tool_name").and_then(Value::as_str) else {
        return false;
    };

    if let Some((tool_pattern, input_pattern)) = parse_if_condition(trimmed) {
        if !matches_text_pattern(tool_pattern, tool_name) {
            return false;
        }

        let input_text = request
            .payload
            .get("tool_input")
            .and_then(render_if_input_text)
            .unwrap_or_default();
        return matches_text_pattern(input_pattern, &input_text);
    }

    matches_text_pattern(trimmed, tool_name)
}

fn parse_if_condition(condition: &str) -> Option<(&str, &str)> {
    let open = condition.find('(')?;
    if !condition.ends_with(')') {
        return None;
    }
    let close = condition.len().saturating_sub(1);
    if close <= open {
        return None;
    }
    Some((condition[..open].trim(), condition[open + 1..close].trim()))
}

fn render_if_input_text(value: &Value) -> Option<String> {
    match value {
        Value::Object(map) => map
            .get("command")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .or_else(|| Some(Value::Object(map.clone()).to_string())),
        Value::String(text) => Some(text.clone()),
        Value::Null => None,
        other => Some(other.to_string()),
    }
}

fn matches_text_pattern(pattern: &str, candidate: &str) -> bool {
    let pattern = pattern.trim();
    if pattern.is_empty() || pattern == "*" {
        return true;
    }

    // Runtime match for `if_condition` patterns. These aren't validated at
    // config-load time, so an invalid pattern fails closed (no match).
    match CompiledMatcher::compile(pattern) {
        Ok(matcher) => matcher.is_match(candidate),
        Err(_) => false,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;
    use crate::config::{HookHandler, HookMatcherGroup, HooksFile, PromptHookConfig};

    #[test]
    fn wildcard_match_supports_globs() {
        assert!(matches_text_pattern("git *", "git status"));
        assert!(matches_text_pattern("shell", "Shell"));
        assert!(!matches_text_pattern("git *", "cargo test"));
    }

    /// AC3.5: an invalid matcher cannot reach the engine. `HooksFile::from_raw`
    /// rejects the config at load, so `Hooks::from_sources` never sees a raw
    /// string matcher. Defense-in-depth via the type system (H22/H27).
    #[test]
    fn review_hook_runtime_ac3_5_engine_never_sees_invalid_matcher() {
        let error = HooksFile::from_json_bytes(
            br#"{
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "(",
                            "hooks": [
                                {
                                    "type": "prompt",
                                    "prompt": "never reached"
                                }
                            ]
                        }
                    ]
                }
            }"#,
        )
        .expect_err("invalid matcher must hard-fail at load");
        let rendered = format!("{error:#}");
        assert!(
            rendered.contains("invalid matcher regex")
                || rendered.contains("invalid regex pattern"),
            "expected compile error, got: {rendered}",
        );
    }

    #[test]
    fn if_condition_matches_tool_name_and_command() {
        let handler = ConfiguredHandler {
            handler_id: "hook".to_owned(),
            plugin_id: PluginId::from("plugin"),
            plugin_root: PathBuf::from("/tmp/plugin"),
            source_path: PathBuf::from("/tmp/plugin/hooks.json"),
            allowed_http_hosts: Vec::new(),
            allowed_env_vars: Vec::new(),
            event_name: HookEventName::PreToolUse,
            handler_type: HookHandlerType::Prompt,
            timeout: Duration::from_secs(1),
            status_message: None,
            if_condition: Some("Shell(git *)".to_owned()),
            once: false,
            matcher: None,
            config: ConfiguredHandlerConfig::File(FileHookHandlerConfig::Prompt(
                PromptHookConfig {
                    prompt: "noop".to_owned(),
                    model: None,
                },
            )),
            priority: HandlerPriority {
                group: HandlerPriorityGroup::PluginFiles,
                plugin_load_order: 0,
                event_declaration_index: 0,
                matcher_group_index: 0,
                hook_index_within_group: 0,
            },
        };

        let request = HookDispatchRequest {
            event_name: HookEventName::PreToolUse,
            matcher_value: Some("Shell".to_owned()),
            payload: json!({
                "tool_name": "Shell",
                "tool_input": { "command": "git status" },
            }),
            fired_hook_ids: BTreeSet::new(),
        };

        assert!(handler.matches(&request));
    }

    #[test]
    fn if_condition_matches_regex_patterns_and_string_inputs() {
        let request = HookDispatchRequest {
            event_name: HookEventName::PreToolUse,
            matcher_value: Some("Read".to_owned()),
            payload: json!({
                "tool_name": "Read",
                "tool_input": "src/lib.rs",
            }),
            fired_hook_ids: BTreeSet::new(),
        };

        assert!(matches_if_condition("^Read$", &request));
        assert!(matches_if_condition("Read(^src/.*\\.rs$)", &request));
        assert!(!matches_if_condition("Write(src/.*)", &request));
    }

    #[test]
    fn if_condition_rejects_non_tool_payloads_and_unbalanced_groups() {
        let request = HookDispatchRequest {
            event_name: HookEventName::Notification,
            matcher_value: None,
            payload: json!({
                "message": "hello"
            }),
            fired_hook_ids: BTreeSet::new(),
        };

        assert!(!matches_if_condition("Shell(git *)", &request));
        assert!(!matches_if_condition("Shell(", &request));
    }

    #[test]
    fn if_condition_rejects_trailing_text_after_group() {
        let request = HookDispatchRequest {
            event_name: HookEventName::PreToolUse,
            matcher_value: Some("Shell".to_owned()),
            payload: json!({
                "tool_name": "Shell",
                "tool_input": { "command": "git status" },
            }),
            fired_hook_ids: BTreeSet::new(),
        };

        assert!(!matches_if_condition("Shell(git *) trailing", &request));
    }

    #[test]
    fn prepare_filters_once_handlers() {
        let hooks = Hooks::from_sources([HookRegistrySource {
            plugin_id: PluginId::from("plugin"),
            plugin_root: PathBuf::from("/tmp/plugin"),
            source_path: PathBuf::from("/tmp/plugin/hooks.json"),
            allowed_http_hosts: Vec::new(),
            allowed_env_vars: Vec::new(),
            file: HooksFile {
                hooks: [(
                    HookEventName::UserPromptSubmit,
                    vec![HookMatcherGroup {
                        matcher: None,
                        hooks: vec![HookHandler {
                            handler_type: HookHandlerType::Prompt,
                            timeout: Duration::from_secs(1),
                            status_message: None,
                            if_condition: None,
                            once: true,
                            config: FileHookHandlerConfig::Prompt(PromptHookConfig {
                                prompt: "noop".to_owned(),
                                model: None,
                            }),
                        }],
                    }],
                )]
                .into_iter()
                .collect(),
            },
        }]);

        let prepared = hooks.prepare(HookDispatchRequest {
            event_name: HookEventName::UserPromptSubmit,
            matcher_value: None,
            payload: json!({}),
            fired_hook_ids: ["plugin:UserPromptSubmit:0:0:0".to_owned()]
                .into_iter()
                .collect(),
        });

        assert!(prepared.matched_handlers().is_empty());
    }
}
