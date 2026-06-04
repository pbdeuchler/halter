// pattern: Functional Core

use std::time::Duration;

use anyhow::Context;
use halter_protocol::HookHandlerType;
use indexmap::{IndexMap, IndexSet};
use serde::Deserialize;
use strum_macros::{EnumString, IntoStaticStr};

use crate::matcher::CompiledMatcher;

#[derive(Debug, Clone, Default)]
/// Parsed `hooks.json` file.
pub struct HooksFile {
    pub hooks: IndexMap<HookEventName, Vec<HookMatcherGroup>>,
}

impl HooksFile {
    /// Parse and validate a hook file from JSON bytes.
    pub fn from_json_bytes(bytes: &[u8]) -> anyhow::Result<(Self, Vec<HooksLoadWarning>)> {
        let raw: HooksFileRaw =
            serde_json::from_slice(bytes).context("failed to parse hooks.json")?;
        Self::from_raw(raw)
    }

    fn from_raw(raw: HooksFileRaw) -> anyhow::Result<(Self, Vec<HooksLoadWarning>)> {
        let mut hooks = IndexMap::new();
        let mut warnings = Vec::new();
        let mut seen = IndexSet::new();

        for (event_alias, matcher_groups) in raw.hooks {
            let Some(event) = HookEventName::from_alias(&event_alias) else {
                warnings.push(HooksLoadWarning::new(
                    "unknown_event",
                    format!("unknown hook event '{event_alias}'"),
                ));
                continue;
            };
            if !seen.insert(event) {
                warnings.push(HooksLoadWarning::new(
                    "duplicate_alias",
                    format!(
                        "duplicate hook alias '{event_alias}' resolved to '{}'",
                        event.canonical_name()
                    ),
                ));
                continue;
            }

            let mut parsed_groups = Vec::new();
            for matcher_group in matcher_groups {
                let group = HookMatcherGroup::from_raw(event, matcher_group, &mut warnings)
                    .with_context(|| {
                        format!(
                            "failed to compile matcher for hook event '{}'",
                            event.canonical_name()
                        )
                    })?;
                if let Some(group) = group
                    && !group.hooks.is_empty()
                {
                    parsed_groups.push(group);
                }
            }

            if !parsed_groups.is_empty() {
                hooks.insert(event, parsed_groups);
            }
        }

        Ok((Self { hooks }, warnings))
    }
}

#[derive(Debug, Clone)]
/// Hooks that share one optional matcher for an event.
pub struct HookMatcherGroup {
    pub matcher: Option<CompiledMatcher>,
    pub hooks: Vec<HookHandler>,
}

impl HookMatcherGroup {
    fn from_raw(
        event: HookEventName,
        raw: HookMatcherGroupRaw,
        warnings: &mut Vec<HooksLoadWarning>,
    ) -> anyhow::Result<Option<Self>> {
        let raw_matcher = raw
            .matcher
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty());

        let matcher = match raw_matcher {
            Some(pattern) => {
                if event.matcher_field().is_none() {
                    anyhow::bail!(
                        "hook event '{}' does not support matcher",
                        event.canonical_name()
                    );
                }
                Some(CompiledMatcher::compile_regex(&pattern).with_context(|| {
                    format!(
                        "invalid matcher regex for '{}': {pattern}",
                        event.canonical_name()
                    )
                })?)
            }
            None => None,
        };

        let mut hooks = Vec::new();
        for handler in raw.hooks {
            if let Some(parsed) = HookHandler::from_raw(handler, warnings) {
                hooks.push(parsed);
            }
        }

        Ok(Some(Self { matcher, hooks }))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Parsed hook handler with common metadata and backend config.
pub struct HookHandler {
    pub handler_type: HookHandlerType,
    pub timeout: Duration,
    pub status_message: Option<String>,
    pub if_condition: Option<String>,
    pub once: bool,
    pub config: HookHandlerConfig,
}

impl HookHandler {
    fn from_raw(raw: HookHandlerRaw, warnings: &mut Vec<HooksLoadWarning>) -> Option<Self> {
        if raw.r#async {
            warnings.push(HooksLoadWarning::new(
                "reserved_async_flag",
                "ignoring reserved async=true hook flag in v1".to_owned(),
            ));
        }

        let timeout_secs = raw
            .timeout
            .or(raw.timeout_sec)
            .unwrap_or_else(|| default_timeout_secs(raw.handler_type));

        match raw.handler_type {
            RawHookHandlerType::Command => {
                let command = raw.command.and_then(trimmed_non_empty).or_else(|| {
                    warnings.push(HooksLoadWarning::new(
                        "missing_field",
                        "command hook is missing the 'command' field".to_owned(),
                    ));
                    None
                })?;
                Some(Self {
                    handler_type: HookHandlerType::Command,
                    timeout: Duration::from_secs(timeout_secs),
                    status_message: raw.status_message.and_then(trimmed_non_empty),
                    if_condition: raw.if_condition.and_then(trimmed_non_empty),
                    once: raw.once,
                    config: HookHandlerConfig::Command(CommandHookConfig {
                        command,
                        shell: raw.shell.unwrap_or_default(),
                        env: raw.env,
                    }),
                })
            }
            RawHookHandlerType::Http => {
                let url = raw.url.and_then(trimmed_non_empty).or_else(|| {
                    warnings.push(HooksLoadWarning::new(
                        "missing_field",
                        "http hook is missing the 'url' field".to_owned(),
                    ));
                    None
                })?;
                Some(Self {
                    handler_type: HookHandlerType::Http,
                    timeout: Duration::from_secs(timeout_secs),
                    status_message: raw.status_message.and_then(trimmed_non_empty),
                    if_condition: raw.if_condition.and_then(trimmed_non_empty),
                    once: raw.once,
                    config: HookHandlerConfig::Http(HttpHookConfig {
                        url,
                        headers: raw.headers,
                        allowed_env_vars: raw.allowed_env_vars,
                    }),
                })
            }
            RawHookHandlerType::Prompt => {
                let prompt = raw.prompt.and_then(trimmed_non_empty).or_else(|| {
                    warnings.push(HooksLoadWarning::new(
                        "missing_field",
                        "prompt hook is missing the 'prompt' field".to_owned(),
                    ));
                    None
                })?;
                Some(Self {
                    handler_type: HookHandlerType::Prompt,
                    timeout: Duration::from_secs(timeout_secs),
                    status_message: raw.status_message.and_then(trimmed_non_empty),
                    if_condition: raw.if_condition.and_then(trimmed_non_empty),
                    once: raw.once,
                    config: HookHandlerConfig::Prompt(PromptHookConfig {
                        prompt,
                        model: raw.model.and_then(trimmed_non_empty),
                    }),
                })
            }
            RawHookHandlerType::Agent => {
                let prompt = raw.prompt.and_then(trimmed_non_empty).or_else(|| {
                    warnings.push(HooksLoadWarning::new(
                        "missing_field",
                        "agent hook is missing the 'prompt' field".to_owned(),
                    ));
                    None
                })?;
                Some(Self {
                    handler_type: HookHandlerType::Agent,
                    timeout: Duration::from_secs(timeout_secs),
                    status_message: raw.status_message.and_then(trimmed_non_empty),
                    if_condition: raw.if_condition.and_then(trimmed_non_empty),
                    once: raw.once,
                    config: HookHandlerConfig::Agent(AgentHookConfig {
                        prompt,
                        model: raw.model.and_then(trimmed_non_empty),
                        allowed_tools: raw
                            .allowed_tools
                            .into_iter()
                            .filter_map(trimmed_non_empty)
                            .collect(),
                        max_turns: raw.max_turns,
                    }),
                })
            }
            RawHookHandlerType::Callback | RawHookHandlerType::Function => {
                warnings.push(HooksLoadWarning::new(
                    "sdk_only_backend",
                    "ignoring sdk-only hook backend in hooks.json".to_owned(),
                ));
                None
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Backend-specific hook handler configuration.
pub enum HookHandlerConfig {
    /// Shell command handler.
    Command(CommandHookConfig),
    /// HTTP request handler.
    Http(HttpHookConfig),
    /// Prompt handler.
    Prompt(PromptHookConfig),
    /// Agent handler.
    Agent(AgentHookConfig),
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Command hook configuration.
pub struct CommandHookConfig {
    pub command: String,
    pub shell: HookShell,
    pub env: IndexMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// HTTP hook configuration.
pub struct HttpHookConfig {
    pub url: String,
    pub headers: IndexMap<String, String>,
    pub allowed_env_vars: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Prompt hook configuration.
pub struct PromptHookConfig {
    pub prompt: String,
    pub model: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Agent hook configuration.
pub struct AgentHookConfig {
    pub prompt: String,
    pub model: Option<String>,
    pub allowed_tools: Vec<String>,
    pub max_turns: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
/// Shell used by command hooks.
pub enum HookShell {
    /// Bash.
    #[default]
    Bash,
    /// PowerShell.
    Pwsh,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, EnumString, IntoStaticStr)]
#[strum(ascii_case_insensitive)]
/// Canonical hook event name.
pub enum HookEventName {
    SessionStart,
    SessionEnd,
    UserPromptSubmit,
    PreToolUse,
    PostToolUse,
    PostToolUseFailure,
    Notification,
    Stop,
    SubagentStart,
    SubagentStop,
    PreCompact,
    PostCompact,
    PermissionRequest,
    PermissionDenied,
    Elicitation,
    ElicitationResult,
    WorktreeCreate,
    WorktreeRemove,
    FileChanged,
    CwdChanged,
    InstructionsLoaded,
    ConfigChange,
    Setup,
    TeammateIdle,
    TaskCreated,
    TaskCompleted,
    StopFailure,
    PostSampling,
}

impl HookEventName {
    /// Canonical PascalCase event spelling.
    #[must_use]
    pub fn canonical_name(self) -> &'static str {
        // `strum::IntoStaticStr` provides a `From<Self> for &'static str`
        // impl that returns the variant's PascalCase identifier.
        self.into()
    }

    /// Payload field used for event matcher evaluation.
    #[must_use]
    pub fn matcher_field(self) -> Option<&'static str> {
        match self {
            Self::PreToolUse | Self::PostToolUse | Self::PostToolUseFailure => Some("tool_name"),
            Self::SessionStart => Some("source"),
            Self::SessionEnd => Some("reason"),
            Self::Notification => Some("notification_type"),
            Self::SubagentStart | Self::SubagentStop => Some("agent_type"),
            Self::PreCompact | Self::PostCompact => Some("trigger"),
            Self::UserPromptSubmit
            | Self::Stop
            | Self::PermissionRequest
            | Self::PermissionDenied
            | Self::Elicitation
            | Self::ElicitationResult
            | Self::WorktreeCreate
            | Self::WorktreeRemove
            | Self::FileChanged
            | Self::CwdChanged
            | Self::InstructionsLoaded
            | Self::ConfigChange
            | Self::Setup
            | Self::TeammateIdle
            | Self::TaskCreated
            | Self::TaskCompleted
            | Self::StopFailure
            | Self::PostSampling => None,
        }
    }

    /// Resolve an alias (PascalCase, snake_case, or camelCase) to its canonical
    /// variant. `strum::EnumString` with `ascii_case_insensitive` handles case
    /// variants; we strip underscores so `pre_tool_use` normalizes to
    /// `PreToolUse` without per-variant serde aliases.
    #[must_use]
    pub fn from_alias(alias: &str) -> Option<Self> {
        let normalized: String = alias.chars().filter(|ch| *ch != '_').collect();
        normalized.parse().ok()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Non-fatal issue found while loading a hook file.
pub struct HooksLoadWarning {
    pub category: String,
    pub message: String,
}

impl HooksLoadWarning {
    /// Build a hook load warning.
    #[must_use]
    pub fn new(category: impl Into<String>, message: String) -> Self {
        Self {
            category: category.into(),
            message,
        }
    }
}

#[derive(Debug, Deserialize)]
struct HooksFileRaw {
    #[serde(default)]
    hooks: IndexMap<String, Vec<HookMatcherGroupRaw>>,
}

#[derive(Debug, Deserialize)]
struct HookMatcherGroupRaw {
    #[serde(default)]
    matcher: Option<String>,
    #[serde(default)]
    hooks: Vec<HookHandlerRaw>,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum RawHookHandlerType {
    Command,
    Http,
    Prompt,
    Agent,
    Callback,
    Function,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "snake_case")]
struct HookHandlerRaw {
    #[serde(rename = "type")]
    handler_type: RawHookHandlerType,
    #[serde(default)]
    timeout: Option<u64>,
    #[serde(default, alias = "timeoutSec")]
    timeout_sec: Option<u64>,
    #[serde(default, alias = "statusMessage")]
    status_message: Option<String>,
    #[serde(default, rename = "if")]
    if_condition: Option<String>,
    #[serde(default)]
    r#async: bool,
    #[serde(default)]
    once: bool,
    #[serde(default)]
    command: Option<String>,
    #[serde(default)]
    url: Option<String>,
    #[serde(default)]
    headers: IndexMap<String, String>,
    #[serde(default, alias = "allowedEnvVars")]
    allowed_env_vars: Vec<String>,
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    model: Option<String>,
    #[serde(default, alias = "allowedTools")]
    allowed_tools: Vec<String>,
    #[serde(default, alias = "maxTurns")]
    max_turns: Option<u32>,
    #[serde(default)]
    shell: Option<HookShell>,
    #[serde(default)]
    env: IndexMap<String, String>,
}

fn default_timeout_secs(handler_type: RawHookHandlerType) -> u64 {
    match handler_type {
        RawHookHandlerType::Command | RawHookHandlerType::Http => 600,
        RawHookHandlerType::Agent => 60,
        RawHookHandlerType::Prompt => 30,
        RawHookHandlerType::Callback | RawHookHandlerType::Function => 30,
    }
}

fn trimmed_non_empty(value: String) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hooks_file_uses_first_alias_for_canonical_event() {
        let (parsed, warnings) = HooksFile::from_json_bytes(
            br#"{
                "hooks": {
                    "PreToolUse": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "echo first"
                                }
                            ]
                        }
                    ],
                    "pre_tool_use": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "echo second"
                                }
                            ]
                        }
                    ]
                }
            }"#,
        )
        .expect("parse hooks");

        let groups = parsed
            .hooks
            .get(&HookEventName::PreToolUse)
            .expect("pre tool use hooks");
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].hooks.len(), 1);
        assert_eq!(
            warnings
                .iter()
                .filter(|warning| warning.message.contains("duplicate hook alias"))
                .count(),
            1
        );
    }

    #[test]
    fn hooks_file_warns_on_unknown_events() {
        let (parsed, warnings) = HooksFile::from_json_bytes(
            br#"{
                "hooks": {
                    "UnknownEvent": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "echo ignored"
                                }
                            ]
                        }
                    ],
                    "Stop": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "echo kept"
                                }
                            ]
                        }
                    ]
                }
            }"#,
        )
        .expect("parse hooks");

        assert!(parsed.hooks.contains_key(&HookEventName::Stop));
        assert_eq!(parsed.hooks.len(), 1);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].message.contains("unknown hook event"));
    }

    #[test]
    fn hooks_file_rejects_malformed_json() {
        let error = HooksFile::from_json_bytes(br#"{ "hooks": { "Stop": [ }"#)
            .expect_err("malformed hooks should fail");

        assert!(error.to_string().contains("failed to parse hooks.json"));
    }

    #[test]
    fn hooks_file_warns_on_reserved_async_flag() {
        let (parsed, warnings) = HooksFile::from_json_bytes(
            br#"{
                "hooks": {
                    "Stop": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "echo keep",
                                    "async": true
                                }
                            ]
                        }
                    ]
                }
            }"#,
        )
        .expect("parse hooks");

        let groups = parsed.hooks.get(&HookEventName::Stop).expect("stop hooks");
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].hooks.len(), 1);
        assert_eq!(warnings.len(), 1);
        assert!(warnings[0].message.contains("async=true"));
    }

    #[test]
    fn hooks_file_ignores_sdk_only_backends() {
        let (parsed, warnings) = HooksFile::from_json_bytes(
            br#"{
                "hooks": {
                    "Stop": [
                        {
                            "hooks": [
                                {
                                    "type": "callback"
                                },
                                {
                                    "type": "function"
                                }
                            ]
                        }
                    ]
                }
            }"#,
        )
        .expect("parse hooks");

        assert!(parsed.hooks.is_empty());
        assert_eq!(warnings.len(), 2);
        assert!(
            warnings
                .iter()
                .all(|warning| warning.message.contains("sdk-only hook backend"))
        );
    }

    #[test]
    fn hooks_file_accepts_snake_case_and_camel_case_handler_fields() {
        let (parsed, warnings) = HooksFile::from_json_bytes(
            br#"{
                "hooks": {
                    "Stop": [
                        {
                            "hooks": [
                                {
                                    "type": "agent",
                                    "prompt": "first",
                                    "status_message": "snake case",
                                    "allowed_tools": ["read"],
                                    "max_turns": 2,
                                    "timeout_sec": 7
                                },
                                {
                                    "type": "agent",
                                    "prompt": "second",
                                    "statusMessage": "camel case",
                                    "allowedTools": ["write"],
                                    "maxTurns": 3,
                                    "timeoutSec": 9
                                }
                            ]
                        }
                    ]
                }
            }"#,
        )
        .expect("parse hooks");

        assert!(warnings.is_empty());
        let groups = parsed.hooks.get(&HookEventName::Stop).expect("stop hooks");
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].hooks.len(), 2);

        let HookHandlerConfig::Agent(first) = &groups[0].hooks[0].config else {
            panic!("expected first hook to be an agent");
        };
        assert_eq!(
            groups[0].hooks[0].status_message.as_deref(),
            Some("snake case")
        );
        assert_eq!(groups[0].hooks[0].timeout, Duration::from_secs(7));
        assert_eq!(first.allowed_tools, vec!["read".to_owned()]);
        assert_eq!(first.max_turns, Some(2));

        let HookHandlerConfig::Agent(second) = &groups[0].hooks[1].config else {
            panic!("expected second hook to be an agent");
        };
        assert_eq!(
            groups[0].hooks[1].status_message.as_deref(),
            Some("camel case")
        );
        assert_eq!(groups[0].hooks[1].timeout, Duration::from_secs(9));
        assert_eq!(second.allowed_tools, vec!["write".to_owned()]);
        assert_eq!(second.max_turns, Some(3));
    }

    #[test]
    fn matcher_on_event_without_matcher_field_is_rejected() {
        let error = HooksFile::from_json_bytes(
            br#"{
                "hooks": {
                    "Stop": [
                        {
                            "matcher": "never",
                            "hooks": [
                                {
                                    "type": "prompt",
                                    "prompt": "noop"
                                }
                            ]
                        }
                    ]
                }
            }"#,
        )
        .expect_err("Stop does not support matcher");

        let rendered = format!("{error:#}");
        assert!(rendered.contains("hook event 'Stop' does not support matcher"));
    }
}
