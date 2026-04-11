// pattern: Functional Core

use std::time::Duration;

use anyhow::Context;
use halter_protocol::HookHandlerType;
use indexmap::{IndexMap, IndexSet};
use regex::Regex;
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HooksFile {
    pub hooks: IndexMap<HookEventName, Vec<HookMatcherGroup>>,
}

impl HooksFile {
    pub fn from_json_bytes(bytes: &[u8]) -> anyhow::Result<(Self, Vec<HooksLoadWarning>)> {
        let raw: HooksFileRaw =
            serde_json::from_slice(bytes).context("failed to parse hooks.json")?;
        Ok(Self::from_raw(raw))
    }

    fn from_raw(raw: HooksFileRaw) -> (Self, Vec<HooksLoadWarning>) {
        let mut hooks = IndexMap::new();
        let mut warnings = Vec::new();
        let mut seen = IndexSet::new();

        for (event_alias, matcher_groups) in raw.hooks {
            let Some(event) = HookEventName::from_alias(&event_alias) else {
                warnings.push(HooksLoadWarning::new(format!(
                    "unknown hook event '{event_alias}'"
                )));
                continue;
            };
            if !seen.insert(event) {
                warnings.push(HooksLoadWarning::new(format!(
                    "duplicate hook alias '{event_alias}' resolved to '{}'",
                    event.canonical_name()
                )));
                continue;
            }

            let mut parsed_groups = Vec::new();
            for matcher_group in matcher_groups {
                match HookMatcherGroup::from_raw(event, matcher_group, &mut warnings) {
                    Some(group) if !group.hooks.is_empty() => parsed_groups.push(group),
                    Some(_) | None => {}
                }
            }

            if !parsed_groups.is_empty() {
                hooks.insert(event, parsed_groups);
            }
        }

        (Self { hooks }, warnings)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HookMatcherGroup {
    pub matcher: Option<String>,
    pub hooks: Vec<HookHandler>,
}

impl HookMatcherGroup {
    fn from_raw(
        event: HookEventName,
        raw: HookMatcherGroupRaw,
        warnings: &mut Vec<HooksLoadWarning>,
    ) -> Option<Self> {
        let matcher = raw
            .matcher
            .map(|value| value.trim().to_owned())
            .filter(|value| !value.is_empty());

        if let Some(pattern) = matcher.as_deref()
            && Regex::new(pattern).is_err()
        {
            warnings.push(HooksLoadWarning::new(format!(
                "invalid matcher regex for '{}': {pattern}",
                event.canonical_name()
            )));
            return None;
        }

        let mut hooks = Vec::new();
        for handler in raw.hooks {
            match HookHandler::from_raw(handler, warnings) {
                Some(parsed) => hooks.push(parsed),
                None => {}
            }
        }

        Some(Self { matcher, hooks })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HookHandler {
    pub handler_type: HookHandlerType,
    pub timeout: Duration,
    pub status_message: Option<String>,
    pub once: bool,
    pub config: HookHandlerConfig,
}

impl HookHandler {
    fn from_raw(raw: HookHandlerRaw, warnings: &mut Vec<HooksLoadWarning>) -> Option<Self> {
        if raw.if_condition.is_some() {
            warnings.push(HooksLoadWarning::new(
                "ignoring hook handler with unsupported 'if' filter".to_owned(),
            ));
            return None;
        }
        if raw.r#async {
            warnings.push(HooksLoadWarning::new(
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
                        "command hook is missing the 'command' field".to_owned(),
                    ));
                    None
                })?;
                Some(Self {
                    handler_type: HookHandlerType::Command,
                    timeout: Duration::from_secs(timeout_secs),
                    status_message: raw.status_message.and_then(trimmed_non_empty),
                    once: raw.once,
                    config: HookHandlerConfig::Command(CommandHookConfig {
                        command,
                        shell: raw.shell.unwrap_or_default(),
                        env: raw.env,
                    }),
                })
            }
            RawHookHandlerType::Http => {
                warnings.push(HooksLoadWarning::new(
                    "ignoring unsupported http hook backend".to_owned(),
                ));
                None
            }
            RawHookHandlerType::Prompt => {
                warnings.push(HooksLoadWarning::new(
                    "ignoring unsupported prompt hook backend".to_owned(),
                ));
                None
            }
            RawHookHandlerType::Agent => {
                warnings.push(HooksLoadWarning::new(
                    "ignoring unsupported agent hook backend".to_owned(),
                ));
                None
            }
            RawHookHandlerType::Callback | RawHookHandlerType::Function => {
                warnings.push(HooksLoadWarning::new(
                    "ignoring sdk-only hook backend in hooks.json".to_owned(),
                ));
                None
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HookHandlerConfig {
    Command(CommandHookConfig),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandHookConfig {
    pub command: String,
    pub shell: HookShell,
    pub env: IndexMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HookShell {
    #[default]
    Bash,
    Pwsh,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
    #[must_use]
    pub fn canonical_name(self) -> &'static str {
        match self {
            Self::SessionStart => "SessionStart",
            Self::SessionEnd => "SessionEnd",
            Self::UserPromptSubmit => "UserPromptSubmit",
            Self::PreToolUse => "PreToolUse",
            Self::PostToolUse => "PostToolUse",
            Self::PostToolUseFailure => "PostToolUseFailure",
            Self::Notification => "Notification",
            Self::Stop => "Stop",
            Self::SubagentStart => "SubagentStart",
            Self::SubagentStop => "SubagentStop",
            Self::PreCompact => "PreCompact",
            Self::PostCompact => "PostCompact",
            Self::PermissionRequest => "PermissionRequest",
            Self::PermissionDenied => "PermissionDenied",
            Self::Elicitation => "Elicitation",
            Self::ElicitationResult => "ElicitationResult",
            Self::WorktreeCreate => "WorktreeCreate",
            Self::WorktreeRemove => "WorktreeRemove",
            Self::FileChanged => "FileChanged",
            Self::CwdChanged => "CwdChanged",
            Self::InstructionsLoaded => "InstructionsLoaded",
            Self::ConfigChange => "ConfigChange",
            Self::Setup => "Setup",
            Self::TeammateIdle => "TeammateIdle",
            Self::TaskCreated => "TaskCreated",
            Self::TaskCompleted => "TaskCompleted",
            Self::StopFailure => "StopFailure",
            Self::PostSampling => "PostSampling",
        }
    }

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

    #[must_use]
    pub fn from_alias(alias: &str) -> Option<Self> {
        let normalized = alias
            .chars()
            .filter(|ch| *ch != '_')
            .flat_map(char::to_lowercase)
            .collect::<String>();
        match normalized.as_str() {
            "sessionstart" => Some(Self::SessionStart),
            "sessionend" => Some(Self::SessionEnd),
            "userpromptsubmit" => Some(Self::UserPromptSubmit),
            "pretooluse" => Some(Self::PreToolUse),
            "posttooluse" => Some(Self::PostToolUse),
            "posttoolusefailure" => Some(Self::PostToolUseFailure),
            "notification" => Some(Self::Notification),
            "stop" => Some(Self::Stop),
            "subagentstart" => Some(Self::SubagentStart),
            "subagentstop" => Some(Self::SubagentStop),
            "precompact" => Some(Self::PreCompact),
            "postcompact" => Some(Self::PostCompact),
            "permissionrequest" => Some(Self::PermissionRequest),
            "permissiondenied" => Some(Self::PermissionDenied),
            "elicitation" => Some(Self::Elicitation),
            "elicitationresult" => Some(Self::ElicitationResult),
            "worktreecreate" => Some(Self::WorktreeCreate),
            "worktreeremove" => Some(Self::WorktreeRemove),
            "filechanged" => Some(Self::FileChanged),
            "cwdchanged" => Some(Self::CwdChanged),
            "instructionsloaded" => Some(Self::InstructionsLoaded),
            "configchange" => Some(Self::ConfigChange),
            "setup" => Some(Self::Setup),
            "teammateidle" => Some(Self::TeammateIdle),
            "taskcreated" => Some(Self::TaskCreated),
            "taskcompleted" => Some(Self::TaskCompleted),
            "stopfailure" => Some(Self::StopFailure),
            "postsampling" => Some(Self::PostSampling),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HooksLoadWarning {
    pub message: String,
}

impl HooksLoadWarning {
    #[must_use]
    pub fn new(message: String) -> Self {
        Self { message }
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
struct HookHandlerRaw {
    #[serde(rename = "type")]
    handler_type: RawHookHandlerType,
    #[serde(default)]
    timeout: Option<u64>,
    #[serde(default, rename = "timeoutSec")]
    timeout_sec: Option<u64>,
    #[serde(default, rename = "statusMessage")]
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
}
