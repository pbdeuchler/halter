// pattern: Functional Core

use halter_protocol::{HookOutputEntry, HookOutputKind};
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct HandlerPriority {
    pub plugin_load_order: usize,
    pub event_declaration_index: usize,
    pub matcher_group_index: usize,
    pub hook_index_within_group: usize,
}

#[derive(Debug, Clone)]
pub struct MergeInput {
    pub handler_id: String,
    pub priority: HandlerPriority,
    pub output: HookOutput,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct HookMergedOutcome {
    pub stop_reason: Option<String>,
    pub block_reason: Option<String>,
    pub permission_decision: Option<PermissionDecision>,
    pub permission_decision_reason: Option<String>,
    pub updated_input: Option<Value>,
    pub updated_output: Option<Value>,
    pub additional_context: Vec<String>,
    pub system_messages: Vec<String>,
    pub suppress_output: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergeConflict {
    pub field: &'static str,
    pub winner: String,
    pub loser: String,
}

#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct HookOutput {
    #[serde(default, rename = "continue")]
    pub continue_execution: Option<bool>,
    #[serde(default, rename = "suppressOutput")]
    pub suppress_output: Option<bool>,
    #[serde(default)]
    pub decision: Option<HookDecision>,
    #[serde(default)]
    pub reason: Option<String>,
    #[serde(default, rename = "stopReason")]
    pub stop_reason: Option<String>,
    #[serde(default, rename = "systemMessage")]
    pub system_message: Option<String>,
    #[serde(default, rename = "hookSpecificOutput")]
    pub hook_specific_output: Option<HookSpecificOutput>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HookDecision {
    Approve,
    Block,
}

#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
pub struct HookSpecificOutput {
    #[serde(default, rename = "hookEventName")]
    pub hook_event_name: Option<String>,
    #[serde(default, rename = "permissionDecision")]
    pub permission_decision: Option<PermissionDecision>,
    #[serde(default, rename = "permissionDecisionReason")]
    pub permission_decision_reason: Option<String>,
    #[serde(default, rename = "updatedInput")]
    pub updated_input: Option<Value>,
    #[serde(default, rename = "updatedMCPToolOutput")]
    pub updated_mcp_tool_output: Option<Value>,
    #[serde(default, rename = "additionalContext")]
    pub additional_context: Option<String>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PermissionDecision {
    Deny,
    Ask,
    Allow,
    Passthrough,
}

pub fn merge_outputs(inputs: &[MergeInput]) -> (HookMergedOutcome, Vec<MergeConflict>) {
    let mut ordered = inputs.to_vec();
    ordered.sort_by(|left, right| left.priority.cmp(&right.priority));

    let mut merged = HookMergedOutcome::default();
    let mut conflicts = Vec::new();
    let mut winning_updated_input: Option<String> = None;
    let mut winning_updated_output: Option<String> = None;
    let mut winning_permission: Option<(PermissionDecision, String, Option<String>)> = None;

    for input in &ordered {
        if matches!(input.output.continue_execution, Some(false)) && merged.stop_reason.is_none() {
            merged.stop_reason = input
                .output
                .stop_reason
                .clone()
                .or_else(|| input.output.reason.clone())
                .or_else(|| Some("hook requested stop".to_owned()));
        }

        if matches!(input.output.decision, Some(HookDecision::Block)) && merged.block_reason.is_none()
        {
            merged.block_reason = input.output.reason.clone();
        }

        if let Some(permission_decision) = input
            .output
            .hook_specific_output
            .as_ref()
            .and_then(|output| output.permission_decision)
        {
            let reason = input
                .output
                .hook_specific_output
                .as_ref()
                .and_then(|output| output.permission_decision_reason.clone());
            match &winning_permission {
                Some((current, _, _))
                    if permission_rank(*current) >= permission_rank(permission_decision) => {}
                _ => {
                    winning_permission = Some((
                        permission_decision,
                        input.handler_id.clone(),
                        reason.clone(),
                    ));
                    merged.permission_decision = Some(permission_decision);
                    merged.permission_decision_reason = reason;
                }
            }
        }

        if let Some(updated_input) = input
            .output
            .hook_specific_output
            .as_ref()
            .and_then(|output| output.updated_input.clone())
        {
            if merged.updated_input.is_none() {
                merged.updated_input = Some(updated_input);
                winning_updated_input = Some(input.handler_id.clone());
            } else if let Some(winner) = winning_updated_input.as_ref() {
                conflicts.push(MergeConflict {
                    field: "updated_input",
                    winner: winner.clone(),
                    loser: input.handler_id.clone(),
                });
            }
        }

        if let Some(updated_output) = input
            .output
            .hook_specific_output
            .as_ref()
            .and_then(|output| output.updated_mcp_tool_output.clone())
        {
            if merged.updated_output.is_none() {
                merged.updated_output = Some(updated_output);
                winning_updated_output = Some(input.handler_id.clone());
            } else if let Some(winner) = winning_updated_output.as_ref() {
                conflicts.push(MergeConflict {
                    field: "updated_output",
                    winner: winner.clone(),
                    loser: input.handler_id.clone(),
                });
            }
        }

        if let Some(context) = input
            .output
            .hook_specific_output
            .as_ref()
            .and_then(|output| output.additional_context.clone())
            .filter(|value| !value.trim().is_empty())
        {
            merged.additional_context.push(context);
        }

        if let Some(message) = input
            .output
            .system_message
            .clone()
            .filter(|value| !value.trim().is_empty())
        {
            merged.system_messages.push(message);
        }

        if matches!(input.output.suppress_output, Some(true)) {
            merged.suppress_output = true;
        }
    }

    if let Some((decision, _, reason)) = winning_permission
        && matches!(decision, PermissionDecision::Deny | PermissionDecision::Ask)
        && merged.block_reason.is_none()
    {
        merged.block_reason = reason;
    }

    (merged, conflicts)
}

pub fn summary_entries(output: &HookOutput) -> Vec<HookOutputEntry> {
    let mut entries = Vec::new();
    if let Some(reason) = output.reason.clone().filter(|value| !value.trim().is_empty()) {
        let kind = if matches!(output.decision, Some(HookDecision::Block)) {
            HookOutputKind::Stop
        } else {
            HookOutputKind::Warning
        };
        entries.push(HookOutputEntry { kind, text: reason });
    }
    if let Some(stop_reason) = output
        .stop_reason
        .clone()
        .filter(|value| !value.trim().is_empty())
    {
        entries.push(HookOutputEntry {
            kind: HookOutputKind::Stop,
            text: stop_reason,
        });
    }
    if let Some(system_message) = output
        .system_message
        .clone()
        .filter(|value| !value.trim().is_empty())
    {
        entries.push(HookOutputEntry {
            kind: HookOutputKind::Feedback,
            text: system_message,
        });
    }
    if let Some(context) = output
        .hook_specific_output
        .as_ref()
        .and_then(|value| value.additional_context.clone())
        .filter(|value| !value.trim().is_empty())
    {
        entries.push(HookOutputEntry {
            kind: HookOutputKind::Context,
            text: context,
        });
    }
    entries
}

fn permission_rank(value: PermissionDecision) -> u8 {
    match value {
        PermissionDecision::Passthrough => 0,
        PermissionDecision::Allow => 1,
        PermissionDecision::Ask => 2,
        PermissionDecision::Deny => 3,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn merge_prefers_highest_priority_updated_input() {
        let (merged, conflicts) = merge_outputs(&[
            MergeInput {
                handler_id: "plugin-a".to_owned(),
                priority: HandlerPriority {
                    plugin_load_order: 0,
                    event_declaration_index: 0,
                    matcher_group_index: 0,
                    hook_index_within_group: 0,
                },
                output: HookOutput {
                    hook_specific_output: Some(HookSpecificOutput {
                        updated_input: Some(json!({"command": "echo a"})),
                        ..HookSpecificOutput::default()
                    }),
                    ..HookOutput::default()
                },
            },
            MergeInput {
                handler_id: "plugin-b".to_owned(),
                priority: HandlerPriority {
                    plugin_load_order: 1,
                    event_declaration_index: 0,
                    matcher_group_index: 0,
                    hook_index_within_group: 0,
                },
                output: HookOutput {
                    hook_specific_output: Some(HookSpecificOutput {
                        updated_input: Some(json!({"command": "echo b"})),
                        ..HookSpecificOutput::default()
                    }),
                    ..HookOutput::default()
                },
            },
        ]);

        assert_eq!(merged.updated_input, Some(json!({"command": "echo a"})));
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].winner, "plugin-a");
        assert_eq!(conflicts[0].loser, "plugin-b");
    }
}
