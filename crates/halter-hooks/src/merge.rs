// pattern: Functional Core

use halter_protocol::{HookOutputEntry, HookOutputKind};
use serde::Deserialize;
use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
/// Priority lane used when ordering hook handlers.
pub enum HandlerPriorityGroup {
    /// SDK hook registered before plugin hooks.
    SdkBeforePlugins,
    /// Hook loaded from plugin files.
    PluginFiles,
    /// SDK hook registered after plugin hooks.
    SdkAfterPlugins,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
/// Full ordering key for one hook handler.
pub struct HandlerPriority {
    pub group: HandlerPriorityGroup,
    pub plugin_load_order: usize,
    pub event_declaration_index: usize,
    pub matcher_group_index: usize,
    pub hook_index_within_group: usize,
}

#[derive(Debug, Clone)]
/// Hook output plus the metadata needed to merge it deterministically.
pub struct MergeInput {
    pub handler_id: String,
    pub priority: HandlerPriority,
    pub output: HookOutput,
}

#[derive(Debug, Clone, Default, PartialEq)]
/// Single effective outcome after all hook outputs are merged.
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

/// Closed set of fields that can carry a merge conflict.
///
/// The `Display` rendering of each variant is load-bearing for tracing
/// observability: consumers of `hooks.merge_conflict` events (including log
/// scrapers and tests) expect `"updated_input"` and `"updated_output"`.
/// Do not change these renderings without also updating the consumers.
///
/// This enum is intentionally not `#[non_exhaustive]` and intentionally does
/// not derive `Serialize`/`Deserialize`: the set is closed, and
/// `MergeConflict` is only constructed and consumed inside this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConflictField {
    /// The `updated_input` hook-specific field conflicted.
    UpdatedInput,
    /// The `updated_output` hook-specific field conflicted.
    UpdatedOutput,
}

impl std::fmt::Display for ConflictField {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConflictField::UpdatedInput => write!(f, "updated_input"),
            ConflictField::UpdatedOutput => write!(f, "updated_output"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Record of a field-level merge conflict.
pub struct MergeConflict {
    pub field: ConflictField,
    pub winner: String,
    pub loser: String,
}

#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
/// Wire-compatible hook output accepted from plugin and SDK handlers.
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
/// Basic allow/block decision emitted by hooks.
pub enum HookDecision {
    /// Approve the operation.
    Approve,
    /// Block the operation.
    Block,
}

#[derive(Debug, Clone, Default, Deserialize, PartialEq)]
/// Hook output fields whose meaning depends on the event type.
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

/// Variants ordered **least-restrictive first** so the derived `Ord` matches
/// the semantic "strength" of the decision: `Passthrough < Allow < Ask < Deny`.
/// Merging two outputs picks the stronger decision with `.max(...)` rather
/// than a hand-rolled rank table (finding L16). Serde uses variant names
/// (snake_case), not declaration position, so the reordering is safe.
#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum PermissionDecision {
    Passthrough,
    Allow,
    Ask,
    Deny,
}

/// Merge ordered hook outputs into one runtime outcome.
///
/// Earlier outputs win for single-writer fields such as updated input/output.
/// Permission decisions use the strongest semantic decision instead.
pub fn merge_outputs(inputs: &[MergeInput]) -> (HookMergedOutcome, Vec<MergeConflict>) {
    // Sort references, not values — `HookOutput` carries `serde_json::Value`
    // payloads whose clones can be large. (M24)
    let mut ordered: Vec<&MergeInput> = inputs.iter().collect();
    ordered.sort_by(|left, right| left.priority.cmp(&right.priority));

    let mut merged = HookMergedOutcome::default();
    let mut conflicts = Vec::new();
    let mut winning_updated_input: Option<String> = None;
    let mut winning_updated_output: Option<String> = None;
    let mut winning_permission: Option<(PermissionDecision, String, Option<String>)> = None;

    for input in &ordered {
        let reason = non_empty(input.output.reason.clone());
        let stop_reason = non_empty(input.output.stop_reason.clone());

        if matches!(input.output.continue_execution, Some(false)) && merged.stop_reason.is_none() {
            merged.stop_reason = stop_reason
                .clone()
                .or(reason.clone())
                .or_else(|| Some(default_stop_reason().to_owned()));
        }

        if matches!(input.output.decision, Some(HookDecision::Block))
            && merged.block_reason.is_none()
        {
            merged.block_reason = Some(
                reason
                    .clone()
                    .unwrap_or_else(|| default_block_reason().to_owned()),
            );
        }

        if let Some(permission_decision) = input
            .output
            .hook_specific_output
            .as_ref()
            .and_then(|output| output.permission_decision)
        {
            let permission_reason = input
                .output
                .hook_specific_output
                .as_ref()
                .and_then(|output| non_empty(output.permission_decision_reason.clone()));
            match &winning_permission {
                Some((current, _, _)) if *current >= permission_decision => {}
                _ => {
                    winning_permission = Some((
                        permission_decision,
                        input.handler_id.clone(),
                        permission_reason.clone(),
                    ));
                    merged.permission_decision = Some(permission_decision);
                    merged.permission_decision_reason = permission_reason;
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
                    field: ConflictField::UpdatedInput,
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
                    field: ConflictField::UpdatedOutput,
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
        merged.block_reason =
            Some(reason.unwrap_or_else(|| default_permission_block_reason(decision).to_owned()));
    }

    (merged, conflicts)
}

/// Convert one hook output into summary entries for event reporting.
pub fn summary_entries(output: &HookOutput) -> Vec<HookOutputEntry> {
    let mut entries = Vec::new();
    if let Some(reason) = output
        .reason
        .clone()
        .filter(|value| !value.trim().is_empty())
    {
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

fn non_empty(value: Option<String>) -> Option<String> {
    value.and_then(|value| {
        let trimmed = value.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_owned())
    })
}

fn default_stop_reason() -> &'static str {
    "hook requested stop"
}

fn default_block_reason() -> &'static str {
    "hook blocked without explanation"
}

fn default_permission_block_reason(decision: PermissionDecision) -> &'static str {
    match decision {
        PermissionDecision::Deny => "hook denied permission without explanation",
        PermissionDecision::Ask => "hook requested permission confirmation without explanation",
        PermissionDecision::Allow | PermissionDecision::Passthrough => {
            "hook blocked without explanation"
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn priority(
        group: HandlerPriorityGroup,
        plugin_load_order: usize,
        event_declaration_index: usize,
        matcher_group_index: usize,
        hook_index_within_group: usize,
    ) -> HandlerPriority {
        HandlerPriority {
            group,
            plugin_load_order,
            event_declaration_index,
            matcher_group_index,
            hook_index_within_group,
        }
    }

    fn merge_input(handler_id: &str, priority: HandlerPriority, output: HookOutput) -> MergeInput {
        MergeInput {
            handler_id: handler_id.to_owned(),
            priority,
            output,
        }
    }

    #[test]
    fn merge_prefers_highest_priority_updated_input() {
        let (merged, conflicts) = merge_outputs(&[
            merge_input(
                "plugin-a",
                priority(HandlerPriorityGroup::PluginFiles, 0, 0, 0, 0),
                HookOutput {
                    hook_specific_output: Some(HookSpecificOutput {
                        updated_input: Some(json!({"command": "echo a"})),
                        ..HookSpecificOutput::default()
                    }),
                    ..HookOutput::default()
                },
            ),
            merge_input(
                "plugin-b",
                priority(HandlerPriorityGroup::PluginFiles, 1, 0, 0, 0),
                HookOutput {
                    hook_specific_output: Some(HookSpecificOutput {
                        updated_input: Some(json!({"command": "echo b"})),
                        ..HookSpecificOutput::default()
                    }),
                    ..HookOutput::default()
                },
            ),
        ]);

        assert_eq!(merged.updated_input, Some(json!({"command": "echo a"})));
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].winner, "plugin-a");
        assert_eq!(conflicts[0].loser, "plugin-b");
    }

    #[test]
    fn merge_synthesizes_block_reason_when_reason_is_missing() {
        let (merged, conflicts) = merge_outputs(&[merge_input(
            "plugin-a",
            priority(HandlerPriorityGroup::PluginFiles, 0, 0, 0, 0),
            HookOutput {
                decision: Some(HookDecision::Block),
                ..HookOutput::default()
            },
        )]);

        assert_eq!(
            merged.block_reason.as_deref(),
            Some("hook blocked without explanation")
        );
        assert!(conflicts.is_empty());
    }

    #[test]
    fn merge_uses_default_stop_reason_when_continue_stops_without_reason() {
        let (merged, conflicts) = merge_outputs(&[merge_input(
            "plugin-a",
            priority(HandlerPriorityGroup::PluginFiles, 0, 0, 0, 0),
            HookOutput {
                continue_execution: Some(false),
                ..HookOutput::default()
            },
        )]);

        assert_eq!(merged.stop_reason.as_deref(), Some("hook requested stop"));
        assert!(conflicts.is_empty());
    }

    #[test]
    fn merge_prefers_strongest_permission_decision() {
        let (merged, conflicts) = merge_outputs(&[
            merge_input(
                "allow",
                priority(HandlerPriorityGroup::PluginFiles, 0, 0, 0, 0),
                HookOutput {
                    hook_specific_output: Some(HookSpecificOutput {
                        permission_decision: Some(PermissionDecision::Allow),
                        permission_decision_reason: Some("allow".to_owned()),
                        ..HookSpecificOutput::default()
                    }),
                    ..HookOutput::default()
                },
            ),
            merge_input(
                "deny",
                priority(HandlerPriorityGroup::PluginFiles, 1, 0, 0, 0),
                HookOutput {
                    hook_specific_output: Some(HookSpecificOutput {
                        permission_decision: Some(PermissionDecision::Deny),
                        permission_decision_reason: Some("deny".to_owned()),
                        ..HookSpecificOutput::default()
                    }),
                    ..HookOutput::default()
                },
            ),
        ]);

        assert_eq!(merged.permission_decision, Some(PermissionDecision::Deny));
        assert_eq!(merged.permission_decision_reason.as_deref(), Some("deny"));
        assert_eq!(merged.block_reason.as_deref(), Some("deny"));
        assert!(conflicts.is_empty());
    }

    #[test]
    fn merge_synthesizes_permission_block_reason_when_missing() {
        let (merged, conflicts) = merge_outputs(&[merge_input(
            "deny",
            priority(HandlerPriorityGroup::PluginFiles, 0, 0, 0, 0),
            HookOutput {
                hook_specific_output: Some(HookSpecificOutput {
                    permission_decision: Some(PermissionDecision::Deny),
                    ..HookSpecificOutput::default()
                }),
                ..HookOutput::default()
            },
        )]);

        assert_eq!(merged.permission_decision, Some(PermissionDecision::Deny));
        assert_eq!(
            merged.block_reason.as_deref(),
            Some("hook denied permission without explanation")
        );
        assert!(conflicts.is_empty());
    }

    #[test]
    fn merge_orders_context_and_system_messages_by_priority() {
        let (merged, conflicts) = merge_outputs(&[
            merge_input(
                "sdk-before",
                priority(HandlerPriorityGroup::SdkBeforePlugins, 0, 0, 0, 0),
                HookOutput {
                    system_message: Some("sdk-before-message".to_owned()),
                    hook_specific_output: Some(HookSpecificOutput {
                        additional_context: Some("sdk-before-context".to_owned()),
                        ..HookSpecificOutput::default()
                    }),
                    ..HookOutput::default()
                },
            ),
            merge_input(
                "plugin",
                priority(HandlerPriorityGroup::PluginFiles, 0, 0, 0, 0),
                HookOutput {
                    system_message: Some("plugin-message".to_owned()),
                    hook_specific_output: Some(HookSpecificOutput {
                        additional_context: Some("plugin-context".to_owned()),
                        ..HookSpecificOutput::default()
                    }),
                    ..HookOutput::default()
                },
            ),
            merge_input(
                "sdk-after",
                priority(HandlerPriorityGroup::SdkAfterPlugins, 0, 0, 0, 0),
                HookOutput {
                    system_message: Some("sdk-after-message".to_owned()),
                    hook_specific_output: Some(HookSpecificOutput {
                        additional_context: Some("sdk-after-context".to_owned()),
                        ..HookSpecificOutput::default()
                    }),
                    ..HookOutput::default()
                },
            ),
        ]);

        assert_eq!(
            merged.additional_context,
            vec![
                "sdk-before-context".to_owned(),
                "plugin-context".to_owned(),
                "sdk-after-context".to_owned(),
            ]
        );
        assert_eq!(
            merged.system_messages,
            vec![
                "sdk-before-message".to_owned(),
                "plugin-message".to_owned(),
                "sdk-after-message".to_owned(),
            ]
        );
        assert!(conflicts.is_empty());
    }

    #[test]
    fn merge_uses_full_priority_tuple_for_tie_breaks() {
        let (merged, conflicts) = merge_outputs(&[
            merge_input(
                "later-matcher",
                priority(HandlerPriorityGroup::PluginFiles, 0, 0, 1, 0),
                HookOutput {
                    hook_specific_output: Some(HookSpecificOutput {
                        updated_input: Some(json!({"command": "echo later"})),
                        ..HookSpecificOutput::default()
                    }),
                    ..HookOutput::default()
                },
            ),
            merge_input(
                "earlier-matcher",
                priority(HandlerPriorityGroup::PluginFiles, 0, 0, 0, 1),
                HookOutput {
                    hook_specific_output: Some(HookSpecificOutput {
                        updated_input: Some(json!({"command": "echo earlier"})),
                        ..HookSpecificOutput::default()
                    }),
                    ..HookOutput::default()
                },
            ),
        ]);

        assert_eq!(
            merged.updated_input,
            Some(json!({"command": "echo earlier"}))
        );
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].winner, "earlier-matcher");
        assert_eq!(conflicts[0].loser, "later-matcher");
    }

    #[test]
    fn merge_prefers_earlier_event_declaration_index() {
        let (merged, conflicts) = merge_outputs(&[
            merge_input(
                "later-event",
                priority(HandlerPriorityGroup::PluginFiles, 0, 1, 0, 0),
                HookOutput {
                    hook_specific_output: Some(HookSpecificOutput {
                        updated_input: Some(json!({"command": "echo later-event"})),
                        ..HookSpecificOutput::default()
                    }),
                    ..HookOutput::default()
                },
            ),
            merge_input(
                "earlier-event",
                priority(HandlerPriorityGroup::PluginFiles, 0, 0, 0, 0),
                HookOutput {
                    hook_specific_output: Some(HookSpecificOutput {
                        updated_input: Some(json!({"command": "echo earlier-event"})),
                        ..HookSpecificOutput::default()
                    }),
                    ..HookOutput::default()
                },
            ),
        ]);

        assert_eq!(
            merged.updated_input,
            Some(json!({"command": "echo earlier-event"}))
        );
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].winner, "earlier-event");
    }

    #[test]
    fn merge_prefers_earlier_hook_index_within_group() {
        let (merged, conflicts) = merge_outputs(&[
            merge_input(
                "later-hook",
                priority(HandlerPriorityGroup::PluginFiles, 0, 0, 0, 1),
                HookOutput {
                    hook_specific_output: Some(HookSpecificOutput {
                        updated_input: Some(json!({"command": "echo later-hook"})),
                        ..HookSpecificOutput::default()
                    }),
                    ..HookOutput::default()
                },
            ),
            merge_input(
                "earlier-hook",
                priority(HandlerPriorityGroup::PluginFiles, 0, 0, 0, 0),
                HookOutput {
                    hook_specific_output: Some(HookSpecificOutput {
                        updated_input: Some(json!({"command": "echo earlier-hook"})),
                        ..HookSpecificOutput::default()
                    }),
                    ..HookOutput::default()
                },
            ),
        ]);

        assert_eq!(
            merged.updated_input,
            Some(json!({"command": "echo earlier-hook"}))
        );
        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].winner, "earlier-hook");
    }

    #[test]
    fn merge_prefers_earlier_priority_for_same_permission_strength() {
        let (merged, conflicts) = merge_outputs(&[
            merge_input(
                "earlier",
                priority(HandlerPriorityGroup::PluginFiles, 0, 0, 0, 0),
                HookOutput {
                    hook_specific_output: Some(HookSpecificOutput {
                        permission_decision: Some(PermissionDecision::Ask),
                        permission_decision_reason: Some("earlier".to_owned()),
                        ..HookSpecificOutput::default()
                    }),
                    ..HookOutput::default()
                },
            ),
            merge_input(
                "later",
                priority(HandlerPriorityGroup::PluginFiles, 1, 0, 0, 0),
                HookOutput {
                    hook_specific_output: Some(HookSpecificOutput {
                        permission_decision: Some(PermissionDecision::Ask),
                        permission_decision_reason: Some("later".to_owned()),
                        ..HookSpecificOutput::default()
                    }),
                    ..HookOutput::default()
                },
            ),
        ]);

        assert_eq!(merged.permission_decision, Some(PermissionDecision::Ask));
        assert_eq!(
            merged.permission_decision_reason.as_deref(),
            Some("earlier")
        );
        assert_eq!(merged.block_reason.as_deref(), Some("earlier"));
        assert!(conflicts.is_empty());
    }

    #[test]
    fn conflict_field_display_renders_legacy_strings() {
        assert_eq!(
            format!("{}", ConflictField::UpdatedInput),
            "updated_input"
        );
        assert_eq!(
            format!("{}", ConflictField::UpdatedOutput),
            "updated_output"
        );
    }

    #[test]
    fn merge_conflict_updated_input_records_conflict_field() {
        let (_, conflicts) = merge_outputs(&[
            merge_input(
                "winner",
                priority(HandlerPriorityGroup::PluginFiles, 0, 0, 0, 0),
                HookOutput {
                    hook_specific_output: Some(HookSpecificOutput {
                        updated_input: Some(json!({"command": "echo winner"})),
                        ..HookSpecificOutput::default()
                    }),
                    ..HookOutput::default()
                },
            ),
            merge_input(
                "loser",
                priority(HandlerPriorityGroup::PluginFiles, 1, 0, 0, 0),
                HookOutput {
                    hook_specific_output: Some(HookSpecificOutput {
                        updated_input: Some(json!({"command": "echo loser"})),
                        ..HookSpecificOutput::default()
                    }),
                    ..HookOutput::default()
                },
            ),
        ]);

        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].field, ConflictField::UpdatedInput);
        assert_eq!(conflicts[0].winner, "winner");
        assert_eq!(conflicts[0].loser, "loser");
    }

    #[test]
    fn merge_conflict_updated_output_records_conflict_field() {
        let (_, conflicts) = merge_outputs(&[
            merge_input(
                "winner",
                priority(HandlerPriorityGroup::PluginFiles, 0, 0, 0, 0),
                HookOutput {
                    hook_specific_output: Some(HookSpecificOutput {
                        updated_mcp_tool_output: Some(json!({"result": "winner"})),
                        ..HookSpecificOutput::default()
                    }),
                    ..HookOutput::default()
                },
            ),
            merge_input(
                "loser",
                priority(HandlerPriorityGroup::PluginFiles, 1, 0, 0, 0),
                HookOutput {
                    hook_specific_output: Some(HookSpecificOutput {
                        updated_mcp_tool_output: Some(json!({"result": "loser"})),
                        ..HookSpecificOutput::default()
                    }),
                    ..HookOutput::default()
                },
            ),
        ]);

        assert_eq!(conflicts.len(), 1);
        assert_eq!(conflicts[0].field, ConflictField::UpdatedOutput);
        assert_eq!(conflicts[0].winner, "winner");
        assert_eq!(conflicts[0].loser, "loser");
    }

    #[test]
    fn merge_conflict_both_fields_in_one_merge() {
        let (merged, conflicts) = merge_outputs(&[
            merge_input(
                "input-winner",
                priority(HandlerPriorityGroup::PluginFiles, 0, 0, 0, 0),
                HookOutput {
                    hook_specific_output: Some(HookSpecificOutput {
                        updated_input: Some(json!({"command": "echo input-winner"})),
                        ..HookSpecificOutput::default()
                    }),
                    ..HookOutput::default()
                },
            ),
            merge_input(
                "input-loser",
                priority(HandlerPriorityGroup::PluginFiles, 1, 0, 0, 0),
                HookOutput {
                    hook_specific_output: Some(HookSpecificOutput {
                        updated_input: Some(json!({"command": "echo input-loser"})),
                        ..HookSpecificOutput::default()
                    }),
                    ..HookOutput::default()
                },
            ),
            merge_input(
                "output-winner",
                priority(HandlerPriorityGroup::PluginFiles, 2, 0, 0, 0),
                HookOutput {
                    hook_specific_output: Some(HookSpecificOutput {
                        updated_mcp_tool_output: Some(json!({"result": "output-winner"})),
                        ..HookSpecificOutput::default()
                    }),
                    ..HookOutput::default()
                },
            ),
            merge_input(
                "output-loser",
                priority(HandlerPriorityGroup::PluginFiles, 3, 0, 0, 0),
                HookOutput {
                    hook_specific_output: Some(HookSpecificOutput {
                        updated_mcp_tool_output: Some(json!({"result": "output-loser"})),
                        ..HookSpecificOutput::default()
                    }),
                    ..HookOutput::default()
                },
            ),
        ]);

        assert_eq!(merged.updated_input, Some(json!({"command": "echo input-winner"})));
        assert_eq!(merged.updated_output, Some(json!({"result": "output-winner"})));
        assert_eq!(conflicts.len(), 2);
        assert_eq!(conflicts[0].field, ConflictField::UpdatedInput);
        assert_eq!(conflicts[0].winner, "input-winner");
        assert_eq!(conflicts[0].loser, "input-loser");
        assert_eq!(conflicts[1].field, ConflictField::UpdatedOutput);
        assert_eq!(conflicts[1].winner, "output-winner");
        assert_eq!(conflicts[1].loser, "output-loser");
    }
}
