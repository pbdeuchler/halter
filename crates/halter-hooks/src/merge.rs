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
    let mut block_handler: Option<String> = None;
    let mut stop_handler: Option<String> = None;

    for input in &ordered {
        let reason = non_empty(input.output.reason.clone());
        let stop_reason = non_empty(input.output.stop_reason.clone());

        if matches!(input.output.continue_execution, Some(false)) && merged.stop_reason.is_none() {
            merged.stop_reason = stop_reason
                .clone()
                .or(reason.clone())
                .or_else(|| Some(default_stop_reason().to_owned()));
            stop_handler = Some(input.handler_id.clone());
        }

        if matches!(input.output.decision, Some(HookDecision::Block))
            && merged.block_reason.is_none()
        {
            merged.block_reason = Some(
                reason
                    .clone()
                    .unwrap_or_else(|| default_block_reason().to_owned()),
            );
            block_handler = Some(input.handler_id.clone());
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

    let permission_handler = winning_permission
        .as_ref()
        .map(|(_, handler_id, _)| handler_id.clone());
    if let Some((decision, handler_id, reason)) = winning_permission
        && matches!(decision, PermissionDecision::Deny | PermissionDecision::Ask)
        && merged.block_reason.is_none()
    {
        merged.block_reason =
            Some(reason.unwrap_or_else(|| default_permission_block_reason(decision).to_owned()));
        block_handler = Some(handler_id);
    }

    if !inputs.is_empty() {
        log_decision(
            &merged,
            DecisionHandlers {
                permission: permission_handler.as_deref(),
                block: block_handler.as_deref(),
                stop: stop_handler.as_deref(),
            },
            conflicts.len(),
            inputs.len(),
        );
    }

    (merged, conflicts)
}

struct DecisionHandlers<'a> {
    permission: Option<&'a str>,
    block: Option<&'a str>,
    stop: Option<&'a str>,
}

/// Emit the merged policy decision. Observability only: the merge result
/// does not depend on it. Outcomes that change execution (block, stop, deny,
/// ask) log at info; everything else at debug. Reasons are hook-authored
/// text and only appear in a separate debug event; rewritten inputs/outputs
/// and payloads are never logged.
fn log_decision(
    merged: &HookMergedOutcome,
    handlers: DecisionHandlers<'_>,
    conflicts: usize,
    handler_count: usize,
) {
    let blocked = merged.block_reason.is_some();
    let stopped = merged.stop_reason.is_some();
    let permission_decision = merged.permission_decision.map_or("none", permission_label);
    let input_rewritten = merged.updated_input.is_some();
    let output_rewritten = merged.updated_output.is_some();
    let notable = blocked
        || stopped
        || matches!(
            merged.permission_decision,
            Some(PermissionDecision::Deny | PermissionDecision::Ask)
        );
    macro_rules! decision_event {
        ($level:ident) => {
            tracing::$level!(
                permission_decision,
                permission_handler = handlers.permission,
                blocked,
                block_handler = handlers.block,
                stopped,
                stop_handler = handlers.stop,
                input_rewritten,
                output_rewritten,
                conflicts,
                handler_count,
                "hooks.decision"
            )
        };
    }
    if notable {
        decision_event!(info);
    } else {
        decision_event!(debug);
    }
    if blocked || stopped || merged.permission_decision_reason.is_some() {
        tracing::debug!(
            block_reason = merged.block_reason.as_deref(),
            stop_reason = merged.stop_reason.as_deref(),
            permission_decision_reason = merged.permission_decision_reason.as_deref(),
            "hooks.decision_reasons"
        );
    }
}

fn permission_label(decision: PermissionDecision) -> &'static str {
    match decision {
        PermissionDecision::Passthrough => "passthrough",
        PermissionDecision::Allow => "allow",
        PermissionDecision::Ask => "ask",
        PermissionDecision::Deny => "deny",
    }
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
        assert_eq!(format!("{}", ConflictField::UpdatedInput), "updated_input");
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

        assert_eq!(
            merged.updated_input,
            Some(json!({"command": "echo input-winner"}))
        );
        assert_eq!(
            merged.updated_output,
            Some(json!({"result": "output-winner"}))
        );
        assert_eq!(conflicts.len(), 2);
        assert_eq!(conflicts[0].field, ConflictField::UpdatedInput);
        assert_eq!(conflicts[0].winner, "input-winner");
        assert_eq!(conflicts[0].loser, "input-loser");
        assert_eq!(conflicts[1].field, ConflictField::UpdatedOutput);
        assert_eq!(conflicts[1].winner, "output-winner");
        assert_eq!(conflicts[1].loser, "output-loser");
    }

    // --- tracing events ---

    mod decision_events {
        use std::collections::BTreeMap;
        use std::fmt;
        use std::sync::{Arc, Mutex};

        use tracing::field::{Field, Visit};
        use tracing::subscriber::Interest;
        use tracing::{Event, Level, Metadata, Subscriber};
        use tracing_subscriber::layer::{Context, Layer, SubscriberExt};

        use super::*;

        #[derive(Debug, Clone)]
        struct CapturedEvent {
            level: Level,
            message: String,
            fields: BTreeMap<String, String>,
        }

        #[derive(Clone, Default)]
        struct Capture(Arc<Mutex<Vec<CapturedEvent>>>);

        #[derive(Default)]
        struct FieldVisitor(BTreeMap<String, String>);

        impl Visit for FieldVisitor {
            fn record_str(&mut self, field: &Field, value: &str) {
                self.0.insert(field.name().to_owned(), value.to_owned());
            }

            fn record_debug(&mut self, field: &Field, value: &dyn fmt::Debug) {
                self.0.insert(field.name().to_owned(), format!("{value:?}"));
            }
        }

        impl<S: Subscriber> Layer<S> for Capture {
            fn register_callsite(&self, _metadata: &'static Metadata<'static>) -> Interest {
                Interest::sometimes()
            }

            fn enabled(&self, _metadata: &Metadata<'_>, _ctx: Context<'_, S>) -> bool {
                true
            }

            fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
                let mut visitor = FieldVisitor::default();
                event.record(&mut visitor);
                let mut fields = visitor.0;
                let message = fields.remove("message").unwrap_or_default();
                self.0.lock().expect("events").push(CapturedEvent {
                    level: *event.metadata().level(),
                    message,
                    fields,
                });
            }
        }

        fn decisions(inputs: &[MergeInput]) -> Vec<CapturedEvent> {
            // With exactly one live dispatcher, tracing-core computes a new
            // callsite's interest from the registering thread's default
            // only, so a parallel test thread without a subscriber could
            // cache `never`. A permanent second dispatcher prevents that.
            static KEEPALIVE: std::sync::OnceLock<tracing::Dispatch> = std::sync::OnceLock::new();
            KEEPALIVE.get_or_init(|| {
                tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default())
            });
            let capture = Capture::default();
            let subscriber = tracing_subscriber::registry().with(capture.clone());
            tracing::subscriber::with_default(subscriber, || {
                merge_outputs(inputs);
            });
            let events = capture.0.lock().expect("events").clone();
            events
                .into_iter()
                .filter(|event| event.message == "hooks.decision")
                .collect()
        }

        fn permission_input(handler_id: &str, decision: PermissionDecision) -> MergeInput {
            merge_input(
                handler_id,
                priority(HandlerPriorityGroup::PluginFiles, 0, 0, 0, 0),
                HookOutput {
                    hook_specific_output: Some(HookSpecificOutput {
                        permission_decision: Some(decision),
                        permission_decision_reason: Some("secret reason text".to_owned()),
                        ..HookSpecificOutput::default()
                    }),
                    ..HookOutput::default()
                },
            )
        }

        #[test]
        fn deny_decision_logs_info_without_reasons() {
            let events = decisions(&[permission_input("deny-hook", PermissionDecision::Deny)]);
            assert_eq!(events.len(), 1, "{events:?}");
            let event = &events[0];
            assert_eq!(event.level, Level::INFO);
            assert_eq!(
                event.fields.get("permission_decision").map(String::as_str),
                Some("deny")
            );
            assert_eq!(
                event.fields.get("permission_handler").map(String::as_str),
                Some("deny-hook")
            );
            assert_eq!(
                event.fields.get("blocked").map(String::as_str),
                Some("true")
            );
            assert_eq!(
                event.fields.get("block_handler").map(String::as_str),
                Some("deny-hook")
            );
            assert!(
                event.fields.keys().all(|key| !key.contains("reason")),
                "{event:?}"
            );
            assert!(
                event
                    .fields
                    .values()
                    .all(|value| !value.contains("secret reason text")),
                "{event:?}"
            );
        }

        #[test]
        fn allow_decision_logs_debug() {
            let events = decisions(&[permission_input("allow-hook", PermissionDecision::Allow)]);
            assert_eq!(events.len(), 1, "{events:?}");
            assert_eq!(events[0].level, Level::DEBUG);
            assert_eq!(
                events[0]
                    .fields
                    .get("permission_decision")
                    .map(String::as_str),
                Some("allow")
            );
        }

        #[test]
        fn empty_inputs_log_nothing() {
            assert!(decisions(&[]).is_empty());
        }
    }
}
