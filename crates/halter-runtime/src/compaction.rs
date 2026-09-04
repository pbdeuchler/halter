// pattern: Functional Core

use std::collections::{BTreeMap, BTreeSet};

use halter_protocol::{
    AssistantPart, CompactedContext, CompactionWindow, Message, MessageSignal,
    PruneSignalThreshold, ToolCall, ToolCallId, ToolName, ToolResult, ToolResultMessage,
    estimate_compacted_prefix_tokens, estimate_message_tokens,
};

/// Token buffer applied before triggering automatic compaction.
pub const COMPACTION_TRIGGER_BUFFER: u64 = 100;

#[derive(Debug, Clone, Copy)]
/// Token thresholds that decide *when* the runtime compacts and how much the
/// provider-delegated strategy prunes first.
pub struct ContextSettings {
    /// Compact once the ledger's effective count (plus
    /// [`COMPACTION_TRIGGER_BUFFER`]) reaches this many tokens.
    pub compaction_threshold: u64,
    /// Evict low-signal history until the projected prefix is below this
    /// target before asking the provider to compact.
    pub pre_compaction_target: u64,
    /// Highest signal tier eligible for pre-compaction eviction.
    pub prune_signal_threshold: PruneSignalThreshold,
    /// Hard cap on the effective count. Checked after compaction has had its
    /// chance, before every provider request; exceeding it fails the turn
    /// with [`ContextCapExceeded`] instead of blowing the provider's window.
    pub max_tokens: Option<u64>,
}

impl Default for ContextSettings {
    fn default() -> Self {
        Self {
            compaction_threshold: 80_000,
            pre_compaction_target: 60_000,
            prune_signal_threshold: PruneSignalThreshold::Normal,
            max_tokens: None,
        }
    }
}

impl ContextSettings {
    /// Whether the effective ledger count is close enough to trigger
    /// compaction.
    #[must_use]
    pub fn compaction_due(&self, effective_tokens: u64) -> bool {
        effective_tokens.saturating_add(COMPACTION_TRIGGER_BUFFER) >= self.compaction_threshold
    }

    /// Enforce the hard cap on the effective ledger count.
    pub fn check_cap(&self, effective_tokens: u64) -> Result<(), ContextCapExceeded> {
        match self.max_tokens {
            Some(max_tokens) if effective_tokens > max_tokens => Err(ContextCapExceeded {
                effective_tokens,
                max_tokens,
            }),
            _ => Ok(()),
        }
    }
}

/// The session context grew past `context.max_tokens` and compaction could
/// not bring it back under. Raised before the provider request that would
/// have carried the oversized context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error(
    "failed to run turn: session context of {effective_tokens} tokens exceeds context.max_tokens ({max_tokens})"
)]
pub struct ContextCapExceeded {
    pub effective_tokens: u64,
    pub max_tokens: u64,
}

#[derive(Debug, Clone)]
/// Messages selected for a provider compaction request.
pub struct CompactionPreparation {
    pub compact_messages: Vec<Message>,
    pub preserved_messages: Vec<Message>,
    pub reserved_response_block: bool,
    pub compacted_message_count: usize,
    pub evicted_unit_count: usize,
}

#[derive(Debug, Clone)]
struct CompactionUnit {
    order: usize,
    message_indices: Vec<usize>,
    signal: MessageSignal,
    estimated_tokens: u64,
}

#[must_use]
/// Select messages to compact after pruning low-signal units.
pub fn prepare_compaction(
    settings: &ContextSettings,
    compacted_context: &CompactedContext,
    window: CompactionWindow,
) -> CompactionPreparation {
    let units = build_compaction_units(&window.eligible_messages);
    let compacted_prefix_tokens = estimate_compacted_prefix_tokens(compacted_context.items());
    let retained_units = prune_units(settings, compacted_prefix_tokens, &units);
    let retained_indices = retained_units
        .iter()
        .flat_map(|unit| unit.message_indices.iter().copied())
        .collect::<BTreeSet<_>>();
    let compact_messages = window
        .eligible_messages
        .iter()
        .enumerate()
        .filter(|(index, _)| retained_indices.contains(index))
        .map(|(_, message)| message.clone())
        .collect::<Vec<_>>();
    let compacted_message_count = compact_messages.len();

    CompactionPreparation {
        compact_messages,
        preserved_messages: window.preserved_messages,
        reserved_response_block: window.reserved_response_block,
        compacted_message_count,
        evicted_unit_count: units.len().saturating_sub(retained_units.len()),
    }
}

#[must_use]
/// Score a message for context pruning.
pub fn score_message(message: &Message) -> MessageSignal {
    match message {
        Message::User(_) => MessageSignal::Anchor,
        Message::System(_) => MessageSignal::High,
        Message::Assistant(message) => assistant_signal(message.parts.as_slice()),
        Message::Tool(message) => score_tool_result(message, None),
    }
}

#[must_use]
/// Human-readable summary for a compaction event.
pub fn render_compaction_event_summary(
    compacted_message_count: usize,
    compacted_item_count: usize,
    evicted_unit_count: usize,
    reserved_response_block: bool,
) -> String {
    format!(
        "Compacted {compacted_message_count} older messages into {compacted_item_count} compact items after evicting {evicted_unit_count} low-signal units; reserved {} latest response block.",
        usize::from(reserved_response_block)
    )
}

/// Iteratively drop the lowest-signal, oldest units until the projected token
/// count drops below `pre_compaction_target`. Replaces the prior bulk-tier
/// eviction, which could overshoot the target by up to ~20× (finding H10):
/// the previous strategy dropped *every* unit of the lowest admissible tier
/// in one pass, even when removing two or three units would have sufficed.
///
/// Eviction order is `(signal ascending, order ascending)`: drop
/// `VeryLow` before `Low`, `Low` before `Normal`, and oldest first within a
/// tier. Units at signals above `prune_signal_threshold` are retained
/// unconditionally — they represent the floor the operator told us not to
/// breach.
fn prune_units(
    settings: &ContextSettings,
    compacted_prefix_tokens: u64,
    units: &[CompactionUnit],
) -> Vec<CompactionUnit> {
    if units.is_empty() {
        return Vec::new();
    }

    let mut retained = units.to_vec();
    if remaining_tokens(&retained, compacted_prefix_tokens) <= settings.pre_compaction_target {
        return retained;
    }

    // Build a per-retained candidate list, ordered from most-droppable to
    // least. Within the allowed threshold, lower-signal (and then older)
    // units go first.
    let mut candidate_orders: Vec<usize> = retained
        .iter()
        .filter(|unit| threshold_allows_signal(settings.prune_signal_threshold, unit.signal))
        .map(|unit| unit.order)
        .collect();
    candidate_orders.sort_by_key(|order| {
        let unit = retained
            .iter()
            .find(|candidate| candidate.order == *order)
            .expect("candidate order references a retained unit");
        (unit.signal, unit.order)
    });

    for order in candidate_orders {
        if remaining_tokens(&retained, compacted_prefix_tokens) <= settings.pre_compaction_target {
            break;
        }
        retained.retain(|unit| unit.order != order);
    }

    retained.sort_by_key(|unit| unit.order);
    retained
}

fn remaining_tokens(units: &[CompactionUnit], compacted_prefix_tokens: u64) -> u64 {
    compacted_prefix_tokens + units.iter().map(|unit| unit.estimated_tokens).sum::<u64>()
}

fn threshold_allows_signal(threshold: PruneSignalThreshold, signal: MessageSignal) -> bool {
    match threshold {
        PruneSignalThreshold::VeryLow => signal == MessageSignal::VeryLow,
        PruneSignalThreshold::Low => signal <= MessageSignal::Low,
        PruneSignalThreshold::Normal => signal <= MessageSignal::Normal,
        PruneSignalThreshold::High => signal <= MessageSignal::High,
    }
}

fn build_compaction_units(messages: &[Message]) -> Vec<CompactionUnit> {
    let mut units = Vec::new();
    let mut index = 0usize;

    while index < messages.len() {
        match &messages[index] {
            Message::Assistant(message) => {
                let tool_calls = assistant_tool_calls(message.parts.as_slice());
                let mut message_indices = vec![index];
                let mut signal = assistant_signal(message.parts.as_slice());
                let mut estimated_tokens = estimate_message_tokens(&messages[index]);
                let mut scan_index = index + 1;

                if !tool_calls.is_empty() {
                    while scan_index < messages.len() {
                        match &messages[scan_index] {
                            Message::Tool(tool) if tool_calls.contains_key(&tool.call_id) => {
                                signal = signal
                                    .max(score_tool_result(tool, tool_calls.get(&tool.call_id)));
                                estimated_tokens += estimate_message_tokens(&messages[scan_index]);
                                message_indices.push(scan_index);
                                scan_index += 1;
                            }
                            _ => break,
                        }
                    }
                }

                units.push(CompactionUnit {
                    order: units.len(),
                    message_indices,
                    signal,
                    estimated_tokens,
                });
                index = scan_index;
            }
            message => {
                units.push(CompactionUnit {
                    order: units.len(),
                    message_indices: vec![index],
                    signal: score_message(message),
                    estimated_tokens: estimate_message_tokens(message),
                });
                index += 1;
            }
        }
    }

    units
}

fn assistant_tool_calls(parts: &[AssistantPart]) -> BTreeMap<ToolCallId, ToolName> {
    parts
        .iter()
        .filter_map(|part| match part {
            AssistantPart::ToolCall(ToolCall { id, name, .. }) => Some((id.clone(), name.clone())),
            AssistantPart::Text { .. } | AssistantPart::Thinking(_) => None,
        })
        .collect()
}

fn assistant_signal(parts: &[AssistantPart]) -> MessageSignal {
    let has_text = parts.iter().any(|part| match part {
        AssistantPart::Text { text } => !text.is_empty(),
        AssistantPart::Thinking(block) => !block.text.is_empty(),
        AssistantPart::ToolCall(_) => false,
    });

    if has_text {
        MessageSignal::VeryHigh
    } else {
        MessageSignal::Normal
    }
}

fn score_tool_result(message: &ToolResultMessage, tool_name: Option<&ToolName>) -> MessageSignal {
    if message.error.is_some() {
        return MessageSignal::Low;
    }

    match &message.content {
        ToolResult::Empty => MessageSignal::VeryLow,
        ToolResult::Text { .. } | ToolResult::Json { .. } => {
            if tool_name.is_some_and(|name| name.0 == "read") {
                MessageSignal::High
            } else {
                MessageSignal::Normal
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use halter_protocol::{
        AssistantMessage, CompactedContext, CompactionWindow, MessageId, ToolError, UserMessage,
    };
    use serde_json::json;

    use super::*;

    #[test]
    fn compaction_due_applies_the_trigger_buffer() {
        let settings = ContextSettings {
            compaction_threshold: 1_000,
            ..ContextSettings::default()
        };
        struct Case {
            effective_tokens: u64,
            due: bool,
        }
        let cases = [
            Case {
                effective_tokens: 0,
                due: false,
            },
            Case {
                effective_tokens: 1_000 - COMPACTION_TRIGGER_BUFFER - 1,
                due: false,
            },
            Case {
                effective_tokens: 1_000 - COMPACTION_TRIGGER_BUFFER,
                due: true,
            },
            Case {
                effective_tokens: u64::MAX,
                due: true,
            },
        ];
        for case in cases {
            assert_eq!(
                settings.compaction_due(case.effective_tokens),
                case.due,
                "{} tokens",
                case.effective_tokens
            );
        }
    }

    #[test]
    fn check_cap_fails_only_past_a_configured_cap() {
        let uncapped = ContextSettings::default();
        assert_eq!(uncapped.check_cap(u64::MAX), Ok(()));

        let capped = ContextSettings {
            max_tokens: Some(500),
            ..ContextSettings::default()
        };
        assert_eq!(capped.check_cap(500), Ok(()));
        let error = capped.check_cap(501).expect_err("over cap");
        assert_eq!(
            error,
            ContextCapExceeded {
                effective_tokens: 501,
                max_tokens: 500,
            }
        );
        assert!(
            error
                .to_string()
                .contains("501 tokens exceeds context.max_tokens (500)")
        );
    }

    #[test]
    fn compaction_window_reserves_latest_assistant_block_and_tail() {
        let messages = vec![
            Message::User(UserMessage::text("first")),
            Message::Assistant(AssistantMessage {
                id: MessageId::new(),
                created_at: Utc::now(),
                parts: vec![AssistantPart::Text {
                    text: "answer".to_owned(),
                }],
                stop_reason: None,
                usage: None,
                replay_meta: Default::default(),
            }),
            Message::User(UserMessage::text("follow up")),
        ];

        let reserved = CompactionWindow::preserve_latest_assistant_response_block(&messages);

        assert_eq!(reserved.eligible_messages.len(), 1);
        assert_eq!(reserved.preserved_messages.len(), 2);
        assert!(reserved.reserved_response_block);
    }

    #[test]
    fn score_message_marks_assistant_text_as_very_high() {
        let message = Message::Assistant(AssistantMessage {
            id: MessageId::new(),
            created_at: Utc::now(),
            parts: vec![AssistantPart::Text {
                text: "done".to_owned(),
            }],
            stop_reason: None,
            usage: None,
            replay_meta: Default::default(),
        });

        assert_eq!(score_message(&message), MessageSignal::VeryHigh);
    }

    #[test]
    fn prepare_compaction_evicts_low_signal_units_before_high() {
        let tool_call_id = ToolCallId::from("call_1");
        let messages = vec![
            Message::Assistant(AssistantMessage {
                id: MessageId::new(),
                created_at: Utc::now(),
                parts: vec![AssistantPart::ToolCall(ToolCall {
                    id: tool_call_id.clone(),
                    name: "write".into(),
                    arguments: json!({"path": "a.txt"}),
                })],
                stop_reason: None,
                usage: None,
                replay_meta: Default::default(),
            }),
            Message::Tool(ToolResultMessage {
                id: MessageId::new(),
                call_id: tool_call_id,
                content: ToolResult::Empty,
                error: None,
                created_at: Utc::now(),
            }),
            Message::Assistant(AssistantMessage {
                id: MessageId::new(),
                created_at: Utc::now(),
                parts: vec![AssistantPart::Text {
                    text: "keep me".repeat(64),
                }],
                stop_reason: None,
                usage: None,
                replay_meta: Default::default(),
            }),
        ];

        let preparation = prepare_compaction(
            &ContextSettings {
                compaction_threshold: 100,
                pre_compaction_target: 1,
                prune_signal_threshold: PruneSignalThreshold::Normal,
                max_tokens: None,
            },
            &CompactedContext::default(),
            CompactionWindow::preserve_latest_assistant_response_block(&messages),
        );

        assert!(preparation.compact_messages.is_empty());
        assert_eq!(preparation.preserved_messages.len(), 1);
    }

    fn build_threshold_fixture() -> Vec<Message> {
        use halter_protocol::SystemMessage;

        let vlow_tool = Message::Tool(ToolResultMessage {
            id: MessageId::new(),
            call_id: ToolCallId::from("vlow"),
            content: ToolResult::Empty,
            error: None,
            created_at: Utc::now(),
        });
        let low_tool = Message::Tool(ToolResultMessage {
            id: MessageId::new(),
            call_id: ToolCallId::from("low"),
            content: ToolResult::Text {
                text: "boom".repeat(32),
            },
            error: Some(ToolError {
                message: "failed".to_owned(),
            }),
            created_at: Utc::now(),
        });
        let normal_tool = Message::Tool(ToolResultMessage {
            id: MessageId::new(),
            call_id: ToolCallId::from("normal"),
            content: ToolResult::Text {
                text: "hit".repeat(64),
            },
            error: None,
            created_at: Utc::now(),
        });
        let high_system = Message::System(SystemMessage {
            id: MessageId::new(),
            created_at: Utc::now(),
            text: "sys".repeat(64),
        });
        let trailing_assistant = Message::Assistant(AssistantMessage {
            id: MessageId::new(),
            created_at: Utc::now(),
            parts: vec![AssistantPart::Text {
                text: "keep me".repeat(64),
            }],
            stop_reason: None,
            usage: None,
            replay_meta: Default::default(),
        });

        vec![
            vlow_tool,
            low_tool,
            normal_tool,
            high_system,
            trailing_assistant,
        ]
    }

    #[test]
    fn prune_threshold_preserves_signals_above_its_ceiling() {
        struct Case {
            threshold: PruneSignalThreshold,
            retained: &'static [MessageSignal],
        }

        let cases = [
            Case {
                threshold: PruneSignalThreshold::VeryLow,
                retained: &[
                    MessageSignal::Low,
                    MessageSignal::Normal,
                    MessageSignal::High,
                ],
            },
            Case {
                threshold: PruneSignalThreshold::Low,
                retained: &[MessageSignal::Normal, MessageSignal::High],
            },
            Case {
                threshold: PruneSignalThreshold::Normal,
                retained: &[MessageSignal::High],
            },
            Case {
                threshold: PruneSignalThreshold::High,
                retained: &[],
            },
        ];

        for case in cases {
            let messages = build_threshold_fixture();
            let preparation = prepare_compaction(
                &ContextSettings {
                    compaction_threshold: 100,
                    pre_compaction_target: 1,
                    prune_signal_threshold: case.threshold,
                    max_tokens: None,
                },
                &CompactedContext::default(),
                CompactionWindow::preserve_latest_assistant_response_block(&messages),
            );

            let surviving: Vec<MessageSignal> = preparation
                .compact_messages
                .iter()
                .map(score_message)
                .collect();

            assert_eq!(
                surviving, case.retained,
                "threshold {:?}: expected {:?}, got {:?}",
                case.threshold, case.retained, surviving
            );
        }
    }

    #[test]
    fn failed_tool_results_score_low() {
        let message = Message::Tool(ToolResultMessage {
            id: MessageId::new(),
            call_id: ToolCallId::from("call_1"),
            content: ToolResult::Text {
                text: "boom".to_owned(),
            },
            error: Some(ToolError {
                message: "failed".to_owned(),
            }),
            created_at: Utc::now(),
        });

        assert_eq!(score_message(&message), MessageSignal::Low);
    }

    #[test]
    fn split_after_last_user_reserves_through_user_prompt() {
        let tool_call_id = ToolCallId::from("call_1");
        let messages = vec![
            Message::User(UserMessage::text("first user")),
            Message::Assistant(AssistantMessage {
                id: MessageId::new(),
                created_at: Utc::now(),
                parts: vec![AssistantPart::Text {
                    text: "first reply".to_owned(),
                }],
                stop_reason: None,
                usage: None,
                replay_meta: Default::default(),
            }),
            Message::User(UserMessage::text("latest user prompt")),
            Message::Assistant(AssistantMessage {
                id: MessageId::new(),
                created_at: Utc::now(),
                parts: vec![AssistantPart::ToolCall(ToolCall {
                    id: tool_call_id.clone(),
                    name: "read".into(),
                    arguments: json!({}),
                })],
                stop_reason: None,
                usage: None,
                replay_meta: Default::default(),
            }),
            Message::Tool(ToolResultMessage {
                id: MessageId::new(),
                call_id: tool_call_id,
                content: ToolResult::Text {
                    text: "tool out".to_owned(),
                },
                error: None,
                created_at: Utc::now(),
            }),
        ];

        let split = CompactionWindow::preserve_through_latest_user(&messages);
        // Reserved up to and including the most recent user message...
        assert_eq!(split.preserved_messages.len(), 3);
        assert!(matches!(
            split.preserved_messages.last(),
            Some(Message::User(_))
        ));
        // ...and the post-user tail (assistant + tool) is eligible for compaction.
        assert_eq!(split.eligible_messages.len(), 2);
    }

    #[test]
    fn inline_strategy_only_compacts_post_user_tail() {
        let tool_call_id = ToolCallId::from("inline_call");
        let messages = vec![
            Message::User(UserMessage::text("anchor user prompt that survives")),
            Message::Assistant(AssistantMessage {
                id: MessageId::new(),
                created_at: Utc::now(),
                parts: vec![AssistantPart::ToolCall(ToolCall {
                    id: tool_call_id.clone(),
                    name: "shell".into(),
                    arguments: json!({"cmd": "ls"}),
                })],
                stop_reason: None,
                usage: None,
                replay_meta: Default::default(),
            }),
            Message::Tool(ToolResultMessage {
                id: MessageId::new(),
                call_id: tool_call_id,
                content: ToolResult::Text {
                    text: "noise that should be summarized".repeat(32),
                },
                error: None,
                created_at: Utc::now(),
            }),
        ];

        let prepared = prepare_compaction(
            &ContextSettings {
                compaction_threshold: 50,
                pre_compaction_target: 1,
                prune_signal_threshold: PruneSignalThreshold::Normal,
                max_tokens: None,
            },
            &CompactedContext::default(),
            CompactionWindow::preserve_through_latest_user(&messages),
        );

        // Reserved suffix = messages we keep verbatim (the user anchor).
        // Compact messages = post-user tail eligible for in-band summarization.
        assert_eq!(prepared.preserved_messages.len(), 1);
        assert!(matches!(prepared.preserved_messages[0], Message::User(_)));
        assert!(!prepared.reserved_response_block);
    }
}
