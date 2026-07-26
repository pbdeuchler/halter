// pattern: Functional Core

use std::collections::{BTreeMap, BTreeSet};

use halter_protocol::{
    AssistantPart, CompactedContext, CompactionWindow, Message, MessageSignal, PromptSegment,
    PruneSignalThreshold, SummarySlice, ToolCall, ToolCallId, ToolName, ToolResult,
    ToolResultMessage,
};
use serde_json::Value;

/// Token buffer applied before triggering automatic compaction.
pub const COMPACTION_TRIGGER_BUFFER: u64 = 100;

#[derive(Debug, Clone, Copy)]
/// Token thresholds used by context planning.
pub struct ContextSettings {
    pub compaction_threshold: u64,
    pub pre_compaction_target: u64,
    pub prune_signal_threshold: PruneSignalThreshold,
}

impl Default for ContextSettings {
    fn default() -> Self {
        Self {
            compaction_threshold: 80_000,
            pre_compaction_target: 60_000,
            prune_signal_threshold: PruneSignalThreshold::Normal,
        }
    }
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
/// Whether the estimated context is close enough to trigger compaction.
pub fn should_trigger_compaction(estimated_tokens: u64, settings: &ContextSettings) -> bool {
    estimated_tokens.saturating_add(COMPACTION_TRIGGER_BUFFER) >= settings.compaction_threshold
}

#[must_use]
/// Select messages to compact after pruning low-signal units.
pub fn prepare_compaction(
    settings: &ContextSettings,
    compacted_context: &CompactedContext,
    window: CompactionWindow,
) -> CompactionPreparation {
    let units = build_compaction_units(&window.eligible_messages);
    let compacted_prefix_tokens = estimate_compacted_context_tokens(compacted_context);
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

/// Estimate total context tokens for prompt, summaries, compacted prefix, and
/// messages.
///
/// Prefers ground truth over the character heuristic. Providers report the
/// exact context size with every assistant response, so when the transcript
/// contains a usable report this anchors on it and estimates only the
/// messages that arrived afterwards — bounding heuristic error to one turn's
/// tail instead of letting it compound across the whole transcript. The
/// reported figure already accounts for the prompt, summaries, and compacted
/// prefix, so those are *not* added on top of an anchor.
///
/// Falls back to estimating everything when no usable report exists: a fresh
/// session, a transcript whose assistant turns all failed, or one whose
/// reports predate the last compaction (see
/// [`SessionState::usage_anchor_floor`](halter_protocol::SessionState)).
#[must_use]
pub fn estimate_context_tokens(
    prompt_segments: &[PromptSegment],
    summaries: &[SummarySlice],
    compacted_prefix: &[Value],
    messages: &[Message],
    usage_anchor_floor: usize,
) -> u64 {
    match find_usage_anchor(messages, usage_anchor_floor) {
        Some(anchor) => anchor
            .context_tokens
            .saturating_add(estimate_messages_tokens(&messages[anchor.index + 1..])),
        None => {
            estimate_segment_tokens(prompt_segments)
                + estimate_summary_tokens(summaries)
                + estimate_compacted_prefix_tokens(compacted_prefix)
                + estimate_messages_tokens(messages)
        }
    }
}

/// A provider-reported context measurement usable as an estimation anchor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UsageAnchor {
    /// Index in `messages` of the assistant message that reported it.
    pub index: usize,
    /// Context size the provider reported at that point.
    pub context_tokens: u64,
}

/// Find the most recent assistant message at or after `floor` carrying a
/// usable context report.
///
/// A report is usable only if the turn actually completed: interrupted and
/// errored turns can report partial or zero usage, which would anchor the
/// whole estimate to a figure far below the real context.
#[must_use]
pub fn find_usage_anchor(messages: &[Message], floor: usize) -> Option<UsageAnchor> {
    messages
        .iter()
        .enumerate()
        .skip(floor)
        .rev()
        .find_map(|(index, message)| {
            let Message::Assistant(assistant) = message else {
                return None;
            };
            if matches!(
                assistant.stop_reason,
                Some(halter_protocol::StopReason::Interrupted | halter_protocol::StopReason::Error)
            ) {
                return None;
            }
            let context_tokens = assistant.usage.as_ref()?.context_tokens();
            (context_tokens > 0).then_some(UsageAnchor {
                index,
                context_tokens,
            })
        })
}

#[must_use]
/// Estimate tokens for prompt segments.
pub fn estimate_segment_tokens(segments: &[PromptSegment]) -> u64 {
    segments
        .iter()
        .map(|segment| estimate_text_tokens(&segment.text))
        .sum()
}

#[must_use]
/// Estimate tokens for carried summaries.
pub fn estimate_summary_tokens(summaries: &[SummarySlice]) -> u64 {
    summaries
        .iter()
        .map(|summary| estimate_text_tokens(&summary.text))
        .sum()
}

#[must_use]
/// Estimate tokens for provider-native compacted prefix items.
pub fn estimate_compacted_prefix_tokens(compacted_prefix: &[Value]) -> u64 {
    compacted_prefix.iter().map(estimate_json_tokens).sum()
}

#[must_use]
/// Estimate tokens for a compacted context wrapper.
pub fn estimate_compacted_context_tokens(compacted_context: &CompactedContext) -> u64 {
    estimate_compacted_prefix_tokens(compacted_context.items())
}

#[must_use]
/// Estimate tokens for a transcript slice.
pub fn estimate_messages_tokens(messages: &[Message]) -> u64 {
    messages.iter().map(estimate_message_tokens).sum()
}

#[must_use]
/// Estimate tokens for one transcript message.
pub fn estimate_message_tokens(message: &Message) -> u64 {
    match message {
        Message::System(message) => estimate_text_tokens(&message.text),
        Message::User(message) => estimate_text_tokens(&message.plain_text()),
        Message::Assistant(message) => message
            .parts
            .iter()
            .map(|part| match part {
                AssistantPart::Text { text } => estimate_text_tokens(text),
                AssistantPart::Thinking(block) => estimate_text_tokens(&block.text),
                AssistantPart::ToolCall(call) => {
                    estimate_text_tokens(&call.name.0)
                        + estimate_json_tokens(&call.arguments)
                        + estimate_text_tokens("tool_call")
                }
            })
            .sum(),
        Message::Tool(message) => match &message.content {
            ToolResult::Empty => 0,
            ToolResult::Text { text } => estimate_text_tokens(text),
            ToolResult::Json { value } => estimate_json_tokens(value),
        },
    }
}

/// Estimates the number of tokens consumed by `text` for budgeting purposes.
///
/// Uses a char-to-token ratio of 10/37 (≈3.7 chars per token), which averages
/// English prose across current BPE tokenizers (cl100k_base, o200k_base, GPT
/// tokenizers). This is intentionally *heuristic*: it runs in O(chars) and
/// avoids loading tokenizer tables, at the cost of being wrong by ±20% for
/// code-heavy or non-Latin text.
///
/// The compaction loop uses this solely for triggering thresholds, not for
/// billing, so the approximation is acceptable. If a per-provider tokenizer
/// becomes available, implement [`TokenEstimator`] and thread an alternative
/// estimator through the context manager.
#[must_use]
pub fn estimate_text_tokens(text: &str) -> u64 {
    CharHeuristicEstimator.estimate(text)
}

/// A pluggable token-budget estimator. Implementors may swap in a
/// model-specific tokenizer when one becomes available; today only
/// [`CharHeuristicEstimator`] is provided.
pub trait TokenEstimator {
    fn estimate(&self, text: &str) -> u64;
}

/// Default estimator: `floor(chars * 10 / 37)` — see
/// [`estimate_text_tokens`] for the rationale and caveats.
#[derive(Debug, Clone, Copy, Default)]
pub struct CharHeuristicEstimator;

impl TokenEstimator for CharHeuristicEstimator {
    fn estimate(&self, text: &str) -> u64 {
        let chars = text.chars().count() as u64;
        (chars * 10) / 37
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

fn estimate_json_tokens(value: &Value) -> u64 {
    estimate_text_tokens(&stable_json(value))
}

pub(crate) fn stable_json(value: &Value) -> String {
    match value {
        Value::Object(map) => {
            let sorted = map.iter().collect::<BTreeMap<_, _>>();
            let body = sorted
                .into_iter()
                .map(|(key, value)| format!("\"{key}\":{}", stable_json(value)))
                .collect::<Vec<_>>()
                .join(",");
            format!("{{{body}}}")
        }
        Value::Array(values) => format!(
            "[{}]",
            values.iter().map(stable_json).collect::<Vec<_>>().join(",")
        ),
        Value::String(text) => serde_json::to_string(text).unwrap_or_else(|_| "\"\"".to_owned()),
        Value::Bool(boolean) => boolean.to_string(),
        Value::Null => "null".to_owned(),
        Value::Number(number) => number.to_string(),
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

    fn assistant(
        text: &str,
        stop_reason: Option<halter_protocol::StopReason>,
        usage: Option<halter_protocol::Usage>,
    ) -> Message {
        Message::Assistant(AssistantMessage {
            id: MessageId::new(),
            created_at: Utc::now(),
            parts: vec![AssistantPart::Text {
                text: text.to_owned(),
            }],
            stop_reason,
            usage,
            replay_meta: Default::default(),
        })
    }

    fn reported(input_tokens: u64, output_tokens: u64) -> Option<halter_protocol::Usage> {
        Some(halter_protocol::Usage {
            input_tokens,
            output_tokens,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        })
    }

    /// The whole point of anchoring: a provider report replaces the heuristic
    /// for everything up to and including the reporting message, and the
    /// prompt/summaries/prefix are not re-added on top because the report
    /// already covered them.
    #[test]
    fn estimate_context_tokens_anchors_on_reported_usage() {
        let messages = vec![
            Message::User(UserMessage::text("x".repeat(4_000).as_str())),
            assistant(
                "done",
                Some(halter_protocol::StopReason::EndTurn),
                reported(50_000, 200),
            ),
            Message::User(UserMessage::text("follow up")),
        ];
        let summaries = vec![SummarySlice {
            id: "s".to_owned(),
            text: "y".repeat(4_000),
        }];

        let estimated = estimate_context_tokens(&[], &summaries, &[], &messages, 0);

        // 50_000 + 200 reported, plus only the trailing user message.
        assert_eq!(
            estimated,
            50_200 + estimate_message_tokens(&messages[2]),
            "anchor must exclude everything the report already covered"
        );
    }

    #[test]
    fn estimate_context_tokens_falls_back_to_heuristic_without_usage() {
        let messages = vec![
            Message::User(UserMessage::text("hello")),
            assistant("done", Some(halter_protocol::StopReason::EndTurn), None),
        ];

        let estimated = estimate_context_tokens(&[], &[], &[], &messages, 0);

        assert_eq!(estimated, estimate_messages_tokens(&messages));
    }

    /// Interrupted and errored turns report partial or zero usage. Anchoring
    /// on those would peg the estimate far below the real context and stop
    /// compaction from ever firing.
    #[test]
    fn find_usage_anchor_rejects_unusable_reports() {
        struct Case {
            name: &'static str,
            message: Message,
        }

        let cases = [
            Case {
                name: "interrupted",
                message: assistant(
                    "partial",
                    Some(halter_protocol::StopReason::Interrupted),
                    reported(50_000, 10),
                ),
            },
            Case {
                name: "error",
                message: assistant(
                    "boom",
                    Some(halter_protocol::StopReason::Error),
                    reported(50_000, 10),
                ),
            },
            Case {
                name: "zero usage",
                message: assistant(
                    "empty",
                    Some(halter_protocol::StopReason::EndTurn),
                    reported(0, 0),
                ),
            },
            Case {
                name: "no usage",
                message: assistant("none", Some(halter_protocol::StopReason::EndTurn), None),
            },
            Case {
                name: "not an assistant message",
                message: Message::User(UserMessage::text("user")),
            },
        ];

        for case in cases {
            assert_eq!(
                find_usage_anchor(std::slice::from_ref(&case.message), 0),
                None,
                "{}: must not anchor",
                case.name
            );
        }
    }

    #[test]
    fn find_usage_anchor_picks_the_most_recent_usable_report() {
        let messages = vec![
            assistant(
                "first",
                Some(halter_protocol::StopReason::EndTurn),
                reported(10, 1),
            ),
            assistant(
                "second",
                Some(halter_protocol::StopReason::EndTurn),
                reported(20, 2),
            ),
            assistant(
                "third",
                Some(halter_protocol::StopReason::Error),
                reported(999, 0),
            ),
        ];

        let anchor = find_usage_anchor(&messages, 0).expect("anchor");

        assert_eq!(anchor.index, 1);
        assert_eq!(anchor.context_tokens, 22);
    }

    /// Compaction preserves a message tail whose usage describes the
    /// pre-compaction context. Without the floor, that stale figure would
    /// re-trigger compaction on every following turn.
    #[test]
    fn find_usage_anchor_ignores_reports_below_the_floor() {
        let messages = vec![
            assistant(
                "pre-compaction",
                Some(halter_protocol::StopReason::EndTurn),
                reported(80_000, 500),
            ),
            Message::User(UserMessage::text("after compaction")),
        ];

        assert_eq!(find_usage_anchor(&messages, 1), None);

        // ...and the estimate falls back to the heuristic rather than 80_500.
        let estimated = estimate_context_tokens(&[], &[], &[], &messages, 1);
        assert_eq!(estimated, estimate_messages_tokens(&messages));
        assert!(estimated < 1_000);
    }

    #[test]
    fn estimate_text_tokens_uses_character_count() {
        assert_eq!(estimate_text_tokens(""), 0);
        assert_eq!(estimate_text_tokens("abcd"), 1);
        assert_eq!(estimate_text_tokens("abcdefghij"), 2);
    }

    #[test]
    fn compacted_context_estimation_uses_stable_json_rendering() {
        let context = CompactedContext::new(vec![json!({
            "b": true,
            "a": "abcd",
        })]);

        let rendered = stable_json(&context.items()[0]);

        assert_eq!(rendered, "{\"a\":\"abcd\",\"b\":true}");
        assert_eq!(
            estimate_compacted_context_tokens(&context),
            estimate_text_tokens(&rendered)
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
