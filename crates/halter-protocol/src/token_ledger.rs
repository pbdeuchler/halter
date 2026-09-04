//! Context-size accounting: the session token ledger and the heuristic
//! estimator it falls back on between provider reports.
// pattern: Functional Core

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{AssistantPart, Message, PromptSegment, StopReason, ToolResult};

/// Running estimate of the tokens the provider will see on the next request.
///
/// `authoritative_tokens` is the context size the provider reported with the
/// last *completed* assistant response; `inferred_tokens` is the heuristic
/// estimate of every message appended since. A new authoritative report
/// replaces the anchor and zeroes the inferred component, so heuristic error
/// is bounded to one response's tail instead of compounding across the
/// transcript. Compaction rewrites history the provider has never seen, so it
/// discards the anchor and re-estimates the compacted state
/// ([`TokenLedger::inferred_from`]).
///
/// Prompt segments (system prompt, skills, hook context) are not counted:
/// every authoritative report already covers them, and between reports the
/// ledger tracks transcript growth only.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
pub struct TokenLedger {
    /// Context size reported by the last completed assistant response, or
    /// zero when no usable report covers the current transcript.
    pub authoritative_tokens: u64,
    /// Heuristic estimate of everything appended since that report.
    pub inferred_tokens: u64,
}

impl TokenLedger {
    /// Tokens the next request is expected to carry.
    #[must_use]
    pub const fn effective_tokens(&self) -> u64 {
        self.authoritative_tokens
            .saturating_add(self.inferred_tokens)
    }

    /// Account for one message appended to the transcript. A completed
    /// assistant response carrying a non-zero usage report replaces the
    /// anchor (its `context_tokens` already include the response itself);
    /// every other message adds its heuristic estimate.
    pub fn record(&mut self, message: &Message) {
        match authoritative_context_tokens(message) {
            Some(tokens) => {
                self.authoritative_tokens = tokens;
                self.inferred_tokens = 0;
            }
            None => {
                self.inferred_tokens = self
                    .inferred_tokens
                    .saturating_add(estimate_message_tokens(message));
            }
        }
    }

    /// Ledger for a context with no usable provider report: everything is
    /// inferred. Used after compaction, when the preserved tail's reports
    /// describe the pre-rewrite context, and for forked subagents, whose
    /// parent's reports describe a different prompt and tool set.
    #[must_use]
    pub fn inferred_from(compacted_prefix: &[Value], messages: &[Message]) -> Self {
        Self {
            authoritative_tokens: 0,
            inferred_tokens: estimate_compacted_prefix_tokens(compacted_prefix)
                .saturating_add(estimate_messages_tokens(messages)),
        }
    }
}

/// Context size a message reports, when that report is usable as an anchor.
/// Interrupted and errored turns report partial or zero usage, which would
/// peg the ledger far below the real context.
fn authoritative_context_tokens(message: &Message) -> Option<u64> {
    let Message::Assistant(assistant) = message else {
        return None;
    };
    if matches!(
        assistant.stop_reason,
        Some(StopReason::Interrupted | StopReason::Error)
    ) {
        return None;
    }
    let tokens = assistant.usage.as_ref()?.context_tokens();
    (tokens > 0).then_some(tokens)
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

/// Estimates the number of tokens consumed by `text` for budgeting purposes.
///
/// Uses a char-to-token ratio of 10/37 (≈3.7 chars per token), which averages
/// English prose across current BPE tokenizers (cl100k_base, o200k_base, GPT
/// tokenizers). This is intentionally *heuristic*: it runs in O(chars) and
/// avoids loading tokenizer tables, at the cost of being wrong by ±20% for
/// code-heavy or non-Latin text.
///
/// The ledger uses this solely for triggering thresholds between provider
/// reports, not for billing, so the approximation is acceptable.
#[must_use]
pub fn estimate_text_tokens(text: &str) -> u64 {
    CharHeuristicEstimator.estimate(text)
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
/// Estimate tokens for provider-native compacted prefix items.
pub fn estimate_compacted_prefix_tokens(compacted_prefix: &[Value]) -> u64 {
    compacted_prefix.iter().map(estimate_json_tokens).sum()
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

#[must_use]
/// Estimate tokens for a JSON value rendered with [`stable_json`].
pub fn estimate_json_tokens(value: &Value) -> u64 {
    estimate_text_tokens(&stable_json(value))
}

/// Render JSON with object keys sorted, so equal values always render (and
/// estimate, and cache-key) identically.
#[must_use]
pub fn stable_json(value: &Value) -> String {
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
    use serde_json::json;

    use super::*;
    use crate::{AssistantMessage, MessageId, Usage, UserMessage};

    fn assistant(text: &str, stop_reason: Option<StopReason>, usage: Option<Usage>) -> Message {
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

    fn reported(input_tokens: u64, output_tokens: u64) -> Option<Usage> {
        Some(Usage {
            input_tokens,
            output_tokens,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        })
    }

    /// The whole point of anchoring: a completed report replaces the
    /// heuristic for everything up to and including the reporting message.
    #[test]
    fn record_replaces_anchor_and_zeroes_inferred() {
        let mut ledger = TokenLedger::default();
        ledger.record(&Message::User(UserMessage::text(
            "x".repeat(4_000).as_str(),
        )));
        assert!(ledger.inferred_tokens > 1_000);

        ledger.record(&assistant(
            "done",
            Some(StopReason::EndTurn),
            reported(50_000, 200),
        ));

        assert_eq!(
            ledger,
            TokenLedger {
                authoritative_tokens: 50_200,
                inferred_tokens: 0,
            }
        );
        assert_eq!(ledger.effective_tokens(), 50_200);
    }

    #[test]
    fn record_adds_heuristic_estimate_on_top_of_anchor() {
        let mut ledger = TokenLedger {
            authoritative_tokens: 50_000,
            inferred_tokens: 0,
        };
        let follow_up = Message::User(UserMessage::text("follow up"));

        ledger.record(&follow_up);

        assert_eq!(ledger.authoritative_tokens, 50_000);
        assert_eq!(ledger.inferred_tokens, estimate_message_tokens(&follow_up));
    }

    /// Interrupted and errored turns report partial or zero usage. Anchoring
    /// on those would peg the ledger far below the real context and stop
    /// compaction from ever firing, so they count like any other message.
    #[test]
    fn record_treats_unusable_reports_as_inferred_appends() {
        struct Case {
            name: &'static str,
            message: Message,
        }

        let cases = [
            Case {
                name: "interrupted",
                message: assistant(
                    "partial",
                    Some(StopReason::Interrupted),
                    reported(50_000, 10),
                ),
            },
            Case {
                name: "error",
                message: assistant("boom", Some(StopReason::Error), reported(50_000, 10)),
            },
            Case {
                name: "zero usage",
                message: assistant("empty", Some(StopReason::EndTurn), reported(0, 0)),
            },
            Case {
                name: "no usage",
                message: assistant("none", Some(StopReason::EndTurn), None),
            },
            Case {
                name: "not an assistant message",
                message: Message::User(UserMessage::text("user")),
            },
        ];

        for case in cases {
            let mut ledger = TokenLedger {
                authoritative_tokens: 7,
                inferred_tokens: 3,
            };
            ledger.record(&case.message);
            assert_eq!(
                ledger.authoritative_tokens, 7,
                "{}: must not anchor",
                case.name
            );
            assert_eq!(
                ledger.inferred_tokens,
                3 + estimate_message_tokens(&case.message),
                "{}: must add its estimate",
                case.name
            );
        }
    }

    #[test]
    fn record_saturates_instead_of_overflowing() {
        let mut ledger = TokenLedger {
            authoritative_tokens: u64::MAX,
            inferred_tokens: u64::MAX,
        };
        ledger.record(&Message::User(UserMessage::text("more")));
        assert_eq!(ledger.inferred_tokens, u64::MAX);
        assert_eq!(ledger.effective_tokens(), u64::MAX);
    }

    #[test]
    fn inferred_from_counts_prefix_and_messages_without_an_anchor() {
        let prefix = vec![json!({"type": "reasoning", "encrypted_content": "abcdefgh"})];
        let messages = vec![
            Message::User(UserMessage::text("kept")),
            assistant(
                "stale report",
                Some(StopReason::EndTurn),
                reported(80_000, 500),
            ),
        ];

        let ledger = TokenLedger::inferred_from(&prefix, &messages);

        assert_eq!(ledger.authoritative_tokens, 0);
        assert_eq!(
            ledger.inferred_tokens,
            estimate_compacted_prefix_tokens(&prefix) + estimate_messages_tokens(&messages)
        );
        assert!(
            ledger.effective_tokens() < 1_000,
            "the stale 80_500 report must not survive the rebuild"
        );
    }

    #[test]
    fn inferred_from_empty_context_is_zero() {
        assert_eq!(TokenLedger::inferred_from(&[], &[]), TokenLedger::default());
    }

    #[test]
    fn estimate_text_tokens_uses_character_count() {
        assert_eq!(estimate_text_tokens(""), 0);
        assert_eq!(estimate_text_tokens("abcd"), 1);
        assert_eq!(estimate_text_tokens("abcdefghij"), 2);
    }

    #[test]
    fn estimate_message_tokens_covers_every_message_kind() {
        let tool_call = Message::Assistant(AssistantMessage {
            id: MessageId::new(),
            created_at: Utc::now(),
            parts: vec![AssistantPart::ToolCall(crate::ToolCall {
                id: crate::ToolCallId::from("call_1"),
                name: "read".into(),
                arguments: json!({"path": "a.txt"}),
            })],
            stop_reason: None,
            usage: None,
            replay_meta: Default::default(),
        });
        let empty_tool = Message::Tool(crate::ToolResultMessage {
            id: MessageId::new(),
            call_id: crate::ToolCallId::from("call_1"),
            content: ToolResult::Empty,
            error: None,
            created_at: Utc::now(),
        });
        let json_tool = Message::Tool(crate::ToolResultMessage {
            id: MessageId::new(),
            call_id: crate::ToolCallId::from("call_2"),
            content: ToolResult::Json {
                value: json!({"b": true, "a": "abcd"}),
            },
            error: None,
            created_at: Utc::now(),
        });
        let system = Message::System(crate::SystemMessage {
            id: MessageId::new(),
            created_at: Utc::now(),
            text: "x".repeat(37),
        });

        assert_eq!(estimate_message_tokens(&empty_tool), 0);
        assert_eq!(
            estimate_message_tokens(&json_tool),
            estimate_text_tokens("{\"a\":\"abcd\",\"b\":true}")
        );
        assert_eq!(estimate_message_tokens(&system), 10);
        assert_eq!(
            estimate_message_tokens(&tool_call),
            estimate_text_tokens("read")
                + estimate_text_tokens("{\"path\":\"a.txt\"}")
                + estimate_text_tokens("tool_call")
        );
    }

    #[test]
    fn stable_json_sorts_object_keys() {
        let rendered = stable_json(&json!({
            "b": true,
            "a": "abcd",
            "c": [null, 1, {"z": 0, "y": 1}],
        }));

        assert_eq!(
            rendered,
            "{\"a\":\"abcd\",\"b\":true,\"c\":[null,1,{\"y\":1,\"z\":0}]}"
        );
        assert_eq!(
            estimate_json_tokens(&json!({"b": true, "a": "abcd"})),
            estimate_text_tokens("{\"a\":\"abcd\",\"b\":true}")
        );
    }
}
