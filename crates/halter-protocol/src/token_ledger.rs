//! Context-size accounting: the session token ledger and the heuristic
//! estimator it falls back on between provider reports.
// pattern: Functional Core

use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    AssistantPart, Message, PromptSegment, StopReason, ToolResult, ToolSpec, Usage, UserPart,
};

// Version 2 rejects output-only reports as context anchors. Rebuild persisted
// ledgers from the transcript once so an old, undersized anchor cannot survive.
const CURRENT_ACCOUNTING_VERSION: u8 = 2;
const MESSAGE_OVERHEAD_TOKENS: u64 = 4;
const PROMPT_SEGMENT_OVERHEAD_TOKENS: u64 = 4;
const TOOL_SPEC_OVERHEAD_TOKENS: u64 = 8;

const fn legacy_accounting_version() -> u8 {
    0
}

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
/// `request_tokens` is the current heuristic size of the provider-visible
/// prompt segments and tool declarations. `request_tokens_at_last_anchor`
/// remembers how much of that base the last authoritative report already
/// covered, so a changed prompt base contributes only its delta.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct TokenLedger {
    /// Context size reported by the last completed assistant response, or
    /// zero when no usable report covers the current transcript.
    pub authoritative_tokens: u64,
    /// Heuristic estimate of everything appended since that report.
    pub inferred_tokens: u64,
    /// Current estimate of prompt segments and provider-visible tool specs.
    #[serde(default)]
    pub request_tokens: u64,
    /// Request-base estimate covered by `authoritative_tokens`.
    #[serde(default)]
    pub request_tokens_at_last_anchor: u64,
    /// Version of the accounting semantics used by this serialized ledger.
    /// Missing or unrecognized versions are rebuilt before use.
    #[doc(hidden)]
    #[serde(default = "legacy_accounting_version")]
    pub accounting_version: u8,
}

impl Default for TokenLedger {
    fn default() -> Self {
        Self {
            authoritative_tokens: 0,
            inferred_tokens: 0,
            request_tokens: 0,
            request_tokens_at_last_anchor: 0,
            accounting_version: CURRENT_ACCOUNTING_VERSION,
        }
    }
}

impl TokenLedger {
    #[must_use]
    pub const fn needs_request_preparation(&self, request_tokens: u64) -> bool {
        self.accounting_version != CURRENT_ACCOUNTING_VERSION
            || self.request_tokens != request_tokens
    }

    /// Tokens the next request is expected to carry.
    #[must_use]
    pub const fn effective_tokens(&self) -> u64 {
        self.projected_tokens(self.request_tokens)
    }

    /// Project the next request against a freshly computed request base.
    #[must_use]
    pub const fn projected_tokens(&self, request_tokens: u64) -> u64 {
        if self.authoritative_tokens == 0 {
            return self.inferred_tokens.saturating_add(request_tokens);
        }
        self.authoritative_tokens
            .saturating_sub(self.request_tokens_at_last_anchor)
            .saturating_add(request_tokens)
            .saturating_add(self.inferred_tokens)
    }

    /// Install the current request-base estimate. Legacy ledgers are rebuilt
    /// from the context first because their authoritative count included an
    /// unknown prompt/tool base and cannot safely be adjusted by a delta.
    pub fn prepare_request(
        &mut self,
        request_tokens: u64,
        compacted_prefix: &[Value],
        messages: &[Message],
    ) {
        if self.accounting_version != CURRENT_ACCOUNTING_VERSION {
            self.authoritative_tokens = 0;
            self.inferred_tokens = estimate_compacted_prefix_tokens(compacted_prefix)
                .saturating_add(estimate_messages_tokens(messages));
            self.request_tokens_at_last_anchor = 0;
            self.accounting_version = CURRENT_ACCOUNTING_VERSION;
        }
        self.request_tokens = request_tokens;
    }

    /// Re-estimate a rewritten transcript while preserving its request base.
    pub fn rebuild_context(&mut self, compacted_prefix: &[Value], messages: &[Message]) {
        self.authoritative_tokens = 0;
        self.inferred_tokens = estimate_compacted_prefix_tokens(compacted_prefix)
            .saturating_add(estimate_messages_tokens(messages));
        self.request_tokens_at_last_anchor = 0;
        self.accounting_version = CURRENT_ACCOUNTING_VERSION;
    }

    /// Account for one message appended to the transcript. A completed
    /// assistant response carrying a non-zero input usage report replaces the
    /// anchor (its `context_tokens` already include the response itself);
    /// every other message adds its heuristic estimate.
    pub fn record(&mut self, message: &Message) {
        match authoritative_context_tokens(message) {
            Some(tokens) => self.anchor(tokens),
            None => self.add_inferred(estimate_message_tokens(message)),
        }
    }

    /// Replace the anchor with a provider-reported context size.
    fn anchor(&mut self, tokens: u64) {
        self.authoritative_tokens = tokens;
        self.inferred_tokens = 0;
        self.request_tokens_at_last_anchor = self.request_tokens;
        self.accounting_version = CURRENT_ACCOUNTING_VERSION;
    }

    /// Add a heuristic estimate for content the provider has not reported on.
    fn add_inferred(&mut self, tokens: u64) {
        self.inferred_tokens = self.inferred_tokens.saturating_add(tokens);
    }

    /// Ledger for a context with no usable provider report: everything is
    /// inferred. Used after compaction, when the preserved tail's reports
    /// describe the pre-rewrite context, and for forked subagents, whose
    /// parent's reports describe a different prompt and tool set.
    #[must_use]
    pub fn inferred_from(compacted_prefix: &[Value], messages: &[Message]) -> Self {
        let mut ledger = Self::default();
        ledger.rebuild_context(compacted_prefix, messages);
        ledger
    }
}

/// Context size a message reports, when that report is usable as an anchor.
/// Interrupted and errored turns report partial or zero usage, which would
/// peg the ledger far below the real context.
fn authoritative_context_tokens(message: &Message) -> Option<u64> {
    let Message::Assistant(assistant) = message else {
        return None;
    };
    usable_report(assistant.stop_reason, assistant.usage.as_ref())
}

fn usable_report(stop_reason: Option<StopReason>, usage: Option<&Usage>) -> Option<u64> {
    if matches!(
        stop_reason,
        Some(StopReason::Interrupted | StopReason::Error)
    ) {
        return None;
    }
    let usage = usage?;
    // Output alone does not report how much history the provider read.
    // Anchoring on it would erase the accumulated transcript estimate.
    (usage.input_tokens > 0).then(|| usage.context_tokens())
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
        .map(|segment| {
            PROMPT_SEGMENT_OVERHEAD_TOKENS.saturating_add(estimate_text_tokens(&segment.text))
        })
        .sum()
}

#[must_use]
/// Estimate provider-visible tool declarations.
pub fn estimate_tool_spec_tokens(tools: &[ToolSpec]) -> u64 {
    tools
        .iter()
        .map(|tool| {
            TOOL_SPEC_OVERHEAD_TOKENS
                .saturating_add(estimate_text_tokens(&tool.name.0))
                .saturating_add(estimate_text_tokens(&tool.description))
                .saturating_add(estimate_json_tokens(&tool.input_schema))
        })
        .sum()
}

#[must_use]
/// Estimate request material that is not part of the transcript.
pub fn estimate_request_tokens(segments: &[PromptSegment], tools: &[ToolSpec]) -> u64 {
    estimate_segment_tokens(segments).saturating_add(estimate_tool_spec_tokens(tools))
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
    MESSAGE_OVERHEAD_TOKENS.saturating_add(match message {
        Message::System(message) => estimate_text_tokens(&message.text),
        Message::User(message) => message
            .parts
            .iter()
            .map(|part| match part {
                UserPart::Text { text } => estimate_text_tokens(text),
                UserPart::Image { media_type, data } | UserPart::Document { media_type, data } => {
                    estimate_media_tokens(media_type, data.len())
                }
            })
            .sum(),
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
    })
}

fn estimate_media_tokens(media_type: &crate::MediaType, byte_len: usize) -> u64 {
    // Providers commonly carry binary input as base64. Estimate that wire
    // representation rather than treating a media-only message as free.
    let encoded_chars = (byte_len as u64).saturating_add(2) / 3 * 4;
    estimate_json_tokens(&serde_json::to_value(media_type).unwrap_or(Value::Null))
        .saturating_add((encoded_chars.saturating_mul(10).saturating_add(36)) / 37)
        .max(1)
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
    use bytes::Bytes;
    use chrono::Utc;
    use indexmap::IndexMap;
    use serde_json::json;

    use super::*;
    use crate::{
        AssistantMessage, CacheScope, MessageId, PromptSegmentId, PromptSegmentKind, SessionState,
        ToolCapabilities, ToolConcurrency, Usage, UserMessage, Volatility,
    };

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
                ..TokenLedger::default()
            }
        );
        assert_eq!(ledger.effective_tokens(), 50_200);
    }

    #[test]
    fn record_adds_heuristic_estimate_on_top_of_anchor() {
        let mut ledger = TokenLedger {
            authoritative_tokens: 50_000,
            inferred_tokens: 0,
            ..TokenLedger::default()
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
                name: "output-only usage cannot describe the input context",
                message: assistant("done", Some(StopReason::EndTurn), reported(0, 50)),
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
                ..TokenLedger::default()
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
            ..TokenLedger::default()
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
    fn request_base_changes_adjust_an_authoritative_anchor_without_double_counting() {
        let mut ledger = TokenLedger::default();
        ledger.prepare_request(100, &[], &[]);
        ledger.record(&assistant(
            "done",
            Some(StopReason::EndTurn),
            reported(900, 100),
        ));

        assert_eq!(ledger.effective_tokens(), 1_000);
        assert_eq!(ledger.projected_tokens(200), 1_100);
        assert_eq!(ledger.projected_tokens(50), 950);

        ledger.prepare_request(200, &[], &[]);
        assert_eq!(ledger.effective_tokens(), 1_100);
    }

    #[test]
    fn legacy_ledger_is_rebuilt_before_request_projection() {
        let mut ledger: TokenLedger = serde_json::from_value(json!({
            "authoritative_tokens": 50_000,
            "inferred_tokens": 0
        }))
        .expect("legacy ledger");
        let messages = vec![Message::User(UserMessage::text("restored transcript"))];

        ledger.prepare_request(250, &[], &messages);

        assert_eq!(ledger.authoritative_tokens, 0);
        assert_eq!(ledger.inferred_tokens, estimate_messages_tokens(&messages));
        assert_eq!(
            ledger.effective_tokens(),
            250 + estimate_messages_tokens(&messages)
        );
        assert_eq!(ledger.accounting_version, CURRENT_ACCOUNTING_VERSION);
    }

    #[test]
    fn unknown_accounting_version_is_rebuilt_before_request_projection() {
        let messages = vec![Message::User(UserMessage::text("restored transcript"))];
        let mut ledger = TokenLedger {
            authoritative_tokens: 50_000,
            inferred_tokens: 0,
            request_tokens: 250,
            request_tokens_at_last_anchor: 250,
            accounting_version: CURRENT_ACCOUNTING_VERSION.saturating_add(1),
        };

        ledger.prepare_request(250, &[], &messages);

        assert_eq!(ledger.authoritative_tokens, 0);
        assert_eq!(ledger.inferred_tokens, estimate_messages_tokens(&messages));
        assert_eq!(ledger.accounting_version, CURRENT_ACCOUNTING_VERSION);
    }

    #[test]
    fn output_only_anchor_from_previous_accounting_is_rebuilt_on_resume() {
        let messages = vec![
            Message::user("x".repeat(80_000)),
            assistant("done", Some(StopReason::EndTurn), reported(0, 50)),
        ];
        let mut ledger: TokenLedger = serde_json::from_value(json!({
            "authoritative_tokens": 50,
            "inferred_tokens": 0,
            "request_tokens": 250,
            "request_tokens_at_last_anchor": 250,
            "accounting_version": 1
        }))
        .expect("previous accounting ledger");

        assert!(ledger.needs_request_preparation(250));
        ledger.prepare_request(250, &[], &messages);

        assert_eq!(ledger.authoritative_tokens, 0);
        assert_eq!(
            ledger.effective_tokens(),
            250 + estimate_messages_tokens(&messages)
        );
        assert!(ledger.effective_tokens() > 20_000);
        assert!(!ledger.needs_request_preparation(250));
    }

    #[test]
    fn legacy_session_without_a_ledger_rebuilds_its_transcript() {
        let message = Message::User(UserMessage::text("legacy transcript"));
        let state = SessionState {
            messages: vec![message.clone()],
            ..SessionState::default()
        };
        let mut json = serde_json::to_value(state).expect("serialize state");
        json.as_object_mut()
            .expect("state object")
            .remove("token_ledger");
        let mut restored: SessionState = serde_json::from_value(json).expect("legacy state");

        restored
            .token_ledger
            .prepare_request(100, &restored.compacted_prefix, &restored.messages);

        assert_eq!(restored.token_ledger.authoritative_tokens, 0);
        assert_eq!(
            restored.token_ledger.inferred_tokens,
            estimate_message_tokens(&message)
        );
        assert_eq!(
            restored.token_ledger.effective_tokens(),
            100 + estimate_message_tokens(&message)
        );
    }

    #[test]
    fn media_only_messages_and_request_declarations_are_not_free() {
        let media = Message::User(UserMessage {
            id: MessageId::new(),
            created_at: Utc::now(),
            parts: vec![UserPart::Image {
                media_type: "image/png".to_owned(),
                data: Bytes::from_static(&[0; 64]),
            }],
        });
        let segment = PromptSegment {
            id: PromptSegmentId::new(),
            text: "system instructions".to_owned(),
            volatility: Volatility::Static,
            cache_scope: CacheScope::PrefixCacheable,
            content_hash: "segment".to_owned(),
            kind: PromptSegmentKind::System,
        };
        let tool = ToolSpec {
            name: "read".into(),
            description: "Read a file".to_owned(),
            input_schema: json!({"type": "object", "properties": {"path": {"type": "string"}}}),
            concurrency: ToolConcurrency::ReadOnly,
            capabilities: ToolCapabilities::default(),
            provider_aliases: IndexMap::new(),
        };

        assert!(estimate_message_tokens(&media) > MESSAGE_OVERHEAD_TOKENS);
        assert!(estimate_request_tokens(&[segment], &[tool]) > 0);
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

        assert_eq!(
            estimate_message_tokens(&empty_tool),
            MESSAGE_OVERHEAD_TOKENS
        );
        assert_eq!(
            estimate_message_tokens(&json_tool),
            MESSAGE_OVERHEAD_TOKENS + estimate_text_tokens("{\"a\":\"abcd\",\"b\":true}")
        );
        assert_eq!(
            estimate_message_tokens(&system),
            MESSAGE_OVERHEAD_TOKENS + 10
        );
        assert_eq!(
            estimate_message_tokens(&tool_call),
            MESSAGE_OVERHEAD_TOKENS
                + estimate_text_tokens("read")
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

#[cfg(kani)]
mod kani_proofs {
    //! Proofs over the ledger's integer core. They deliberately avoid
    //! constructing `Message`: its `serde_json::Value` fields pull B-tree drop
    //! glue into the model, which CBMC cannot bound. `record` is glue over
    //! `usable_report`, `anchor`, and `add_inferred`, which are proved here.
    use super::*;

    fn any_ledger() -> TokenLedger {
        TokenLedger {
            authoritative_tokens: kani::any(),
            inferred_tokens: kani::any(),
            request_tokens: kani::any(),
            request_tokens_at_last_anchor: kani::any(),
            accounting_version: CURRENT_ACCOUNTING_VERSION,
        }
    }

    fn any_stop_reason() -> Option<StopReason> {
        match kani::any::<u8>() {
            0 => None,
            1 => Some(StopReason::EndTurn),
            2 => Some(StopReason::ToolUse),
            3 => Some(StopReason::Interrupted),
            4 => Some(StopReason::MaxTokens),
            _ => Some(StopReason::Error),
        }
    }

    fn any_usage() -> Usage {
        Usage {
            input_tokens: kani::any(),
            output_tokens: kani::any(),
            cache_creation_input_tokens: kani::any(),
            cache_read_input_tokens: kani::any(),
        }
    }

    /// A larger request base never shrinks the projection.
    #[kani::proof]
    fn projection_is_monotone_in_request_base() {
        let ledger = any_ledger();
        let smaller: u64 = kani::any();
        let larger: u64 = kani::any_where(|larger| *larger >= smaller);
        assert!(ledger.projected_tokens(smaller) <= ledger.projected_tokens(larger));
    }

    /// Interrupted or errored turns, and empty reports, never anchor.
    /// Every other reported context size does.
    #[kani::proof]
    fn only_complete_nonzero_reports_are_usable() {
        let stop_reason = any_stop_reason();
        let usage = any_usage();
        let has_usage: bool = kani::any();
        let report = usable_report(stop_reason, has_usage.then_some(&usage));

        let partial = matches!(
            stop_reason,
            Some(StopReason::Interrupted | StopReason::Error)
        );
        let expected =
            (!partial && has_usage && usage.input_tokens > 0).then(|| usage.context_tokens());
        assert_eq!(report, expected);
    }

    /// Anchoring projects the next request at the reported context size, or
    /// at the request base if the provider reported less than that estimate.
    #[kani::proof]
    fn anchor_projects_reported_size() {
        let mut ledger = any_ledger();
        let reported: u64 = kani::any();
        ledger.anchor(reported);
        assert_eq!(ledger.inferred_tokens, 0);
        assert_eq!(
            ledger.effective_tokens(),
            reported.max(ledger.request_tokens)
        );
    }

    /// After anchoring, a changed prompt/tool base moves the projection by
    /// exactly the delta, with no double counting of the old base.
    #[kani::proof]
    fn request_base_change_contributes_only_its_delta() {
        let mut ledger = any_ledger();
        let reported: u64 = kani::any_where(|reported| *reported >= ledger.request_tokens);
        ledger.anchor(reported);

        let new_base: u64 = kani::any();
        ledger.prepare_request(new_base, &[], &[]);

        if let Some(expected) =
            (reported - ledger.request_tokens_at_last_anchor).checked_add(new_base)
        {
            assert_eq!(ledger.effective_tokens(), expected);
        }
    }

    /// Inferred content only ever grows the projection and never moves the
    /// anchor.
    #[kani::proof]
    fn inferred_content_never_moves_anchor_or_shrinks_projection() {
        let mut ledger = any_ledger();
        let before = ledger;
        ledger.add_inferred(kani::any());
        assert_eq!(ledger.authoritative_tokens, before.authoritative_tokens);
        assert_eq!(
            ledger.request_tokens_at_last_anchor,
            before.request_tokens_at_last_anchor
        );
        assert!(ledger.effective_tokens() >= before.effective_tokens());
    }
}
