//! Cache-prefix stability, as a property of every codec.
//!
//! Appending to a history must leave the encoded bytes of everything before
//! it unchanged: tools, the system prompt, and every earlier message. A
//! provider prompt cache is a byte (token) prefix, so anything else re-bills
//! the whole conversation as a cache write.
//!
//! The unit of appending is a *step*: anything that can end a request the
//! runtime sends. That is a user message, a system message, an assistant
//! reply without tool calls, or an assistant reply with tool calls together
//! with its results and the system messages hooks interleave with them. No
//! request is sent between a tool call and its result, so the states inside
//! a step are never encoded on their own.
//!
//! Legitimate exceptions, kept out of the property's domain rather than
//! weakening its assertions:
//! - System messages before the first other message are the session's
//!   leading system content and belong to the system prompt. Generated
//!   histories start with a user message after them, so appending can never
//!   add to that run. `leading_system_messages_stay_in_the_system_prompt`
//!   pins their encoding.
//! - Compaction and rollover replace the window and compacted prefix. They
//!   are not appends and change the prefix once, by design.
//! - Anthropic's `cache_control` marker on the last user message moves
//!   forward by design. Messages are compared without it, and the marker is
//!   checked separately: it may only move forward, within the 20-block
//!   lookback Anthropic uses to find the previous cache entry.
use crate::{
    anthropic_codec,
    openai_codec::{
        self, ResponsesInstructionMode,
        ResponsesInstructionMode::{DeveloperMessage as Dev, TopLevelRequired as Top},
        ResponsesRequestOptions,
    },
};
use bytes::Bytes;
use halter_protocol::{
    ApiKind, AssembledPrompt, AssistantMessage, AssistantPart, CacheBreakpoints, CacheScope,
    Message, MessageId, ModelId, ModelRole, PromptSegment, PromptSegmentId, PromptSegmentKind,
    ProviderCompactionRequest, ProviderKind, ProviderName, ProviderRequest, ResolvedModel,
    SystemMessage, ThinkingBlock, ToolAlias, ToolCall, ToolCallId, ToolCapabilities,
    ToolConcurrency, ToolError, ToolResult, ToolResultMessage, ToolSpec, TurnId, UserMessage,
    UserPart, Volatility,
};
use proptest::prelude::*;
use serde_json::{Value, json};

/// Anthropic finds the previous cache entry by walking back at most this many
/// content blocks from the newest breakpoint.
const ANTHROPIC_LOOKBACK_BLOCKS: usize = 20;

fn system(text: &str) -> Message {
    Message::System(SystemMessage {
        id: MessageId::new(),
        created_at: chrono::Utc::now(),
        text: text.into(),
    })
}

fn user(parts: Vec<UserPart>) -> Message {
    Message::User(UserMessage {
        id: MessageId::new(),
        created_at: chrono::Utc::now(),
        parts,
    })
}

fn assistant(id: String, parts: Vec<AssistantPart>) -> Message {
    Message::Assistant(AssistantMessage {
        id: MessageId::from(id),
        created_at: chrono::Utc::now(),
        parts,
        stop_reason: None,
        usage: None,
        replay_meta: Default::default(),
    })
}

fn tool_result(call_id: &str, text: &str, failed: bool) -> Message {
    Message::Tool(ToolResultMessage {
        id: MessageId::new(),
        call_id: ToolCallId::from(call_id),
        content: ToolResult::Text {
            text: text.to_owned(),
        },
        error: failed.then(|| ToolError::new("failed")),
        created_at: chrono::Utc::now(),
    })
}

fn segment(text: &str, kind: PromptSegmentKind) -> PromptSegment {
    PromptSegment {
        id: PromptSegmentId::new(),
        text: text.to_owned(),
        volatility: Volatility::Static,
        cache_scope: CacheScope::PrefixCacheable,
        content_hash: text.to_owned(),
        kind,
    }
}

/// A request as the runtime assembles it. `sections` selects the Anthropic
/// block-form system prompt with all four breakpoints; without it the codec
/// takes the flat-string path non-runtime callers use.
fn request(api_kind: ApiKind, messages: Vec<Message>, sections: bool) -> ProviderRequest {
    let provider_kind = match api_kind {
        ApiKind::AnthropicMessages => ProviderKind::Anthropic,
        _ => ProviderKind::OpenAi,
    };
    let ordered_segments = vec![
        segment("follow plan", PromptSegmentKind::System),
        segment("skills: none", PromptSegmentKind::Skill),
    ];
    let tool = |name: &str, alias: Option<&str>| ToolSpec {
        name: name.into(),
        description: format!("{name} tool"),
        input_schema: json!({"type": "object"}),
        concurrency: ToolConcurrency::ReadOnly,
        capabilities: ToolCapabilities::default(),
        provider_aliases: alias
            .map(|alias| (provider_kind, ToolAlias::from(alias)))
            .into_iter()
            .collect(),
    };
    ProviderRequest {
        session_id: Default::default(),
        turn_id: TurnId::new(),
        model: ResolvedModel {
            role: ModelRole::default(),
            id: ModelId::from("model"),
            provider: ProviderName::from("provider"),
            provider_kind,
            api_kind,
            model: "model".to_owned(),
            max_input_tokens: Some(200_000),
            max_output_tokens: Some(8_192),
            reasoning: None,
            tokens_per_minute: None,
        },
        prompt: AssembledPrompt {
            segments: ordered_segments.clone(),
            transcript: messages.clone(),
            rendered_prefix: "follow plan\n\nskills: none".to_owned(),
            ordered_segments,
            prefix_cache_key: "cache-key".to_owned(),
            rendered_transcript: String::new(),
            rendered: String::new(),
            cache_breakpoints: if sections {
                CacheBreakpoints::all()
            } else {
                CacheBreakpoints::default()
            },
            system_segment_count: 1,
            skill_segment_count: 1,
        },
        compacted_prefix: Vec::new(),
        messages,
        tools: vec![tool("read", Some("fs_read")), tool("write", None)],
        previous_response_id: None,
        new_messages_start: 0,
    }
}

fn text() -> impl Strategy<Value = String> {
    prop_oneof![
        Just(String::new()),
        Just("  ".to_owned()),
        "[a-z]{1,6}( [a-z]{1,6})?"
    ]
}

fn user_part() -> impl Strategy<Value = UserPart> {
    prop_oneof![
        3 => text().prop_map(|text| UserPart::Text { text }),
        1 => Just(UserPart::Image {
            media_type: "image/png".to_owned(),
            data: Bytes::from_static(b"png"),
        }),
        1 => Just(UserPart::Document {
            media_type: "application/pdf".to_owned(),
            data: Bytes::from_static(b"pdf"),
        }),
    ]
}

fn reply_part() -> impl Strategy<Value = AssistantPart> {
    prop_oneof![
        text().prop_map(|text| AssistantPart::Text { text }),
        (text(), proptest::option::of("[a-z]{4}")).prop_map(|(text, signature)| {
            AssistantPart::Thinking(ThinkingBlock { text, signature })
        }),
    ]
}

/// One step, as the messages the runtime appends for it. Ids are drawn
/// from `seed` so a call and its result agree and steps stay distinct.
fn step() -> impl Strategy<Value = Vec<Message>> {
    let call = (
        prop_oneof![Just("read"), Just("write")],
        any::<bool>(),
        any::<bool>(),
        proptest::option::of(text()),
    );
    prop_oneof![
        proptest::collection::vec(user_part(), 1..3).prop_map(|parts| vec![user(parts)]),
        text().prop_map(|text| vec![system(&text)]),
        (any::<u32>(), proptest::collection::vec(reply_part(), 0..3))
            .prop_map(|(seed, parts)| vec![assistant(format!("msg_{seed}"), parts)]),
        (
            any::<u32>(),
            proptest::collection::vec(reply_part(), 0..2),
            proptest::collection::vec(call, 1..4),
            proptest::option::of(text()),
        )
            .prop_map(|(seed, mut parts, calls, trailing_system)| {
                let id = |index: usize| format!("call_{seed}_{index}");
                parts.extend(calls.iter().enumerate().map(|(index, (name, ..))| {
                    AssistantPart::ToolCall(ToolCall {
                        id: ToolCallId::from(id(index)),
                        name: (*name).into(),
                        arguments: json!({"index": index}),
                    })
                }));
                // A hook on a call (PreToolUse, PostToolUse) lands its system
                // message before that call's result.
                std::iter::once(assistant(format!("msg_{seed}"), parts))
                    .chain(
                        calls
                            .iter()
                            .enumerate()
                            .flat_map(|(index, (_, failed, _, hook))| {
                                hook.as_deref().map(system).into_iter().chain([tool_result(
                                    &id(index),
                                    "done",
                                    *failed,
                                )])
                            }),
                    )
                    .chain(trailing_system.as_deref().map(system))
                    .collect()
            }),
    ]
}

/// Leading system messages, the first user message, and any later steps.
fn history() -> impl Strategy<Value = Vec<Message>> {
    (
        proptest::collection::vec(text(), 0..3),
        proptest::collection::vec(user_part(), 1..3),
        proptest::collection::vec(step(), 0..6),
    )
        .prop_map(|(leading, first, steps)| {
            leading
                .iter()
                .map(|text| system(text))
                .chain([user(first)])
                .chain(steps.into_iter().flatten())
                .collect()
        })
}

fn without_cache_control(messages: &Value) -> Vec<Value> {
    messages
        .as_array()
        .expect("messages")
        .iter()
        .map(|message| {
            let mut message = message.clone();
            for block in message["content"].as_array_mut().into_iter().flatten() {
                block
                    .as_object_mut()
                    .map(|block| block.remove("cache_control"));
            }
            message
        })
        .collect()
}

/// Flattened index of the last message block carrying `cache_control`.
fn last_breakpoint(messages: &Value) -> Option<usize> {
    messages
        .as_array()?
        .iter()
        .flat_map(|message| message["content"].as_array().into_iter().flatten())
        .enumerate()
        .filter(|(_, block)| block.get("cache_control").is_some())
        .map(|(index, _)| index)
        .last()
}

/// Request shapes each API rejects: a role it does not accept in its
/// message list, or anything other than results between tool calls and
/// their results.
fn assert_wire_valid(api_kind: ApiKind, body: &Value) {
    match api_kind {
        ApiKind::AnthropicMessages => {
            let messages = body["messages"].as_array().expect("messages");
            for (index, message) in messages.iter().enumerate() {
                assert!(
                    matches!(message["role"].as_str(), Some("user" | "assistant")),
                    "{message}"
                );
                let calls = message["content"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter(|block| block["type"] == "tool_use")
                    .count();
                if calls > 0 {
                    let next = &messages[index + 1]["content"];
                    let results = next.as_array().expect("results").iter();
                    assert!(
                        results
                            .take(calls)
                            .all(|block| block["type"] == "tool_result"),
                        "tool_use must be answered first in the next message: {next}"
                    );
                }
            }
        }
        ApiKind::OpenAiResponses => {
            let input = body["input"].as_array().expect("input");
            let mut open = 0usize;
            for item in input {
                match item["type"].as_str() {
                    Some("function_call") => open += 1,
                    Some("function_call_output") => open -= 1,
                    _ => assert_eq!(open, 0, "item between calls and results: {item}"),
                }
            }
        }
        ApiKind::OpenAiChat => {
            let messages = body["messages"].as_array().expect("messages");
            let mut open = 0usize;
            for message in messages {
                match message["role"].as_str() {
                    Some("tool") => open -= 1,
                    role => {
                        assert_eq!(open, 0, "message between calls and results: {message}");
                        assert!(matches!(role, Some("system" | "user" | "assistant")));
                        open = message["tool_calls"].as_array().map_or(0, Vec::len);
                    }
                }
            }
        }
        ApiKind::Fake => {}
    }
}

fn encode(api_kind: ApiKind, request: &ProviderRequest, mode: ResponsesInstructionMode) -> Value {
    match api_kind {
        ApiKind::AnthropicMessages => {
            anthropic_codec::encode_request(request, None).expect("encode anthropic")
        }
        ApiKind::OpenAiResponses => openai_codec::encode_responses_request(
            request,
            ResponsesRequestOptions {
                stream: false,
                store: None,
                prompt_cache_key: None,
                include_encrypted_reasoning: false,
                reasoning_summary: None,
                instruction_mode: mode,
                temperature: None,
                routing: None,
            },
        )
        .expect("encode responses"),
        ApiKind::OpenAiChat => {
            openai_codec::encode_chat_request(request, false).expect("encode chat")
        }
        ApiKind::Fake => unreachable!("the fake provider has no wire format"),
    }
}

/// Every dialect and encoder configuration the property covers.
const CONFIGS: [(ApiKind, bool, ResponsesInstructionMode); 5] = [
    (ApiKind::AnthropicMessages, false, Dev),
    (ApiKind::AnthropicMessages, true, Dev),
    (ApiKind::OpenAiResponses, true, Dev),
    (ApiKind::OpenAiResponses, true, Top),
    (ApiKind::OpenAiChat, true, Dev),
];

fn check_append(base: &[Message], appended: &[Message]) {
    let extended = [base, appended].concat();
    let has_document = extended.iter().any(|message| {
        matches!(message, Message::User(user)
            if user.parts.iter().any(|part| matches!(part, UserPart::Document { .. })))
    });
    for (api_kind, sections, mode) in CONFIGS {
        if api_kind == ApiKind::OpenAiChat && has_document {
            continue; // Chat Completions has no document input.
        }
        let before = encode(api_kind, &request(api_kind, base.to_vec(), sections), mode);
        let after = encode(
            api_kind,
            &request(api_kind, extended.clone(), sections),
            mode,
        );
        assert_wire_valid(api_kind, &after);
        let context = format!("{api_kind:?} sections={sections} {mode:?}");
        for key in ["system", "instructions", "tools"] {
            assert_eq!(
                before.get(key),
                after.get(key),
                "{context}: `{key}` changed"
            );
        }
        let list = if api_kind == ApiKind::OpenAiResponses {
            "input"
        } else {
            "messages"
        };
        let (before_items, after_items) = (
            without_cache_control(&before[list]),
            without_cache_control(&after[list]),
        );
        assert!(
            after_items.starts_with(&before_items),
            "{context}: earlier {list} changed\nbefore: {before_items:#?}\nafter: {after_items:#?}"
        );
        if let (Some(old), Some(new)) = (
            last_breakpoint(&before[list]),
            last_breakpoint(&after[list]),
        ) {
            assert!(
                (old..=old + ANTHROPIC_LOOKBACK_BLOCKS).contains(&new),
                "{context}: breakpoint moved from block {old} to {new}"
            );
        }
    }

    // A chained Responses request sends exactly what the full request
    // would add: nothing dropped, nothing repeated.
    let full = |messages: &[Message]| {
        let request = request(ApiKind::OpenAiResponses, messages.to_vec(), true);
        encode(ApiKind::OpenAiResponses, &request, Dev)
    };
    let mut chained = request(ApiKind::OpenAiResponses, extended.clone(), true);
    chained.previous_response_id = Some("resp_1".to_owned());
    chained.new_messages_start = base.len();
    let chained = encode(ApiKind::OpenAiResponses, &chained, Dev);
    let mut stitched = full(base)["input"].as_array().expect("input").clone();
    stitched.extend(chained["input"].as_array().expect("input").iter().cloned());
    assert_eq!(
        &stitched,
        full(&extended)["input"].as_array().expect("input"),
        "chained Responses input is not the suffix of the full input"
    );
}

/// A fixed seed keeps CI reproducible. The nightly soak sets `PROPTEST_CASES`
/// to run many more cases, and samples fresh ones.
fn case_seed() -> proptest::test_runner::RngSeed {
    match std::env::var_os("PROPTEST_CASES") {
        Some(_) => proptest::test_runner::RngSeed::Random,
        None => proptest::test_runner::RngSeed::Fixed(0x6361_6368),
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        rng_seed: case_seed(),
        failure_persistence: None,
        ..ProptestConfig::default()
    })]

    #[test]
    fn appending_a_step_never_rewrites_the_encoded_prefix(
        base in history(),
        appended in step(),
    ) {
        check_append(&base, &appended);
    }
}

/// The system prompt and the conversation as dialect-neutral labels:
/// `user`, `assistant`, `results` (one tool batch's results) or
/// `system:<text>` for a system message encoded in place.
fn transcript(api_kind: ApiKind, body: &Value) -> (String, Vec<String>) {
    let text = |value: &Value| value.as_str().unwrap_or_default().to_owned();
    let (system, items) = match api_kind {
        ApiKind::AnthropicMessages => (
            text(&body["system"]),
            body["messages"].as_array().expect("messages").clone(),
        ),
        ApiKind::OpenAiResponses => (
            text(&body["instructions"]),
            body["input"].as_array().expect("input").clone(),
        ),
        _ => {
            let messages = body["messages"].as_array().expect("messages");
            (text(&messages[0]["content"]), messages[1..].to_vec())
        }
    };
    let mut labels = items
        .iter()
        .map(|item| {
            let block = &item["content"][0];
            match (item["role"].as_str(), item["type"].as_str()) {
                (Some("user"), _) if block["type"] == "tool_result" => "results".to_owned(),
                (_, Some("function_call_output")) | (Some("tool"), _) => "results".to_owned(),
                (Some("user"), _) => match text(&block["text"])
                    .strip_prefix("<system-reminder>\n")
                    .and_then(|text| text.strip_suffix("\n</system-reminder>"))
                {
                    Some(reminder) => format!("system:{reminder}"),
                    None => "user".to_owned(),
                },
                (Some("developer"), _) => format!("system:{}", text(&block["text"])),
                (Some("system"), _) => format!("system:{}", text(&item["content"])),
                _ => "assistant".to_owned(),
            }
        })
        .collect::<Vec<_>>();
    labels.dedup();
    (system, labels)
}

#[test]
fn system_messages_encode_by_position() {
    let calls = |ids: &[&str]| {
        let parts = ids.iter().map(|id| {
            AssistantPart::ToolCall(ToolCall {
                id: ToolCallId::from(*id),
                name: "read".into(),
                arguments: json!({}),
            })
        });
        assistant("msg_1".to_owned(), parts.collect())
    };
    let hi = || user(vec![UserPart::Text { text: "hi".into() }]);
    let reminder = "Context window 0 has reached 50% of its threshold.";
    let prompt = "follow plan\n\nskills: none";
    let cases: [(&str, Vec<Message>, String, &[&str]); 5] = [
        (
            "leading system messages join the system prompt once",
            vec![system("lead"), system(" "), system("more"), hi()],
            format!("{prompt}\n\nlead\n\nmore"),
            &["user"],
        ),
        (
            // The regression: CleanWindow's reminder after a tool batch.
            "a later system message stays in place",
            vec![
                hi(),
                calls(&["c1"]),
                tool_result("c1", "ok", false),
                system(reminder),
            ],
            prompt.to_owned(),
            &[
                "user",
                "assistant",
                "results",
                &format!("system:{reminder}"),
            ],
        ),
        (
            "a hook's system message inside a tool batch follows the batch",
            vec![
                hi(),
                calls(&["c1", "c2"]),
                system("pre"),
                tool_result("c1", "ok", false),
                system("post"),
                tool_result("c2", "ok", true),
                hi(),
            ],
            prompt.to_owned(),
            &[
                "user",
                "assistant",
                "results",
                "system:pre",
                "system:post",
                "user",
            ],
        ),
        (
            "an empty later system message is dropped",
            vec![hi(), system("  "), hi()],
            prompt.to_owned(),
            &["user"],
        ),
        (
            "a system message after a reply is kept verbatim",
            vec![
                hi(),
                assistant("msg_2".to_owned(), Vec::new()),
                system(" spaced "),
            ],
            prompt.to_owned(),
            &["user", "system: spaced "],
        ),
    ];
    for (name, messages, system_prompt, labels) in cases {
        for api_kind in [
            ApiKind::AnthropicMessages,
            ApiKind::OpenAiResponses,
            ApiKind::OpenAiChat,
        ] {
            let body = encode(api_kind, &request(api_kind, messages.clone(), false), Top);
            assert_wire_valid(api_kind, &body);
            assert_eq!(
                transcript(api_kind, &body),
                (
                    system_prompt.clone(),
                    labels.iter().map(|label| (*label).to_owned()).collect()
                ),
                "{name} ({api_kind:?})"
            );
        }
    }
}

#[test]
fn responses_compaction_input_keeps_later_system_messages() {
    let hi = || user(vec![UserPart::Text { text: "hi".into() }]);
    let request = request(
        ApiKind::OpenAiResponses,
        vec![system("lead"), hi(), system("reminder")],
        false,
    );
    let body = openai_codec::encode_responses_compact_request(&ProviderCompactionRequest {
        session_id: request.session_id,
        model: request.model,
        compacted_prefix: Vec::new(),
        messages: request.messages,
        tools: request.tools,
        instructions: "Summarize".to_owned(),
    })
    .expect("encode compaction");
    let input = body["input"].as_array().expect("input");
    assert_eq!(input.len(), 2, "{input:?}");
    assert_eq!(input[0]["role"], "user");
    assert_eq!(input[1]["role"], "developer");
    assert_eq!(input[1]["content"][0]["text"], "reminder");
}
