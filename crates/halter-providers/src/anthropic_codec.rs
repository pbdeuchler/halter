// pattern: Functional Core

use base64::Engine;
use halter_protocol::{
    ApiKind, AssistantPart, BlockId, Message, MessageId, PromptSegment, ProviderRequest,
    ReasoningEffort, StopReason, StreamEvent, ToolResultMessage, Usage, UserPart,
};
use serde_json::{Map, Value, json};

use crate::codec_common::{normalized_tool_call_id, tool_name_for_provider, tool_result_text};

const CACHE_CONTROL_EPHEMERAL: &str = "ephemeral";

pub(crate) fn encode_request(request: &ProviderRequest, temperature: f32) -> anyhow::Result<Value> {
    if request.model.api_kind != ApiKind::AnthropicMessages {
        anyhow::bail!(
            "failed to encode anthropic request: unsupported api kind '{}'",
            request.model.api_kind as u8
        );
    }

    let mut body = Map::new();
    body.insert(
        "model".to_owned(),
        Value::String(request.model.model.clone()),
    );
    body.insert(
        "max_tokens".to_owned(),
        json!(request.model.max_output_tokens.unwrap_or(4096)),
    );
    body.insert("temperature".to_owned(), json!(temperature));
    body.insert(
        "messages".to_owned(),
        Value::Array(encode_messages(request)?),
    );

    if let Some(system) = encode_system_blocks(request) {
        body.insert("system".to_owned(), system);
    }
    if !request.tools.is_empty() {
        body.insert("tools".to_owned(), Value::Array(encode_tools(request)));
    }
    if let Some(thinking) =
        encode_thinking(request.model.reasoning, request.model.max_output_tokens)
    {
        body.insert("thinking".to_owned(), thinking);
    }

    Ok(Value::Object(body))
}

/// Encode the system field as either a flat string (when no breakpoints
/// land in this section) or an array of text blocks with `cache_control`
/// attached to the last block of each section the runtime asked us to
/// pin. Anthropic supports a maximum of four cache breakpoints per
/// request, which is exactly the shape the assembler emits.
fn encode_system_blocks(request: &ProviderRequest) -> Option<Value> {
    let breakpoints = request.prompt.cache_breakpoints;
    let want_blocks = breakpoints.after_system || breakpoints.after_skills;
    if !want_blocks {
        // Fast path: hand the provider one flat system string, identical
        // to the legacy behavior so callers that bypass the assembler
        // (and thus omit section metadata) keep working.
        return collect_system_text_legacy(request).map(Value::String);
    }

    let system_blob = render_segments(request.prompt.system_segments());
    let skill_blob = render_segments(request.prompt.skill_segments());
    let tail_blob = render_system_tail(request);

    let mut blocks: Vec<Value> = Vec::new();
    if let Some(text) = system_blob {
        blocks.push(text_block(text, breakpoints.after_system));
    }
    if let Some(text) = skill_blob {
        blocks.push(text_block(text, breakpoints.after_skills));
    }
    if let Some(text) = tail_blob {
        blocks.push(text_block(text, false));
    }
    if blocks.is_empty() {
        None
    } else {
        Some(Value::Array(blocks))
    }
}

/// Legacy single-string system field. Keeps backward compatibility for
/// requests assembled outside the runtime path (test fixtures, hand-built
/// `ProviderRequest`s).
fn collect_system_text_legacy(request: &ProviderRequest) -> Option<String> {
    let mut sections = Vec::new();
    let rendered_prefix = request.prompt.rendered_prefix.trim();
    if !rendered_prefix.is_empty() {
        sections.push(rendered_prefix.to_owned());
    }
    for message in &request.messages {
        if let Message::System(system) = message {
            let text = system.text.trim();
            if !text.is_empty() {
                sections.push(text.to_owned());
            }
        }
    }
    if sections.is_empty() {
        None
    } else {
        Some(sections.join("\n\n"))
    }
}

fn render_segments(segments: &[PromptSegment]) -> Option<String> {
    let combined = segments
        .iter()
        .map(|seg| seg.text.as_str().trim_end())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    if combined.is_empty() {
        None
    } else {
        Some(combined)
    }
}

/// Everything the assembler placed in `rendered_prefix` that does NOT
/// belong to the system or skills sections — append segments, summaries,
/// in-band compacted prefix items, and any `Message::System` payloads.
fn render_system_tail(request: &ProviderRequest) -> Option<String> {
    let mut sections = Vec::new();
    if let Some(text) = render_segments(request.prompt.append_segments()) {
        sections.push(text);
    }
    for message in &request.messages {
        if let Message::System(system) = message {
            let text = system.text.trim();
            if !text.is_empty() {
                sections.push(text.to_owned());
            }
        }
    }
    if sections.is_empty() {
        None
    } else {
        Some(sections.join("\n\n"))
    }
}

fn text_block(text: String, cache_breakpoint: bool) -> Value {
    let mut block = Map::new();
    block.insert("type".to_owned(), Value::String("text".to_owned()));
    block.insert("text".to_owned(), Value::String(text));
    if cache_breakpoint {
        block.insert(
            "cache_control".to_owned(),
            json!({ "type": CACHE_CONTROL_EPHEMERAL }),
        );
    }
    Value::Object(block)
}

pub(crate) fn decode_response(
    request: &ProviderRequest,
    response: &Value,
) -> anyhow::Result<Vec<StreamEvent>> {
    let message_id = response
        .get("id")
        .and_then(Value::as_str)
        .map(|id| MessageId::from(id.to_owned()))
        .unwrap_or_default();
    let mut events = vec![StreamEvent::MessageStart {
        id: message_id.clone(),
    }];

    if let Some(content) = response.get("content").and_then(Value::as_array) {
        for block in content {
            match block
                .get("type")
                .and_then(Value::as_str)
                .unwrap_or_default()
            {
                "text" => {
                    if let Some(text) = block.get("text").and_then(Value::as_str) {
                        let block_id = BlockId::new();
                        events.push(StreamEvent::TextStart {
                            id: block_id.clone(),
                        });
                        events.push(StreamEvent::TextDelta {
                            id: block_id.clone(),
                            delta: text.to_owned(),
                        });
                        events.push(StreamEvent::TextEnd { id: block_id });
                    }
                }
                "thinking" => {
                    if let Some(text) = block
                        .get("thinking")
                        .and_then(Value::as_str)
                        .or_else(|| block.get("text").and_then(Value::as_str))
                    {
                        let block_id = BlockId::new();
                        events.push(StreamEvent::ThinkingStart {
                            id: block_id.clone(),
                        });
                        events.push(StreamEvent::ThinkingDelta {
                            id: block_id.clone(),
                            delta: text.to_owned(),
                        });
                        events.push(StreamEvent::ThinkingEnd {
                            id: block_id,
                            signature: block
                                .get("signature")
                                .and_then(Value::as_str)
                                .map(str::to_owned),
                        });
                    }
                }
                "tool_use" => {
                    let block_id = BlockId::new();
                    let tool_call_id = block
                        .get("id")
                        .and_then(Value::as_str)
                        .map(|id| id.into())
                        .unwrap_or_default();
                    let tool_name = block
                        .get("name")
                        .and_then(Value::as_str)
                        .map(|name| {
                            crate::codec_common::canonical_tool_name(
                                name,
                                &request.tools,
                                request.model.provider_kind,
                            )
                        })
                        .unwrap_or_else(|| "tool".into());
                    let arguments = block
                        .get("input")
                        .cloned()
                        .unwrap_or_else(|| json!({}))
                        .to_string();
                    events.push(StreamEvent::ToolCallStart {
                        id: block_id.clone(),
                        tool_call_id,
                        name: tool_name,
                    });
                    events.push(StreamEvent::ToolArgsDelta {
                        id: block_id.clone(),
                        delta: arguments,
                    });
                    events.push(StreamEvent::ToolCallEnd { id: block_id });
                }
                _ => {}
            }
        }
    }

    let usage = decode_usage(response);
    events.push(StreamEvent::UsageUpdate { usage });
    events.push(StreamEvent::MessageEnd {
        id: message_id,
        stop_reason: decode_stop_reason(response),
        response_id: None,
    });
    Ok(events)
}

fn encode_messages(request: &ProviderRequest) -> anyhow::Result<Vec<Value>> {
    let mut encoded = Vec::new();
    let mut pending_tool_results = Vec::new();

    for message in &request.messages {
        match message {
            Message::System(_) => {}
            Message::User(user) => {
                flush_tool_results(&mut encoded, &mut pending_tool_results);
                encoded.push(json!({
                    "role": "user",
                    "content": encode_user_parts(&user.parts),
                }));
            }
            Message::Assistant(assistant) => {
                flush_tool_results(&mut encoded, &mut pending_tool_results);
                let content = encode_assistant_parts(request, assistant)?;
                if !content.is_empty() {
                    encoded.push(json!({
                        "role": "assistant",
                        "content": content,
                    }));
                }
            }
            Message::Tool(tool) => pending_tool_results.push(encode_tool_result(tool)),
        }
    }

    flush_tool_results(&mut encoded, &mut pending_tool_results);

    if request.prompt.cache_breakpoints.after_user_prompt {
        attach_cache_breakpoint_to_last_user_message(&mut encoded);
    }
    Ok(encoded)
}

/// Find the most recent user-role message in the encoded payload and put
/// `cache_control: ephemeral` on its last content block. The Anthropic
/// docs are explicit: cache_control belongs on the last block of the
/// section you want pinned, not on the message envelope.
fn attach_cache_breakpoint_to_last_user_message(messages: &mut [Value]) {
    let Some(message) = messages.iter_mut().rev().find(|message| {
        message
            .get("role")
            .and_then(Value::as_str)
            .is_some_and(|role| role == "user")
    }) else {
        return;
    };
    let Some(content) = message.get_mut("content").and_then(Value::as_array_mut) else {
        return;
    };
    let Some(last_block) = content.last_mut() else {
        return;
    };
    if let Value::Object(map) = last_block {
        map.insert(
            "cache_control".to_owned(),
            json!({ "type": CACHE_CONTROL_EPHEMERAL }),
        );
    }
}

fn encode_user_parts(parts: &[UserPart]) -> Vec<Value> {
    parts
        .iter()
        .map(|part| match part {
            UserPart::Text { text } => json!({
                "type": "text",
                "text": text,
            }),
            UserPart::Image { media_type, data } => json!({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64::engine::general_purpose::STANDARD.encode(data),
                }
            }),
            UserPart::Document { media_type, data } => json!({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64::engine::general_purpose::STANDARD.encode(data),
                }
            }),
        })
        .collect()
}

fn encode_assistant_parts(
    request: &ProviderRequest,
    message: &halter_protocol::AssistantMessage,
) -> anyhow::Result<Vec<Value>> {
    let mut content = Vec::new();

    for part in &message.parts {
        match part {
            AssistantPart::Text { text } if !text.is_empty() => content.push(json!({
                "type": "text",
                "text": text,
            })),
            AssistantPart::Text { .. } | AssistantPart::Thinking(_) => {}
            AssistantPart::ToolCall(call) => {
                content.push(json!({
                    "type": "tool_use",
                    "id": normalized_tool_call_id(&call.id),
                    "name": tool_name_for_provider(
                        &call.name,
                        &request.tools,
                        request.model.provider_kind,
                    ),
                    "input": call.arguments,
                }));
            }
        }
    }

    Ok(content)
}

fn encode_tool_result(message: &ToolResultMessage) -> Value {
    json!({
        "type": "tool_result",
        "tool_use_id": normalized_tool_call_id(&message.call_id),
        "content": tool_result_text(&message.content, &message.error),
        "is_error": message.error.is_some(),
    })
}

fn flush_tool_results(encoded: &mut Vec<Value>, pending_tool_results: &mut Vec<Value>) {
    if pending_tool_results.is_empty() {
        return;
    }

    encoded.push(json!({
        "role": "user",
        "content": std::mem::take(pending_tool_results),
    }));
}

fn encode_tools(request: &ProviderRequest) -> Vec<Value> {
    let last_index = request.tools.len().saturating_sub(1);
    let pin_last = request.prompt.cache_breakpoints.after_tools;
    request
        .tools
        .iter()
        .enumerate()
        .map(|(index, tool)| {
            let mut spec = json!({
                "name": tool_name_for_provider(&tool.name, &request.tools, request.model.provider_kind),
                "description": tool.description,
                "input_schema": tool.input_schema,
            });
            if pin_last && index == last_index && let Value::Object(map) = &mut spec {
                map.insert(
                    "cache_control".to_owned(),
                    json!({ "type": CACHE_CONTROL_EPHEMERAL }),
                );
            }
            spec
        })
        .collect()
}

fn encode_thinking(
    reasoning: Option<ReasoningEffort>,
    max_output_tokens: Option<u32>,
) -> Option<Value> {
    let reasoning = reasoning?;
    let Some(max_output_tokens) = max_output_tokens else {
        tracing::warn!(
            ?reasoning,
            "reasoning requested but max_output_tokens is unset; anthropic thinking block dropped silently. Set max_output_tokens or remove reasoning effort."
        );
        return None;
    };
    if max_output_tokens <= 1_024 {
        return None;
    }

    let desired_budget = match reasoning {
        ReasoningEffort::Low => 1_024,
        ReasoningEffort::Medium => 4_096,
        ReasoningEffort::High => 8_192,
        ReasoningEffort::Xhigh => 8_192,
    };
    let budget_tokens = desired_budget.min(max_output_tokens.saturating_sub(1));
    if budget_tokens < 1_024 {
        return None;
    }

    Some(json!({
        "type": "enabled",
        "budget_tokens": budget_tokens,
    }))
}

fn decode_stop_reason(response: &Value) -> StopReason {
    match response
        .get("stop_reason")
        .and_then(Value::as_str)
        .unwrap_or("end_turn")
    {
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        "pause_turn" => StopReason::Interrupted,
        "refusal" => StopReason::Error,
        _ => StopReason::EndTurn,
    }
}

fn decode_usage(response: &Value) -> Usage {
    Usage {
        input_tokens: response
            .pointer("/usage/input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or_default(),
        output_tokens: response
            .pointer("/usage/output_tokens")
            .and_then(Value::as_u64)
            .unwrap_or_default(),
        cache_creation_input_tokens: response
            .pointer("/usage/cache_creation_input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or_default(),
        cache_read_input_tokens: response
            .pointer("/usage/cache_read_input_tokens")
            .and_then(Value::as_u64)
            .unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use chrono::Utc;
    use halter_protocol::{
        ApiKind, AssembledPrompt, AssistantMessage, AssistantPart, CacheBreakpoints, CacheScope,
        Message, MessageId, ModelId, ModelRole, PromptSegment, PromptSegmentId, PromptSegmentKind,
        ProviderKind, ProviderName, ProviderRequest, ResolvedModel, ToolAlias, ToolCall,
        ToolCallId, ToolCapabilities, ToolConcurrency, ToolSpec, TurnId, UserMessage, Volatility,
    };
    use indexmap::IndexMap;
    use serde_json::json;

    use super::*;

    #[test]
    fn request_hoists_system_and_groups_tool_results() {
        let request = sample_request(vec![
            Message::User(UserMessage {
                id: MessageId::new(),
                created_at: Utc::now(),
                parts: vec![
                    UserPart::Text {
                        text: "look".to_owned(),
                    },
                    UserPart::Document {
                        media_type: "application/pdf".to_owned(),
                        data: Bytes::from_static(b"pdf"),
                    },
                ],
            }),
            Message::Assistant(AssistantMessage {
                id: MessageId::new(),
                created_at: Utc::now(),
                parts: vec![AssistantPart::ToolCall(ToolCall {
                    id: ToolCallId::from("call with spaces"),
                    name: "read".into(),
                    arguments: json!({"path": "README.md"}),
                })],
                stop_reason: None,
                usage: None,
                replay_meta: Default::default(),
            }),
            Message::Tool(ToolResultMessage {
                id: MessageId::new(),
                call_id: ToolCallId::from("call with spaces"),
                content: halter_protocol::ToolResult::Text {
                    text: "done".to_owned(),
                },
                error: None,
                created_at: Utc::now(),
            }),
        ]);

        let body =
            encode_request(&request, halter_protocol::DEFAULT_TEMPERATURE).expect("encode request");

        assert_eq!(
            body.get("system").and_then(Value::as_str),
            Some("follow plan")
        );
        assert_eq!(body["messages"].as_array().expect("messages").len(), 3);
        assert_eq!(body["messages"][1]["content"][0]["name"], "fs_read");
        assert_eq!(
            body["messages"][2]["content"][0]["tool_use_id"],
            "call_with_spaces"
        );
        assert_eq!(
            body["temperature"],
            json!(halter_protocol::DEFAULT_TEMPERATURE)
        );
    }

    #[test]
    fn anthropic_request_forwards_configured_temperature_override() {
        let request = sample_request(Vec::new());
        let body = encode_request(&request, 0.25).expect("encode request");

        assert_eq!(body["temperature"], json!(0.25_f32));
    }

    #[test]
    fn response_maps_text_and_tool_use_blocks() {
        let request = sample_request(Vec::new());
        let response = json!({
            "id": "msg_123",
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 11,
                "output_tokens": 7,
            },
            "content": [
                { "type": "text", "text": "working" },
                { "type": "tool_use", "id": "toolu_123", "name": "fs_read", "input": { "path": "README.md" } }
            ]
        });

        let events = decode_response(&request, &response).expect("decode response");

        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::TextDelta { delta, .. } if delta == "working"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::ToolCallStart { name, .. } if name.0 == "read"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::MessageEnd {
                stop_reason: StopReason::ToolUse,
                ..
            }
        )));
    }

    fn sample_request(messages: Vec<Message>) -> ProviderRequest {
        let mut provider_aliases = IndexMap::new();
        provider_aliases.insert(ProviderKind::Anthropic, ToolAlias::from("fs_read"));
        ProviderRequest {
            session_id: Default::default(),
            turn_id: TurnId::new(),
            model: ResolvedModel {
                role: ModelRole::default(),
                id: ModelId::from("claude_default"),
                provider: ProviderName::from("anthropic"),
                provider_kind: ProviderKind::Anthropic,
                api_kind: ApiKind::AnthropicMessages,
                model: "claude-sonnet-4-5".to_owned(),
                max_input_tokens: Some(200_000),
                max_output_tokens: Some(8_192),
                reasoning: None,
                tokens_per_minute: None,
            },
            prompt: AssembledPrompt {
                segments: vec![PromptSegment {
                    id: PromptSegmentId::new(),
                    text: "follow plan".to_owned(),
                    volatility: Volatility::Static,
                    cache_scope: CacheScope::PrefixCacheable,
                    content_hash: "hash".to_owned(),
                    kind: PromptSegmentKind::System,
                }],
                transcript: messages.clone(),
                ordered_segments: Vec::new(),
                prefix_cache_key: "cache-key".to_owned(),
                rendered_prefix: "follow plan".to_owned(),
                rendered_transcript: String::new(),
                rendered: String::new(),
                cache_breakpoints: CacheBreakpoints::default(),
                system_segment_count: 0,
                skill_segment_count: 0,
            },
            compacted_prefix: Vec::new(),
            messages,
            tools: vec![ToolSpec {
                name: "read".into(),
                description: "Read a file".to_owned(),
                input_schema: json!({"type": "object"}),
                concurrency: ToolConcurrency::ReadOnly,
                capabilities: ToolCapabilities::default(),
                provider_aliases,
            }],
            previous_response_id: None,
            new_messages_start: 0,
        }
    }
}
