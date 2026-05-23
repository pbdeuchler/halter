// pattern: Functional Core

use std::collections::BTreeMap;

use base64::Engine;
use halter_protocol::{
    ApiKind, AssistantPart, BlockId, Message, MessageId, PromptSegment, ProviderCompactionRequest,
    ProviderCompactionResponse, ProviderKind, ProviderRequest, ReasoningEffort, StopReason,
    StreamEvent, ToolCallId, ToolName, ToolResultMessage, ToolSpec, Usage, UserPart,
};
use serde_json::{Map, Value, json};

use crate::codec_common::{
    canonical_tool_name, normalized_tool_call_id, tool_name_for_provider, tool_result_text,
};

const CACHE_CONTROL_EPHEMERAL: &str = "ephemeral";
const COMPACTED_CONTEXT_TYPE: &str = "halter_compacted_context";
const COMPACTED_CONTEXT_OPEN: &str = "<compacted_context>";
const COMPACTED_CONTEXT_CLOSE: &str = "</compacted_context>";

#[derive(Debug, Clone, Copy)]
struct AnthropicRequestOptions {
    stream: bool,
}

#[cfg(test)]
pub(crate) fn encode_request(
    request: &ProviderRequest,
    temperature: Option<f32>,
) -> anyhow::Result<Value> {
    encode_request_with_options(
        request,
        temperature,
        AnthropicRequestOptions { stream: false },
    )
}

pub(crate) fn encode_stream_request(
    request: &ProviderRequest,
    temperature: Option<f32>,
) -> anyhow::Result<Value> {
    encode_request_with_options(
        request,
        temperature,
        AnthropicRequestOptions { stream: true },
    )
}

fn encode_request_with_options(
    request: &ProviderRequest,
    temperature: Option<f32>,
    options: AnthropicRequestOptions,
) -> anyhow::Result<Value> {
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
    if let Some(temperature) = temperature {
        body.insert("temperature".to_owned(), json!(temperature));
    }
    body.insert(
        "messages".to_owned(),
        Value::Array(encode_messages(request)?),
    );
    if options.stream {
        body.insert("stream".to_owned(), Value::Bool(true));
    }

    if let Some(system) = encode_system_blocks(request) {
        body.insert("system".to_owned(), system);
    }
    if !request.tools.is_empty() {
        body.insert("tools".to_owned(), Value::Array(encode_tools(request)));
    }
    if let Some(thinking) = encode_thinking(
        &request.model.model,
        request.model.reasoning,
        request.model.max_output_tokens,
    ) {
        if let Some(output_config) = thinking.output_config {
            body.insert("output_config".to_owned(), output_config);
        }
        body.insert("thinking".to_owned(), thinking.thinking);
    }

    Ok(Value::Object(body))
}

pub(crate) fn encode_compaction_request(
    request: &ProviderCompactionRequest,
    temperature: Option<f32>,
) -> anyhow::Result<Value> {
    if request.model.api_kind != ApiKind::AnthropicMessages {
        anyhow::bail!("failed to encode anthropic compaction request: unsupported api kind");
    }

    let mut system_sections = vec![request.instructions.trim().to_owned()];
    if let Some(compacted) = render_compacted_prefix(&request.compacted_prefix) {
        system_sections.push(compacted);
    }
    let system = system_sections
        .into_iter()
        .filter(|section| !section.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");

    let mut body = Map::new();
    body.insert(
        "model".to_owned(),
        Value::String(request.model.model.clone()),
    );
    body.insert(
        "max_tokens".to_owned(),
        json!(request.model.max_output_tokens.unwrap_or(4096)),
    );
    if let Some(temperature) = temperature {
        body.insert("temperature".to_owned(), json!(temperature));
    }
    body.insert("system".to_owned(), Value::String(system));
    body.insert(
        "messages".to_owned(),
        json!([
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": render_compaction_messages(&request.messages),
                    }
                ],
            }
        ]),
    );

    Ok(Value::Object(body))
}

pub(crate) fn decode_compaction_response(
    response: &Value,
) -> anyhow::Result<ProviderCompactionResponse> {
    let text = response_text(response);
    if text.trim().is_empty() {
        return Ok(ProviderCompactionResponse {
            output: Vec::new(),
            usage: decode_usage(response),
        });
    }

    Ok(ProviderCompactionResponse {
        output: vec![compacted_context_item(text)],
        usage: decode_usage(response),
    })
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
    if let Some(text) = render_compacted_prefix(&request.compacted_prefix) {
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

#[cfg(test)]
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
            AssistantPart::Thinking(block) => {
                let mut encoded = Map::new();
                encoded.insert("type".to_owned(), json!("thinking"));
                encoded.insert("thinking".to_owned(), json!(block.text));
                if let Some(signature) = &block.signature {
                    encoded.insert("signature".to_owned(), json!(signature));
                }
                content.push(Value::Object(encoded));
            }
            AssistantPart::Text { .. } => {}
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

#[derive(Debug, Clone)]
struct EncodedThinking {
    thinking: Value,
    output_config: Option<Value>,
}

fn encode_thinking(
    model: &str,
    reasoning: Option<ReasoningEffort>,
    max_output_tokens: Option<u32>,
) -> Option<EncodedThinking> {
    let reasoning = reasoning?;
    if prefers_adaptive_thinking(model) {
        let mut output_config = Map::new();
        output_config.insert(
            "effort".to_owned(),
            json!(adaptive_effort_for_model(model, reasoning)),
        );
        return Some(EncodedThinking {
            thinking: json!({
                "type": "adaptive",
                "display": "summarized",
            }),
            output_config: Some(Value::Object(output_config)),
        });
    }

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

    Some(EncodedThinking {
        thinking: json!({
            "type": "enabled",
            "budget_tokens": budget_tokens,
        }),
        output_config: None,
    })
}

fn prefers_adaptive_thinking(model: &str) -> bool {
    let model = model.to_ascii_lowercase();
    model.contains("claude-opus-4-7")
        || model.contains("claude-opus-4-6")
        || model.contains("claude-sonnet-4-6")
        || model.contains("claude-mythos")
}

fn adaptive_effort_for_model(model: &str, reasoning: ReasoningEffort) -> &'static str {
    match reasoning {
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::Xhigh if model.to_ascii_lowercase().contains("claude-opus-4-7") => "xhigh",
        ReasoningEffort::Xhigh => "high",
    }
}

#[cfg(test)]
fn decode_stop_reason(response: &Value) -> StopReason {
    decode_stop_reason_value(
        response
            .get("stop_reason")
            .and_then(Value::as_str)
            .unwrap_or("end_turn"),
    )
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

#[derive(Debug, Clone)]
pub(crate) struct AnthropicStreamDecoder {
    tool_specs: Vec<ToolSpec>,
    provider_kind: ProviderKind,
    message_id: Option<MessageId>,
    stop_reason: StopReason,
    usage: Usage,
    blocks: BTreeMap<usize, AnthropicStreamBlock>,
}

#[derive(Debug, Clone)]
enum AnthropicStreamBlock {
    Text {
        id: BlockId,
    },
    Thinking {
        id: BlockId,
        signature: Option<String>,
    },
    ToolUse {
        id: BlockId,
    },
}

impl AnthropicStreamDecoder {
    pub(crate) fn new(request: &ProviderRequest) -> Self {
        Self {
            tool_specs: request.tools.clone(),
            provider_kind: request.model.provider_kind,
            message_id: None,
            stop_reason: StopReason::EndTurn,
            usage: Usage::default(),
            blocks: BTreeMap::new(),
        }
    }

    pub(crate) fn decode(&mut self, event: &Value) -> anyhow::Result<Vec<StreamEvent>> {
        let event_type = event
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default();
        match event_type {
            "message_start" => Ok(self.decode_message_start(event)),
            "content_block_start" => self.decode_content_block_start(event),
            "content_block_delta" => self.decode_content_block_delta(event),
            "content_block_stop" => self.decode_content_block_stop(event),
            "message_delta" => Ok(self.decode_message_delta(event)),
            "message_stop" => Ok(self.decode_message_stop()),
            "ping" => Ok(Vec::new()),
            "error" => {
                let message = event
                    .pointer("/error/message")
                    .and_then(Value::as_str)
                    .unwrap_or("anthropic stream error");
                Ok(vec![StreamEvent::Error {
                    error: halter_protocol::ProviderError::new(
                        format!("failed to execute provider request: {message}"),
                        false,
                    ),
                }])
            }
            _ => Ok(Vec::new()),
        }
    }

    fn decode_message_start(&mut self, event: &Value) -> Vec<StreamEvent> {
        let message_id = event
            .pointer("/message/id")
            .and_then(Value::as_str)
            .map(|id| MessageId::from(id.to_owned()))
            .unwrap_or_default();
        self.message_id = Some(message_id.clone());
        self.usage = decode_usage(event.pointer("/message").unwrap_or(event));
        vec![StreamEvent::MessageStart { id: message_id }]
    }

    fn decode_content_block_start(&mut self, event: &Value) -> anyhow::Result<Vec<StreamEvent>> {
        let index = event_index(event)?;
        let content_block = event.get("content_block").unwrap_or(event);
        match content_block
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "text" => {
                let id = BlockId::new();
                self.blocks
                    .insert(index, AnthropicStreamBlock::Text { id: id.clone() });
                let mut events = vec![StreamEvent::TextStart { id: id.clone() }];
                if let Some(text) = content_block.get("text").and_then(Value::as_str)
                    && !text.is_empty()
                {
                    events.push(StreamEvent::TextDelta {
                        id,
                        delta: text.to_owned(),
                    });
                }
                Ok(events)
            }
            "thinking" => {
                let id = BlockId::new();
                let signature = content_block
                    .get("signature")
                    .and_then(Value::as_str)
                    .filter(|signature| !signature.is_empty())
                    .map(str::to_owned);
                self.blocks.insert(
                    index,
                    AnthropicStreamBlock::Thinking {
                        id: id.clone(),
                        signature,
                    },
                );
                let mut events = vec![StreamEvent::ThinkingStart { id: id.clone() }];
                if let Some(text) = content_block.get("thinking").and_then(Value::as_str)
                    && !text.is_empty()
                {
                    events.push(StreamEvent::ThinkingDelta {
                        id,
                        delta: text.to_owned(),
                    });
                }
                Ok(events)
            }
            "tool_use" => {
                let id = BlockId::new();
                let tool_call_id = content_block
                    .get("id")
                    .and_then(Value::as_str)
                    .map(ToolCallId::from)
                    .unwrap_or_default();
                let name = content_block
                    .get("name")
                    .and_then(Value::as_str)
                    .map(|name| canonical_tool_name(name, &self.tool_specs, self.provider_kind))
                    .unwrap_or_else(|| ToolName::from("tool"));
                self.blocks
                    .insert(index, AnthropicStreamBlock::ToolUse { id: id.clone() });
                let mut events = vec![StreamEvent::ToolCallStart {
                    id: id.clone(),
                    tool_call_id,
                    name,
                }];
                if let Some(input) = content_block.get("input")
                    && input.as_object().is_some_and(|object| !object.is_empty())
                {
                    events.push(StreamEvent::ToolArgsDelta {
                        id,
                        delta: input.to_string(),
                    });
                }
                Ok(events)
            }
            _ => Ok(Vec::new()),
        }
    }

    fn decode_content_block_delta(&mut self, event: &Value) -> anyhow::Result<Vec<StreamEvent>> {
        let index = event_index(event)?;
        let Some(block) = self.blocks.get_mut(&index) else {
            return Ok(Vec::new());
        };
        let delta = event.get("delta").unwrap_or(event);
        match delta
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or_default()
        {
            "text_delta" => match block {
                AnthropicStreamBlock::Text { id } => Ok(delta
                    .get("text")
                    .and_then(Value::as_str)
                    .filter(|text| !text.is_empty())
                    .map(|text| {
                        vec![StreamEvent::TextDelta {
                            id: id.clone(),
                            delta: text.to_owned(),
                        }]
                    })
                    .unwrap_or_default()),
                AnthropicStreamBlock::Thinking { .. } | AnthropicStreamBlock::ToolUse { .. } => {
                    Ok(Vec::new())
                }
            },
            "thinking_delta" => match block {
                AnthropicStreamBlock::Thinking { id, .. } => Ok(delta
                    .get("thinking")
                    .and_then(Value::as_str)
                    .filter(|text| !text.is_empty())
                    .map(|text| {
                        vec![StreamEvent::ThinkingDelta {
                            id: id.clone(),
                            delta: text.to_owned(),
                        }]
                    })
                    .unwrap_or_default()),
                AnthropicStreamBlock::Text { .. } | AnthropicStreamBlock::ToolUse { .. } => {
                    Ok(Vec::new())
                }
            },
            "signature_delta" => {
                if let AnthropicStreamBlock::Thinking { signature, .. } = block {
                    *signature = delta
                        .get("signature")
                        .and_then(Value::as_str)
                        .map(str::to_owned);
                }
                Ok(Vec::new())
            }
            "input_json_delta" => match block {
                AnthropicStreamBlock::ToolUse { id } => Ok(delta
                    .get("partial_json")
                    .and_then(Value::as_str)
                    .filter(|text| !text.is_empty())
                    .map(|text| {
                        vec![StreamEvent::ToolArgsDelta {
                            id: id.clone(),
                            delta: text.to_owned(),
                        }]
                    })
                    .unwrap_or_default()),
                AnthropicStreamBlock::Text { .. } | AnthropicStreamBlock::Thinking { .. } => {
                    Ok(Vec::new())
                }
            },
            _ => Ok(Vec::new()),
        }
    }

    fn decode_content_block_stop(&mut self, event: &Value) -> anyhow::Result<Vec<StreamEvent>> {
        let index = event_index(event)?;
        let Some(block) = self.blocks.remove(&index) else {
            return Ok(Vec::new());
        };
        Ok(match block {
            AnthropicStreamBlock::Text { id } => vec![StreamEvent::TextEnd { id }],
            AnthropicStreamBlock::Thinking { id, signature } => {
                vec![StreamEvent::ThinkingEnd { id, signature }]
            }
            AnthropicStreamBlock::ToolUse { id } => vec![StreamEvent::ToolCallEnd { id }],
        })
    }

    fn decode_message_delta(&mut self, event: &Value) -> Vec<StreamEvent> {
        if let Some(stop_reason) = event.pointer("/delta/stop_reason").and_then(Value::as_str) {
            self.stop_reason = decode_stop_reason_value(stop_reason);
        }
        if let Some(usage) = event.get("usage") {
            self.usage.output_tokens = usage
                .get("output_tokens")
                .and_then(Value::as_u64)
                .unwrap_or(self.usage.output_tokens);
        }
        vec![StreamEvent::UsageUpdate {
            usage: self.usage.clone(),
        }]
    }

    fn decode_message_stop(&mut self) -> Vec<StreamEvent> {
        vec![StreamEvent::MessageEnd {
            id: self.message_id.clone().unwrap_or_default(),
            stop_reason: self.stop_reason,
            response_id: None,
        }]
    }
}

fn event_index(event: &Value) -> anyhow::Result<usize> {
    event
        .get("index")
        .and_then(Value::as_u64)
        .map(|index| index as usize)
        .ok_or_else(|| anyhow::anyhow!("failed to decode anthropic stream event: missing index"))
}

fn decode_stop_reason_value(value: &str) -> StopReason {
    match value {
        "tool_use" => StopReason::ToolUse,
        "max_tokens" => StopReason::MaxTokens,
        "pause_turn" => StopReason::Interrupted,
        "refusal" => StopReason::Error,
        _ => StopReason::EndTurn,
    }
}

fn response_text(response: &Value) -> String {
    response
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|block| match block.get("type").and_then(Value::as_str) {
            Some("text") => block.get("text").and_then(Value::as_str),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn compacted_context_item(text: String) -> Value {
    json!({
        "type": COMPACTED_CONTEXT_TYPE,
        "text": text,
    })
}

fn render_compacted_prefix(items: &[Value]) -> Option<String> {
    if items.is_empty() {
        return None;
    }
    let rendered = items
        .iter()
        .map(render_compacted_prefix_item)
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    if rendered.trim().is_empty() {
        None
    } else {
        Some(format!(
            "{COMPACTED_CONTEXT_OPEN}\n{rendered}\n{COMPACTED_CONTEXT_CLOSE}"
        ))
    }
}

fn render_compacted_prefix_item(item: &Value) -> String {
    if item.get("type").and_then(Value::as_str) == Some(COMPACTED_CONTEXT_TYPE)
        && let Some(text) = item.get("text").and_then(Value::as_str)
    {
        return text.to_owned();
    }
    item.to_string()
}

fn render_compaction_messages(messages: &[Message]) -> String {
    if messages.is_empty() {
        return "No prior messages are eligible for compaction.".to_owned();
    }
    messages
        .iter()
        .map(render_compaction_message)
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn render_compaction_message(message: &Message) -> String {
    match message {
        Message::System(system) => format!("system:\n{}", system.text),
        Message::User(user) => format!("user:\n{}", render_user_parts(&user.parts)),
        Message::Assistant(assistant) => {
            let parts = assistant
                .parts
                .iter()
                .map(|part| match part {
                    AssistantPart::Text { text } => text.clone(),
                    AssistantPart::Thinking(block) => format!("[thinking]\n{}", block.text),
                    AssistantPart::ToolCall(call) => {
                        format!("[tool_call {}]\n{}", call.name, call.arguments)
                    }
                })
                .collect::<Vec<_>>()
                .join("\n");
            format!("assistant:\n{parts}")
        }
        Message::Tool(tool) => format!(
            "tool_result {}:\n{}",
            tool.call_id,
            tool_result_text(&tool.content, &tool.error)
        ),
    }
}

fn render_user_parts(parts: &[UserPart]) -> String {
    parts
        .iter()
        .map(|part| match part {
            UserPart::Text { text } => text.clone(),
            UserPart::Image { media_type, .. } => format!("[image: {media_type}]"),
            UserPart::Document { media_type, .. } => format!("[document: {media_type}]"),
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;
    use chrono::Utc;
    use halter_protocol::{
        ApiKind, AssembledPrompt, AssistantMessage, AssistantPart, CacheBreakpoints, CacheScope,
        Message, MessageId, ModelId, ModelRole, PromptSegment, PromptSegmentId, PromptSegmentKind,
        ProviderCompactionRequest, ProviderKind, ProviderName, ProviderRequest, ReasoningEffort,
        ResolvedModel, ToolAlias, ToolCall, ToolCallId, ToolCapabilities, ToolConcurrency,
        ToolResult, ToolSpec, TurnId, UserMessage, Volatility,
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

        let body = encode_request(&request, None).expect("encode request");

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
        assert!(body.get("temperature").is_none());
    }

    #[test]
    fn anthropic_request_forwards_configured_temperature_override() {
        let request = sample_request(Vec::new());
        let body = encode_request(&request, Some(0.25)).expect("encode request");

        assert_eq!(body["temperature"], json!(0.25_f32));
    }

    #[test]
    fn stream_request_enables_sse_and_renders_compacted_prefix_after_cache_boundaries() {
        let mut request = sample_request(Vec::new());
        request.prompt.ordered_segments = request.prompt.segments.clone();
        request.prompt.system_segment_count = 1;
        request.prompt.cache_breakpoints.after_system = true;
        request.compacted_prefix = vec![json!({
            "type": COMPACTED_CONTEXT_TYPE,
            "text": "older decisions are summarized",
        })];

        let body = encode_stream_request(&request, None).expect("encode");

        assert_eq!(body["stream"], true);
        let system = body["system"].as_array().expect("system blocks");
        assert_eq!(system[0]["cache_control"]["type"], CACHE_CONTROL_EPHEMERAL);
        assert!(
            system
                .last()
                .and_then(|block| block.get("text"))
                .and_then(Value::as_str)
                .is_some_and(|text| text.contains("older decisions are summarized"))
        );
    }

    #[test]
    fn request_replays_thinking_blocks_with_signatures() {
        let request = sample_request(vec![Message::Assistant(AssistantMessage {
            id: MessageId::new(),
            created_at: Utc::now(),
            parts: vec![
                AssistantPart::Thinking(halter_protocol::ThinkingBlock {
                    text: "reasoned".to_owned(),
                    signature: Some("sig-123".to_owned()),
                }),
                AssistantPart::Text {
                    text: "answer".to_owned(),
                },
            ],
            stop_reason: None,
            usage: None,
            replay_meta: Default::default(),
        })]);

        let body = encode_request(&request, None).expect("encode");

        assert_eq!(body["messages"][0]["content"][0]["type"], "thinking");
        assert_eq!(body["messages"][0]["content"][0]["thinking"], "reasoned");
        assert_eq!(body["messages"][0]["content"][0]["signature"], "sig-123");
    }

    #[test]
    fn adaptive_models_use_current_anthropic_thinking_shape() {
        let mut request = sample_request(Vec::new());
        request.model.model = "claude-opus-4-7-latest".to_owned();
        request.model.reasoning = Some(ReasoningEffort::Xhigh);

        let body = encode_stream_request(&request, None).expect("encode");

        assert_eq!(body["thinking"]["type"], "adaptive");
        assert_eq!(body["thinking"]["display"], "summarized");
        assert_eq!(body["output_config"]["effort"], "xhigh");
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

    #[test]
    fn stream_decoder_maps_text_thinking_tool_use_usage_and_stop_reason() {
        let request = sample_request(Vec::new());
        let mut decoder = AnthropicStreamDecoder::new(&request);
        let mut events = Vec::new();
        for event in [
            json!({
                "type": "message_start",
                "message": {
                    "id": "msg_123",
                    "usage": {
                        "input_tokens": 11,
                        "output_tokens": 1,
                        "cache_creation_input_tokens": 2,
                        "cache_read_input_tokens": 3,
                    }
                }
            }),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "done"}}),
            json!({"type": "content_block_stop", "index": 0}),
            json!({"type": "content_block_start", "index": 1, "content_block": {"type": "thinking", "thinking": ""}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "thinking_delta", "thinking": "think"}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "signature_delta", "signature": "sig-123"}}),
            json!({"type": "content_block_stop", "index": 1}),
            json!({"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "toolu_123", "name": "fs_read", "input": {}}}),
            json!({"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": "{\"path\""}}),
            json!({"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": ":\"README.md\"}"}}),
            json!({"type": "content_block_stop", "index": 2}),
            json!({"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 7}}),
            json!({"type": "message_stop"}),
        ] {
            events.extend(decoder.decode(&event).expect("decode event"));
        }

        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::MessageStart { id } if id.0 == "msg_123"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::TextDelta { delta, .. } if delta == "done"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::ThinkingDelta { delta, .. } if delta == "think"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::ThinkingEnd { signature, .. } if signature.as_deref() == Some("sig-123")
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::ToolCallStart { name, .. } if name.0 == "read"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::ToolArgsDelta { delta, .. } if delta.contains("README.md")
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::UsageUpdate { usage } if usage.input_tokens == 11 && usage.output_tokens == 7
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::MessageEnd { stop_reason, .. } if *stop_reason == StopReason::ToolUse
        )));
    }

    #[test]
    fn compaction_request_and_response_use_anthropic_inline_summary_shape() {
        let request = sample_compaction_request();
        let body = encode_compaction_request(&request, None).expect("encode compaction");

        assert_eq!(body["model"], "claude-sonnet-4-5");
        assert!(body.get("temperature").is_none());
        assert!(
            body["messages"][0]["content"][0]["text"]
                .as_str()
                .is_some_and(|text| text.contains("assistant:"))
        );

        let response = json!({
            "id": "msg_summary",
            "content": [
                {"type": "text", "text": "Summary of the compacted turn"}
            ],
            "usage": {
                "input_tokens": 30,
                "output_tokens": 9,
            }
        });
        let decoded = decode_compaction_response(&response).expect("decode compaction");

        assert_eq!(decoded.usage.input_tokens, 30);
        assert_eq!(decoded.output[0]["type"], COMPACTED_CONTEXT_TYPE);
        assert_eq!(decoded.output[0]["text"], "Summary of the compacted turn");
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

    fn sample_compaction_request() -> ProviderCompactionRequest {
        let request = sample_request(vec![
            Message::User(UserMessage {
                id: MessageId::new(),
                created_at: Utc::now(),
                parts: vec![UserPart::Text {
                    text: "inspect README".to_owned(),
                }],
            }),
            Message::Assistant(AssistantMessage {
                id: MessageId::new(),
                created_at: Utc::now(),
                parts: vec![AssistantPart::Text {
                    text: "I inspected it.".to_owned(),
                }],
                stop_reason: None,
                usage: None,
                replay_meta: Default::default(),
            }),
            Message::Tool(ToolResultMessage {
                id: MessageId::new(),
                call_id: ToolCallId::from("toolu_123"),
                content: ToolResult::Text {
                    text: "README contents".to_owned(),
                },
                error: None,
                created_at: Utc::now(),
            }),
        ]);

        ProviderCompactionRequest {
            session_id: Default::default(),
            model: request.model,
            compacted_prefix: vec![json!({
                "type": COMPACTED_CONTEXT_TYPE,
                "text": "Earlier summary",
            })],
            messages: request.messages,
            tools: request.tools,
            instructions: "Summarize the session".to_owned(),
        }
    }
}
