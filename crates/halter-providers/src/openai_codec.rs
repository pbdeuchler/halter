// pattern: Functional Core

use std::collections::{BTreeMap, BTreeSet};

use anyhow::Context;
use async_openai::types::responses::{
    FunctionToolCall, OutputItem, OutputMessage, OutputMessageContent, ReasoningItem, Response,
    ResponseStreamEvent, SummaryPart,
};
use halter_protocol::{
    ApiKind, AssistantPart, BlockId, Message, MessageId, ProviderCompactionRequest,
    ProviderCompactionResponse, ProviderRequest, ReasoningEffort, StopReason, StreamEvent, Usage,
    UserPart,
};
use serde_json::{Map, Value, json};
use tracing::warn;

use crate::codec_common::{
    assistant_text, bounded_provider_id, bounded_provider_id_with_prefix, canonical_tool_name,
    collect_system_text, data_url, document_filename, has_user_media, tool_name_for_provider,
    tool_result_text, user_text,
};

// Alias the consolidated PROVIDER_ID_MAX_LEN from codec_common (finding L14).
// Kept as a named alias because the local sites read more clearly with the
// responses-item-specific identifier at the callsite.
const RESPONSES_ITEM_ID_MAX_LEN: usize = crate::codec_common::PROVIDER_ID_MAX_LEN;
/// Tag pair that wraps an in-band (non-dedicated) compaction summary so
/// the model can clearly distinguish lossy summarized history from
/// authoritative system content. The runtime relies on these literal
/// markers when computing eligibility windows, so changing them requires
/// a coordinated update.
pub(crate) const COMPACTION_OPEN_TAG: &str = "<compaction>\n";
pub(crate) const COMPACTION_CLOSE_TAG: &str = "\n</compaction>";

#[derive(Debug, Clone, Copy)]
pub(crate) struct ResponsesRequestOptions<'a> {
    pub stream: bool,
    pub store: Option<bool>,
    pub prompt_cache_key: Option<&'a str>,
    pub include_encrypted_reasoning: bool,
    pub reasoning_summary: Option<&'a str>,
    /// Sampling temperature forwarded to the Responses API.
    pub temperature: f32,
}

pub(crate) fn encode_responses_request(
    request: &ProviderRequest,
    options: ResponsesRequestOptions<'_>,
) -> anyhow::Result<Value> {
    if request.model.api_kind != ApiKind::OpenAiResponses {
        anyhow::bail!("failed to encode openai responses request: unsupported api kind");
    }

    let mut body = Map::new();
    body.insert(
        "model".to_owned(),
        Value::String(request.model.model.clone()),
    );

    // When chaining via previous_response_id, only send new messages.
    // The server retains the prior conversation state.
    if let Some(ref prev_id) = request.previous_response_id {
        body.insert(
            "previous_response_id".to_owned(),
            Value::String(prev_id.clone()),
        );
        let new_messages = &request.messages[request.new_messages_start..];
        let input = encode_responses_input_slice(new_messages, request)?;
        validate_responses_input_item_ids(&input)?;
        body.insert("input".to_owned(), Value::Array(input));
    } else {
        let input = encode_responses_input(request)?;
        validate_responses_input_item_ids(&input)?;
        body.insert("input".to_owned(), Value::Array(input));
    }

    body.insert("stream".to_owned(), Value::Bool(options.stream));
    body.insert("temperature".to_owned(), json!(options.temperature));

    if let Some(max_output_tokens) = request.model.max_output_tokens {
        body.insert("max_output_tokens".to_owned(), json!(max_output_tokens));
    }
    if let Some(store) = options.store {
        body.insert("store".to_owned(), Value::Bool(store));
    }
    if let Some(prompt_cache_key) = options.prompt_cache_key {
        body.insert(
            "prompt_cache_key".to_owned(),
            Value::String(prompt_cache_key.to_owned()),
        );
    }
    if let Some(reasoning) =
        encode_openai_reasoning(request.model.reasoning, options.reasoning_summary)
    {
        body.insert("reasoning".to_owned(), reasoning);
        if options.include_encrypted_reasoning {
            body.insert("include".to_owned(), json!(["reasoning.encrypted_content"]));
        }
    }
    if !request.tools.is_empty() {
        body.insert(
            "tools".to_owned(),
            Value::Array(encode_responses_tools(request)),
        );
    }

    Ok(Value::Object(body))
}

pub(crate) fn encode_responses_compact_request(
    request: &ProviderCompactionRequest,
) -> anyhow::Result<Value> {
    if request.model.api_kind != ApiKind::OpenAiResponses {
        anyhow::bail!("failed to encode openai compaction request: unsupported api kind");
    }

    let input = encode_responses_compact_input(request)?;
    validate_responses_input_item_ids(&input)?;
    Ok(json!({
        "model": request.model.model,
        "input": input,
        "instructions": request.instructions,
    }))
}

pub(crate) fn encode_openrouter_compact_request(
    request: &ProviderCompactionRequest,
) -> anyhow::Result<Value> {
    if request.model.api_kind != ApiKind::OpenAiResponses {
        anyhow::bail!("failed to encode openrouter compaction request: unsupported api kind");
    }

    let input = encode_responses_compact_input(request)?;
    validate_responses_input_item_ids(&input)?;
    Ok(json!({
        "model": request.model.model.clone(),
        "input": input,
        "instructions": request.instructions.clone(),
        "stream": false,
        "store": false,
    }))
}

pub(crate) fn decode_responses_compact_response(
    response: &Value,
) -> anyhow::Result<ProviderCompactionResponse> {
    let output = response
        .get("output")
        .and_then(Value::as_array)
        .cloned()
        .ok_or_else(|| {
            anyhow::anyhow!("failed to decode openai compaction response: missing output array")
        })?;

    Ok(ProviderCompactionResponse {
        output,
        usage: decode_openai_usage(response),
    })
}

pub(crate) fn decode_openrouter_compact_response(
    response: &Value,
) -> anyhow::Result<ProviderCompactionResponse> {
    let output = response
        .get("output")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            anyhow::anyhow!("failed to decode openrouter compaction response: missing output array")
        })?;
    let compacted_text = openrouter_compaction_output_text(output);
    let output = if compacted_text.is_empty() {
        Vec::new()
    } else {
        vec![encode_responses_developer_message(&format!(
            "{COMPACTION_OPEN_TAG}{compacted_text}{COMPACTION_CLOSE_TAG}"
        ))]
    };

    Ok(ProviderCompactionResponse {
        output,
        usage: decode_openai_usage(response),
    })
}

fn encode_responses_compact_input(
    request: &ProviderCompactionRequest,
) -> anyhow::Result<Vec<Value>> {
    let mut input = request.compacted_prefix.clone();
    for message in &request.messages {
        match message {
            Message::System(_) => {}
            Message::User(user) => input.push(encode_responses_user_message(user)?),
            Message::Assistant(assistant) => {
                if let Some(message) = encode_responses_assistant_message(assistant) {
                    input.push(message);
                }
                for part in &assistant.parts {
                    if let AssistantPart::ToolCall(call) = part {
                        input.push(json!({
                            "type": "function_call",
                            "id": responses_function_call_item_id(&call.id),
                            "call_id": call.id,
                            "name": tool_name_for_provider(
                                &call.name,
                                &request.tools,
                                request.model.provider_kind,
                            ),
                            "arguments": call.arguments.to_string(),
                            "status": "completed",
                        }));
                    }
                }
            }
            Message::Tool(tool) => input.push(encode_responses_tool_output(tool)),
        }
    }

    Ok(input)
}

#[cfg(test)]
pub(crate) fn decode_responses_response(
    request: &ProviderRequest,
    response: &Value,
) -> anyhow::Result<Vec<StreamEvent>> {
    let message_id = response_output_message_id(response).unwrap_or_default();
    let mut events = vec![StreamEvent::MessageStart {
        id: message_id.clone(),
    }];
    let mut saw_tool_call = false;

    for item in response
        .get("output")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        match item.get("type").and_then(Value::as_str).unwrap_or_default() {
            "message" => {
                if let Some(content) = item.get("content").and_then(Value::as_array) {
                    for block in content {
                        if let Some(text) = block
                            .get("text")
                            .and_then(Value::as_str)
                            .or_else(|| block.get("output_text").and_then(Value::as_str))
                        {
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
                }
            }
            "reasoning" => {
                let text = reasoning_text(item);
                if !text.is_empty() {
                    let block_id = BlockId::new();
                    events.push(StreamEvent::ThinkingStart {
                        id: block_id.clone(),
                    });
                    events.push(StreamEvent::ThinkingDelta {
                        id: block_id.clone(),
                        delta: text,
                    });
                    events.push(StreamEvent::ThinkingEnd {
                        id: block_id,
                        signature: item
                            .get("encrypted_content")
                            .and_then(Value::as_str)
                            .map(str::to_owned),
                    });
                }
            }
            "function_call" => {
                let block_id = BlockId::new();
                saw_tool_call = true;
                let tool_call_id = item
                    .get("call_id")
                    .and_then(Value::as_str)
                    .or_else(|| item.get("id").and_then(Value::as_str))
                    .map(|value| value.into())
                    .unwrap_or_default();
                let tool_name = item
                    .get("name")
                    .and_then(Value::as_str)
                    .map(|name| {
                        canonical_tool_name(name, &request.tools, request.model.provider_kind)
                    })
                    .unwrap_or_else(|| "tool".into());
                let arguments = item
                    .get("arguments")
                    .map(openai_arguments_string)
                    .unwrap_or_else(|| "{}".to_owned());
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

    events.push(StreamEvent::UsageUpdate {
        usage: decode_openai_usage(response),
    });
    events.push(StreamEvent::MessageEnd {
        id: message_id,
        stop_reason: decode_responses_stop_reason(response, saw_tool_call),
        response_id: response_output_response_id(response),
    });
    Ok(events)
}

#[derive(Debug, Clone)]
pub(crate) struct ResponsesStreamDecoder {
    tool_specs: Vec<halter_protocol::ToolSpec>,
    provider_kind: halter_protocol::ProviderKind,
    track_response_id: bool,
    response_id: Option<String>,
    message_id: Option<MessageId>,
    started: bool,
    saw_tool_call: bool,
    active_text: Option<PendingTextBlock>,
    active_reasoning: Option<PendingReasoningBlock>,
    active_tool_calls: BTreeMap<String, PendingToolCallBlock>,
    completed_tool_calls: BTreeSet<String>,
}

#[derive(Debug, Clone)]
struct PendingTextBlock {
    id: BlockId,
    text: String,
}

#[derive(Debug, Clone)]
struct PendingReasoningBlock {
    id: BlockId,
    text: String,
    mode: Option<ReasoningStreamMode>,
    summary_parts: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReasoningStreamMode {
    Summary,
    Text,
}

#[derive(Debug, Clone)]
struct PendingToolCallBlock {
    id: BlockId,
    arguments: String,
}

impl ResponsesStreamDecoder {
    #[must_use]
    pub(crate) fn new(request: &ProviderRequest, track_response_id: bool) -> Self {
        Self {
            tool_specs: request.tools.clone(),
            provider_kind: request.model.provider_kind,
            track_response_id,
            response_id: None,
            message_id: None,
            started: false,
            saw_tool_call: false,
            active_text: None,
            active_reasoning: None,
            active_tool_calls: BTreeMap::new(),
            completed_tool_calls: BTreeSet::new(),
        }
    }

    pub(crate) fn decode(
        &mut self,
        event: ResponseStreamEvent,
    ) -> anyhow::Result<Vec<StreamEvent>> {
        let mut events = Vec::new();
        match event {
            ResponseStreamEvent::ResponseCreated(event) => {
                self.remember_response_id(&event.response.id);
            }
            ResponseStreamEvent::ResponseOutputItemAdded(event) => match event.item {
                OutputItem::Message(message) => {
                    self.ensure_message_started(Some(&message.id), &mut events);
                    self.open_text_block(&mut events);
                    let initial_text = output_message_text(&message);
                    self.push_text_delta(&initial_text, &mut events);
                }
                OutputItem::Reasoning(reasoning) => {
                    self.ensure_message_started(None, &mut events);
                    self.open_reasoning_block(&mut events);
                    let initial_text = reasoning_item_text(&reasoning);
                    if !initial_text.is_empty() {
                        let mode = if reasoning.summary.is_empty() {
                            ReasoningStreamMode::Text
                        } else {
                            ReasoningStreamMode::Summary
                        };
                        self.push_reasoning_delta(&initial_text, mode, &mut events);
                    }
                }
                OutputItem::FunctionCall(call) => {
                    self.ensure_message_started(None, &mut events);
                    self.start_tool_call(
                        tool_item_key(call.id.as_deref(), event.output_index),
                        &call,
                        &mut events,
                    );
                }
                _ => {}
            },
            ResponseStreamEvent::ResponseOutputTextDelta(event) => {
                self.push_text_delta(&event.delta, &mut events);
            }
            ResponseStreamEvent::ResponseOutputTextDone(event) => {
                self.finish_text_block(Some(&event.text), &mut events);
            }
            ResponseStreamEvent::ResponseRefusalDelta(event) => {
                self.push_text_delta(&event.delta, &mut events);
            }
            ResponseStreamEvent::ResponseRefusalDone(event) => {
                self.finish_text_block(Some(&event.refusal), &mut events);
            }
            ResponseStreamEvent::ResponseReasoningSummaryPartAdded(_) => {
                self.start_reasoning_summary_part(&mut events);
            }
            ResponseStreamEvent::ResponseReasoningSummaryTextDelta(event) => {
                self.push_reasoning_delta(&event.delta, ReasoningStreamMode::Summary, &mut events);
            }
            ResponseStreamEvent::ResponseReasoningSummaryTextDone(event) => {
                self.finish_reasoning_text_fragment(
                    &event.text,
                    ReasoningStreamMode::Summary,
                    &mut events,
                );
            }
            ResponseStreamEvent::ResponseReasoningTextDelta(event) => {
                self.push_reasoning_delta(&event.delta, ReasoningStreamMode::Text, &mut events);
            }
            ResponseStreamEvent::ResponseReasoningTextDone(event) => {
                self.finish_reasoning_text_fragment(
                    &event.text,
                    ReasoningStreamMode::Text,
                    &mut events,
                );
            }
            ResponseStreamEvent::ResponseFunctionCallArgumentsDelta(event) => {
                self.push_tool_args_delta(&event.item_id, &event.delta, &mut events)?;
            }
            ResponseStreamEvent::ResponseFunctionCallArgumentsDone(event) => {
                self.finish_tool_call(
                    &event.item_id,
                    Some(&event.arguments),
                    None,
                    None,
                    &mut events,
                )?;
            }
            ResponseStreamEvent::ResponseOutputItemDone(event) => match event.item {
                OutputItem::Message(message) => {
                    self.finish_text_block(Some(&output_message_text(&message)), &mut events);
                }
                OutputItem::Reasoning(reasoning) => {
                    self.finish_reasoning_block(
                        Some(&reasoning_item_text(&reasoning)),
                        reasoning.encrypted_content,
                        &mut events,
                    );
                }
                OutputItem::FunctionCall(call) => {
                    self.finish_tool_call(
                        &tool_item_key(call.id.as_deref(), event.output_index),
                        Some(&call.arguments),
                        Some(&call.call_id),
                        Some(&call.name),
                        &mut events,
                    )?;
                }
                _ => {}
            },
            ResponseStreamEvent::ResponseCompleted(event) => {
                self.finish_response(&event.response, &mut events);
            }
            ResponseStreamEvent::ResponseIncomplete(event) => {
                self.finish_response(&event.response, &mut events);
            }
            ResponseStreamEvent::ResponseFailed(event) => {
                anyhow::bail!(response_failure_message(&event.response));
            }
            ResponseStreamEvent::ResponseError(event) => {
                anyhow::bail!("failed to execute provider request: {}", event.message);
            }
            ResponseStreamEvent::ResponseInProgress(_)
            | ResponseStreamEvent::ResponseContentPartAdded(_)
            | ResponseStreamEvent::ResponseContentPartDone(_)
            | ResponseStreamEvent::ResponseQueued(_)
            | ResponseStreamEvent::ResponseReasoningSummaryPartDone(_)
            | ResponseStreamEvent::ResponseFileSearchCallInProgress(_)
            | ResponseStreamEvent::ResponseFileSearchCallSearching(_)
            | ResponseStreamEvent::ResponseFileSearchCallCompleted(_)
            | ResponseStreamEvent::ResponseWebSearchCallInProgress(_)
            | ResponseStreamEvent::ResponseWebSearchCallSearching(_)
            | ResponseStreamEvent::ResponseWebSearchCallCompleted(_)
            | ResponseStreamEvent::ResponseImageGenerationCallCompleted(_)
            | ResponseStreamEvent::ResponseImageGenerationCallGenerating(_)
            | ResponseStreamEvent::ResponseImageGenerationCallInProgress(_)
            | ResponseStreamEvent::ResponseImageGenerationCallPartialImage(_)
            | ResponseStreamEvent::ResponseMCPCallArgumentsDelta(_)
            | ResponseStreamEvent::ResponseMCPCallArgumentsDone(_)
            | ResponseStreamEvent::ResponseMCPCallCompleted(_)
            | ResponseStreamEvent::ResponseMCPCallFailed(_)
            | ResponseStreamEvent::ResponseMCPCallInProgress(_)
            | ResponseStreamEvent::ResponseMCPListToolsCompleted(_)
            | ResponseStreamEvent::ResponseMCPListToolsFailed(_)
            | ResponseStreamEvent::ResponseMCPListToolsInProgress(_)
            | ResponseStreamEvent::ResponseCodeInterpreterCallInProgress(_)
            | ResponseStreamEvent::ResponseCodeInterpreterCallInterpreting(_)
            | ResponseStreamEvent::ResponseCodeInterpreterCallCompleted(_)
            | ResponseStreamEvent::ResponseCodeInterpreterCallCodeDelta(_)
            | ResponseStreamEvent::ResponseCodeInterpreterCallCodeDone(_)
            | ResponseStreamEvent::ResponseOutputTextAnnotationAdded(_)
            | ResponseStreamEvent::ResponseCustomToolCallInputDelta(_)
            | ResponseStreamEvent::ResponseCustomToolCallInputDone(_) => {}
        }

        Ok(events)
    }

    fn ensure_message_started(&mut self, message_id: Option<&str>, events: &mut Vec<StreamEvent>) {
        if self.started {
            return;
        }

        // Use `MessageId::new` explicitly rather than `unwrap_or_default`: the
        // intent here is "mint a fresh unique id", not "fall back to whatever
        // Default happens to produce". Default currently delegates to `new`,
        // but binding us to that would make changing Default (e.g. to `""`) a
        // silent correctness regression. (finding M23)
        #[allow(clippy::unwrap_or_default)]
        let message_id = self.message_id.clone().unwrap_or_else(|| {
            message_id
                .filter(|id| is_responses_message_item_id(id))
                .map(|id| MessageId::from(id.to_owned()))
                .or_else(|| {
                    self.response_id
                        .as_deref()
                        .map(synthesized_responses_message_id)
                })
                .unwrap_or_else(MessageId::new)
        });
        self.message_id = Some(message_id.clone());
        self.started = true;
        events.push(StreamEvent::MessageStart { id: message_id });
    }

    fn open_text_block(&mut self, events: &mut Vec<StreamEvent>) {
        self.finish_text_block(None, events);
        let block = PendingTextBlock {
            id: BlockId::new(),
            text: String::new(),
        };
        events.push(StreamEvent::TextStart {
            id: block.id.clone(),
        });
        self.active_text = Some(block);
    }

    fn push_text_delta(&mut self, delta: &str, events: &mut Vec<StreamEvent>) {
        if delta.is_empty() {
            return;
        }
        if self.active_text.is_none() {
            self.open_text_block(events);
        }
        let block = self.active_text.as_mut().expect("text block initialized");
        block.text.push_str(delta);
        events.push(StreamEvent::TextDelta {
            id: block.id.clone(),
            delta: delta.to_owned(),
        });
    }

    fn finish_text_block(&mut self, full_text: Option<&str>, events: &mut Vec<StreamEvent>) {
        let pending_delta = match (self.active_text.as_ref(), full_text) {
            (Some(block), Some(full_text)) => full_text.strip_prefix(&block.text),
            _ => None,
        };
        if let Some(delta) = pending_delta {
            self.push_text_delta(delta, events);
        }
        if let Some(block) = self.active_text.take() {
            events.push(StreamEvent::TextEnd { id: block.id });
        }
    }

    fn open_reasoning_block(&mut self, events: &mut Vec<StreamEvent>) {
        self.finish_reasoning_block(None, None, events);
        let block = PendingReasoningBlock {
            id: BlockId::new(),
            text: String::new(),
            mode: None,
            summary_parts: 0,
        };
        events.push(StreamEvent::ThinkingStart {
            id: block.id.clone(),
        });
        self.active_reasoning = Some(block);
    }

    fn start_reasoning_summary_part(&mut self, events: &mut Vec<StreamEvent>) {
        if self.active_reasoning.is_none() {
            self.open_reasoning_block(events);
        }
        let needs_separator = {
            let block = self
                .active_reasoning
                .as_ref()
                .expect("reasoning block initialized");
            if matches!(block.mode, Some(ReasoningStreamMode::Text)) {
                return;
            }
            block.summary_parts > 0
        };
        if needs_separator {
            self.push_reasoning_delta("\n\n", ReasoningStreamMode::Summary, events);
        }
        let block = self
            .active_reasoning
            .as_mut()
            .expect("reasoning block initialized");
        block.summary_parts += 1;
        block.mode = Some(ReasoningStreamMode::Summary);
    }

    fn push_reasoning_delta(
        &mut self,
        delta: &str,
        mode: ReasoningStreamMode,
        events: &mut Vec<StreamEvent>,
    ) {
        if delta.is_empty() {
            return;
        }
        if self.active_reasoning.is_none() {
            self.open_reasoning_block(events);
        }
        let block = self
            .active_reasoning
            .as_mut()
            .expect("reasoning block initialized");
        if let Some(existing_mode) = block.mode {
            if existing_mode != mode {
                warn!(
                    reasoning_id = %block.id,
                    existing_mode = ?existing_mode,
                    dropped_mode = ?mode,
                    dropped_chars = delta.chars().count(),
                    "dropping reasoning delta: mode mismatch within a single reasoning block"
                );
                return;
            }
        } else {
            block.mode = Some(mode);
        }
        block.text.push_str(delta);
        events.push(StreamEvent::ThinkingDelta {
            id: block.id.clone(),
            delta: delta.to_owned(),
        });
    }

    fn finish_reasoning_text_fragment(
        &mut self,
        full_text: &str,
        mode: ReasoningStreamMode,
        events: &mut Vec<StreamEvent>,
    ) {
        let Some(block) = self.active_reasoning.as_ref() else {
            return;
        };
        if matches!(block.mode, Some(existing_mode) if existing_mode != mode) {
            return;
        }
        if let Some(delta) = full_text.strip_prefix(&block.text) {
            self.push_reasoning_delta(delta, mode, events);
        }
    }

    fn finish_reasoning_block(
        &mut self,
        full_text: Option<&str>,
        signature: Option<String>,
        events: &mut Vec<StreamEvent>,
    ) {
        let pending = match (self.active_reasoning.as_ref(), full_text) {
            (Some(block), Some(full_text)) => full_text
                .strip_prefix(&block.text)
                .map(|delta| (delta, block.mode.unwrap_or(ReasoningStreamMode::Summary))),
            _ => None,
        };
        if let Some((delta, mode)) = pending {
            self.push_reasoning_delta(delta, mode, events);
        }
        if let Some(block) = self.active_reasoning.take() {
            events.push(StreamEvent::ThinkingEnd {
                id: block.id,
                signature,
            });
        }
    }

    fn start_tool_call(
        &mut self,
        item_key: String,
        call: &FunctionToolCall,
        events: &mut Vec<StreamEvent>,
    ) {
        if self.active_tool_calls.contains_key(&item_key)
            || self.completed_tool_calls.contains(&item_key)
        {
            return;
        }
        let tool_call_id: halter_protocol::ToolCallId = call.call_id.as_str().into();
        let name = canonical_tool_name(&call.name, &self.tool_specs, self.provider_kind);
        let block = PendingToolCallBlock {
            id: BlockId::new(),
            arguments: String::new(),
        };
        events.push(StreamEvent::ToolCallStart {
            id: block.id.clone(),
            tool_call_id,
            name,
        });
        self.active_tool_calls.insert(item_key.clone(), block);
        self.saw_tool_call = true;
        self.push_tool_args_delta(&item_key, &call.arguments, events)
            .expect("tool call inserted");
    }

    fn push_tool_args_delta(
        &mut self,
        item_key: &str,
        delta: &str,
        events: &mut Vec<StreamEvent>,
    ) -> anyhow::Result<()> {
        if delta.is_empty() {
            return Ok(());
        }
        let pending = self.active_tool_calls.get_mut(item_key).with_context(|| {
            format!(
                "failed to decode openai responses stream: missing tool call '{}'",
                item_key
            )
        })?;
        pending.arguments.push_str(delta);
        events.push(StreamEvent::ToolArgsDelta {
            id: pending.id.clone(),
            delta: delta.to_owned(),
        });
        Ok(())
    }

    fn finish_tool_call(
        &mut self,
        item_key: &str,
        full_arguments: Option<&str>,
        call_id: Option<&str>,
        name: Option<&str>,
        events: &mut Vec<StreamEvent>,
    ) -> anyhow::Result<()> {
        let pending_delta = match (self.active_tool_calls.get(item_key), full_arguments) {
            (Some(pending), Some(full_arguments)) => {
                full_arguments.strip_prefix(&pending.arguments)
            }
            _ => None,
        };
        if let Some(delta) = pending_delta {
            self.push_tool_args_delta(item_key, delta, events)?;
        }
        if self.active_tool_calls.contains_key(item_key) {
            let pending = self
                .active_tool_calls
                .remove(item_key)
                .expect("tool call present after lookup");
            self.completed_tool_calls.insert(item_key.to_owned());
            events.push(StreamEvent::ToolCallEnd { id: pending.id });
            return Ok(());
        }

        if self.completed_tool_calls.contains(item_key) {
            return Ok(());
        }

        let Some(call_id) = call_id else {
            return Ok(());
        };
        let Some(name) = name else {
            return Ok(());
        };
        let tool_call_id = call_id.into();
        let tool_name = canonical_tool_name(name, &self.tool_specs, self.provider_kind);
        let block_id = BlockId::new();
        events.push(StreamEvent::ToolCallStart {
            id: block_id.clone(),
            tool_call_id,
            name: tool_name,
        });
        if let Some(full_arguments) = full_arguments
            && !full_arguments.is_empty()
        {
            events.push(StreamEvent::ToolArgsDelta {
                id: block_id.clone(),
                delta: full_arguments.to_owned(),
            });
        }
        self.completed_tool_calls.insert(item_key.to_owned());
        events.push(StreamEvent::ToolCallEnd { id: block_id });
        Ok(())
    }

    fn finish_response(&mut self, response: &Response, events: &mut Vec<StreamEvent>) {
        self.remember_response_id(&response.id);
        self.ensure_message_started(None, events);
        if let Some(usage) = &response.usage {
            events.push(StreamEvent::UsageUpdate {
                usage: decode_responses_usage(usage),
            });
        }
        events.push(StreamEvent::MessageEnd {
            id: self
                .message_id
                .clone()
                .expect("message id initialized before message end"),
            stop_reason: decode_responses_stop_reason_from_response(response, self.saw_tool_call),
            response_id: self.response_id.clone(),
        });
    }

    fn remember_response_id(&mut self, response_id: &str) {
        if self.track_response_id && !response_id.is_empty() {
            self.response_id = Some(response_id.to_owned());
        }
    }
}

#[cfg(test)]
pub(crate) fn encode_chat_request(
    request: &ProviderRequest,
    include_reasoning: bool,
) -> anyhow::Result<Value> {
    if request.model.api_kind != ApiKind::OpenAiChat {
        anyhow::bail!("failed to encode openai chat request: unsupported api kind");
    }

    let mut body = Map::new();
    body.insert(
        "model".to_owned(),
        Value::String(request.model.model.clone()),
    );
    body.insert("stream".to_owned(), Value::Bool(false));
    body.insert(
        "messages".to_owned(),
        Value::Array(encode_chat_messages(request)?),
    );

    if let Some(max_output_tokens) = request.model.max_output_tokens {
        body.insert("max_tokens".to_owned(), json!(max_output_tokens));
    }
    if include_reasoning
        && let Some(reasoning) = encode_openai_reasoning(request.model.reasoning, None)
    {
        body.insert("reasoning".to_owned(), reasoning);
    }
    if !request.tools.is_empty() {
        body.insert("tools".to_owned(), Value::Array(encode_chat_tools(request)));
        body.insert("tool_choice".to_owned(), json!("auto"));
    }

    Ok(Value::Object(body))
}

#[cfg(test)]
pub(crate) fn decode_chat_response(
    request: &ProviderRequest,
    response: &Value,
) -> anyhow::Result<Vec<StreamEvent>> {
    let message_id = response
        .get("id")
        .and_then(Value::as_str)
        .map(|id| MessageId::from(id.to_owned()))
        .unwrap_or_default();
    let choice = response
        .pointer("/choices/0")
        .with_context(|| "failed to decode openai chat response: missing first choice")?;
    let message = choice
        .get("message")
        .with_context(|| "failed to decode openai chat response: missing message")?;
    let mut events = vec![StreamEvent::MessageStart {
        id: message_id.clone(),
    }];

    if let Some(text) = chat_content_text(message.get("content")) {
        let block_id = BlockId::new();
        events.push(StreamEvent::TextStart {
            id: block_id.clone(),
        });
        events.push(StreamEvent::TextDelta {
            id: block_id.clone(),
            delta: text,
        });
        events.push(StreamEvent::TextEnd { id: block_id });
    }

    for tool_call in message
        .get("tool_calls")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let block_id = BlockId::new();
        let tool_call_id = tool_call
            .get("id")
            .and_then(Value::as_str)
            .map(|value| value.into())
            .unwrap_or_default();
        let function = tool_call.get("function").unwrap_or(tool_call);
        let tool_name = function
            .get("name")
            .and_then(Value::as_str)
            .map(|name| canonical_tool_name(name, &request.tools, request.model.provider_kind))
            .unwrap_or_else(|| "tool".into());
        let arguments = function
            .get("arguments")
            .map(openai_arguments_string)
            .unwrap_or_else(|| "{}".to_owned());
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

    events.push(StreamEvent::UsageUpdate {
        usage: decode_chat_usage(response),
    });
    events.push(StreamEvent::MessageEnd {
        id: message_id,
        stop_reason: decode_chat_stop_reason(choice),
        response_id: None,
    });
    Ok(events)
}

fn encode_responses_input(request: &ProviderRequest) -> anyhow::Result<Vec<Value>> {
    let mut input = Vec::new();
    if let Some(system) = collect_system_text(request) {
        input.push(encode_responses_developer_message(&system));
    }
    input.extend(request.compacted_prefix.clone());

    for message in &request.messages {
        match message {
            Message::System(_) => {}
            Message::User(user) => input.push(encode_responses_user_message(user)?),
            Message::Assistant(assistant) => {
                if let Some(message) = encode_responses_assistant_message(assistant) {
                    input.push(message);
                }
                for part in &assistant.parts {
                    if let AssistantPart::ToolCall(call) = part {
                        input.push(encode_responses_tool_call(call, request));
                    }
                }
            }
            Message::Tool(tool) => input.push(encode_responses_tool_output(tool)),
        }
    }

    Ok(input)
}

/// Encode only the given message slice — used when chaining via `previous_response_id`.
/// Omits the developer/system message since the server already has it.
fn encode_responses_input_slice(
    messages: &[Message],
    request: &ProviderRequest,
) -> anyhow::Result<Vec<Value>> {
    let mut input = Vec::new();
    for message in messages {
        match message {
            Message::System(_) => {}
            Message::User(user) => input.push(encode_responses_user_message(user)?),
            Message::Assistant(assistant) => {
                if let Some(message) = encode_responses_assistant_message(assistant) {
                    input.push(message);
                }
                for part in &assistant.parts {
                    if let AssistantPart::ToolCall(call) = part {
                        input.push(encode_responses_tool_call(call, request));
                    }
                }
            }
            Message::Tool(tool) => input.push(encode_responses_tool_output(tool)),
        }
    }
    Ok(input)
}

fn encode_responses_developer_message(text: &str) -> Value {
    json!({
        "type": "message",
        "role": "developer",
        "content": [
            {
                "type": "input_text",
                "text": text,
            }
        ],
    })
}

fn encode_responses_user_message(message: &halter_protocol::UserMessage) -> anyhow::Result<Value> {
    let mut content = Vec::new();
    if has_user_media(message) {
        for part in &message.parts {
            match part {
                UserPart::Text { text } => content.push(json!({
                    "type": "input_text",
                    "text": text,
                })),
                UserPart::Image { media_type, data } => content.push(json!({
                    "type": "input_image",
                    "image_url": data_url(media_type, data),
                })),
                UserPart::Document { media_type, data } => content.push(json!({
                    "type": "input_file",
                    "filename": document_filename(media_type),
                    "file_data": data_url(media_type, data),
                })),
            }
        }
    } else {
        content.push(json!({
            "type": "input_text",
            "text": user_text(message),
        }));
    }

    Ok(json!({
        "type": "message",
        "role": "user",
        "content": content,
    }))
}

fn encode_responses_assistant_message(
    message: &halter_protocol::AssistantMessage,
) -> Option<Value> {
    let text = assistant_text(message);
    if text.is_empty() {
        return None;
    }

    let mut encoded = Map::new();
    encoded.insert("type".to_owned(), json!("message"));
    encoded.insert("role".to_owned(), json!("assistant"));
    encoded.insert("status".to_owned(), json!("completed"));
    encoded.insert(
        "content".to_owned(),
        json!([
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
            }
        ]),
    );
    if let Some(message_id) = encode_responses_message_item_id(&message.id) {
        encoded.insert("id".to_owned(), Value::String(message_id));
    }

    Some(Value::Object(encoded))
}

fn encode_responses_tool_call(
    call: &halter_protocol::ToolCall,
    request: &ProviderRequest,
) -> Value {
    json!({
        "type": "function_call",
        "id": responses_function_call_item_id(&call.id),
        "call_id": call.id,
        "name": tool_name_for_provider(
            &call.name,
            &request.tools,
            request.model.provider_kind,
        ),
        "arguments": call.arguments.to_string(),
        "status": "completed",
    })
}

fn encode_responses_tool_output(message: &halter_protocol::ToolResultMessage) -> Value {
    json!({
        "type": "function_call_output",
        "id": responses_function_call_output_item_id(&message.call_id),
        "call_id": message.call_id,
        "output": tool_result_text(&message.content, &message.error),
    })
}

fn responses_function_call_item_id(call_id: &halter_protocol::ToolCallId) -> String {
    bounded_provider_id_with_prefix("fc_", &call_id.0, RESPONSES_ITEM_ID_MAX_LEN, "fc")
}

fn responses_function_call_output_item_id(call_id: &halter_protocol::ToolCallId) -> String {
    bounded_provider_id_with_prefix(
        "fc_output_",
        &call_id.0,
        RESPONSES_ITEM_ID_MAX_LEN,
        "fc_output",
    )
}

fn synthesized_responses_message_id(response_id: &str) -> MessageId {
    if is_responses_message_item_id(response_id) {
        return MessageId::from(bounded_provider_id(
            response_id,
            RESPONSES_ITEM_ID_MAX_LEN,
            "msg",
        ));
    }
    MessageId::from(bounded_provider_id_with_prefix(
        "msg_response_",
        response_id,
        RESPONSES_ITEM_ID_MAX_LEN,
        "msg_response",
    ))
}

#[cfg(test)]
fn response_output_message_id(response: &Value) -> Option<MessageId> {
    response
        .get("output")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .find_map(|item| {
            if item.get("type").and_then(Value::as_str) != Some("message") {
                return None;
            }
            item.get("id")
                .and_then(Value::as_str)
                .filter(|id| is_responses_message_item_id(id))
                .map(|id| MessageId::from(id.to_owned()))
        })
        .or_else(|| {
            response
                .get("id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
                .map(synthesized_responses_message_id)
        })
}

#[cfg(test)]
fn response_output_response_id(response: &Value) -> Option<String> {
    response
        .get("id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(|id| id.to_owned())
}

fn encode_responses_message_item_id(message_id: &MessageId) -> Option<String> {
    is_responses_message_item_id(&message_id.0)
        .then(|| bounded_provider_id(&message_id.0, RESPONSES_ITEM_ID_MAX_LEN, "msg"))
}

fn is_responses_message_item_id(id: &str) -> bool {
    let Some(rest) = id.strip_prefix("msg_") else {
        return false;
    };
    !rest.is_empty()
        && rest
            .chars()
            .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_' || ch == '-')
}

fn validate_responses_input_item_ids(input: &[Value]) -> anyhow::Result<()> {
    for (index, item) in input.iter().enumerate() {
        let Some(id) = item.get("id").and_then(Value::as_str) else {
            continue;
        };
        if id.len() > RESPONSES_ITEM_ID_MAX_LEN {
            anyhow::bail!(
                "failed to encode openai responses request: input[{index}].id length {} exceeds {}",
                id.len(),
                RESPONSES_ITEM_ID_MAX_LEN
            );
        }
    }

    Ok(())
}

fn encode_responses_tools(request: &ProviderRequest) -> Vec<Value> {
    request
        .tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "name": tool_name_for_provider(&tool.name, &request.tools, request.model.provider_kind),
                "description": tool.description,
                "parameters": tool.input_schema,
            })
        })
        .collect()
}

#[cfg(test)]
fn encode_chat_messages(request: &ProviderRequest) -> anyhow::Result<Vec<Value>> {
    let mut messages = Vec::new();
    if let Some(system) = collect_system_text(request) {
        messages.push(json!({
            "role": "system",
            "content": system,
        }));
    }

    for message in &request.messages {
        match message {
            Message::System(_) => {}
            Message::User(user) => messages.push(encode_chat_user_message(user)?),
            Message::Assistant(assistant) => {
                let text = assistant_text(assistant);
                let tool_calls = assistant
                    .parts
                    .iter()
                    .filter_map(|part| match part {
                        AssistantPart::ToolCall(call) => Some(json!({
                            "id": call.id,
                            "type": "function",
                            "function": {
                                "name": tool_name_for_provider(
                                    &call.name,
                                    &request.tools,
                                    request.model.provider_kind,
                                ),
                                "arguments": call.arguments.to_string(),
                            }
                        })),
                        AssistantPart::Text { .. } | AssistantPart::Thinking(_) => None,
                    })
                    .collect::<Vec<_>>();
                let mut assistant_message = Map::new();
                assistant_message.insert("role".to_owned(), json!("assistant"));
                if !text.is_empty() {
                    assistant_message.insert("content".to_owned(), Value::String(text));
                } else if tool_calls.is_empty() {
                    continue;
                }
                if !tool_calls.is_empty() {
                    assistant_message.insert("tool_calls".to_owned(), Value::Array(tool_calls));
                }
                messages.push(Value::Object(assistant_message));
            }
            Message::Tool(tool) => messages.push(json!({
                "role": "tool",
                "tool_call_id": tool.call_id,
                "content": tool_result_text(&tool.content, &tool.error),
            })),
        }
    }

    Ok(messages)
}

#[cfg(test)]
fn encode_chat_user_message(message: &halter_protocol::UserMessage) -> anyhow::Result<Value> {
    if !has_user_media(message) {
        return Ok(json!({
            "role": "user",
            "content": user_text(message),
        }));
    }

    let mut content = Vec::new();
    for part in &message.parts {
        match part {
            UserPart::Text { text } => content.push(json!({
                "type": "text",
                "text": text,
            })),
            UserPart::Image { media_type, data } => content.push(json!({
                "type": "image_url",
                "image_url": {
                    "url": data_url(media_type, data),
                }
            })),
            UserPart::Document { media_type, .. } => {
                anyhow::bail!(
                    "failed to encode openai chat request: document inputs are not supported by chat completions ({media_type})"
                );
            }
        }
    }

    Ok(json!({
        "role": "user",
        "content": content,
    }))
}

#[cfg(test)]
fn encode_chat_tools(request: &ProviderRequest) -> Vec<Value> {
    request
        .tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool_name_for_provider(&tool.name, &request.tools, request.model.provider_kind),
                    "description": tool.description,
                    "parameters": tool.input_schema,
                }
            })
        })
        .collect()
}

fn encode_openai_reasoning(
    reasoning: Option<ReasoningEffort>,
    summary: Option<&str>,
) -> Option<Value> {
    let effort = match reasoning? {
        ReasoningEffort::Low => "low",
        ReasoningEffort::Medium => "medium",
        ReasoningEffort::High => "high",
        ReasoningEffort::Xhigh => "xhigh",
    };
    let mut body = Map::new();
    body.insert("effort".to_owned(), json!(effort));
    if let Some(summary) = summary {
        body.insert("summary".to_owned(), json!(summary));
    }
    Some(Value::Object(body))
}

#[cfg(test)]
fn decode_responses_stop_reason(response: &Value, saw_tool_call: bool) -> StopReason {
    if saw_tool_call {
        return StopReason::ToolUse;
    }
    match response
        .pointer("/incomplete_details/reason")
        .and_then(Value::as_str)
        .unwrap_or_default()
    {
        "max_output_tokens" | "max_completion_tokens" => StopReason::MaxTokens,
        _ => StopReason::EndTurn,
    }
}

fn decode_responses_stop_reason_from_response(
    response: &Response,
    saw_tool_call: bool,
) -> StopReason {
    if saw_tool_call {
        return StopReason::ToolUse;
    }
    match response
        .incomplete_details
        .as_ref()
        .map(|details| details.reason.as_str())
        .unwrap_or_default()
    {
        "max_output_tokens" | "max_completion_tokens" => StopReason::MaxTokens,
        _ => StopReason::EndTurn,
    }
}

#[cfg(test)]
fn decode_chat_stop_reason(choice: &Value) -> StopReason {
    match choice
        .get("finish_reason")
        .and_then(Value::as_str)
        .unwrap_or("stop")
    {
        "tool_calls" | "function_call" => StopReason::ToolUse,
        "length" => StopReason::MaxTokens,
        "content_filter" => StopReason::Error,
        _ => StopReason::EndTurn,
    }
}

fn decode_openai_usage(response: &Value) -> Usage {
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
            .pointer("/usage/input_tokens_details/cache_creation_tokens")
            .and_then(Value::as_u64)
            .unwrap_or_default(),
        cache_read_input_tokens: response
            .pointer("/usage/input_tokens_details/cached_tokens")
            .and_then(Value::as_u64)
            .unwrap_or_default(),
    }
}

fn decode_responses_usage(usage: &async_openai::types::responses::ResponseUsage) -> Usage {
    Usage {
        input_tokens: u64::from(usage.input_tokens),
        output_tokens: u64::from(usage.output_tokens),
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: u64::from(usage.input_tokens_details.cached_tokens),
    }
}

#[cfg(test)]
fn decode_chat_usage(response: &Value) -> Usage {
    Usage {
        input_tokens: response
            .pointer("/usage/prompt_tokens")
            .and_then(Value::as_u64)
            .unwrap_or_default(),
        output_tokens: response
            .pointer("/usage/completion_tokens")
            .and_then(Value::as_u64)
            .unwrap_or_default(),
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: response
            .pointer("/usage/prompt_tokens_details/cached_tokens")
            .and_then(Value::as_u64)
            .unwrap_or_default(),
    }
}

#[cfg(test)]
fn openai_arguments_string(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        _ => value.to_string(),
    }
}

#[cfg(test)]
fn reasoning_text(item: &Value) -> String {
    if let Some(summary) = item.get("summary").and_then(Value::as_array) {
        let text = summary
            .iter()
            .filter_map(|entry| {
                entry
                    .get("text")
                    .and_then(Value::as_str)
                    .or_else(|| entry.as_str())
            })
            .collect::<Vec<_>>()
            .join("\n");
        if !text.is_empty() {
            return text;
        }
    }

    item.get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|entry| entry.get("text").and_then(Value::as_str))
        .collect::<Vec<_>>()
        .join("\n")
}

fn reasoning_item_text(item: &ReasoningItem) -> String {
    let summary = item
        .summary
        .iter()
        .map(summary_part_text)
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    if !summary.is_empty() {
        return summary;
    }

    item.content
        .as_ref()
        .into_iter()
        .flatten()
        .map(|entry| entry.text.clone())
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn openrouter_compaction_output_text(output: &[Value]) -> String {
    output
        .iter()
        .filter_map(openrouter_compaction_message_text)
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn openrouter_compaction_message_text(item: &Value) -> Option<String> {
    if item.get("type").and_then(Value::as_str) != Some("message") {
        return None;
    }

    let text = item
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(openrouter_compaction_content_text)
        .collect::<Vec<_>>()
        .join("");
    if text.is_empty() { None } else { Some(text) }
}

fn openrouter_compaction_content_text(content: &Value) -> Option<&str> {
    content
        .get("text")
        .and_then(Value::as_str)
        .or_else(|| content.get("output_text").and_then(Value::as_str))
}

fn output_message_text(message: &OutputMessage) -> String {
    message
        .content
        .iter()
        .map(output_message_content_text)
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("")
}

fn output_message_content_text(content: &OutputMessageContent) -> String {
    match content {
        OutputMessageContent::OutputText(text) => text.text.clone(),
        OutputMessageContent::Refusal(refusal) => refusal.refusal.clone(),
    }
}

fn summary_part_text(part: &SummaryPart) -> String {
    match part {
        SummaryPart::SummaryText(text) => text.text.clone(),
    }
}

fn response_failure_message(response: &Response) -> String {
    if let Some(error) = &response.error {
        return format!("failed to execute provider request: {}", error.message);
    }
    if let Some(details) = &response.incomplete_details {
        return format!(
            "failed to execute provider request: incomplete: {}",
            details.reason
        );
    }
    "failed to execute provider request: openai response failed".to_owned()
}

fn tool_item_key(item_id: Option<&str>, output_index: u32) -> String {
    // The synthesized prefix starts with `__halter_synth__` — a sentinel that
    // OpenAI item ids can never match (real ids are `fc_...`/`msg_...`). This
    // guarantees no collision if a later event arrives with a real id whose
    // string happens to equal `output_index:N`.
    item_id
        .map(str::to_owned)
        .unwrap_or_else(|| format!("__halter_synth__:output_index:{output_index}"))
}

#[cfg(test)]
fn chat_content_text(content: Option<&Value>) -> Option<String> {
    match content {
        Some(Value::String(text)) if !text.is_empty() => Some(text.clone()),
        Some(Value::Array(parts)) => {
            let text = parts
                .iter()
                .filter_map(|part| {
                    part.get("text")
                        .and_then(Value::as_str)
                        .or_else(|| part.as_str())
                })
                .collect::<Vec<_>>()
                .join("\n");
            if text.is_empty() { None } else { Some(text) }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use halter_protocol::{
        ApiKind, AssembledPrompt, AssistantMessage, AssistantPart, CacheBreakpoints, CacheScope,
        DEFAULT_TEMPERATURE, Message, MessageId, ModelId, ModelRole, PromptSegment,
        PromptSegmentId, PromptSegmentKind, ProviderKind, ProviderName, ResolvedModel, ToolAlias,
        ToolCall, ToolCallId, ToolCapabilities, ToolConcurrency, ToolResult, ToolResultMessage,
        ToolSpec, TurnId, UserMessage, Volatility,
    };
    use indexmap::IndexMap;
    use serde_json::json;

    use super::*;

    #[test]
    fn is_responses_message_item_id_rejects_cross_provider_shapes() {
        assert!(is_responses_message_item_id("msg_abc123"));
        assert!(is_responses_message_item_id("msg_abcdef0123456789"));
        assert!(is_responses_message_item_id("msg_abc-123_xyz"));

        // Anthropic-style ids contain uppercase letters.
        assert!(!is_responses_message_item_id("msg_01ABC"));
        assert!(!is_responses_message_item_id("msg_01AbCdEf"));

        // Missing prefix or empty body.
        assert!(!is_responses_message_item_id("msg_"));
        assert!(!is_responses_message_item_id("resp_abc"));
        assert!(!is_responses_message_item_id(""));

        // Disallowed punctuation.
        assert!(!is_responses_message_item_id("msg_abc.def"));
    }

    #[test]
    fn openai_responses_request_includes_prompt_cache_key_and_structured_history() {
        let request = sample_request(ApiKind::OpenAiResponses, ProviderKind::OpenAi);
        let body = encode_responses_request(
            &request,
            ResponsesRequestOptions {
                stream: false,
                store: Some(false),
                prompt_cache_key: Some(request.prompt.prefix_cache_key.as_str()),
                include_encrypted_reasoning: true,
                reasoning_summary: Some("auto"),
                temperature: 0.5,
            },
        )
        .expect("encode request");

        assert_eq!(body["prompt_cache_key"], "cache-key");
        assert_eq!(body["store"], false);
        assert_eq!(body["stream"], false);
        assert_eq!(body["temperature"], json!(0.5_f32));
        assert_eq!(body["include"], json!(["reasoning.encrypted_content"]));
        assert_eq!(body["reasoning"]["effort"], "medium");
        assert_eq!(body["reasoning"]["summary"], "auto");
        assert!(
            body["input"]
                .as_array()
                .expect("input")
                .iter()
                .any(|item| item["type"] == "function_call_output")
        );
        assert!(body["input"].as_array().expect("input").iter().any(|item| {
            item["type"] == "message"
                && item["role"] == "assistant"
                && item["id"] == "msg_history_1"
                && item["status"] == "completed"
        }));
        assert!(
            body["tools"]
                .as_array()
                .expect("tools")
                .iter()
                .any(|tool| tool["name"] == "fs_read")
        );
    }

    #[test]
    fn openai_responses_request_omits_invalid_assistant_message_ids() {
        let mut request = sample_request(ApiKind::OpenAiResponses, ProviderKind::OpenAi);
        let Message::Assistant(assistant) = &mut request.messages[1] else {
            panic!("expected assistant history");
        };
        assistant.id = MessageId::from("resp_123");

        let body = encode_responses_request(
            &request,
            ResponsesRequestOptions {
                stream: false,
                store: Some(false),
                prompt_cache_key: None,
                include_encrypted_reasoning: false,
                reasoning_summary: None,
                temperature: DEFAULT_TEMPERATURE,
            },
        )
        .expect("encode request");

        let assistant_message = body["input"]
            .as_array()
            .expect("input")
            .iter()
            .find(|item| item["type"] == "message" && item["role"] == "assistant")
            .expect("assistant message");
        assert!(assistant_message.get("id").is_none());
    }

    #[test]
    fn openrouter_responses_request_omits_openai_only_fields() {
        let request = sample_request(ApiKind::OpenAiResponses, ProviderKind::OpenRouter);
        let body = encode_responses_request(
            &request,
            ResponsesRequestOptions {
                stream: true,
                store: None,
                prompt_cache_key: None,
                include_encrypted_reasoning: false,
                reasoning_summary: None,
                temperature: DEFAULT_TEMPERATURE,
            },
        )
        .expect("encode request");
        let input = body["input"].as_array().expect("input");

        assert!(body.get("store").is_none());
        assert!(body.get("prompt_cache_key").is_none());
        assert!(body.get("include").is_none());
        assert_eq!(body["reasoning"]["effort"], "medium");
        assert!(body["reasoning"].get("summary").is_none());
        assert!(input.iter().any(|item| {
            item["type"] == "function_call"
                && item["id"] == "fc_call_123"
                && item["call_id"] == "call_123"
        }));
        assert!(input.iter().any(|item| {
            item["type"] == "function_call_output"
                && item["id"] == "fc_output_call_123"
                && item["call_id"] == "call_123"
        }));
    }

    #[test]
    fn openrouter_compaction_request_uses_responses_shape() {
        let request = sample_compaction_request(ProviderKind::OpenRouter);
        let body = encode_openrouter_compact_request(&request).expect("encode compaction request");
        let input = body["input"].as_array().expect("input");

        assert_eq!(body["model"], "gpt-5");
        assert_eq!(body["instructions"], "Summarize the session");
        assert_eq!(body["stream"], false);
        assert_eq!(body["store"], false);
        assert!(input.iter().any(|item| item["role"] == "developer"));
        assert!(input.iter().any(|item| item["role"] == "user"));
        assert!(input.iter().any(|item| item["type"] == "function_call"));
        assert!(
            input
                .iter()
                .any(|item| item["type"] == "function_call_output")
        );
    }

    #[test]
    fn openrouter_compaction_response_wraps_summary_in_developer_message() {
        let response = json!({
            "output": [
                {
                    "type": "reasoning",
                    "summary": [{"text": "ignore reasoning"}]
                },
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "## User Intent"
                        },
                        {
                            "type": "output_text",
                            "text": "\n- finish the fix"
                        }
                    ]
                },
                {
                    "type": "message",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "## Completed Work"
                        }
                    ]
                }
            ],
            "usage": {
                "input_tokens": 17,
                "output_tokens": 9,
                "input_tokens_details": {
                    "cache_creation_tokens": 0,
                    "cached_tokens": 0
                }
            }
        });

        let decoded =
            decode_openrouter_compact_response(&response).expect("decode compaction response");

        assert_eq!(decoded.usage.input_tokens, 17);
        assert_eq!(decoded.usage.output_tokens, 9);
        assert_eq!(
            decoded.output,
            vec![json!({
                "type": "message",
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": "<compaction>\n## User Intent\n- finish the fix\n\n## Completed Work\n</compaction>"
                    }
                ],
            })]
        );
    }

    #[test]
    fn openrouter_compaction_response_returns_empty_output_for_empty_text() {
        let response = json!({
            "output": [
                { "type": "reasoning", "summary": [{"text": "ignore"}] },
                { "type": "function_call", "name": "read" }
            ],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 0
            }
        });

        let decoded =
            decode_openrouter_compact_response(&response).expect("decode compaction response");

        assert!(decoded.output.is_empty());
        assert_eq!(decoded.usage.input_tokens, 1);
    }

    #[test]
    fn openai_responses_request_bounds_legacy_message_and_tool_item_ids() {
        let mut request = sample_request(ApiKind::OpenAiResponses, ProviderKind::OpenAi);
        let long_response_id = "resp_06e44240522f2cae0169ddd62f88908190a4bc4203232681a5";
        let long_call_id = "call_1234567890abcdef1234567890abcdef1234567890abcdef12345".to_owned();
        let legacy_message_id = format!("msg_response_{long_response_id}");

        let Message::Assistant(assistant) = &mut request.messages[1] else {
            panic!("expected assistant history");
        };
        assistant.id = MessageId::from(legacy_message_id.clone());
        let tool_call = assistant
            .parts
            .iter_mut()
            .find_map(|part| match part {
                AssistantPart::ToolCall(call) => Some(call),
                _ => None,
            })
            .expect("assistant tool call");
        tool_call.id = ToolCallId::from(long_call_id.clone());

        let Message::Tool(tool) = &mut request.messages[2] else {
            panic!("expected tool message");
        };
        tool.call_id = ToolCallId::from(long_call_id.clone());

        let body = encode_responses_request(
            &request,
            ResponsesRequestOptions {
                stream: false,
                store: Some(false),
                prompt_cache_key: None,
                include_encrypted_reasoning: false,
                reasoning_summary: None,
                temperature: DEFAULT_TEMPERATURE,
            },
        )
        .expect("encode request");

        let input = body["input"].as_array().expect("input");
        for item in input {
            if let Some(id) = item.get("id").and_then(Value::as_str) {
                assert!(
                    id.len() <= RESPONSES_ITEM_ID_MAX_LEN,
                    "expected bounded input id, got {id}"
                );
            }
        }

        let assistant_message = input
            .iter()
            .find(|item| item["type"] == "message" && item["role"] == "assistant")
            .expect("assistant message");
        let assistant_id = assistant_message["id"]
            .as_str()
            .expect("assistant message id");
        assert!(assistant_id.starts_with("msg_response_"));
        assert!(assistant_id.len() <= RESPONSES_ITEM_ID_MAX_LEN);
        assert_ne!(assistant_id, legacy_message_id);

        let function_call = input
            .iter()
            .find(|item| item["type"] == "function_call")
            .expect("function call");
        assert_eq!(function_call["call_id"], long_call_id);
        assert!(
            function_call["id"]
                .as_str()
                .expect("function call id")
                .starts_with("fc_")
        );
        assert!(
            function_call["id"]
                .as_str()
                .expect("function call id")
                .len()
                <= RESPONSES_ITEM_ID_MAX_LEN
        );

        let function_call_output = input
            .iter()
            .find(|item| item["type"] == "function_call_output")
            .expect("function call output");
        assert_eq!(function_call_output["call_id"], long_call_id);
        assert!(
            function_call_output["id"]
                .as_str()
                .expect("function call output id")
                .starts_with("fc_output_")
        );
        assert!(
            function_call_output["id"]
                .as_str()
                .expect("function call output id")
                .len()
                <= RESPONSES_ITEM_ID_MAX_LEN
        );
    }

    #[test]
    fn synthesized_responses_message_id_stays_within_limit() {
        let response_id = "resp_06e44240522f2cae0169ddd62f88908190a4bc4203232681a5";

        let message_id = synthesized_responses_message_id(response_id);

        assert!(message_id.0.starts_with("msg_response_"));
        assert!(message_id.0.len() <= RESPONSES_ITEM_ID_MAX_LEN);
    }

    #[test]
    fn validate_responses_input_item_ids_rejects_overlong_ids() {
        let overlong_id = "x".repeat(RESPONSES_ITEM_ID_MAX_LEN + 1);

        let error = validate_responses_input_item_ids(&[json!({
            "type": "message",
            "id": overlong_id,
        })])
        .expect_err("expected validation failure");

        assert_eq!(
            error.to_string(),
            "failed to encode openai responses request: input[0].id length 65 exceeds 64"
        );
    }

    #[test]
    fn chat_response_maps_text_and_function_calls() {
        let request = sample_request(ApiKind::OpenAiChat, ProviderKind::OpenRouter);
        let response = json!({
            "id": "chatcmpl_123",
            "choices": [
                {
                    "finish_reason": "tool_calls",
                    "message": {
                        "role": "assistant",
                        "content": "done",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "fs_read",
                                    "arguments": "{\"path\":\"README.md\"}"
                                }
                            }
                        ]
                    }
                }
            ],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 4,
                "prompt_tokens_details": { "cached_tokens": 3 }
            }
        });

        let events = decode_chat_response(&request, &response).expect("decode response");

        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::TextDelta { delta, .. } if delta == "done"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::ToolCallStart { name, .. } if name.0 == "read"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::UsageUpdate { usage } if usage.cache_read_input_tokens == 3
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::MessageEnd { stop_reason, .. } if *stop_reason == StopReason::ToolUse
        )));
    }

    #[test]
    fn responses_response_maps_text_and_function_calls() {
        let request = sample_request(ApiKind::OpenAiResponses, ProviderKind::OpenAi);
        let response = json!({
            "id": "resp_123",
            "output": [
                {
                    "type": "message",
                    "content": [
                        { "type": "output_text", "text": "done" }
                    ]
                },
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "fs_read",
                    "arguments": "{\"path\":\"README.md\"}"
                }
            ],
            "usage": {
                "input_tokens": 12,
                "output_tokens": 4,
                "input_tokens_details": { "cached_tokens": 3 }
            }
        });

        let events = decode_responses_response(&request, &response).expect("decode response");

        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::TextDelta { delta, .. } if delta == "done"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::ToolCallStart { name, .. } if name.0 == "read"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::UsageUpdate { usage } if usage.cache_read_input_tokens == 3
        )));
    }

    #[test]
    fn responses_stream_decoder_maps_text_reasoning_tool_calls_and_usage() {
        let request = sample_request(ApiKind::OpenAiResponses, ProviderKind::OpenAi);
        let mut decoder = ResponsesStreamDecoder::new(&request, true);
        let stream_events = vec![
            json!({
                "type": "response.created",
                "sequence_number": 0,
                "response": {
                    "id": "resp_123",
                    "created_at": 0,
                    "model": "gpt-5",
                    "object": "response",
                    "output": [],
                    "status": "in_progress"
                }
            }),
            json!({
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": []
                }
            }),
            json!({
                "type": "response.output_text.delta",
                "sequence_number": 2,
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": "done"
            }),
            json!({
                "type": "response.output_item.done",
                "sequence_number": 3,
                "output_index": 0,
                "item": {
                    "type": "message",
                    "id": "msg_1",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "done",
                            "annotations": []
                        }
                    ]
                }
            }),
            json!({
                "type": "response.output_item.added",
                "sequence_number": 4,
                "output_index": 1,
                "item": {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [],
                    "status": "in_progress"
                }
            }),
            json!({
                "type": "response.reasoning_summary_part.added",
                "sequence_number": 5,
                "item_id": "rs_1",
                "output_index": 1,
                "summary_index": 0,
                "part": {
                    "type": "summary_text",
                    "text": ""
                }
            }),
            json!({
                "type": "response.reasoning_summary_text.delta",
                "sequence_number": 6,
                "item_id": "rs_1",
                "output_index": 1,
                "summary_index": 0,
                "delta": "think"
            }),
            json!({
                "type": "response.output_item.done",
                "sequence_number": 7,
                "output_index": 1,
                "item": {
                    "type": "reasoning",
                    "id": "rs_1",
                    "summary": [
                        {
                            "type": "summary_text",
                            "text": "think"
                        }
                    ],
                    "encrypted_content": "enc_123",
                    "status": "completed"
                }
            }),
            json!({
                "type": "response.output_item.added",
                "sequence_number": 8,
                "output_index": 2,
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_123",
                    "name": "fs_read",
                    "arguments": "",
                    "status": "in_progress"
                }
            }),
            json!({
                "type": "response.function_call_arguments.delta",
                "sequence_number": 9,
                "item_id": "fc_1",
                "output_index": 2,
                "delta": "{\"path\":"
            }),
            json!({
                "type": "response.function_call_arguments.done",
                "sequence_number": 10,
                "item_id": "fc_1",
                "output_index": 2,
                "arguments": "{\"path\":\"README.md\"}"
            }),
            json!({
                "type": "response.output_item.done",
                "sequence_number": 11,
                "output_index": 2,
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_123",
                    "name": "fs_read",
                    "arguments": "{\"path\":\"README.md\"}",
                    "status": "completed"
                }
            }),
            json!({
                "type": "response.completed",
                "sequence_number": 12,
                "response": {
                    "id": "resp_123",
                    "created_at": 0,
                    "model": "gpt-5",
                    "object": "response",
                    "output": [],
                    "status": "completed",
                    "usage": {
                        "input_tokens": 12,
                        "input_tokens_details": {
                            "cached_tokens": 3
                        },
                        "output_tokens": 4,
                        "output_tokens_details": {
                            "reasoning_tokens": 1
                        },
                        "total_tokens": 16
                    }
                }
            }),
        ];

        let decoded = stream_events
            .into_iter()
            .flat_map(|event| {
                let event = serde_json::from_value(event).expect("parse stream event");
                decoder.decode(event).expect("decode stream event")
            })
            .collect::<Vec<_>>();

        assert!(decoded.iter().any(|event| matches!(
            event,
            StreamEvent::MessageStart { id } if id.0 == "msg_1"
        )));
        assert!(decoded.iter().any(|event| matches!(
            event,
            StreamEvent::TextDelta { delta, .. } if delta == "done"
        )));
        assert!(decoded.iter().any(|event| matches!(
            event,
            StreamEvent::ThinkingDelta { delta, .. } if delta == "think"
        )));
        assert!(decoded.iter().any(|event| matches!(
            event,
            StreamEvent::ThinkingEnd { signature, .. } if signature.as_deref() == Some("enc_123")
        )));
        assert!(decoded.iter().any(|event| matches!(
            event,
            StreamEvent::ToolCallStart { name, .. } if name.0 == "read"
        )));
        assert!(decoded.iter().any(|event| matches!(
            event,
            StreamEvent::ToolArgsDelta { delta, .. } if delta.contains("README.md")
        )));
        assert!(decoded.iter().any(|event| matches!(
            event,
            StreamEvent::UsageUpdate { usage } if usage.cache_read_input_tokens == 3
        )));
        assert!(decoded.iter().any(|event| matches!(
            event,
            StreamEvent::MessageEnd { stop_reason, .. } if *stop_reason == StopReason::ToolUse
        )));
    }

    #[test]
    fn responses_stream_decoder_does_not_reuse_response_id_for_tool_only_turns() {
        let request = sample_request(ApiKind::OpenAiResponses, ProviderKind::OpenAi);
        let mut decoder = ResponsesStreamDecoder::new(&request, true);
        let stream_events = vec![
            json!({
                "type": "response.created",
                "sequence_number": 0,
                "response": {
                    "id": "resp_123",
                    "created_at": 0,
                    "model": "gpt-5",
                    "object": "response",
                    "output": [],
                    "status": "in_progress"
                }
            }),
            json!({
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_123",
                    "name": "fs_read",
                    "arguments": "",
                    "status": "in_progress"
                }
            }),
            json!({
                "type": "response.function_call_arguments.done",
                "sequence_number": 2,
                "item_id": "fc_1",
                "output_index": 0,
                "arguments": "{\"path\":\"README.md\"}"
            }),
            json!({
                "type": "response.output_item.done",
                "sequence_number": 3,
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_123",
                    "name": "fs_read",
                    "arguments": "{\"path\":\"README.md\"}",
                    "status": "completed"
                }
            }),
            json!({
                "type": "response.completed",
                "sequence_number": 4,
                "response": {
                    "id": "resp_123",
                    "created_at": 0,
                    "model": "gpt-5",
                    "object": "response",
                    "output": [],
                    "status": "completed",
                    "usage": {
                        "input_tokens": 12,
                        "input_tokens_details": {
                            "cached_tokens": 0
                        },
                        "output_tokens": 4,
                        "output_tokens_details": {
                            "reasoning_tokens": 0
                        },
                        "total_tokens": 16
                    }
                }
            }),
        ];

        let decoded = stream_events
            .into_iter()
            .flat_map(|event| {
                let event = serde_json::from_value(event).expect("parse stream event");
                decoder.decode(event).expect("decode stream event")
            })
            .collect::<Vec<_>>();

        assert!(decoded.iter().any(|event| matches!(
            event,
            StreamEvent::MessageStart { id } if id.0 == "msg_response_resp_123"
        )));
        assert!(decoded.iter().any(|event| matches!(
            event,
            StreamEvent::ToolCallStart { name, .. } if name.0 == "read"
        )));
    }

    #[test]
    fn responses_response_uses_stable_tool_only_message_id() {
        let request = sample_request(ApiKind::OpenAiResponses, ProviderKind::OpenAi);
        let response = json!({
            "id": "resp_123",
            "output": [
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "fs_read",
                    "arguments": "{\"path\":\"README.md\"}"
                }
            ],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1
            }
        });

        let events = decode_responses_response(&request, &response).expect("decode response");

        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::MessageStart { id } if id.0 == "msg_response_resp_123"
        )));
    }

    #[test]
    fn responses_stream_decoder_ignores_duplicate_tool_completion_events() {
        let request = sample_request(ApiKind::OpenAiResponses, ProviderKind::OpenAi);
        let mut decoder = ResponsesStreamDecoder::new(&request, true);
        let stream_events = vec![
            json!({
                "type": "response.created",
                "sequence_number": 0,
                "response": {
                    "id": "resp_123",
                    "created_at": 0,
                    "model": "gpt-5",
                    "object": "response",
                    "output": [],
                    "status": "in_progress"
                }
            }),
            json!({
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_123",
                    "name": "fs_read",
                    "arguments": "",
                    "status": "in_progress"
                }
            }),
            json!({
                "type": "response.function_call_arguments.delta",
                "sequence_number": 2,
                "item_id": "fc_1",
                "output_index": 0,
                "delta": "{\"path\":"
            }),
            json!({
                "type": "response.function_call_arguments.done",
                "sequence_number": 3,
                "item_id": "fc_1",
                "output_index": 0,
                "arguments": "{\"path\":\"README.md\"}"
            }),
            json!({
                "type": "response.output_item.done",
                "sequence_number": 4,
                "output_index": 0,
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_123",
                    "name": "fs_read",
                    "arguments": "{\"path\":\"README.md\"}",
                    "status": "completed"
                }
            }),
        ];

        let decoded = stream_events
            .into_iter()
            .flat_map(|event| {
                let event = serde_json::from_value(event).expect("parse stream event");
                decoder.decode(event).expect("decode stream event")
            })
            .collect::<Vec<_>>();

        assert_eq!(
            decoded
                .iter()
                .filter(|event| matches!(event, StreamEvent::ToolCallStart { .. }))
                .count(),
            1
        );
        assert_eq!(
            decoded
                .iter()
                .filter(|event| matches!(event, StreamEvent::ToolArgsDelta { .. }))
                .count(),
            2
        );
        assert_eq!(
            decoded
                .iter()
                .filter(|event| matches!(event, StreamEvent::ToolCallEnd { .. }))
                .count(),
            1
        );
    }

    #[test]
    fn chat_request_uses_tool_messages_and_aliases() {
        let request = sample_request(ApiKind::OpenAiChat, ProviderKind::OpenRouter);
        let body = encode_chat_request(&request, false).expect("encode chat");
        let messages = body["messages"].as_array().expect("messages");

        assert_eq!(messages[0]["role"], "system");
        assert!(messages.iter().any(|message| message["role"] == "tool"));
        assert!(
            body["tools"]
                .as_array()
                .expect("tools")
                .iter()
                .any(|tool| tool["function"]["name"] == "fs_read")
        );
    }

    fn sample_request(api_kind: ApiKind, provider_kind: ProviderKind) -> ProviderRequest {
        let mut provider_aliases = IndexMap::new();
        provider_aliases.insert(provider_kind, ToolAlias::from("fs_read"));
        ProviderRequest {
            session_id: Default::default(),
            turn_id: TurnId::new(),
            model: ResolvedModel {
                role: ModelRole::default(),
                id: ModelId::from("model"),
                provider: ProviderName::from("provider"),
                provider_kind,
                api_kind,
                model: "gpt-5".to_owned(),
                max_input_tokens: Some(200_000),
                max_output_tokens: Some(8_192),
                reasoning: Some(ReasoningEffort::Medium),
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
                transcript: Vec::new(),
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
            messages: vec![
                Message::User(UserMessage::text("hello")),
                Message::Assistant(AssistantMessage {
                    id: MessageId::from("msg_history_1"),
                    created_at: Utc::now(),
                    parts: vec![
                        AssistantPart::Text {
                            text: "checked".to_owned(),
                        },
                        AssistantPart::ToolCall(ToolCall {
                            id: ToolCallId::from("call_123"),
                            name: "read".into(),
                            arguments: json!({"path": "README.md"}),
                        }),
                    ],
                    stop_reason: None,
                    usage: None,
                    replay_meta: Default::default(),
                }),
                Message::Tool(ToolResultMessage {
                    id: MessageId::from("tool_output_1"),
                    call_id: ToolCallId::from("call_123"),
                    content: ToolResult::Json {
                        value: json!({"ok": true}),
                    },
                    error: None,
                    created_at: Utc::now(),
                }),
            ],
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

    fn sample_compaction_request(provider_kind: ProviderKind) -> ProviderCompactionRequest {
        let request = sample_request(ApiKind::OpenAiResponses, provider_kind);
        ProviderCompactionRequest {
            session_id: request.session_id,
            model: request.model,
            compacted_prefix: vec![json!({
                "type": "message",
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": "[Compacted context]\n\nEarlier summary"
                    }
                ],
            })],
            messages: request.messages,
            tools: request.tools,
            instructions: "Summarize the session".to_owned(),
        }
    }
}
