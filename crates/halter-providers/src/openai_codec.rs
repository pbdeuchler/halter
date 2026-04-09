// pattern: Functional Core

use std::collections::BTreeMap;

use anyhow::Context;
use async_openai::types::responses::{
    FunctionToolCall, OutputItem, OutputMessage, OutputMessageContent, ReasoningItem, Response,
    ResponseStreamEvent, SummaryPart,
};
use halter_protocol::{
    ApiKind, AssistantPart, BlockId, Message, MessageId, ProviderRequest, ReasoningEffort,
    StopReason, StreamEvent, Usage, UserPart,
};
use serde_json::{Map, Value, json};

use crate::codec_common::{
    assistant_text, canonical_tool_name, collect_system_text, data_url, document_filename,
    has_user_media, normalized_tool_call_id, tool_name_for_provider, tool_result_text, user_text,
};

#[derive(Debug, Clone, Copy)]
pub(crate) struct ResponsesRequestOptions<'a> {
    pub stream: bool,
    pub store: Option<bool>,
    pub prompt_cache_key: Option<&'a str>,
    pub include_encrypted_reasoning: bool,
    pub reasoning_summary: Option<&'a str>,
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
    body.insert(
        "input".to_owned(),
        Value::Array(encode_responses_input(request)?),
    );
    body.insert("stream".to_owned(), Value::Bool(options.stream));

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

#[cfg(test)]
pub(crate) fn decode_responses_response(
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
    });
    Ok(events)
}

#[derive(Debug, Clone)]
pub(crate) struct ResponsesStreamDecoder {
    tool_specs: Vec<halter_protocol::ToolSpec>,
    provider_kind: halter_protocol::ProviderKind,
    message_id: Option<MessageId>,
    response_id: Option<MessageId>,
    started: bool,
    saw_tool_call: bool,
    active_text: Option<PendingTextBlock>,
    active_reasoning: Option<PendingReasoningBlock>,
    active_tool_calls: BTreeMap<String, PendingToolCallBlock>,
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
    pub(crate) fn new(request: &ProviderRequest) -> Self {
        Self {
            tool_specs: request.tools.clone(),
            provider_kind: request.model.provider_kind,
            message_id: None,
            response_id: None,
            started: false,
            saw_tool_call: false,
            active_text: None,
            active_reasoning: None,
            active_tool_calls: BTreeMap::new(),
        }
    }

    pub(crate) fn decode(
        &mut self,
        event: ResponseStreamEvent,
    ) -> anyhow::Result<Vec<StreamEvent>> {
        let mut events = Vec::new();
        match event {
            ResponseStreamEvent::ResponseCreated(event) => {
                self.response_id = Some(MessageId::from(event.response.id));
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

        let message_id = self.message_id.clone().unwrap_or_else(|| {
            message_id
                .map(|id| MessageId::from(id.to_owned()))
                .or_else(|| self.response_id.clone())
                .unwrap_or_default()
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
            events.push(StreamEvent::ToolCallEnd { id: pending.id });
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
        events.push(StreamEvent::ToolCallEnd { id: block_id });
        Ok(())
    }

    fn finish_response(&mut self, response: &Response, events: &mut Vec<StreamEvent>) {
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
        });
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
    });
    Ok(events)
}

fn encode_responses_input(request: &ProviderRequest) -> anyhow::Result<Vec<Value>> {
    let mut input = Vec::new();
    if let Some(system) = collect_system_text(request) {
        input.push(encode_responses_developer_message(&system));
    }

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

    Some(json!({
        "type": "message",
        "role": "assistant",
        "id": message.id,
        "status": "completed",
        "content": [
            {
                "type": "output_text",
                "text": text,
                "annotations": [],
            }
        ],
    }))
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
    format!("fc_{}", normalized_tool_call_id(call_id))
}

fn responses_function_call_output_item_id(call_id: &halter_protocol::ToolCallId) -> String {
    format!("fc_output_{}", normalized_tool_call_id(call_id))
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

#[cfg(test)]
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
    item_id
        .map(str::to_owned)
        .unwrap_or_else(|| format!("output_index:{output_index}"))
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
        ApiKind, AssembledPrompt, AssistantMessage, AssistantPart, CacheScope, Message, MessageId,
        ModelId, ModelRole, PromptSegment, PromptSegmentId, ProviderKind, ProviderName,
        ResolvedModel, ToolAlias, ToolCall, ToolCallId, ToolCapabilities, ToolConcurrency,
        ToolResult, ToolResultMessage, ToolSpec, TurnId, UserMessage, Volatility,
    };
    use indexmap::IndexMap;
    use serde_json::json;

    use super::*;

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
            },
        )
        .expect("encode request");

        assert_eq!(body["prompt_cache_key"], "cache-key");
        assert_eq!(body["store"], false);
        assert_eq!(body["stream"], false);
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
        let mut decoder = ResponsesStreamDecoder::new(&request);
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
            },
            prompt: AssembledPrompt {
                segments: vec![PromptSegment {
                    id: PromptSegmentId::new(),
                    text: "follow plan".to_owned(),
                    volatility: Volatility::Static,
                    cache_scope: CacheScope::PrefixCacheable,
                    content_hash: "hash".to_owned(),
                }],
                transcript: Vec::new(),
                ordered_segments: Vec::new(),
                prefix_cache_key: "cache-key".to_owned(),
                rendered_prefix: "follow plan".to_owned(),
                rendered_transcript: String::new(),
                rendered: String::new(),
            },
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
        }
    }
}
