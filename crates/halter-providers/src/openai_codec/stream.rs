// pattern: Imperative Shell

use std::collections::{BTreeMap, BTreeSet};

use anyhow::Context;
use async_openai::types::responses::{
    FunctionToolCall, OutputItem, OutputMessage, OutputMessageContent, ReasoningItem, Response,
    ResponseStreamEvent, SummaryPart,
};
use halter_protocol::{
    BlockId, MessageId, ProviderKind, ProviderRequest, StopReason, StreamEvent, ToolSpec, Usage,
};

use crate::codec_common::canonical_tool_name;

use super::{is_responses_message_item_id, synthesized_responses_message_id};

#[derive(Debug, Clone)]
pub(crate) struct ResponsesStreamDecoder {
    tool_specs: Vec<ToolSpec>,
    provider_kind: ProviderKind,
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

        let message_id = self.message_id.clone().unwrap_or_else(|| {
            message_id
                .filter(|id| is_responses_message_item_id(id))
                .map(|id| MessageId::from(id.to_owned()))
                .or_else(|| {
                    self.response_id
                        .as_deref()
                        .map(synthesized_responses_message_id)
                })
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

fn decode_responses_usage(usage: &async_openai::types::responses::ResponseUsage) -> Usage {
    Usage {
        input_tokens: u64::from(usage.input_tokens),
        output_tokens: u64::from(usage.output_tokens),
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: u64::from(usage.input_tokens_details.cached_tokens),
    }
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
