// pattern: Functional Core

mod compact;
mod stream;

pub(crate) use compact::{
    decode_openrouter_compact_response, decode_responses_compact_response,
    encode_openrouter_compact_request, encode_responses_compact_request,
};
pub(crate) use stream::ResponsesStreamDecoder;

use halter_protocol::{
    ApiKind, AssistantPart, Message, MessageId, ProviderKind, ProviderRequest, ReasoningEffort,
    ToolSpec, Usage, UserPart,
};
use serde_json::{Map, Value, json};

use crate::codec_common::{
    assistant_text, bounded_provider_id, bounded_provider_id_with_prefix, collect_system_text,
    data_url, document_filename, has_user_media, tool_name_for_provider, tool_result_text,
    user_text,
};

const RESPONSES_ITEM_ID_MAX_LEN: usize = 64;

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
    if request.model.api_kind() != ApiKind::OpenAiResponses {
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
) -> anyhow::Result<Vec<halter_protocol::StreamEvent>> {
    use halter_protocol::{BlockId, StreamEvent};

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
                saw_tool_call = true;
                let block_id = BlockId::new();
                let tool_call_id = halter_protocol::ToolCallId::from(
                    item.get("call_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned(),
                );
                let raw_name = item.get("name").and_then(Value::as_str).unwrap_or_default();
                let tool_name = crate::codec_common::canonical_tool_name(
                    raw_name,
                    &request.tools,
                    request.model.provider_kind,
                );
                let arguments = item
                    .get("arguments")
                    .map(openai_arguments_string)
                    .unwrap_or_default();
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

fn encode_responses_input(request: &ProviderRequest) -> anyhow::Result<Vec<Value>> {
    let mut input = Vec::new();
    if let Some(system) = collect_system_text(request) {
        input.push(encode_responses_developer_message(&system));
    }
    input.extend(request.compacted_prefix.clone());
    append_responses_messages(
        &mut input,
        &request.messages,
        &request.tools,
        request.model.provider_kind,
    )?;
    Ok(input)
}

/// Encode only the given message slice — used when chaining via `previous_response_id`.
/// Omits the developer/system message since the server already has it.
fn encode_responses_input_slice(
    messages: &[Message],
    request: &ProviderRequest,
) -> anyhow::Result<Vec<Value>> {
    let mut input = Vec::new();
    append_responses_messages(
        &mut input,
        messages,
        &request.tools,
        request.model.provider_kind,
    )?;
    Ok(input)
}

pub(super) fn append_responses_messages(
    input: &mut Vec<Value>,
    messages: &[Message],
    tools: &[ToolSpec],
    provider_kind: ProviderKind,
) -> anyhow::Result<()> {
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
                        input.push(encode_responses_tool_call(call, tools, provider_kind));
                    }
                }
            }
            Message::Tool(tool) => input.push(encode_responses_tool_output(tool)),
        }
    }
    Ok(())
}

pub(super) fn encode_responses_developer_message(text: &str) -> Value {
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
    tools: &[ToolSpec],
    provider_kind: ProviderKind,
) -> Value {
    json!({
        "type": "function_call",
        "id": responses_function_call_item_id(&call.id),
        "call_id": call.id,
        "name": tool_name_for_provider(&call.name, tools, provider_kind),
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

pub(super) fn synthesized_responses_message_id(response_id: &str) -> MessageId {
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

pub(super) fn is_responses_message_item_id(id: &str) -> bool {
    id.starts_with("msg_")
}

pub(super) fn validate_responses_input_item_ids(input: &[Value]) -> anyhow::Result<()> {
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
fn decode_responses_stop_reason(
    response: &Value,
    saw_tool_call: bool,
) -> halter_protocol::StopReason {
    use halter_protocol::StopReason;

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

pub(super) fn decode_openai_usage(response: &Value) -> Usage {
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

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use halter_protocol::{
        AssembledPrompt, AssistantMessage, AssistantPart, CacheScope, Message, MessageId, ModelId,
        ModelRole, ProviderCompactionRequest, PromptSegment, PromptSegmentId, ProviderKind,
        ProviderName, ProviderRequest, ResolvedModel, StopReason, StreamEvent, ToolAlias, ToolCall,
        ToolCallId, ToolCapabilities, ToolConcurrency, ToolResult, ToolResultMessage, ToolSpec,
        TurnId, UserMessage, Volatility,
    };
    use indexmap::IndexMap;
    use serde_json::json;

    use super::*;

    #[test]
    fn openai_responses_request_includes_prompt_cache_key_and_structured_history() {
        let request = sample_request(ProviderKind::OpenAi);
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
    fn openai_responses_request_omits_invalid_assistant_message_ids() {
        let mut request = sample_request(ProviderKind::OpenAi);
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
        let request = sample_request(ProviderKind::OpenRouter);
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
                        "text": "[Compacted context]\n\n## User Intent\n- finish the fix\n\n## Completed Work"
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
        let mut request = sample_request(ProviderKind::OpenAi);
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
    fn responses_response_maps_text_and_function_calls() {
        let request = sample_request(ProviderKind::OpenAi);
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
        let request = sample_request(ProviderKind::OpenAi);
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
        let request = sample_request(ProviderKind::OpenAi);
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
        let request = sample_request(ProviderKind::OpenAi);
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
        let request = sample_request(ProviderKind::OpenAi);
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
            json!({
                "type": "response.output_item.done",
                "sequence_number": 5,
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

    fn sample_request(provider_kind: ProviderKind) -> ProviderRequest {
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
                model: "gpt-5".to_owned(),
                max_input_tokens: Some(200_000),
                max_output_tokens: Some(8_192),
                reasoning: Some(halter_protocol::ReasoningEffort::Medium),
                tokens_per_minute: None,
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
        let request = sample_request(provider_kind);
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
