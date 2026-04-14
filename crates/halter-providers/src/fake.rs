// pattern: Functional Core

use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};
use halter_protocol::{
    BlockId, Message, MessageId, ProviderCapabilities, ProviderCompactionRequest,
    ProviderCompactionResponse, ProviderError, ProviderRequest, StopReason, StreamEvent, Usage,
};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::Provider;

#[derive(Debug, Clone)]
pub struct FakeProvider {
    prefix: String,
}

impl Default for FakeProvider {
    fn default() -> Self {
        Self {
            prefix: "fake>".to_owned(),
        }
    }
}

impl FakeProvider {
    #[must_use]
    pub fn with_prefix(prefix: impl Into<String>) -> Self {
        Self {
            prefix: prefix.into(),
        }
    }

    fn render_reply(&self, request: &ProviderRequest) -> String {
        let latest_user_text = request
            .messages
            .iter()
            .rev()
            .find_map(|message| match message {
                Message::User(user) => Some(user.plain_text()),
                Message::System(_) | Message::Assistant(_) | Message::Tool(_) => None,
            })
            .unwrap_or_else(|| "empty turn".to_owned());

        format!(
            "{} {} [{}]",
            self.prefix, latest_user_text, request.model.model
        )
    }
}

#[async_trait]
impl Provider for FakeProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supports_compaction: true,
            ..ProviderCapabilities::default()
        }
    }

    async fn stream(
        &self,
        request: ProviderRequest,
        _cancel: CancellationToken,
    ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
        let message_id = MessageId::new();
        let block_id = BlockId::new();
        let reply = self.render_reply(&request);
        let usage = Usage {
            input_tokens: request.messages.len() as u64 * 8,
            output_tokens: reply.split_whitespace().count() as u64,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        };
        let events = vec![
            Ok(StreamEvent::MessageStart {
                id: message_id.clone(),
            }),
            Ok(StreamEvent::TextStart {
                id: block_id.clone(),
            }),
            Ok(StreamEvent::TextDelta {
                id: block_id.clone(),
                delta: reply.clone(),
            }),
            Ok(StreamEvent::TextEnd {
                id: block_id.clone(),
            }),
            Ok(StreamEvent::UsageUpdate {
                usage: usage.clone(),
            }),
            Ok(StreamEvent::MessageEnd {
                id: message_id,
                stop_reason: StopReason::EndTurn,
                response_id: None,
            }),
        ];
        Ok(stream::iter(events).boxed())
    }

    async fn compact(
        &self,
        request: ProviderCompactionRequest,
        _cancel: CancellationToken,
    ) -> anyhow::Result<ProviderCompactionResponse> {
        let mut items = request
            .compacted_prefix
            .iter()
            .map(render_compaction_input_item)
            .collect::<Vec<_>>();
        items.extend(request.messages.iter().map(render_compaction_message));
        let summary = items
            .into_iter()
            .filter(|item| !item.is_empty())
            .collect::<Vec<_>>()
            .join("\n");
        Ok(ProviderCompactionResponse {
            output: vec![json!({
                "type": "compaction",
                "id": format!("cmp_{}", request.session_id.0),
                "encrypted_content": summary,
            })],
            usage: Usage {
                input_tokens: (request.compacted_prefix.len() + request.messages.len()) as u64 * 8,
                output_tokens: 8,
                cache_creation_input_tokens: 0,
                cache_read_input_tokens: 0,
            },
        })
    }
}

fn render_compaction_input_item(item: &Value) -> String {
    match item {
        Value::Object(map) => {
            if let Some(role) = map.get("role").and_then(Value::as_str) {
                return format!("{role}: {}", item);
            }
            if let Some(kind) = map.get("type").and_then(Value::as_str) {
                return format!("{kind}: {}", item);
            }
            item.to_string()
        }
        _ => item.to_string(),
    }
}

fn render_compaction_message(message: &Message) -> String {
    match message {
        Message::System(message) => format!("system: {}", message.text),
        Message::User(message) => format!("user: {}", message.plain_text()),
        Message::Assistant(message) => format!(
            "assistant: {}",
            message
                .parts
                .iter()
                .map(|part| match part {
                    halter_protocol::AssistantPart::Text { text } => text.clone(),
                    halter_protocol::AssistantPart::Thinking(block) => block.text.clone(),
                    halter_protocol::AssistantPart::ToolCall(call) => {
                        format!("tool_call {} {}", call.name, call.arguments)
                    }
                })
                .collect::<Vec<_>>()
                .join("\n")
        ),
        Message::Tool(message) => match &message.content {
            halter_protocol::ToolResult::Empty => "tool: <empty>".to_owned(),
            halter_protocol::ToolResult::Text { text } => format!("tool: {text}"),
            halter_protocol::ToolResult::Json { value } => format!("tool: {value}"),
        },
    }
}
