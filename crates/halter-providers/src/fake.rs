// pattern: Functional Core

use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};
use halter_protocol::{
    BlockId, Message, MessageId, ProviderCapabilities, ProviderError, ProviderRequest, StopReason,
    StreamEvent, Usage,
};
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
        ProviderCapabilities::default()
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
}
