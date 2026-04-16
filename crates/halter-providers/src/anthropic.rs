// pattern: Imperative Shell

use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};
use halter_protocol::{
    ApiKind, ProviderCapabilities, ProviderError, ProviderKind, ProviderRequest, StreamEvent,
};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

use crate::Provider;
use crate::anthropic_codec;
use crate::codec_common::provider_url;
use crate::http_client::{JsonHttpClient, JsonRequest};

const ANTHROPIC_MESSAGES_PATH: &str = "/v1/messages";

#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    api_key: String,
    base_url: String,
    client: JsonHttpClient,
}

impl AnthropicProvider {
    #[must_use]
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            client: JsonHttpClient::default(),
        }
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::for_provider(ProviderKind::Anthropic)
    }

    async fn stream(
        &self,
        request: ProviderRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
        if request.model.api_kind() != ApiKind::AnthropicMessages {
            anyhow::bail!(
                "failed to execute provider request: anthropic provider requires anthropic_messages api kind"
            );
        }
        info!(
            provider = "anthropic",
            session_id = %request.session_id,
            turn_id = %request.turn_id,
            model = %request.model.model,
            message_count = request.messages.len(),
            tool_count = request.tools.len(),
            "starting anthropic request"
        );

        let body = anthropic_codec::encode_request(&request)?;
        let response = self
            .client
            .post_json(
                JsonRequest {
                    provider_label: "anthropic",
                    url: provider_url(&self.base_url, ANTHROPIC_MESSAGES_PATH),
                    headers: vec![
                        ("x-api-key".to_owned(), self.api_key.clone()),
                        ("anthropic-version".to_owned(), "2023-06-01".to_owned()),
                    ],
                    body,
                },
                cancel,
            )
            .await?;
        let events = anthropic_codec::decode_response(&request, &response)?;
        debug!(event_count = events.len(), "decoded anthropic response");
        Ok(stream::iter(events.into_iter().map(Ok)).boxed())
    }
}

