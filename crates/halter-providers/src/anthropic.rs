// pattern: Imperative Shell

use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};
use halter_protocol::{
    ApiKind, DEFAULT_TEMPERATURE, ProviderCapabilities, ProviderError, ProviderRequest,
    StreamEvent, ToolCallIdPolicy,
};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

use crate::Provider;
use crate::anthropic_codec;
use crate::header_overrides::HeaderOverrides;
use crate::http_client::{JsonHttpClient, JsonRequest};
use crate::secret::SecretString;

const ANTHROPIC_MESSAGES_PATH: &str = "/v1/messages";

#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    api_key: SecretString,
    base_url: String,
    client: JsonHttpClient,
    header_overrides: HeaderOverrides,
    temperature: f32,
}

impl AnthropicProvider {
    pub fn new(
        api_key: impl Into<SecretString>,
        base_url: impl Into<String>,
    ) -> anyhow::Result<Self> {
        Self::new_with_headers(api_key, base_url, &[], DEFAULT_TEMPERATURE)
    }

    /// Same as [`AnthropicProvider::new`] but also accepts user-configured
    /// header overrides that replace any default or hardcoded header
    /// (`x-api-key`, `anthropic-version`, `Content-Type`) case-insensitively.
    /// `temperature` is forwarded verbatim into every request body.
    pub fn new_with_headers(
        api_key: impl Into<SecretString>,
        base_url: impl Into<String>,
        header_overrides: &[(String, String)],
        temperature: f32,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            client: JsonHttpClient::try_new()?,
            header_overrides: HeaderOverrides::new(header_overrides)?,
            temperature,
        })
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supports_tools: true,
            supports_streaming: false,
            supports_reasoning: true,
            supports_interleaved_reasoning: false,
            supports_images: true,
            supports_documents: true,
            supports_prompt_cache: false,
            supports_compaction: false,
            supports_tool_result_media: false,
            requires_non_empty_assistant_content: true,
            tool_call_id_policy: ToolCallIdPolicy::StableReplayNormalized,
            max_input_tokens: 200_000,
            max_output_tokens: 8_192,
        }
    }

    async fn stream(
        &self,
        request: ProviderRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
        if request.model.api_kind != ApiKind::AnthropicMessages {
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

        let body = anthropic_codec::encode_request(&request, self.temperature)?;
        let default_headers = vec![
            (
                "x-api-key".to_owned(),
                self.api_key.expose_secret().to_owned(),
            ),
            ("anthropic-version".to_owned(), "2023-06-01".to_owned()),
        ];
        let response = self
            .client
            .post_json(
                JsonRequest {
                    provider_label: "anthropic",
                    url: provider_url(&self.base_url, ANTHROPIC_MESSAGES_PATH),
                    headers: self.header_overrides.merge_string_pairs(default_headers),
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

fn provider_url(base_url: &str, path: &str) -> String {
    format!("{}{}", base_url.trim_end_matches('/'), path)
}
