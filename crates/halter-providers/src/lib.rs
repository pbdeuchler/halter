// pattern: Functional Core

mod anthropic;
mod anthropic_codec;
mod codec_common;
mod fake;
mod http_client;
mod openai;
mod openai_codec;
mod openai_rate_limit;
mod openai_rate_limit_policy;
mod openrouter;
mod registry;
mod responses_provider;
mod responses_transport;
mod unsupported;

use async_trait::async_trait;
use futures::stream::BoxStream;
use halter_protocol::{ProviderCapabilities, ProviderError, ProviderRequest, StreamEvent};
use tokio_util::sync::CancellationToken;

pub use anthropic::AnthropicProvider;
pub use fake::FakeProvider;
pub use openai::OpenAiProvider;
pub use openrouter::OpenRouterProvider;
pub use registry::ModelRegistry;
pub use unsupported::UnsupportedProvider;

#[async_trait]
pub trait Provider: Send + Sync {
    fn capabilities(&self) -> ProviderCapabilities;

    async fn stream(
        &self,
        request: ProviderRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>>;
}
