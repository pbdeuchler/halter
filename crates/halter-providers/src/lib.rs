// pattern: Functional Core

mod anthropic;
mod anthropic_codec;
mod codec_common;
mod fake;
mod http_client;
mod openai_codec;
mod openai_error;
mod openai_rate_limit;
mod openai_rate_limit_policy;
mod registry;
mod responses_provider;
mod responses_transport;
#[cfg(test)]
mod test_http;
mod unsupported;

use async_trait::async_trait;
use futures::stream::BoxStream;
use halter_protocol::{
    ProviderCapabilities, ProviderCompactionRequest, ProviderCompactionResponse, ProviderError,
    ProviderRequest, StreamEvent,
};
use tokio_util::sync::CancellationToken;

pub use anthropic::AnthropicProvider;
pub use fake::FakeProvider;
pub use registry::ModelRegistry;
pub use responses_provider::ResponsesProvider;
pub use unsupported::UnsupportedProvider;

#[async_trait]
pub trait Provider: Send + Sync {
    fn capabilities(&self) -> ProviderCapabilities;

    async fn stream(
        &self,
        request: ProviderRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>>;

    async fn compact(
        &self,
        _request: ProviderCompactionRequest,
        _cancel: CancellationToken,
    ) -> anyhow::Result<ProviderCompactionResponse> {
        anyhow::bail!("failed to compact session: provider does not support compaction");
    }
}
