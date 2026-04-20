// pattern: Crate Boundary
//
// This file is the `halter-providers` crate root. It declares modules and
// re-exports the public `Provider` trait + provider constructors. The trait
// declaration is neither a "functional core" nor an "imperative shell"
// (finding L9) — those labels apply to the concrete provider impls (e.g.
// `anthropic_codec.rs` = Functional Core, `responses_transport.rs` =
// Imperative Shell).

mod anthropic;
mod anthropic_codec;
mod codec_common;
mod fake;
mod http_client;
mod openai;
mod openai_codec;
mod openai_error;
mod openai_rate_limit;
mod openai_rate_limit_policy;
mod openrouter;
mod registry;
mod responses_provider;
mod responses_transport;
mod retry;
mod secret;
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
pub use openai::OpenAiProvider;
pub use openrouter::OpenRouterProvider;
pub use registry::ModelRegistry;
pub use secret::SecretString;
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
