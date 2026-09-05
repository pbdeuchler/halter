//! Provider adapters and model registry support.
//!
//! Providers translate the protocol-level [`ProviderRequest`] into upstream
//! API calls and stream protocol-level [`StreamEvent`] values back to the
//! runtime. The concrete adapters keep transport concerns here so the runtime
//! can stay provider-agnostic.
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
mod header_overrides;
mod http_client;
mod model_judge;
mod openai;
mod openai_codec;
mod openai_error;
mod openai_rate_limit;
mod openai_rate_limit_policy;
mod openrouter;
mod registry;
mod resilience;
mod responses_provider;
mod responses_transport;
mod retry;
mod secret;
#[cfg(test)]
pub(crate) mod test_http;
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
pub use halter_protocol::ProviderErrorKind;
pub use model_judge::{
    Candidate, FullTurnJudgePlan, FullTurnPanelist, MODEL_JUDGE_RANK_TOOL,
    MODEL_JUDGE_TRACE_TARGET, ModelJudgeMember, ModelJudgeProvider, run_panel_synthesis,
    synthesis_guidance_message,
};
pub use openai::{OpenAiOAuthCredentials, OpenAiProvider};
pub use openrouter::OpenRouterProvider;
pub use registry::ModelRegistry;
pub use resilience::{
    DefaultProviderErrorClassifier, ProviderErrorClassifier, ProviderTimeouts, ResiliencePolicy,
    ResilientProvider,
};
pub use retry::RetryPolicy;
pub use secret::SecretString;
pub use unsupported::UnsupportedProvider;

#[async_trait]
/// Common interface implemented by all model providers.
pub trait Provider: Send + Sync {
    /// Capability flags used by the runtime and prompt codecs.
    fn capabilities(&self) -> ProviderCapabilities;

    /// Start a streaming generation request.
    async fn stream(
        &self,
        request: ProviderRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>>;

    /// Compact a transcript into provider-native context items.
    ///
    /// This is the transport primitive behind the runtime's provider-delegated
    /// compaction strategy. Which messages a rewrite may replace is that
    /// strategy's policy, derived from
    /// [`ProviderCapabilities::compaction_strategy`]; an adapter only encodes
    /// the request it is handed. It is invoked when the runtime's token ledger
    /// says so and never by a server-side trigger: an adapter must not enable
    /// a provider's own automatic compaction.
    ///
    /// The default implementation rejects the call for providers without
    /// compaction support.
    async fn compact(
        &self,
        _request: ProviderCompactionRequest,
        _cancel: CancellationToken,
    ) -> anyhow::Result<ProviderCompactionResponse> {
        anyhow::bail!("failed to compact session: provider does not support compaction");
    }
}
