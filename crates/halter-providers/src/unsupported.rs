// pattern: Functional Core

use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};
use halter_protocol::{
    ProviderCapabilities, ProviderError, ProviderKind, ProviderRequest, StreamEvent,
};
use tokio_util::sync::CancellationToken;

use crate::Provider;

#[derive(Debug, Clone)]
/// Provider stub used when a provider kind is configured but unavailable.
pub struct UnsupportedProvider {
    kind: ProviderKind,
}

impl UnsupportedProvider {
    /// Construct a stub for the unavailable provider kind.
    #[must_use]
    pub fn new(kind: ProviderKind) -> Self {
        Self { kind }
    }
}

#[async_trait]
impl Provider for UnsupportedProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::default()
    }

    async fn stream(
        &self,
        _request: ProviderRequest,
        _cancel: CancellationToken,
    ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
        let error = ProviderError::new(
            format!(
                "failed to execute provider request: {:?} transport is not wired in this build",
                self.kind
            ),
            false,
        );
        Ok(stream::iter(vec![Err(error)]).boxed())
    }
}
