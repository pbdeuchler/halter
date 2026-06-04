// pattern: Functional Core
//
// Cloud browser provider abstraction. Each concrete provider returns the CDP
// websocket endpoint for a freshly-created remote browser session and knows
// how to release it. Configuration discovery and validation live in the
// concrete impls — the trait only carries lifecycle.

use async_trait::async_trait;

/// A live remote browser session. The CDP url is what playwright connects to;
/// the id is the provider-side handle used for [`BrowserProvider::close_session`].
#[derive(Debug, Clone)]
pub struct RemoteSession {
    pub id: String,
    pub cdp_url: String,
    pub features: Vec<&'static str>,
}

#[async_trait]
/// Provider abstraction for allocating and closing remote browser sessions.
pub trait BrowserProvider: Send + Sync + std::fmt::Debug {
    /// Short, stable identifier for diagnostics (`browserbase`, `local`, ...).
    fn name(&self) -> &'static str;

    /// Allocate a new remote browser and return its CDP endpoint.
    async fn create_session(&self) -> anyhow::Result<RemoteSession>;

    /// Release the provider-side resources for `session_id`. Must tolerate
    /// "already gone" without raising.
    async fn close_session(&self, session_id: &str) -> anyhow::Result<()>;
}
