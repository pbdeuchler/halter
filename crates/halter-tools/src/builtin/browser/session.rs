// pattern: Imperative Shell
//
// One BrowserSession per halter SessionId. Holds a provider-side cloud
// session, a playwright Browser/Page connected to that session via CDP, and
// drops both on close. The playwright server process itself is shared across
// all sessions in this process via a OnceCell — spawning the Node driver per
// session would be wasteful.

use std::sync::Arc;
use std::time::Duration;

use playwright_rs::protocol::{
    GotoOptions, Page, Playwright, ScreenshotOptions, ScreenshotType, WaitUntil,
};
use tokio::sync::OnceCell;
use tracing::{debug, warn};

use super::provider::{BrowserProvider, RemoteSession};

const NAVIGATION_TIMEOUT: Duration = Duration::from_secs(30);

static PLAYWRIGHT: OnceCell<Arc<Playwright>> = OnceCell::const_new();

/// Returns the process-wide Playwright server handle, launching it on first
/// use. Subsequent calls reuse the same Node driver subprocess.
async fn shared_playwright() -> anyhow::Result<Arc<Playwright>> {
    PLAYWRIGHT
        .get_or_try_init(|| async {
            let pw = Playwright::launch().await.map_err(|err| {
                anyhow::anyhow!(
                    "failed to launch playwright driver (is Node.js + the playwright \
                     browsers installed? `npx playwright install`): {err}"
                )
            })?;
            Ok::<_, anyhow::Error>(Arc::new(pw))
        })
        .await
        .cloned()
}

/// Live browser session bound to a specific halter SessionId.
///
/// The session holds:
/// 1. The provider-side cloud session id (so we can release it on close).
/// 2. The provider handle (so close can speak to the right API).
/// 3. The playwright Page we drive on every action.
///
/// `Drop` spawns a best-effort cleanup task for cases where the agent never
/// fires the explicit `close` action. Callers should still prefer the
/// explicit `close` action — Drop cannot block on the cloud release call.
pub struct BrowserSession {
    provider: Arc<dyn BrowserProvider>,
    remote: RemoteSession,
    page: Page,
    /// Held to keep the connection alive for the page's lifetime.
    _browser: playwright_rs::protocol::Browser,
    last_url: Option<String>,
}

impl BrowserSession {
    pub async fn open(provider: Arc<dyn BrowserProvider>) -> anyhow::Result<Self> {
        let remote = provider.create_session().await?;
        let playwright = shared_playwright().await?;
        let browser = playwright
            .chromium()
            .connect_over_cdp(&remote.cdp_url, None)
            .await
            .map_err(|err| {
                anyhow::anyhow!(
                    "failed to connect to {} cdp endpoint: {err}",
                    provider.name()
                )
            })?;

        // BrowserBase / cloud providers usually surface an existing default
        // context with one page. Fall back to creating a fresh one if not.
        let page = match browser.contexts().into_iter().next() {
            Some(context) => {
                let mut pages = context.pages();
                if let Some(existing) = pages.pop() {
                    existing
                } else {
                    context.new_page().await.map_err(|err| {
                        anyhow::anyhow!("failed to open new page in cloud context: {err}")
                    })?
                }
            }
            None => {
                let context = browser
                    .new_context()
                    .await
                    .map_err(|err| anyhow::anyhow!("failed to create browser context: {err}"))?;
                context
                    .new_page()
                    .await
                    .map_err(|err| anyhow::anyhow!("failed to open new page: {err}"))?
            }
        };

        Ok(Self {
            provider,
            remote,
            page,
            _browser: browser,
            last_url: None,
        })
    }

    pub fn provider_name(&self) -> &'static str {
        self.provider.name()
    }

    pub fn features(&self) -> &[&'static str] {
        &self.remote.features
    }

    pub fn page(&self) -> &Page {
        &self.page
    }

    pub fn record_url(&mut self, url: String) {
        self.last_url = Some(url);
    }

    pub fn last_url(&self) -> Option<&str> {
        self.last_url.as_deref()
    }

    /// Eagerly closes the cloud session and the local connection. Idempotent
    /// — safe to call from both the explicit close action and the Drop path.
    pub async fn close(self) {
        let provider = self.provider.clone();
        let id = self.remote.id.clone();
        if let Err(err) = self.page.close().await {
            debug!(error = %err, "page.close failed during session shutdown");
        }
        if let Err(err) = provider.close_session(&id).await {
            warn!(error = %err, session_id = %id, "failed to release cloud browser session");
        }
    }
}

impl Drop for BrowserSession {
    fn drop(&mut self) {
        // Best-effort: try to release the remote session even when the agent
        // forgot to call `close`. Spawn a detached task because Drop can't
        // await — if no runtime is available (e.g. shutdown), we silently
        // accept the leaked session rather than panic. Cloud-side timeouts
        // will eventually reap it.
        let provider = self.provider.clone();
        let id = self.remote.id.clone();
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                if let Err(err) = provider.close_session(&id).await {
                    debug!(error = %err, "background browser session close failed");
                }
            });
        }
    }
}

/// Build a [`GotoOptions`] preconfigured for a forgiving navigation: wait
/// until the DOM is loaded but don't insist on every network request to
/// settle (which can hang on long-poll endpoints).
pub fn default_goto_options() -> GotoOptions {
    GotoOptions::new()
        .timeout(NAVIGATION_TIMEOUT)
        .wait_until(WaitUntil::DomContentLoaded)
}

/// Build a PNG full-page screenshot configuration.
pub fn default_screenshot_options() -> ScreenshotOptions {
    ScreenshotOptions::builder()
        .screenshot_type(ScreenshotType::Png)
        .full_page(true)
        .build()
}
