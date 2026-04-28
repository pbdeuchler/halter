// pattern: Imperative Shell
//
// BrowserBase (https://browserbase.com) cloud browser adapter.
//
// Reads credentials from environment at construction time so the policy /
// runtime layer can decide *once* whether to register the browser tool. The
// actual session create/close calls happen later when the agent navigates,
// and use the in-memory copy of the credentials.

use async_trait::async_trait;
use reqwest::Client;
use serde_json::{Value, json};
use tracing::{info, warn};

use super::provider::{BrowserProvider, RemoteSession};

const DEFAULT_BASE_URL: &str = "https://api.browserbase.com";

#[derive(Debug, Clone)]
pub struct BrowserbaseConfig {
    pub api_key: String,
    pub project_id: String,
    pub base_url: String,
    pub proxies: bool,
    pub keep_alive: bool,
    pub session_timeout_ms: Option<u64>,
}

impl BrowserbaseConfig {
    /// Build a config from `BROWSERBASE_*` environment variables, returning
    /// `None` when the required credentials are absent. Optional knobs default
    /// to the values that worked best in the upstream Python implementation.
    pub fn from_env() -> Option<Self> {
        let api_key = read_required_env("BROWSERBASE_API_KEY")?;
        let project_id = read_required_env("BROWSERBASE_PROJECT_ID")?;
        let base_url = std::env::var("BROWSERBASE_BASE_URL")
            .ok()
            .and_then(|raw| {
                let trimmed = raw.trim().trim_end_matches('/').to_owned();
                if trimmed.is_empty() {
                    None
                } else {
                    Some(trimmed)
                }
            })
            .unwrap_or_else(|| DEFAULT_BASE_URL.to_owned());
        Some(Self {
            api_key,
            project_id,
            base_url,
            proxies: read_bool_env("BROWSERBASE_PROXIES", true),
            keep_alive: read_bool_env("BROWSERBASE_KEEP_ALIVE", true),
            session_timeout_ms: std::env::var("BROWSERBASE_SESSION_TIMEOUT")
                .ok()
                .and_then(|raw| raw.trim().parse::<u64>().ok())
                .filter(|&value| value > 0),
        })
    }
}

fn read_required_env(name: &str) -> Option<String> {
    let raw = std::env::var(name).ok()?;
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

fn read_bool_env(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(value) => match value.trim().to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "on" => true,
            "false" | "0" | "no" | "off" => false,
            _ => default,
        },
        Err(_) => default,
    }
}

#[derive(Debug)]
pub struct BrowserbaseProvider {
    config: BrowserbaseConfig,
    client: Client,
}

impl BrowserbaseProvider {
    pub fn new(config: BrowserbaseConfig) -> anyhow::Result<Self> {
        let client = Client::builder()
            .user_agent(concat!("halter-tools/", env!("CARGO_PKG_VERSION")))
            .build()
            .map_err(|err| anyhow::anyhow!("failed to build browserbase http client: {err}"))?;
        Ok(Self { config, client })
    }
}

#[async_trait]
impl BrowserProvider for BrowserbaseProvider {
    fn name(&self) -> &'static str {
        "browserbase"
    }

    async fn create_session(&self) -> anyhow::Result<RemoteSession> {
        // Build the body. Keep this aligned with the Python adapter's
        // semantics: keep_alive + proxies are best-effort; the API rejects
        // them with 402 when the plan doesn't allow them, and we retry
        // without each one rather than failing the whole call.
        let mut body = json!({ "projectId": self.config.project_id });
        if self.config.keep_alive {
            body["keepAlive"] = json!(true);
        }
        if let Some(timeout) = self.config.session_timeout_ms {
            body["timeout"] = json!(timeout);
        }
        if self.config.proxies {
            body["proxies"] = json!(true);
        }

        let mut features = Vec::new();
        if self.config.keep_alive {
            features.push("keep_alive");
        }
        if self.config.proxies {
            features.push("proxies");
        }
        if self.config.session_timeout_ms.is_some() {
            features.push("custom_timeout");
        }

        let session = self
            .create_session_with_fallback(&mut body, &mut features)
            .await?;

        let id = session
            .get("id")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                anyhow::anyhow!("failed to create browserbase session: response missing 'id' field")
            })?
            .to_owned();
        let cdp_url = session
            .get("connectUrl")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "failed to create browserbase session: response missing 'connectUrl' field"
                )
            })?
            .to_owned();

        info!(
            session_id = %id,
            features = ?features,
            "created browserbase session"
        );
        Ok(RemoteSession {
            id,
            cdp_url,
            features,
        })
    }

    async fn close_session(&self, session_id: &str) -> anyhow::Result<()> {
        let url = format!("{}/v1/sessions/{}", self.config.base_url, session_id);
        let response = self
            .client
            .post(&url)
            .header("X-BB-API-Key", &self.config.api_key)
            .header("Content-Type", "application/json")
            .json(&json!({
                "projectId": self.config.project_id,
                "status": "REQUEST_RELEASE",
            }))
            .send()
            .await
            .map_err(|err| {
                anyhow::anyhow!("failed to close browserbase session {session_id}: {err}")
            })?;
        let status = response.status();
        if status.is_success() || status.as_u16() == 404 {
            return Ok(());
        }
        let detail = response.text().await.unwrap_or_default();
        anyhow::bail!("failed to close browserbase session {session_id}: HTTP {status} {detail}")
    }
}

impl BrowserbaseProvider {
    async fn create_session_with_fallback(
        &self,
        body: &mut Value,
        features: &mut Vec<&'static str>,
    ) -> anyhow::Result<Value> {
        let response = self.post_create(body).await?;
        if !is_payment_required(&response) {
            return parse_session_response(response).await;
        }
        // Retry without keepAlive — paid feature on most plans.
        if body.get("keepAlive").is_some() {
            warn!("browserbase rejected keepAlive (402); retrying without");
            body.as_object_mut().unwrap().remove("keepAlive");
            features.retain(|name| *name != "keep_alive");
            let response = self.post_create(body).await?;
            if !is_payment_required(&response) {
                return parse_session_response(response).await;
            }
        }
        // Retry without proxies — also a paid feature on starter plans.
        if body.get("proxies").is_some() {
            warn!("browserbase rejected proxies (402); retrying without");
            body.as_object_mut().unwrap().remove("proxies");
            features.retain(|name| *name != "proxies");
            let response = self.post_create(body).await?;
            return parse_session_response(response).await;
        }
        parse_session_response(response).await
    }

    async fn post_create(&self, body: &Value) -> anyhow::Result<reqwest::Response> {
        let url = format!("{}/v1/sessions", self.config.base_url);
        self.client
            .post(&url)
            .header("X-BB-API-Key", &self.config.api_key)
            .header("Content-Type", "application/json")
            .json(body)
            .send()
            .await
            .map_err(|err| anyhow::anyhow!("failed to create browserbase session: {err}"))
    }
}

fn is_payment_required(response: &reqwest::Response) -> bool {
    response.status().as_u16() == 402
}

async fn parse_session_response(response: reqwest::Response) -> anyhow::Result<Value> {
    let status = response.status();
    let bytes = response
        .bytes()
        .await
        .map_err(|err| anyhow::anyhow!("failed to read browserbase response: {err}"))?;
    if !status.is_success() {
        let detail = String::from_utf8_lossy(&bytes);
        anyhow::bail!("failed to create browserbase session: HTTP {status} {detail}");
    }
    serde_json::from_slice(&bytes)
        .map_err(|err| anyhow::anyhow!("failed to decode browserbase response: {err}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_bool_env_understands_common_values() {
        // unsafe block required by std::env::set_var on edition 2024 — single-threaded test.
        unsafe {
            std::env::set_var("HALTER_TEST_BOOL", "true");
        }
        assert!(read_bool_env("HALTER_TEST_BOOL", false));
        unsafe {
            std::env::set_var("HALTER_TEST_BOOL", "0");
        }
        assert!(!read_bool_env("HALTER_TEST_BOOL", true));
        unsafe {
            std::env::set_var("HALTER_TEST_BOOL", "garbage");
        }
        assert!(read_bool_env("HALTER_TEST_BOOL", true));
        assert!(!read_bool_env("HALTER_TEST_BOOL", false));
        unsafe {
            std::env::remove_var("HALTER_TEST_BOOL");
        }
    }

    #[test]
    fn from_env_requires_both_credentials() {
        unsafe {
            std::env::remove_var("BROWSERBASE_API_KEY");
            std::env::remove_var("BROWSERBASE_PROJECT_ID");
        }
        assert!(BrowserbaseConfig::from_env().is_none());

        unsafe {
            std::env::set_var("BROWSERBASE_API_KEY", "api");
        }
        assert!(BrowserbaseConfig::from_env().is_none());

        unsafe {
            std::env::set_var("BROWSERBASE_PROJECT_ID", "proj");
        }
        let config = BrowserbaseConfig::from_env().expect("config");
        assert_eq!(config.api_key, "api");
        assert_eq!(config.project_id, "proj");
        assert_eq!(config.base_url, DEFAULT_BASE_URL);

        unsafe {
            std::env::remove_var("BROWSERBASE_API_KEY");
            std::env::remove_var("BROWSERBASE_PROJECT_ID");
        }
    }
}
