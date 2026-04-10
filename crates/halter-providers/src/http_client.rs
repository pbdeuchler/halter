// pattern: Imperative Shell

use anyhow::Context;
use reqwest::Client;
use serde_json::Value;
use tokio::select;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

#[derive(Debug, Clone)]
pub(crate) struct JsonRequest {
    pub provider_label: &'static str,
    pub url: String,
    pub headers: Vec<(String, String)>,
    pub body: Value,
}

#[derive(Debug, Clone)]
pub(crate) struct JsonHttpClient {
    client: Client,
}

impl Default for JsonHttpClient {
    fn default() -> Self {
        let client = Client::builder()
            .user_agent(concat!("halter/", env!("CARGO_PKG_VERSION")))
            .build()
            .expect("provider client must build");
        Self { client }
    }
}

impl JsonHttpClient {
    pub(crate) async fn post_json(
        &self,
        request: JsonRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<Value> {
        let JsonRequest {
            provider_label,
            url,
            headers,
            body,
        } = request;
        let body_bytes = body.to_string().len();
        debug!(
            provider = provider_label,
            url = %url,
            body_bytes,
            "sending json request"
        );
        let mut builder = self.client.post(&url).json(&body);
        for (name, value) in headers {
            builder = builder.header(name, value);
        }

        let response = select! {
            _ = cancel.cancelled() => anyhow::bail!("failed to execute provider request: request cancelled"),
            result = builder.send() => result,
        }
        .with_context(|| format!("failed to execute {} request", provider_label))?;
        let status = response.status();
        let body = select! {
            _ = cancel.cancelled() => anyhow::bail!("failed to execute provider request: request cancelled"),
            result = response.text() => result,
        }
        .with_context(|| format!("failed to read {} response body", provider_label))?;
        debug!(
            provider = provider_label,
            url = %url,
            status = %status,
            response_bytes = body.len(),
            "received json response"
        );

        if !status.is_success() {
            let detail = response_error_message(&body);
            warn!(
                provider = provider_label,
                url = %url,
                status = %status,
                detail = %detail,
                "provider request failed"
            );
            if detail.is_empty() {
                anyhow::bail!(
                    "failed to execute provider request: {} returned {}",
                    provider_label,
                    status
                );
            }
            anyhow::bail!(
                "failed to execute provider request: {} returned {}: {}",
                provider_label,
                status,
                detail
            );
        }

        serde_json::from_str(&body)
            .with_context(|| format!("failed to decode {} response json", provider_label))
    }
}

fn response_error_message(body: &str) -> String {
    let parsed = serde_json::from_str::<Value>(body);
    if let Ok(value) = parsed
        && let Some(message) = value
            .pointer("/error/message")
            .and_then(Value::as_str)
            .or_else(|| value.get("message").and_then(Value::as_str))
    {
        return message.to_owned();
    }

    body.trim().to_owned()
}
