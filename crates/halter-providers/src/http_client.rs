// pattern: Imperative Shell

use anyhow::Context;
use reqwest::Client;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue};
use serde_json::Value;
use tokio::select;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

#[derive(Debug, Clone)]
pub(crate) struct JsonRequest {
    pub provider_label: &'static str,
    pub url: String,
    /// Headers applied with insert semantics — later entries override
    /// earlier entries with the same (case-insensitive) name. A
    /// `Content-Type: application/json` default is set by `post_json` before
    /// these are applied, so a caller-supplied `Content-Type` header will
    /// replace it cleanly (no duplicate values).
    pub headers: Vec<(String, String)>,
    pub body: Value,
}

#[derive(Debug, Clone)]
pub(crate) struct JsonHttpClient {
    client: Client,
}

impl JsonHttpClient {
    pub(crate) fn try_new() -> anyhow::Result<Self> {
        let client = Client::builder()
            .user_agent(concat!("halter/", env!("CARGO_PKG_VERSION")))
            .build()
            .context("failed to build provider http client")?;
        Ok(Self { client })
    }
}

impl JsonHttpClient {
    /// Posts a JSON body and buffers the entire response into memory as a
    /// `String` before decoding. Suitable for small unary endpoints
    /// (Anthropic messages, OpenAI non-streaming responses) where the full
    /// payload is bounded by the provider's per-request output cap.
    ///
    /// **Do not use for streaming endpoints** — it fully consumes the
    /// response before returning, defeating SSE/chunked transport. Use
    /// `ResponsesTransport` (or an Anthropic-equivalent streaming client)
    /// for token-by-token delivery. (finding M26)
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
        let body_bytes_vec = serde_json::to_vec(&body)
            .with_context(|| format!("failed to encode {} request body", provider_label))?;
        let body_bytes = body_bytes_vec.len();
        debug!(
            provider = provider_label,
            url = %url,
            body_bytes,
            "sending json request"
        );
        let mut header_map = HeaderMap::new();
        header_map.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        for (name, value) in headers {
            let header_name = HeaderName::from_bytes(name.as_bytes()).with_context(|| {
                format!(
                    "failed to encode http header '{}' for {} request",
                    name, provider_label
                )
            })?;
            let header_value = HeaderValue::from_str(&value).with_context(|| {
                format!(
                    "failed to encode http header value for '{}' in {} request",
                    name, provider_label
                )
            })?;
            header_map.insert(header_name, header_value);
        }
        let builder = self
            .client
            .post(&url)
            .headers(header_map)
            .body(body_bytes_vec);

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
