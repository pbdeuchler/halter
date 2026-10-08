//! An [`opentelemetry_http::HttpClient`] backed by the workspace's existing
//! `reqwest` 0.12 client.
//!
//! `opentelemetry-http`'s own `reqwest`/`reqwest-rustls` features pull in
//! `reqwest ^0.13`, a different major version than the one already pinned
//! across the workspace (`reqwest 0.12.24`). Implementing the trait directly
//! against `reqwest::Client` keeps exactly one `reqwest` major version in the
//! dependency graph when the `otel` feature is enabled, and avoids the
//! `grpc-tonic` transport entirely.

use async_trait::async_trait;
use opentelemetry_http::{Bytes, HttpClient, HttpError, Request, Response};

/// Response bodies larger than this are rejected rather than buffered fully
/// into memory, matching the limit `opentelemetry-http`'s own reqwest
/// adapters use.
const MAX_RESPONSE_BODY_BYTES: usize = 4 * 1024 * 1024;

/// Error returned when an HTTP response body exceeds [`MAX_RESPONSE_BODY_BYTES`].
#[derive(Debug, Default)]
struct ResponseBodyTooLarge;

impl std::fmt::Display for ResponseBodyTooLarge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "response body exceeded maximum allowed 4 MiB limit")
    }
}

impl std::error::Error for ResponseBodyTooLarge {}

/// Adapts a `reqwest::Client` to [`HttpClient`] for OTLP HTTP export.
#[derive(Debug, Clone)]
pub(crate) struct ReqwestOtlpClient(reqwest::Client);

impl ReqwestOtlpClient {
    pub(crate) fn new(client: reqwest::Client) -> Self {
        Self(client)
    }
}

impl Default for ReqwestOtlpClient {
    fn default() -> Self {
        Self::new(reqwest::Client::new())
    }
}

#[async_trait]
impl HttpClient for ReqwestOtlpClient {
    async fn send_bytes(&self, request: Request<Bytes>) -> Result<Response<Bytes>, HttpError> {
        let request: reqwest::Request = request.try_into()?;
        let mut response = self.0.execute(request).await?;

        let capacity = response
            .content_length()
            .unwrap_or(0)
            .min(MAX_RESPONSE_BODY_BYTES as u64) as usize;
        let status = response.status();
        let headers = std::mem::take(response.headers_mut());

        let mut body = bytes::BytesMut::with_capacity(capacity);
        while let Some(chunk) = response.chunk().await? {
            if body.len() + chunk.len() > MAX_RESPONSE_BODY_BYTES {
                return Err(Box::new(ResponseBodyTooLarge));
            }
            body.extend_from_slice(&chunk);
        }

        let mut http_response = Response::builder().status(status).body(body.freeze())?;
        *http_response.headers_mut() = headers;
        Ok(http_response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exporter wiring (`OtelConfig::build`) only exercises this client
    /// against an in-memory exporter in the rest of the test suite; this test
    /// is the one place that actually drives bytes through
    /// `ReqwestOtlpClient::send_bytes` against a real (if unroutable)
    /// endpoint, catching gross wiring bugs like a broken `TryFrom`
    /// conversion even without a live collector.
    #[tokio::test]
    async fn send_bytes_against_unroutable_endpoint_fails_without_hanging() {
        let client = ReqwestOtlpClient::new(
            reqwest::Client::builder()
                .timeout(std::time::Duration::from_millis(500))
                .build()
                .expect("client"),
        );
        let request = Request::post("http://127.0.0.1:1")
            .header("content-type", "application/x-protobuf")
            .body(Bytes::from_static(b"not real protobuf"))
            .expect("request");

        let result = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            client.send_bytes(request),
        )
        .await
        .expect("send_bytes should not hang");

        assert!(result.is_err(), "connecting to a closed port must fail");
    }
}
