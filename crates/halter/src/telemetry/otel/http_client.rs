//! An [`opentelemetry_http::HttpClient`] backed by the workspace's existing
//! `reqwest` 0.12 client.
//!
//! `opentelemetry-http`'s own `reqwest`/`reqwest-rustls` features pull in
//! `reqwest ^0.13`, a different major version than the one already pinned
//! across the workspace (`reqwest 0.12.24`). Implementing the trait directly
//! against `reqwest::Client` keeps exactly one `reqwest` major version in the
//! dependency graph when the `otel` feature is enabled, and avoids the
//! `grpc-tonic` transport entirely.
//!
//! # Runtime
//!
//! `opentelemetry_sdk`'s default `BatchSpanProcessor` and `PeriodicReader`
//! export from their own OS threads by driving `send_bytes` with a plain
//! `block_on`, where no Tokio reactor is available. `reqwest`'s async client
//! needs one: called from those threads it panics ("there is no reactor
//! running"), killing the export threads. The client therefore owns a
//! dedicated single-worker Tokio runtime and runs every request on it.
//! Awaiting the returned `JoinHandle` works from any executor, so
//! `send_bytes` is usable from the SDK threads and from inside an
//! application runtime alike, and export keeps working while the
//! application's own runtime is shutting down.

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

/// Adapts a `reqwest::Client` to [`HttpClient`] for OTLP HTTP export, running
/// each request on a runtime it owns (see the module docs).
#[derive(Debug)]
pub(crate) struct ReqwestOtlpClient {
    client: reqwest::Client,
    runtime: ExportRuntime,
}

impl ReqwestOtlpClient {
    pub(crate) fn new(client: reqwest::Client) -> std::io::Result<Self> {
        Ok(Self {
            client,
            runtime: ExportRuntime::new()?,
        })
    }
}

#[async_trait]
impl HttpClient for ReqwestOtlpClient {
    async fn send_bytes(&self, request: Request<Bytes>) -> Result<Response<Bytes>, HttpError> {
        let request: reqwest::Request = request.try_into()?;
        let client = self.client.clone();
        self.runtime
            .handle()
            .spawn(execute(client, request))
            .await
            .map_err(|error| -> HttpError { Box::new(error) })?
    }
}

/// Owns the export runtime and shuts it down without blocking, so dropping
/// the client (which can happen inside an application's async context when
/// the providers are dropped there) never panics.
#[derive(Debug)]
struct ExportRuntime(Option<tokio::runtime::Runtime>);

impl ExportRuntime {
    fn new() -> std::io::Result<Self> {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .thread_name("halter-otlp-http")
            .enable_all()
            .build()
            .map(|runtime| Self(Some(runtime)))
    }

    fn handle(&self) -> &tokio::runtime::Handle {
        self.0
            .as_ref()
            .expect("export runtime is present until drop")
            .handle()
    }
}

impl Drop for ExportRuntime {
    fn drop(&mut self) {
        if let Some(runtime) = self.0.take() {
            runtime.shutdown_background();
        }
    }
}

async fn execute(
    client: reqwest::Client,
    request: reqwest::Request,
) -> Result<Response<Bytes>, HttpError> {
    let mut response = client.execute(request).await?;

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
        )
        .expect("export runtime");
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

    /// The SDK's batch processor and periodic reader call `send_bytes` from
    /// plain OS threads via `block_on`, with no Tokio reactor. This must not
    /// panic and must actually complete the request.
    #[test]
    fn send_bytes_works_from_a_thread_without_a_tokio_runtime() {
        use std::io::{Read, Write};

        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind");
        let url = format!("http://{}/v1/traces", listener.local_addr().expect("addr"));
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().expect("accept");
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf).expect("read request");
            stream
                .write_all(b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: close\r\n\r\nok")
                .expect("write response");
        });

        let client = ReqwestOtlpClient::new(reqwest::Client::new()).expect("export runtime");
        let request = Request::post(url)
            .header("content-type", "application/x-protobuf")
            .body(Bytes::from_static(b"payload"))
            .expect("request");

        let response =
            std::thread::spawn(move || futures::executor::block_on(client.send_bytes(request)))
                .join()
                .expect("send_bytes must not panic outside a Tokio runtime")
                .expect("request should succeed");
        server.join().expect("server thread");

        assert_eq!(response.status(), 200);
        assert_eq!(response.body().as_ref(), b"ok");
    }

    /// Exporters (and so this client) can be dropped inside an application's
    /// async context; dropping a Tokio runtime there normally panics.
    #[tokio::test]
    async fn dropping_the_client_inside_an_async_context_does_not_panic() {
        let client = ReqwestOtlpClient::new(reqwest::Client::new()).expect("export runtime");
        drop(client);
    }
}
