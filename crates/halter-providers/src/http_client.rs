// pattern: Imperative Shell

use std::time::Duration;

use anyhow::Context;
use eventsource_stream::Eventsource;
use futures::{StreamExt, stream::BoxStream};
use halter_protocol::{ProviderError, ProviderErrorKind};
use reqwest::Client;
use reqwest::header::{CONTENT_TYPE, HeaderMap, HeaderName, HeaderValue, RETRY_AFTER};
use serde_json::Value;
use tokio::select;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use crate::openai_error::classify_error_payload;
use crate::resilience::ProviderTimeouts;

/// Build the (unary, streaming) reqwest client pair shared by every provider
/// transport. The unary client bounds the whole request with
/// `timeouts.request`; the streaming client instead bounds only the idle gap
/// between body reads so long-lived SSE responses are not killed mid-stream.
pub(crate) fn provider_http_clients(
    timeouts: ProviderTimeouts,
) -> anyhow::Result<(Client, Client)> {
    let unary_client = Client::builder()
        .user_agent(concat!("halter/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(timeouts.connect)
        .timeout(timeouts.request)
        .build()
        .context("failed to build provider http client")?;
    let streaming_client = Client::builder()
        .user_agent(concat!("halter/", env!("CARGO_PKG_VERSION")))
        .connect_timeout(timeouts.connect)
        .read_timeout(timeouts.stream_idle)
        .build()
        .context("failed to build provider streaming http client")?;
    Ok((unary_client, streaming_client))
}

/// Join a provider base URL and request path, trimming a duplicate slash.
pub(crate) fn join_url(base_url: &str, path: &str) -> String {
    format!("{}{}", base_url.trim_end_matches('/'), path)
}

/// Recover the typed [`ProviderError`] attached by this module's failure
/// paths, falling back to a transient wrapper for untyped errors.
pub(crate) fn provider_error_from_anyhow(error: anyhow::Error) -> ProviderError {
    match error.downcast::<ProviderError>() {
        Ok(provider_error) => provider_error,
        Err(error) => ProviderError::with_kind(format!("{error:#}"), ProviderErrorKind::Transient),
    }
}

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
    unary_client: Client,
    streaming_client: Client,
    request_timeout: Duration,
}

impl JsonHttpClient {
    pub(crate) fn try_new_with_timeouts(timeouts: ProviderTimeouts) -> anyhow::Result<Self> {
        let (unary_client, streaming_client) = provider_http_clients(timeouts)?;
        Ok(Self {
            unary_client,
            streaming_client,
            request_timeout: timeouts.request,
        })
    }
}

impl JsonHttpClient {
    /// Errors from every failure path carry a typed [`ProviderError`]
    /// (recoverable via [`provider_error_from_anyhow`]): encode and header
    /// failures are `Fatal`, cancellation is `Cancelled`, network faults and
    /// timeouts are `Transient`, and non-success statuses are classified by
    /// status and payload tags with any `Retry-After` header as backoff hint.
    pub(crate) async fn post_json_event_stream(
        &self,
        request: JsonRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<BoxStream<'static, anyhow::Result<Value>>> {
        let JsonRequest {
            provider_label,
            url,
            headers,
            body,
        } = request;
        let body_bytes_vec = encode_request_body(provider_label, &body)?;
        debug!(
            provider = provider_label,
            url = %url,
            body_bytes = body_bytes_vec.len(),
            "sending streaming json request"
        );
        let header_map = request_headers(provider_label, headers)?;
        let builder = self
            .streaming_client
            .post(&url)
            .headers(header_map)
            .body(body_bytes_vec);

        let response = select! {
            _ = cancel.cancelled() => return Err(cancelled_error()),
            result = tokio::time::timeout(self.request_timeout, builder.send()) => match result {
                Ok(result) => result.map_err(|error| transient_error(format!(
                    "failed to execute {provider_label} request: {error}"
                )))?,
                Err(_) => {
                    return Err(transient_error(format!(
                        "failed to execute provider request: {} request timed out after {}s",
                        provider_label,
                        self.request_timeout.as_secs()
                    )));
                }
            },
        };
        let status = response.status();
        if !status.is_success() {
            let retry_after = retry_after_header(response.headers());
            let body = select! {
                _ = cancel.cancelled() => return Err(cancelled_error()),
                result = response.text() => result.map_err(|error| transient_error(format!(
                    "failed to read {provider_label} response body: {error}"
                )))?,
            };
            return Err(http_status_error(
                provider_label,
                &url,
                status,
                retry_after,
                &body,
            ));
        }

        let stream = response
            .bytes_stream()
            .eventsource()
            .take_until(cancel.cancelled_owned())
            .filter_map(move |event| async move {
                match event {
                    Ok(event) if event.data == "[DONE]" => None,
                    Ok(event) => {
                        Some(serde_json::from_str::<Value>(&event.data).with_context(|| {
                            format!("failed to decode {} stream event json", provider_label)
                        }))
                    }
                    Err(error) => Some(Err(anyhow::anyhow!(
                        "failed to read {} stream event: {error}",
                        provider_label
                    ))),
                }
            })
            .boxed();

        Ok(stream)
    }

    /// Posts a JSON body and buffers the entire response into memory as a
    /// `String` before decoding. Suitable for small unary endpoints
    /// (Anthropic messages, OpenAI non-streaming responses) where the full
    /// payload is bounded by the provider's per-request output cap.
    ///
    /// **Do not use for streaming endpoints** — it fully consumes the
    /// response before returning, defeating SSE/chunked transport. Use
    /// `ResponsesTransport` (or an Anthropic-equivalent streaming client)
    /// for token-by-token delivery. (finding M26)
    ///
    /// Errors carry a typed [`ProviderError`] with the same classification
    /// rules as [`JsonHttpClient::post_json_event_stream`].
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
        let body_bytes_vec = encode_request_body(provider_label, &body)?;
        debug!(
            provider = provider_label,
            url = %url,
            body_bytes = body_bytes_vec.len(),
            "sending json request"
        );
        let header_map = request_headers(provider_label, headers)?;
        let builder = self
            .unary_client
            .post(&url)
            .headers(header_map)
            .body(body_bytes_vec);

        let response = select! {
            _ = cancel.cancelled() => return Err(cancelled_error()),
            result = builder.send() => result.map_err(|error| transient_error(format!(
                "failed to execute {provider_label} request: {error}"
            )))?,
        };
        let status = response.status();
        let retry_after = retry_after_header(response.headers());
        let body = select! {
            _ = cancel.cancelled() => return Err(cancelled_error()),
            result = response.text() => result.map_err(|error| transient_error(format!(
                "failed to read {provider_label} response body: {error}"
            )))?,
        };
        debug!(
            provider = provider_label,
            url = %url,
            status = %status,
            response_bytes = body.len(),
            "received json response"
        );

        if !status.is_success() {
            return Err(http_status_error(
                provider_label,
                &url,
                status,
                retry_after,
                &body,
            ));
        }

        serde_json::from_str(&body).map_err(|error| {
            fatal_error(format!(
                "failed to decode {provider_label} response json: {error}"
            ))
        })
    }
}

fn encode_request_body(provider_label: &str, body: &Value) -> anyhow::Result<Vec<u8>> {
    serde_json::to_vec(body).map_err(|error| {
        fatal_error(format!(
            "failed to encode {provider_label} request body: {error}"
        ))
    })
}

fn request_headers(
    provider_label: &'static str,
    headers: Vec<(String, String)>,
) -> anyhow::Result<HeaderMap> {
    let mut header_map = HeaderMap::new();
    header_map.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    for (name, value) in headers {
        let header_name = HeaderName::from_bytes(name.as_bytes()).map_err(|error| {
            fatal_error(format!(
                "failed to encode http header '{name}' for {provider_label} request: {error}"
            ))
        })?;
        let mut header_value = HeaderValue::from_str(&value).map_err(|error| {
            fatal_error(format!(
                "failed to encode http header value for '{name}' in {provider_label} request: {error}"
            ))
        })?;
        if is_sensitive_header(&header_name) {
            header_value.set_sensitive(true);
        }
        header_map.insert(header_name, header_value);
    }
    Ok(header_map)
}

/// Credential-bearing headers must never appear in debug output; mark them
/// sensitive so reqwest/http redact them (matches the `Authorization`
/// treatment in `responses_transport`).
fn is_sensitive_header(name: &HeaderName) -> bool {
    name == reqwest::header::AUTHORIZATION || name.as_str() == "x-api-key"
}

fn http_status_error(
    provider_label: &'static str,
    url: &str,
    status: reqwest::StatusCode,
    retry_after: Option<Duration>,
    body: &str,
) -> anyhow::Error {
    let (detail, error_type) = response_error_detail(body);
    warn!(
        provider = provider_label,
        url = %url,
        status = %status,
        detail = %detail,
        "provider request failed"
    );
    let message = if detail.is_empty() {
        format!("failed to execute provider request: {provider_label} returned {status}")
    } else {
        format!("failed to execute provider request: {provider_label} returned {status}: {detail}")
    };
    let retryability =
        classify_error_payload(Some(status.as_u16()), error_type.as_deref(), &detail);
    anyhow::Error::new(
        ProviderError::with_kind(message, retryability.kind)
            .with_backoff_hint(retry_after.or(retryability.backoff_hint)),
    )
}

fn cancelled_error() -> anyhow::Error {
    anyhow::Error::new(ProviderError::cancelled())
}

fn transient_error(message: String) -> anyhow::Error {
    anyhow::Error::new(ProviderError::with_kind(
        message,
        ProviderErrorKind::Transient,
    ))
}

fn fatal_error(message: String) -> anyhow::Error {
    anyhow::Error::new(ProviderError::with_kind(message, ProviderErrorKind::Fatal))
}

/// Parse a `Retry-After` header expressed in whole seconds. HTTP-date values
/// are ignored — providers in this crate only emit the seconds form.
fn retry_after_header(headers: &HeaderMap) -> Option<Duration> {
    headers
        .get(RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .map(Duration::from_secs)
}

/// Extract the human-readable detail plus the payload's `error.type` tag
/// (used for retryability classification) from an error response body.
fn response_error_detail(body: &str) -> (String, Option<String>) {
    let Ok(value) = serde_json::from_str::<Value>(body) else {
        return (body.trim().to_owned(), None);
    };
    let error_type = value
        .pointer("/error/type")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let message = value
        .pointer("/error/message")
        .and_then(Value::as_str)
        .or_else(|| value.get("message").and_then(Value::as_str))
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| body.trim().to_owned());
    (message, error_type)
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;

    use super::*;
    use crate::test_http::read_http_request;

    fn client() -> JsonHttpClient {
        JsonHttpClient::try_new_with_timeouts(ProviderTimeouts::default()).expect("http client")
    }

    fn request(url: String, headers: Vec<(String, String)>) -> JsonRequest {
        JsonRequest {
            provider_label: "test",
            url,
            headers,
            body: json!({"ping": true}),
        }
    }

    async fn spawn_response_server(response: String) -> String {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind listener");
        let address = listener.local_addr().expect("local addr");
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept socket");
            read_http_request(&mut socket).await.expect("read request");
            socket
                .write_all(response.as_bytes())
                .await
                .expect("write response");
        });
        format!("http://{address}")
    }

    fn http_response(status_line: &str, extra_headers: &str, body: &str) -> String {
        format!(
            "HTTP/1.1 {status_line}\r\ncontent-type: application/json\r\ncontent-length: {}\r\n{extra_headers}connection: close\r\n\r\n{body}",
            body.len()
        )
    }

    #[test]
    fn join_url_trims_duplicate_slash() {
        let cases = [
            (
                "https://api.anthropic.com/",
                "/v1/messages",
                "https://api.anthropic.com/v1/messages",
            ),
            (
                "https://api.anthropic.com",
                "/v1/messages",
                "https://api.anthropic.com/v1/messages",
            ),
        ];
        for (base, path, want) in cases {
            assert_eq!(join_url(base, path), want);
        }
    }

    #[test]
    fn request_headers_apply_defaults_overrides_and_sensitivity() {
        let headers = request_headers(
            "test",
            vec![
                ("x-api-key".to_owned(), "secret".to_owned()),
                ("Authorization".to_owned(), "Bearer secret".to_owned()),
                ("X-Trace-Id".to_owned(), "trace-1".to_owned()),
            ],
        )
        .expect("headers");

        assert_eq!(
            headers
                .get(CONTENT_TYPE)
                .and_then(|value| value.to_str().ok()),
            Some("application/json")
        );
        assert!(headers.get("x-api-key").expect("x-api-key").is_sensitive());
        assert!(
            headers
                .get(reqwest::header::AUTHORIZATION)
                .expect("authorization")
                .is_sensitive()
        );
        assert!(!headers.get("x-trace-id").expect("trace").is_sensitive());
    }

    #[test]
    fn request_headers_reject_invalid_names_and_values() {
        let bad_name = request_headers("test", vec![("bad name".to_owned(), "v".to_owned())])
            .expect_err("invalid header name should fail");
        assert_eq!(
            provider_error_from_anyhow(bad_name).kind,
            ProviderErrorKind::Fatal
        );

        let bad_value = request_headers("test", vec![("x-ok".to_owned(), "bad\nvalue".to_owned())])
            .expect_err("invalid header value should fail");
        let bad_value = provider_error_from_anyhow(bad_value);
        assert_eq!(bad_value.kind, ProviderErrorKind::Fatal);
        assert!(bad_value.message.contains("x-ok"));
    }

    #[test]
    fn response_error_detail_extracts_message_and_type() {
        let cases: &[(&str, &str, Option<&str>)] = &[
            (
                r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
                "Overloaded",
                Some("overloaded_error"),
            ),
            (r#"{"message":"top level"}"#, "top level", None),
            ("not json at all", "not json at all", None),
            (r#"{"error":{}}"#, r#"{"error":{}}"#, None),
        ];
        for (body, want_detail, want_type) in cases {
            let (detail, error_type) = response_error_detail(body);
            assert_eq!(detail, *want_detail, "{body}");
            assert_eq!(error_type.as_deref(), *want_type, "{body}");
        }
    }

    #[test]
    fn retry_after_header_parses_seconds_only() {
        let mut headers = HeaderMap::new();
        assert_eq!(retry_after_header(&headers), None);
        headers.insert(RETRY_AFTER, HeaderValue::from_static("7"));
        assert_eq!(retry_after_header(&headers), Some(Duration::from_secs(7)));
        headers.insert(
            RETRY_AFTER,
            HeaderValue::from_static("Wed, 21 Oct 2026 07:28:00 GMT"),
        );
        assert_eq!(retry_after_header(&headers), None);
    }

    #[test]
    fn provider_error_from_anyhow_falls_back_to_transient() {
        let typed = provider_error_from_anyhow(anyhow::Error::new(ProviderError::with_kind(
            "typed",
            ProviderErrorKind::Fatal,
        )));
        assert_eq!(typed.kind, ProviderErrorKind::Fatal);

        let untyped = provider_error_from_anyhow(anyhow::anyhow!("plain failure"));
        assert_eq!(untyped.kind, ProviderErrorKind::Transient);
        assert!(untyped.message.contains("plain failure"));
    }

    #[tokio::test]
    async fn post_json_decodes_success_response() {
        let base_url = spawn_response_server(http_response("200 OK", "", r#"{"ok":true}"#)).await;
        let value = client()
            .post_json(request(base_url, Vec::new()), CancellationToken::new())
            .await
            .expect("json response");
        assert_eq!(value, json!({"ok": true}));
    }

    #[tokio::test]
    async fn post_json_classifies_rate_limit_with_retry_after_header() {
        let body = r#"{"type":"error","error":{"type":"rate_limit_error","message":"Number of requests exceeded"}}"#;
        let base_url = spawn_response_server(http_response(
            "429 Too Many Requests",
            "retry-after: 3\r\n",
            body,
        ))
        .await;
        let error = client()
            .post_json(request(base_url, Vec::new()), CancellationToken::new())
            .await
            .expect_err("429 should fail");
        let error = provider_error_from_anyhow(error);
        assert_eq!(error.kind, ProviderErrorKind::RateLimited);
        assert_eq!(error.backoff_hint, Some(Duration::from_secs(3)));
        assert!(error.message.contains("Number of requests exceeded"));
    }

    #[tokio::test]
    async fn post_json_classifies_client_error_fatal_with_raw_body_detail() {
        let base_url =
            spawn_response_server(http_response("400 Bad Request", "", "malformed {json")).await;
        let error = client()
            .post_json(request(base_url, Vec::new()), CancellationToken::new())
            .await
            .expect_err("400 should fail");
        let error = provider_error_from_anyhow(error);
        assert_eq!(error.kind, ProviderErrorKind::Fatal);
        assert!(error.message.contains("malformed {json"));
    }

    #[tokio::test]
    async fn post_json_treats_undecodable_success_body_as_fatal() {
        let base_url = spawn_response_server(http_response("200 OK", "", "not json")).await;
        let error = client()
            .post_json(request(base_url, Vec::new()), CancellationToken::new())
            .await
            .expect_err("undecodable body should fail");
        let error = provider_error_from_anyhow(error);
        assert_eq!(error.kind, ProviderErrorKind::Fatal);
        assert!(
            error
                .message
                .contains("failed to decode test response json")
        );
    }

    #[tokio::test]
    async fn post_json_honors_pre_cancelled_token() {
        let cancel = CancellationToken::new();
        cancel.cancel();
        let error = client()
            .post_json(request("http://127.0.0.1:1".to_owned(), Vec::new()), cancel)
            .await
            .expect_err("cancelled request should fail");
        assert!(provider_error_from_anyhow(error).is_cancelled());
    }

    #[tokio::test]
    async fn post_json_event_stream_yields_decoded_events() {
        let body = "data: {\"a\":1}\n\ndata: [DONE]\n\n";
        let response = format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        let base_url = spawn_response_server(response).await;
        let mut stream = client()
            .post_json_event_stream(request(base_url, Vec::new()), CancellationToken::new())
            .await
            .expect("event stream");
        let first = stream
            .next()
            .await
            .expect("first event")
            .expect("decoded event");
        assert_eq!(first, json!({"a": 1}));
        assert!(stream.next().await.is_none(), "[DONE] must end the stream");
    }

    #[tokio::test]
    async fn post_json_event_stream_classifies_overloaded_status() {
        let body = r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#;
        let base_url = spawn_response_server(http_response(
            "529 Site Overloaded",
            "retry-after: 2\r\n",
            body,
        ))
        .await;
        let error = client()
            .post_json_event_stream(request(base_url, Vec::new()), CancellationToken::new())
            .await
            .err()
            .expect("529 should fail");
        let error = provider_error_from_anyhow(error);
        assert_eq!(error.kind, ProviderErrorKind::Transient);
        assert_eq!(error.backoff_hint, Some(Duration::from_secs(2)));
    }

    #[tokio::test]
    async fn post_json_event_stream_times_out_request_setup() {
        // Server accepts the connection but never responds.
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind listener");
        let address = listener.local_addr().expect("local addr");
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept socket");
            let _ = read_http_request(&mut socket).await;
            std::future::pending::<()>().await;
        });

        let timeouts = ProviderTimeouts {
            connect: Duration::from_secs(1),
            request: Duration::from_millis(50),
            stream_idle: Duration::from_secs(1),
        };
        let client = JsonHttpClient::try_new_with_timeouts(timeouts).expect("http client");
        let error = client
            .post_json_event_stream(
                request(format!("http://{address}"), Vec::new()),
                CancellationToken::new(),
            )
            .await
            .err()
            .expect("setup should time out");
        let error = provider_error_from_anyhow(error);
        assert_eq!(error.kind, ProviderErrorKind::Transient);
        assert!(error.message.contains("timed out"));
    }

    #[tokio::test]
    async fn post_json_event_stream_honors_pre_cancelled_token() {
        let cancel = CancellationToken::new();
        cancel.cancel();
        let error = client()
            .post_json_event_stream(request("http://127.0.0.1:1".to_owned(), Vec::new()), cancel)
            .await
            .err()
            .expect("cancelled request should fail");
        assert!(provider_error_from_anyhow(error).is_cancelled());
    }
}
