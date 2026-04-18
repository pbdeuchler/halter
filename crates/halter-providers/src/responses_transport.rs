// pattern: Imperative Shell

use std::time::Duration;

use async_openai::{
    error::{ApiError, OpenAIError, StreamError},
    types::responses::{ResponseStream, ResponseStreamEvent},
};
use eventsource_stream::Eventsource;
use futures::StreamExt;
use reqwest::{Client as ReqwestClient, Response, StatusCode};
use serde_json::Value;
use thiserror::Error;
use tokio::select;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_util::sync::CancellationToken;
use tracing::info;

use crate::openai_error::{
    Retryability, SYNTHETIC_SERVER_ERROR_CODE, classify, openai_api_error_retry_after,
    parse_openai_http_error, parse_openai_stream_error,
};
use crate::openai_rate_limit::{OpenAiRateLimitPermit, OpenAiRateLimiter};
use crate::openai_rate_limit_policy::OpenAiReservation;

/// Result of a transport-layer call. The variant carries the retryability
/// decision so callers do not re-classify by inspecting message text.
#[derive(Debug, Error)]
pub(crate) enum TransportError {
    /// Client-initiated cancellation. Maps to `ProviderError::cancelled()`
    /// at the provider boundary; never retried.
    #[error("failed to execute provider request: request cancelled")]
    Cancelled,
    /// Upstream signaled a transient failure (rate limit, 5xx, network
    /// blip). Caller may retry; `backoff_hint` carries any server-supplied
    /// delay (e.g. `Please try again in 1.25s`).
    #[error("retryable provider failure: {source}")]
    Retryable {
        #[source]
        source: OpenAIError,
        backoff_hint: Option<Duration>,
    },
    /// Upstream signaled a permanent failure (4xx, malformed request,
    /// non-retryable decode error). Caller must propagate.
    #[error("fatal provider failure: {source}")]
    Fatal {
        #[source]
        source: OpenAIError,
    },
}

impl TransportError {
    pub(crate) fn from_openai(source: OpenAIError) -> Self {
        match classify(&source) {
            Retryability::Retryable { backoff_hint } => Self::Retryable {
                source,
                backoff_hint,
            },
            Retryability::Fatal => Self::Fatal { source },
        }
    }

    pub(crate) fn from_reqwest(error: reqwest::Error, label: &str) -> Self {
        let wrapped = OpenAIError::Reqwest(error);
        match classify(&wrapped) {
            Retryability::Retryable { backoff_hint } => Self::Retryable {
                source: OpenAIError::ApiError(ApiError {
                    message: format!("failed to execute {label} request: {wrapped}"),
                    r#type: None,
                    param: None,
                    code: Some(SYNTHETIC_SERVER_ERROR_CODE.to_owned()),
                }),
                backoff_hint,
            },
            Retryability::Fatal => Self::Fatal { source: wrapped },
        }
    }
}

const RESPONSES_PATH: &str = "/v1/responses";
const RESPONSES_COMPACT_PATH: &str = "/v1/responses/compact";

/// An event sent by the OpenAI Responses streaming API that the `async-openai`
/// SDK does not yet model (e.g. `keepalive` heartbeat pings).
#[derive(Debug, Clone)]
enum NonStandardStreamEvent {
    Keepalive { sequence_number: u64 },
}

impl NonStandardStreamEvent {
    fn parse(data: &Value) -> Option<Self> {
        match data.get("type")?.as_str()? {
            "keepalive" => Some(Self::Keepalive {
                sequence_number: data.get("sequence_number")?.as_u64()?,
            }),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResponsesRateLimitStrategy {
    OpenAiHeaders,
}

#[derive(Debug, Clone)]
pub(crate) struct ResponsesTransportRequest {
    pub provider_label: &'static str,
    pub model: String,
    pub reservation: OpenAiReservation,
    pub rate_limit_strategy: Option<ResponsesRateLimitStrategy>,
    pub tokens_per_minute: Option<u64>,
}

#[derive(Debug, Clone)]
pub(crate) struct ResponsesTransport {
    client: ReqwestClient,
    api_key: String,
    base_url: String,
    openai_rate_limiter: OpenAiRateLimiter,
}

impl ResponsesTransport {
    #[must_use]
    pub(crate) fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        let api_key = api_key.into();
        let base_url = base_url.into();
        let client = ReqwestClient::builder()
            .user_agent(concat!("halter/", env!("CARGO_PKG_VERSION")))
            .build()
            .expect("responses transport client must build");

        Self {
            client,
            openai_rate_limiter: OpenAiRateLimiter::new(&api_key, &base_url),
            api_key,
            base_url,
        }
    }

    pub(crate) async fn responses_stream(
        &self,
        request: Value,
        request_meta: ResponsesTransportRequest,
        cancel: CancellationToken,
    ) -> Result<ResponseStream, TransportError> {
        let response = self
            .send_json_request(RESPONSES_PATH, request, request_meta.clone(), cancel)
            .await?;
        Ok(stream_response(
            response,
            OpenAiStreamRateLimitObserver {
                limiter: self.openai_rate_limiter.clone(),
                model: request_meta.model,
            },
        ))
    }

    pub(crate) async fn responses_compact(
        &self,
        request: Value,
        request_meta: ResponsesTransportRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<Value> {
        let provider_label = request_meta.provider_label;
        let response = self
            .send_json_request(
                RESPONSES_COMPACT_PATH,
                request,
                request_meta,
                cancel.child_token(),
            )
            .await
            .map_err(transport_error_to_anyhow)?;
        select! {
            _ = cancel.cancelled() => anyhow::bail!("failed to execute provider request: request cancelled"),
            result = response.json::<Value>() => result,
        }
        .map_err(|error| anyhow::anyhow!(
            "failed to decode {} compaction response: {error}",
            provider_label
        ))
    }

    pub(crate) async fn responses_json(
        &self,
        request: Value,
        request_meta: ResponsesTransportRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<Value> {
        let provider_label = request_meta.provider_label;
        let response = self
            .send_json_request(RESPONSES_PATH, request, request_meta, cancel.child_token())
            .await
            .map_err(transport_error_to_anyhow)?;
        select! {
            _ = cancel.cancelled() => anyhow::bail!("failed to execute provider request: request cancelled"),
            result = response.json::<Value>() => result,
        }
        .map_err(|error| anyhow::anyhow!("failed to decode {} response: {error}", provider_label))
    }

    async fn send_json_request(
        &self,
        path: &str,
        request: Value,
        request_meta: ResponsesTransportRequest,
        cancel: CancellationToken,
    ) -> Result<Response, TransportError> {
        let mut permit = self
            .rate_limit_permit(&request_meta, cancel.child_token())
            .await
            .map_err(|error| TransportError::Fatal {
                source: OpenAIError::ApiError(ApiError {
                    message: format!("failed to acquire rate-limit permit: {error}"),
                    r#type: None,
                    param: None,
                    code: None,
                }),
            })?;
        let request_builder = self
            .client
            .post(provider_url(&self.base_url, path))
            .bearer_auth(&self.api_key)
            .json(&request);

        let response = select! {
            _ = cancel.cancelled() => return Err(TransportError::Cancelled),
            result = request_builder.send() => result,
        }
        .map_err(|error| TransportError::from_reqwest(error, request_meta.provider_label))?;
        let status = response.status();
        let headers = response.headers().clone();
        if let Some(permit) = permit.as_mut() {
            permit.update_from_headers(&headers, status);
        }

        if !status.is_success() {
            let body = select! {
                _ = cancel.cancelled() => return Err(TransportError::Cancelled),
                result = response.bytes() => result,
            }
            .map_err(|error| TransportError::from_reqwest(error, request_meta.provider_label))?;
            let error = decode_openai_error(status, &body);
            // Push the server-supplied backoff hint into the rate limiter
            // here (single classification source). The classifier extracts
            // the same hint downstream when callers retry.
            if let OpenAIError::ApiError(api_error) = &error
                && let Some(retry_after) = openai_api_error_retry_after(api_error)
            {
                self.openai_rate_limiter
                    .apply_retry_after(&request_meta.model, retry_after);
            }
            return Err(TransportError::from_openai(error));
        }

        Ok(response)
    }

    async fn rate_limit_permit(
        &self,
        request_meta: &ResponsesTransportRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<Option<OpenAiRateLimitPermit>> {
        match request_meta.rate_limit_strategy {
            Some(ResponsesRateLimitStrategy::OpenAiHeaders) => Ok(Some(
                self.openai_rate_limiter
                    .acquire(
                        &request_meta.model,
                        request_meta.reservation,
                        request_meta.tokens_per_minute,
                        cancel,
                    )
                    .await?,
            )),
            None => Ok(None),
        }
    }
}

fn transport_error_to_anyhow(error: TransportError) -> anyhow::Error {
    anyhow::anyhow!("failed to execute provider request: {error}")
}

fn provider_url(base_url: &str, path: &str) -> String {
    format!("{}{}", base_url.trim_end_matches('/'), path)
}

#[derive(Debug, Clone)]
struct OpenAiStreamRateLimitObserver {
    limiter: OpenAiRateLimiter,
    model: String,
}

impl OpenAiStreamRateLimitObserver {
    fn record_api_error(&self, error: &ApiError) {
        if let Some(retry_after) = openai_api_error_retry_after(error) {
            self.limiter.apply_retry_after(&self.model, retry_after);
        }
    }
}

fn stream_response(
    response: Response,
    rate_limits: OpenAiStreamRateLimitObserver,
) -> ResponseStream {
    let stream = response.bytes_stream().eventsource();
    let (tx, rx) = mpsc::unbounded_channel();

    tokio::spawn(async move {
        let mut stream = Box::pin(stream);
        while let Some(event) = stream.next().await {
            match event {
                Err(error) => {
                    if tx
                        .send(Err(OpenAIError::StreamError(Box::new(
                            StreamError::EventStream(error.to_string()),
                        ))))
                        .is_err()
                    {
                        break;
                    }
                }
                Ok(event) => {
                    if event.data == "[DONE]" {
                        break;
                    }

                    match decode_stream_event(&event.data, &rate_limits) {
                        Ok(Some(event)) => {
                            if tx.send(Ok(event)).is_err() {
                                break;
                            }
                        }
                        Ok(None) => {} // non-standard event already handled (e.g. keepalive)
                        Err(err) => {
                            if tx.send(Err(err)).is_err() {
                                break;
                            }
                        }
                    }
                }
            }
        }
    });

    Box::pin(UnboundedReceiverStream::new(rx))
}

fn decode_stream_event(
    data: &str,
    rate_limits: &OpenAiStreamRateLimitObserver,
) -> Result<Option<ResponseStreamEvent>, OpenAIError> {
    if let Some(api_error) = parse_openai_stream_error(data) {
        rate_limits.record_api_error(&api_error);
        return Err(OpenAIError::ApiError(api_error));
    }

    if let Ok(raw) = serde_json::from_str::<Value>(data)
        && let Some(event) = NonStandardStreamEvent::parse(&raw)
    {
        match event {
            NonStandardStreamEvent::Keepalive { sequence_number } => {
                info!(sequence_number, "received keepalive from responses stream");
            }
        }
        return Ok(None);
    }

    serde_json::from_str::<ResponseStreamEvent>(data)
        .map(Some)
        .map_err(|error| OpenAIError::JSONDeserialize(error, data.to_owned()))
}

fn decode_openai_error(status: StatusCode, body: &[u8]) -> OpenAIError {
    // 5xx without a parseable JSON body → stamp the synthetic code so the
    // shared `classify` routes it as Retryable without needing the original
    // HTTP status. This replaces ad-hoc substring tests on the message.
    if status.is_server_error() {
        let message = String::from_utf8_lossy(body).trim().to_owned();
        return OpenAIError::ApiError(ApiError {
            message,
            r#type: None,
            param: None,
            code: Some(SYNTHETIC_SERVER_ERROR_CODE.to_owned()),
        });
    }

    parse_openai_http_error(body)
        .map(OpenAIError::ApiError)
        .unwrap_or_else(|| {
            OpenAIError::ApiError(ApiError {
                message: String::from_utf8_lossy(body).trim().to_owned(),
                r#type: None,
                param: None,
                code: None,
            })
        })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;
    use crate::openai_rate_limit_policy::estimate_openai_request_cost;

    #[tokio::test]
    async fn responses_stream_honors_openai_header_waits() {
        let request_times = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let base_url = spawn_sse_server(request_times.clone(), 2).await;
        let transport = ResponsesTransport::new("test-key", base_url);
        let request = json!({
            "model": "gpt-5",
            "input": [{ "role": "user", "content": [{ "type": "input_text", "text": "hello" }] }],
            "stream": true,
        });
        let request_meta = ResponsesTransportRequest {
            provider_label: "openai",
            model: "gpt-5".to_owned(),
            reservation: estimate_openai_request_cost(request.to_string().len(), Some(32)),
            rate_limit_strategy: Some(ResponsesRateLimitStrategy::OpenAiHeaders),
            tokens_per_minute: None,
        };

        let mut first = transport
            .responses_stream(
                request.clone(),
                request_meta.clone(),
                CancellationToken::new(),
            )
            .await
            .expect("first stream");
        first.next().await.expect("first event").expect("ok event");

        let mut second = transport
            .responses_stream(request, request_meta, CancellationToken::new())
            .await
            .expect("second stream");
        second
            .next()
            .await
            .expect("second event")
            .expect("ok event");

        let request_times = request_times.lock().await.clone();
        assert_eq!(request_times.len(), 2);
        let gap = request_times[1].saturating_duration_since(request_times[0]);
        assert!(
            gap >= Duration::from_millis(100),
            "expected limiter to delay second request, saw gap {gap:?}"
        );
    }

    async fn spawn_sse_server(
        request_times: Arc<tokio::sync::Mutex<Vec<Instant>>>,
        expected_requests: usize,
    ) -> String {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind listener");
        let address = listener.local_addr().expect("local addr");

        tokio::spawn(async move {
            for _ in 0..expected_requests {
                let (mut socket, _) = listener.accept().await.expect("accept socket");
                read_http_request(&mut socket).await.expect("read request");
                request_times.lock().await.push(Instant::now());

                let event = json!({
                    "type": "response.created",
                    "sequence_number": 0,
                    "response": {
                        "id": "resp_test",
                        "created_at": 0,
                        "model": "gpt-5",
                        "object": "response",
                        "output": [],
                        "status": "in_progress"
                    }
                });
                let body = format!("data: {event}\n\ndata: [DONE]\n\n");
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nx-ratelimit-limit-requests: 1\r\nx-ratelimit-remaining-requests: 0\r\nx-ratelimit-reset-requests: 120ms\r\nx-ratelimit-limit-tokens: 1000\r\nx-ratelimit-remaining-tokens: 900\r\nx-ratelimit-reset-tokens: 120ms\r\nconnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                socket
                    .write_all(response.as_bytes())
                    .await
                    .expect("write response");
            }
        });

        format!("http://{address}")
    }

    async fn read_http_request(socket: &mut tokio::net::TcpStream) -> anyhow::Result<()> {
        let mut buffer = Vec::new();
        let mut chunk = [0u8; 1024];

        loop {
            let read = socket.read(&mut chunk).await?;
            if read == 0 {
                break;
            }
            buffer.extend_from_slice(&chunk[..read]);
            if let Some(headers_end) = find_headers_end(&buffer) {
                let header_text = String::from_utf8_lossy(&buffer[..headers_end]);
                let content_length = header_text
                    .lines()
                    .find_map(|line| {
                        line.split_once(':').and_then(|(name, value)| {
                            name.trim()
                                .eq_ignore_ascii_case("content-length")
                                .then(|| value.trim().parse::<usize>().ok())
                                .flatten()
                        })
                    })
                    .unwrap_or(0);
                let body_bytes = buffer.len().saturating_sub(headers_end + 4);
                if body_bytes >= content_length {
                    return Ok(());
                }
            }
        }

        anyhow::bail!("incomplete http request")
    }

    fn find_headers_end(buffer: &[u8]) -> Option<usize> {
        buffer.windows(4).position(|window| window == b"\r\n\r\n")
    }
}
