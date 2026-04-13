// pattern: Imperative Shell

use async_openai::{
    error::{ApiError, OpenAIError, StreamError, WrappedError},
    types::responses::{ResponseStream, ResponseStreamEvent},
};
use eventsource_stream::Eventsource;
use futures::StreamExt;
use reqwest::{Client as ReqwestClient, Response, StatusCode};
use serde_json::Value;
use tokio::select;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_util::sync::CancellationToken;

use crate::openai_rate_limit::{OpenAiRateLimitPermit, OpenAiRateLimiter};
use crate::openai_rate_limit_policy::OpenAiReservation;

const RESPONSES_PATH: &str = "/v1/responses";

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
    ) -> anyhow::Result<ResponseStream> {
        let mut permit = self
            .rate_limit_permit(&request_meta, cancel.child_token())
            .await?;
        let request_builder = self
            .client
            .post(provider_url(&self.base_url, RESPONSES_PATH))
            .bearer_auth(&self.api_key)
            .json(&request);

        let response = select! {
            _ = cancel.cancelled() => anyhow::bail!("failed to execute provider request: request cancelled"),
            result = request_builder.send() => result,
        }
        .map_err(|error| anyhow::anyhow!(
            "failed to execute {} request: {error}",
            request_meta.provider_label
        ))?;
        let status = response.status();
        let headers = response.headers().clone();
        if let Some(permit) = permit.as_mut() {
            permit.update_from_headers(&headers, status);
        }

        if !status.is_success() {
            let body = select! {
                _ = cancel.cancelled() => anyhow::bail!("failed to execute provider request: request cancelled"),
                result = response.bytes() => result,
            }
            .map_err(|error| anyhow::anyhow!(
                "failed to read {} response body: {error}",
                request_meta.provider_label
            ))?;
            let error = decode_openai_error(status, &body);
            anyhow::bail!("failed to execute provider request: {error}");
        }

        Ok(stream_response(response))
    }

    async fn rate_limit_permit(
        &self,
        request_meta: &ResponsesTransportRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<Option<OpenAiRateLimitPermit>> {
        match request_meta.rate_limit_strategy {
            Some(ResponsesRateLimitStrategy::OpenAiHeaders) => Ok(Some(
                self.openai_rate_limiter
                    .acquire(&request_meta.model, request_meta.reservation, cancel)
                    .await?,
            )),
            None => Ok(None),
        }
    }
}

fn provider_url(base_url: &str, path: &str) -> String {
    format!("{}{}", base_url.trim_end_matches('/'), path)
}

fn stream_response(response: Response) -> ResponseStream {
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

                    let parsed = serde_json::from_str::<ResponseStreamEvent>(&event.data)
                        .map_err(|error| OpenAIError::JSONDeserialize(error, event.data.clone()));

                    if tx.send(parsed).is_err() {
                        break;
                    }
                }
            }
        }
    });

    Box::pin(UnboundedReceiverStream::new(rx))
}

fn decode_openai_error(status: StatusCode, body: &[u8]) -> OpenAIError {
    if status.is_server_error() {
        let message = String::from_utf8_lossy(body).trim().to_owned();
        return OpenAIError::ApiError(ApiError {
            message,
            r#type: None,
            param: None,
            code: None,
        });
    }

    serde_json::from_slice::<WrappedError>(body)
        .map(|wrapped| OpenAIError::ApiError(wrapped.error))
        .unwrap_or_else(|_| {
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
