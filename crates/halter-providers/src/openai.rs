// pattern: Imperative Shell

use async_trait::async_trait;
use futures::stream::BoxStream;
use halter_protocol::{
    ProviderCapabilities, ProviderCompactionRequest, ProviderCompactionResponse, ProviderError,
    ProviderRequest, StreamEvent, ToolCallIdPolicy,
};
use tokio_util::sync::CancellationToken;

use crate::Provider;
use crate::responses_provider::{
    CompactStrategy, ResponsesProvider, ResponsesProviderConfig, ResponsesProviderRequestConfig,
};
use crate::responses_transport::ResponsesRateLimitStrategy;
use crate::retry::RetryPolicy;
use crate::secret::SecretString;

#[derive(Debug, Clone)]
pub struct OpenAiProvider {
    inner: ResponsesProvider,
}

impl OpenAiProvider {
    pub fn new(
        api_key: impl Into<SecretString>,
        base_url: impl Into<String>,
    ) -> anyhow::Result<Self> {
        Self::new_with_headers(api_key, base_url, &[])
    }

    /// Construct an OpenAI provider with user-configured HTTP header
    /// overrides. Overrides replace any default or hardcoded header
    /// (`Authorization`, `Content-Type`) case-insensitively.
    pub fn new_with_headers(
        api_key: impl Into<SecretString>,
        base_url: impl Into<String>,
        header_overrides: &[(String, String)],
    ) -> anyhow::Result<Self> {
        Ok(Self {
            inner: ResponsesProvider::try_new(config(), api_key, base_url, header_overrides)?,
        })
    }
}

#[async_trait]
impl Provider for OpenAiProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        self.inner.capabilities()
    }

    async fn stream(
        &self,
        request: ProviderRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
        self.inner.stream(request, cancel).await
    }

    async fn compact(
        &self,
        request: ProviderCompactionRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<ProviderCompactionResponse> {
        self.inner.compact(request, cancel).await
    }
}

fn config() -> ResponsesProviderConfig {
    ResponsesProviderConfig {
        label: "openai",
        capabilities: ProviderCapabilities {
            supports_tools: true,
            supports_streaming: true,
            supports_reasoning: true,
            supports_interleaved_reasoning: false,
            supports_images: true,
            supports_documents: true,
            supports_prompt_cache: true,
            supports_compaction: true,
            supports_tool_result_media: false,
            requires_non_empty_assistant_content: false,
            tool_call_id_policy: ToolCallIdPolicy::ProviderSupplied,
            max_input_tokens: 200_000,
            max_output_tokens: 8_192,
        },
        request: ResponsesProviderRequestConfig {
            store: None,
            include_prompt_cache_key: true,
            include_encrypted_reasoning: true,
            reasoning_summary: Some("auto"),
        },
        compact_strategy: Some(CompactStrategy::DedicatedEndpoint),
        rate_limit_strategy: Some(ResponsesRateLimitStrategy::OpenAiHeaders),
        retry_policy: RetryPolicy::default(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use futures::StreamExt;
    use halter_protocol::{
        ApiKind, AssembledPrompt, ModelId, ModelRole, ProviderKind, ProviderName, ResolvedModel,
        SessionId, StreamEvent, TurnId,
    };
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;

    #[tokio::test]
    async fn openai_provider_rejects_chat_api_kind() {
        let provider =
            OpenAiProvider::new("test-key", "https://api.openai.com").expect("openai provider");
        let error = match provider
            .stream(
                sample_request(ApiKind::OpenAiChat),
                CancellationToken::new(),
            )
            .await
        {
            Ok(_) => panic!("openai provider should reject chat requests"),
            Err(error) => error,
        };

        assert!(
            error
                .to_string()
                .contains("openai provider requires openai_responses api kind")
        );
    }

    #[tokio::test]
    async fn openai_provider_retries_streamed_token_rate_limits() {
        let request_times = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let base_url = spawn_retrying_stream_server(request_times.clone()).await;
        let provider = OpenAiProvider::new("test-key", base_url).expect("openai provider");
        let mut stream = provider
            .stream(
                sample_request(ApiKind::OpenAiResponses),
                CancellationToken::new(),
            )
            .await
            .expect("provider stream");

        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            let event = item.expect("stream event");
            let done = matches!(event, StreamEvent::MessageEnd { .. });
            events.push(event);
            if done {
                break;
            }
        }

        assert_eq!(
            events
                .iter()
                .filter(|event| matches!(event, StreamEvent::MessageStart { .. }))
                .count(),
            1
        );
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::TextDelta { delta, .. } if delta == "done"
        )));

        let request_times = request_times.lock().await.clone();
        assert_eq!(request_times.len(), 2);
        let gap = request_times[1].saturating_duration_since(request_times[0]);
        assert!(
            gap >= Duration::from_millis(15),
            "expected retry to honor cooldown, saw gap {gap:?}"
        );
    }

    #[tokio::test]
    async fn openai_provider_applies_header_overrides() {
        let captured = Arc::new(tokio::sync::Mutex::new(Vec::<u8>::new()));
        let base_url = spawn_header_capture_server(captured.clone()).await;
        let overrides = vec![
            ("authorization".to_owned(), "Bearer override-token".to_owned()),
            ("X-Trace-Id".to_owned(), "trace-1".to_owned()),
        ];
        let provider = OpenAiProvider::new_with_headers("default-key", base_url, &overrides)
            .expect("openai provider");

        let mut stream = provider
            .stream(
                sample_request(ApiKind::OpenAiResponses),
                CancellationToken::new(),
            )
            .await
            .expect("provider stream");
        while stream.next().await.is_some() {}

        let captured = captured.lock().await.clone();
        let headers_end = find_headers_end(&captured).expect("headers end");
        let request_text = String::from_utf8_lossy(&captured[..headers_end]);

        let bearer_lines = request_text
            .lines()
            .filter(|line| line.to_ascii_lowercase().starts_with("authorization:"))
            .count();
        assert_eq!(bearer_lines, 1, "authorization should appear once");
        assert!(
            request_text
                .lines()
                .any(|line| line.eq_ignore_ascii_case("authorization: Bearer override-token")),
            "override bearer must replace default, got:\n{request_text}"
        );
        assert!(
            request_text
                .lines()
                .any(|line| line.eq_ignore_ascii_case("x-trace-id: trace-1")),
            "custom header must be forwarded, got:\n{request_text}"
        );
        let content_type_lines = request_text
            .lines()
            .filter(|line| line.to_ascii_lowercase().starts_with("content-type:"))
            .count();
        assert_eq!(content_type_lines, 1, "content-type should not duplicate");
    }

    async fn spawn_header_capture_server(captured: Arc<tokio::sync::Mutex<Vec<u8>>>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind listener");
        let address = listener.local_addr().expect("local addr");

        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept socket");
            let buffer = read_http_request_buffer(&mut socket)
                .await
                .expect("read request");
            *captured.lock().await = buffer;

            let completed = json!({
                "type": "response.completed",
                "sequence_number": 0,
                "response": {
                    "id": "resp_hdr",
                    "created_at": 0,
                    "model": "gpt-5.4",
                    "object": "response",
                    "output": [],
                    "status": "completed",
                    "usage": {
                        "input_tokens": 0,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens": 0,
                        "output_tokens_details": {"reasoning_tokens": 0},
                        "total_tokens": 0
                    }
                }
            });
            let body = format!("data: {completed}\n\ndata: [DONE]\n\n");
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("write response");
        });

        format!("http://{address}")
    }

    async fn read_http_request_buffer(
        socket: &mut tokio::net::TcpStream,
    ) -> anyhow::Result<Vec<u8>> {
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
                    return Ok(buffer);
                }
            }
        }

        anyhow::bail!("incomplete http request")
    }

    fn sample_request(api_kind: ApiKind) -> ProviderRequest {
        ProviderRequest {
            session_id: SessionId::new(),
            turn_id: TurnId::new(),
            model: ResolvedModel {
                role: ModelRole::default(),
                id: ModelId::from("default"),
                provider: ProviderName::from("openai"),
                provider_kind: ProviderKind::OpenAi,
                api_kind,
                model: "gpt-5".to_owned(),
                max_input_tokens: Some(200_000),
                max_output_tokens: Some(8_192),
                reasoning: None,
                tokens_per_minute: None,
            },
            prompt: AssembledPrompt {
                segments: Vec::new(),
                transcript: Vec::new(),
                ordered_segments: Vec::new(),
                prefix_cache_key: "cache-key".to_owned(),
                rendered_prefix: String::new(),
                rendered_transcript: String::new(),
                rendered: String::new(),
            },
            compacted_prefix: Vec::new(),
            messages: Vec::new(),
            tools: Vec::new(),
            previous_response_id: None,
            new_messages_start: 0,
        }
    }

    async fn spawn_retrying_stream_server(
        request_times: Arc<tokio::sync::Mutex<Vec<Instant>>>,
    ) -> String {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind listener");
        let address = listener.local_addr().expect("local addr");

        tokio::spawn(async move {
            for attempt in 0..2 {
                let (mut socket, _) = listener.accept().await.expect("accept socket");
                read_http_request(&mut socket).await.expect("read request");
                request_times.lock().await.push(Instant::now());
                let body = if attempt == 0 {
                    let warmup = json!({
                        "type": "response.output_item.added",
                        "sequence_number": 0,
                        "output_index": 0,
                        "item": {
                            "type": "message",
                            "id": "msg_retry",
                            "role": "assistant",
                            "status": "in_progress",
                            "content": []
                        }
                    });
                    let error = json!({
                        "type": "error",
                        "error": {
                            "type": "tokens",
                            "code": "rate_limit_exceeded",
                            "message": "Rate limit reached for gpt-5.4 in organization org_test on tokens per min (TPM): Limit 500000, Used 414695, Requested 94256. Please try again in 0.02s. Visit https://platform.openai.com/account/rate-limits to learn more.",
                            "param": null
                        },
                        "sequence_number": 1
                    });
                    format!("data: {warmup}\n\ndata: {error}\n\ndata: [DONE]\n\n")
                } else {
                    let created = json!({
                        "type": "response.created",
                        "sequence_number": 0,
                        "response": {
                            "id": "resp_success",
                            "created_at": 0,
                            "model": "gpt-5.4",
                            "object": "response",
                            "output": [],
                            "status": "in_progress"
                        }
                    });
                    let added = json!({
                        "type": "response.output_item.added",
                        "sequence_number": 1,
                        "output_index": 0,
                        "item": {
                            "type": "message",
                            "id": "msg_success",
                            "role": "assistant",
                            "status": "in_progress",
                            "content": []
                        }
                    });
                    let delta = json!({
                        "type": "response.output_text.delta",
                        "sequence_number": 2,
                        "item_id": "msg_success",
                        "output_index": 0,
                        "content_index": 0,
                        "delta": "done"
                    });
                    let item_done = json!({
                        "type": "response.output_item.done",
                        "sequence_number": 3,
                        "output_index": 0,
                        "item": {
                            "type": "message",
                            "id": "msg_success",
                            "role": "assistant",
                            "status": "completed",
                            "content": [
                                {
                                    "type": "output_text",
                                    "text": "done",
                                    "annotations": []
                                }
                            ]
                        }
                    });
                    let completed = json!({
                        "type": "response.completed",
                        "sequence_number": 4,
                        "response": {
                            "id": "resp_success",
                            "created_at": 0,
                            "model": "gpt-5.4",
                            "object": "response",
                            "output": [],
                            "status": "completed",
                            "usage": {
                                "input_tokens": 12,
                                "input_tokens_details": {
                                    "cached_tokens": 0
                                },
                                "output_tokens": 4,
                                "output_tokens_details": {
                                    "reasoning_tokens": 0
                                },
                                "total_tokens": 16
                            }
                        }
                    });
                    format!(
                        "data: {created}\n\ndata: {added}\n\ndata: {delta}\n\ndata: {item_done}\n\ndata: {completed}\n\ndata: [DONE]\n\n"
                    )
                };
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
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
