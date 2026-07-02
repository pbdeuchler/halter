// pattern: Imperative Shell

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};
use halter_protocol::{
    ApiKind, CompactionWindow, Message, ProviderCapabilities, ProviderCompactionRequest,
    ProviderCompactionResponse, ProviderCompactionStrategy, ProviderError, ProviderErrorKind,
    ProviderRequest, StreamEvent, ToolCallIdPolicy,
};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

use crate::Provider;
use crate::anthropic_codec;
use crate::header_overrides::HeaderOverrides;
use crate::http_client::{JsonHttpClient, JsonRequest, join_url, provider_error_from_anyhow};
use crate::resilience::{ProviderErrorClassifier, ResiliencePolicy, ResilientProvider};
use crate::secret::SecretString;

const ANTHROPIC_MESSAGES_PATH: &str = "/v1/messages";

#[derive(Debug, Clone)]
/// Anthropic Messages API provider.
///
/// Like [`crate::OpenAiProvider`] and [`crate::OpenRouterProvider`], the
/// transport core is wrapped in a [`ResilientProvider`], so every constructor
/// yields bounded retries with backoff for transient failures (429/529,
/// overload, network faults) on both `stream` and `compact`.
pub struct AnthropicProvider {
    inner: ResilientProvider<AnthropicMessagesProvider>,
}

impl AnthropicProvider {
    /// Construct an Anthropic provider with default headers and no temperature override.
    pub fn new(
        api_key: impl Into<SecretString>,
        base_url: impl Into<String>,
    ) -> anyhow::Result<Self> {
        Self::new_with_headers(api_key, base_url, &[], None)
    }

    /// Same as [`AnthropicProvider::new`] but also accepts user-configured
    /// header overrides that replace any default or hardcoded header
    /// (`x-api-key`, `anthropic-version`, `Content-Type`) case-insensitively.
    /// When `temperature` is `Some`, it is forwarded verbatim into every
    /// request body; otherwise request bodies omit temperature.
    pub fn new_with_headers(
        api_key: impl Into<SecretString>,
        base_url: impl Into<String>,
        header_overrides: &[(String, String)],
        temperature: Option<f32>,
    ) -> anyhow::Result<Self> {
        Self::new_with_headers_and_resilience(
            api_key,
            base_url,
            header_overrides,
            temperature,
            ResiliencePolicy::default(),
            Arc::new(crate::DefaultProviderErrorClassifier),
        )
    }

    /// Same as [`AnthropicProvider::new_with_headers`] with an explicit
    /// provider-request resilience policy and error classifier.
    pub fn new_with_headers_and_resilience(
        api_key: impl Into<SecretString>,
        base_url: impl Into<String>,
        header_overrides: &[(String, String)],
        temperature: Option<f32>,
        resilience_policy: ResiliencePolicy,
        classifier: Arc<dyn ProviderErrorClassifier>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            inner: ResilientProvider::new_with_classifier(
                "anthropic",
                AnthropicMessagesProvider::try_new(
                    api_key,
                    base_url,
                    header_overrides,
                    temperature,
                    resilience_policy,
                )?,
                resilience_policy,
                classifier,
            ),
        })
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        self.inner.capabilities()
    }

    fn compaction_window(&self, messages: &[Message]) -> Option<CompactionWindow> {
        self.inner.compaction_window(messages)
    }

    async fn stream(
        &self,
        request: ProviderRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
        if request.model.api_kind != ApiKind::AnthropicMessages {
            anyhow::bail!(
                "failed to execute provider request: anthropic provider requires anthropic_messages api kind"
            );
        }
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

/// Transport core for the Anthropic Messages API. Owns only transport-level
/// concerns (HTTP client, encode/decode); retry, backoff, and error
/// classification live in the [`ResilientProvider`] wrapper.
#[derive(Debug, Clone)]
struct AnthropicMessagesProvider {
    api_key: SecretString,
    base_url: String,
    client: JsonHttpClient,
    header_overrides: HeaderOverrides,
    temperature: Option<f32>,
}

impl AnthropicMessagesProvider {
    fn try_new(
        api_key: impl Into<SecretString>,
        base_url: impl Into<String>,
        header_overrides: &[(String, String)],
        temperature: Option<f32>,
        resilience_policy: ResiliencePolicy,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            client: JsonHttpClient::try_new_with_timeouts(resilience_policy.timeouts)?,
            header_overrides: HeaderOverrides::new(header_overrides)?,
            temperature,
        })
    }

    fn default_headers(&self, enable_interleaved_thinking: bool) -> Vec<(String, String)> {
        let mut headers = vec![
            (
                "x-api-key".to_owned(),
                self.api_key.expose_secret().to_owned(),
            ),
            ("anthropic-version".to_owned(), "2023-06-01".to_owned()),
        ];
        if enable_interleaved_thinking {
            headers.push((
                "anthropic-beta".to_owned(),
                "interleaved-thinking-2025-05-14".to_owned(),
            ));
        }
        headers
    }
}

#[async_trait]
impl Provider for AnthropicMessagesProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities {
            supports_tools: true,
            supports_streaming: true,
            supports_reasoning: true,
            supports_interleaved_reasoning: true,
            supports_images: true,
            supports_documents: true,
            supports_prompt_cache: true,
            supports_compaction: true,
            compaction_strategy: Some(ProviderCompactionStrategy::Inline),
            supports_tool_result_media: false,
            requires_non_empty_assistant_content: true,
            tool_call_id_policy: ToolCallIdPolicy::StableReplayNormalized,
            max_input_tokens: 200_000,
            max_output_tokens: 8_192,
        }
    }

    fn compaction_window(&self, messages: &[Message]) -> Option<CompactionWindow> {
        Some(CompactionWindow::preserve_through_latest_user(messages))
    }

    async fn stream(
        &self,
        request: ProviderRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
        info!(
            provider = "anthropic",
            session_id = %request.session_id,
            turn_id = %request.turn_id,
            model = %request.model.model,
            message_count = request.messages.len(),
            tool_count = request.tools.len(),
            "starting anthropic request"
        );

        // Deterministic encode failures are Fatal so the resilience wrapper
        // short-circuits instead of burning the retry budget (H4).
        let body = match anthropic_codec::encode_stream_request(&request, self.temperature) {
            Ok(body) => body,
            Err(error) => {
                return Ok(single_error_stream(ProviderError::with_kind(
                    format!("failed to encode anthropic request: {error:#}"),
                    ProviderErrorKind::Fatal,
                )));
            }
        };
        let enable_interleaved_thinking =
            request.model.reasoning.is_some() && !request.tools.is_empty();
        let raw_stream = match self
            .client
            .post_json_event_stream(
                JsonRequest {
                    provider_label: "anthropic",
                    url: join_url(&self.base_url, ANTHROPIC_MESSAGES_PATH),
                    headers: self
                        .header_overrides
                        .merge_string_pairs(self.default_headers(enable_interleaved_thinking)),
                    body,
                },
                cancel,
            )
            .await
        {
            Ok(raw_stream) => raw_stream,
            // Transport failures carry their status/tag classification from
            // the shared classifier (429 → RateLimited with Retry-After
            // hint, 5xx → Transient, other 4xx → Fatal), so the resilience
            // wrapper sees the same taxonomy as the OpenAI-family providers.
            Err(error) => return Ok(single_error_stream(provider_error_from_anyhow(error))),
        };
        let mut decoder = anthropic_codec::AnthropicStreamDecoder::new(&request);
        Ok(raw_stream
            .flat_map(move |item| {
                let events = match item {
                    Ok(event) => decoder
                        .decode(&event)
                        .map(|events| {
                            events
                                .into_iter()
                                .map(|event| match event {
                                    StreamEvent::Error { error } => Err(error),
                                    event => Ok(event),
                                })
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_else(|error| {
                            vec![Err(ProviderError::with_kind(
                                format!("failed to decode anthropic stream: {error:#}"),
                                ProviderErrorKind::Fatal,
                            ))]
                        }),
                    // Mid-stream SSE read failures are connection faults —
                    // inherently transient, mirroring the OpenAI-family
                    // treatment of stream/network errors.
                    Err(error) => vec![Err(ProviderError::with_kind(
                        format!("{error:#}"),
                        ProviderErrorKind::Transient,
                    ))],
                };
                debug!(event_count = events.len(), "decoded anthropic stream event");
                stream::iter(events)
            })
            .boxed())
    }

    async fn compact(
        &self,
        request: ProviderCompactionRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<ProviderCompactionResponse> {
        if request.model.api_kind != ApiKind::AnthropicMessages {
            anyhow::bail!(
                "failed to compact session: anthropic provider requires anthropic_messages api kind"
            );
        }
        info!(
            provider = "anthropic",
            session_id = %request.session_id,
            model = %request.model.model,
            compacted_prefix_items = request.compacted_prefix.len(),
            message_count = request.messages.len(),
            "starting anthropic compaction request"
        );

        let body = anthropic_codec::encode_compaction_request(&request, self.temperature)?;
        let response = self
            .client
            .post_json(
                JsonRequest {
                    provider_label: "anthropic",
                    url: join_url(&self.base_url, ANTHROPIC_MESSAGES_PATH),
                    headers: self
                        .header_overrides
                        .merge_string_pairs(self.default_headers(false)),
                    body,
                },
                cancel,
            )
            .await?;
        anthropic_codec::decode_compaction_response(&response)
    }
}

fn single_error_stream(
    error: ProviderError,
) -> BoxStream<'static, Result<StreamEvent, ProviderError>> {
    stream::iter([Err(error)]).boxed()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use chrono::Utc;
    use futures::StreamExt;
    use halter_protocol::{
        ApiKind, AssembledPrompt, AssistantMessage, AssistantPart, CacheBreakpoints, Message,
        MessageId, ModelId, ModelRole, ProviderKind, ProviderName, ReasoningEffort, ResolvedModel,
        StopReason, ToolAlias, ToolCall, ToolCallId, ToolCapabilities, ToolConcurrency, ToolResult,
        ToolResultMessage, ToolSpec, TurnId, UserMessage, UserPart,
    };
    use indexmap::IndexMap;
    use serde_json::{Value, json};
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;

    use super::*;
    use crate::RetryPolicy;
    use crate::resilience::ProviderTimeouts;
    use crate::test_http::{find_headers_end, read_http_request};

    #[test]
    fn anthropic_provider_reports_current_feature_support() {
        let capabilities = AnthropicProvider::new("test-key", "https://api.anthropic.com")
            .expect("anthropic provider")
            .capabilities();

        assert!(capabilities.supports_streaming);
        assert!(capabilities.supports_prompt_cache);
        assert!(capabilities.supports_compaction);
        assert_eq!(
            capabilities.compaction_strategy,
            Some(ProviderCompactionStrategy::Inline)
        );
        assert!(capabilities.supports_interleaved_reasoning);
    }

    #[tokio::test]
    async fn anthropic_provider_rejects_non_messages_api_kind() {
        let provider = AnthropicProvider::new("test-key", "https://api.anthropic.com")
            .expect("anthropic provider");
        let mut request = sample_request();
        request.model.api_kind = ApiKind::OpenAiResponses;

        let error = match provider.stream(request, CancellationToken::new()).await {
            Ok(_) => panic!("anthropic provider should reject non-messages requests"),
            Err(error) => error,
        };

        assert!(
            error
                .to_string()
                .contains("anthropic provider requires anthropic_messages api kind")
        );
    }

    #[test]
    fn anthropic_provider_compaction_window_preserves_through_latest_user() {
        let provider = AnthropicProvider::new("test-key", "https://api.anthropic.com")
            .expect("anthropic provider");
        let tool_call_id = ToolCallId::from("call_1");
        let messages = vec![
            Message::User(UserMessage::text("first")),
            assistant_text("answer"),
            Message::User(UserMessage::text("latest")),
            Message::Assistant(AssistantMessage {
                id: MessageId::new(),
                created_at: Utc::now(),
                parts: vec![AssistantPart::ToolCall(ToolCall {
                    id: tool_call_id.clone(),
                    name: "read".into(),
                    arguments: json!({}),
                })],
                stop_reason: None,
                usage: None,
                replay_meta: Default::default(),
            }),
            Message::Tool(ToolResultMessage {
                id: MessageId::new(),
                call_id: tool_call_id,
                content: ToolResult::Text {
                    text: "tail".to_owned(),
                },
                error: None,
                created_at: Utc::now(),
            }),
        ];

        let window = provider
            .compaction_window(&messages)
            .expect("compaction window");

        assert_eq!(window.preserved_messages.len(), 3);
        assert!(matches!(
            window.preserved_messages.last(),
            Some(Message::User(_))
        ));
        assert_eq!(window.eligible_messages.len(), 2);
        assert!(!window.reserved_response_block);
    }

    #[tokio::test]
    async fn anthropic_provider_streams_sse_messages() {
        let captured = std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let base_url = spawn_stream_server(captured.clone()).await;
        let provider = AnthropicProvider::new("test-key", base_url).expect("anthropic provider");
        let mut stream = provider
            .stream(sample_request(), CancellationToken::new())
            .await
            .expect("provider stream");

        let mut events = Vec::new();
        while let Some(event) = stream.next().await {
            let event = event.expect("stream event");
            let done = matches!(event, StreamEvent::MessageEnd { .. });
            events.push(event);
            if done {
                break;
            }
        }

        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::TextDelta { delta, .. } if delta == "hello"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::MessageEnd { stop_reason, .. } if *stop_reason == StopReason::EndTurn
        )));

        let captured = captured.lock().await.clone();
        let headers_end = find_headers_end(&captured).expect("headers end");
        let request_text = String::from_utf8_lossy(&captured[..headers_end]);
        assert!(request_text.starts_with("POST /v1/messages HTTP/1.1"));
        assert!(request_text.contains("x-api-key: test-key"));
        assert!(request_text.contains("anthropic-version: 2023-06-01"));
        assert!(request_text.contains("anthropic-beta: interleaved-thinking-2025-05-14"));

        let body: Value = serde_json::from_slice(&captured[headers_end + 4..]).expect("parse body");
        assert_eq!(body["stream"], true);
    }

    /// Anthropic now shares the resilience wrapper: an HTTP 429 with a
    /// Retry-After hint must be retried, matching the OpenAI treatment.
    /// Previously the direct constructor issued exactly one request and
    /// surfaced a non-retryable error.
    #[tokio::test]
    async fn anthropic_provider_retries_rate_limited_requests() {
        let attempts = std::sync::Arc::new(tokio::sync::Mutex::new(0u32));
        let base_url = spawn_rate_limited_then_success_server(attempts.clone()).await;
        let provider = AnthropicProvider::new_with_headers_and_resilience(
            "test-key",
            base_url,
            &[],
            None,
            test_policy(3),
            std::sync::Arc::new(crate::DefaultProviderErrorClassifier),
        )
        .expect("anthropic provider");

        let mut stream = provider
            .stream(sample_request(), CancellationToken::new())
            .await
            .expect("provider stream");
        let mut events = Vec::new();
        while let Some(event) = stream.next().await {
            let event = event.expect("stream should recover after retry");
            let done = matches!(event, StreamEvent::MessageEnd { .. });
            events.push(event);
            if done {
                break;
            }
        }

        assert_eq!(*attempts.lock().await, 2, "429 must trigger one retry");
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::TextDelta { delta, .. } if delta == "hello"
        )));
    }

    /// A 4xx client error must stay fatal: exactly one request, no retry.
    #[tokio::test]
    async fn anthropic_provider_does_not_retry_invalid_requests() {
        let attempts = std::sync::Arc::new(tokio::sync::Mutex::new(0u32));
        let base_url = spawn_invalid_request_server(attempts.clone()).await;
        let provider = AnthropicProvider::new_with_headers_and_resilience(
            "test-key",
            base_url,
            &[],
            None,
            test_policy(3),
            std::sync::Arc::new(crate::DefaultProviderErrorClassifier),
        )
        .expect("anthropic provider");

        let mut stream = provider
            .stream(sample_request(), CancellationToken::new())
            .await
            .expect("provider stream");
        let mut errors = Vec::new();
        while let Some(item) = stream.next().await {
            errors.push(item.expect_err("invalid request should surface an error"));
        }

        assert_eq!(*attempts.lock().await, 1, "4xx must not be retried");
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].kind, ProviderErrorKind::Fatal);
        assert!(errors[0].message.contains("max_tokens"));
    }

    fn test_policy(max_attempts: u32) -> ResiliencePolicy {
        ResiliencePolicy {
            timeouts: ProviderTimeouts {
                connect: Duration::from_secs(1),
                request: Duration::from_secs(5),
                stream_idle: Duration::from_secs(5),
            },
            request_retry: RetryPolicy {
                max_attempts,
                base_backoff: Duration::from_millis(1),
                max_backoff: Duration::from_millis(5),
                deadline: Duration::from_secs(10),
                jitter_pct: 0,
            },
        }
    }

    fn sse_success_response() -> String {
        let body = [
            r#"event: message_start"#,
            r#"data: {"type":"message_start","message":{"id":"msg_123","usage":{"input_tokens":1,"output_tokens":0}}}"#,
            "",
            r#"event: content_block_start"#,
            r#"data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
            "",
            r#"event: content_block_delta"#,
            r#"data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}"#,
            "",
            r#"event: content_block_stop"#,
            r#"data: {"type":"content_block_stop","index":0}"#,
            "",
            r#"event: message_delta"#,
            r#"data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}"#,
            "",
            r#"event: message_stop"#,
            r#"data: {"type":"message_stop"}"#,
            "",
            "",
        ]
        .join("\n");
        format!(
            "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
            body.len(),
            body
        )
    }

    async fn spawn_stream_server(captured: std::sync::Arc<tokio::sync::Mutex<Vec<u8>>>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind listener");
        let address = listener.local_addr().expect("local addr");
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept socket");
            let request = read_http_request(&mut socket).await.expect("read request");
            *captured.lock().await = request;

            socket
                .write_all(sse_success_response().as_bytes())
                .await
                .expect("write response");
        });
        format!("http://{address}")
    }

    async fn spawn_rate_limited_then_success_server(
        attempts: std::sync::Arc<tokio::sync::Mutex<u32>>,
    ) -> String {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind listener");
        let address = listener.local_addr().expect("local addr");
        tokio::spawn(async move {
            for attempt in 0..2 {
                let (mut socket, _) = listener.accept().await.expect("accept socket");
                read_http_request(&mut socket).await.expect("read request");
                *attempts.lock().await += 1;
                let response = if attempt == 0 {
                    let body = r#"{"type":"error","error":{"type":"rate_limit_error","message":"Number of requests exceeded your rate limit"}}"#;
                    format!(
                        "HTTP/1.1 429 Too Many Requests\r\ncontent-type: application/json\r\nretry-after: 0\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                        body.len(),
                        body
                    )
                } else {
                    sse_success_response()
                };
                socket
                    .write_all(response.as_bytes())
                    .await
                    .expect("write response");
            }
        });
        format!("http://{address}")
    }

    async fn spawn_invalid_request_server(
        attempts: std::sync::Arc<tokio::sync::Mutex<u32>>,
    ) -> String {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind listener");
        let address = listener.local_addr().expect("local addr");
        tokio::spawn(async move {
            for _ in 0..4 {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                if read_http_request(&mut socket).await.is_err() {
                    return;
                }
                *attempts.lock().await += 1;
                let body = r#"{"type":"error","error":{"type":"invalid_request_error","message":"max_tokens too large"}}"#;
                let response = format!(
                    "HTTP/1.1 400 Bad Request\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = socket.write_all(response.as_bytes()).await;
            }
        });
        format!("http://{address}")
    }

    fn sample_request() -> ProviderRequest {
        let mut provider_aliases = IndexMap::new();
        provider_aliases.insert(ProviderKind::Anthropic, ToolAlias::from("fs_read"));
        ProviderRequest {
            session_id: Default::default(),
            turn_id: TurnId::new(),
            model: ResolvedModel {
                role: ModelRole::default(),
                id: ModelId::from("claude_default"),
                provider: ProviderName::from("anthropic"),
                provider_kind: ProviderKind::Anthropic,
                api_kind: ApiKind::AnthropicMessages,
                model: "claude-sonnet-4-5".to_owned(),
                max_input_tokens: Some(200_000),
                max_output_tokens: Some(8_192),
                reasoning: Some(ReasoningEffort::Medium),
                tokens_per_minute: None,
            },
            prompt: AssembledPrompt {
                segments: Vec::new(),
                transcript: Vec::new(),
                ordered_segments: Vec::new(),
                prefix_cache_key: "cache-key".to_owned(),
                rendered_prefix: "follow plan".to_owned(),
                rendered_transcript: String::new(),
                rendered: String::new(),
                cache_breakpoints: CacheBreakpoints::default(),
                system_segment_count: 0,
                skill_segment_count: 0,
            },
            compacted_prefix: Vec::new(),
            messages: vec![Message::User(UserMessage {
                id: MessageId::new(),
                created_at: Utc::now(),
                parts: vec![UserPart::Text {
                    text: "hi".to_owned(),
                }],
            })],
            tools: vec![ToolSpec {
                name: "read".into(),
                description: "Read a file".to_owned(),
                input_schema: json!({"type": "object"}),
                concurrency: ToolConcurrency::ReadOnly,
                capabilities: ToolCapabilities::default(),
                provider_aliases,
            }],
            previous_response_id: None,
            new_messages_start: 0,
        }
    }

    fn assistant_text(text: &str) -> Message {
        Message::Assistant(AssistantMessage {
            id: MessageId::new(),
            created_at: Utc::now(),
            parts: vec![AssistantPart::Text {
                text: text.to_owned(),
            }],
            stop_reason: None,
            usage: None,
            replay_meta: Default::default(),
        })
    }
}
