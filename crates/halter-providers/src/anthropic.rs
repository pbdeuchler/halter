// pattern: Imperative Shell

use async_trait::async_trait;
use futures::stream::{self, BoxStream, StreamExt};
use halter_protocol::{
    ApiKind, CompactionWindow, Message, ProviderCapabilities, ProviderCompactionRequest,
    ProviderCompactionResponse, ProviderCompactionStrategy, ProviderError, ProviderRequest,
    StreamEvent, ToolCallIdPolicy,
};
use tokio_util::sync::CancellationToken;
use tracing::{debug, info};

use crate::Provider;
use crate::anthropic_codec;
use crate::header_overrides::HeaderOverrides;
use crate::http_client::{JsonHttpClient, JsonRequest};
use crate::resilience::ProviderTimeouts;
use crate::secret::SecretString;

const ANTHROPIC_MESSAGES_PATH: &str = "/v1/messages";

#[derive(Debug, Clone)]
/// Anthropic Messages API provider.
pub struct AnthropicProvider {
    api_key: SecretString,
    base_url: String,
    client: JsonHttpClient,
    header_overrides: HeaderOverrides,
    temperature: Option<f32>,
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
        Self::new_with_headers_and_timeouts(
            api_key,
            base_url,
            header_overrides,
            temperature,
            ProviderTimeouts::default(),
        )
    }

    pub fn new_with_headers_and_timeouts(
        api_key: impl Into<SecretString>,
        base_url: impl Into<String>,
        header_overrides: &[(String, String)],
        temperature: Option<f32>,
        timeouts: ProviderTimeouts,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            api_key: api_key.into(),
            base_url: base_url.into(),
            client: JsonHttpClient::try_new_with_timeouts(timeouts)?,
            header_overrides: HeaderOverrides::new(header_overrides)?,
            temperature,
        })
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
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
        if request.model.api_kind != ApiKind::AnthropicMessages {
            anyhow::bail!(
                "failed to execute provider request: anthropic provider requires anthropic_messages api kind"
            );
        }
        info!(
            provider = "anthropic",
            session_id = %request.session_id,
            turn_id = %request.turn_id,
            model = %request.model.model,
            message_count = request.messages.len(),
            tool_count = request.tools.len(),
            "starting anthropic request"
        );

        let body = anthropic_codec::encode_stream_request(&request, self.temperature)?;
        let enable_interleaved_thinking =
            request.model.reasoning.is_some() && !request.tools.is_empty();
        let raw_stream = self
            .client
            .post_json_event_stream(
                JsonRequest {
                    provider_label: "anthropic",
                    url: provider_url(&self.base_url, ANTHROPIC_MESSAGES_PATH),
                    headers: self
                        .header_overrides
                        .merge_string_pairs(self.default_headers(enable_interleaved_thinking)),
                    body,
                },
                cancel,
            )
            .await?;
        let mut decoder = anthropic_codec::AnthropicStreamDecoder::new(&request);
        Ok(raw_stream
            .flat_map(move |item| {
                let events = match item {
                    Ok(event) => decoder.decode(&event).unwrap_or_else(|error| {
                        vec![StreamEvent::Error {
                            error: ProviderError::new(
                                format!("failed to decode anthropic stream: {error}"),
                                false,
                            ),
                        }]
                    }),
                    Err(error) => vec![StreamEvent::Error {
                        error: ProviderError::new(error.to_string(), false),
                    }],
                };
                debug!(event_count = events.len(), "decoded anthropic stream event");
                stream::iter(events.into_iter().map(Ok))
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
                    url: provider_url(&self.base_url, ANTHROPIC_MESSAGES_PATH),
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

impl AnthropicProvider {
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

fn provider_url(base_url: &str, path: &str) -> String {
    format!("{}{}", base_url.trim_end_matches('/'), path)
}

#[cfg(test)]
mod tests {
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
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;

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

    async fn spawn_stream_server(captured: std::sync::Arc<tokio::sync::Mutex<Vec<u8>>>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind listener");
        let address = listener.local_addr().expect("local addr");
        tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept socket");
            let request = read_http_request(&mut socket).await.expect("read request");
            *captured.lock().await = request;

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

    async fn read_http_request(socket: &mut tokio::net::TcpStream) -> anyhow::Result<Vec<u8>> {
        let mut request = Vec::new();
        let mut buf = [0_u8; 1024];
        loop {
            let read = socket.read(&mut buf).await?;
            if read == 0 {
                break;
            }
            request.extend_from_slice(&buf[..read]);
            if let Some(headers_end) = find_headers_end(&request) {
                let content_length = content_length(&request[..headers_end]).unwrap_or_default();
                if request.len() >= headers_end + 4 + content_length {
                    break;
                }
            }
        }
        Ok(request)
    }

    fn find_headers_end(bytes: &[u8]) -> Option<usize> {
        bytes.windows(4).position(|window| window == b"\r\n\r\n")
    }

    fn content_length(headers: &[u8]) -> Option<usize> {
        let text = String::from_utf8_lossy(headers);
        text.lines().find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse().ok())
                .flatten()
        })
    }
}
