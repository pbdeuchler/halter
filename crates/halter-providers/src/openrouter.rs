// pattern: Imperative Shell

use async_trait::async_trait;
use futures::stream::BoxStream;
use halter_protocol::{
    DEFAULT_TEMPERATURE, ProviderCapabilities, ProviderCompactionRequest,
    ProviderCompactionResponse, ProviderError, ProviderRequest, StreamEvent, ToolCallIdPolicy,
};
use tokio_util::sync::CancellationToken;

use crate::Provider;
use crate::responses_provider::{
    CompactStrategy, ResponsesProvider, ResponsesProviderConfig, ResponsesProviderRequestConfig,
};
use crate::retry::RetryPolicy;
use crate::secret::SecretString;

#[derive(Debug, Clone)]
pub struct OpenRouterProvider {
    inner: ResponsesProvider,
}

impl OpenRouterProvider {
    pub fn new(
        api_key: impl Into<SecretString>,
        base_url: impl Into<String>,
    ) -> anyhow::Result<Self> {
        Self::new_with_headers(api_key, base_url, &[], DEFAULT_TEMPERATURE)
    }

    /// Construct an OpenRouter provider with user-configured HTTP header
    /// overrides. Overrides replace any default or hardcoded header
    /// (`Authorization`, `Content-Type`) case-insensitively. `temperature`
    /// is forwarded verbatim to every request body; callers typically pull
    /// it from the resolved provider config.
    pub fn new_with_headers(
        api_key: impl Into<SecretString>,
        base_url: impl Into<String>,
        header_overrides: &[(String, String)],
        temperature: f32,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            inner: ResponsesProvider::try_new(
                config(),
                api_key,
                base_url,
                header_overrides,
                temperature,
            )?,
        })
    }
}

#[async_trait]
impl Provider for OpenRouterProvider {
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
        label: "openrouter",
        capabilities: ProviderCapabilities {
            supports_tools: true,
            supports_streaming: true,
            supports_reasoning: true,
            supports_interleaved_reasoning: false,
            supports_images: true,
            supports_documents: false,
            supports_prompt_cache: true,
            supports_compaction: true,
            compaction_strategy: None,
            supports_tool_result_media: false,
            requires_non_empty_assistant_content: false,
            tool_call_id_policy: ToolCallIdPolicy::ProviderSupplied,
            max_input_tokens: 200_000,
            max_output_tokens: 8_192,
        },
        request: ResponsesProviderRequestConfig {
            store: Some(false),
            include_prompt_cache_key: true,
            include_encrypted_reasoning: false,
            reasoning_summary: Some("auto"),
        },
        compact_strategy: Some(CompactStrategy::InlineResponses),
        rate_limit_strategy: None,
        retry_policy: RetryPolicy::default(),
    }
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use halter_protocol::{
        ApiKind, AssembledPrompt, AssistantMessage, AssistantPart, Message, MessageId, ModelId,
        ModelRole, ProviderKind, ProviderName, ProviderRequest, ResolvedModel, SessionId, ToolCall,
        ToolCallId, ToolCapabilities, ToolConcurrency, ToolResult, ToolResultMessage, ToolSpec,
        TurnId, UserMessage,
    };
    use indexmap::IndexMap;
    use serde_json::{Value, json};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;
    use crate::Provider;

    #[tokio::test]
    async fn openrouter_provider_rejects_chat_api_kind() {
        let provider = OpenRouterProvider::new("test-key", "https://openrouter.ai/api")
            .expect("openrouter provider");
        let error = match provider
            .stream(
                sample_request(ApiKind::OpenAiChat),
                CancellationToken::new(),
            )
            .await
        {
            Ok(_) => panic!("openrouter provider should reject chat requests"),
            Err(error) => error,
        };

        assert!(
            error
                .to_string()
                .contains("openrouter provider requires openai_responses api kind")
        );
    }

    #[test]
    fn openrouter_provider_reports_compaction_and_prompt_cache_support() {
        let capabilities = OpenRouterProvider::new("test-key", "https://openrouter.ai/api")
            .expect("openrouter provider")
            .capabilities();

        assert!(capabilities.supports_prompt_cache);
        assert!(capabilities.supports_compaction);
    }

    #[tokio::test]
    async fn openrouter_provider_compacts_via_inline_responses_request() {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind listener");
        let address = listener.local_addr().expect("local addr");
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.expect("accept socket");
            let request = read_http_request(&mut socket).await.expect("read request");

            let headers_end = find_headers_end(&request).expect("headers end");
            let request_text = String::from_utf8_lossy(&request[..headers_end]);
            assert!(request_text.starts_with("POST /v1/responses HTTP/1.1"));

            let body: Value =
                serde_json::from_slice(&request[headers_end + 4..]).expect("parse request body");
            assert_eq!(body["model"], "gpt-5");
            assert_eq!(body["instructions"], "Summarize the session");
            assert_eq!(body["stream"], false);
            assert_eq!(body["store"], false);
            assert!(body.get("tools").is_none());
            assert!(
                body["input"]
                    .as_array()
                    .expect("input")
                    .iter()
                    .any(|item| item["role"] == "developer")
            );
            assert!(
                body["input"]
                    .as_array()
                    .expect("input")
                    .iter()
                    .any(|item| item["type"] == "function_call")
            );

            let response_body = json!({
                "id": "resp_openrouter_compaction",
                "output": [
                    {
                        "type": "reasoning",
                        "summary": [{"text": "ignore"}]
                    },
                    {
                        "type": "message",
                        "id": "msg_summary",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": "## User Intent\nShip compaction support",
                                "annotations": []
                            }
                        ]
                    }
                ],
                "usage": {
                    "input_tokens": 42,
                    "output_tokens": 13,
                    "input_tokens_details": {
                        "cache_creation_tokens": 0,
                        "cached_tokens": 0
                    }
                }
            })
            .to_string();
            let response = format!(
                "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                response_body.len(),
                response_body
            );
            socket
                .write_all(response.as_bytes())
                .await
                .expect("write response");
        });

        let provider = OpenRouterProvider::new("test-key", format!("http://{address}"))
            .expect("openrouter provider");
        let response = provider
            .compact(sample_compaction_request(), CancellationToken::new())
            .await
            .expect("provider compaction");

        assert_eq!(response.usage.input_tokens, 42);
        assert_eq!(
            response.output,
            vec![json!({
                "type": "message",
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": "<compaction>\n## User Intent\nShip compaction support\n</compaction>"
                    }
                ],
            })]
        );

        server.await.expect("server task");
    }

    fn sample_request(api_kind: ApiKind) -> ProviderRequest {
        ProviderRequest {
            session_id: SessionId::new(),
            turn_id: TurnId::new(),
            model: ResolvedModel {
                role: ModelRole::default(),
                id: ModelId::from("default"),
                provider: ProviderName::from("openrouter"),
                provider_kind: ProviderKind::OpenRouter,
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
                cache_breakpoints: halter_protocol::CacheBreakpoints::default(),
                system_segment_count: 0,
                skill_segment_count: 0,
            },
            compacted_prefix: Vec::new(),
            messages: Vec::new(),
            tools: Vec::new(),
            previous_response_id: None,
            new_messages_start: 0,
        }
    }

    fn sample_compaction_request() -> ProviderCompactionRequest {
        let mut provider_aliases = IndexMap::new();
        provider_aliases.insert(ProviderKind::OpenRouter, "fs_read".into());
        ProviderCompactionRequest {
            session_id: SessionId::new(),
            model: ResolvedModel {
                role: ModelRole::default(),
                id: ModelId::from("default"),
                provider: ProviderName::from("openrouter"),
                provider_kind: ProviderKind::OpenRouter,
                api_kind: ApiKind::OpenAiResponses,
                model: "gpt-5".to_owned(),
                max_input_tokens: Some(200_000),
                max_output_tokens: Some(8_192),
                reasoning: None,
                tokens_per_minute: None,
            },
            compacted_prefix: vec![json!({
                "type": "message",
                "role": "developer",
                "content": [
                    {
                        "type": "input_text",
                        "text": "[Compacted context]\n\nEarlier summary"
                    }
                ],
            })],
            messages: vec![
                Message::User(UserMessage::text("ship compaction support")),
                Message::Assistant(AssistantMessage {
                    id: MessageId::from("msg_history_1"),
                    created_at: Utc::now(),
                    parts: vec![
                        AssistantPart::Text {
                            text: "Investigating".to_owned(),
                        },
                        AssistantPart::ToolCall(ToolCall {
                            id: ToolCallId::from("call_123"),
                            name: "read".into(),
                            arguments: json!({"path": "docs/openrouter-compaction-design.md"}),
                        }),
                    ],
                    stop_reason: None,
                    usage: None,
                    replay_meta: Default::default(),
                }),
                Message::Tool(ToolResultMessage {
                    id: MessageId::from("tool_output_1"),
                    call_id: ToolCallId::from("call_123"),
                    content: ToolResult::Json {
                        value: json!({"ok": true}),
                    },
                    error: None,
                    created_at: Utc::now(),
                }),
            ],
            tools: vec![ToolSpec {
                name: "read".into(),
                description: "Read a file".to_owned(),
                input_schema: json!({"type": "object"}),
                concurrency: ToolConcurrency::ReadOnly,
                capabilities: ToolCapabilities::default(),
                provider_aliases,
            }],
            instructions: "Summarize the session".to_owned(),
        }
    }

    async fn read_http_request(socket: &mut tokio::net::TcpStream) -> anyhow::Result<Vec<u8>> {
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

    fn find_headers_end(buffer: &[u8]) -> Option<usize> {
        buffer.windows(4).position(|window| window == b"\r\n\r\n")
    }
}
