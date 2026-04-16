// pattern: Imperative Shell

use async_openai::error::OpenAIError;
use async_trait::async_trait;
use futures::{StreamExt, channel::mpsc, stream::BoxStream};
use halter_protocol::{
    ApiKind, ProviderCapabilities, ProviderCompactionRequest, ProviderCompactionResponse,
    ProviderError, ProviderKind, ProviderRequest, StreamEvent,
};
use tokio::select;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::Provider;
use crate::openai_codec::{self, ResponsesRequestOptions};
use crate::openai_error::{
    openai_api_error_is_rate_limit, openai_message_is_rate_limit, parse_openai_stream_error,
};
use crate::openai_rate_limit_policy::estimate_openai_request_cost;
use crate::responses_transport::{
    ResponsesRateLimitStrategy, ResponsesTransport, ResponsesTransportRequest,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompactStrategy {
    DedicatedEndpoint,
    InlineResponses,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct ResponsesProviderRequestConfig {
    pub store: Option<bool>,
    pub include_prompt_cache_key: bool,
    pub include_encrypted_reasoning: bool,
    pub reasoning_summary: Option<&'static str>,
}

#[derive(Debug, Clone)]
pub(crate) struct ResponsesProviderConfig {
    pub label: &'static str,
    pub capabilities: ProviderCapabilities,
    pub request: ResponsesProviderRequestConfig,
    pub compact_strategy: Option<CompactStrategy>,
    pub rate_limit_strategy: Option<ResponsesRateLimitStrategy>,
}

#[derive(Debug, Clone)]
pub struct ResponsesProvider {
    config: ResponsesProviderConfig,
    transport: ResponsesTransport,
}

impl ResponsesProvider {
    #[must_use]
    pub(crate) fn new(
        config: ResponsesProviderConfig,
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            config,
            transport: ResponsesTransport::new(api_key, base_url),
        }
    }

    /// Construct a `ResponsesProvider` for OpenAI (platform.openai.com).
    #[must_use]
    pub fn openai(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self::new(config_for(ProviderKind::OpenAi), api_key, base_url)
    }

    /// Construct a `ResponsesProvider` for OpenRouter (openrouter.ai).
    #[must_use]
    pub fn openrouter(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self::new(config_for(ProviderKind::OpenRouter), api_key, base_url)
    }
}

#[async_trait]
impl Provider for ResponsesProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        let mut capabilities = self.config.capabilities.clone();
        capabilities.supports_compaction = self.config.compact_strategy.is_some();
        capabilities
    }

    async fn stream(
        &self,
        request: ProviderRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
        validate_responses_request(self.config.label, &request)?;
        info!(
            provider = self.config.label,
            session_id = %request.session_id,
            turn_id = %request.turn_id,
            model = %request.model.model,
            message_count = request.messages.len(),
            tool_count = request.tools.len(),
            "starting responses request"
        );

        let request_body = openai_codec::encode_responses_request(
            &request,
            ResponsesRequestOptions {
                stream: true,
                store: self.config.request.store,
                prompt_cache_key: self
                    .config
                    .request
                    .include_prompt_cache_key
                    .then_some(request.prompt.prefix_cache_key.as_str()),
                include_encrypted_reasoning: self.config.request.include_encrypted_reasoning,
                reasoning_summary: self.config.request.reasoning_summary,
            },
        )?;
        let request_bytes = request_body.to_string().len();
        debug!(
            provider = self.config.label,
            request_bytes, "encoded responses request"
        );
        let request_meta = ResponsesTransportRequest {
            provider_label: self.config.label,
            model: request.model.model.clone(),
            reservation: estimate_openai_request_cost(
                request_bytes,
                request.model.max_output_tokens,
            ),
            rate_limit_strategy: self.config.rate_limit_strategy,
            tokens_per_minute: request.model.tokens_per_minute,
        };
        let track_response_id = self.config.request.store != Some(false);
        let (tx, rx) = mpsc::unbounded();
        let provider_label = self.config.label;
        let transport = self.transport.clone();

        tokio::spawn(async move {
            loop {
                let mut response_stream = match transport
                    .responses_stream(
                        request_body.clone(),
                        request_meta.clone(),
                        cancel.child_token(),
                    )
                    .await
                {
                    Ok(stream) => stream,
                    Err(error) => {
                        let message = error.to_string();
                        if openai_message_is_rate_limit(&message) {
                            warn!(
                                provider = provider_label,
                                error = %message,
                                "retrying responses request after rate limit"
                            );
                            continue;
                        }
                        let _ = tx.unbounded_send(Err(provider_error_from_transport_error(error)));
                        return;
                    }
                };
                let mut decoder =
                    openai_codec::ResponsesStreamDecoder::new(&request, track_response_id);
                let mut pending_events = Vec::new();
                let mut committed = false;

                loop {
                    select! {
                        _ = cancel.cancelled() => {
                            warn!(provider = provider_label, "responses request cancelled");
                            let _ = tx.unbounded_send(Err(ProviderError::new(
                                "failed to execute provider request: request cancelled",
                                false,
                            )));
                            return;
                        }
                        item = response_stream.next() => {
                            match item {
                                Some(Ok(event)) => match decoder.decode(event) {
                                    Ok(events) => {
                                        if committed {
                                            for event in events {
                                                if tx.unbounded_send(Ok(event)).is_err() {
                                                    return;
                                                }
                                            }
                                            continue;
                                        }

                                        pending_events.extend(events);
                                        if pending_events.iter().any(stream_event_commits_attempt) {
                                            committed = true;
                                            for event in pending_events.drain(..) {
                                                if tx.unbounded_send(Ok(event)).is_err() {
                                                    return;
                                                }
                                            }
                                        }
                                    }
                                    Err(error) => {
                                        error!(provider = provider_label, error = %error, "failed to decode responses stream");
                                        let _ = tx.unbounded_send(Err(ProviderError::new(error.to_string(), false)));
                                        return;
                                    }
                                },
                                Some(Err(error)) => {
                                    if !committed && stream_error_is_retryable(&error) {
                                        warn!(
                                            provider = provider_label,
                                            error = %error,
                                            "retrying responses stream after rate limit"
                                        );
                                        break;
                                    }
                                    warn!(provider = provider_label, error = %error, "responses stream returned provider error");
                                    let _ = tx.unbounded_send(Err(provider_error_from_stream_error(error)));
                                    return;
                                }
                                None => {
                                    if !pending_events.is_empty() {
                                        for event in pending_events.drain(..) {
                                            if tx.unbounded_send(Ok(event)).is_err() {
                                                return;
                                            }
                                        }
                                    }
                                    debug!(provider = provider_label, "responses stream completed");
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        });

        Ok(rx.boxed())
    }

    async fn compact(
        &self,
        request: ProviderCompactionRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<ProviderCompactionResponse> {
        validate_responses_compaction_request(self.config.label, &self.config, &request)?;
        info!(
            provider = self.config.label,
            session_id = %request.session_id,
            model = %request.model.model,
            compacted_prefix_items = request.compacted_prefix.len(),
            message_count = request.messages.len(),
            "starting responses compaction request"
        );

        match self.config.compact_strategy {
            Some(CompactStrategy::DedicatedEndpoint) => {
                self.compact_via_endpoint(&request, cancel).await
            }
            Some(CompactStrategy::InlineResponses) => {
                self.compact_via_responses(&request, cancel).await
            }
            None => anyhow::bail!(
                "failed to compact session: {} provider does not support compaction",
                self.config.label
            ),
        }
    }
}

impl ResponsesProvider {
    async fn compact_via_endpoint(
        &self,
        request: &ProviderCompactionRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<ProviderCompactionResponse> {
        let request_body = openai_codec::encode_responses_compact_request(request)?;
        let request_bytes = request_body.to_string().len();
        let response = self
            .transport
            .responses_compact(
                request_body,
                self.compaction_transport_request(request, request_bytes),
                cancel,
            )
            .await?;
        openai_codec::decode_responses_compact_response(&response)
    }

    async fn compact_via_responses(
        &self,
        request: &ProviderCompactionRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<ProviderCompactionResponse> {
        let request_body = openai_codec::encode_openrouter_compact_request(request)?;
        let request_bytes = request_body.to_string().len();
        let response = self
            .transport
            .responses_json(
                request_body,
                self.compaction_transport_request(request, request_bytes),
                cancel,
            )
            .await?;
        openai_codec::decode_openrouter_compact_response(&response)
    }

    fn compaction_transport_request(
        &self,
        request: &ProviderCompactionRequest,
        request_bytes: usize,
    ) -> ResponsesTransportRequest {
        ResponsesTransportRequest {
            provider_label: self.config.label,
            model: request.model.model.clone(),
            reservation: estimate_openai_request_cost(
                request_bytes,
                request.model.max_output_tokens,
            ),
            rate_limit_strategy: self.config.rate_limit_strategy,
            tokens_per_minute: request.model.tokens_per_minute,
        }
    }
}

fn validate_responses_request(label: &str, request: &ProviderRequest) -> anyhow::Result<()> {
    if request.model.api_kind() != ApiKind::OpenAiResponses {
        anyhow::bail!(
            "failed to execute provider request: {label} provider requires openai_responses api kind"
        );
    }

    Ok(())
}

fn validate_responses_compaction_request(
    label: &str,
    config: &ResponsesProviderConfig,
    request: &ProviderCompactionRequest,
) -> anyhow::Result<()> {
    if request.model.api_kind() != ApiKind::OpenAiResponses {
        anyhow::bail!(
            "failed to compact session: {label} provider requires openai_responses api kind"
        );
    }
    if config.compact_strategy.is_none() {
        anyhow::bail!("failed to compact session: {label} provider does not support compaction");
    }

    Ok(())
}

fn config_for(kind: ProviderKind) -> ResponsesProviderConfig {
    match kind {
        ProviderKind::OpenAi => ResponsesProviderConfig {
            label: "openai",
            capabilities: ProviderCapabilities::for_provider(ProviderKind::OpenAi),
            request: ResponsesProviderRequestConfig {
                store: None,
                include_prompt_cache_key: true,
                include_encrypted_reasoning: true,
                reasoning_summary: Some("auto"),
            },
            compact_strategy: Some(CompactStrategy::DedicatedEndpoint),
            rate_limit_strategy: Some(ResponsesRateLimitStrategy::OpenAiHeaders),
        },
        ProviderKind::OpenRouter => ResponsesProviderConfig {
            label: "openrouter",
            capabilities: ProviderCapabilities::for_provider(ProviderKind::OpenRouter),
            request: ResponsesProviderRequestConfig {
                store: Some(false),
                include_prompt_cache_key: true,
                include_encrypted_reasoning: false,
                reasoning_summary: Some("auto"),
            },
            compact_strategy: Some(CompactStrategy::InlineResponses),
            rate_limit_strategy: None,
        },
        ProviderKind::Anthropic | ProviderKind::Fake => {
            panic!("ResponsesProvider does not support {kind:?}")
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use chrono::Utc;
    use futures::StreamExt;
    use halter_protocol::{
        AssembledPrompt, AssistantMessage, AssistantPart, Message, MessageId, ModelId, ModelRole,
        ProviderName, ResolvedModel, SessionId, ToolCall, ToolCallId, ToolCapabilities,
        ToolConcurrency, ToolResult, ToolResultMessage, ToolSpec, TurnId, UserMessage,
    };
    use indexmap::IndexMap;
    use serde_json::{Value, json};
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;

    use super::*;
    use crate::test_http::{find_headers_end, read_http_request};

    #[tokio::test]
    async fn responses_provider_without_compaction_strategy_rejects_compaction() {
        let provider = ResponsesProvider::new(
            ResponsesProviderConfig {
                label: "responses-test",
                capabilities: ProviderCapabilities {
                    supports_compaction: true,
                    ..ProviderCapabilities::default()
                },
                request: ResponsesProviderRequestConfig {
                    store: None,
                    include_prompt_cache_key: false,
                    include_encrypted_reasoning: false,
                    reasoning_summary: None,
                },
                compact_strategy: None,
                rate_limit_strategy: None,
            },
            "test-key",
            "http://127.0.0.1:1",
        );

        let error = provider
            .compact(
                sample_compaction_request(ProviderKind::OpenAi, "responses-test"),
                CancellationToken::new(),
            )
            .await
            .expect_err("compaction should fail without a strategy");

        assert!(
            error
                .to_string()
                .contains("responses-test provider does not support compaction")
        );
        assert!(!provider.capabilities().supports_compaction);
    }

    #[tokio::test]
    async fn openai_provider_retries_streamed_token_rate_limits() {
        let request_times = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let base_url = spawn_retrying_stream_server(request_times.clone()).await;
        let provider = ResponsesProvider::openai("test-key", base_url);
        let mut stream = provider
            .stream(
                sample_request(ProviderKind::OpenAi),
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

    #[test]
    fn openrouter_provider_reports_compaction_and_prompt_cache_support() {
        let capabilities =
            ResponsesProvider::openrouter("test-key", "https://openrouter.ai/api").capabilities();

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

        let provider = ResponsesProvider::openrouter("test-key", format!("http://{address}"));
        let response = provider
            .compact(
                sample_openrouter_compaction_request(),
                CancellationToken::new(),
            )
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
                        "text": "[Compacted context]\n\n## User Intent\nShip compaction support"
                    }
                ],
            })]
        );

        server.await.expect("server task");
    }

    fn sample_request(kind: ProviderKind) -> ProviderRequest {
        ProviderRequest {
            session_id: SessionId::new(),
            turn_id: TurnId::new(),
            model: ResolvedModel {
                role: ModelRole::default(),
                id: ModelId::from("default"),
                provider: ProviderName::from(match kind {
                    ProviderKind::OpenAi => "openai",
                    ProviderKind::OpenRouter => "openrouter",
                    other => panic!("unsupported provider kind in test: {other:?}"),
                }),
                provider_kind: kind,
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

    fn sample_compaction_request(kind: ProviderKind, provider: &str) -> ProviderCompactionRequest {
        ProviderCompactionRequest {
            session_id: SessionId::new(),
            model: ResolvedModel {
                role: ModelRole::default(),
                id: ModelId::from("default"),
                provider: ProviderName::from(provider),
                provider_kind: kind,
                model: "gpt-5".to_owned(),
                max_input_tokens: Some(200_000),
                max_output_tokens: Some(8_192),
                reasoning: None,
                tokens_per_minute: None,
            },
            compacted_prefix: Vec::new(),
            messages: Vec::new(),
            tools: Vec::new(),
            instructions: "Summarize".to_owned(),
        }
    }

    fn sample_openrouter_compaction_request() -> ProviderCompactionRequest {
        let mut provider_aliases = IndexMap::new();
        provider_aliases.insert(ProviderKind::OpenRouter, "fs_read".into());
        ProviderCompactionRequest {
            session_id: SessionId::new(),
            model: ResolvedModel {
                role: ModelRole::default(),
                id: ModelId::from("default"),
                provider: ProviderName::from("openrouter"),
                provider_kind: ProviderKind::OpenRouter,
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
}

fn provider_error_from_stream_error(error: OpenAIError) -> ProviderError {
    match error {
        OpenAIError::ApiError(api_error) => ProviderError::new(
            format!("failed to execute provider request: {}", api_error.message),
            openai_api_error_is_rate_limit(&api_error),
        ),
        OpenAIError::JSONDeserialize(json_error, content) => {
            if let Some(api_error) = parse_openai_stream_error(&content) {
                return ProviderError::new(
                    format!("failed to execute provider request: {}", api_error.message),
                    openai_api_error_is_rate_limit(&api_error),
                );
            }
            ProviderError::new(
                format!(
                    "failed to execute provider request: failed to deserialize api response: error:{json_error} content:{content}"
                ),
                false,
            )
        }
        other => {
            let retryable = matches!(other, OpenAIError::Reqwest(_) | OpenAIError::StreamError(_));
            ProviderError::new(
                format!("failed to execute provider request: {other}"),
                retryable,
            )
        }
    }
}

fn provider_error_from_transport_error(error: anyhow::Error) -> ProviderError {
    let message = error.to_string();
    ProviderError::new(message.clone(), openai_message_is_rate_limit(&message))
}

fn stream_error_is_retryable(error: &OpenAIError) -> bool {
    match error {
        OpenAIError::ApiError(api_error) => openai_api_error_is_rate_limit(api_error),
        OpenAIError::JSONDeserialize(_, content) => parse_openai_stream_error(content)
            .as_ref()
            .is_some_and(openai_api_error_is_rate_limit),
        OpenAIError::Reqwest(_)
        | OpenAIError::StreamError(_)
        | OpenAIError::FileSaveError(_)
        | OpenAIError::FileReadError(_)
        | OpenAIError::InvalidArgument(_) => false,
    }
}

fn stream_event_commits_attempt(event: &StreamEvent) -> bool {
    matches!(
        event,
        StreamEvent::TextDelta { .. }
            | StreamEvent::ThinkingDelta { .. }
            | StreamEvent::ToolCallStart { .. }
            | StreamEvent::ToolArgsDelta { .. }
            | StreamEvent::ToolCallEnd { .. }
            | StreamEvent::MessageEnd { .. }
            | StreamEvent::ProviderWarning { .. }
            | StreamEvent::Error { .. }
    )
}
