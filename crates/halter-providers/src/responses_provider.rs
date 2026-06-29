// pattern: Imperative Shell

use futures::{
    StreamExt,
    stream::{self, BoxStream},
};
use halter_protocol::{
    ApiKind, CompactionWindow, Message, ProviderCapabilities, ProviderCompactionRequest,
    ProviderCompactionResponse, ProviderCompactionStrategy, ProviderError, ProviderErrorKind,
    ProviderRequest, StreamEvent,
};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info};

use crate::Provider;
use crate::openai_codec::{self, ResponsesInstructionMode, ResponsesRequestOptions};
use crate::openai_error::classify;
use crate::openai_rate_limit_policy::estimate_openai_request_cost;
use crate::resilience::ResiliencePolicy;
use crate::responses_transport::{
    ResponsesEndpointMode, ResponsesRateLimitStrategy, ResponsesTransport,
    ResponsesTransportRequest, TransportError,
};
use crate::secret::SecretString;

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
    pub instruction_mode: ResponsesInstructionMode,
}

#[derive(Debug, Clone)]
pub(crate) struct ResponsesProviderConfig {
    pub label: &'static str,
    pub capabilities: ProviderCapabilities,
    pub request: ResponsesProviderRequestConfig,
    pub compact_strategy: Option<CompactStrategy>,
    pub rate_limit_strategy: Option<ResponsesRateLimitStrategy>,
    /// Resilience policy used by the outer provider decorator and the raw
    /// transport's HTTP client.
    pub resilience_policy: ResiliencePolicy,
}

#[derive(Debug, Clone)]
pub(crate) struct ResponsesProvider {
    config: ResponsesProviderConfig,
    transport: ResponsesTransport,
    temperature: Option<f32>,
}

impl ResponsesProvider {
    pub(crate) fn try_new(
        config: ResponsesProviderConfig,
        bearer_token: impl Into<SecretString>,
        base_url: impl Into<String>,
        header_overrides: &[(String, String)],
        temperature: Option<f32>,
    ) -> anyhow::Result<Self> {
        let timeouts = config.resilience_policy.timeouts;
        Ok(Self {
            transport: ResponsesTransport::try_new_with_timeouts(
                bearer_token,
                base_url,
                header_overrides,
                timeouts,
            )?,
            config,
            temperature,
        })
    }

    pub(crate) fn try_new_with_endpoint_mode(
        config: ResponsesProviderConfig,
        bearer_token: impl Into<SecretString>,
        base_url: impl Into<String>,
        header_overrides: &[(String, String)],
        temperature: Option<f32>,
        endpoint_mode: ResponsesEndpointMode,
    ) -> anyhow::Result<Self> {
        let timeouts = config.resilience_policy.timeouts;
        Ok(Self {
            transport: ResponsesTransport::try_new_with_endpoint_mode(
                bearer_token,
                base_url,
                header_overrides,
                endpoint_mode,
                timeouts,
            )?,
            config,
            temperature,
        })
    }

    #[must_use]
    pub(crate) fn capabilities(&self) -> ProviderCapabilities {
        let mut capabilities = self.config.capabilities.clone();
        capabilities.supports_compaction = self.config.compact_strategy.is_some();
        capabilities.compaction_strategy =
            self.config.compact_strategy.map(|strategy| match strategy {
                CompactStrategy::DedicatedEndpoint => ProviderCompactionStrategy::Dedicated,
                CompactStrategy::InlineResponses => ProviderCompactionStrategy::Inline,
            });
        capabilities
    }

    pub(crate) fn compaction_window(&self, messages: &[Message]) -> Option<CompactionWindow> {
        match self.config.compact_strategy {
            Some(CompactStrategy::DedicatedEndpoint) => Some(
                CompactionWindow::preserve_latest_assistant_response_block(messages),
            ),
            Some(CompactStrategy::InlineResponses) => {
                Some(CompactionWindow::preserve_through_latest_user(messages))
            }
            None => None,
        }
    }

    pub(crate) async fn stream(
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
                instruction_mode: self.config.request.instruction_mode,
                temperature: self.temperature,
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
        let response_stream = match self
            .transport
            .responses_stream(request_body, request_meta, cancel)
            .await
        {
            Ok(stream) => stream,
            Err(error) => {
                return Ok(stream::iter([Err(provider_error_from_transport(error))]).boxed());
            }
        };
        let provider_label = self.config.label;
        let mut decoder = openai_codec::ResponsesStreamDecoder::new(&request, track_response_id);
        Ok(response_stream
            .flat_map(move |item| {
                let events = match item {
                    Ok(event) => match decoder.decode(event) {
                        Ok(events) => events.into_iter().map(Ok).collect::<Vec<_>>(),
                        Err(error) => {
                            error!(
                                provider = provider_label,
                                error = %error,
                                "failed to decode responses stream"
                            );
                            vec![Err(ProviderError::with_kind(
                                error.to_string(),
                                ProviderErrorKind::Fatal,
                            ))]
                        }
                    },
                    Err(error) => vec![Err(provider_error_from_openai(error))],
                };
                stream::iter(events)
            })
            .boxed())
    }

    pub(crate) async fn compact(
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

#[async_trait::async_trait]
impl Provider for ResponsesProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        Self::capabilities(self)
    }

    fn compaction_window(&self, messages: &[Message]) -> Option<CompactionWindow> {
        Self::compaction_window(self, messages)
    }

    async fn stream(
        &self,
        request: ProviderRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
        Self::stream(self, request, cancel).await
    }

    async fn compact(
        &self,
        request: ProviderCompactionRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<ProviderCompactionResponse> {
        Self::compact(self, request, cancel).await
    }
}

fn validate_responses_request(label: &str, request: &ProviderRequest) -> anyhow::Result<()> {
    if request.model.api_kind != ApiKind::OpenAiResponses {
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
    if request.model.api_kind != ApiKind::OpenAiResponses {
        anyhow::bail!(
            "failed to compact session: {label} provider requires openai_responses api kind"
        );
    }
    if config.compact_strategy.is_none() {
        anyhow::bail!("failed to compact session: {label} provider does not support compaction");
    }

    Ok(())
}

fn provider_error_from_transport(error: TransportError) -> ProviderError {
    match error {
        TransportError::Cancelled => ProviderError::cancelled(),
        TransportError::Retryable {
            source,
            backoff_hint,
        } => provider_error_from_openai(source).with_backoff_hint(backoff_hint),
        TransportError::Fatal { source } => provider_error_from_openai(source),
    }
}

/// Convert an `OpenAIError` to a typed `ProviderError`. Centralizing the
/// formatting here ensures every error string has the same prefix.
fn provider_error_from_openai(error: async_openai::error::OpenAIError) -> ProviderError {
    let retryability = classify(&error);
    let message = match &error {
        async_openai::error::OpenAIError::ApiError(api_error) => {
            format!("failed to execute provider request: {}", api_error.message)
        }
        async_openai::error::OpenAIError::JSONDeserialize(json_error, content) => format!(
            "failed to execute provider request: failed to deserialize api response: error:{json_error} content:{content}"
        ),
        other => format!("failed to execute provider request: {other}"),
    };
    ProviderError::with_kind(message, retryability.kind)
        .with_backoff_hint(retryability.backoff_hint)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::Duration;

    use futures::StreamExt;
    use halter_protocol::{
        AssembledPrompt, ModelId, ModelRole, ProviderKind, ProviderName, ResolvedModel, SessionId,
        TurnId,
    };
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;
    use crate::{ResilientProvider, RetryPolicy};

    #[tokio::test]
    async fn responses_provider_without_compaction_strategy_rejects_compaction() {
        let provider = ResponsesProvider::try_new(
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
                    instruction_mode: ResponsesInstructionMode::DeveloperMessage,
                },
                compact_strategy: None,
                rate_limit_strategy: None,
                resilience_policy: ResiliencePolicy::default(),
            },
            "test-key",
            "http://127.0.0.1:1",
            &[],
            None,
        )
        .expect("responses provider");

        let error = provider
            .compact(sample_compaction_request(), CancellationToken::new())
            .await
            .expect_err("compaction should fail without a strategy");

        assert!(
            error
                .to_string()
                .contains("responses-test provider does not support compaction")
        );
        assert!(!provider.capabilities().supports_compaction);
    }

    fn sample_compaction_request() -> ProviderCompactionRequest {
        ProviderCompactionRequest {
            session_id: SessionId::new(),
            model: ResolvedModel {
                role: ModelRole::default(),
                id: ModelId::from("default"),
                provider: ProviderName::from("responses-test"),
                provider_kind: ProviderKind::OpenAi,
                api_kind: ApiKind::OpenAiResponses,
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

    /// Contract test: the unified `classify` and the resulting
    /// `provider_error_from_openai` must agree with the previous
    /// hand-written `provider_error_from_stream_error` /
    /// `stream_error_is_retryable` pair on the retryability of canonical
    /// errors. Locks AC3.7 so subsequent refactors of the classifier do not
    /// silently regress the field semantics consumers depend on.
    #[test]
    fn provider_error_retryability_matches_classify() {
        use async_openai::error::{ApiError, OpenAIError, StreamError};

        let cases: Vec<(&str, OpenAIError, bool)> = vec![
            (
                "rate-limit api error",
                OpenAIError::ApiError(ApiError {
                    message: "rate limit reached".to_owned(),
                    r#type: Some("tokens".to_owned()),
                    param: None,
                    code: Some("rate_limit_exceeded".to_owned()),
                }),
                true,
            ),
            (
                "synthetic 5xx",
                OpenAIError::ApiError(ApiError {
                    message: "Internal Server Error".to_owned(),
                    r#type: None,
                    param: None,
                    code: Some(crate::openai_error::SYNTHETIC_SERVER_ERROR_CODE.to_owned()),
                }),
                true,
            ),
            (
                "client error",
                OpenAIError::ApiError(ApiError {
                    message: "missing required parameter".to_owned(),
                    r#type: Some("invalid_request".to_owned()),
                    param: None,
                    code: Some("invalid_request_error".to_owned()),
                }),
                false,
            ),
            (
                "stream framing",
                OpenAIError::StreamError(Box::new(StreamError::EventStream("eof".to_owned()))),
                true,
            ),
        ];

        for (label, error, expected_retryable) in cases {
            let retryability = classify(&error);
            let provider_error = provider_error_from_openai(error);
            assert_eq!(
                retryability.is_retryable(),
                expected_retryable,
                "{label}: classify mismatch"
            );
            assert_eq!(
                provider_error.retryable(),
                expected_retryable,
                "{label}: ProviderError retryability disagrees with classify"
            );
            assert!(
                provider_error
                    .message
                    .starts_with("failed to execute provider request:"),
                "{label}: message missing canonical prefix"
            );
        }
    }

    #[test]
    fn provider_error_cancelled_is_distinguishable() {
        let err = ProviderError::cancelled();
        assert!(err.is_cancelled());
        assert!(!err.retryable());
        assert_eq!(err.message, ProviderError::CANCELLED_MESSAGE);
    }

    /// AC3.4: When every attempt fails with a retryable error, the bounded
    /// `RetryPolicy::max_attempts` budget must be honored — the provider
    /// must stop after exactly that many requests, not loop forever, and the
    /// final emitted `ProviderError` must preserve its typed classification
    /// for any downstream policy that decides whether to retry the whole turn.
    #[tokio::test]
    async fn responses_provider_stops_after_retry_budget() {
        let attempts = Arc::new(tokio::sync::Mutex::new(0u32));
        let base_url = spawn_always_failing_stream_server(attempts.clone()).await;

        let max_attempts = 3u32;
        let policy = ResiliencePolicy {
            request_retry: RetryPolicy {
                max_attempts,
                base_backoff: Duration::from_millis(1),
                max_backoff: Duration::from_millis(5),
                deadline: Duration::from_secs(60),
                jitter_pct: 0,
            },
            ..ResiliencePolicy::default()
        };
        let provider = ResilientProvider::new(
            "responses-budget",
            ResponsesProvider::try_new(
                ResponsesProviderConfig {
                    label: "responses-budget",
                    capabilities: ProviderCapabilities::default(),
                    request: ResponsesProviderRequestConfig {
                        store: Some(false),
                        include_prompt_cache_key: false,
                        include_encrypted_reasoning: false,
                        reasoning_summary: None,
                        instruction_mode: ResponsesInstructionMode::DeveloperMessage,
                    },
                    compact_strategy: None,
                    rate_limit_strategy: None,
                    resilience_policy: policy,
                },
                "test-key",
                base_url,
                &[],
                None,
            )
            .expect("responses provider"),
            policy,
        );

        let mut stream = provider
            .stream(sample_responses_request(), CancellationToken::new())
            .await
            .expect("provider stream");

        let mut errors = Vec::new();
        while let Some(item) = stream.next().await {
            match item {
                Ok(_) => {}
                Err(error) => errors.push(error),
            }
        }

        let final_error = errors
            .pop()
            .expect("provider should surface a final error after exhausting budget");
        assert_eq!(final_error.kind, ProviderErrorKind::RateLimited);
        assert!(final_error.retryable());
        assert!(
            final_error
                .message
                .starts_with("failed to execute provider request:"),
            "final error message missing canonical prefix: {final_error:?}"
        );

        let attempts = *attempts.lock().await;
        assert_eq!(
            attempts, max_attempts,
            "expected exactly {max_attempts} requests, observed {attempts}"
        );
    }

    fn sample_responses_request() -> ProviderRequest {
        ProviderRequest {
            session_id: SessionId::new(),
            turn_id: TurnId::new(),
            model: ResolvedModel {
                role: ModelRole::default(),
                id: ModelId::from("default"),
                provider: ProviderName::from("responses-budget"),
                provider_kind: ProviderKind::OpenAi,
                api_kind: ApiKind::OpenAiResponses,
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

    /// Test fixture: HTTP/SSE server that returns an in-stream rate-limit
    /// error on every connection. Models the worst-case where every retry
    /// attempt fails with a retryable error so we can verify the budget caps
    /// the loop. Accepts up to 16 connections so a misconfigured (unbounded)
    /// retry would eventually trip an assertion rather than hang the test.
    async fn spawn_always_failing_stream_server(attempts: Arc<tokio::sync::Mutex<u32>>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind listener");
        let address = listener.local_addr().expect("local addr");

        tokio::spawn(async move {
            for _ in 0..16 {
                let Ok((mut socket, _)) = listener.accept().await else {
                    return;
                };
                if read_http_request(&mut socket).await.is_err() {
                    return;
                }
                *attempts.lock().await += 1;
                let error = json!({
                    "type": "error",
                    "error": {
                        "type": "tokens",
                        "code": "rate_limit_exceeded",
                        "message": "Rate limit reached. Please try again in 0.01s.",
                        "param": null
                    },
                    "sequence_number": 0
                });
                let body = format!("data: {error}\n\ndata: [DONE]\n\n");
                let response = format!(
                    "HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
                    body.len(),
                    body
                );
                let _ = socket.write_all(response.as_bytes()).await;
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
