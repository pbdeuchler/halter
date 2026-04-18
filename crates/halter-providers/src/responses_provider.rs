// pattern: Imperative Shell

use std::pin::Pin;
use std::task::{Context, Poll};

use async_openai::error::OpenAIError;
use futures::{Stream, StreamExt, channel::mpsc, stream::BoxStream};
use halter_protocol::{
    ApiKind, ProviderCapabilities, ProviderCompactionRequest, ProviderCompactionResponse,
    ProviderError, ProviderRequest, StreamEvent,
};
use tokio::select;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::openai_codec::{self, ResponsesRequestOptions};
use crate::openai_error::classify;
use crate::openai_rate_limit_policy::estimate_openai_request_cost;
use crate::responses_transport::{
    ResponsesRateLimitStrategy, ResponsesTransport, ResponsesTransportRequest, TransportError,
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
pub(crate) struct ResponsesProvider {
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

    #[must_use]
    pub(crate) fn capabilities(&self) -> ProviderCapabilities {
        let mut capabilities = self.config.capabilities.clone();
        capabilities.supports_compaction = self.config.compact_strategy.is_some();
        capabilities
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
        // Stream-scoped child token: cancellable by either the caller's
        // outer `cancel` or by `CancelOnDrop` when the consumer drops the
        // returned stream. Cancelling this child does not affect siblings
        // of the caller's broader scope.
        let stream_cancel = cancel.child_token();
        let task_cancel = stream_cancel.clone();

        tokio::spawn(async move {
            let cancel = task_cancel;
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
                    Err(TransportError::Cancelled) => {
                        warn!(provider = provider_label, "responses request cancelled");
                        let _ = tx.unbounded_send(Err(ProviderError::cancelled()));
                        return;
                    }
                    Err(TransportError::Retryable {
                        source,
                        backoff_hint,
                    }) => {
                        warn!(
                            provider = provider_label,
                            error = %source,
                            backoff_hint = ?backoff_hint,
                            "retrying responses request after retryable transport failure"
                        );
                        continue;
                    }
                    Err(TransportError::Fatal { source }) => {
                        let _ = tx.unbounded_send(Err(provider_error_from_openai(source, false)));
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
                            let _ = tx.unbounded_send(Err(ProviderError::cancelled()));
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
                                    let retryability = classify(&error);
                                    if !committed && retryability.is_retryable() {
                                        warn!(
                                            provider = provider_label,
                                            error = %error,
                                            backoff_hint = ?retryability.backoff_hint(),
                                            "retrying responses stream after retryable failure"
                                        );
                                        break;
                                    }
                                    warn!(provider = provider_label, error = %error, "responses stream returned provider error");
                                    let _ = tx.unbounded_send(Err(provider_error_from_openai(
                                        error,
                                        retryability.is_retryable(),
                                    )));
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

        // Wrap the receiver so that consumer drop fires `stream_cancel`,
        // which propagates to the spawned worker and to the cancel-aware
        // SSE decode task in `ResponsesTransport::stream_response`. Without
        // this wrapper, dropping the stream would only signal back-pressure
        // to the worker via channel send failure on the *next* event —
        // leaving the SSE task parked on `byte_stream.next()` indefinitely.
        Ok(CancelOnDrop {
            inner: rx.boxed(),
            cancel: stream_cancel,
        }
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

/// Convert an `OpenAIError` to a `ProviderError`, using the caller-provided
/// `retryable` flag (already decided by `classify`). Centralizing the
/// formatting here ensures every error string has the same prefix.
fn provider_error_from_openai(error: OpenAIError, retryable: bool) -> ProviderError {
    let message = match &error {
        OpenAIError::ApiError(api_error) => {
            format!("failed to execute provider request: {}", api_error.message)
        }
        OpenAIError::JSONDeserialize(json_error, content) => format!(
            "failed to execute provider request: failed to deserialize api response: error:{json_error} content:{content}"
        ),
        other => format!("failed to execute provider request: {other}"),
    };
    ProviderError::new(message, retryable)
}

/// Stream wrapper that cancels its `CancellationToken` when dropped. Used to
/// propagate consumer drop back into the spawned SSE decode task so the task
/// exits promptly instead of leaking on a parked `byte_stream.next()` (AC3.1).
struct CancelOnDrop<S> {
    inner: S,
    cancel: CancellationToken,
}

impl<S: Stream + Unpin> Stream for CancelOnDrop<S> {
    type Item = S::Item;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Pin::new(&mut self.inner).poll_next(cx)
    }
}

impl<S> Drop for CancelOnDrop<S> {
    fn drop(&mut self) {
        self.cancel.cancel();
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

#[cfg(test)]
mod tests {
    use halter_protocol::{
        ModelId, ModelRole, ProviderKind, ProviderName, ResolvedModel, SessionId,
    };

    use super::*;

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
            let provider_error = provider_error_from_openai(error, retryability.is_retryable());
            assert_eq!(
                retryability.is_retryable(),
                expected_retryable,
                "{label}: classify mismatch"
            );
            assert_eq!(
                provider_error.retryable, expected_retryable,
                "{label}: ProviderError.retryable disagrees with classify"
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
        assert!(!err.retryable);
        assert_eq!(err.message, ProviderError::CANCELLED_MESSAGE);
    }

    /// AC3.1 wiring: `CancelOnDrop` must cancel its token on `Drop` so that
    /// the parent token's downstream listeners (the spawned worker loop and
    /// the cancel-aware SSE decoder) observe consumer drop without depending
    /// on the channel surfacing a failed send.
    #[test]
    fn cancel_on_drop_cancels_token_on_drop() {
        let token = CancellationToken::new();
        {
            let _wrapper = CancelOnDrop {
                inner: futures::stream::empty::<i32>(),
                cancel: token.clone(),
            };
            assert!(!token.is_cancelled());
        }
        assert!(token.is_cancelled());
    }

    /// `CancelOnDrop` must remain a transparent stream proxy — adding the
    /// drop side-effect should not perturb forward iteration semantics.
    #[tokio::test]
    async fn cancel_on_drop_passes_through_stream_items() {
        let token = CancellationToken::new();
        let mut wrapper = CancelOnDrop {
            inner: futures::stream::iter(vec![1, 2, 3]),
            cancel: token.clone(),
        };
        assert_eq!(wrapper.next().await, Some(1));
        assert_eq!(wrapper.next().await, Some(2));
        assert_eq!(wrapper.next().await, Some(3));
        assert_eq!(wrapper.next().await, None);
        assert!(!token.is_cancelled());
        drop(wrapper);
        assert!(token.is_cancelled());
    }
}
