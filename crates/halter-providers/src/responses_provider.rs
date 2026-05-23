// pattern: Imperative Shell

use std::collections::HashSet;
use std::pin::Pin;
use std::task::{Context, Poll};

use async_openai::error::OpenAIError;
use futures::{Stream, StreamExt, channel::mpsc, stream::BoxStream};
use halter_protocol::{
    ApiKind, MessageId, ProviderCapabilities, ProviderCompactionRequest,
    ProviderCompactionResponse, ProviderCompactionStrategy, ProviderError, ProviderRequest,
    StreamEvent,
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
use crate::retry::{RetryGate, RetryPolicy};
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
}

#[derive(Debug, Clone)]
pub(crate) struct ResponsesProviderConfig {
    pub label: &'static str,
    pub capabilities: ProviderCapabilities,
    pub request: ResponsesProviderRequestConfig,
    pub compact_strategy: Option<CompactStrategy>,
    pub rate_limit_strategy: Option<ResponsesRateLimitStrategy>,
    /// Retry budget for the streaming pipeline. Defaults are appropriate
    /// for production; tests inject smaller values to keep the suite fast.
    pub retry_policy: RetryPolicy,
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
        api_key: impl Into<SecretString>,
        base_url: impl Into<String>,
        header_overrides: &[(String, String)],
        temperature: Option<f32>,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            config,
            transport: ResponsesTransport::try_new(api_key, base_url, header_overrides)?,
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
        let (tx, rx) = mpsc::unbounded();
        let provider_label = self.config.label;
        let transport = self.transport.clone();
        let retry_policy = self.config.retry_policy;
        // Stream-scoped child token: cancellable by either the caller's
        // outer `cancel` or by `CancelOnDrop` when the consumer drops the
        // returned stream. Cancelling this child does not affect siblings
        // of the caller's broader scope.
        let stream_cancel = cancel.child_token();
        let task_cancel = stream_cancel.clone();

        tokio::spawn(async move {
            let cancel = task_cancel;
            // Single retry gate spans both startup failures and mid-stream
            // pre-commit failures (AC3.4). Without a single gate, a stream
            // that reliably failed *after* connecting could loop forever
            // even though the transport startup retry counter was bounded.
            let mut gate = RetryGate::new(retry_policy);
            // Cross-attempt MessageStart dedup (AC3.6). When an earlier
            // attempt's `pending_events` buffer is discarded by retry, the
            // next attempt's decoder will re-emit `MessageStart` for the
            // same response; we suppress the duplicate before forwarding.
            let mut commit_dedup = CommitDedup::default();

            loop {
                let attempt_id = gate.next_attempt_id();
                // Single attempt: try to start the stream, then consume
                // events. Returns `Some((source, hint))` when this attempt
                // failed *retryably* (either at startup or pre-commit) and
                // the outer loop should consult the gate. Terminal outcomes
                // (cancellation, fatal failure, post-commit failure, normal
                // completion) handle their own consumer emissions and
                // `return` directly out of the spawned task.
                let retry_reason: Option<(OpenAIError, Option<std::time::Duration>)> = 'attempt: {
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
                        Err(TransportError::Fatal { source }) => {
                            let _ =
                                tx.unbounded_send(Err(provider_error_from_openai(source, false)));
                            return;
                        }
                        Err(TransportError::Retryable {
                            source,
                            backoff_hint,
                        }) => {
                            warn!(
                                provider = provider_label,
                                attempt = attempt_id,
                                error = %source,
                                backoff_hint = ?backoff_hint,
                                "retryable transport startup failure"
                            );
                            break 'attempt Some((source, backoff_hint));
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
                            item = response_stream.next() => match item {
                                Some(Ok(event)) => match decoder.decode(event) {
                                    Ok(events) => {
                                        if committed {
                                            for event in events {
                                                if !commit_dedup.allow(&event) {
                                                    continue;
                                                }
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
                                                if !commit_dedup.allow(&event) {
                                                    continue;
                                                }
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
                                        let hint = retryability.backoff_hint();
                                        warn!(
                                            provider = provider_label,
                                            attempt = attempt_id,
                                            error = %error,
                                            backoff_hint = ?hint,
                                            "retryable pre-commit stream failure"
                                        );
                                        break 'attempt Some((error, hint));
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
                                            if !commit_dedup.allow(&event) {
                                                continue;
                                            }
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
                };

                // Retry decision: consult the gate, sleep, and loop. On
                // exhaustion, surface the latest source as a non-retryable
                // ProviderError (the retryable signal has been "spent" by
                // the budget; further retries would be unbounded).
                if let Some((source, hint)) = retry_reason {
                    match gate.record_failure_and_next_backoff(hint) {
                        Some(delay) => {
                            info!(
                                provider = provider_label,
                                attempt = attempt_id,
                                retry_in_ms = delay.as_millis() as u64,
                                "retrying responses request"
                            );
                            tokio::time::sleep(delay).await;
                        }
                        None => {
                            warn!(
                                provider = provider_label,
                                attempt = attempt_id,
                                "responses retry budget exhausted"
                            );
                            let _ =
                                tx.unbounded_send(Err(provider_error_from_openai(source, false)));
                            return;
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

/// Cross-attempt dedup for commit-eligible boundary events. When an attempt
/// fails before commit, the next attempt's decoder will re-emit the same
/// `MessageStart` for the same response id; without suppression, downstream
/// consumers would see two starts for one logical message (AC3.6).
#[derive(Debug, Default)]
struct CommitDedup {
    seen_message_starts: HashSet<MessageId>,
}

impl CommitDedup {
    fn allow(&mut self, event: &StreamEvent) -> bool {
        if let StreamEvent::MessageStart { id } = event {
            return self.seen_message_starts.insert(id.clone());
        }
        true
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
    use std::sync::Arc;
    use std::time::Duration;

    use futures::StreamExt;
    use halter_protocol::{
        AssembledPrompt, BlockId, MessageId, ModelId, ModelRole, ProviderKind, ProviderName,
        ResolvedModel, SessionId, TurnId,
    };
    use serde_json::json;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;

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
                },
                compact_strategy: None,
                rate_limit_strategy: None,
                retry_policy: RetryPolicy::default(),
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

    /// AC3.6: When a pre-commit attempt fails and is retried, the next
    /// attempt's decoder will re-emit `MessageStart` for the same response;
    /// `CommitDedup` must suppress the duplicate so consumers see exactly one
    /// start per logical message id. Distinct ids must still pass through —
    /// otherwise a multi-message conversation would be silently truncated.
    #[test]
    fn commit_dedup_suppresses_duplicate_message_start() {
        let mut dedup = CommitDedup::default();
        let id = MessageId::from("msg_alpha");
        let other = MessageId::from("msg_beta");

        assert!(dedup.allow(&StreamEvent::MessageStart { id: id.clone() }));
        assert!(!dedup.allow(&StreamEvent::MessageStart { id: id.clone() }));
        assert!(dedup.allow(&StreamEvent::MessageStart { id: other }));
    }

    /// `CommitDedup` must only intercept `MessageStart`. All other events —
    /// in particular repeats of the same `BlockId` for `TextStart` /
    /// `TextDelta` — must always pass through, since the codec uses block
    /// identity for normal in-stream segmentation, not for retry suppression.
    #[test]
    fn commit_dedup_passes_through_non_message_start_events() {
        let mut dedup = CommitDedup::default();
        let block = BlockId::from("blk_one");
        let event = StreamEvent::TextStart { id: block.clone() };
        assert!(dedup.allow(&event));
        assert!(dedup.allow(&event));
        assert!(dedup.allow(&StreamEvent::TextEnd { id: block }));
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

    /// AC3.4: When every attempt fails with a retryable error, the bounded
    /// `RetryPolicy::max_attempts` budget must be honored — the provider
    /// must stop after exactly that many requests, not loop forever, and the
    /// final emitted `ProviderError` must be `retryable=false` (the budget
    /// has been "spent"; further retries are no longer authorized).
    #[tokio::test]
    async fn responses_provider_stops_after_retry_budget() {
        let attempts = Arc::new(tokio::sync::Mutex::new(0u32));
        let base_url = spawn_always_failing_stream_server(attempts.clone()).await;

        let max_attempts = 3u32;
        let provider = ResponsesProvider::try_new(
            ResponsesProviderConfig {
                label: "responses-budget",
                capabilities: ProviderCapabilities::default(),
                request: ResponsesProviderRequestConfig {
                    store: Some(false),
                    include_prompt_cache_key: false,
                    include_encrypted_reasoning: false,
                    reasoning_summary: None,
                },
                compact_strategy: None,
                rate_limit_strategy: None,
                retry_policy: RetryPolicy {
                    max_attempts,
                    base_backoff: Duration::from_millis(1),
                    max_backoff: Duration::from_millis(5),
                    deadline: Duration::from_secs(60),
                    jitter_pct: 0,
                },
            },
            "test-key",
            base_url,
            &[],
            None,
        )
        .expect("responses provider");

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
        assert!(
            !final_error.retryable,
            "final error after budget exhaustion must be non-retryable: {final_error:?}"
        );
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
