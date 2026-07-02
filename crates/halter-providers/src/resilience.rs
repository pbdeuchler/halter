// pattern: Imperative Shell

use std::collections::HashSet;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use async_trait::async_trait;
use futures::{SinkExt, Stream, StreamExt, channel::mpsc, stream::BoxStream};
use halter_protocol::{
    CompactionWindow, Message, MessageId, ProviderCapabilities, ProviderCompactionRequest,
    ProviderCompactionResponse, ProviderError, ProviderErrorKind, ProviderRequest, StreamEvent,
};
use tokio::select;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::Provider;
use crate::retry::{RetryGate, RetryPolicy};

/// Bound on events buffered between the retry worker and a slow consumer.
/// Small power of two: enough to smooth bursts, small enough that a stalled
/// consumer exerts backpressure instead of accumulating a whole stream.
const EVENT_CHANNEL_CAPACITY: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProviderTimeouts {
    pub connect: Duration,
    pub request: Duration,
    pub stream_idle: Duration,
}

impl Default for ProviderTimeouts {
    fn default() -> Self {
        Self {
            connect: Duration::from_secs(10),
            request: Duration::from_secs(60),
            stream_idle: Duration::from_secs(60),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ResiliencePolicy {
    pub timeouts: ProviderTimeouts,
    pub request_retry: RetryPolicy,
}

pub trait ProviderErrorClassifier: Send + Sync {
    fn classify(&self, error: ProviderError) -> ProviderError;
}

#[derive(Debug, Default)]
pub struct DefaultProviderErrorClassifier;

impl ProviderErrorClassifier for DefaultProviderErrorClassifier {
    fn classify(&self, error: ProviderError) -> ProviderError {
        error
    }
}

/// Decorator that adds bounded retries with backoff to an inner [`Provider`].
///
/// `stream` retries pre-commit retryable failures (transport faults, rate
/// limits, in-band error events) and never retries once user-visible content
/// has been emitted. `compact` is unary and gets the same retry/timeout
/// treatment: attempts whose error carries a retryable [`ProviderError`]
/// (attached by the built-in transports) are retried; untyped errors are
/// treated as fatal so capability errors like "provider does not support
/// compaction" are not retry-looped.
///
/// All built-in providers ([`crate::AnthropicProvider`],
/// [`crate::OpenAiProvider`], [`crate::OpenRouterProvider`]) construct this
/// wrapper internally; the higher-level builder only supplies the policy.
#[derive(Clone)]
pub struct ResilientProvider<P> {
    label: &'static str,
    inner: Arc<P>,
    policy: ResiliencePolicy,
    classifier: Arc<dyn ProviderErrorClassifier>,
}

impl<P> fmt::Debug for ResilientProvider<P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ResilientProvider")
            .field("label", &self.label)
            .field("policy", &self.policy)
            .finish_non_exhaustive()
    }
}

impl<P> ResilientProvider<P> {
    pub fn new(label: &'static str, inner: P, policy: ResiliencePolicy) -> Self {
        Self::new_with_classifier(
            label,
            inner,
            policy,
            Arc::new(DefaultProviderErrorClassifier),
        )
    }

    pub fn new_with_classifier(
        label: &'static str,
        inner: P,
        policy: ResiliencePolicy,
        classifier: Arc<dyn ProviderErrorClassifier>,
    ) -> Self {
        Self {
            label,
            inner: Arc::new(inner),
            policy,
            classifier,
        }
    }
}

#[async_trait]
impl<P> Provider for ResilientProvider<P>
where
    P: Provider + 'static,
{
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
        let (tx, rx) = mpsc::channel(EVENT_CHANNEL_CAPACITY);
        let stream_cancel = cancel.child_token();
        let task_cancel = stream_cancel.clone();
        let inner = self.inner.clone();
        let policy = self.policy;
        let classifier = self.classifier.clone();
        let label = self.label;

        tokio::spawn(async move {
            let cancel = task_cancel;
            let mut tx = tx;
            let mut gate = RetryGate::new(policy.request_retry);
            let mut commit_dedup = CommitDedup::default();

            loop {
                let attempt_id = gate.next_attempt_id();
                let attempt_cancel = cancel.child_token();
                let mut attempt_stream = match tokio::time::timeout(
                    policy.timeouts.request,
                    inner.stream(request.clone(), attempt_cancel.clone()),
                )
                .await
                {
                    Ok(Ok(stream)) => stream,
                    Ok(Err(error)) => {
                        // Recover the typed classification when the provider
                        // attached one (deterministic setup failures are
                        // Fatal); only untyped errors default to Transient.
                        let error = classifier.classify(setup_provider_error(error));
                        attempt_cancel.cancel();
                        if !retry_or_emit(
                            RetryContext {
                                label,
                                attempt_id,
                                gate: &mut gate,
                                tx: &mut tx,
                                cancel: &cancel,
                            },
                            error,
                        )
                        .await
                        {
                            return;
                        }
                        continue;
                    }
                    Err(_) => {
                        let error = provider_timeout_error(
                            format!(
                                "failed to execute provider request: request timed out after {}s",
                                policy.timeouts.request.as_secs()
                            ),
                            None,
                        );
                        attempt_cancel.cancel();
                        if !retry_or_emit(
                            RetryContext {
                                label,
                                attempt_id,
                                gate: &mut gate,
                                tx: &mut tx,
                                cancel: &cancel,
                            },
                            error,
                        )
                        .await
                        {
                            return;
                        }
                        continue;
                    }
                };

                let mut pending_events = Vec::new();
                let mut committed = false;

                loop {
                    let item = select! {
                        _ = cancel.cancelled() => {
                            let _ = tx.try_send(Err(ProviderError::cancelled()));
                            return;
                        }
                        item = tokio::time::timeout(policy.timeouts.stream_idle, attempt_stream.next()) => item,
                    };

                    match item {
                        Err(_) => {
                            let error = provider_timeout_error(
                                format!(
                                    "failed to execute provider request: stream idle timeout after {}s",
                                    policy.timeouts.stream_idle.as_secs()
                                ),
                                None,
                            );
                            if !committed && error.retryable() {
                                warn!(
                                    provider = label,
                                    attempt = attempt_id,
                                    error = %error,
                                    "retryable pre-commit provider stream timeout"
                                );
                                attempt_cancel.cancel();
                                if retry_or_emit(
                                    RetryContext {
                                        label,
                                        attempt_id,
                                        gate: &mut gate,
                                        tx: &mut tx,
                                        cancel: &cancel,
                                    },
                                    error,
                                )
                                .await
                                {
                                    break;
                                }
                                return;
                            }
                            let _ = forward(&mut tx, &cancel, Err(error)).await;
                            return;
                        }
                        Ok(Some(Ok(StreamEvent::Error { error }))) => {
                            let error = classifier.classify(error);
                            if !committed && error.retryable() {
                                warn!(
                                    provider = label,
                                    attempt = attempt_id,
                                    error = %error,
                                    backoff_hint = ?error.backoff_hint,
                                    "retryable pre-commit provider stream error event"
                                );
                                attempt_cancel.cancel();
                                if retry_or_emit(
                                    RetryContext {
                                        label,
                                        attempt_id,
                                        gate: &mut gate,
                                        tx: &mut tx,
                                        cancel: &cancel,
                                    },
                                    error,
                                )
                                .await
                                {
                                    break;
                                }
                                return;
                            }
                            let _ = forward(&mut tx, &cancel, Err(error)).await;
                            return;
                        }
                        Ok(Some(Ok(event))) => {
                            if committed {
                                if commit_dedup.allow(&event)
                                    && !forward(&mut tx, &cancel, Ok(event)).await
                                {
                                    return;
                                }
                                continue;
                            }

                            pending_events.push(event);
                            if pending_events
                                .last()
                                .is_some_and(stream_event_commits_attempt)
                            {
                                committed = true;
                                for event in pending_events.drain(..) {
                                    if !commit_dedup.allow(&event) {
                                        continue;
                                    }
                                    if !forward(&mut tx, &cancel, Ok(event)).await {
                                        return;
                                    }
                                }
                            }
                        }
                        Ok(Some(Err(error))) => {
                            let error = classifier.classify(error);
                            if !committed && error.retryable() {
                                warn!(
                                    provider = label,
                                    attempt = attempt_id,
                                    error = %error,
                                    backoff_hint = ?error.backoff_hint,
                                    "retryable pre-commit provider stream failure"
                                );
                                attempt_cancel.cancel();
                                if retry_or_emit(
                                    RetryContext {
                                        label,
                                        attempt_id,
                                        gate: &mut gate,
                                        tx: &mut tx,
                                        cancel: &cancel,
                                    },
                                    error,
                                )
                                .await
                                {
                                    break;
                                }
                                return;
                            }
                            let _ = forward(&mut tx, &cancel, Err(error)).await;
                            return;
                        }
                        Ok(None) => {
                            for event in pending_events.drain(..) {
                                if !commit_dedup.allow(&event) {
                                    continue;
                                }
                                if !forward(&mut tx, &cancel, Ok(event)).await {
                                    return;
                                }
                            }
                            debug!(provider = label, "provider stream completed");
                            return;
                        }
                    }
                }
            }
        });

        Ok(CancelOnDrop::new(rx, stream_cancel).boxed())
    }

    async fn compact(
        &self,
        request: ProviderCompactionRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<ProviderCompactionResponse> {
        let mut gate = RetryGate::new(self.policy.request_retry);
        loop {
            let attempt_id = gate.next_attempt_id();
            let attempt = select! {
                biased;
                _ = cancel.cancelled() => return Err(anyhow::Error::new(ProviderError::cancelled())),
                attempt = tokio::time::timeout(
                    self.policy.timeouts.request,
                    self.inner.compact(request.clone(), cancel.child_token()),
                ) => attempt,
            };
            let error = match attempt {
                Ok(Ok(response)) => return Ok(response),
                // Untyped errors default to Fatal here (unlike stream setup):
                // the trait's own compact fallback rejects unsupported
                // compaction with an untyped error, which must not be
                // retry-looped. Built-in transports attach typed errors.
                Ok(Err(error)) => self.classifier.classify(compact_provider_error(error)),
                Err(_) => provider_timeout_error(
                    format!(
                        "failed to compact session: request timed out after {}s",
                        self.policy.timeouts.request.as_secs()
                    ),
                    None,
                ),
            };
            if !error.retryable() {
                return Err(anyhow::Error::new(error));
            }
            match gate.record_failure_and_next_backoff(error.backoff_hint) {
                Some(delay) => {
                    info!(
                        provider = self.label,
                        attempt = attempt_id,
                        retry_in_ms = delay.as_millis() as u64,
                        "retrying provider compaction request"
                    );
                    select! {
                        _ = cancel.cancelled() => {
                            return Err(anyhow::Error::new(ProviderError::cancelled()));
                        }
                        _ = tokio::time::sleep(delay) => {}
                    }
                }
                None => {
                    warn!(
                        provider = self.label,
                        attempt = attempt_id,
                        kind = ?error.kind,
                        "provider compaction retry budget exhausted"
                    );
                    return Err(anyhow::Error::new(error));
                }
            }
        }
    }
}

/// Classify a stream-setup failure: prefer the typed [`ProviderError`]
/// attached by the provider, default the rest to `Transient` (network-layer
/// failures from foreign `Provider` impls are more often recoverable).
fn setup_provider_error(error: anyhow::Error) -> ProviderError {
    match error.downcast::<ProviderError>() {
        Ok(provider_error) => provider_error,
        Err(error) => ProviderError::with_kind(format!("{error:#}"), ProviderErrorKind::Transient),
    }
}

/// Classify a compaction failure: prefer the typed [`ProviderError`],
/// default the rest to `Fatal` so untyped capability errors short-circuit.
fn compact_provider_error(error: anyhow::Error) -> ProviderError {
    match error.downcast::<ProviderError>() {
        Ok(provider_error) => provider_error,
        Err(error) => ProviderError::with_kind(format!("{error:#}"), ProviderErrorKind::Fatal),
    }
}

/// Send an item to the consumer, racing the cancellation token so a stalled
/// consumer cannot wedge the worker past cancellation. Returns `false` when
/// the worker should stop (consumer dropped or stream cancelled); a
/// best-effort cancellation marker is queued in the cancel case.
async fn forward(
    tx: &mut mpsc::Sender<Result<StreamEvent, ProviderError>>,
    cancel: &CancellationToken,
    item: Result<StreamEvent, ProviderError>,
) -> bool {
    select! {
        _ = cancel.cancelled() => {
            let _ = tx.try_send(Err(ProviderError::cancelled()));
            false
        }
        result = tx.send(item) => result.is_ok(),
    }
}

struct RetryContext<'a> {
    label: &'static str,
    attempt_id: u32,
    gate: &'a mut RetryGate,
    tx: &'a mut mpsc::Sender<Result<StreamEvent, ProviderError>>,
    cancel: &'a CancellationToken,
}

async fn retry_or_emit(context: RetryContext<'_>, error: ProviderError) -> bool {
    if !error.retryable() {
        let _ = forward(context.tx, context.cancel, Err(error)).await;
        return false;
    }

    match context
        .gate
        .record_failure_and_next_backoff(error.backoff_hint)
    {
        Some(delay) => {
            info!(
                provider = context.label,
                attempt = context.attempt_id,
                retry_in_ms = delay.as_millis() as u64,
                "retrying provider request"
            );
            select! {
                _ = context.cancel.cancelled() => {
                    let _ = context.tx.try_send(Err(ProviderError::cancelled()));
                    false
                }
                _ = tokio::time::sleep(delay) => true,
            }
        }
        None => {
            warn!(
                provider = context.label,
                attempt = context.attempt_id,
                kind = ?error.kind,
                "provider retry budget exhausted"
            );
            let _ = forward(context.tx, context.cancel, Err(error)).await;
            false
        }
    }
}

fn provider_timeout_error(message: String, backoff_hint: Option<Duration>) -> ProviderError {
    ProviderError::with_kind(message, ProviderErrorKind::Transient).with_backoff_hint(backoff_hint)
}

/// Stream wrapper that cancels its token when dropped. Used to propagate
/// consumer drop back into provider worker tasks so idle streams do not leak.
pub(crate) struct CancelOnDrop<S> {
    inner: S,
    cancel: CancellationToken,
}

impl<S> CancelOnDrop<S> {
    pub(crate) fn new(inner: S, cancel: CancellationToken) -> Self {
        Self { inner, cancel }
    }
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

/// Suppresses duplicate `MessageStart` events *within a single committed
/// stream* — a defensive guard against upstreams that replay the boundary
/// event. Retries never cross the commit boundary, so this never has to (and
/// does not) dedup across attempts.
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

/// Whether an event represents real generated content that forecloses
/// retrying the attempt. `ProviderWarning` is deliberately excluded: an
/// early warning is diagnostic, not user-visible content, and must not
/// spend the retry budget before anything was delivered.
fn stream_event_commits_attempt(event: &StreamEvent) -> bool {
    matches!(
        event,
        StreamEvent::TextDelta { .. }
            | StreamEvent::ThinkingDelta { .. }
            | StreamEvent::ToolCallStart { .. }
            | StreamEvent::ToolArgsDelta { .. }
            | StreamEvent::ToolCallEnd { .. }
            | StreamEvent::MessageEnd { .. }
    )
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use async_trait::async_trait;
    use futures::{StreamExt, stream};
    use halter_protocol::{
        ApiKind, AssembledPrompt, BlockId, CacheBreakpoints, ModelId, ModelRole, ProviderKind,
        ProviderName, ResolvedModel, SessionId, StopReason, TurnId,
    };

    use super::*;

    type Script = Vec<Result<StreamEvent, ProviderError>>;

    #[derive(Debug)]
    struct ScriptedProvider {
        attempts: Arc<AtomicUsize>,
        produced: Arc<AtomicUsize>,
        scripts: Arc<Mutex<VecDeque<Script>>>,
    }

    impl ScriptedProvider {
        fn new(scripts: Vec<Script>) -> Self {
            Self {
                attempts: Arc::new(AtomicUsize::new(0)),
                produced: Arc::new(AtomicUsize::new(0)),
                scripts: Arc::new(Mutex::new(scripts.into())),
            }
        }

        fn attempts(&self) -> Arc<AtomicUsize> {
            self.attempts.clone()
        }

        /// Count of items the resilience worker has pulled off the inner
        /// stream — the observable for backpressure assertions.
        fn produced_events(&self) -> Arc<AtomicUsize> {
            self.produced.clone()
        }
    }

    #[async_trait]
    impl Provider for ScriptedProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }

        async fn stream(
            &self,
            _request: ProviderRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
            self.attempts.fetch_add(1, Ordering::SeqCst);
            let script = self
                .scripts
                .lock()
                .expect("script mutex")
                .pop_front()
                .unwrap_or_default();
            let produced = self.produced.clone();
            Ok(stream::iter(script)
                .inspect(move |_| {
                    produced.fetch_add(1, Ordering::SeqCst);
                })
                .boxed())
        }
    }

    #[derive(Debug)]
    struct StartupFailureProvider {
        attempts: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Provider for StartupFailureProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }

        async fn stream(
            &self,
            _request: ProviderRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
            if self.attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                anyhow::bail!("temporary transport failure");
            }
            Ok(stream::iter(success_events("msg_startup", "ok")).boxed())
        }
    }

    #[derive(Debug)]
    struct HangingStartupProvider {
        attempts: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Provider for HangingStartupProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }

        async fn stream(
            &self,
            _request: ProviderRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
            self.attempts.fetch_add(1, Ordering::SeqCst);
            std::future::pending::<()>().await;
            Ok(stream::empty().boxed())
        }
    }

    #[derive(Debug)]
    struct PendingStreamProvider;

    #[async_trait]
    impl Provider for PendingStreamProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }

        async fn stream(
            &self,
            _request: ProviderRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
            Ok(stream::pending().boxed())
        }
    }

    struct StartupClassifier;

    impl ProviderErrorClassifier for StartupClassifier {
        fn classify(&self, error: ProviderError) -> ProviderError {
            if error.message.contains("temporary transport") {
                return ProviderError::with_kind(error.message, ProviderErrorKind::Transient);
            }
            error
        }
    }

    #[tokio::test]
    async fn retries_pre_commit_retryable_stream_failure() {
        let provider = ScriptedProvider::new(vec![
            vec![Err(transient_error("rate limited"))],
            success_events("msg_retry", "done"),
        ]);
        let attempts = provider.attempts();
        let resilient = ResilientProvider::new("test", provider, test_policy(3));

        let events = collect_events(resilient).await;

        assert_eq!(attempts.load(Ordering::SeqCst), 2);
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
    }

    #[tokio::test]
    async fn does_not_retry_after_commit_boundary() {
        let provider = ScriptedProvider::new(vec![
            vec![
                Ok(StreamEvent::MessageStart {
                    id: MessageId::from("msg_committed"),
                }),
                Ok(StreamEvent::TextDelta {
                    id: BlockId::from("text_committed"),
                    delta: "partial".to_owned(),
                }),
                Err(transient_error("upstream reset")),
            ],
            success_events("msg_second", "should not run"),
        ]);
        let attempts = provider.attempts();
        let resilient = ResilientProvider::new("test", provider, test_policy(3));

        let mut stream = resilient
            .stream(sample_request(), CancellationToken::new())
            .await
            .expect("stream");
        let mut saw_error = false;
        while let Some(item) = stream.next().await {
            if item.is_err() {
                saw_error = true;
            }
        }

        assert!(saw_error, "post-commit error should be propagated");
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn classifier_can_make_startup_errors_retryable() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let provider = StartupFailureProvider {
            attempts: attempts.clone(),
        };
        let resilient = ResilientProvider::new_with_classifier(
            "test",
            provider,
            test_policy(2),
            Arc::new(StartupClassifier),
        );

        let events = collect_events(resilient).await;

        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::TextDelta { delta, .. } if delta == "ok"
        )));
    }

    #[derive(Debug)]
    struct TypedFatalStartupProvider {
        attempts: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Provider for TypedFatalStartupProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }

        async fn stream(
            &self,
            _request: ProviderRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
            self.attempts.fetch_add(1, Ordering::SeqCst);
            Err(anyhow::Error::new(ProviderError::with_kind(
                "failed to encode request",
                ProviderErrorKind::Fatal,
            )))
        }
    }

    /// A setup `Err` carrying a typed Fatal `ProviderError` must
    /// short-circuit: previously every setup error was rewrapped as
    /// `Transient` and burned the whole retry budget.
    #[tokio::test]
    async fn typed_fatal_startup_error_short_circuits_retry_budget() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let provider = TypedFatalStartupProvider {
            attempts: attempts.clone(),
        };
        let resilient = ResilientProvider::new("test", provider, test_policy(5));

        let items = collect_items(resilient, CancellationToken::new()).await;

        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        assert_eq!(items.len(), 1);
        let error = items[0].as_ref().expect_err("fatal setup error");
        assert_eq!(error.kind, ProviderErrorKind::Fatal);
        assert_eq!(error.message, "failed to encode request");
    }

    #[tokio::test]
    async fn retries_startup_errors_by_default() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let provider = StartupFailureProvider {
            attempts: attempts.clone(),
        };
        let resilient = ResilientProvider::new("test", provider, test_policy(2));

        let events = collect_events(resilient).await;

        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::TextDelta { delta, .. } if delta == "ok"
        )));
    }

    #[tokio::test]
    async fn exhausts_retry_budget_and_emits_final_error() {
        let provider = ScriptedProvider::new(vec![
            vec![Err(transient_error("first"))],
            vec![Err(transient_error("final"))],
        ]);
        let attempts = provider.attempts();
        let resilient = ResilientProvider::new("test", provider, test_policy(2));

        let items = collect_items(resilient, CancellationToken::new()).await;

        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert_eq!(items.len(), 1);
        let error = items[0].as_ref().expect_err("final item should be error");
        assert_eq!(error.message, "final");
        assert!(error.retryable());
    }

    #[tokio::test]
    async fn emits_fatal_pre_commit_error_without_retry() {
        let provider = ScriptedProvider::new(vec![
            vec![Err(ProviderError::with_kind(
                "bad request",
                ProviderErrorKind::Fatal,
            ))],
            success_events("msg_second", "should not run"),
        ]);
        let attempts = provider.attempts();
        let resilient = ResilientProvider::new("test", provider, test_policy(3));

        let items = collect_items(resilient, CancellationToken::new()).await;

        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        assert_eq!(items.len(), 1);
        let error = items[0].as_ref().expect_err("fatal item should be error");
        assert_eq!(error.kind, ProviderErrorKind::Fatal);
    }

    #[tokio::test]
    async fn retries_pre_commit_in_band_error_event() {
        let provider = ScriptedProvider::new(vec![
            vec![Ok(StreamEvent::Error {
                error: transient_error("provider event error"),
            })],
            success_events("msg_retry_event", "done"),
        ]);
        let attempts = provider.attempts();
        let resilient = ResilientProvider::new("test", provider, test_policy(3));

        let events = collect_events(resilient).await;

        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::TextDelta { delta, .. } if delta == "done"
        )));
    }

    #[tokio::test]
    async fn request_setup_timeout_emits_retryable_error() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let provider = HangingStartupProvider {
            attempts: attempts.clone(),
        };
        let resilient = ResilientProvider::new("test", provider, timeout_policy());

        let items = collect_items(resilient, CancellationToken::new()).await;

        assert_eq!(attempts.load(Ordering::SeqCst), 1);
        let error = items
            .first()
            .and_then(|item| item.as_ref().err())
            .expect("timeout should emit an error");
        assert_eq!(error.kind, ProviderErrorKind::Transient);
        assert!(error.message.contains("request timed out"));
    }

    #[tokio::test]
    async fn stream_idle_timeout_emits_retryable_error() {
        let resilient = ResilientProvider::new("test", PendingStreamProvider, timeout_policy());

        let items = collect_items(resilient, CancellationToken::new()).await;

        let error = items
            .first()
            .and_then(|item| item.as_ref().err())
            .expect("idle timeout should emit an error");
        assert_eq!(error.kind, ProviderErrorKind::Transient);
        assert!(error.message.contains("stream idle timeout"));
    }

    #[tokio::test]
    async fn cancellation_during_backoff_emits_cancelled_error() {
        let provider = ScriptedProvider::new(vec![
            vec![Err(transient_error("retry me"))],
            success_events("msg_late", "late"),
        ]);
        let attempts = provider.attempts();
        let policy = ResiliencePolicy {
            request_retry: RetryPolicy {
                base_backoff: Duration::from_secs(30),
                max_backoff: Duration::from_secs(30),
                ..test_policy(3).request_retry
            },
            ..test_policy(3)
        };
        let resilient = ResilientProvider::new("test", provider, policy);
        let cancel = CancellationToken::new();
        let mut stream = resilient
            .stream(sample_request(), cancel.clone())
            .await
            .expect("stream");

        tokio::task::yield_now().await;
        cancel.cancel();
        let item = tokio::time::timeout(Duration::from_secs(1), stream.next())
            .await
            .expect("cancel should produce an item")
            .expect("cancelled item");

        assert!(
            item.expect_err("item should be cancellation")
                .is_cancelled()
        );
        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    /// L6: a `ProviderWarning` before any content must not commit the
    /// attempt — a subsequent transient failure is still retried. Under the
    /// old behavior the warning committed, attempts stayed at 1, and the
    /// error was surfaced instead of retried.
    #[tokio::test]
    async fn provider_warning_does_not_commit_attempt() {
        let provider = ScriptedProvider::new(vec![
            vec![
                Ok(StreamEvent::ProviderWarning {
                    message: "provider degraded".into(),
                }),
                Err(transient_error("upstream reset")),
            ],
            success_events("msg_after_warning", "done"),
        ]);
        let attempts = provider.attempts();
        let resilient = ResilientProvider::new("test", provider, test_policy(3));

        let events = collect_events(resilient).await;

        assert_eq!(
            attempts.load(Ordering::SeqCst),
            2,
            "warning must not foreclose the retry"
        );
        assert!(events.iter().any(|event| matches!(
            event,
            StreamEvent::TextDelta { delta, .. } if delta == "done"
        )));
    }

    /// M13: the worker-to-consumer channel is bounded. A consumer that never
    /// polls must stall the worker after roughly the channel capacity, not
    /// let it drain (and buffer) the entire upstream.
    #[tokio::test]
    async fn bounded_channel_applies_backpressure_to_stalled_consumer() {
        let total_events = 1_000usize;
        let mut script: Vec<Result<StreamEvent, ProviderError>> =
            vec![Ok(StreamEvent::MessageStart {
                id: MessageId::from("msg_backpressure"),
            })];
        script.extend((0..total_events).map(|index| {
            Ok(StreamEvent::TextDelta {
                id: BlockId::from("text_backpressure"),
                delta: format!("chunk {index}"),
            })
        }));
        let provider = ScriptedProvider::new(vec![script]);
        let produced = provider.produced_events();
        let resilient = ResilientProvider::new("test", provider, test_policy(1));

        let stream = resilient
            .stream(sample_request(), CancellationToken::new())
            .await
            .expect("stream");

        // Give the worker ample time to pull as much as the channel allows
        // while the consumer never polls.
        tokio::time::sleep(Duration::from_millis(100)).await;
        let pulled = produced.load(Ordering::SeqCst);
        assert!(
            pulled <= EVENT_CHANNEL_CAPACITY + 8,
            "worker pulled {pulled} events despite a stalled consumer"
        );

        // Draining the consumer must release the backpressure and deliver
        // every event without loss.
        let events: Vec<_> = stream.collect().await;
        assert_eq!(events.len(), total_events + 1);
    }

    /// M13: a stalled consumer must not wedge the worker past cancellation —
    /// the blocked send races the token and the stream terminates.
    #[tokio::test]
    async fn cancellation_unblocks_worker_wedged_on_stalled_consumer() {
        let mut script: Vec<Result<StreamEvent, ProviderError>> =
            vec![Ok(StreamEvent::MessageStart {
                id: MessageId::from("msg_wedged"),
            })];
        script.extend((0..1_000).map(|index| {
            Ok(StreamEvent::TextDelta {
                id: BlockId::from("text_wedged"),
                delta: format!("chunk {index}"),
            })
        }));
        let provider = ScriptedProvider::new(vec![script]);
        let resilient = ResilientProvider::new("test", provider, test_policy(1));
        let cancel = CancellationToken::new();

        let stream = resilient
            .stream(sample_request(), cancel.clone())
            .await
            .expect("stream");
        tokio::time::sleep(Duration::from_millis(50)).await;
        cancel.cancel();

        // The worker must exit (dropping its sender), so draining the
        // buffered items terminates promptly instead of hanging.
        let drained = tokio::time::timeout(Duration::from_secs(1), stream.collect::<Vec<_>>())
            .await
            .expect("worker must exit after cancellation");
        assert!(drained.len() <= EVENT_CHANNEL_CAPACITY + 8);
    }

    #[derive(Debug)]
    struct ScriptedCompactProvider {
        attempts: Arc<AtomicUsize>,
        results: Arc<Mutex<VecDeque<anyhow::Result<ProviderCompactionResponse>>>>,
    }

    impl ScriptedCompactProvider {
        fn new(results: Vec<anyhow::Result<ProviderCompactionResponse>>) -> Self {
            Self {
                attempts: Arc::new(AtomicUsize::new(0)),
                results: Arc::new(Mutex::new(results.into())),
            }
        }
    }

    #[async_trait]
    impl Provider for ScriptedCompactProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }

        async fn stream(
            &self,
            _request: ProviderRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
            Ok(stream::empty().boxed())
        }

        async fn compact(
            &self,
            _request: ProviderCompactionRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<ProviderCompactionResponse> {
            self.attempts.fetch_add(1, Ordering::SeqCst);
            self.results
                .lock()
                .expect("results mutex")
                .pop_front()
                .unwrap_or_else(|| anyhow::bail!("script exhausted"))
        }
    }

    fn compaction_response() -> ProviderCompactionResponse {
        ProviderCompactionResponse {
            output: Vec::new(),
            usage: Default::default(),
        }
    }

    fn sample_compaction_request() -> ProviderCompactionRequest {
        ProviderCompactionRequest {
            session_id: SessionId::new(),
            model: sample_request().model,
            compacted_prefix: Vec::new(),
            messages: Vec::new(),
            tools: Vec::new(),
            instructions: "Summarize".to_owned(),
        }
    }

    #[tokio::test]
    async fn compact_returns_first_success() {
        let provider = ScriptedCompactProvider::new(vec![Ok(compaction_response())]);
        let attempts = provider.attempts.clone();
        let resilient = ResilientProvider::new("test", provider, test_policy(3));

        resilient
            .compact(sample_compaction_request(), CancellationToken::new())
            .await
            .expect("compaction succeeds");

        assert_eq!(attempts.load(Ordering::SeqCst), 1);
    }

    /// M11: `compact` now gets the same retry treatment as `stream` — a
    /// typed retryable failure is retried. Previously compact delegated
    /// directly and any failure was final.
    #[tokio::test]
    async fn compact_retries_typed_retryable_failures() {
        let provider = ScriptedCompactProvider::new(vec![
            Err(anyhow::Error::new(ProviderError::with_kind(
                "rate limited",
                ProviderErrorKind::RateLimited,
            ))),
            Ok(compaction_response()),
        ]);
        let attempts = provider.attempts.clone();
        let resilient = ResilientProvider::new("test", provider, test_policy(3));

        resilient
            .compact(sample_compaction_request(), CancellationToken::new())
            .await
            .expect("compaction succeeds after retry");

        assert_eq!(attempts.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn compact_exhausts_retry_budget_and_returns_typed_error() {
        let provider = ScriptedCompactProvider::new(vec![
            Err(anyhow::Error::new(ProviderError::with_kind(
                "first",
                ProviderErrorKind::Transient,
            ))),
            Err(anyhow::Error::new(ProviderError::with_kind(
                "final",
                ProviderErrorKind::Transient,
            ))),
        ]);
        let attempts = provider.attempts.clone();
        let resilient = ResilientProvider::new("test", provider, test_policy(2));

        let error = resilient
            .compact(sample_compaction_request(), CancellationToken::new())
            .await
            .expect_err("budget exhaustion should fail");

        assert_eq!(attempts.load(Ordering::SeqCst), 2);
        let provider_error = error
            .downcast::<ProviderError>()
            .expect("typed provider error");
        assert_eq!(provider_error.message, "final");
    }

    /// Untyped compact failures (e.g. the trait's own "does not support
    /// compaction" fallback) must not be retry-looped.
    #[tokio::test]
    async fn compact_does_not_retry_untyped_or_fatal_failures() {
        for scripted in [
            anyhow::anyhow!("failed to compact session: provider does not support compaction"),
            anyhow::Error::new(ProviderError::with_kind(
                "bad request",
                ProviderErrorKind::Fatal,
            )),
        ] {
            let provider = ScriptedCompactProvider::new(vec![Err(scripted)]);
            let attempts = provider.attempts.clone();
            let resilient = ResilientProvider::new("test", provider, test_policy(3));

            let error = resilient
                .compact(sample_compaction_request(), CancellationToken::new())
                .await
                .expect_err("fatal compaction should fail");

            assert_eq!(attempts.load(Ordering::SeqCst), 1, "{error:#}");
        }
    }

    #[derive(Debug)]
    struct HangingCompactProvider;

    #[async_trait]
    impl Provider for HangingCompactProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }

        async fn stream(
            &self,
            _request: ProviderRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
            Ok(stream::empty().boxed())
        }

        async fn compact(
            &self,
            _request: ProviderCompactionRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<ProviderCompactionResponse> {
            std::future::pending().await
        }
    }

    #[tokio::test]
    async fn compact_times_out_hanging_requests() {
        let resilient = ResilientProvider::new("test", HangingCompactProvider, timeout_policy());

        let error = resilient
            .compact(sample_compaction_request(), CancellationToken::new())
            .await
            .expect_err("hanging compaction should time out");

        let provider_error = error
            .downcast::<ProviderError>()
            .expect("typed provider error");
        assert_eq!(provider_error.kind, ProviderErrorKind::Transient);
        assert!(provider_error.message.contains("timed out"));
    }

    #[tokio::test]
    async fn compact_honors_pre_cancelled_token() {
        let provider = ScriptedCompactProvider::new(vec![Ok(compaction_response())]);
        let attempts = provider.attempts.clone();
        let resilient = ResilientProvider::new("test", provider, test_policy(3));
        let cancel = CancellationToken::new();
        cancel.cancel();

        let error = resilient
            .compact(sample_compaction_request(), cancel)
            .await
            .expect_err("cancelled compaction should fail");

        assert!(
            error
                .downcast::<ProviderError>()
                .expect("typed provider error")
                .is_cancelled()
        );
        assert_eq!(attempts.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn cancel_on_drop_cancels_token_on_drop() {
        let token = CancellationToken::new();
        {
            let _wrapper = CancelOnDrop::new(futures::stream::empty::<i32>(), token.clone());
            assert!(!token.is_cancelled());
        }
        assert!(token.is_cancelled());
    }

    #[tokio::test]
    async fn cancel_on_drop_passes_through_stream_items() {
        let token = CancellationToken::new();
        let mut wrapper = CancelOnDrop::new(futures::stream::iter(vec![1, 2, 3]), token.clone());
        assert_eq!(wrapper.next().await, Some(1));
        assert_eq!(wrapper.next().await, Some(2));
        assert_eq!(wrapper.next().await, Some(3));
        assert_eq!(wrapper.next().await, None);
        assert!(!token.is_cancelled());
        drop(wrapper);
        assert!(token.is_cancelled());
    }

    #[test]
    fn commit_dedup_suppresses_duplicate_message_start() {
        let mut dedup = CommitDedup::default();
        let id = MessageId::from("msg_alpha");
        let other = MessageId::from("msg_beta");

        assert!(dedup.allow(&StreamEvent::MessageStart { id: id.clone() }));
        assert!(!dedup.allow(&StreamEvent::MessageStart { id: id.clone() }));
        assert!(dedup.allow(&StreamEvent::MessageStart { id: other }));
    }

    #[test]
    fn commit_dedup_passes_through_non_message_start_events() {
        let mut dedup = CommitDedup::default();
        let block = BlockId::from("blk_one");
        let event = StreamEvent::TextStart { id: block.clone() };
        assert!(dedup.allow(&event));
        assert!(dedup.allow(&event));
        assert!(dedup.allow(&StreamEvent::TextEnd { id: block }));
    }

    async fn collect_events(
        provider: ResilientProvider<impl Provider + 'static>,
    ) -> Vec<StreamEvent> {
        let mut stream = provider
            .stream(sample_request(), CancellationToken::new())
            .await
            .expect("stream");
        let mut events = Vec::new();
        while let Some(item) = stream.next().await {
            events.push(item.expect("stream event"));
        }
        events
    }

    async fn collect_items(
        provider: ResilientProvider<impl Provider + 'static>,
        cancel: CancellationToken,
    ) -> Vec<Result<StreamEvent, ProviderError>> {
        let mut stream = provider
            .stream(sample_request(), cancel)
            .await
            .expect("stream");
        let mut items = Vec::new();
        while let Some(item) = stream.next().await {
            items.push(item);
        }
        items
    }

    fn success_events(message_id: &str, text: &str) -> Vec<Result<StreamEvent, ProviderError>> {
        let message_id = MessageId::from(message_id);
        let block_id = BlockId::from(format!("text_{message_id}"));
        vec![
            Ok(StreamEvent::MessageStart {
                id: message_id.clone(),
            }),
            Ok(StreamEvent::TextDelta {
                id: block_id,
                delta: text.to_owned(),
            }),
            Ok(StreamEvent::MessageEnd {
                id: message_id,
                stop_reason: StopReason::EndTurn,
                response_id: None,
            }),
        ]
    }

    fn transient_error(message: &str) -> ProviderError {
        ProviderError::with_kind(message, ProviderErrorKind::Transient)
    }

    fn test_policy(max_attempts: u32) -> ResiliencePolicy {
        ResiliencePolicy {
            timeouts: ProviderTimeouts {
                connect: Duration::from_secs(1),
                request: Duration::from_secs(1),
                stream_idle: Duration::from_secs(1),
            },
            request_retry: RetryPolicy {
                max_attempts,
                base_backoff: Duration::from_millis(1),
                max_backoff: Duration::from_millis(1),
                deadline: Duration::from_secs(5),
                jitter_pct: 0,
            },
        }
    }

    fn timeout_policy() -> ResiliencePolicy {
        ResiliencePolicy {
            timeouts: ProviderTimeouts {
                connect: Duration::from_millis(20),
                request: Duration::from_millis(20),
                stream_idle: Duration::from_millis(20),
            },
            request_retry: RetryPolicy {
                max_attempts: 1,
                base_backoff: Duration::from_millis(1),
                max_backoff: Duration::from_millis(1),
                deadline: Duration::from_millis(50),
                jitter_pct: 0,
            },
        }
    }

    fn sample_request() -> ProviderRequest {
        ProviderRequest {
            session_id: SessionId::new(),
            turn_id: TurnId::new(),
            model: ResolvedModel {
                role: ModelRole::default(),
                id: ModelId::from("default"),
                provider: ProviderName::from("test"),
                provider_kind: ProviderKind::OpenAi,
                api_kind: ApiKind::OpenAiResponses,
                model: "test-model".to_owned(),
                max_input_tokens: None,
                max_output_tokens: None,
                reasoning: None,
                tokens_per_minute: None,
            },
            prompt: AssembledPrompt {
                segments: Vec::new(),
                transcript: Vec::new(),
                ordered_segments: Vec::new(),
                prefix_cache_key: String::new(),
                rendered_prefix: String::new(),
                rendered_transcript: String::new(),
                rendered: String::new(),
                cache_breakpoints: CacheBreakpoints::default(),
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
}
