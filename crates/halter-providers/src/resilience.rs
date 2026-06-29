// pattern: Imperative Shell

use std::collections::HashSet;
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use async_trait::async_trait;
use futures::{Stream, StreamExt, channel::mpsc, stream::BoxStream};
use halter_protocol::{
    CompactionWindow, Message, MessageId, ProviderCapabilities, ProviderCompactionRequest,
    ProviderCompactionResponse, ProviderError, ProviderErrorKind, ProviderRequest, StreamEvent,
};
use tokio::select;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::Provider;
use crate::retry::{RetryGate, RetryPolicy};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResiliencePolicy {
    pub timeouts: ProviderTimeouts,
    pub request_retry: RetryPolicy,
}

impl Default for ResiliencePolicy {
    fn default() -> Self {
        Self {
            timeouts: ProviderTimeouts::default(),
            request_retry: RetryPolicy::default(),
        }
    }
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
        let (tx, rx) = mpsc::unbounded();
        let stream_cancel = cancel.child_token();
        let task_cancel = stream_cancel.clone();
        let inner = self.inner.clone();
        let policy = self.policy;
        let classifier = self.classifier.clone();
        let label = self.label;

        tokio::spawn(async move {
            let cancel = task_cancel;
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
                        let error = classifier.classify(ProviderError::with_kind(
                            format!("{error:#}"),
                            ProviderErrorKind::Transient,
                        ));
                        attempt_cancel.cancel();
                        if !retry_or_emit(
                            RetryContext {
                                label,
                                attempt_id,
                                gate: &mut gate,
                                tx: &tx,
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
                                tx: &tx,
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
                            let _ = tx.unbounded_send(Err(ProviderError::cancelled()));
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
                                        tx: &tx,
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
                            let _ = tx.unbounded_send(Err(error));
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
                                        tx: &tx,
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
                            let _ = tx.unbounded_send(Err(error));
                            return;
                        }
                        Ok(Some(Ok(event))) => {
                            if committed {
                                if commit_dedup.allow(&event)
                                    && tx.unbounded_send(Ok(event)).is_err()
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
                                    if tx.unbounded_send(Ok(event)).is_err() {
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
                                        tx: &tx,
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
                            let _ = tx.unbounded_send(Err(error));
                            return;
                        }
                        Ok(None) => {
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
                            debug!(provider = label, "provider stream completed");
                            return;
                        }
                    }
                }
            }
        });

        Ok(CancelOnDrop::new(rx.boxed(), stream_cancel).boxed())
    }

    async fn compact(
        &self,
        request: ProviderCompactionRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<ProviderCompactionResponse> {
        self.inner.compact(request, cancel).await
    }
}

struct RetryContext<'a> {
    label: &'static str,
    attempt_id: u32,
    gate: &'a mut RetryGate,
    tx: &'a mpsc::UnboundedSender<Result<StreamEvent, ProviderError>>,
    cancel: &'a CancellationToken,
}

async fn retry_or_emit(context: RetryContext<'_>, error: ProviderError) -> bool {
    if !error.retryable() {
        let _ = context.tx.unbounded_send(Err(error));
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
                    let _ = context.tx.unbounded_send(Err(ProviderError::cancelled()));
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
            let _ = context.tx.unbounded_send(Err(error));
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

/// Dedup for repeated commit-eligible boundary events within a committed stream.
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

    #[derive(Debug)]
    struct ScriptedProvider {
        attempts: Arc<AtomicUsize>,
        scripts: Arc<Mutex<VecDeque<Vec<Result<StreamEvent, ProviderError>>>>>,
    }

    impl ScriptedProvider {
        fn new(scripts: Vec<Vec<Result<StreamEvent, ProviderError>>>) -> Self {
            Self {
                attempts: Arc::new(AtomicUsize::new(0)),
                scripts: Arc::new(Mutex::new(scripts.into())),
            }
        }

        fn attempts(&self) -> Arc<AtomicUsize> {
            self.attempts.clone()
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
            Ok(stream::iter(script).boxed())
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
