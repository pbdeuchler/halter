// pattern: Integration Tests

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use futures::{StreamExt, TryStreamExt, stream};
use halter_protocol::{
    ApiKind, BlockId, Message, MessageId, ModelId, ModelRole, ProviderCapabilities, ProviderError,
    ProviderKind, ProviderName, ProviderRequest, ResolvedModel, SessionEvent, SessionEventPayload,
    SessionState, SessionStatus, StreamEvent,
};
use halter_providers::{FakeProvider, ModelRegistry, Provider};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

use crate::{
    EventBus, RuntimeServices, SessionError, SessionEventStream, SessionInit, SessionRuntime,
};

fn services(provider: Arc<dyn Provider>) -> Arc<RuntimeServices> {
    let mut models = ModelRegistry::new();
    let model = ResolvedModel {
        role: ModelRole::default(),
        id: ModelId::from("default"),
        provider: ProviderName::from("test"),
        provider_kind: ProviderKind::Fake,
        api_kind: ApiKind::Fake,
        model: "halter/test".into(),
        max_input_tokens: Some(32_000),
        max_output_tokens: Some(4_096),
        reasoning: None,
        tokens_per_minute: None,
    };
    models.set_default_model(model.clone());
    models.set_subagent_model(model.clone());
    models.set_small_model(model);
    models.register_provider(ProviderName::from("test"), provider);
    Arc::new(RuntimeServices {
        models: Arc::new(models),
        event_bus: Arc::new(EventBus::new(2)),
        ..Default::default()
    })
}

async fn until_status(events: &mut SessionEventStream, status: SessionStatus) -> Vec<SessionEvent> {
    tokio::time::timeout(Duration::from_secs(5), async {
        let mut seen = Vec::new();
        loop {
            let event = events.next().await.expect("live stream").expect("event");
            let matched = matches!(event.payload, SessionEventPayload::SessionStatusChanged { status: current } if current == status);
            seen.push(event);
            if matched { return seen; }
        }
    }).await.expect("session status")
}

#[tokio::test]
async fn session_stream_spans_submissions_and_reopens_idle_with_stale_handles_closed() {
    let temp = tempfile::tempdir().unwrap();
    let services = services(Arc::new(FakeProvider::default()));
    let runtime = SessionRuntime::new(services.clone());
    let (session, mut events) = runtime
        .create_session(SessionInit {
            working_dir: temp.path().into(),
            ..Default::default()
        })
        .await
        .unwrap();
    let stale = session.clone();
    let mut observed = until_status(&mut events, SessionStatus::Idle).await;
    for text in ["first", "second"] {
        let message = Message::user(text);
        let id = session.submit(message.clone()).await.unwrap();
        observed.extend(until_status(&mut events, SessionStatus::Running).await);
        observed.extend(until_status(&mut events, SessionStatus::Idle).await);
        assert_eq!(
            session.submit(message).await.unwrap(),
            id,
            "same id is acknowledged without executing twice"
        );
    }
    session.shutdown().await.unwrap();
    observed.extend(events.try_collect::<Vec<_>>().await.unwrap());
    let log = session.replay().await.unwrap();
    assert_eq!(
        observed
            .iter()
            .map(SessionEvent::sequence)
            .collect::<Vec<_>>(),
        log.iter().map(SessionEvent::sequence).collect::<Vec<_>>()
    );
    assert_eq!(
        log.iter()
            .filter(|e| matches!(e.payload, SessionEventPayload::InputAccepted { .. }))
            .count(),
        2
    );
    assert!(matches!(
        stale.submit(Message::user("stale")).await,
        Err(SessionError::Closed)
    ));
    let (reopened, mut events) = runtime.resume_session(session.id()).await.unwrap();
    until_status(&mut events, SessionStatus::Idle).await;
    assert_eq!(reopened.id(), session.id());
    assert!(matches!(
        runtime.resume_session(session.id()).await,
        Err(SessionError::AlreadyOpen(_))
    ));
    reopened.shutdown().await.unwrap();
}

struct BlockingFirst {
    calls: AtomicUsize,
    started: Arc<Notify>,
    requests: std::sync::Mutex<Vec<ProviderRequest>>,
    block_creation: bool,
}

impl BlockingFirst {
    fn new(block_creation: bool) -> Self {
        Self {
            calls: AtomicUsize::new(0),
            started: Arc::new(Notify::new()),
            requests: Default::default(),
            block_creation,
        }
    }
}

#[async_trait]
impl Provider for BlockingFirst {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::default()
    }
    async fn stream(
        &self,
        request: ProviderRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<stream::BoxStream<'static, Result<StreamEvent, ProviderError>>> {
        self.requests.lock().unwrap().push(request.clone());
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            if self.block_creation {
                self.started.notify_one();
                return std::future::pending().await;
            }
            let block = BlockId::new();
            let prefix = stream::iter(vec![
                Ok(StreamEvent::MessageStart {
                    id: MessageId::new(),
                }),
                Ok(StreamEvent::TextStart { id: block.clone() }),
                Ok(StreamEvent::TextDelta {
                    id: block,
                    delta: "partial reply".into(),
                }),
            ]);
            let started = self.started.clone();
            let tail = stream::once(async move {
                started.notify_one();
                std::future::pending::<Result<StreamEvent, ProviderError>>().await
            });
            return Ok(prefix.chain(tail).boxed());
        }
        FakeProvider::default().stream(request, cancel).await
    }
}

#[tokio::test]
async fn interrupt_awaits_finalization_preserves_pending_input_and_next_submit_continues() {
    let temp = tempfile::tempdir().unwrap();
    let provider = Arc::new(BlockingFirst::new(false));
    let services = services(provider.clone());
    let runtime = SessionRuntime::new(services.clone());
    let (session, mut events) = runtime
        .create_session(SessionInit {
            working_dir: temp.path().into(),
            ..Default::default()
        })
        .await
        .unwrap();
    session.submit(Message::user("start")).await.unwrap();
    provider.started.notified().await;
    // Observe the first request has consumed its text before interrupting it.
    // Input admission is serialized after the executor's opening commits.
    let queued = session
        .submit(Message::user("queued correction"))
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), session.interrupt())
        .await
        .expect("uncooperative stream is cancellable")
        .unwrap();
    let mut stored = services
        .sessions
        .load_session(session.id())
        .await
        .unwrap()
        .unwrap();
    crate::session::hydrate_stored_session(services.sessions.as_ref(), &mut stored)
        .await
        .unwrap();
    assert!(stored.state.open_turn.is_none());
    assert_eq!(
        stored
            .state
            .pending_inputs
            .iter()
            .map(|m| &m.id)
            .collect::<Vec<_>>(),
        vec![&queued]
    );
    assert!(stored.state.messages.iter().any(|message| matches!(message, Message::Assistant(a) if a.parts.iter().any(|part| matches!(part, halter_protocol::AssistantPart::Text { text } if text == "partial reply")))));
    assert_eq!(
        provider.calls.load(Ordering::SeqCst),
        1,
        "interrupt does not restart accepted input"
    );
    // Drain the interrupted execution, then continue on the same stream.
    until_status(&mut events, SessionStatus::Running).await;
    until_status(&mut events, SessionStatus::Idle).await;
    session.submit(Message::user("continue")).await.unwrap();
    until_status(&mut events, SessionStatus::Running).await;
    until_status(&mut events, SessionStatus::Idle).await;
    let requests = provider.requests.lock().unwrap().clone();
    let texts = requests
        .last()
        .unwrap()
        .messages
        .iter()
        .filter_map(|m| match m {
            Message::User(u) => Some(u.plain_text()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert!(texts.contains(&"queued correction".to_owned()));
    assert!(texts.contains(&"continue".to_owned()));
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn inbox_capacity_is_bounded_while_provider_creation_is_blocked() {
    let provider = Arc::new(BlockingFirst::new(true));
    let runtime = SessionRuntime::new(services(provider.clone()));
    let (session, _) = runtime
        .create_session(SessionInit::default())
        .await
        .unwrap();
    session.submit(Message::user("start")).await.unwrap();
    provider.started.notified().await;
    for i in 0..128 {
        session
            .submit(Message::user(format!("queued {i}")))
            .await
            .unwrap();
    }
    assert!(matches!(
        session.submit(Message::user("overflow")).await,
        Err(SessionError::InboxFull)
    ));
    tokio::time::timeout(Duration::from_secs(5), session.interrupt())
        .await
        .expect("provider creation cancels")
        .unwrap();
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn lagged_stream_replays_every_committed_event_once_and_runtime_closes_idle_sessions() {
    let runtime = SessionRuntime::new(services(Arc::new(FakeProvider::default())));
    let (session, events) = runtime
        .create_session(SessionInit::default())
        .await
        .unwrap();
    for i in 0..20 {
        session
            .submit(Message::user(format!("message {i}")))
            .await
            .unwrap();
    }
    session.interrupt().await.unwrap();
    let report = runtime.shutdown(Duration::from_secs(5)).await;
    assert!(!report.timed_out);
    let received = tokio::time::timeout(Duration::from_secs(5), events.try_collect::<Vec<_>>())
        .await
        .unwrap()
        .unwrap();
    let log = session.replay().await.unwrap();
    assert_eq!(
        received
            .iter()
            .map(SessionEvent::sequence)
            .collect::<Vec<_>>(),
        log.iter().map(SessionEvent::sequence).collect::<Vec<_>>()
    );
    assert!(matches!(
        log.last().unwrap().payload,
        SessionEventPayload::SessionStatusChanged {
            status: SessionStatus::Closed
        }
    ));
    assert!(matches!(
        session.submit(Message::user("closed")).await,
        Err(SessionError::Closed)
    ));
    assert!(matches!(
        runtime.create_session(SessionInit::default()).await,
        Err(SessionError::Closed)
    ));
}

struct AdmissionFailure {
    store: halter_session::InMemorySessionStore,
    fail: AtomicBool,
    fail_running: AtomicBool,
    fail_idle: AtomicBool,
    fail_terminal: AtomicBool,
    opening: Option<(Arc<Notify>, Arc<tokio::sync::Semaphore>)>,
}

#[async_trait]
impl halter_session::SessionStore for AdmissionFailure {
    async fn create_session(&self, session: halter_session::StoredSession) -> anyhow::Result<()> {
        if let Some((started, release)) = &self.opening {
            started.notify_one();
            let _permit = release.acquire().await.unwrap();
        }
        self.store.create_session(session).await
    }
    async fn load_session(
        &self,
        id: &halter_protocol::SessionId,
    ) -> anyhow::Result<Option<halter_session::StoredSession>> {
        self.store.load_session(id).await
    }
    async fn commit(
        &self,
        id: &halter_protocol::SessionId,
        snapshot: Option<Arc<halter_protocol::ResourceSnapshot>>,
        expected: Option<u64>,
        state: Option<SessionState>,
        events: Vec<halter_protocol::PendingEvent>,
    ) -> anyhow::Result<Vec<SessionEvent>> {
        if self.fail_terminal.load(Ordering::SeqCst)
            && events.iter().any(|event| {
                matches!(
                    event.payload,
                    SessionEventPayload::TurnCompleted { .. }
                        | SessionEventPayload::TurnFailed { .. }
                )
            })
        {
            anyhow::bail!("terminal execution commit unavailable");
        }
        if events.iter().any(|event| {
            matches!(
                event.payload,
                SessionEventPayload::SessionStatusChanged {
                    status: SessionStatus::Running
                }
            )
        }) && self.fail_running.swap(false, Ordering::SeqCst)
        {
            anyhow::bail!("execution status commit unavailable");
        }
        if events.iter().any(|event| {
            matches!(
                event.payload,
                SessionEventPayload::SessionStatusChanged {
                    status: SessionStatus::Idle
                }
            )
        }) && self.fail_idle.swap(false, Ordering::SeqCst)
        {
            anyhow::bail!("idle status commit unavailable");
        }
        if self.fail.load(Ordering::SeqCst)
            && events
                .iter()
                .any(|e| matches!(e.payload, SessionEventPayload::InputAccepted { .. }))
        {
            anyhow::bail!("admission storage unavailable");
        }
        self.store
            .commit(id, snapshot, expected, state, events)
            .await
    }
    async fn replay(&self, id: &halter_protocol::SessionId) -> anyhow::Result<Vec<SessionEvent>> {
        self.store.replay(id).await
    }
    async fn list_sessions(&self) -> anyhow::Result<Vec<halter_protocol::SessionBlueprint>> {
        self.store.list_sessions().await
    }
}

#[tokio::test]
async fn failed_admission_is_not_acknowledged_or_recorded_and_can_be_retried() {
    let store = Arc::new(AdmissionFailure {
        store: Default::default(),
        fail: AtomicBool::new(true),
        fail_running: AtomicBool::new(false),
        fail_idle: AtomicBool::new(false),
        fail_terminal: AtomicBool::new(false),
        opening: None,
    });
    let mut services = (*services(Arc::new(FakeProvider::default()))).clone();
    services.sessions = store.clone();
    let runtime = SessionRuntime::new(Arc::new(services));
    let (session, mut events) = runtime
        .create_session(SessionInit::default())
        .await
        .unwrap();
    let message = Message::user("retry me");
    assert!(matches!(
        session.submit(message.clone()).await,
        Err(SessionError::Operation(_))
    ));
    assert!(
        !session
            .replay()
            .await
            .unwrap()
            .iter()
            .any(|e| matches!(e.payload, SessionEventPayload::InputAccepted { .. }))
    );
    store.fail.store(false, Ordering::SeqCst);
    session.submit(message).await.unwrap();
    until_status(&mut events, SessionStatus::Running).await;
    until_status(&mut events, SessionStatus::Idle).await;
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn explicit_compaction_finishes_idle_and_subsequent_submission_has_its_own_statuses() {
    let temp = tempfile::tempdir().unwrap();
    let mut services = (*services(Arc::new(FakeProvider::default()))).clone();
    services.compaction = Arc::new(crate::ProviderDefault);
    let runtime = SessionRuntime::new(Arc::new(services));
    let (session, mut events) = runtime
        .create_session(SessionInit {
            working_dir: temp.path().into(),
            ..Default::default()
        })
        .await
        .unwrap();
    session
        .submit(Message::user("before compaction"))
        .await
        .unwrap();
    until_status(&mut events, SessionStatus::Running).await;
    until_status(&mut events, SessionStatus::Idle).await;
    session.compact("manual", None).await.unwrap();
    let log = session.replay().await.unwrap();
    assert!(matches!(
        log.last().unwrap().payload,
        SessionEventPayload::SessionStatusChanged {
            status: SessionStatus::Idle
        }
    ));
    let id = session
        .submit(Message::user("after compaction"))
        .await
        .unwrap();
    loop {
        let event = events.next().await.unwrap().unwrap();
        if matches!(event.payload, SessionEventPayload::InputAccepted { message } if message.id == id)
        {
            break;
        }
    }
    until_status(&mut events, SessionStatus::Running).await;
    let delivered = until_status(&mut events, SessionStatus::Idle).await;
    assert!(delivered.iter().any(|event| matches!(&event.payload, SessionEventPayload::MessageItem { message: Message::User(message) } if message.id == id)));
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn cancelled_open_releases_reservation_and_runtime_shutdown_refuses_delayed_open() {
    for cancel_open in [true, false] {
        let started = Arc::new(Notify::new());
        let release = Arc::new(tokio::sync::Semaphore::new(0));
        let store = Arc::new(AdmissionFailure {
            store: Default::default(),
            fail: AtomicBool::new(false),
            fail_running: AtomicBool::new(false),
            fail_idle: AtomicBool::new(false),
            fail_terminal: AtomicBool::new(false),
            opening: Some((started.clone(), release.clone())),
        });
        let mut services = (*services(Arc::new(FakeProvider::default()))).clone();
        services.sessions = store;
        let runtime = SessionRuntime::new(Arc::new(services));
        let opening = runtime.clone();
        let id = halter_protocol::SessionId::new();
        let init = SessionInit {
            session_id: Some(id.clone()),
            ..Default::default()
        };
        let task_init = init.clone();
        let task = tokio::spawn(async move { opening.create_session(task_init).await });
        started.notified().await;
        if cancel_open {
            task.abort();
            assert!(matches!(task.await, Err(error) if error.is_cancelled()));
            release.add_permits(1);
            let (session, _) = runtime
                .create_session(init)
                .await
                .expect("cancelled opener releases its reservation");
            session.shutdown().await.unwrap();
        } else {
            let report = runtime.shutdown(Duration::from_secs(5)).await;
            assert!(!report.timed_out);
            release.add_permits(1);
            assert!(matches!(task.await.unwrap(), Err(SessionError::Closed)));
        }
    }
}

#[tokio::test]
async fn concurrent_shutdown_calls_are_idempotent() {
    let runtime = SessionRuntime::new(services(Arc::new(FakeProvider::default())));
    let (session, _) = runtime
        .create_session(SessionInit::default())
        .await
        .unwrap();
    let results = futures::future::join_all((0..32).map(|_| session.shutdown())).await;
    assert!(results.into_iter().all(|result| result.is_ok()));
    session.shutdown().await.unwrap();
}

#[tokio::test]
async fn driver_storage_failure_closes_stream_cleans_up_and_keeps_input_resumable() {
    let store = Arc::new(AdmissionFailure {
        store: Default::default(),
        fail: AtomicBool::new(false),
        fail_running: AtomicBool::new(true),
        fail_idle: AtomicBool::new(false),
        fail_terminal: AtomicBool::new(false),
        opening: None,
    });
    let mut services = (*services(Arc::new(FakeProvider::default()))).clone();
    services.sessions = store;
    let runtime = SessionRuntime::new(Arc::new(services));
    let (session, events) = runtime
        .create_session(SessionInit::default())
        .await
        .unwrap();
    let message = Message::user("durable input");
    session.submit(message.clone()).await.unwrap();
    let result = tokio::time::timeout(Duration::from_secs(5), events.try_collect::<Vec<_>>())
        .await
        .unwrap();
    assert!(
        result.is_err(),
        "driver failure must reach stream consumers"
    );
    assert!(matches!(
        session.shutdown().await,
        Err(SessionError::Operation(_))
    ));
    assert!(
        session
            .replay()
            .await
            .unwrap()
            .iter()
            .any(|event| matches!(event.payload, SessionEventPayload::SessionShutdownComplete)),
        "cleanup commits use storage after the failed actor stops"
    );
    let (reopened, mut events) = runtime.resume_session(session.id()).await.unwrap();
    reopened.submit(message).await.unwrap();
    until_status(&mut events, SessionStatus::Running).await;
    let delivered = until_status(&mut events, SessionStatus::Idle).await;
    assert!(delivered.iter().any(|event| matches!(&event.payload, SessionEventPayload::MessageItem { message: Message::User(message) } if message.plain_text() == "durable input")));
    reopened.shutdown().await.unwrap();
}

#[tokio::test]
async fn terminal_execution_commit_failure_closes_the_stream_and_cleans_up_the_driver() {
    for interrupted in [false, true] {
        let provider = Arc::new(BlockingFirst::new(true));
        let execution_provider: Arc<dyn Provider> = if interrupted {
            provider.clone()
        } else {
            Arc::new(FakeProvider::default())
        };
        let store = Arc::new(AdmissionFailure {
            store: Default::default(),
            fail: AtomicBool::new(false),
            fail_running: AtomicBool::new(false),
            fail_idle: AtomicBool::new(false),
            fail_terminal: AtomicBool::new(true),
            opening: None,
        });
        let mut services = (*services(execution_provider)).clone();
        services.sessions = store;
        let runtime = SessionRuntime::new(Arc::new(services));
        let (session, events) = runtime
            .create_session(SessionInit::default())
            .await
            .unwrap();
        session
            .submit(Message::user("persist this result"))
            .await
            .unwrap();
        if interrupted {
            tokio::time::timeout(Duration::from_secs(5), provider.started.notified())
                .await
                .expect("foreground provider starts");
            let error = tokio::time::timeout(Duration::from_secs(5), session.interrupt())
                .await
                .expect("failed finalization settles interrupt")
                .expect_err("interrupt must report its failed finalization");
            assert!(matches!(error, SessionError::Operation(_)));
            assert!(
                error
                    .to_string()
                    .contains("terminal execution commit unavailable")
            );
        }
        let error = tokio::time::timeout(Duration::from_secs(5), events.try_collect::<Vec<_>>())
            .await
            .expect("failed finalization closes the session stream")
            .expect_err("continuous stream must expose storage failure");
        assert!(
            error
                .to_string()
                .contains("terminal execution commit unavailable")
        );
        assert!(matches!(
            session.shutdown().await,
            Err(SessionError::Operation(_))
        ));
        assert!(matches!(
            session.submit(Message::user("closed")).await,
            Err(SessionError::Closed)
        ));
        let log = session.replay().await.unwrap();
        assert!(
            log.iter()
                .any(|event| matches!(event.payload, SessionEventPayload::SessionShutdownComplete))
        );
        assert!(!log.iter().any(|event| matches!(
            event.payload,
            SessionEventPayload::TurnCompleted { .. } | SessionEventPayload::TurnFailed { .. }
        )));
    }
}

struct BlockingCompaction {
    started: Arc<Notify>,
}

#[async_trait]
impl crate::CompactionStrategy for BlockingCompaction {
    async fn compact(
        &self,
        context: crate::CompactionContext<'_>,
    ) -> anyhow::Result<Option<crate::CompactionEffects>> {
        self.started.notify_one();
        context.cancel().cancelled().await;
        Ok(None)
    }
}

#[tokio::test]
async fn compaction_idle_commit_failure_reaches_interrupt_compact_and_stream_after_cleanup() {
    let started = Arc::new(Notify::new());
    let store = Arc::new(AdmissionFailure {
        store: Default::default(),
        fail: AtomicBool::new(false),
        fail_running: AtomicBool::new(false),
        fail_idle: AtomicBool::new(false),
        fail_terminal: AtomicBool::new(false),
        opening: None,
    });
    let mut services = (*services(Arc::new(FakeProvider::default()))).clone();
    services.sessions = store.clone();
    services.compaction = Arc::new(BlockingCompaction {
        started: started.clone(),
    });
    let runtime = SessionRuntime::new(Arc::new(services));
    let (session, events) = runtime
        .create_session(SessionInit::default())
        .await
        .unwrap();
    let compact_session = session.clone();
    let compact = tokio::spawn(async move { compact_session.compact("manual", None).await });
    tokio::time::timeout(Duration::from_secs(5), started.notified())
        .await
        .expect("compaction starts before interruption");
    store.fail_idle.store(true, Ordering::SeqCst);

    let interrupted = tokio::time::timeout(Duration::from_secs(5), session.interrupt())
        .await
        .expect("interrupt settles the failed compaction");
    let compacted = compact.await.unwrap();
    for result in [interrupted, compacted] {
        let error = result.expect_err("failed Idle persistence must reach each caller");
        assert!(matches!(error, SessionError::Operation(_)));
        assert!(error.to_string().contains("idle status commit unavailable"));
    }
    let stream_error = tokio::time::timeout(Duration::from_secs(5), events.try_collect::<Vec<_>>())
        .await
        .expect("failed driver closes its event stream")
        .expect_err("stream consumers receive the persistence failure");
    assert!(
        stream_error
            .to_string()
            .contains("idle status commit unavailable")
    );
    assert!(matches!(
        session.shutdown().await,
        Err(SessionError::Operation(_))
    ));
    assert!(
        session
            .replay()
            .await
            .unwrap()
            .iter()
            .any(|event| matches!(event.payload, SessionEventPayload::SessionShutdownComplete)),
        "interrupt returns only after failed-driver cleanup has committed"
    );
}
