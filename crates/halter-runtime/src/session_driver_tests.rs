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

pub(crate) fn services(provider: Arc<dyn Provider>) -> Arc<RuntimeServices> {
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

async fn until_event(
    events: &mut SessionEventStream,
    matches: impl Fn(&SessionEvent) -> bool,
) -> Vec<SessionEvent> {
    tokio::time::timeout(Duration::from_secs(5), async {
        let mut seen = Vec::new();
        loop {
            let event = events.next().await.expect("live stream").expect("event");
            let matched = matches(&event);
            seen.push(event);
            if matched {
                return seen;
            }
        }
    })
    .await
    .expect("durable session event")
}

async fn until_started(events: &mut SessionEventStream) -> Vec<SessionEvent> {
    until_event(events, |event| {
        matches!(event.payload, SessionEventPayload::TurnStarted { .. })
    })
    .await
}

async fn wait_status(session: &crate::SessionHandle, expected: SessionStatus) {
    tokio::time::timeout(Duration::from_secs(5), async {
        let mut status = session.subscribe_status();
        loop {
            if *status.borrow_and_update() == expected {
                break;
            }
            status.changed().await.expect("status remains attached");
        }
    })
    .await
    .expect("live session activity");
}

async fn until_finished(
    session: &crate::SessionHandle,
    events: &mut SessionEventStream,
) -> Vec<SessionEvent> {
    let mut seen = until_event(events, |event| {
        event.session_id == *session.id()
            && matches!(
                event.payload,
                SessionEventPayload::TurnCompleted { .. } | SessionEventPayload::TurnFailed { .. }
            )
    })
    .await;
    wait_status(session, SessionStatus::Idle).await;
    let head = session.replay().await.unwrap().last().unwrap().sequence();
    if seen.last().unwrap().sequence() < head {
        seen.extend(until_event(events, |event| event.sequence() >= head).await);
    }
    seen
}

async fn assert_clean_closure(
    services: &RuntimeServices,
    id: &halter_protocol::SessionId,
    mut seen: Vec<SessionEvent>,
    events: SessionEventStream,
) -> Vec<SessionEvent> {
    seen.extend(
        tokio::time::timeout(Duration::from_secs(5), events.try_collect::<Vec<_>>())
            .await
            .expect("released session stream closes")
            .expect("clean session closure"),
    );
    assert!(matches!(
        seen.last().unwrap().payload,
        SessionEventPayload::SessionStatusChanged {
            status: SessionStatus::Closed
        }
    ));
    assert_eq!(seen.last().unwrap().sequence(), 0, "closure is transient");
    assert_eq!(
        seen.iter()
            .filter(|event| matches!(
                event.payload,
                SessionEventPayload::SessionStatusChanged { .. }
            ))
            .count(),
        1,
        "each stream reports clean closure once"
    );
    let log = services.sessions.replay(id).await.unwrap();
    assert!(!log.iter().any(|event| matches!(
        event.payload,
        SessionEventPayload::SessionStatusChanged { .. }
    )));
    assert_eq!(
        seen.iter()
            .filter(|event| !matches!(
                event.payload,
                SessionEventPayload::SessionStatusChanged { .. }
            ))
            .map(SessionEvent::sequence)
            .collect::<Vec<_>>(),
        log.iter().map(SessionEvent::sequence).collect::<Vec<_>>(),
        "closure follows every committed event, including cleanup"
    );
    seen
}

#[tokio::test]
async fn last_idle_handle_drop_releases_once_without_stream_or_status_retention() {
    use halter_hooks::{
        Hook, HookEventName, HookResponse, RegisteredHookPriority, RegisteredHooks,
    };

    let reasons = Arc::new(std::sync::Mutex::new(Vec::new()));
    let captured = reasons.clone();
    let mut hooks = RegisteredHooks::default();
    hooks.register(
        halter_protocol::PluginId::from("record-release"),
        RegisteredHookPriority::AfterPlugins,
        Hook::callback(HookEventName::SessionEnd, move |input| {
            let captured = captured.clone();
            async move {
                captured
                    .lock()
                    .unwrap()
                    .push(input.string_field("reason").unwrap().to_owned());
                HookResponse::passthrough()
            }
        }),
    );
    let mut services = services(Arc::new(FakeProvider::default()));
    Arc::get_mut(&mut services).unwrap().registered_hooks = Arc::new(hooks);
    let runtime = SessionRuntime::new(services.clone());
    let (session, events) = runtime
        .create_session(SessionInit::default())
        .await
        .unwrap();
    let id = session.id().clone();
    let status = session.subscribe_status();
    drop(session);
    assert_clean_closure(&services, &id, Vec::new(), events).await;
    assert_eq!(*status.borrow(), SessionStatus::Closed);
    assert_eq!(*reasons.lock().unwrap(), ["session_released"]);

    let (resumed, _) = runtime.resume_session(&id).await.unwrap();
    assert_eq!(resumed.status(), SessionStatus::Idle);
    resumed.shutdown(None).await.unwrap();
}

#[tokio::test]
async fn released_stream_is_fenced_from_reopened_incarnation_before_and_after_closed_event() {
    for consume_closed_before_reopen in [false, true] {
        let services = services(Arc::new(FakeProvider::default()));
        let runtime = SessionRuntime::new(services.clone());
        let (session, mut old_events) = runtime
            .create_session(SessionInit::default())
            .await
            .unwrap();
        let id = session.id().clone();
        let mut status = session.subscribe_status();
        drop(session);
        tokio::time::timeout(Duration::from_secs(5), async {
            while *status.borrow_and_update() != SessionStatus::Closed {
                status.changed().await.unwrap();
            }
        })
        .await
        .expect("released incarnation finishes cleanup");
        let old_log = services.sessions.replay(&id).await.unwrap();
        let mut seen = if consume_closed_before_reopen {
            until_event(&mut old_events, |event| {
                matches!(
                    event.payload,
                    SessionEventPayload::SessionStatusChanged {
                        status: SessionStatus::Closed
                    }
                )
            })
            .await
        } else {
            Vec::new()
        };

        let (resumed, mut new_events) = runtime.resume_session(&id).await.unwrap();
        let accepted = resumed
            .submit(Message::user("input for the new incarnation"))
            .await
            .unwrap();
        until_finished(&resumed, &mut new_events).await;
        assert!(accepted.sequence > old_log.last().unwrap().sequence());

        let tail = tokio::time::timeout(Duration::from_secs(5), old_events.try_collect::<Vec<_>>())
            .await
            .expect("old stream reaches EOF despite new incarnation activity")
            .unwrap();
        if consume_closed_before_reopen {
            assert!(tail.is_empty(), "Closed is immediately followed by EOF");
        }
        seen.extend(tail);
        assert!(matches!(
            seen.last().unwrap().payload,
            SessionEventPayload::SessionStatusChanged {
                status: SessionStatus::Closed
            }
        ));
        assert_eq!(seen.last().unwrap().sequence(), 0);
        assert_eq!(
            seen.iter()
                .filter(|event| matches!(
                    event.payload,
                    SessionEventPayload::SessionStatusChanged { .. }
                ))
                .count(),
            1
        );
        assert_eq!(
            seen.iter()
                .filter(|event| !matches!(
                    event.payload,
                    SessionEventPayload::SessionStatusChanged { .. }
                ))
                .map(SessionEvent::sequence)
                .collect::<Vec<_>>(),
            old_log
                .iter()
                .map(SessionEvent::sequence)
                .collect::<Vec<_>>(),
            "old stream contains only its own incarnation's durable events"
        );
        resumed.shutdown(None).await.unwrap();
    }
}

#[tokio::test]
async fn handle_clones_keep_idle_sessions_available_until_the_last_drop() {
    let services = services(Arc::new(FakeProvider::default()));
    let runtime = SessionRuntime::new(services.clone());
    let (session, mut events) = runtime
        .create_session(SessionInit::default())
        .await
        .unwrap();
    let id = session.id().clone();
    let retained = session.clone();
    drop(session);
    retained
        .submit(Message::user("the remaining handle still works"))
        .await
        .unwrap();
    let mut seen = until_started(&mut events).await;
    seen.extend(until_finished(&retained, &mut events).await);
    drop(retained);
    assert_clean_closure(&services, &id, seen, events).await;
}

struct GatedProvider {
    started: Notify,
    release: tokio::sync::Semaphore,
}

#[async_trait]
impl Provider for GatedProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::default()
    }

    async fn stream(
        &self,
        request: ProviderRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<stream::BoxStream<'static, Result<StreamEvent, ProviderError>>> {
        self.started.notify_one();
        self.release.acquire().await.unwrap().forget();
        assert!(
            !cancel.is_cancelled(),
            "handle release preserves active execution"
        );
        FakeProvider::default().stream(request, cancel).await
    }
}

#[tokio::test]
async fn last_handle_drop_during_foreground_execution_finishes_before_release() {
    let provider = Arc::new(GatedProvider {
        started: Notify::new(),
        release: tokio::sync::Semaphore::new(0),
    });
    let services = services(provider.clone());
    let runtime = SessionRuntime::new(services.clone());
    let (session, events) = runtime
        .create_session(SessionInit::default())
        .await
        .unwrap();
    let id = session.id().clone();
    session
        .submit(Message::user("finish normally"))
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), provider.started.notified())
        .await
        .expect("foreground execution starts");
    drop(session);
    provider.release.add_permits(1);
    let seen = assert_clean_closure(&services, &id, Vec::new(), events).await;
    assert!(
        seen.iter()
            .any(|event| matches!(event.payload, SessionEventPayload::TurnCompleted { .. }))
    );
    assert!(
        !seen
            .iter()
            .any(|event| matches!(event.payload, SessionEventPayload::TurnFailed { .. }))
    );
}

#[tokio::test]
async fn running_background_job_retains_released_session_until_its_exit() {
    use halter_tools::{
        BackgroundTool, DefaultToolPolicy, NoopToolEventSink, PolicySettings, Tool, ToolContext,
    };

    let root = tempfile::tempdir().unwrap();
    let services = services(Arc::new(FakeProvider::default()));
    let runtime = SessionRuntime::new(services.clone());
    let (session, events) = runtime
        .create_session(SessionInit {
            working_dir: root.path().into(),
            ..Default::default()
        })
        .await
        .unwrap();
    let id = session.id().clone();
    let status = session.subscribe_status();
    let context = ToolContext {
        session_id: id.clone(),
        working_dir: root.path().into(),
        path_locks: services.path_locks.clone(),
        tool_sessions: services.tool_sessions.clone(),
        snapshot: services.resources.snapshot(),
        cancel: CancellationToken::new(),
        emit: Arc::new(NoopToolEventSink),
        policy: Arc::new(DefaultToolPolicy::new(PolicySettings {
            allowed_read_roots: vec![root.path().into()],
            allowed_shell_commands: vec!["sleep".into()],
            ..Default::default()
        })),
        shell_timeout_secs: 30,
        subagent_parent: None,
    };
    let halter_protocol::ToolResult::Json { value: job } = BackgroundTool
        .execute(
            context.clone(),
            serde_json::json!({"action": "spawn", "command": "sleep 30"}),
        )
        .await
        .unwrap()
    else {
        panic!("background spawn returns its job record");
    };
    drop(session);
    assert!(matches!(
        runtime.resume_session(&id).await,
        Err(SessionError::AlreadyOpen(_))
    ));
    assert_eq!(*status.borrow(), SessionStatus::Idle);
    assert!(services.tool_sessions.has_running_jobs(&id));

    BackgroundTool
        .execute(
            context,
            serde_json::json!({"action": "kill", "id": job["id"]}),
        )
        .await
        .unwrap();
    assert_clean_closure(&services, &id, Vec::new(), events).await;
    assert!(!services.tool_sessions.has_running_jobs(&id));
    assert!(!services.tool_sessions.has_process_state(&id));
    assert_eq!(*status.borrow(), SessionStatus::Closed);
    let (resumed, _) = runtime.resume_session(&id).await.unwrap();
    resumed.shutdown(None).await.unwrap();
}

#[tokio::test]
async fn running_subagent_retains_released_parent_until_its_completion() {
    let provider = Arc::new(GatedProvider {
        started: Notify::new(),
        release: tokio::sync::Semaphore::new(0),
    });
    let services = services(provider.clone());
    let runtime = SessionRuntime::new(services.clone());
    let (session, events) = runtime
        .create_session(SessionInit::default())
        .await
        .unwrap();
    let id = session.id().clone();
    let stored = services.sessions.load_session(&id).await.unwrap().unwrap();
    let parent = halter_tools::SubagentParentContext {
        model: stored.blueprint.default_model.clone(),
        subagent_model: stored.blueprint.subagent_model.clone(),
        blueprint: stored.blueprint,
        state: stored.state,
        snapshot: stored.snapshot,
    };
    let control = runtime.subagent_control();
    let child = control
        .spawn(
            &parent,
            halter_protocol::SpawnSubagentRequest {
                message: "finish the delegated task".into(),
                agent_type: None,
                fork_context: false,
                model: None,
            },
            CancellationToken::new(),
        )
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), provider.started.notified())
        .await
        .expect("subagent execution starts");
    drop(session);
    assert!(matches!(
        runtime.resume_session(&id).await,
        Err(SessionError::AlreadyOpen(_))
    ));
    assert!(runtime.subagents.has_running_subagents(&id));
    provider.release.add_permits(1);
    let seen = assert_clean_closure(&services, &id, Vec::new(), events).await;
    assert!(seen.iter().any(|event| matches!(&event.payload,
        SessionEventPayload::SubagentUpdated { record }
            if record.status.agent_id == child.agent_id
                && record.status.state == halter_protocol::SubagentState::Completed
    )));
    assert!(!runtime.subagents.has_running_subagents(&id));
    let (resumed, _) = runtime.resume_session(&id).await.unwrap();
    resumed.shutdown(None).await.unwrap();
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
    assert_eq!(session.status(), SessionStatus::Idle);
    let mut observed = Vec::new();
    for text in ["first", "second"] {
        let message = Message::user(text);
        let id = session.submit(message.clone()).await.unwrap();
        observed.extend(until_started(&mut events).await);
        observed.extend(until_finished(&session, &mut events).await);
        assert_eq!(
            session.submit(message).await.unwrap(),
            id,
            "same id is acknowledged without executing twice"
        );
    }
    session.shutdown(None).await.unwrap();
    observed.extend(events.try_collect::<Vec<_>>().await.unwrap());
    let log = session.replay().await.unwrap();
    assert_eq!(
        observed
            .iter()
            .filter(|event| !matches!(
                event.payload,
                SessionEventPayload::SessionStatusChanged { .. }
            ))
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
    let (reopened, _events) = runtime.resume_session(session.id()).await.unwrap();
    assert_eq!(reopened.status(), SessionStatus::Idle);
    assert_eq!(reopened.id(), session.id());
    assert!(matches!(
        runtime.resume_session(session.id()).await,
        Err(SessionError::AlreadyOpen(_))
    ));
    reopened.shutdown(None).await.unwrap();
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
    let primary = session
        .submit(Message::user("start"))
        .await
        .unwrap()
        .message_id;
    provider.started.notified().await;
    // Observe the first request has consumed its text before interrupting it.
    // Input admission is serialized after the executor's opening commits.
    let queued = session
        .submit(Message::user("queued correction"))
        .await
        .unwrap()
        .message_id;
    tokio::time::timeout(Duration::from_secs(5), session.interrupt(None))
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
    until_started(&mut events).await;
    let interrupted = until_finished(&session, &mut events).await;
    assert!(interrupted.iter().any(|event| matches!(&event.payload,
        SessionEventPayload::InputDelivered { message_id } if message_id == &primary)));
    assert!(interrupted.iter().any(|event| matches!(&event.payload,
        SessionEventPayload::InputDeferred { message_id, reason: halter_protocol::InputDeferredReason::Interrupted } if message_id == &queued)));
    session.submit(Message::user("continue")).await.unwrap();
    until_started(&mut events).await;
    until_finished(&session, &mut events).await;
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
    session.shutdown(None).await.unwrap();
}

#[tokio::test]
async fn retained_deferred_input_counts_toward_capacity_and_idle_discard_releases_a_slot() {
    let provider = Arc::new(BlockingFirst::new(true));
    let runtime = SessionRuntime::new(services(provider.clone()));
    let (session, mut events) = runtime
        .create_session(SessionInit::default())
        .await
        .unwrap();
    session.submit(Message::user("start")).await.unwrap();
    provider.started.notified().await;
    let mut pending = Vec::new();
    for i in 0..128 {
        let receipt = session
            .submit(Message::user(format!("queued {i}")))
            .await
            .unwrap();
        pending.push(receipt.message_id);
    }
    assert!(matches!(
        session.discard(&pending[0]).await,
        Err(SessionError::Busy)
    ));
    assert!(matches!(
        session.submit(Message::user("overflow")).await,
        Err(SessionError::InboxFull)
    ));
    tokio::time::timeout(Duration::from_secs(5), session.interrupt(None))
        .await
        .expect("provider creation cancels")
        .unwrap();
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    until_finished(&session, &mut events).await;
    assert!(matches!(
        session
            .submit(Message::user("still full after interruption"))
            .await,
        Err(SessionError::InboxFull)
    ));
    assert!(session.discard(&pending[0]).await.unwrap());
    assert!(!session.discard(&pending[0]).await.unwrap());
    assert!(
        !session
            .discard(&MessageId::from("unknown-input"))
            .await
            .unwrap()
    );
    let fresh = session
        .submit(Message::user("use released slot"))
        .await
        .unwrap();
    until_finished(&session, &mut events).await;
    assert!(!session.discard(&fresh.message_id).await.unwrap());
    let log = session.replay().await.unwrap();
    assert_eq!(
        log.iter()
            .filter(|event| matches!(&event.payload,
                SessionEventPayload::InputRejected { message_id, .. } if message_id == &pending[0]
            ))
            .count(),
        1
    );
    assert!(!log.iter().any(|event| matches!(&event.payload,
        SessionEventPayload::MessageItem { message: Message::User(message) } if message.id == pending[0]
    )));
    assert!(log.iter().any(|event| matches!(&event.payload,
        SessionEventPayload::InputDelivered { message_id } if message_id == &fresh.message_id
    )));
    session.shutdown(None).await.unwrap();
    assert!(matches!(
        session.discard(&pending[0]).await,
        Err(SessionError::Closed)
    ));
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
    session.interrupt(None).await.unwrap();
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
            .filter(|event| !matches!(
                event.payload,
                SessionEventPayload::SessionStatusChanged { .. }
            ))
            .map(SessionEvent::sequence)
            .collect::<Vec<_>>(),
        log.iter().map(SessionEvent::sequence).collect::<Vec<_>>()
    );
    assert_eq!(session.status(), SessionStatus::Closed);
    assert!(
        !log.iter().any(|event| matches!(
            event.payload,
            SessionEventPayload::SessionStatusChanged { .. }
        )),
        "live activity is not persisted"
    );
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
    fail_start: AtomicBool,
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
        if events
            .iter()
            .any(|event| matches!(event.payload, SessionEventPayload::TurnStarted { .. }))
            && self.fail_start.swap(false, Ordering::SeqCst)
        {
            anyhow::bail!("execution start commit unavailable");
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
        fail_start: AtomicBool::new(false),
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
    until_started(&mut events).await;
    until_finished(&session, &mut events).await;
    session.shutdown(None).await.unwrap();
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
    until_started(&mut events).await;
    until_finished(&session, &mut events).await;
    session.compact("manual", None).await.unwrap();
    let log = session.replay().await.unwrap();
    assert_eq!(session.status(), SessionStatus::Idle);
    assert!(
        log.iter()
            .any(|event| matches!(event.payload, SessionEventPayload::ContextCompacted { .. }))
    );
    let id = session
        .submit(Message::user("after compaction"))
        .await
        .unwrap()
        .message_id;
    loop {
        let event = events.next().await.unwrap().unwrap();
        if matches!(event.payload, SessionEventPayload::InputAccepted { message } if message.id == id)
        {
            break;
        }
    }
    until_started(&mut events).await;
    let delivered = until_finished(&session, &mut events).await;
    assert!(delivered.iter().any(|event| matches!(&event.payload, SessionEventPayload::MessageItem { message: Message::User(message) } if message.id == id)));
    session.shutdown(None).await.unwrap();
}

#[tokio::test]
async fn cancelled_open_releases_reservation_and_runtime_shutdown_refuses_delayed_open() {
    for cancel_open in [true, false] {
        let started = Arc::new(Notify::new());
        let release = Arc::new(tokio::sync::Semaphore::new(0));
        let store = Arc::new(AdmissionFailure {
            store: Default::default(),
            fail: AtomicBool::new(false),
            fail_start: AtomicBool::new(false),
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
            session.shutdown(None).await.unwrap();
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
    let results = futures::future::join_all((0..32).map(|_| session.shutdown(None))).await;
    assert!(results.into_iter().all(|result| result.is_ok()));
    session.shutdown(None).await.unwrap();
}

struct RecordingProvider {
    requests: std::sync::Mutex<Vec<ProviderRequest>>,
}

#[async_trait]
impl Provider for RecordingProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::default()
    }

    async fn stream(
        &self,
        request: ProviderRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<stream::BoxStream<'static, Result<StreamEvent, ProviderError>>> {
        self.requests.lock().unwrap().push(request.clone());
        FakeProvider::default().stream(request, cancel).await
    }
}

#[tokio::test]
async fn failed_execution_start_defers_its_input_and_fresh_input_skips_it_until_explicit_retry() {
    let store = Arc::new(AdmissionFailure {
        store: Default::default(),
        fail: AtomicBool::new(false),
        fail_start: AtomicBool::new(true),
        fail_terminal: AtomicBool::new(false),
        opening: None,
    });
    let provider = Arc::new(RecordingProvider {
        requests: Default::default(),
    });
    let mut services = (*services(provider.clone())).clone();
    services.sessions = store;
    let runtime = SessionRuntime::new(Arc::new(services));
    let (session, mut events) = runtime
        .create_session(SessionInit::default())
        .await
        .unwrap();
    let message = Message::user("failing input");
    let original = session.submit(message.clone()).await.unwrap();
    let failed = until_event(&mut events, |event| matches!(&event.payload,
        SessionEventPayload::InputDeferred {
            message_id,
            reason: halter_protocol::InputDeferredReason::ExecutionFailed { error, retryable: true },
        } if message_id == &original.message_id && error.contains("execution start commit unavailable")
    )).await;
    wait_status(&session, SessionStatus::Idle).await;
    assert!(
        !failed
            .iter()
            .any(|event| matches!(event.payload, SessionEventPayload::TurnStarted { .. }))
    );
    assert!(provider.requests.lock().unwrap().is_empty());

    let fresh = session.submit(Message::user("new task")).await.unwrap();
    let delivered = until_finished(&session, &mut events).await;
    assert!(delivered.iter().any(|event| matches!(&event.payload,
        SessionEventPayload::InputDelivered { message_id } if message_id == &fresh.message_id
    )));
    assert!(!delivered.iter().any(|event| matches!(&event.payload,
        SessionEventPayload::InputDelivered { message_id } if message_id == &original.message_id
    )));
    let users = provider.requests.lock().unwrap()[0]
        .messages
        .iter()
        .filter_map(|message| {
            if let Message::User(user) = message {
                Some(user.plain_text())
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(users, ["new task"]);

    let retry = session.submit(message.clone()).await.unwrap();
    assert!(retry.sequence > original.sequence);
    until_finished(&session, &mut events).await;
    assert_eq!(session.submit(message).await.unwrap(), retry);
    let log = session.replay().await.unwrap();
    assert_eq!(log.iter().filter(|event| matches!(&event.payload,
        SessionEventPayload::InputDelivered { message_id } if message_id == &original.message_id
    )).count(), 1);
    assert_eq!(provider.requests.lock().unwrap().len(), 2);
    session.shutdown(None).await.unwrap();
}

#[tokio::test]
async fn release_preserves_failed_input_for_explicit_retry_after_resume() {
    let store = Arc::new(AdmissionFailure {
        store: Default::default(),
        fail: AtomicBool::new(false),
        fail_start: AtomicBool::new(true),
        fail_terminal: AtomicBool::new(false),
        opening: None,
    });
    let mut configured = (*services(Arc::new(FakeProvider::default()))).clone();
    configured.sessions = store;
    let services = Arc::new(configured);
    let runtime = SessionRuntime::new(services.clone());
    let (session, mut events) = runtime
        .create_session(SessionInit::default())
        .await
        .unwrap();
    let id = session.id().clone();
    let input = Message::user("retry after reopening");
    let accepted = session.submit(input.clone()).await.unwrap();
    let seen = until_event(&mut events, |event| {
        matches!(&event.payload,
            SessionEventPayload::InputDeferred {
                message_id,
                reason: halter_protocol::InputDeferredReason::ExecutionFailed { .. },
            } if message_id == &accepted.message_id
        )
    })
    .await;
    wait_status(&session, SessionStatus::Idle).await;
    drop(session);
    assert_clean_closure(&services, &id, seen, events).await;

    let mut stored = services.sessions.load_session(&id).await.unwrap().unwrap();
    crate::session::hydrate_stored_session(services.sessions.as_ref(), &mut stored)
        .await
        .unwrap();
    assert_eq!(stored.state.pending_inputs.len(), 1);
    assert_eq!(stored.state.pending_inputs[0].id, accepted.message_id);
    let (resumed, mut events) = runtime.resume_session(&id).await.unwrap();
    assert_eq!(resumed.status(), SessionStatus::Idle);
    let retry = resumed.submit(input).await.unwrap();
    assert!(retry.sequence > accepted.sequence);
    let delivered = until_finished(&resumed, &mut events).await;
    assert!(delivered.iter().any(|event| matches!(&event.payload,
        SessionEventPayload::InputDelivered { message_id } if message_id == &accepted.message_id
    )));
    resumed.shutdown(None).await.unwrap();
}

#[tokio::test]
async fn release_preserves_interrupted_input_for_next_submission_after_resume() {
    let provider = Arc::new(BlockingFirst::new(true));
    let services = services(provider.clone());
    let runtime = SessionRuntime::new(services.clone());
    let (session, events) = runtime
        .create_session(SessionInit::default())
        .await
        .unwrap();
    let id = session.id().clone();
    session.submit(Message::user("start")).await.unwrap();
    tokio::time::timeout(Duration::from_secs(5), provider.started.notified())
        .await
        .expect("foreground execution starts");
    let queued = session
        .submit(Message::user("keep this queued input"))
        .await
        .unwrap();
    session.interrupt(None).await.unwrap();
    assert_eq!(session.status(), SessionStatus::Idle);
    drop(session);
    assert_clean_closure(&services, &id, Vec::new(), events).await;

    let mut stored = services.sessions.load_session(&id).await.unwrap().unwrap();
    crate::session::hydrate_stored_session(services.sessions.as_ref(), &mut stored)
        .await
        .unwrap();
    assert_eq!(stored.state.pending_inputs.len(), 1);
    assert_eq!(stored.state.pending_inputs[0].id, queued.message_id);
    let (resumed, mut events) = runtime.resume_session(&id).await.unwrap();
    assert_eq!(resumed.status(), SessionStatus::Idle);
    let fresh = resumed.submit(Message::user("continue")).await.unwrap();
    let delivered = until_finished(&resumed, &mut events).await;
    for expected in [&queued.message_id, &fresh.message_id] {
        assert!(delivered.iter().any(|event| matches!(&event.payload,
            SessionEventPayload::InputDelivered { message_id } if message_id == expected
        )));
    }
    resumed.shutdown(None).await.unwrap();
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
            fail_start: AtomicBool::new(false),
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
            let error = tokio::time::timeout(Duration::from_secs(5), session.interrupt(None))
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
            session.shutdown(None).await,
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

struct FailingCompaction {
    started: Arc<Notify>,
}

#[tokio::test]
async fn admitted_input_crossing_threshold_compacts_before_the_next_provider_request() {
    use halter_hooks::{
        Hook, HookEventName, HookResponse, RegisteredHookPriority, RegisteredHooks,
    };

    struct Checkpoint;
    #[async_trait]
    impl crate::CompactionStrategy for Checkpoint {
        async fn compact(
            &self,
            context: crate::CompactionContext<'_>,
        ) -> anyhow::Result<Option<crate::CompactionEffects>> {
            Ok(Some(crate::CompactionEffects {
                messages: vec![Message::user("checkpoint of admitted input")],
                compacted_context: Default::default(),
                result: halter_protocol::CompactionResult {
                    compacted_count: context.state().messages.len(),
                    summary: "checkpointed".into(),
                },
                usage: Default::default(),
            }))
        }
    }

    struct RecordingProvider(std::sync::Mutex<Vec<ProviderRequest>>);
    #[async_trait]
    impl Provider for RecordingProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }
        async fn stream(
            &self,
            request: ProviderRequest,
            cancel: CancellationToken,
        ) -> anyhow::Result<stream::BoxStream<'static, Result<StreamEvent, ProviderError>>>
        {
            self.0.lock().unwrap().push(request.clone());
            FakeProvider::default().stream(request, cancel).await
        }
    }

    // Exercise admission after the opening boundary and after the final
    // response's boundary. Both must budget the new input before inference.
    for event_name in [HookEventName::UserPromptSubmit, HookEventName::Stop] {
        let entered = Arc::new(Notify::new());
        let release = Arc::new(Notify::new());
        let first = Arc::new(AtomicBool::new(true));
        let mut hooks = RegisteredHooks::default();
        let gate_entered = entered.clone();
        let gate_release = release.clone();
        hooks.register(
            halter_protocol::PluginId::from("admission-boundary"),
            RegisteredHookPriority::AfterPlugins,
            Hook::callback(event_name, move |_| {
                let first = first.clone();
                let entered = gate_entered.clone();
                let release = gate_release.clone();
                async move {
                    if first.swap(false, Ordering::SeqCst) {
                        entered.notify_one();
                        release.notified().await;
                    }
                    HookResponse::passthrough()
                }
            }),
        );
        let provider = Arc::new(RecordingProvider(Default::default()));
        let mut runtime_services = services(provider.clone());
        let configured = Arc::get_mut(&mut runtime_services).unwrap();
        configured.registered_hooks = Arc::new(hooks);
        configured.context = crate::ContextSettings {
            compaction_threshold: 10_000,
            max_tokens: Some(15_000),
        };
        configured.compaction = Arc::new(Checkpoint);
        let runtime = SessionRuntime::new(runtime_services);
        let (session, mut events) = runtime
            .create_session(SessionInit::default())
            .await
            .unwrap();
        session.submit(Message::user("start")).await.unwrap();
        tokio::time::timeout(Duration::from_secs(5), entered.notified())
            .await
            .expect("hook reached");
        let accepted = session
            .submit(Message::user("x".repeat(80_000)))
            .await
            .unwrap();
        release.notify_one();
        let seen = until_finished(&session, &mut events).await;
        assert!(
            !seen
                .iter()
                .any(|event| matches!(event.payload, SessionEventPayload::TurnFailed { .. })),
            "{event_name:?}: compaction must precede cap enforcement"
        );
        let log = session.replay().await.unwrap();
        let position = |predicate: &dyn Fn(&SessionEventPayload) -> bool| {
            log.iter()
                .position(|event| predicate(&event.payload))
                .unwrap()
        };
        let delivered = position(
            &|payload| matches!(payload, SessionEventPayload::MessageItem { message: Message::User(user) } if user.id == accepted.message_id),
        );
        let compacted =
            position(&|payload| matches!(payload, SessionEventPayload::ContextCompacted { .. }));
        assert!(
            delivered < compacted,
            "{event_name:?}: the checkpoint must include admitted input (delivery {delivered}, compaction {compacted})"
        );
        assert!(provider.0.lock().unwrap().iter().any(|request| request.messages.iter().any(|message| matches!(message, Message::User(user) if user.plain_text() == "checkpoint of admitted input"))), "{event_name:?}: the next inference must see the checkpoint");
        session.shutdown(None).await.unwrap();
    }
}

#[async_trait]
impl crate::CompactionStrategy for FailingCompaction {
    async fn compact(
        &self,
        context: crate::CompactionContext<'_>,
    ) -> anyhow::Result<Option<crate::CompactionEffects>> {
        self.started.notify_one();
        context.cancel().cancelled().await;
        anyhow::bail!("strategy unavailable")
    }
}

#[tokio::test]
async fn compaction_failure_reaches_waiters_and_returns_the_live_session_to_idle() {
    let started = Arc::new(Notify::new());
    let mut services = (*services(Arc::new(FakeProvider::default()))).clone();
    services.compaction = Arc::new(FailingCompaction {
        started: started.clone(),
    });
    let runtime = SessionRuntime::new(Arc::new(services));
    let (session, mut events) = runtime
        .create_session(SessionInit::default())
        .await
        .unwrap();
    let compact_session = session.clone();
    let compact = tokio::spawn(async move { compact_session.compact("manual", None).await });
    tokio::time::timeout(Duration::from_secs(5), started.notified())
        .await
        .expect("compaction starts");
    assert_eq!(session.status(), SessionStatus::Running);
    let interrupted = tokio::time::timeout(Duration::from_secs(5), session.interrupt(None))
        .await
        .expect("interrupt settles compaction failure");
    for result in [interrupted, compact.await.unwrap()] {
        let error = result.expect_err("strategy failure reaches each waiter");
        assert!(
            error.to_string().contains("strategy unavailable"),
            "{error}"
        );
    }
    assert_eq!(session.status(), SessionStatus::Idle);
    session
        .submit(Message::user("continue after compaction failure"))
        .await
        .unwrap();
    until_finished(&session, &mut events).await;
    session.shutdown(None).await.unwrap();
}

struct ReleasedCompaction {
    started: Arc<Notify>,
    release: Arc<tokio::sync::Semaphore>,
}

#[async_trait]
impl crate::CompactionStrategy for ReleasedCompaction {
    async fn compact(
        &self,
        _context: crate::CompactionContext<'_>,
    ) -> anyhow::Result<Option<crate::CompactionEffects>> {
        self.started.notify_one();
        self.release.acquire().await.unwrap().forget();
        Ok(None)
    }
}

#[tokio::test]
async fn input_submitted_during_compaction_is_delivered_to_foreground_execution() {
    let started = Arc::new(Notify::new());
    let release = Arc::new(tokio::sync::Semaphore::new(0));
    let mut services = (*services(Arc::new(FakeProvider::default()))).clone();
    services.compaction = Arc::new(ReleasedCompaction {
        started: started.clone(),
        release: release.clone(),
    });
    let runtime = SessionRuntime::new(Arc::new(services));
    let (session, mut events) = runtime
        .create_session(SessionInit::default())
        .await
        .unwrap();
    assert_eq!(session.status(), SessionStatus::Idle);
    let compact_session = session.clone();
    let compact = tokio::spawn(async move { compact_session.compact("manual", None).await });
    tokio::time::timeout(Duration::from_secs(5), started.notified())
        .await
        .unwrap();
    assert_eq!(session.status(), SessionStatus::Running);
    let id = session
        .submit(Message::user("submitted during compaction"))
        .await
        .unwrap()
        .message_id;
    assert!(
        !session
            .replay()
            .await
            .unwrap()
            .iter()
            .any(|event| matches!(&event.payload,
                SessionEventPayload::InputDelivered { message_id } if message_id == &id
            ))
    );
    release.add_permits(1);
    compact.await.unwrap().unwrap();
    let execution = until_finished(&session, &mut events).await;
    assert_eq!(
        execution
            .iter()
            .filter(|event| matches!(&event.payload,
                SessionEventPayload::InputDelivered { message_id } if message_id == &id
            ))
            .count(),
        1
    );
    assert!(execution.iter().any(|event| matches!(&event.payload,
        SessionEventPayload::MessageItem { message: Message::User(message) } if message.id == id
    )));
    session.shutdown(None).await.unwrap();
}

#[tokio::test]
async fn retry_of_deferred_input_has_a_fresh_acceptance_boundary_and_one_delivery() {
    let provider = Arc::new(BlockingFirst::new(false));
    let runtime = SessionRuntime::new(services(provider.clone()));
    let (session, mut events) = runtime
        .create_session(SessionInit::default())
        .await
        .unwrap();
    session.submit(Message::user("start")).await.unwrap();
    provider.started.notified().await;
    let input = Message::user("pending input");
    let original = session.submit(input.clone()).await.unwrap();
    let id = original.message_id;
    session.interrupt(None).await.unwrap();
    let before = session.replay().await.unwrap();
    assert!(before.iter().any(|event| matches!(&event.payload,
        SessionEventPayload::InputDeferred { message_id, .. } if message_id == &id)));
    let retry = session.submit(input).await.unwrap();
    assert!(original.sequence < retry.sequence);
    // The stream still contains the first attempt's acceptance and deferral.
    // The receipt excludes them without requiring the caller to drain first.
    let after = tokio::time::timeout(Duration::from_secs(5), async {
        let mut after = Vec::new();
        loop {
            let event = events.next().await.unwrap().unwrap();
            if event.sequence() < retry.sequence { continue; }
            let delivered = matches!(&event.payload, SessionEventPayload::InputDelivered { message_id } if message_id == &id);
            after.push(event);
            if delivered { break after; }
        }
    }).await.unwrap();
    let accepted = after
        .iter()
        .position(|event| {
            matches!(&event.payload,
        SessionEventPayload::InputAccepted { message } if message.id == id)
        })
        .unwrap();
    let delivered = after
        .iter()
        .position(|event| {
            matches!(&event.payload,
        SessionEventPayload::InputDelivered { message_id } if message_id == &id)
        })
        .unwrap();
    assert!(accepted < delivered);
    assert_eq!(
        after
            .iter()
            .filter(|event| matches!(&event.payload,
        SessionEventPayload::InputDelivered { message_id } if message_id == &id))
            .count(),
        1
    );
    session.shutdown(None).await.unwrap();
}

#[tokio::test]
async fn failed_pending_retry_does_not_wake_a_deferred_input() {
    let provider = Arc::new(BlockingFirst::new(false));
    let store = Arc::new(AdmissionFailure {
        store: Default::default(),
        fail: AtomicBool::new(false),
        fail_start: AtomicBool::new(false),
        fail_terminal: AtomicBool::new(false),
        opening: None,
    });
    let mut services = (*services(provider.clone())).clone();
    services.sessions = store.clone();
    services.compaction = Arc::new(ReleasedCompaction {
        started: Arc::new(Notify::new()),
        release: Arc::new(tokio::sync::Semaphore::new(1)),
    });
    let runtime = SessionRuntime::new(Arc::new(services));
    let (session, _events) = runtime
        .create_session(SessionInit::default())
        .await
        .unwrap();
    session.submit(Message::user("start")).await.unwrap();
    provider.started.notified().await;
    let pending = Message::user("deferred");
    session.submit(pending.clone()).await.unwrap();
    session.interrupt(None).await.unwrap();
    store.fail.store(true, Ordering::SeqCst);
    assert!(session.submit(pending).await.is_err());
    // A successful idle operation gives the actor a chance to process the
    // failed retry. It must not start the queued provider request instead.
    tokio::time::timeout(Duration::from_secs(5), session.compact("barrier", None))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
    session.shutdown(None).await.unwrap();
}
