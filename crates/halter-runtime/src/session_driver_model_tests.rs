// pattern: Integration Tests
//
// Model-based tests for the session driver. Proptest generates sequences of
// concurrent client calls, scripted provider behaviour, and injected store
// faults, then runs them against the real `SessionRuntime`. Rather than
// predicting exact interleavings, each run is checked against invariants over
// the durable log, the live event streams, and the call results.
//
// Runs use a current-thread runtime with paused time: when every task is
// blocked, tokio advances the clock, so a deadlock surfaces as a call timeout
// within milliseconds of wall time.
//
// Each run uses one durable backend. With the in-memory store a turn usually
// finishes before the next operation, so those runs mostly exercise delivery.
// SQLite commits run on blocking threads, so interrupts, shutdowns, and crashes
// land mid-turn far more often; those runs mostly exercise cancellation and
// recovery against real persistence. The two are complementary.
//
// `PROPTEST_CASES=2000 cargo test -p halter-runtime session_driver_model`
// runs a longer soak of both.

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use futures::{StreamExt, stream};
use halter_protocol::fold::{covered_state_matches, fold_events};
use halter_protocol::{
    BlockId, Message, MessageId, PendingEvent, ProviderCapabilities, ProviderError,
    ProviderRequest, ResourceSnapshot, SessionBlueprint, SessionEvent, SessionEventPayload,
    SessionId, SessionState, SessionStatus, StreamEvent,
};
use halter_providers::{FakeProvider, Provider};
use halter_session::{InMemorySessionStore, SessionStore, SqliteSessionStore, StoredSession};
use proptest::prelude::*;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::{
    SessionError, SessionEventStream, SessionHandle, SessionInit, SessionRuntime, Submission,
};

/// Distinct user messages a run draws from. Small, so resubmission and
/// discard of the same input are common.
const MESSAGE_POOL: usize = 4;
/// Paused-clock budget for one client call. Only a deadlock exhausts it.
const CALL_TIMEOUT: Duration = Duration::from_secs(600);
/// Paused-clock budget for `WaitIdle`. A hung provider legitimately keeps
/// the session running, so exhausting this is not a failure.
const IDLE_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, Copy)]
enum ProviderStep {
    Complete,
    FailCreate,
    HangCreate,
    HangMidStream,
}

#[derive(Debug, Clone, Copy)]
enum StoreFault {
    Admission(bool),
    Terminal(bool),
    NextTurnStart,
    NextCommit,
}

#[derive(Debug, Clone, Copy)]
enum Op {
    Submit(usize),
    Discard(usize),
    Interrupt,
    Compact,
    Shutdown,
    /// Let spawned calls and the driver make progress.
    Yield(u8),
    /// Await every outstanding client call.
    Barrier,
    WaitIdle,
    /// Shut down, drop the handle, and resume from the store.
    Reopen,
    /// Abandon the process without cleanup and resume in a fresh runtime
    /// over the same durable store.
    Crash,
    Script(ProviderStep),
    Fault(StoreFault),
}

fn op_strategy() -> impl Strategy<Value = Op> {
    let message = 0..MESSAGE_POOL;
    prop_oneof![
        6 => message.clone().prop_map(Op::Submit),
        2 => message.prop_map(Op::Discard),
        2 => Just(Op::Interrupt),
        1 => Just(Op::Compact),
        1 => Just(Op::Shutdown),
        3 => (1u8..8).prop_map(Op::Yield),
        2 => Just(Op::Barrier),
        2 => Just(Op::WaitIdle),
        1 => Just(Op::Reopen),
        1 => Just(Op::Crash),
        3 => prop_oneof![
            Just(ProviderStep::Complete),
            Just(ProviderStep::FailCreate),
            Just(ProviderStep::HangCreate),
            Just(ProviderStep::HangMidStream),
        ]
        .prop_map(Op::Script),
        2 => prop_oneof![
            any::<bool>().prop_map(StoreFault::Admission),
            any::<bool>().prop_map(StoreFault::Terminal),
            Just(StoreFault::NextTurnStart),
            Just(StoreFault::NextCommit),
        ]
        .prop_map(Op::Fault),
    ]
}

// ---------------------------------------------------------------------------
// Scripted collaborators
// ---------------------------------------------------------------------------

/// Provider whose next calls follow a script; an empty script completes.
#[derive(Default)]
struct ScriptedProvider {
    script: Mutex<VecDeque<ProviderStep>>,
    /// User-message texts of every request, in call order.
    requests: Mutex<Vec<Vec<String>>>,
}

#[async_trait]
impl Provider for ScriptedProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::default()
    }

    async fn stream(
        &self,
        request: ProviderRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<stream::BoxStream<'static, Result<StreamEvent, ProviderError>>> {
        self.requests.lock().unwrap().push(
            request
                .messages
                .iter()
                .filter_map(|message| match message {
                    Message::User(user) => Some(user.plain_text()),
                    _ => None,
                })
                .collect(),
        );
        let step = self
            .script
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or(ProviderStep::Complete);
        match step {
            ProviderStep::Complete => FakeProvider::default().stream(request, cancel).await,
            ProviderStep::FailCreate => anyhow::bail!("scripted provider failure"),
            // Deliberately ignores cancellation: the runtime must still be
            // able to abandon an uncooperative provider.
            ProviderStep::HangCreate => std::future::pending().await,
            ProviderStep::HangMidStream => {
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
                Ok(prefix.chain(stream::pending()).boxed())
            }
        }
    }
}

/// Durable storage that outlives any one process.
#[derive(Debug, Clone, Copy)]
enum Backend {
    Memory,
    Sqlite,
}

/// Opens each process's handle on the durable store: memory processes share
/// one in-process store, SQLite processes open the same database file.
enum Durable {
    Memory(InMemorySessionStore),
    Sqlite(PathBuf),
}

impl Durable {
    fn new(backend: Backend, dir: &Path) -> Self {
        match backend {
            Backend::Memory => Self::Memory(InMemorySessionStore::default()),
            Backend::Sqlite => Self::Sqlite(dir.join("sessions.db")),
        }
    }

    fn open(&self) -> Arc<dyn SessionStore> {
        match self {
            Self::Memory(store) => Arc::new(store.clone()),
            Self::Sqlite(path) => {
                Arc::new(SqliteSessionStore::open(path).expect("open sqlite store"))
            }
        }
    }
}

/// One process's view of the durable store, with switchable commit
/// failures. A failed commit is rejected before reaching the inner store, so
/// it is never durable. Once `dead`, the process has crashed and none of its
/// writes land.
struct FaultyStore {
    inner: Arc<dyn SessionStore>,
    dead: AtomicBool,
    fail_admission: AtomicBool,
    fail_terminal: AtomicBool,
    fail_next_turn_start: AtomicBool,
    fail_next_commit: AtomicBool,
}

impl FaultyStore {
    fn over(inner: Arc<dyn SessionStore>) -> Self {
        Self {
            inner,
            dead: AtomicBool::default(),
            fail_admission: AtomicBool::default(),
            fail_terminal: AtomicBool::default(),
            fail_next_turn_start: AtomicBool::default(),
            fail_next_commit: AtomicBool::default(),
        }
    }

    fn apply(&self, fault: StoreFault) {
        match fault {
            StoreFault::Admission(on) => self.fail_admission.store(on, Ordering::SeqCst),
            StoreFault::Terminal(on) => self.fail_terminal.store(on, Ordering::SeqCst),
            StoreFault::NextTurnStart => {
                self.fail_next_turn_start.store(true, Ordering::SeqCst);
            }
            StoreFault::NextCommit => self.fail_next_commit.store(true, Ordering::SeqCst),
        }
    }
}

#[async_trait]
impl SessionStore for FaultyStore {
    async fn create_session(&self, session: StoredSession) -> anyhow::Result<()> {
        anyhow::ensure!(!self.dead.load(Ordering::SeqCst), "process crashed");
        self.inner.create_session(session).await
    }

    async fn load_session(&self, id: &SessionId) -> anyhow::Result<Option<StoredSession>> {
        self.inner.load_session(id).await
    }

    async fn commit(
        &self,
        id: &SessionId,
        snapshot: Option<Arc<ResourceSnapshot>>,
        expected: Option<u64>,
        state: Option<SessionState>,
        events: Vec<PendingEvent>,
    ) -> anyhow::Result<Vec<SessionEvent>> {
        let has = |matches: fn(&SessionEventPayload) -> bool| {
            events.iter().any(|event| matches(&event.payload))
        };
        anyhow::ensure!(!self.dead.load(Ordering::SeqCst), "process crashed");
        if self.fail_next_commit.swap(false, Ordering::SeqCst) {
            anyhow::bail!("injected commit failure");
        }
        if self.fail_admission.load(Ordering::SeqCst)
            && has(|payload| matches!(payload, SessionEventPayload::InputAccepted { .. }))
        {
            anyhow::bail!("injected admission failure");
        }
        if has(|payload| matches!(payload, SessionEventPayload::TurnStarted { .. }))
            && self.fail_next_turn_start.swap(false, Ordering::SeqCst)
        {
            anyhow::bail!("injected turn start failure");
        }
        if self.fail_terminal.load(Ordering::SeqCst)
            && has(|payload| {
                matches!(
                    payload,
                    SessionEventPayload::TurnCompleted { .. }
                        | SessionEventPayload::TurnFailed { .. }
                )
            })
        {
            anyhow::bail!("injected terminal commit failure");
        }
        self.inner
            .commit(id, snapshot, expected, state, events)
            .await
    }

    async fn replay(&self, id: &SessionId) -> anyhow::Result<Vec<SessionEvent>> {
        self.inner.replay(id).await
    }

    async fn list_sessions(&self) -> anyhow::Result<Vec<SessionBlueprint>> {
        self.inner.list_sessions().await
    }
}

// ---------------------------------------------------------------------------
// Recorded run
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ErrorKind {
    Closed,
    Busy,
    InboxFull,
    Operation,
    Other,
}

impl From<&SessionError> for ErrorKind {
    fn from(error: &SessionError) -> Self {
        match error {
            SessionError::Closed => Self::Closed,
            SessionError::Busy => Self::Busy,
            SessionError::InboxFull => Self::InboxFull,
            SessionError::Operation(_) => Self::Operation,
            _ => Self::Other,
        }
    }
}

#[derive(Debug)]
enum Outcome {
    Submit {
        message: MessageId,
        result: Result<Submission, ErrorKind>,
    },
    Discard {
        message: MessageId,
        result: Result<bool, ErrorKind>,
    },
    Interrupt(Result<(), ErrorKind>),
    Compact(Result<(), ErrorKind>),
    Shutdown(Result<(), ErrorKind>),
    /// The call did not settle within `CALL_TIMEOUT`.
    Hung(&'static str),
}

/// One client call. `start` and `end` come from a shared logical clock, so
/// `a.end < b.start` means `a` returned before `b` was issued.
#[derive(Debug)]
struct Call {
    incarnation: usize,
    start: u64,
    end: u64,
    outcome: Outcome,
}

#[derive(Debug)]
struct Incarnation {
    events: Vec<SessionEvent>,
    stream_error: Option<String>,
}

#[derive(Debug, Default)]
struct Run {
    calls: Vec<Call>,
    incarnations: Vec<Incarnation>,
    /// Harness-level failures, such as a stream that never closed.
    failures: Vec<String>,
    /// Logs captured whenever `WaitIdle` observed an idle session.
    idle_logs: Vec<Vec<SessionEvent>>,
    log: Vec<SessionEvent>,
    stored_state: SessionState,
    provider_requests: Vec<Vec<String>>,
}

type Drain = JoinHandle<(Vec<SessionEvent>, Option<String>)>;

fn drain(mut events: SessionEventStream) -> Drain {
    tokio::spawn(async move {
        let mut seen = Vec::new();
        while let Some(item) = events.next().await {
            match item {
                Ok(event) => seen.push(event),
                Err(error) => return (seen, Some(format!("{error:#}"))),
            }
        }
        (seen, None)
    })
}

struct Harness {
    runtime: SessionRuntime,
    services: Arc<crate::RuntimeServices>,
    provider: Arc<ScriptedProvider>,
    durable: Durable,
    store: Arc<FaultyStore>,
    id: SessionId,
    handle: Option<SessionHandle>,
    stream: Option<Drain>,
    incarnation: usize,
    clock: Arc<AtomicU64>,
    messages: Vec<Message>,
    pending: Vec<JoinHandle<Call>>,
    /// Runtimes of crashed processes, kept alive so their tasks can only
    /// fail against the dead store rather than vanish mid-await.
    crashed: Vec<SessionRuntime>,
    run: Run,
}

impl Harness {
    /// A fresh process: its own services, leases, and store view.
    fn process(
        provider: &Arc<ScriptedProvider>,
        durable: Arc<dyn SessionStore>,
    ) -> (
        SessionRuntime,
        Arc<crate::RuntimeServices>,
        Arc<FaultyStore>,
    ) {
        let store = Arc::new(FaultyStore::over(durable));
        let mut services = (*crate::session_driver_tests::services(provider.clone())).clone();
        services.sessions = store.clone();
        let services = Arc::new(services);
        (SessionRuntime::new(services.clone()), services, store)
    }

    async fn new(working_dir: &Path, durable: Durable) -> Self {
        let provider = Arc::new(ScriptedProvider::default());
        let (runtime, services, store) = Self::process(&provider, durable.open());
        let (handle, events) = runtime
            .create_session(SessionInit {
                working_dir: working_dir.into(),
                ..Default::default()
            })
            .await
            .expect("fresh session opens without faults");
        Self {
            runtime,
            services,
            provider,
            durable,
            store,
            id: handle.id().clone(),
            handle: Some(handle),
            stream: Some(drain(events)),
            incarnation: 0,
            clock: Arc::new(AtomicU64::new(0)),
            messages: (0..MESSAGE_POOL)
                .map(|index| Message::user(format!("message {index}")))
                .collect(),
            pending: Vec::new(),
            crashed: Vec::new(),
            run: Run::default(),
        }
    }

    fn spawn_call<F, Fut>(&mut self, label: &'static str, call: F)
    where
        F: FnOnce(SessionHandle) -> Fut,
        Fut: Future<Output = Outcome> + Send + 'static,
    {
        let Some(handle) = self.handle.clone() else {
            return;
        };
        let call = call(handle);
        let clock = self.clock.clone();
        let incarnation = self.incarnation;
        self.pending.push(tokio::spawn(async move {
            let start = clock.fetch_add(1, Ordering::SeqCst);
            let outcome = tokio::time::timeout(CALL_TIMEOUT, call)
                .await
                .unwrap_or(Outcome::Hung(label));
            let end = clock.fetch_add(1, Ordering::SeqCst);
            Call {
                incarnation,
                start,
                end,
                outcome,
            }
        }));
    }

    async fn barrier(&mut self) {
        for call in std::mem::take(&mut self.pending) {
            self.run.calls.push(call.await.expect("call task"));
        }
    }

    async fn apply(&mut self, op: Op) {
        match op {
            Op::Submit(index) => {
                let message = self.messages[index].clone();
                let id = message_id(&message);
                self.spawn_call("submit", |handle| async move {
                    Outcome::Submit {
                        message: id,
                        result: handle.submit(message).await.map_err(|e| (&e).into()),
                    }
                });
            }
            Op::Discard(index) => {
                let id = message_id(&self.messages[index]);
                self.spawn_call("discard", |handle| async move {
                    Outcome::Discard {
                        result: handle.discard(&id).await.map_err(|e| (&e).into()),
                        message: id,
                    }
                });
            }
            Op::Interrupt => self.spawn_call("interrupt", |handle| async move {
                Outcome::Interrupt(handle.interrupt(None).await.map_err(|e| (&e).into()))
            }),
            // Compaction waits on the provider, so a hung provider can
            // legitimately stall it. The liveness rule is that an interrupt
            // always frees it.
            Op::Compact => self.spawn_call("compact", |handle| async move {
                let stage = CALL_TIMEOUT / 3;
                let compact = handle.compact("model test", None);
                tokio::pin!(compact);
                if let Ok(result) = tokio::time::timeout(stage, &mut compact).await {
                    return Outcome::Compact(result.map_err(|e| (&e).into()));
                }
                if tokio::time::timeout(stage, handle.interrupt(None))
                    .await
                    .is_err()
                {
                    return Outcome::Hung("interrupt of stalled compaction");
                }
                match tokio::time::timeout(stage, compact).await {
                    Ok(result) => Outcome::Compact(result.map_err(|e| (&e).into())),
                    Err(_) => Outcome::Hung("compaction after interrupt"),
                }
            }),
            Op::Shutdown => self.spawn_call("shutdown", |handle| async move {
                Outcome::Shutdown(handle.shutdown(None).await.map_err(|e| (&e).into()))
            }),
            Op::Yield(times) => {
                for _ in 0..times {
                    tokio::task::yield_now().await;
                }
            }
            Op::Barrier => self.barrier().await,
            Op::WaitIdle => self.wait_idle().await,
            Op::Reopen => self.reopen().await,
            Op::Crash => self.crash().await,
            Op::Script(step) => self.provider.script.lock().unwrap().push_back(step),
            Op::Fault(fault) => self.store.apply(fault),
        }
    }

    async fn wait_idle(&mut self) {
        let Some(handle) = self.handle.clone() else {
            return;
        };
        let mut status = handle.subscribe_status();
        let settled = tokio::time::timeout(IDLE_TIMEOUT, async {
            loop {
                let current = *status.borrow_and_update();
                if current != SessionStatus::Running {
                    return current;
                }
                if status.changed().await.is_err() {
                    return SessionStatus::Closed;
                }
            }
        })
        .await;
        // A liveness snapshot is sound only if no client call can add input
        // while it is taken: with none in flight, and the session idle both
        // before and after the replay, the log is the idle driver's inbox.
        let quiescent = |pending: &[JoinHandle<Call>]| pending.iter().all(JoinHandle::is_finished);
        if settled == Ok(SessionStatus::Idle)
            && quiescent(&self.pending)
            && let Ok(log) = handle.replay().await
            && quiescent(&self.pending)
            && handle.status() == SessionStatus::Idle
        {
            self.run.idle_logs.push(log);
        }
    }

    /// Close the current incarnation and record its stream.
    async fn close(&mut self) {
        self.barrier().await;
        if let Some(handle) = self.handle.take() {
            let shutdown = tokio::time::timeout(CALL_TIMEOUT, handle.shutdown(None)).await;
            if shutdown.is_err() {
                self.run.failures.push("shutdown hung while closing".into());
            }
        }
        if let Some(stream) = self.stream.take() {
            match tokio::time::timeout(CALL_TIMEOUT, stream).await {
                Ok(drained) => {
                    let (events, stream_error) = drained.expect("stream task");
                    self.run.incarnations.push(Incarnation {
                        events,
                        stream_error,
                    });
                }
                Err(_) => self.run.failures.push(format!(
                    "incarnation {} stream never closed",
                    self.incarnation
                )),
            }
        }
    }

    async fn reopen(&mut self) {
        self.close().await;
        self.resume().await;
    }

    /// Kill the process: nothing it does from here on is durable. In-flight
    /// calls finish against the dead store and are still checked.
    async fn crash(&mut self) {
        self.store.dead.store(true, Ordering::SeqCst);
        self.barrier().await;
        self.handle = None;
        if let Some(stream) = self.stream.take() {
            stream.abort();
        }
        let (runtime, services, store) = Self::process(&self.provider, self.durable.open());
        self.crashed
            .push(std::mem::replace(&mut self.runtime, runtime));
        self.services = services;
        self.store = store;
        self.resume().await;
    }

    async fn resume(&mut self) {
        self.incarnation += 1;
        match tokio::time::timeout(CALL_TIMEOUT, self.runtime.resume_session(&self.id)).await {
            Ok(Ok((handle, events))) => {
                self.handle = Some(handle);
                self.stream = Some(drain(events));
            }
            // A fault may fail the resume; later ops no-op until a retry.
            Ok(Err(_)) => {}
            Err(_) => self.run.failures.push("resume hung".into()),
        }
    }

    async fn finish(mut self) -> Run {
        self.close().await;
        let report = self.runtime.shutdown(Duration::from_secs(5)).await;
        if report.timed_out {
            self.run.failures.push("runtime shutdown timed out".into());
        }
        for crashed in std::mem::take(&mut self.crashed) {
            crashed.shutdown(Duration::from_secs(5)).await;
        }
        self.run.log = self.services.sessions.replay(&self.id).await.unwrap();
        let mut stored = self
            .services
            .sessions
            .load_session(&self.id)
            .await
            .unwrap()
            .unwrap();
        crate::session::hydrate_stored_session(self.services.sessions.as_ref(), &mut stored)
            .await
            .unwrap();
        self.run.stored_state = stored.state;
        self.run.provider_requests = std::mem::take(&mut *self.provider.requests.lock().unwrap());
        self.run
    }
}

fn message_id(message: &Message) -> MessageId {
    let Message::User(user) = message else {
        unreachable!("pool holds user messages");
    };
    user.id.clone()
}

fn execute(ops: &[Op], backend: Backend) -> Run {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .start_paused(true)
        .build()
        .unwrap();
    let workspace = tempfile::tempdir().unwrap();
    let storage = tempfile::tempdir().unwrap();
    runtime.block_on(async {
        let durable = Durable::new(backend, storage.path());
        let mut harness = Harness::new(workspace.path(), durable).await;
        for &op in ops {
            harness.apply(op).await;
            tokio::task::yield_now().await;
        }
        harness.finish().await
    })
}

// ---------------------------------------------------------------------------
// Invariants
// ---------------------------------------------------------------------------

/// Log positions of each kind of input event, per message.
#[derive(Default)]
struct InputHistory {
    accepted: Vec<u64>,
    delivered: Vec<u64>,
    transcribed: Vec<u64>,
    rejected: Vec<u64>,
}

fn input_histories(log: &[SessionEvent]) -> HashMap<MessageId, InputHistory> {
    let mut histories: HashMap<MessageId, InputHistory> = HashMap::new();
    for event in log {
        let sequence = event.sequence();
        match &event.payload {
            SessionEventPayload::InputAccepted { message } => {
                histories
                    .entry(message.id.clone())
                    .or_default()
                    .accepted
                    .push(sequence);
            }
            SessionEventPayload::InputDelivered { message_id } => {
                histories
                    .entry(message_id.clone())
                    .or_default()
                    .delivered
                    .push(sequence);
            }
            SessionEventPayload::InputRejected { message_id, .. } => {
                histories
                    .entry(message_id.clone())
                    .or_default()
                    .rejected
                    .push(sequence);
            }
            SessionEventPayload::MessageItem {
                message: Message::User(user),
            } => {
                histories
                    .entry(user.id.clone())
                    .or_default()
                    .transcribed
                    .push(sequence);
            }
            _ => {}
        }
    }
    histories
}

fn halter_protocol_compaction_marker() -> &'static str {
    static MARKER: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    MARKER.get_or_init(|| {
        let instructions = crate::compaction_instructions(None);
        instructions.lines().next().unwrap_or_default().to_owned()
    })
}

/// Scheduling rules, checked by replaying the log with the fold:
///
/// - A turn starts only while some pending input is ready: neither deferred
///   nor carried across a resume since its latest acceptance. Interrupted,
///   failed, and resumed input waits for the next submission.
/// - Input deferred as `ExecutionFailed` is delivered only after an explicit
///   resubmission re-accepts it.
/// - Discard is refused while a turn runs, so no rejection lands in a turn.
fn check_scheduling(log: &[SessionEvent]) {
    let mut state = SessionState::default();
    let mut ready: HashMap<MessageId, bool> = HashMap::new();
    let mut failed: BTreeSet<MessageId> = BTreeSet::new();
    let mut in_turn = false;
    for event in log {
        let sequence = event.sequence();
        match &event.payload {
            SessionEventPayload::InputAccepted { message } => {
                ready.insert(message.id.clone(), true);
                failed.remove(&message.id);
            }
            SessionEventPayload::InputDeferred { message_id, reason } => {
                ready.insert(message_id.clone(), false);
                if matches!(
                    reason,
                    halter_protocol::InputDeferredReason::ExecutionFailed { .. }
                ) {
                    failed.insert(message_id.clone());
                }
            }
            SessionEventPayload::SessionResumed => ready.values_mut().for_each(|r| *r = false),
            SessionEventPayload::TurnStarted { .. } => {
                assert!(
                    state
                        .pending_inputs
                        .iter()
                        .any(|input| ready.get(&input.id).copied().unwrap_or(false)),
                    "turn started at {sequence} without ready input"
                );
                in_turn = true;
            }
            SessionEventPayload::TurnCompleted { .. }
            | SessionEventPayload::TurnFailed { .. }
            | SessionEventPayload::SessionShutdownComplete => in_turn = false,
            SessionEventPayload::InputDelivered { message_id } => assert!(
                !failed.contains(message_id),
                "failed input {message_id} delivered at {sequence} without resubmission"
            ),
            SessionEventPayload::InputRejected { message_id, .. } => assert!(
                !in_turn,
                "input {message_id} rejected at {sequence} while a turn was running"
            ),
            _ => {}
        }
        halter_protocol::fold::apply_event(&mut state, &event.payload);
    }
}

fn check(run: &Run) {
    assert!(
        run.failures.is_empty(),
        "harness failures: {:?}",
        run.failures
    );
    for call in &run.calls {
        if let Outcome::Hung(label) = call.outcome {
            panic!(
                "client call deadlocked: {label} (incarnation {})",
                call.incarnation
            );
        }
    }

    let log = &run.log;
    let by_sequence: BTreeMap<u64, &SessionEvent> =
        log.iter().map(|event| (event.sequence(), event)).collect();
    assert!(
        log.windows(2)
            .all(|pair| pair[0].sequence() < pair[1].sequence()),
        "log sequences must strictly increase"
    );

    let histories = input_histories(log);
    let receipts: BTreeSet<u64> = run
        .calls
        .iter()
        .filter_map(|call| match &call.outcome {
            Outcome::Submit {
                result: Ok(receipt),
                ..
            } => Some(receipt.sequence),
            _ => None,
        })
        .collect();

    for call in &run.calls {
        match &call.outcome {
            // A receipt is a durability promise: it names the acceptance.
            Outcome::Submit {
                message,
                result: Ok(receipt),
            } => {
                assert_eq!(&receipt.message_id, message, "receipt names its input");
                let accepted = by_sequence
                    .get(&receipt.sequence)
                    .map(|event| &event.payload);
                assert!(
                    matches!(accepted, Some(SessionEventPayload::InputAccepted { message: m }) if &m.id == message),
                    "receipt {receipt:?} does not point at its InputAccepted event: {accepted:?}"
                );
            }
            // A successful discard is durable.
            Outcome::Discard {
                message,
                result: Ok(true),
            } => assert!(
                histories
                    .get(message)
                    .is_some_and(|h| !h.rejected.is_empty()),
                "discard of {message} succeeded without an InputRejected"
            ),
            _ => {}
        }
    }

    // Runtime-authored user messages (compaction prompts and summaries) are
    // transcribed without being inputs; the input rules apply to inputs.
    for (id, history) in histories.iter().filter(|(_, h)| !h.accepted.is_empty()) {
        // No acceptance is recorded unless a caller was told it succeeded.
        for sequence in &history.accepted {
            assert!(
                receipts.contains(sequence),
                "InputAccepted for {id} at {sequence} was never acknowledged"
            );
        }
        // Exactly-once delivery, and the transcript agrees with delivery.
        assert!(
            history.delivered.len() <= 1,
            "{id} delivered {} times",
            history.delivered.len()
        );
        assert!(
            history.transcribed.len() <= 1,
            "{id} transcribed {} times",
            history.transcribed.len()
        );
        assert_eq!(
            history.delivered.is_empty(),
            history.transcribed.is_empty(),
            "{id}: delivery and transcript disagree"
        );
        if let Some(&delivered) = history.delivered.first() {
            assert!(
                history
                    .accepted
                    .iter()
                    .any(|&accepted| accepted < delivered),
                "{id} delivered without prior acceptance"
            );
            // A rejected input is only delivered after a fresh acceptance.
            for &rejected in history.rejected.iter().filter(|&&r| r < delivered) {
                assert!(
                    history
                        .accepted
                        .iter()
                        .any(|&accepted| rejected < accepted && accepted < delivered),
                    "{id} delivered after rejection at {rejected} without re-acceptance"
                );
            }
        }
    }

    // Once shutdown returns, the incarnation admits nothing.
    for shutdown in &run.calls {
        if !matches!(shutdown.outcome, Outcome::Shutdown(Ok(()))) {
            continue;
        }
        for call in run
            .calls
            .iter()
            .filter(|call| call.incarnation == shutdown.incarnation && call.start > shutdown.end)
        {
            let closed = match &call.outcome {
                Outcome::Submit { result, .. } => result.as_ref().err() == Some(&ErrorKind::Closed),
                Outcome::Discard { result, .. } => {
                    result.as_ref().err() == Some(&ErrorKind::Closed)
                }
                Outcome::Interrupt(result) | Outcome::Compact(result) => {
                    result.as_ref().err() == Some(&ErrorKind::Closed)
                }
                Outcome::Shutdown(_) | Outcome::Hung(_) => true,
            };
            assert!(
                closed,
                "call after completed shutdown was not refused: {call:?}"
            );
        }
    }

    // Turns never overlap and end exactly once. Only a shutdown may leave
    // a turn open (a failed terminal commit closes the session), and resume
    // must close such an orphaned turn before the session runs again.
    let mut open_turn = None;
    let mut orphaned = BTreeSet::new();
    let mut ended = BTreeSet::new();
    for event in log {
        let sequence = event.sequence();
        match &event.payload {
            SessionEventPayload::TurnStarted { turn_id, .. } => {
                assert!(
                    open_turn.is_none(),
                    "turn started at {sequence} inside an open turn"
                );
                open_turn = Some(turn_id.clone());
            }
            SessionEventPayload::TurnCompleted { turn_id, .. }
            | SessionEventPayload::TurnFailed { turn_id, .. } => {
                assert!(
                    ended.insert(turn_id.clone()),
                    "turn ended twice at {sequence}"
                );
                if open_turn.as_ref() == Some(turn_id) {
                    open_turn = None;
                } else {
                    assert!(
                        orphaned.remove(turn_id),
                        "turn ended at {sequence} without being open"
                    );
                }
            }
            SessionEventPayload::SessionShutdownComplete => {
                orphaned.extend(open_turn.take());
            }
            SessionEventPayload::SessionResumed => {
                assert!(
                    orphaned.is_empty(),
                    "resumed at {sequence} with orphaned turns {orphaned:?}"
                );
            }
            _ => {}
        }
    }

    check_scheduling(log);

    // A compaction prompt belongs only to its own request, as the final
    // message. Anywhere earlier, a stale prompt leaked into the transcript.
    let marker = halter_protocol_compaction_marker();
    for (index, request) in run.provider_requests.iter().enumerate() {
        let earlier = &request[..request.len().saturating_sub(1)];
        assert!(
            // `starts_with`: FakeProvider echoes the prompt into the summary.
            !earlier.iter().any(|text| text.starts_with(marker)),
            "provider request {index} carries a stale compaction prompt"
        );
    }

    // The checkpoint is a cache of the log.
    let folded = fold_events(SessionState::default(), log);
    assert!(
        covered_state_matches(&folded, &run.stored_state),
        "checkpoint diverged from the log\nfolded pending: {:?}\nstored pending: {:?}",
        folded
            .pending_inputs
            .iter()
            .map(|m| &m.id)
            .collect::<Vec<_>>(),
        run.stored_state
            .pending_inputs
            .iter()
            .map(|m| &m.id)
            .collect::<Vec<_>>(),
    );

    // Liveness: an idle session holds no input that is ready to run. Every
    // pending input must have been deferred since its latest acceptance.
    for idle_log in &run.idle_logs {
        let pending = fold_events(SessionState::default(), idle_log).pending_inputs;
        let histories = input_histories(idle_log);
        for input in pending {
            let last_accepted = histories[&input.id]
                .accepted
                .iter()
                .max()
                .copied()
                .unwrap_or(0);
            let deferred = idle_log.iter().any(|event| {
                event.sequence() > last_accepted
                    && matches!(&event.payload,
                        SessionEventPayload::InputDeferred { message_id, .. } if message_id == &input.id)
            });
            assert!(deferred, "idle session holds ready input {}", input.id);
        }
    }

    // Each live stream shows committed events once, in order, without gaps.
    for (index, incarnation) in run.incarnations.iter().enumerate() {
        let committed: Vec<&SessionEvent> = incarnation
            .events
            .iter()
            .filter(|event| {
                !matches!(
                    event.payload,
                    SessionEventPayload::SessionStatusChanged { .. }
                )
            })
            .collect();
        assert!(
            committed
                .windows(2)
                .all(|pair| pair[0].sequence() < pair[1].sequence()),
            "incarnation {index} stream repeated or reordered events"
        );
        for event in &committed {
            assert_eq!(
                by_sequence.get(&event.sequence()).copied(),
                Some(*event),
                "incarnation {index} streamed an event that is not in the log"
            );
        }
        if let (Some(first), Some(last)) = (committed.first(), committed.last()) {
            let expected = by_sequence
                .range(first.sequence()..=last.sequence())
                .count();
            assert_eq!(
                committed.len(),
                expected,
                "incarnation {index} stream skipped committed events (stream error: {:?})",
                incarnation.stream_error
            );
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 128,
        ..ProptestConfig::default()
    })]

    #[test]
    fn session_driver_model(ops in prop::collection::vec(op_strategy(), 1..40)) {
        check(&execute(&ops, Backend::Memory));
    }
}

proptest! {
    // Each SQLite case does real file I/O, so it samples fewer cases per run;
    // the nightly soak raises `PROPTEST_CASES` for both backends.
    #![proptest_config(ProptestConfig {
        cases: 32,
        ..ProptestConfig::default()
    })]

    #[test]
    fn session_driver_model_sqlite(ops in prop::collection::vec(op_strategy(), 1..40)) {
        check(&execute(&ops, Backend::Sqlite));
    }
}

/// Shrunk cases from earlier runs, pinned so they replay regardless of how
/// the generator evolves. Each once failed an invariant that was then
/// corrected to match the driver's intended behaviour.
#[test]
fn pinned_scenarios() {
    use {Op::*, ProviderStep::*, StoreFault::*};
    let scenarios: &[&[Op]] = &[
        // The hang meant for the turn is consumed by compaction instead,
        // which must stay cancellable.
        &[
            Script(HangCreate),
            Submit(0),
            Interrupt,
            Barrier,
            Compact,
            Yield(1),
        ],
        // A compaction summary echoes the prompt; a second pass is clean.
        &[Submit(0), WaitIdle, Compact, Barrier, Compact],
        // A failed terminal commit orphans the turn; resume must close it.
        &[
            Fault(Terminal(true)),
            Submit(0),
            WaitIdle,
            Fault(Terminal(false)),
            Reopen,
        ],
        // A delivered input resubmitted after resume keeps its receipt.
        &[Submit(0), Reopen, Submit(0)],
        // Input left ready by a crash must wait for the next submission.
        &[
            Script(HangMidStream),
            Submit(0),
            Yield(4),
            Submit(1),
            Yield(4),
            Crash,
            WaitIdle,
        ],
    ];
    for ops in scenarios {
        for backend in [Backend::Memory, Backend::Sqlite] {
            check(&execute(ops, backend));
        }
    }
}

/// Diagnostic, not a check: prints how many generated runs reach each event
/// and call outcome, to spot behaviour the generator never exercises.
/// `cargo test -p halter-runtime generator_coverage -- --ignored --nocapture`
#[test]
#[ignore = "diagnostic report"]
fn generator_coverage() {
    use proptest::strategy::ValueTree;
    use proptest::test_runner::TestRunner;
    let mut runner = TestRunner::deterministic();
    let strategy = prop::collection::vec(op_strategy(), 1..40);
    let mut counts: BTreeMap<String, usize> = BTreeMap::new();
    let runs = 200;
    for _ in 0..runs {
        let ops = strategy.new_tree(&mut runner).unwrap().current();
        let run = execute(&ops, Backend::Memory);
        let mut seen = BTreeSet::new();
        for event in &run.log {
            let name = format!("{:?}", event.payload);
            let name = name.split([' ', '{', '(']).next().unwrap().to_owned();
            let name = match &event.payload {
                SessionEventPayload::InputDeferred { reason, .. } => format!(
                    "InputDeferred::{}",
                    format!("{reason:?}").split([' ', '{']).next().unwrap()
                ),
                _ => name,
            };
            seen.insert(name);
        }
        for call in &run.calls {
            let name = match &call.outcome {
                Outcome::Submit { result, .. } => {
                    format!("call submit {:?}", result.as_ref().map(|_| "ok"))
                }
                Outcome::Discard { result, .. } => format!("call discard {result:?}"),
                Outcome::Interrupt(r) => format!("call interrupt {r:?}"),
                Outcome::Compact(r) => format!("call compact {r:?}"),
                Outcome::Shutdown(r) => format!("call shutdown {r:?}"),
                Outcome::Hung(l) => format!("call hung {l}"),
            };
            seen.insert(name);
        }
        if run.incarnations.len() > 1 {
            seen.insert("reopened".into());
        }
        if run.incarnations.iter().any(|i| i.stream_error.is_some()) {
            seen.insert("stream error".into());
        }
        if run.incarnations.iter().any(|i| {
            i.events
                .iter()
                .any(|e| matches!(e.payload, SessionEventPayload::Lagged { .. }))
        }) {
            seen.insert("stream lagged".into());
        }
        if !run.idle_logs.is_empty() {
            seen.insert("idle liveness checked".into());
        }
        if !run.stored_state.pending_inputs.is_empty() {
            seen.insert("ends with pending input".into());
        }
        for name in seen {
            *counts.entry(name).or_default() += 1;
        }
    }
    for (name, count) in counts {
        println!("{count:>4}/{runs} {name}");
    }
}
