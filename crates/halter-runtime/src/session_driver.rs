//! A session owns admission, its durable inbox, and its live execution.
// pattern: Imperative Shell

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use async_trait::async_trait;
use futures::{FutureExt, StreamExt, TryStreamExt, stream};
use halter_protocol::{
    InputDeferredReason, Message, MessageId, PendingEvent, ResourceSnapshot, SessionBlueprint,
    SessionEvent, SessionEventPayload, SessionId, SessionState, SessionStatus, Turn, TurnId,
    UserMessage,
};
use halter_session::{SessionStore, StoredSession};
use thiserror::Error;
use tokio::sync::{broadcast, mpsc, oneshot, watch};
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use crate::session::{
    RuntimeServices, SessionEventStream, SessionExecutor, hydrate_stored_session,
};
use crate::subagents::RuntimeSubagentControl;

const INBOX_CAPACITY: usize = 128;
const FORWARDED_EVENT_CAPACITY: usize = 128;

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("session is closed")]
    Closed,
    #[error("session '{0}' is already open")]
    AlreadyOpen(SessionId),
    #[error("unknown session '{0}'")]
    NotFound(SessionId),
    #[error("only user messages can be submitted")]
    InvalidMessage,
    #[error("session inbox is full")]
    InboxFull,
    #[error("session is busy")]
    Busy,
    #[error("session cleanup exceeded the requested timeout")]
    TimedOut,
    #[error(transparent)]
    Operation(#[from] anyhow::Error),
}

type Reply<T> = oneshot::Sender<Result<T, SessionError>>;

impl SessionError {
    /// The original operation failure, including its concrete error type.
    pub fn operation_error(&self) -> Option<&anyhow::Error> {
        let Self::Operation(error) = self else {
            return None;
        };
        let mut original = error;
        while let Some(shared) = original.downcast_ref::<SharedOperationError>() {
            original = shared.0.as_ref();
        }
        Some(original)
    }
}

/// Durable admission receipt. This acknowledges queuing, not completion of
/// the request or background work. Repeating a delivered ID returns its receipt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Submission {
    pub message_id: MessageId,
    pub sequence: u64,
}

/// A cloneable connection to one live incarnation of a stored session.
/// Once every handle is dropped, the session closes after foreground work,
/// background jobs and subagents settle. Event streams do not keep it open.
#[derive(Clone)]
pub struct SessionHandle {
    control: Arc<DriverControl>,
    _token: Arc<HandleToken>,
}

// Only public handles hold this token. Internal control references and event
// streams cannot revive handle ownership after the final handle is dropped.
struct HandleToken {
    stop_tx: mpsc::UnboundedSender<Command>,
}

impl Drop for HandleToken {
    fn drop(&mut self) {
        let _ = self.stop_tx.send(Command::Released);
    }
}

struct DriverControl {
    id: SessionId,
    tx: mpsc::Sender<Command>,
    stop_tx: mpsc::UnboundedSender<Command>,
    closed: CancellationToken,
    // Published before releasing the incarnation reservation. Old streams
    // must never replay events appended by a subsequently opened driver.
    final_head: AtomicU64,
    services: Arc<RuntimeServices>,
    forwarded: broadcast::Sender<SessionEvent>,
    status: watch::Sender<SessionStatus>,
    failure: std::sync::Mutex<Option<SharedOperationError>>,
}

impl SessionHandle {
    #[must_use]
    pub fn id(&self) -> &SessionId {
        &self.control.id
    }

    #[must_use]
    pub fn session_id(&self) -> &SessionId {
        self.id()
    }

    /// Current foreground activity. Background processes may run while idle.
    #[must_use]
    pub fn status(&self) -> SessionStatus {
        *self.control.status.borrow()
    }

    /// Observe current activity. Intermediate changes may be coalesced; this
    /// is not an acknowledgement of delivery or completion of submitted input.
    #[must_use]
    pub fn subscribe_status(&self) -> watch::Receiver<SessionStatus> {
        self.control.status.subscribe()
    }

    /// Remove an input that is still queued. Requires an idle session to avoid
    /// racing delivery. Returns false if it has already left the inbox.
    pub async fn discard(&self, message_id: &MessageId) -> Result<bool, SessionError> {
        self.request(|reply| Command::Discard(message_id.clone(), reply))
            .await
    }

    /// Record input before acknowledging it. Active execution sees it at its
    /// next safe boundary; an idle session starts execution.
    pub async fn submit(&self, message: Message) -> Result<Submission, SessionError> {
        let Message::User(message) = message else {
            return Err(SessionError::InvalidMessage);
        };
        self.request(|reply| Command::Submit(message, reply)).await
    }

    /// Stop foreground execution, await cleanup and commit its final state.
    /// Earlier pending inputs remain recorded but do not restart execution.
    /// `None` waits without a deadline. A timeout requests forced recovery;
    /// call with `None` to await its settlement.
    pub async fn interrupt(&self, timeout: Option<Duration>) -> Result<(), SessionError> {
        let deadline = timeout_deadline(timeout)?;
        let response = self
            .control
            .enqueue_stop(|reply| Command::Interrupt { deadline, reply })?;
        let request = async { response.await.map_err(|_| SessionError::Closed)? };
        match deadline {
            Some(deadline) => tokio::time::timeout_at(deadline, request)
                .await
                .unwrap_or(Err(SessionError::TimedOut)),
            None => request.await,
        }
    }

    /// Close this incarnation and await cleanup of its session resources.
    /// `None` waits without a deadline. A timeout requests forced cleanup and
    /// returns `TimedOut`; the session stays fenced until cleanup settles.
    pub async fn shutdown(&self, timeout: Option<Duration>) -> Result<(), SessionError> {
        self.control.shutdown(timeout).await
    }

    /// Compact an idle session explicitly.
    pub async fn compact(
        &self,
        reason: &str,
        instructions: Option<&str>,
    ) -> Result<(), SessionError> {
        self.request(|reply| Command::Compact {
            reason: reason.to_owned(),
            instructions: instructions.map(str::to_owned),
            reply,
        })
        .await
    }

    pub async fn replay(&self) -> anyhow::Result<Vec<SessionEvent>> {
        self.control.services.sessions.replay(self.id()).await
    }

    pub async fn export_trace(&self) -> anyhow::Result<String> {
        crate::export_session_trace(self.control.services.sessions.as_ref(), self.id()).await
    }

    async fn request<T>(
        &self,
        command: impl FnOnce(Reply<T>) -> Command,
    ) -> Result<T, SessionError> {
        if self.control.closed.is_cancelled() {
            return Err(SessionError::Closed);
        }
        let (reply, response) = oneshot::channel();
        self.control
            .tx
            .send(command(reply))
            .await
            .map_err(|_| SessionError::Closed)?;
        response.await.map_err(|_| SessionError::Closed)?
    }
}

impl DriverControl {
    // Runtime shutdown holds controls directly; it must not create a new
    // public handle after handle ownership has ended.
    async fn shutdown(&self, timeout: Option<Duration>) -> Result<(), SessionError> {
        let deadline = timeout_deadline(timeout)?;
        let response = if self.closed.is_cancelled() {
            None
        } else {
            match self.enqueue_stop(|reply| Command::Shutdown { deadline, reply }) {
                Ok(response) => Some(response),
                Err(SessionError::Closed) => None,
                Err(error) => return Err(error),
            }
        };
        let cleanup = self.shutdown_until(response);
        match deadline {
            Some(deadline) => tokio::time::timeout_at(deadline, cleanup)
                .await
                .unwrap_or(Err(SessionError::TimedOut)),
            None => cleanup.await,
        }
    }

    async fn shutdown_until(
        &self,
        response: Option<oneshot::Receiver<Result<(), SessionError>>>,
    ) -> Result<(), SessionError> {
        if let Some(response) = response {
            match response.await.unwrap_or(Err(SessionError::Closed)) {
                Err(SessionError::Closed) => self.closed.cancelled().await,
                result => return result,
            }
        } else if !self.closed.is_cancelled() {
            self.closed.cancelled().await;
        }
        match self
            .failure
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
        {
            Some(error) => Err(SessionError::Operation(anyhow::Error::new(error.clone()))),
            None => Ok(()),
        }
    }

    fn enqueue_stop(
        &self,
        command: impl FnOnce(Reply<()>) -> Command,
    ) -> Result<oneshot::Receiver<Result<(), SessionError>>, SessionError> {
        if self.closed.is_cancelled() {
            return Err(SessionError::Closed);
        }
        let (reply, response) = oneshot::channel();
        let command = command(reply);
        // Stop admission must survive a timed-out caller and cannot wait for
        // ordinary inbox capacity. Its separate lane also preserves call order.
        self.stop_tx
            .send(command)
            .map_err(|_| SessionError::Closed)?;
        Ok(response)
    }
}

pub(crate) struct SessionDrivers {
    entries: Mutex<HashMap<SessionId, DriverSlot>>,
    store: Arc<dyn SessionStore>,
}

enum DriverSlot {
    Opening(SessionId),
    Open(Weak<DriverControl>),
    // Keep reopen fenced while failure cleanup writes directly to storage.
    Failed(Weak<DriverControl>),
}

pub(crate) struct OpeningSession {
    drivers: Arc<SessionDrivers>,
    id: SessionId,
    reservation: SessionId,
}

impl Drop for OpeningSession {
    fn drop(&mut self) {
        let drivers = self.drivers.clone();
        let id = self.id.clone();
        let reservation = self.reservation.clone();
        let mut entries = drivers.lock_entries();
        if matches!(entries.get(&id), Some(DriverSlot::Opening(current)) if current == &reservation)
        {
            entries.remove(&id);
        }
    }
}

impl SessionDrivers {
    fn lock_entries(&self) -> std::sync::MutexGuard<'_, HashMap<SessionId, DriverSlot>> {
        self.entries.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub(crate) fn new(store: Arc<dyn SessionStore>) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            store,
        }
    }

    pub(crate) fn routed_store(self: &Arc<Self>) -> Arc<dyn SessionStore> {
        Arc::new(RoutedStore {
            drivers: Arc::downgrade(self),
            store: self.store.clone(),
        })
    }

    pub(crate) fn reserve(
        self: &Arc<Self>,
        id: &SessionId,
    ) -> Result<OpeningSession, SessionError> {
        let mut entries = self.lock_entries();
        let occupied = match entries.get(id) {
            Some(DriverSlot::Opening(_)) => true,
            Some(DriverSlot::Open(driver) | DriverSlot::Failed(driver)) => {
                driver.upgrade().is_some_and(|d| !d.closed.is_cancelled())
            }
            None => false,
        };
        if occupied {
            return Err(SessionError::AlreadyOpen(id.clone()));
        }
        let reservation = SessionId::new();
        entries.insert(id.clone(), DriverSlot::Opening(reservation.clone()));
        Ok(OpeningSession {
            drivers: self.clone(),
            id: id.clone(),
            reservation,
        })
    }

    pub(crate) fn release(&self, id: &SessionId) {
        self.lock_entries().remove(id);
    }

    pub(crate) async fn close_all(&self, deadline: Option<Instant>) -> bool {
        let controls = self
            .lock_entries()
            .values()
            .filter_map(|entry| match entry {
                DriverSlot::Open(driver) | DriverSlot::Failed(driver) => driver.upgrade(),
                DriverSlot::Opening(_) => None,
            })
            .collect::<Vec<_>>();
        futures::future::join_all(controls.into_iter().map(|control| async move {
            if let Err(error) = control.shutdown(deadline.map(|d| d.saturating_duration_since(Instant::now()))).await {
                tracing::warn!(session_id = %control.id, %error, "session cleanup failed during runtime shutdown");
                return matches!(error, SessionError::TimedOut);
            }
            false
        }))
        .await.into_iter().any(|timed_out| timed_out)
    }

    pub(crate) async fn open(
        self: &Arc<Self>,
        executor: SessionExecutor,
        services: Arc<RuntimeServices>,
        subagents: RuntimeSubagentControl,
        after: u64,
        history: Option<Vec<SessionEvent>>,
    ) -> Result<(SessionHandle, SessionEventStream), SessionError> {
        let id = executor.session_id().clone();
        let mut stored = services
            .sessions
            .load_session(&id)
            .await?
            .ok_or_else(|| SessionError::NotFound(id.clone()))?;
        hydrate_stored_session(services.sessions.as_ref(), &mut stored).await?;
        let (tx, rx) = mpsc::channel(INBOX_CAPACITY);
        let (stop_tx, stop_rx) = mpsc::unbounded_channel();
        let control = Arc::new(DriverControl {
            id: id.clone(),
            tx: tx.clone(),
            stop_tx,
            closed: CancellationToken::new(),
            final_head: AtomicU64::new(u64::MAX),
            services: services.clone(),
            forwarded: broadcast::channel(FORWARDED_EVENT_CAPACITY).0,
            status: watch::channel(SessionStatus::Idle).0,
            failure: std::sync::Mutex::new(None),
        });
        let token = Arc::new(HandleToken {
            stop_tx: control.stop_tx.clone(),
        });
        let events = session_events(control.clone(), after);
        let store = Arc::new(SessionInbox { tx });
        let executor = executor.with_inbox(store);
        let mut driver = Driver {
            executor,
            services: services.clone(),
            subagents,
            store: self.store.clone(),
            control: control.clone(),
            handles: Arc::downgrade(&token),
            rx,
            stop_rx,
            head: stored.head_sequence,
            last_state_commit: stored.head_sequence,
            inbox: Inbox {
                pending: stored.state.pending_inputs,
                ..Inbox::default()
            },
            phase: Phase::Idle,
        };
        let history = match history {
            Some(history) => history,
            None => services.sessions.replay(&id).await?,
        };
        for event in history {
            let sequence = event.sequence();
            match event.payload {
                SessionEventPayload::InputAccepted { message } => {
                    driver.inbox.deferred.remove(&message.id);
                    driver.inbox.accepted.insert(message.id, sequence);
                }
                SessionEventPayload::InputDeferred { message_id, reason } => {
                    driver.inbox.deferred.insert(message_id, reason);
                }
                SessionEventPayload::InputDelivered { message_id }
                | SessionEventPayload::InputRejected { message_id, .. } => {
                    driver.inbox.deferred.remove(&message_id);
                }
                _ => {}
            }
        }
        driver
            .inbox
            .deferred
            .retain(|id, _| driver.inbox.pending.iter().any(|m| &m.id == id));
        driver.defer_pending(InputDeferredReason::Resumed).await?;
        {
            let mut entries = self.lock_entries();
            if services.turn_registry.is_shutting_down() {
                return Err(SessionError::Closed);
            }
            entries.insert(id.clone(), DriverSlot::Open(Arc::downgrade(&control)));
        }
        services.tool_sessions.open_session(&id);
        let registry = self.clone();
        let task_control = control.clone();
        tokio::spawn(async move {
            let result = std::panic::AssertUnwindSafe(driver.run())
                .catch_unwind()
                .await
                .unwrap_or_else(|_| Err(anyhow::anyhow!("session driver panicked")));
            let failure = result
                .err()
                .map(|error| SharedOperationError(Arc::new(error)));
            if let Some(error) = &failure {
                *task_control
                    .failure
                    .lock()
                    .unwrap_or_else(|p| p.into_inner()) = Some(error.clone());
                tracing::error!(session_id = %id, %error, "session driver failed");
                driver.rx.close();
                driver.stop_rx.close();
                while let Ok(command) = driver.rx.try_recv() {
                    drop(command);
                }
                while let Ok(command) = driver.stop_rx.try_recv() {
                    drop(command);
                }
                registry.lock_entries().insert(
                    id.clone(),
                    DriverSlot::Failed(Arc::downgrade(&task_control)),
                );
                driver.recover_failed_driver().await;
            }
            let replies = driver
                .phase
                .stop_mut()
                .map(StopRequest::take_replies)
                .unwrap_or_default();
            task_control
                .final_head
                .store(driver.head, Ordering::Release);
            registry.release(&id);
            task_control.status.send_replace(SessionStatus::Closed);
            task_control.closed.cancel();
            respond(
                replies,
                failure.map_or(Ok(()), |error| Err(anyhow::Error::new(error))),
            );
        });
        Ok((
            SessionHandle {
                control,
                _token: token,
            },
            events,
        ))
    }
}

// Lifecycle hooks and child-status updates use the same commit arbiter as
// execution. Sessions without a live driver keep their ordinary store path.
struct RoutedStore {
    drivers: Weak<SessionDrivers>,
    store: Arc<dyn SessionStore>,
}

#[async_trait]
impl SessionStore for RoutedStore {
    async fn create_session(&self, session: StoredSession) -> anyhow::Result<()> {
        self.store.create_session(session).await
    }
    async fn load_session(&self, id: &SessionId) -> anyhow::Result<Option<StoredSession>> {
        self.store.load_session(id).await
    }
    async fn commit(
        &self,
        id: &SessionId,
        snapshot: Option<Arc<ResourceSnapshot>>,
        expected: Option<u64>,
        state: Option<SessionState>,
        events: Vec<PendingEvent>,
    ) -> anyhow::Result<Vec<SessionEvent>> {
        let control = if let Some(drivers) = self.drivers.upgrade() {
            match drivers.lock_entries().get(id) {
                Some(DriverSlot::Open(control)) => control.upgrade(),
                _ => None,
            }
        } else {
            None
        };
        if let Some(control) = control {
            let (reply, response) = oneshot::channel();
            control
                .tx
                .send(Command::Commit {
                    snapshot,
                    expected,
                    state: state.map(Box::new),
                    events,
                    reply,
                })
                .await
                .map_err(|_| anyhow::anyhow!("session is closed"))?;
            response
                .await
                .map_err(|_| anyhow::anyhow!("session is closed"))?
        } else {
            self.store
                .commit(id, snapshot, expected, state, events)
                .await
        }
    }
    async fn replay(&self, id: &SessionId) -> anyhow::Result<Vec<SessionEvent>> {
        self.store.replay(id).await
    }
    async fn synchronize(&self, id: &SessionId) -> anyhow::Result<()> {
        self.store.synchronize(id).await
    }
    async fn replay_after(&self, id: &SessionId, after: u64) -> anyhow::Result<Vec<SessionEvent>> {
        self.store.replay_after(id, after).await
    }
    async fn list_sessions(&self) -> anyhow::Result<Vec<SessionBlueprint>> {
        self.store.list_sessions().await
    }
    fn transcript_path(&self, id: &SessionId) -> Option<PathBuf> {
        self.store.transcript_path(id)
    }
}

// The executor reads pending input at its safe delivery boundaries.
pub(crate) struct SessionInbox {
    tx: mpsc::Sender<Command>,
}

impl SessionInbox {
    /// Record which input owns a failing admission hook. The actor serializes
    /// this with stop requests, so a stopped executor cannot claim more work.
    pub(crate) async fn attempt(&self, message_id: MessageId) -> anyhow::Result<()> {
        let (reply, response) = oneshot::channel();
        self.tx
            .send(Command::AttemptInput(message_id, reply))
            .await
            .map_err(|_| anyhow::anyhow!("session is closed"))?;
        response
            .await
            .map_err(|_| anyhow::anyhow!("session is closed"))?
    }
    /// Await the actor after the executor has stopped sending writes. In-flight
    /// blocking storage operations must finish before recovery reloads state.
    pub(crate) async fn synchronize(&self) -> anyhow::Result<()> {
        self.pending().await.map(|_| ())
    }

    pub(crate) async fn pending(&self) -> anyhow::Result<Vec<UserMessage>> {
        let (reply, response) = oneshot::channel();
        self.tx
            .send(Command::Pending(reply))
            .await
            .map_err(|_| anyhow::anyhow!("session is closed"))?;
        response
            .await
            .map_err(|_| anyhow::anyhow!("session is closed"))
    }
}

enum Command {
    Released,
    Submit(UserMessage, Reply<Submission>),
    Discard(MessageId, Reply<bool>),
    AttemptInput(MessageId, oneshot::Sender<anyhow::Result<()>>),
    Interrupt {
        deadline: Option<Instant>,
        reply: Reply<()>,
    },
    Shutdown {
        deadline: Option<Instant>,
        reply: Reply<()>,
    },
    Compact {
        reason: String,
        instructions: Option<String>,
        reply: Reply<()>,
    },
    Pending(oneshot::Sender<Vec<UserMessage>>),
    Commit {
        snapshot: Option<Arc<ResourceSnapshot>>,
        expected: Option<u64>,
        state: Option<Box<SessionState>>,
        events: Vec<PendingEvent>,
        reply: oneshot::Sender<anyhow::Result<Vec<SessionEvent>>>,
    },
}

struct Active {
    cancel: CancellationToken,
    task: JoinHandle<anyhow::Result<TaskOutcome>>,
}

struct Execution {
    active: Active,
    turn_id: TurnId,
    attempted_input: MessageId,
}

struct Compaction {
    active: Active,
    reply: Reply<()>,
}

enum TaskOutcome {
    Complete,
    TurnFailed,
    // No durable TurnStarted was observed: execution has not begun.
    NotStarted(anyhow::Error),
}

enum RunningWork {
    Execution(Execution),
    Compaction(Compaction),
}

impl RunningWork {
    fn active_mut(&mut self) -> &mut Active {
        match self {
            Self::Execution(work) => &mut work.active,
            Self::Compaction(work) => &mut work.active,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum StopIntent {
    Interrupt,
    Shutdown,
    Release,
}

impl StopIntent {
    fn is_shutdown(self) -> bool {
        self != Self::Interrupt
    }

    fn reason(self) -> &'static str {
        match self {
            Self::Release => "session_released",
            Self::Shutdown | Self::Interrupt => "session_closed",
        }
    }
}

struct StopRequest {
    intent: StopIntent,
    deadline: Option<Instant>,
    interrupt_replies: Vec<Reply<()>>,
    shutdown_replies: Vec<Reply<()>>,
}

impl StopRequest {
    fn new(intent: StopIntent, deadline: Option<Instant>) -> Self {
        Self {
            intent,
            deadline,
            interrupt_replies: Vec::new(),
            shutdown_replies: Vec::new(),
        }
    }

    fn add_deadline(&mut self, deadline: Option<Instant>) {
        if let Some(deadline) = deadline {
            self.deadline = Some(self.deadline.map_or(deadline, |old| old.min(deadline)));
        }
    }

    fn take_replies(&mut self) -> Vec<Reply<()>> {
        self.interrupt_replies
            .drain(..)
            .chain(self.shutdown_replies.drain(..))
            .collect()
    }
}

// Each phase owns its work and the callers waiting for that work to settle.
// A shutdown can upgrade Stopping, but cleanup starts exactly once.
enum Phase {
    Idle,
    Executing(Execution),
    Compacting(Compaction),
    Stopping {
        work: RunningWork,
        stop: StopRequest,
    },
    Closing {
        active: Active,
        stop: StopRequest,
    },
    Closed(StopRequest),
}

impl Phase {
    fn status(&self) -> SessionStatus {
        match self {
            Self::Idle => SessionStatus::Idle,
            Self::Closed(_) => SessionStatus::Closed,
            _ => SessionStatus::Running,
        }
    }

    fn active_mut(&mut self) -> Option<&mut Active> {
        match self {
            Self::Executing(work) => Some(&mut work.active),
            Self::Compacting(work) => Some(&mut work.active),
            Self::Stopping { work, .. } => Some(work.active_mut()),
            Self::Closing { active, .. } => Some(active),
            Self::Idle | Self::Closed(_) => None,
        }
    }

    fn stop_mut(&mut self) -> Option<&mut StopRequest> {
        match self {
            Self::Stopping { stop, .. } | Self::Closing { stop, .. } | Self::Closed(stop) => {
                Some(stop)
            }
            _ => None,
        }
    }

    fn is_closing(&self) -> bool {
        matches!(self, Self::Closing { .. } | Self::Closed(_))
            || matches!(self, Self::Stopping { stop, .. } if stop.intent.is_shutdown())
    }

    fn attempted_input(&self) -> Option<&MessageId> {
        match self {
            Self::Executing(work)
            | Self::Stopping {
                work: RunningWork::Execution(work),
                ..
            } => Some(&work.attempted_input),
            _ => None,
        }
    }
}

#[derive(Default)]
struct Inbox {
    pending: Vec<UserMessage>,
    accepted: HashMap<MessageId, u64>,
    deferred: HashMap<MessageId, InputDeferredReason>,
}

impl Inbox {
    fn ready(&self) -> bool {
        self.pending
            .iter()
            .any(|message| !self.deferred.contains_key(&message.id))
    }

    fn deliverable(&self) -> Vec<UserMessage> {
        self.pending
            .iter()
            .filter(|message| {
                !matches!(
                    self.deferred.get(&message.id),
                    Some(InputDeferredReason::ExecutionFailed { .. })
                )
            })
            .cloned()
            .collect()
    }
}

struct Driver {
    executor: SessionExecutor,
    services: Arc<RuntimeServices>,
    store: Arc<dyn SessionStore>,
    subagents: RuntimeSubagentControl,
    control: Arc<DriverControl>,
    handles: Weak<HandleToken>,
    rx: mpsc::Receiver<Command>,
    stop_rx: mpsc::UnboundedReceiver<Command>,
    head: u64,
    // Admission commits only modify pending_inputs. Routed state writers may
    // cross those commits because that field is overlaid below. They may never
    // cross another routed state replacement.
    last_state_commit: u64,
    inbox: Inbox,
    phase: Phase,
}

impl Driver {
    fn transition(&mut self, phase: Phase) {
        let status = phase.status();
        self.phase = phase;
        self.control.status.send_if_modified(|current| {
            if *current == status {
                false
            } else {
                *current = status;
                true
            }
        });
    }

    async fn run(&mut self) -> anyhow::Result<()> {
        let runtime_cancel = self.services.turn_registry.child_token();
        // Subscribe before the first check so completion between a check and
        // select remains observable. These watches never own a public handle.
        let mut jobs = self.services.tool_sessions.subscribe_activity();
        let mut children = self.subagents.subscribe_activity();
        loop {
            if matches!(self.phase, Phase::Idle) && self.inbox.ready() {
                self.start_execution();
            }
            self.release_if_unowned().await?;
            let active = self.phase.active_mut().is_some();
            let deadline = self.phase.stop_mut().and_then(|stop| stop.deadline);
            tokio::select! {
                biased;
                _ = wait_deadline(deadline), if deadline.is_some() => self.force_active(),
                Some(command) = self.stop_rx.recv() => self.command(command).await?,
                _ = runtime_cancel.cancelled(), if !self.phase.is_closing() => {
                    self.request_stop(StopIntent::Shutdown, None, None).await?;
                }
                Some(command) = self.rx.recv() => self.command(command).await?,
                outcome = async { (&mut self.phase.active_mut().expect("active phase").task).await }, if active => {
                    let result = outcome.map_err(anyhow::Error::from).and_then(|result| result);
                    if self.finish_work(result).await? { return Ok(()); }
                }
                _ = jobs.changed() => self.release_if_unowned().await?,
                _ = children.changed() => self.release_if_unowned().await?,
            }
        }
    }

    async fn command(&mut self, command: Command) -> anyhow::Result<()> {
        match command {
            Command::Released => {
                if !self.phase.is_closing() && !self.should_release() {
                    tracing::info!(
                        session_id = %self.control.id,
                        foreground = ?self.phase.status(),
                        "session has no remaining handles; waiting for owned work before release"
                    );
                }
                self.release_if_unowned().await?;
            }
            Command::Commit {
                snapshot,
                expected,
                state,
                events,
                reply,
            } => {
                let result = if (state.is_some() || snapshot.is_some()) && events.is_empty() {
                    Err(anyhow::anyhow!("session checkpoint requires an event"))
                } else if let Some(expected) = expected
                    && (expected < self.last_state_commit || expected > self.head)
                {
                    Err(halter_session::SessionCommitConflict {
                        session_id: self.control.id.clone(),
                        expected_head_sequence: expected,
                        actual_head_sequence: self.head,
                    }
                    .into())
                } else {
                    let result = self
                        .commit(snapshot, state.map(|state| *state), events)
                        .await;
                    if result.is_ok() {
                        self.last_state_commit = self.head;
                    }
                    result
                };
                let _ = reply.send(result);
            }
            Command::Pending(reply) => {
                let _ = reply.send(if matches!(self.phase, Phase::Executing(_)) {
                    self.inbox.deliverable()
                } else {
                    Vec::new()
                });
            }
            Command::AttemptInput(id, reply) => {
                let result = match &mut self.phase {
                    Phase::Executing(work) => {
                        work.attempted_input = id;
                        Ok(())
                    }
                    _ => Err(halter_protocol::ProviderError::cancelled().into()),
                };
                let _ = reply.send(result);
            }
            Command::Submit(message, reply) => {
                if self.phase.is_closing() {
                    let _ = reply.send(Err(SessionError::Closed));
                } else if let Some(sequence) = self.inbox.accepted.get(&message.id).copied() {
                    if let Some(pending) = self
                        .inbox
                        .pending
                        .iter()
                        .find(|pending| pending.id == message.id)
                        .cloned()
                    {
                        let id = pending.id.clone();
                        let result = self
                            .commit_payload(SessionEventPayload::InputAccepted { message: pending })
                            .await;
                        let _ = reply.send(
                            result
                                .map(|sequence| Submission {
                                    message_id: id,
                                    sequence,
                                })
                                .map_err(SessionError::from),
                        );
                    } else {
                        let _ = reply.send(Ok(Submission {
                            message_id: message.id,
                            sequence,
                        }));
                    }
                } else if self.inbox.pending.len() >= INBOX_CAPACITY {
                    let _ = reply.send(Err(SessionError::InboxFull));
                } else {
                    let id = message.id.clone();
                    let result = self
                        .commit_payload(SessionEventPayload::InputAccepted { message })
                        .await;
                    let _ = reply.send(
                        result
                            .map(|sequence| Submission {
                                message_id: id,
                                sequence,
                            })
                            .map_err(SessionError::from),
                    );
                }
            }
            Command::Discard(id, reply) => {
                let result = if self.phase.is_closing() {
                    Err(SessionError::Closed)
                } else if !matches!(self.phase, Phase::Idle) {
                    Err(SessionError::Busy)
                } else if !self.inbox.pending.iter().any(|message| message.id == id) {
                    Ok(false)
                } else {
                    self.commit_payload(SessionEventPayload::InputRejected {
                        message_id: id,
                        reason: "discarded by client".into(),
                    })
                    .await
                    .map(|_| true)
                    .map_err(SessionError::from)
                };
                let _ = reply.send(result);
            }
            Command::Interrupt { deadline, reply } => {
                self.request_stop(StopIntent::Interrupt, deadline, Some(reply))
                    .await?;
            }
            Command::Shutdown { deadline, reply } => {
                self.request_stop(StopIntent::Shutdown, deadline, Some(reply))
                    .await?;
            }
            Command::Compact {
                reason,
                instructions,
                reply,
            } => {
                if self.phase.is_closing() {
                    let _ = reply.send(Err(SessionError::Closed));
                } else if !matches!(self.phase, Phase::Idle) {
                    let _ = reply.send(Err(SessionError::Busy));
                } else {
                    let cancel = self.services.turn_registry.child_token();
                    let task_cancel = cancel.clone();
                    let executor = self.executor.clone();
                    let task = tokio::spawn(async move {
                        executor
                            .compact_with_cancel(&reason, instructions.as_deref(), task_cancel)
                            .await
                            .map(|()| TaskOutcome::Complete)
                    });
                    self.transition(Phase::Compacting(Compaction {
                        active: Active { cancel, task },
                        reply,
                    }));
                }
            }
        }
        Ok(())
    }

    fn should_release(&self) -> bool {
        matches!(self.phase, Phase::Idle)
            && !self.inbox.ready()
            && self.handles.upgrade().is_none()
            && !self
                .services
                .tool_sessions
                .has_running_jobs(&self.control.id)
            && !self.subagents.has_running_subagents(&self.control.id)
    }

    async fn release_if_unowned(&mut self) -> anyhow::Result<()> {
        if self.should_release() {
            // Deferred input is already durable and does not request work.
            // With no token left, clients cannot submit or clone a handle
            // between this check and closing admission.
            self.request_stop(StopIntent::Release, None, None).await?;
        }
        Ok(())
    }

    async fn request_stop(
        &mut self,
        intent: StopIntent,
        deadline: Option<Instant>,
        reply: Option<Reply<()>>,
    ) -> anyhow::Result<()> {
        if intent == StopIntent::Interrupt && self.phase.is_closing() {
            if let Some(reply) = reply {
                let _ = reply.send(Err(SessionError::Closed));
            }
            return Ok(());
        }
        let first_shutdown = intent.is_shutdown() && !self.phase.is_closing();
        let phase = std::mem::replace(&mut self.phase, Phase::Idle);
        let stop = match phase {
            Phase::Idle => {
                let mut stop = StopRequest::new(intent, deadline);
                if let Some(reply) = reply {
                    if intent.is_shutdown() {
                        stop.shutdown_replies.push(reply);
                    } else {
                        stop.interrupt_replies.push(reply);
                    }
                }
                if intent == StopIntent::Interrupt {
                    self.defer_pending(InputDeferredReason::Interrupted).await?;
                    respond(stop.take_replies(), Ok(()));
                    return Ok(());
                }
                self.start_cleanup(stop);
                if first_shutdown {
                    self.services
                        .tool_sessions
                        .request_stop_session(&self.control.id)
                        .await;
                }
                return Ok(());
            }
            Phase::Executing(work) => {
                work.active.cancel.cancel();
                let stop = StopRequest::new(intent, deadline);
                self.phase = Phase::Stopping {
                    work: RunningWork::Execution(work),
                    stop,
                };
                self.phase.stop_mut().unwrap()
            }
            Phase::Compacting(work) => {
                work.active.cancel.cancel();
                let stop = StopRequest::new(intent, deadline);
                self.phase = Phase::Stopping {
                    work: RunningWork::Compaction(work),
                    stop,
                };
                self.phase.stop_mut().unwrap()
            }
            Phase::Stopping { work, stop } => {
                self.phase = Phase::Stopping { work, stop };
                self.phase.stop_mut().unwrap()
            }
            Phase::Closing { active, stop } => {
                self.phase = Phase::Closing { active, stop };
                self.phase.stop_mut().unwrap()
            }
            Phase::Closed(stop) => {
                self.phase = Phase::Closed(stop);
                self.phase.stop_mut().unwrap()
            }
        };
        stop.add_deadline(deadline);
        if intent.is_shutdown() && !stop.intent.is_shutdown() {
            stop.intent = intent;
        }
        if let Some(reply) = reply {
            if intent.is_shutdown() {
                stop.shutdown_replies.push(reply);
            } else {
                stop.interrupt_replies.push(reply);
            }
        }
        if first_shutdown {
            self.services
                .tool_sessions
                .request_stop_session(&self.control.id)
                .await;
        }
        Ok(())
    }

    fn start_execution(&mut self) {
        let message = self
            .inbox
            .deliverable()
            .into_iter()
            .next()
            .expect("ready input");
        let attempted_input = message.id.clone();
        let executor = self.executor.clone();
        let cancel = self.services.turn_registry.child_token();
        let task_cancel = cancel.clone();
        let turn = Turn {
            id: TurnId::new(),
            user_message: message,
            default_model: None,
            subagent_model: None,
        };
        let turn_id = turn.id.clone();
        let execution_id = turn_id.clone();
        let session_id = self.control.id.clone();
        let forwarded = self.control.forwarded.clone();
        let task = tokio::spawn(async move {
            let mut events = match executor.submit_turn_with_cancel(turn, task_cancel).await {
                Ok(events) => events,
                Err(error) => return Ok(TaskOutcome::NotStarted(error)),
            };
            let mut outcome = None;
            let mut started = false;
            loop {
                let event = match events.try_next().await {
                    Ok(Some(event)) => event,
                    Ok(None) => break,
                    Err(error) if !started => return Ok(TaskOutcome::NotStarted(error)),
                    Err(error) => return Err(error),
                };
                if event.session_id != session_id
                    || matches!(event.payload, SessionEventPayload::Lagged { .. })
                {
                    let _ = forwarded.send(event);
                } else {
                    match &event.payload {
                        SessionEventPayload::TurnStarted { turn_id, .. }
                            if turn_id == &execution_id =>
                        {
                            started = true
                        }
                        SessionEventPayload::TurnFailed { turn_id, .. }
                            if turn_id == &execution_id =>
                        {
                            outcome = Some(TaskOutcome::TurnFailed)
                        }
                        SessionEventPayload::TurnCompleted { turn_id, .. }
                            if turn_id == &execution_id =>
                        {
                            outcome = Some(TaskOutcome::Complete)
                        }
                        _ => {}
                    }
                }
            }
            outcome.ok_or_else(|| anyhow::anyhow!("execution ended without a terminal event"))
        });
        self.transition(Phase::Executing(Execution {
            active: Active { cancel, task },
            turn_id,
            attempted_input,
        }));
    }

    async fn finish_work(&mut self, result: anyhow::Result<TaskOutcome>) -> anyhow::Result<bool> {
        let result = match result {
            Ok(result) => result,
            Err(error) if matches!(self.phase, Phase::Compacting(_)) => {
                let Phase::Compacting(work) = std::mem::replace(&mut self.phase, Phase::Idle)
                else {
                    unreachable!()
                };
                let _ = work.reply.send(Err(SessionError::Operation(error)));
                self.transition(Phase::Idle);
                return Ok(false);
            }
            Err(error)
                if matches!(
                    self.phase,
                    Phase::Stopping {
                        work: RunningWork::Compaction(_),
                        ..
                    }
                ) =>
            {
                let Phase::Stopping {
                    work: RunningWork::Compaction(work),
                    mut stop,
                } = std::mem::replace(&mut self.phase, Phase::Idle)
                else {
                    unreachable!()
                };
                let shared = SharedOperationError(Arc::new(error));
                let _ = work
                    .reply
                    .send(Err(SessionError::Operation(anyhow::Error::new(
                        shared.clone(),
                    ))));
                if stop.intent.is_shutdown() {
                    self.start_cleanup(stop);
                } else {
                    self.transition(Phase::Idle);
                    respond(stop.take_replies(), Err(anyhow::Error::new(shared)));
                }
                return Ok(false);
            }
            Err(error) => return Err(error),
        };
        if let TaskOutcome::NotStarted(error) = &result {
            tracing::warn!(session_id = %self.control.id, %error, "session execution failed before durable start");
            let reason = if error
                .downcast_ref::<halter_protocol::ProviderError>()
                .is_some_and(halter_protocol::ProviderError::is_cancelled)
            {
                InputDeferredReason::Interrupted
            } else {
                InputDeferredReason::ExecutionFailed {
                    error: format!("{error:#}"),
                    retryable: true,
                }
            };
            self.defer_after_failure(reason).await?;
        }
        if matches!(self.phase, Phase::Closing { .. }) {
            self.defer_pending(InputDeferredReason::Shutdown).await?;
            let Phase::Closing { stop, .. } = std::mem::replace(&mut self.phase, Phase::Idle)
            else {
                unreachable!()
            };
            self.phase = Phase::Closed(stop);
            return Ok(true);
        }
        if matches!(self.phase, Phase::Stopping { .. }) {
            let reason = if self.phase.is_closing() {
                InputDeferredReason::Shutdown
            } else {
                InputDeferredReason::Interrupted
            };
            self.defer_pending(reason).await?;
        }
        match std::mem::replace(&mut self.phase, Phase::Idle) {
            Phase::Executing(_) => {
                if self.inbox.ready() {
                    self.start_execution();
                } else {
                    self.transition(Phase::Idle);
                }
            }
            Phase::Compacting(work) => {
                let _ = work.reply.send(Ok(()));
                if self.inbox.ready() {
                    self.start_execution();
                } else {
                    self.transition(Phase::Idle);
                }
            }
            Phase::Stopping { work, mut stop } => {
                if let RunningWork::Compaction(work) = work {
                    let _ = work.reply.send(Err(SessionError::Operation(
                        halter_protocol::ProviderError::cancelled().into(),
                    )));
                }
                if stop.intent.is_shutdown() {
                    self.start_cleanup(stop);
                } else {
                    self.transition(Phase::Idle);
                    respond(stop.take_replies(), Ok(()));
                }
            }
            _ => unreachable!("task completion requires active phase"),
        }
        Ok(false)
    }

    fn start_cleanup(&mut self, stop: StopRequest) {
        let executor = self.executor.clone();
        let sessions = self.services.tool_sessions.clone();
        let subagents = self.subagents.clone();
        let id = self.control.id.clone();
        let deadline = stop.deadline;
        let reason = stop.intent.reason();
        let task = tokio::spawn(async move {
            let children = subagents.close_session_with_deadline(&id, deadline).await;
            let resources = sessions.shutdown_session(&id).await;
            let hooks = executor.shutdown(reason).await;
            children
                .and(resources)
                .and(hooks)
                .map(|()| TaskOutcome::Complete)
        });
        self.transition(Phase::Closing {
            active: Active {
                cancel: CancellationToken::new(),
                task,
            },
            stop,
        });
    }

    fn force_active(&mut self) {
        let phase = std::mem::replace(&mut self.phase, Phase::Idle);
        let sessions = self.services.tool_sessions.clone();
        let subagents = self.subagents.clone();
        let id = self.control.id.clone();
        match phase {
            Phase::Stopping { work, mut stop } => {
                stop.deadline = None;
                let closing = stop.intent.is_shutdown();
                let force_resources = async move {
                    if closing {
                        sessions.force_stop_session(&id).await;
                        subagents.force_close_session(&id).await;
                    }
                };
                let work = match work {
                    RunningWork::Execution(work) => {
                        work.active.cancel.cancel();
                        work.active.task.abort();
                        let executor = self.executor.clone();
                        let recovery_id = work.turn_id.clone();
                        let task = tokio::spawn(async move {
                            let _ = work.active.task.await;
                            force_resources.await;
                            executor
                                .force_interrupt_turn(&recovery_id)
                                .await
                                .map(|()| TaskOutcome::TurnFailed)
                        });
                        RunningWork::Execution(Execution {
                            active: Active {
                                cancel: work.active.cancel,
                                task,
                            },
                            turn_id: work.turn_id,
                            attempted_input: work.attempted_input,
                        })
                    }
                    RunningWork::Compaction(work) => {
                        work.active.cancel.cancel();
                        work.active.task.abort();
                        let task = tokio::spawn(async move {
                            let _ = work.active.task.await;
                            force_resources.await;
                            Ok(TaskOutcome::Complete)
                        });
                        RunningWork::Compaction(Compaction {
                            active: Active {
                                cancel: work.active.cancel,
                                task,
                            },
                            reply: work.reply,
                        })
                    }
                };
                self.phase = Phase::Stopping { work, stop };
            }
            Phase::Closing { active, mut stop } => {
                stop.deadline = None;
                let task = tokio::spawn(async move {
                    sessions.force_stop_session(&id).await;
                    subagents.force_close_session(&id).await;
                    active.task.await.map_err(anyhow::Error::from)?
                });
                self.phase = Phase::Closing {
                    active: Active {
                        cancel: active.cancel,
                        task,
                    },
                    stop,
                };
            }
            phase => self.phase = phase,
        }
    }

    async fn recover_failed_driver(&mut self) {
        let phase = std::mem::replace(&mut self.phase, Phase::Idle);
        let (work, cleanup, stop) = match phase {
            Phase::Executing(work) => (
                Some(RunningWork::Execution(work)),
                None,
                StopRequest::new(StopIntent::Shutdown, None),
            ),
            Phase::Compacting(work) => (
                Some(RunningWork::Compaction(work)),
                None,
                StopRequest::new(StopIntent::Shutdown, None),
            ),
            Phase::Stopping { work, mut stop } => {
                stop.intent = StopIntent::Shutdown;
                (Some(work), None, stop)
            }
            Phase::Closing { active, stop } => (None, Some(active), stop),
            Phase::Closed(stop) => {
                self.phase = Phase::Closed(stop);
                return;
            }
            Phase::Idle => (None, None, StopRequest::new(StopIntent::Shutdown, None)),
        };
        self.services
            .tool_sessions
            .force_stop_session(&self.control.id)
            .await;
        self.subagents.force_close_session(&self.control.id).await;
        if let Some(work) = work {
            match work {
                RunningWork::Execution(work) => {
                    work.active.cancel.cancel();
                    work.active.task.abort();
                    // Fatal finalization can arrive from a task the actor
                    // already joined. A completed JoinHandle cannot be polled
                    // twice; only unfinished work still needs settlement.
                    if !work.active.task.is_finished() {
                        let _ = work.active.task.await;
                    }
                    if let Ok(executor) =
                        SessionExecutor::new(self.services.clone(), self.control.id.clone())
                    {
                        let _ = executor.force_interrupt_turn(&work.turn_id).await;
                    }
                }
                RunningWork::Compaction(work) => {
                    work.active.cancel.cancel();
                    work.active.task.abort();
                    if !work.active.task.is_finished() {
                        let _ = work.active.task.await;
                    }
                    let _ = work.reply.send(Err(SessionError::Closed));
                }
            }
        }
        if let Some(cleanup) = cleanup {
            // Its result was already awaited by the actor. Do not poll a
            // completed JoinHandle again or rerun lifecycle hooks.
            drop(cleanup);
        } else {
            let _ = self
                .subagents
                .close_session_with_deadline(&self.control.id, stop.deadline)
                .await;
            let _ = self
                .services
                .tool_sessions
                .shutdown_session(&self.control.id)
                .await;
            let _ = self.executor.shutdown("session_failed").await;
        }
        // Fatal errors route directly to the store. Repair remaining inbox
        // metadata against its current head, never the failed actor's cursor.
        if let Ok(Some(mut stored)) = self.store.load_session(&self.control.id).await
            && hydrate_stored_session(self.store.as_ref(), &mut stored)
                .await
                .is_ok()
        {
            self.head = stored.head_sequence;
            let events = stored
                .state
                .pending_inputs
                .iter()
                .filter(|message| !self.inbox.deferred.contains_key(&message.id))
                .map(|message| {
                    PendingEvent::new(
                        self.control.id.clone(),
                        halter_protocol::Delivery::Lossless,
                        SessionEventPayload::InputDeferred {
                            message_id: message.id.clone(),
                            reason: InputDeferredReason::Shutdown,
                        },
                    )
                })
                .collect::<Vec<_>>();
            if !events.is_empty()
                && let Ok(committed) = self
                    .store
                    .commit(
                        &self.control.id,
                        None,
                        Some(stored.head_sequence),
                        None,
                        events,
                    )
                    .await
            {
                for event in committed {
                    self.head = event.sequence();
                    self.services.event_bus.publish(event);
                }
            }
        }
        self.phase = Phase::Closed(stop);
    }

    async fn defer_pending(&mut self, reason: InputDeferredReason) -> anyhow::Result<()> {
        let ids = self
            .inbox
            .pending
            .iter()
            .filter(|m| !self.inbox.deferred.contains_key(&m.id))
            .map(|m| m.id.clone())
            .collect::<Vec<_>>();
        for message_id in ids {
            self.commit_payload(SessionEventPayload::InputDeferred {
                message_id,
                reason: reason.clone(),
            })
            .await?;
        }
        Ok(())
    }

    async fn defer_after_failure(&mut self, reason: InputDeferredReason) -> anyhow::Result<()> {
        let attempted = self.phase.attempted_input().cloned();
        let ids = self
            .inbox
            .pending
            .iter()
            .filter(|m| {
                !self.inbox.deferred.contains_key(&m.id) || Some(&m.id) == attempted.as_ref()
            })
            .map(|m| m.id.clone())
            .collect::<Vec<_>>();
        for message_id in ids {
            let reason = if Some(&message_id) == attempted.as_ref() {
                reason.clone()
            } else {
                InputDeferredReason::ExecutionStopped
            };
            self.commit_payload(SessionEventPayload::InputDeferred { message_id, reason })
                .await?;
        }
        Ok(())
    }

    async fn commit_payload(&mut self, payload: SessionEventPayload) -> anyhow::Result<u64> {
        debug_assert!(matches!(
            payload,
            SessionEventPayload::InputAccepted { .. }
                | SessionEventPayload::InputRejected { .. }
                | SessionEventPayload::InputDeferred { .. }
        ));
        let event = PendingEvent::new(
            self.control.id.clone(),
            halter_protocol::Delivery::Lossless,
            payload,
        );
        let committed = self.commit(None, None, vec![event]).await?;
        let sequence = committed.last().map_or(self.head, SessionEvent::sequence);
        for event in committed {
            self.services.event_bus.publish(event.clone());
            if let Some(recorder) = &self.services.trace_recorder {
                recorder.record(&event);
            }
        }
        Ok(sequence)
    }

    async fn commit(
        &mut self,
        snapshot: Option<Arc<ResourceSnapshot>>,
        mut state: Option<SessionState>,
        events: Vec<PendingEvent>,
    ) -> anyhow::Result<Vec<SessionEvent>> {
        let mut projection = SessionState {
            pending_inputs: self.inbox.pending.clone(),
            ..SessionState::default()
        };
        let mut admitted = self.inbox.accepted.clone();
        let mut deferred = self.inbox.deferred.clone();
        let attempted = self.phase.attempted_input().cloned();
        let mut expanded = Vec::with_capacity(events.len());
        for event in events {
            let delivered = match &event.payload {
                SessionEventPayload::MessageItem {
                    message: Message::User(message),
                } if projection
                    .pending_inputs
                    .iter()
                    .any(|pending| pending.id == message.id) =>
                {
                    Some(message.id.clone())
                }
                _ => None,
            };
            if let SessionEventPayload::InputAccepted { message } = &event.payload {
                deferred.remove(&message.id);
            }
            if let SessionEventPayload::InputDeferred { message_id, reason } = &event.payload {
                deferred.insert(message_id.clone(), reason.clone());
            }
            halter_protocol::fold::apply_event(&mut projection, &event.payload);
            let failure = match &event.payload {
                SessionEventPayload::TurnFailed {
                    cancelled,
                    error,
                    retryable,
                    ..
                } => Some(if *cancelled {
                    if self.phase.is_closing() {
                        InputDeferredReason::Shutdown
                    } else {
                        InputDeferredReason::Interrupted
                    }
                } else {
                    InputDeferredReason::ExecutionFailed {
                        error: error.clone(),
                        retryable: *retryable,
                    }
                }),
                _ => None,
            };
            expanded.push(event);
            if let Some(message_id) = delivered {
                expanded.push(PendingEvent::new(
                    self.control.id.clone(),
                    halter_protocol::Delivery::Lossless,
                    SessionEventPayload::InputDelivered { message_id },
                ));
            }
            if let Some(reason) = failure {
                for message in &projection.pending_inputs {
                    if deferred.contains_key(&message.id)
                        && !(Some(&message.id) == attempted.as_ref()
                            && matches!(reason, InputDeferredReason::ExecutionFailed { .. }))
                    {
                        continue;
                    }
                    let reason = if matches!(reason, InputDeferredReason::ExecutionFailed { .. })
                        && Some(&message.id) != attempted.as_ref()
                    {
                        InputDeferredReason::ExecutionStopped
                    } else {
                        reason.clone()
                    };
                    deferred.insert(message.id.clone(), reason.clone());
                    expanded.push(PendingEvent::new(
                        self.control.id.clone(),
                        halter_protocol::Delivery::Lossless,
                        SessionEventPayload::InputDeferred {
                            message_id: message.id.clone(),
                            reason,
                        },
                    ));
                }
            }
        }
        if let Some(state) = &mut state {
            state.pending_inputs = projection.pending_inputs.clone();
        }
        let committed = self
            .store
            .commit(&self.control.id, snapshot, Some(self.head), state, expanded)
            .await?;
        self.head = committed.last().map_or(self.head, SessionEvent::sequence);
        for event in &committed {
            if let SessionEventPayload::InputAccepted { message } = &event.payload {
                admitted.insert(message.id.clone(), event.sequence());
            }
        }
        self.inbox.pending = projection.pending_inputs;
        deferred.retain(|id, _| self.inbox.pending.iter().any(|m| &m.id == id));
        self.inbox.accepted = admitted;
        self.inbox.deferred = deferred;
        Ok(committed)
    }
}

async fn wait_deadline(deadline: Option<Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline).await,
        None => std::future::pending().await,
    }
}

fn timeout_deadline(timeout: Option<Duration>) -> Result<Option<Instant>, SessionError> {
    timeout
        .map(|duration| {
            Instant::now()
                .checked_add(duration)
                .ok_or_else(|| SessionError::Operation(anyhow::anyhow!("invalid session timeout")))
        })
        .transpose()
}

#[derive(Debug, Clone)]
struct SharedOperationError(Arc<anyhow::Error>);

impl std::fmt::Display for SharedOperationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:#}", self.0)
    }
}

impl std::error::Error for SharedOperationError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.0.as_ref().as_ref())
    }
}

fn respond(replies: Vec<Reply<()>>, result: anyhow::Result<()>) {
    let error = result
        .err()
        .map(|error| SharedOperationError(Arc::new(error)));
    for reply in replies {
        let response = match &error {
            Some(error) => Err(SessionError::Operation(anyhow::Error::new(error.clone()))),
            None => Ok(()),
        };
        let _ = reply.send(response);
    }
}

#[cfg(test)]
#[path = "session_event_cursor_tests.rs"]
mod cursor_tests;

struct EventCursor {
    control: Arc<DriverControl>,
    receiver: broadcast::Receiver<SessionEvent>,
    forwarded: broadcast::Receiver<SessionEvent>,
    sequence: u64,
    buffered: std::collections::VecDeque<SessionEvent>,
    replay_needed: bool,
    closed_reported: bool,
}

fn session_events(control: Arc<DriverControl>, sequence: u64) -> SessionEventStream {
    let receiver = control.services.event_bus.subscribe_raw();
    let forwarded = control.forwarded.subscribe();
    stream::try_unfold(EventCursor { control, receiver, forwarded, sequence, buffered: Default::default(), replay_needed: true, closed_reported: false }, |mut cursor| async move {
        loop {
            if cursor.closed_reported { return Ok(None); }
            if let Some(event) = cursor.buffered.pop_front() {
                if event.sequence() > cursor.control.final_head.load(Ordering::Acquire) { continue; }
                cursor.sequence = event.sequence();
                return Ok(Some((event, cursor)));
            }
            if cursor.replay_needed || cursor.control.closed.is_cancelled() {
                let replay = cursor.control.services.sessions.replay_after(&cursor.control.id, cursor.sequence).await?;
                // Read the fence after awaiting replay: closure and reopen
                // may both have happened while the store was being queried.
                let final_head = cursor.control.final_head.load(Ordering::Acquire);
                cursor.buffered.extend(replay.into_iter().filter(|event| event.sequence() <= final_head));
                cursor.replay_needed = false;
            }
            if !cursor.buffered.is_empty() { continue; }
            match cursor.forwarded.try_recv() {
                Ok(event) => return Ok(Some((event, cursor))),
                Err(broadcast::error::TryRecvError::Lagged(dropped)) => {
                    return Ok(Some((crate::event_bus::lagged_event(dropped), cursor)));
                },
                Err(broadcast::error::TryRecvError::Empty | broadcast::error::TryRecvError::Closed) => {},
            }
            if cursor.control.closed.is_cancelled() {
                if let Some(error) = cursor.control.failure.lock().unwrap_or_else(|p| p.into_inner()).as_ref() {
                    return Err(anyhow::Error::new(error.clone()));
                }
                if !cursor.closed_reported {
                    cursor.closed_reported = true;
                    // Like Lagged, this is a stream-local notification with
                    // no durable sequence. Emit after the final replay even
                    // if this reader was paused throughout session cleanup.
                    let event = PendingEvent::new(
                        cursor.control.id.clone(),
                        halter_protocol::Delivery::BestEffort,
                        SessionEventPayload::SessionStatusChanged { status: SessionStatus::Closed },
                    ).into_committed(0);
                    return Ok(Some((event, cursor)));
                }
                return Ok(None);
            }
            tokio::select! {
                _ = cursor.control.closed.cancelled() => { cursor.replay_needed = true; },
                event = cursor.receiver.recv() => {
                    match event {
                        Ok(event) if event.session_id == cursor.control.id && event.sequence() > cursor.sequence => {
                            cursor.replay_needed = true;
                        },
                        // A gap can include this session even if retained events do not.
                        Err(broadcast::error::RecvError::Lagged(_)) => { cursor.replay_needed = true; },
                        Err(broadcast::error::RecvError::Closed) => return Ok(None),
                        Ok(_) => {},
                    }
                },
                event = cursor.forwarded.recv() => {
                    match event {
                        Ok(event) => return Ok(Some((event, cursor))),
                        Err(broadcast::error::RecvError::Lagged(dropped)) => {
                            return Ok(Some((crate::event_bus::lagged_event(dropped), cursor)));
                        },
                        Err(broadcast::error::RecvError::Closed) => {},
                    }
                },
            }
        }
    }).boxed()
}

#[cfg(test)]
mod commit_tests {
    use super::*;

    #[tokio::test]
    async fn shared_stop_failures_retain_the_original_typed_cause() {
        let id = SessionId::new();
        let (first, first_response) = oneshot::channel();
        let (second, second_response) = oneshot::channel();
        respond(
            vec![first, second],
            Err(halter_session::SessionCommitConflict {
                session_id: id.clone(),
                expected_head_sequence: 5,
                actual_head_sequence: 7,
            }
            .into()),
        );
        for response in [first_response, second_response] {
            let error = response.await.unwrap().unwrap_err();
            let original = error
                .operation_error()
                .unwrap()
                .downcast_ref::<halter_session::SessionCommitConflict>()
                .unwrap();
            assert_eq!(original.session_id, id);
            assert_eq!(original.actual_head_sequence, 7);
        }
    }

    #[tokio::test]
    async fn timed_out_stop_is_still_admitted_when_the_mailbox_was_full() {
        for shutdown in [false, true] {
            let (tx, mut rx) = mpsc::channel(1);
            let (stop_tx, mut stop_rx) = mpsc::unbounded_channel();
            let (reply, _response) = oneshot::channel();
            tx.send(Command::Pending(reply)).await.unwrap();
            let control = Arc::new(DriverControl {
                id: SessionId::new(),
                tx,
                stop_tx,
                closed: CancellationToken::new(),
                final_head: AtomicU64::new(u64::MAX),
                services: crate::session_driver_tests::services(Arc::new(
                    halter_providers::FakeProvider::default(),
                )),
                forwarded: broadcast::channel(1).0,
                status: watch::channel(SessionStatus::Idle).0,
                failure: Mutex::new(None),
            });
            let token = Arc::new(HandleToken {
                stop_tx: control.stop_tx.clone(),
            });
            let handle = SessionHandle {
                control,
                _token: token,
            };
            let result = if shutdown {
                handle.shutdown(Some(Duration::ZERO)).await
            } else {
                handle.interrupt(Some(Duration::ZERO)).await
            };
            assert!(matches!(result, Err(SessionError::TimedOut)));
            assert!(matches!(rx.recv().await, Some(Command::Pending(_))));
            let stop = stop_rx
                .try_recv()
                .expect("stop admitted independently of caller timeout");
            assert!(matches!(
                (shutdown, stop),
                (
                    true,
                    Command::Shutdown {
                        deadline: Some(_),
                        ..
                    }
                ) | (
                    false,
                    Command::Interrupt {
                        deadline: Some(_),
                        ..
                    }
                )
            ));
        }
    }

    #[tokio::test]
    async fn resume_preserves_delivery_without_inventing_request_completion() {
        let services = crate::session_driver_tests::services(Arc::new(
            halter_providers::FakeProvider::default(),
        ));
        let runtime = crate::SessionRuntime::new(services);
        let executor = runtime
            .new_session(crate::SessionInit::default())
            .await
            .unwrap();
        let store = executor.services().sessions.clone();
        let id = executor.session_id();
        let stored = store.load_session(id).await.unwrap().unwrap();
        let Message::User(input) = Message::user("delivered before crash") else {
            unreachable!()
        };
        let turn_id = TurnId::new();
        let payloads = [
            SessionEventPayload::InputAccepted {
                message: input.clone(),
            },
            SessionEventPayload::TurnStarted {
                turn_id: turn_id.clone(),
                default_model: None,
                subagent_model: None,
            },
            SessionEventPayload::MessageItem {
                message: Message::User(input.clone()),
            },
            SessionEventPayload::InputDelivered {
                message_id: input.id.clone(),
            },
        ];
        let mut state = stored.state;
        for payload in &payloads {
            halter_protocol::fold::apply_event(&mut state, payload);
        }
        state.open_turn = Some(turn_id);
        store
            .commit(
                id,
                None,
                Some(stored.head_sequence),
                Some(state),
                payloads
                    .into_iter()
                    .map(|payload| {
                        PendingEvent::new(id.clone(), halter_protocol::Delivery::Lossless, payload)
                    })
                    .collect(),
            )
            .await
            .unwrap();
        let (session, _events) = runtime.resume_session(id).await.unwrap();
        let log = session.replay().await.unwrap();
        assert_eq!(
            log.iter()
                .filter(|event| matches!(&event.payload,
            SessionEventPayload::InputDelivered { message_id } if message_id == &input.id))
                .count(),
            1
        );
        session.shutdown(None).await.unwrap();
    }

    #[tokio::test]
    async fn checkpoints_rebase_across_inbox_metadata_but_reject_stale_transcripts() {
        let services = crate::session_driver_tests::services(Arc::new(
            halter_providers::FakeProvider::default(),
        ));
        let runtime = crate::SessionRuntime::new(services);
        let executor = runtime
            .new_session(crate::SessionInit::default())
            .await
            .unwrap();
        let services = executor.services().clone();
        let id = executor.session_id().clone();
        let stored = services.sessions.load_session(&id).await.unwrap().unwrap();
        let base = stored.head_sequence;
        let (tx, rx) = mpsc::channel(INBOX_CAPACITY);
        let (stop_tx, stop_rx) = mpsc::unbounded_channel();
        let control = Arc::new(DriverControl {
            id: id.clone(),
            tx,
            stop_tx,
            closed: CancellationToken::new(),
            final_head: AtomicU64::new(u64::MAX),
            services: services.clone(),
            forwarded: broadcast::channel(FORWARDED_EVENT_CAPACITY).0,
            status: watch::channel(SessionStatus::Idle).0,
            failure: Mutex::new(None),
        });
        let mut driver = Driver {
            executor,
            services: services.clone(),
            store: services.sessions.clone(),
            subagents: RuntimeSubagentControl::new(services.clone()),
            control,
            handles: Weak::new(),
            rx,
            stop_rx,
            head: base,
            last_state_commit: base,
            inbox: Inbox::default(),
            phase: Phase::Idle,
        };
        let (reply, response) = oneshot::channel();
        driver
            .command(Command::Commit {
                snapshot: None,
                expected: Some(base),
                state: Some(Box::new(stored.state.clone())),
                events: Vec::new(),
                reply,
            })
            .await
            .unwrap();
        assert!(
            response
                .await
                .unwrap()
                .unwrap_err()
                .to_string()
                .contains("checkpoint requires an event")
        );
        let Message::User(input) = Message::user("queued") else {
            unreachable!()
        };
        driver
            .commit_payload(SessionEventPayload::InputAccepted {
                message: input.clone(),
            })
            .await
            .unwrap();
        let mut newer = stored.state.clone();
        newer.messages.push(Message::user("new transcript"));
        let (reply, response) = oneshot::channel();
        driver
            .command(Command::Commit {
                snapshot: None,
                expected: Some(base),
                state: Some(Box::new(newer)),
                events: vec![PendingEvent::new(
                    id.clone(),
                    halter_protocol::Delivery::Lossless,
                    SessionEventPayload::Warning {
                        message: "checkpoint".into(),
                    },
                )],
                reply,
            })
            .await
            .unwrap();
        response.await.unwrap().unwrap();
        let (reply, response) = oneshot::channel();
        driver
            .command(Command::Commit {
                snapshot: None,
                expected: Some(base),
                state: Some(Box::new(stored.state)),
                events: vec![PendingEvent::new(
                    id.clone(),
                    halter_protocol::Delivery::Lossless,
                    SessionEventPayload::Warning {
                        message: "stale checkpoint".into(),
                    },
                )],
                reply,
            })
            .await
            .unwrap();
        let error = response.await.unwrap().unwrap_err();
        assert!(
            error
                .downcast_ref::<halter_session::SessionCommitConflict>()
                .is_some()
        );
        let final_state = services
            .sessions
            .load_session(&id)
            .await
            .unwrap()
            .unwrap()
            .state;
        assert_eq!(final_state.pending_inputs, vec![input]);
        assert!(final_state.messages.iter().any(|message| matches!(message,
            Message::User(message) if message.plain_text() == "new transcript")));
    }
}
