//! A session owns admission, its durable inbox, and its live execution.
// pattern: Imperative Shell

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, Weak};
use std::time::Duration;

use async_trait::async_trait;
use futures::{FutureExt, StreamExt, TryStreamExt, stream};
use halter_protocol::{
    InputDeferredReason, InputOutcome, Message, MessageId, PendingEvent, ResourceSnapshot,
    SessionBlueprint, SessionEvent, SessionEventPayload, SessionId, SessionState, SessionStatus,
    Turn, TurnId, UserMessage,
};
use halter_session::{SessionStore, StoredSession};
use thiserror::Error;
use tokio::sync::{broadcast, mpsc, oneshot};
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

/// Durable admission receipt. Ignore events below `sequence` when waiting for
/// this attempt's outcome. Repeating an already-settled ID returns its previous
/// receipt; its outcome remains available through replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Submission {
    pub message_id: MessageId,
    pub sequence: u64,
}

/// A cloneable connection to one live incarnation of a stored session.
/// Dropping a handle or event stream does not stop execution.
#[derive(Clone)]
pub struct SessionHandle {
    control: Arc<DriverControl>,
}

struct DriverControl {
    id: SessionId,
    tx: mpsc::Sender<Command>,
    stop_tx: mpsc::UnboundedSender<Command>,
    closed: CancellationToken,
    services: Arc<RuntimeServices>,
    forwarded: broadcast::Sender<SessionEvent>,
    failure: std::sync::Mutex<Option<String>>,
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
        let deadline = timeout
            .map(|duration| {
                Instant::now().checked_add(duration).ok_or_else(|| {
                    SessionError::Operation(anyhow::anyhow!("invalid session timeout"))
                })
            })
            .transpose()?;
        let response = self.enqueue_stop(|reply| Command::Interrupt { deadline, reply })?;
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
        let deadline = timeout
            .map(|duration| {
                Instant::now().checked_add(duration).ok_or_else(|| {
                    SessionError::Operation(anyhow::anyhow!("invalid session timeout"))
                })
            })
            .transpose()?;
        let response = if self.control.closed.is_cancelled() {
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
                Err(SessionError::Closed) => self.control.closed.cancelled().await,
                result => return result,
            }
        } else if !self.control.closed.is_cancelled() {
            self.control.closed.cancelled().await;
        }
        match self
            .control
            .failure
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .as_ref()
        {
            Some(error) => Err(SessionError::Operation(anyhow::anyhow!("{error}"))),
            None => Ok(()),
        }
    }

    fn enqueue_stop(
        &self,
        command: impl FnOnce(Reply<()>) -> Command,
    ) -> Result<oneshot::Receiver<Result<(), SessionError>>, SessionError> {
        if self.control.closed.is_cancelled() {
            return Err(SessionError::Closed);
        }
        let (reply, response) = oneshot::channel();
        let command = command(reply);
        // Stop admission must survive a timed-out caller and cannot wait for
        // ordinary inbox capacity. Its separate lane also preserves call order.
        self.control
            .stop_tx
            .send(command)
            .map_err(|_| SessionError::Closed)?;
        Ok(response)
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
        let handles = self
            .lock_entries()
            .values()
            .filter_map(|entry| match entry {
                DriverSlot::Open(driver) | DriverSlot::Failed(driver) => {
                    driver.upgrade().map(|control| SessionHandle { control })
                }
                DriverSlot::Opening(_) => None,
            })
            .collect::<Vec<_>>();
        futures::future::join_all(handles.into_iter().map(|handle| async move {
            if let Err(error) = handle.shutdown(deadline.map(|d| d.saturating_duration_since(Instant::now()))).await {
                tracing::warn!(session_id = %handle.id(), %error, "session cleanup failed during runtime shutdown");
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
            services: services.clone(),
            forwarded: broadcast::channel(FORWARDED_EVENT_CAPACITY).0,
            failure: std::sync::Mutex::new(None),
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
            rx,
            stop_rx,
            head: stored.head_sequence,
            last_state_commit: stored.head_sequence,
            pending: stored.state.pending_inputs,
            status: SessionStatus::Idle,
            active: None,
            wake_requested: false,
            closing: false,
            interrupt_waiters: Vec::new(),
            shutdown_waiters: Vec::new(),
            cancel_deadline: None,
            shutdown_deadline: None,
            accepted: HashMap::new(),
            delivered: HashSet::new(),
            deferred: HashSet::new(),
        };
        let mut delivered = HashSet::new();
        let mut unsettled = HashMap::new();
        let history = match history {
            Some(history) => history,
            None => services.sessions.replay(&id).await?,
        };
        for event in history {
            let sequence = event.sequence();
            if let Some(outcome) = input_outcome(&event.payload) {
                for message_id in delivered.drain() {
                    unsettled.insert(message_id, outcome.clone());
                }
            }
            match event.payload {
                SessionEventPayload::InputAccepted { message } => {
                    driver.accepted.insert(message.id, sequence);
                }
                SessionEventPayload::MessageItem {
                    message: Message::User(message),
                } if driver.accepted.contains_key(&message.id) => {
                    delivered.insert(message.id);
                }
                SessionEventPayload::InputSettled { message_id, .. } => {
                    unsettled.remove(&message_id);
                }
                _ => {}
            }
        }
        // Resume may have recovered a terminal event before the driver existed.
        // Complete its input correlation from the same durable history.
        let mut recovered = unsettled.into_iter().collect::<Vec<_>>();
        recovered.sort_by(|(a, _), (b, _)| a.0.cmp(&b.0));
        for (message_id, outcome) in recovered {
            driver
                .commit_payload(SessionEventPayload::InputSettled {
                    message_id,
                    outcome,
                })
                .await?;
        }
        driver.set_status(SessionStatus::Idle).await?;
        driver.defer_pending(InputDeferredReason::Resumed).await?;
        {
            let mut entries = self.lock_entries();
            // Shutdown cancels the runtime before collecting drivers. Under
            // this lock an opener either joins that collection or is refused.
            if services.turn_registry.is_shutting_down() {
                return Err(SessionError::Closed);
            }
            entries.insert(id.clone(), DriverSlot::Open(Arc::downgrade(&control)));
        }
        let registry = self.clone();
        let task_control = control.clone();
        tokio::spawn(async move {
            let result = std::panic::AssertUnwindSafe(driver.run())
                .catch_unwind()
                .await
                .unwrap_or_else(|_| Err(anyhow::anyhow!("session driver panicked")));
            if let Err(error) = &result {
                *task_control
                    .failure
                    .lock()
                    .unwrap_or_else(|p| p.into_inner()) = Some(format!("{error:#}"));
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
                // Signal resources before executor joins or recovery writes,
                // which may still be waiting on uncooperative plugin code.
                driver.services.tool_sessions.force_stop_session(&id).await;
                driver.subagents.force_close_session(&id).await;
                if let Some(active) = driver.active.take() {
                    active.cancel.cancel();
                    active.task.abort();
                    let _ = active.task.await;
                    if let Some(turn_id) = &active.turn_id {
                        // The actor cannot serve more commits after failure.
                        // The Failed slot routes recovery directly to storage.
                        if let Ok(executor) =
                            SessionExecutor::new(driver.services.clone(), id.clone())
                        {
                            let _ = executor.force_interrupt_turn(turn_id).await;
                        }
                    }
                }
                let _ = driver
                    .subagents
                    .close_session_with_deadline(&id, driver.shutdown_deadline)
                    .await;
                let _ = driver.services.tool_sessions.shutdown_session(&id).await;
                let _ = driver.executor.shutdown("session_failed").await;
            }
            registry.release(&id);
            task_control.closed.cancel();
            for reply in driver
                .shutdown_waiters
                .drain(..)
                .chain(driver.interrupt_waiters.drain(..))
            {
                let response = match &result {
                    Ok(()) => Ok(()),
                    Err(error) => Err(SessionError::Operation(anyhow::anyhow!("{error:#}"))),
                };
                let _ = reply.send(response);
            }
        });
        Ok((SessionHandle { control }, events))
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
    Submit(UserMessage, Reply<Submission>),
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
    kind: Work,
    turn_id: Option<TurnId>,
}

enum TaskOutcome {
    Complete,
    TurnFailed,
}

enum Work {
    Execution,
    Compact(Reply<()>),
    Cleanup,
}

struct Driver {
    executor: SessionExecutor,
    services: Arc<RuntimeServices>,
    store: Arc<dyn SessionStore>,
    subagents: RuntimeSubagentControl,
    control: Arc<DriverControl>,
    rx: mpsc::Receiver<Command>,
    stop_rx: mpsc::UnboundedReceiver<Command>,
    head: u64,
    // Routed checkpoints may cross admission/status commits, whose only state
    // changes are overlaid below. They must never cross another state writer.
    last_state_commit: u64,
    pending: Vec<UserMessage>,
    accepted: HashMap<MessageId, u64>,
    delivered: HashSet<MessageId>,
    deferred: HashSet<MessageId>,
    status: SessionStatus,
    active: Option<Active>,
    wake_requested: bool,
    closing: bool,
    interrupt_waiters: Vec<Reply<()>>,
    shutdown_waiters: Vec<Reply<()>>,
    cancel_deadline: Option<Instant>,
    shutdown_deadline: Option<Instant>,
}

impl Driver {
    async fn run(&mut self) -> anyhow::Result<()> {
        let runtime_cancel = self.services.turn_registry.child_token();
        loop {
            if self.active.is_none() {
                if self.closing {
                    self.start_cleanup();
                } else if self.wake_requested && !self.pending.is_empty() {
                    self.start_execution().await?;
                }
            }
            tokio::select! {
                biased;
                _ = async { tokio::time::sleep_until(self.cancel_deadline.unwrap()).await }, if self.cancel_deadline.is_some() => {
                    self.cancel_deadline = None;
                    self.force_active();
                }
                command = self.stop_rx.recv() => {
                    if let Some(command) = command { self.command(command).await?; }
                    else { self.closing = true; }
                }
                _ = runtime_cancel.cancelled(), if !self.closing => {
                    self.closing = true;
                    self.wake_requested = false;
                    if let Some(active) = &self.active { active.cancel.cancel(); }
                }
                command = self.rx.recv() => {
                    if let Some(command) = command { self.command(command).await?; }
                    else { self.closing = true; }
                }
                outcome = async { (&mut self.active.as_mut().unwrap().task).await }, if self.active.is_some() => {
                    let active = self.active.take().unwrap();
                    let result = outcome.map_err(anyhow::Error::from).and_then(|result| result);
                    match active.kind {
                        Work::Cleanup => {
                            let status = self.set_status(SessionStatus::Closed).await;
                            return result.map(|_| ()).and(status);
                        }
                        Work::Compact(reply) => {
                            if let Err(error) = self.set_status(SessionStatus::Idle).await {
                                let _ = reply.send(Err(SessionError::Operation(anyhow::anyhow!("{error:#}"))));
                                return Err(error);
                            }
                            let response = match &result {
                                Ok(_) => Ok(()),
                                Err(error) => Err(SessionError::Operation(anyhow::anyhow!("{error:#}"))),
                            };
                            let _ = reply.send(response);
                        }
                        Work::Execution => {
                            if matches!(result, Ok(TaskOutcome::Complete)) && !active.cancel.is_cancelled() && !self.closing && !self.pending.is_empty() {
                                self.wake_requested = true;
                            }
                            if let Err(error) = &result {
                                if !error.downcast_ref::<halter_protocol::ProviderError>().is_some_and(halter_protocol::ProviderError::is_cancelled) {
                                    // Ordinary execution failures are durable
                                    // TurnFailed events. A stream error means
                                    // finalization failed; close this incarnation.
                                    return Err(anyhow::anyhow!("{error:#}"));
                                }
                                tracing::warn!(session_id = %self.control.id, %error, "session execution failed");
                            }
                        }
                    }
                    if self.status != SessionStatus::Idle && (!self.wake_requested || self.pending.is_empty() || !self.interrupt_waiters.is_empty()) {
                        self.set_status(SessionStatus::Idle).await?;
                    }
                    if active.cancel.is_cancelled() {
                        self.defer_pending(if self.closing { InputDeferredReason::Shutdown } else { InputDeferredReason::Interrupted }).await?;
                    }
                    for reply in self.interrupt_waiters.drain(..) {
                        let response = match &result {
                            Err(error) if !error.downcast_ref::<halter_protocol::ProviderError>().is_some_and(halter_protocol::ProviderError::is_cancelled) => {
                                Err(SessionError::Operation(anyhow::anyhow!("{error:#}")))
                            }
                            _ => Ok(()),
                        };
                        let _ = reply.send(response);
                    }
                    if !self.closing { self.cancel_deadline = None; }
                }
            }
        }
    }

    async fn command(&mut self, command: Command) -> anyhow::Result<()> {
        match command {
            Command::Commit {
                snapshot,
                expected,
                state,
                events,
                reply,
            } => {
                let result = if (state.is_some() || snapshot.is_some()) && events.is_empty() {
                    // Head-based concurrency cannot distinguish two state-only
                    // replacements at one sequence. Executor mutations always
                    // carry an event; enforce that contract at the arbiter.
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
                let stopped = self.closing
                    || self
                        .active
                        .as_ref()
                        .is_some_and(|active| active.cancel.is_cancelled());
                let _ = reply.send(if stopped {
                    Vec::new()
                } else {
                    self.pending.clone()
                });
            }
            Command::Submit(message, reply) => {
                if self.closing {
                    let _ = reply.send(Err(SessionError::Closed));
                } else if let Some(sequence) = self.accepted.get(&message.id).copied() {
                    if let Some(pending) = self
                        .pending
                        .iter()
                        .find(|pending| pending.id == message.id)
                        .cloned()
                    {
                        let id = pending.id.clone();
                        let result = self
                            .commit_payload(SessionEventPayload::InputAccepted { message: pending })
                            .await;
                        if let Ok(sequence) = result {
                            self.deferred.remove(&id);
                            self.accepted.insert(id.clone(), sequence);
                            self.wake_requested = true;
                        }
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
                } else if self.pending.len() >= INBOX_CAPACITY {
                    let _ = reply.send(Err(SessionError::InboxFull));
                } else {
                    let id = message.id.clone();
                    let result = self
                        .commit_payload(SessionEventPayload::InputAccepted { message })
                        .await;
                    if let Ok(sequence) = result {
                        self.accepted.insert(id.clone(), sequence);
                        self.wake_requested = true;
                    }
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
            Command::Interrupt { deadline, reply } => {
                self.set_cancel_deadline(deadline);
                self.wake_requested = false;
                if self.closing {
                    let _ = reply.send(Err(SessionError::Closed));
                } else if let Some(active) = &self.active {
                    active.cancel.cancel();
                    self.interrupt_waiters.push(reply);
                } else {
                    self.cancel_deadline = None;
                    self.defer_pending(InputDeferredReason::Interrupted).await?;
                    let _ = reply.send(Ok(()));
                }
            }
            Command::Shutdown { deadline, reply } => {
                self.set_cancel_deadline(deadline);
                if let Some(deadline) = deadline {
                    self.shutdown_deadline = Some(
                        self.shutdown_deadline
                            .map_or(deadline, |existing| existing.min(deadline)),
                    );
                }
                self.closing = true;
                self.wake_requested = false;
                self.services
                    .tool_sessions
                    .request_stop_session(&self.control.id)
                    .await;
                if let Some(active) = &self.active {
                    active.cancel.cancel();
                }
                self.shutdown_waiters.push(reply);
            }
            Command::Compact {
                reason,
                instructions,
                reply,
            } => {
                if self.closing {
                    let _ = reply.send(Err(SessionError::Closed));
                } else if self.active.is_some() {
                    let _ = reply.send(Err(SessionError::Busy));
                } else {
                    self.set_status(SessionStatus::Running).await?;
                    let cancel = self.services.turn_registry.child_token();
                    let task_cancel = cancel.clone();
                    let executor = self.executor.clone();
                    let task = tokio::spawn(async move {
                        executor
                            .compact_with_cancel(&reason, instructions.as_deref(), task_cancel)
                            .await
                            .map(|()| TaskOutcome::Complete)
                    });
                    self.active = Some(Active {
                        cancel,
                        task,
                        kind: Work::Compact(reply),
                        turn_id: None,
                    });
                }
            }
        }
        Ok(())
    }

    async fn start_execution(&mut self) -> anyhow::Result<()> {
        self.delivered.clear();
        self.wake_requested = false;
        self.set_status(SessionStatus::Running).await?;
        let executor = self.executor.clone();
        let cancel = self.services.turn_registry.child_token();
        let task_cancel = cancel.clone();
        let turn = Turn {
            id: TurnId::new(),
            user_message: self.pending[0].clone(),
            default_model: None,
            subagent_model: None,
        };
        let turn_id = turn.id.clone();
        let active_turn_id = turn_id.clone();
        let session_id = self.control.id.clone();
        let forwarded = self.control.forwarded.clone();
        let task = tokio::spawn(async move {
            let mut events = executor.submit_turn_with_cancel(turn, task_cancel).await?;
            let mut outcome = None;
            // Semantic failures are recorded as events. Drain to completion so
            // cleanup finishes, but do not automatically retry undelivered input.
            while let Some(event) = events.try_next().await? {
                if event.session_id != session_id
                    || matches!(event.payload, SessionEventPayload::Lagged { .. })
                {
                    let _ = forwarded.send(event);
                } else if matches!(&event.payload, SessionEventPayload::TurnFailed { turn_id: failed, .. } if failed == &turn_id)
                {
                    outcome = Some(TaskOutcome::TurnFailed);
                } else if matches!(&event.payload, SessionEventPayload::TurnCompleted { turn_id: completed, .. } if completed == &turn_id)
                {
                    outcome = Some(TaskOutcome::Complete);
                }
            }
            outcome.ok_or_else(|| anyhow::anyhow!("execution ended without a terminal event"))
        });
        self.active = Some(Active {
            cancel,
            task,
            kind: Work::Execution,
            turn_id: Some(active_turn_id),
        });
        Ok(())
    }

    fn start_cleanup(&mut self) {
        let executor = self.executor.clone();
        let sessions = self.services.tool_sessions.clone();
        let subagents = self.subagents.clone();
        let id = self.control.id.clone();
        let deadline = self.shutdown_deadline;
        let task = tokio::spawn(async move {
            let children = subagents.close_session_with_deadline(&id, deadline).await;
            let resources = sessions.shutdown_session(&id).await;
            let hooks = executor.shutdown("session_closed").await;
            children
                .and(resources)
                .and(hooks)
                .map(|()| TaskOutcome::Complete)
        });
        self.active = Some(Active {
            cancel: CancellationToken::new(),
            task,
            kind: Work::Cleanup,
            turn_id: None,
        });
    }

    fn set_cancel_deadline(&mut self, deadline: Option<Instant>) {
        if let Some(deadline) = deadline {
            self.cancel_deadline = Some(
                self.cancel_deadline
                    .map_or(deadline, |existing| existing.min(deadline)),
            );
        }
    }

    fn force_active(&mut self) {
        let closing = self.closing;
        let sessions = self.services.tool_sessions.clone();
        let subagents = self.subagents.clone();
        let id = self.control.id.clone();
        let force_resources = async move {
            if closing {
                sessions.force_stop_session(&id).await;
                subagents.force_close_session(&id).await;
            }
        };
        let Some(active) = self.active.take() else {
            return;
        };
        active.cancel.cancel();
        match active.kind {
            Work::Execution => {
                let executor = self.executor.clone();
                let turn_id = active.turn_id.expect("execution has an internal turn id");
                let recovery_id = turn_id.clone();
                let task = tokio::spawn(async move {
                    active.task.abort();
                    let _ = active.task.await;
                    force_resources.await;
                    let result = executor.force_interrupt_turn(&recovery_id).await;
                    result.map(|()| TaskOutcome::TurnFailed)
                });
                self.active = Some(Active {
                    cancel: active.cancel,
                    task,
                    kind: Work::Execution,
                    turn_id: Some(turn_id),
                });
            }
            Work::Compact(reply) => {
                active.task.abort();
                let task = tokio::spawn(async move {
                    let _ = active.task.await;
                    force_resources.await;
                    Err(anyhow::Error::new(SessionError::TimedOut))
                });
                self.active = Some(Active {
                    cancel: active.cancel,
                    task,
                    kind: Work::Compact(reply),
                    turn_id: None,
                });
            }
            Work::Cleanup => {
                // Cleanup remains owned and the incarnation fenced until it
                // settles. The caller's deadline bounds waiting, not SQL writes
                // or already-running blocking plugin code.
                let task = tokio::spawn(async move {
                    force_resources.await;
                    active.task.await.map_err(anyhow::Error::from)?
                });
                self.active = Some(Active {
                    cancel: active.cancel,
                    task,
                    kind: Work::Cleanup,
                    turn_id: None,
                });
            }
        }
    }

    async fn set_status(&mut self, status: SessionStatus) -> anyhow::Result<()> {
        if status == SessionStatus::Closed {
            self.defer_pending(InputDeferredReason::Shutdown).await?;
        }
        self.commit_payload(SessionEventPayload::SessionStatusChanged { status })
            .await
            .map(|_| ())
    }

    async fn defer_pending(&mut self, reason: InputDeferredReason) -> anyhow::Result<()> {
        let ids = self
            .pending
            .iter()
            .filter(|m| !self.deferred.contains(&m.id))
            .map(|m| m.id.clone())
            .collect::<Vec<_>>();
        for message_id in ids {
            self.commit_payload(SessionEventPayload::InputDeferred {
                message_id: message_id.clone(),
                reason: reason.clone(),
            })
            .await?;
            self.deferred.insert(message_id);
        }
        Ok(())
    }

    async fn commit_payload(&mut self, payload: SessionEventPayload) -> anyhow::Result<u64> {
        // Driver-owned commits change only inbox/lifecycle metadata. Any new
        // state mutation must use the routed commit path and advance the
        // last_state_commit watermark, or stale checkpoints could overwrite it.
        debug_assert!(matches!(
            payload,
            SessionEventPayload::InputAccepted { .. }
                | SessionEventPayload::SessionStatusChanged { .. }
                | SessionEventPayload::InputDeferred { .. }
                | SessionEventPayload::InputSettled { .. }
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
        mut events: Vec<PendingEvent>,
    ) -> anyhow::Result<Vec<SessionEvent>> {
        let mut projection = SessionState {
            pending_inputs: self.pending.clone(),
            session_status: self.status,
            ..SessionState::default()
        };
        let mut delivered = self.delivered.clone();
        let mut terminal = None;
        for event in &events {
            if let SessionEventPayload::MessageItem {
                message: Message::User(message),
            } = &event.payload
                && projection
                    .pending_inputs
                    .iter()
                    .any(|pending| pending.id == message.id)
            {
                delivered.insert(message.id.clone());
            }
            if let Some(outcome) = input_outcome(&event.payload) {
                terminal = Some(outcome);
            }
            halter_protocol::fold::apply_event(&mut projection, &event.payload);
        }
        let mut deferred_ids = Vec::new();
        if let Some(outcome) = &terminal {
            // Settlement shares the terminal commit: no completed execution
            // can be replayed without the outcome of its delivered inputs.
            let mut ids = delivered.iter().cloned().collect::<Vec<_>>();
            ids.sort_by(|a, b| a.0.cmp(&b.0));
            for message_id in ids {
                events.push(PendingEvent::new(
                    self.control.id.clone(),
                    halter_protocol::Delivery::Lossless,
                    SessionEventPayload::InputSettled {
                        message_id,
                        outcome: outcome.clone(),
                    },
                ));
            }
            let reason = match outcome {
                InputOutcome::Interrupted => Some(InputDeferredReason::Interrupted),
                InputOutcome::Failed { error, retryable } => {
                    Some(InputDeferredReason::ExecutionFailed {
                        error: error.clone(),
                        retryable: *retryable,
                    })
                }
                InputOutcome::Completed if self.closing => Some(InputDeferredReason::Shutdown),
                InputOutcome::Completed => None,
            };
            if let Some(reason) = reason {
                for message in &projection.pending_inputs {
                    if !self.deferred.contains(&message.id) {
                        deferred_ids.push(message.id.clone());
                        events.push(PendingEvent::new(
                            self.control.id.clone(),
                            halter_protocol::Delivery::Lossless,
                            SessionEventPayload::InputDeferred {
                                message_id: message.id.clone(),
                                reason: reason.clone(),
                            },
                        ));
                    }
                }
            }
        }
        if let Some(state) = &mut state {
            state.pending_inputs = projection.pending_inputs.clone();
            state.session_status = projection.session_status;
        }
        let committed = self
            .store
            .commit(&self.control.id, snapshot, Some(self.head), state, events)
            .await?;
        self.head = committed.last().map_or(self.head, SessionEvent::sequence);
        self.pending = projection.pending_inputs;
        self.status = projection.session_status;
        self.delivered = if terminal.is_some() {
            HashSet::new()
        } else {
            delivered
        };
        self.deferred.extend(deferred_ids);
        self.deferred
            .retain(|id| self.pending.iter().any(|m| &m.id == id));
        if terminal.is_some_and(|outcome| outcome != InputOutcome::Completed) {
            self.wake_requested = false;
        }
        Ok(committed)
    }
}

fn input_outcome(payload: &SessionEventPayload) -> Option<InputOutcome> {
    match payload {
        SessionEventPayload::TurnCompleted { .. } => Some(InputOutcome::Completed),
        SessionEventPayload::TurnFailed {
            cancelled: true, ..
        } => Some(InputOutcome::Interrupted),
        SessionEventPayload::TurnFailed {
            error, retryable, ..
        } => Some(InputOutcome::Failed {
            error: error.clone(),
            retryable: *retryable,
        }),
        _ => None,
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
}

fn session_events(control: Arc<DriverControl>, sequence: u64) -> SessionEventStream {
    let receiver = control.services.event_bus.subscribe_raw();
    let forwarded = control.forwarded.subscribe();
    stream::try_unfold(EventCursor { control, receiver, forwarded, sequence, buffered: Default::default(), replay_needed: true }, |mut cursor| async move {
        loop {
            if let Some(event) = cursor.buffered.pop_front() {
                cursor.sequence = event.sequence();
                return Ok(Some((event, cursor)));
            }
            if cursor.replay_needed || cursor.control.closed.is_cancelled() {
                cursor.buffered.extend(cursor.control.services.sessions.replay_after(&cursor.control.id, cursor.sequence).await?);
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
                    return Err(anyhow::anyhow!("{error}"));
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
                services: crate::session_driver_tests::services(Arc::new(
                    halter_providers::FakeProvider::default(),
                )),
                forwarded: broadcast::channel(1).0,
                failure: Mutex::new(None),
            });
            let handle = SessionHandle { control };
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
    async fn resume_settles_delivered_input_from_a_crashed_execution() {
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
        assert_eq!(log.iter().filter(|event| matches!(&event.payload,
            SessionEventPayload::InputSettled { message_id, outcome: InputOutcome::Interrupted } if message_id == &input.id)).count(), 1);
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
            services: services.clone(),
            forwarded: broadcast::channel(FORWARDED_EVENT_CAPACITY).0,
            failure: Mutex::new(None),
        });
        let mut driver = Driver {
            executor,
            services: services.clone(),
            store: services.sessions.clone(),
            subagents: RuntimeSubagentControl::new(services.clone()),
            control,
            rx,
            stop_rx,
            head: base,
            last_state_commit: base,
            pending: Vec::new(),
            accepted: HashMap::new(),
            delivered: HashSet::new(),
            deferred: HashSet::new(),
            status: SessionStatus::Idle,
            active: None,
            wake_requested: false,
            closing: false,
            interrupt_waiters: Vec::new(),
            shutdown_waiters: Vec::new(),
            cancel_deadline: None,
            shutdown_deadline: None,
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
