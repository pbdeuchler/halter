//! A session owns admission, its durable inbox, and its live execution.
// pattern: Imperative Shell

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, Weak};

use async_trait::async_trait;
use futures::{FutureExt, StreamExt, TryStreamExt, stream};
use halter_protocol::{
    Message, MessageId, PendingEvent, ResourceSnapshot, SessionBlueprint, SessionEvent,
    SessionEventPayload, SessionId, SessionState, SessionStatus, Turn, UserMessage,
};
use halter_session::{SessionStore, StoredSession};
use thiserror::Error;
use tokio::sync::{broadcast, mpsc, oneshot};
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::session::{HalterSession, RuntimeServices, SessionEventStream, hydrate_stored_session};
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
    #[error(transparent)]
    Operation(#[from] anyhow::Error),
}

type Reply<T> = oneshot::Sender<Result<T, SessionError>>;

/// A cloneable connection to one live incarnation of a stored session.
/// Dropping a handle or event stream does not stop execution.
#[derive(Clone)]
pub struct SessionHandle {
    control: Arc<DriverControl>,
}

struct DriverControl {
    id: SessionId,
    tx: mpsc::Sender<Command>,
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
    pub async fn submit(&self, message: Message) -> Result<MessageId, SessionError> {
        let Message::User(message) = message else {
            return Err(SessionError::InvalidMessage);
        };
        self.request(|reply| Command::Submit(message, reply)).await
    }

    /// Stop foreground execution, await cleanup and commit its final state.
    /// Earlier pending inputs remain recorded but do not restart execution.
    pub async fn interrupt(&self) -> Result<(), SessionError> {
        self.request(Command::Interrupt).await
    }

    /// Close this incarnation and await cleanup of its session resources.
    pub async fn shutdown(&self) -> Result<(), SessionError> {
        if !self.control.closed.is_cancelled() {
            match self.request(Command::Shutdown).await {
                Err(SessionError::Closed) => self.control.closed.cancelled().await,
                result => return result,
            }
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

    pub(crate) async fn close_all(&self) {
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
            if let Err(error) = handle.shutdown().await {
                tracing::warn!(session_id = %handle.id(), %error, "session cleanup failed during runtime shutdown");
            }
        }))
        .await;
    }

    pub(crate) async fn open(
        self: &Arc<Self>,
        executor: HalterSession,
        services: Arc<RuntimeServices>,
        subagents: RuntimeSubagentControl,
        after: u64,
    ) -> Result<(SessionHandle, SessionEventStream), SessionError> {
        let id = executor.session_id().clone();
        let mut stored = services
            .sessions
            .load_session(&id)
            .await?
            .ok_or_else(|| SessionError::NotFound(id.clone()))?;
        hydrate_stored_session(services.sessions.as_ref(), &mut stored).await?;
        let (tx, rx) = mpsc::channel(INBOX_CAPACITY);
        let control = Arc::new(DriverControl {
            id: id.clone(),
            tx: tx.clone(),
            closed: CancellationToken::new(),
            services: services.clone(),
            forwarded: broadcast::channel(FORWARDED_EVENT_CAPACITY).0,
            failure: std::sync::Mutex::new(None),
        });
        let events = session_events(control.clone(), after);
        let store = Arc::new(SessionInbox { tx });
        let executor = executor.with_driver(store);
        let mut driver = Driver {
            executor,
            services: services.clone(),
            subagents,
            store: self.store.clone(),
            control: control.clone(),
            rx,
            head: stored.head_sequence,
            pending: stored.state.pending_inputs,
            status: SessionStatus::Idle,
            active: None,
            wake_requested: false,
            closing: false,
            interrupt_waiters: Vec::new(),
            shutdown_waiters: Vec::new(),
            accepted: HashSet::new(),
        };
        for event in services.sessions.replay(&id).await? {
            if let SessionEventPayload::InputAccepted { message } = event.payload {
                driver.accepted.insert(message.id);
            }
        }
        driver.set_status(SessionStatus::Idle).await?;
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
                while let Ok(command) = driver.rx.try_recv() {
                    drop(command);
                }
                if let Some(active) = driver.active.take() {
                    active.cancel.cancel();
                    let _ = active.task.await;
                }
                registry.lock_entries().insert(
                    id.clone(),
                    DriverSlot::Failed(Arc::downgrade(&task_control)),
                );
                let _ = driver.subagents.close_session(&id).await;
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
    Submit(UserMessage, Reply<MessageId>),
    Interrupt(Reply<()>),
    Shutdown(Reply<()>),
    Compact {
        reason: String,
        instructions: Option<String>,
        reply: Reply<()>,
    },
    Pending(oneshot::Sender<Vec<UserMessage>>),
    Commit {
        snapshot: Option<Arc<ResourceSnapshot>>,
        state: Option<Box<SessionState>>,
        events: Vec<PendingEvent>,
        reply: oneshot::Sender<anyhow::Result<Vec<SessionEvent>>>,
    },
}

struct Active {
    cancel: CancellationToken,
    task: JoinHandle<anyhow::Result<TaskOutcome>>,
    kind: Work,
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
    executor: HalterSession,
    services: Arc<RuntimeServices>,
    store: Arc<dyn SessionStore>,
    subagents: RuntimeSubagentControl,
    control: Arc<DriverControl>,
    rx: mpsc::Receiver<Command>,
    head: u64,
    pending: Vec<UserMessage>,
    accepted: HashSet<MessageId>,
    status: SessionStatus,
    active: Option<Active>,
    wake_requested: bool,
    closing: bool,
    interrupt_waiters: Vec<Reply<()>>,
    shutdown_waiters: Vec<Reply<()>>,
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
                    for reply in self.interrupt_waiters.drain(..) {
                        let response = match &result {
                            Err(error) if !error.downcast_ref::<halter_protocol::ProviderError>().is_some_and(halter_protocol::ProviderError::is_cancelled) => {
                                Err(SessionError::Operation(anyhow::anyhow!("{error:#}")))
                            }
                            _ => Ok(()),
                        };
                        let _ = reply.send(response);
                    }
                }
            }
        }
    }

    async fn command(&mut self, command: Command) -> anyhow::Result<()> {
        match command {
            Command::Commit {
                snapshot,
                state,
                events,
                reply,
            } => {
                let _ = reply.send(
                    self.commit(snapshot, state.map(|state| *state), events)
                        .await,
                );
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
                } else if self.accepted.contains(&message.id) {
                    if self.pending.iter().any(|pending| pending.id == message.id) {
                        self.wake_requested = true;
                    }
                    let _ = reply.send(Ok(message.id));
                } else if self.pending.len() >= INBOX_CAPACITY {
                    let _ = reply.send(Err(SessionError::InboxFull));
                } else {
                    let id = message.id.clone();
                    let result = self
                        .commit_payload(SessionEventPayload::InputAccepted { message })
                        .await;
                    if result.is_ok() {
                        self.accepted.insert(id.clone());
                        self.wake_requested = true;
                    }
                    let _ = reply.send(result.map(|_| id).map_err(SessionError::from));
                }
            }
            Command::Interrupt(reply) => {
                self.wake_requested = false;
                if self.closing {
                    let _ = reply.send(Err(SessionError::Closed));
                } else if let Some(active) = &self.active {
                    active.cancel.cancel();
                    self.interrupt_waiters.push(reply);
                } else {
                    let _ = reply.send(Ok(()));
                }
            }
            Command::Shutdown(reply) => {
                self.closing = true;
                self.wake_requested = false;
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
                    });
                }
            }
        }
        Ok(())
    }

    async fn start_execution(&mut self) -> anyhow::Result<()> {
        self.wake_requested = false;
        self.set_status(SessionStatus::Running).await?;
        let executor = self.executor.clone();
        let cancel = self.services.turn_registry.child_token();
        let task_cancel = cancel.clone();
        let mut turn = Turn::user("");
        turn.user_message = self.pending[0].clone();
        let turn_id = turn.id.clone();
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
        });
        Ok(())
    }

    fn start_cleanup(&mut self) {
        let executor = self.executor.clone();
        let sessions = self.services.tool_sessions.clone();
        let subagents = self.subagents.clone();
        let id = self.control.id.clone();
        let task = tokio::spawn(async move {
            let children = subagents.close_session(&id).await;
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
        });
    }

    async fn set_status(&mut self, status: SessionStatus) -> anyhow::Result<()> {
        self.commit_payload(SessionEventPayload::SessionStatusChanged { status })
            .await
    }

    async fn commit_payload(&mut self, payload: SessionEventPayload) -> anyhow::Result<()> {
        let event = PendingEvent::new(
            self.control.id.clone(),
            halter_protocol::Delivery::Lossless,
            payload,
        );
        let committed = self.commit(None, None, vec![event]).await?;
        for event in committed {
            self.services.event_bus.publish(event.clone());
            if let Some(recorder) = &self.services.trace_recorder {
                recorder.record(&event);
            }
        }
        Ok(())
    }

    async fn commit(
        &mut self,
        snapshot: Option<Arc<ResourceSnapshot>>,
        mut state: Option<SessionState>,
        events: Vec<PendingEvent>,
    ) -> anyhow::Result<Vec<SessionEvent>> {
        let mut projection = SessionState {
            pending_inputs: self.pending.clone(),
            session_status: self.status,
            ..SessionState::default()
        };
        for event in &events {
            halter_protocol::fold::apply_event(&mut projection, &event.payload);
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
        Ok(committed)
    }
}

struct EventCursor {
    control: Arc<DriverControl>,
    receiver: broadcast::Receiver<SessionEvent>,
    forwarded: broadcast::Receiver<SessionEvent>,
    sequence: u64,
    buffered: std::collections::VecDeque<SessionEvent>,
}

fn session_events(control: Arc<DriverControl>, sequence: u64) -> SessionEventStream {
    let receiver = control.services.event_bus.subscribe_raw();
    let forwarded = control.forwarded.subscribe();
    stream::try_unfold(EventCursor { control, receiver, forwarded, sequence, buffered: Default::default() }, |mut cursor| async move {
        loop {
            if let Some(event) = cursor.buffered.pop_front() {
                cursor.sequence = event.sequence();
                return Ok(Some((event, cursor)));
            }
            cursor.buffered.extend(cursor.control.services.sessions.replay_after(&cursor.control.id, cursor.sequence).await?);
            if !cursor.buffered.is_empty() { continue; }
            match cursor.forwarded.try_recv() {
                Ok(event) => return Ok(Some((event, cursor))),
                Err(broadcast::error::TryRecvError::Lagged(dropped)) => {
                    return Ok(Some((forwarding_lagged_event(dropped), cursor)));
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
                _ = cursor.control.closed.cancelled() => {},
                event = cursor.receiver.recv() => {
                    if matches!(event, Err(broadcast::error::RecvError::Closed)) { return Ok(None); }
                },
                event = cursor.forwarded.recv() => {
                    match event {
                        Ok(event) => return Ok(Some((event, cursor))),
                        Err(broadcast::error::RecvError::Lagged(dropped)) => {
                            return Ok(Some((forwarding_lagged_event(dropped), cursor)));
                        },
                        Err(broadcast::error::RecvError::Closed) => {},
                    }
                },
            }
        }
    }).boxed()
}

fn forwarding_lagged_event(dropped_events: u64) -> SessionEvent {
    PendingEvent::new(
        SessionId::from(crate::event_bus::BUS_SESSION_ID),
        halter_protocol::Delivery::BestEffort,
        SessionEventPayload::Lagged { dropped_events },
    )
    .into_committed(0)
}
