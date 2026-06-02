// pattern: Imperative Shell

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, Weak};

use anyhow::Context;
use arc_swap::ArcSwap;
use chrono::Utc;
use futures::stream::{BoxStream, StreamExt};
use halter_hooks::{Hooks, RegisteredHooks};
use halter_protocol::{
    AssistantMessage, AssistantPart, BlockId, CacheScope, ContentHash, Delivery,
    HookSessionStartSource, HookWarning, Message, MessageId, ModelId, ObservedState, PendingEvent,
    PendingToolCall, PromptSegment, PromptSegmentId, PromptSegmentKind, ProviderError,
    ProviderRequest, ReplayMeta, ResourceSnapshot, SessionBlueprint, SessionEvent,
    SessionEventPayload, SessionId, SessionState, StopReason, StreamEvent, SubagentEventForwarding,
    SystemMessage, ToolCall, ToolError, ToolExecutionOutcome, ToolResult, ToolResultMessage, Turn,
    TurnId, Usage, Volatility,
};
use halter_providers::ModelRegistry;
use halter_session::{SessionStore, StoredSession};
use halter_tools::{
    PathLockMap, SubagentControl, SubagentParentContext, ToolEventSink, ToolPolicy, ToolRuntime,
    ToolRuntimeEvent, ToolSessionStore,
};
use sha2::{Digest, Sha256};
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

#[cfg(test)]
use crate::DefaultContextManager;
use crate::model_selection::select_models;
use crate::turn_registry::TurnRegistry;
use crate::{
    ContextManager, EventBus, ExecutedHookDispatch, HookInvocationContext, PromptAssembler,
    run_notification, run_post_compact, run_post_tool_use, run_post_tool_use_failure,
    run_pre_compact, run_pre_tool_use, run_session_end, run_session_start, run_stop,
    run_user_prompt_submit,
};

pub type SessionEventStream = BoxStream<'static, anyhow::Result<SessionEvent>>;

const PROVIDER_STREAM_OUTPUT_CAP_BYTES: usize = 4 * 1024 * 1024;
const PROVIDER_STREAM_EVENT_CAP: usize = 8_192;
const TOOL_RUNTIME_EVENT_CAP: usize = 4_096;
const TOOL_RUNTIME_EVENT_BYTES_CAP: usize = 1024 * 1024;

pub struct RuntimeServices {
    pub resources: Arc<ResourceHandle>,
    pub registered_hooks: Arc<RegisteredHooks>,
    pub session_hook_store: Arc<Mutex<HashMap<SessionId, Arc<Hooks>>>>,
    pub models: Arc<ModelRegistry>,
    pub tools: Arc<ToolRuntime>,
    pub path_locks: Arc<PathLockMap>,
    pub tool_sessions: Arc<ToolSessionStore>,
    pub sessions: Arc<dyn SessionStore>,
    pub policy: Arc<dyn ToolPolicy>,
    pub prompt_assembler: Arc<dyn PromptAssembler>,
    pub context_manager: Arc<dyn ContextManager>,
    pub event_bus: Arc<EventBus>,
    pub parent_streams: Arc<ParentStreamRegistry>,
    pub turn_registry: Arc<TurnRegistry>,
    pub subagent_event_forwarding: SubagentEventForwarding,
    pub subagent_event_forwarding_cap: u64,
    pub shell_timeout_secs: u64,
    /// Optional sink that mirrors every committed `SessionEvent` into a
    /// per-session JSONL trace file. Disabled (`None`) when
    /// `runtime.traces_dir` is not configured.
    pub trace_recorder: Option<Arc<crate::TraceRecorder>>,
}

#[derive(Debug, Clone)]
pub struct ResourceHandle {
    current: Arc<ArcSwap<ResourceState>>,
}

#[derive(Clone, Debug)]
struct ResourceState {
    snapshot: ResourceSnapshot,
    hooks: Arc<Hooks>,
    hook_warnings: Arc<Vec<HookWarning>>,
}

impl ResourceHandle {
    #[must_use]
    pub fn new(
        snapshot: ResourceSnapshot,
        hooks: Arc<Hooks>,
        hook_warnings: Vec<HookWarning>,
    ) -> Self {
        Self {
            current: Arc::new(ArcSwap::from_pointee(ResourceState {
                snapshot,
                hooks,
                hook_warnings: Arc::new(hook_warnings),
            })),
        }
    }

    #[must_use]
    pub fn snapshot(&self) -> Arc<ResourceSnapshot> {
        Arc::new(self.current.load().snapshot.clone())
    }

    #[must_use]
    pub fn hooks(&self) -> Arc<Hooks> {
        self.current.load().hooks.clone()
    }

    #[must_use]
    pub fn hook_warnings(&self) -> Arc<Vec<HookWarning>> {
        self.current.load().hook_warnings.clone()
    }

    pub fn replace(
        &self,
        snapshot: ResourceSnapshot,
        hooks: Arc<Hooks>,
        hook_warnings: Vec<HookWarning>,
    ) {
        info!(revision = %snapshot.revision, "replaced resource snapshot");
        self.current.store(Arc::new(ResourceState {
            snapshot,
            hooks,
            hook_warnings: Arc::new(hook_warnings),
        }));
    }
}

#[derive(Debug, Clone)]
pub struct SessionInit {
    pub session_id: Option<SessionId>,
    pub parent_session_id: Option<SessionId>,
    pub working_dir: PathBuf,
    pub system_prompt_seed: Vec<PromptSegment>,
    pub max_turns: Option<u32>,
    pub default_model: Option<ModelId>,
    pub subagent_model: Option<ModelId>,
    pub subagent_event_forwarding: Option<SubagentEventForwarding>,
    pub subagent_depth: u32,
}

impl Default for SessionInit {
    fn default() -> Self {
        Self {
            session_id: None,
            parent_session_id: None,
            working_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            system_prompt_seed: vec![crate::prompt::default_system_prompt_segment()],
            max_turns: None,
            default_model: None,
            subagent_model: None,
            subagent_event_forwarding: None,
            subagent_depth: 0,
        }
    }
}

impl SessionInit {
    #[must_use]
    pub fn with_default_model(mut self, model: impl Into<ModelId>) -> Self {
        self.default_model = Some(model.into());
        self
    }

    #[must_use]
    pub fn with_subagent_model(mut self, model: impl Into<ModelId>) -> Self {
        self.subagent_model = Some(model.into());
        self
    }

    #[must_use]
    pub fn with_subagent_event_forwarding(mut self, mode: SubagentEventForwarding) -> Self {
        self.subagent_event_forwarding = Some(mode);
        self
    }
}

/// Drop-time hook eviction guard. Held inside an `Arc` on the session
/// handle so eviction fires only when the *last* clone of the handle is
/// dropped. Pre-Phase-3 code put `Drop` on `HalterSession` itself, which
/// meant any short-lived clone (e.g. a clone moved into a `tokio::spawn`
/// turn loop) would evict the hooks for the still-live original handle.
///
/// The trace recorder is intentionally *not* closed here. Calls to
/// `HalterSession::new` for an already-live session (e.g. the parent's
/// short-lived handle constructed inside subagent hook dispatch) yield
/// independent `EvictionGuard`s; closing the trace on a temporary handle's
/// drop would silently remove the parent's writer entry while real work was
/// still streaming events under that session id. The recorder flushes per
/// line and is cleaned up when the runtime drops it.
struct EvictionGuard {
    services: Arc<RuntimeServices>,
    session_id: SessionId,
}

impl Drop for EvictionGuard {
    fn drop(&mut self) {
        evict_session_hooks(&self.services, &self.session_id);
    }
}

/// Cheaply-cloneable handle to a halter session. Cloning bumps the inner
/// `Arc`s; the session's hooks are only evicted from the runtime store
/// when every clone of the handle has been dropped.
#[derive(Clone)]
pub struct SessionHandle {
    services: Arc<RuntimeServices>,
    session_id: SessionId,
    session_hooks: Arc<Hooks>,
    /// Held only for its `Drop` side-effect on the last clone — see
    /// `EvictionGuard`.
    #[allow(dead_code)]
    eviction: Arc<EvictionGuard>,
}

/// Backwards-compatible alias for the public session type. Prefer
/// `SessionHandle` in new code.
pub type HalterSession = SessionHandle;

#[derive(Debug, Default)]
struct ToolEventBuffer {
    events: Vec<ToolRuntimeEvent>,
    bytes: usize,
    truncated: bool,
}

struct SessionToolEventSink {
    buffer: Arc<Mutex<ToolEventBuffer>>,
}

impl ToolEventSink for SessionToolEventSink {
    fn emit(&self, event: ToolRuntimeEvent) {
        let mut buffer = self
            .buffer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let event_bytes = tool_runtime_event_bytes(&event);
        let over_count = buffer.events.len() >= TOOL_RUNTIME_EVENT_CAP;
        let over_bytes = buffer.bytes.saturating_add(event_bytes) > TOOL_RUNTIME_EVENT_BYTES_CAP;
        if over_count || over_bytes {
            if !buffer.truncated {
                let tool_name = tool_runtime_event_tool_name(&event).to_owned();
                let chunk = format!(
                    "\n[tool output truncated after {} bytes]\n",
                    TOOL_RUNTIME_EVENT_BYTES_CAP
                );
                buffer.bytes = buffer.bytes.saturating_add(chunk.len());
                buffer
                    .events
                    .push(ToolRuntimeEvent::ToolOutput { tool_name, chunk });
                buffer.truncated = true;
            }
            return;
        }
        buffer.bytes = buffer.bytes.saturating_add(event_bytes);
        buffer.events.push(event);
    }
}

struct ToolEventDrain {
    buffer: Arc<Mutex<ToolEventBuffer>>,
}

impl ToolEventDrain {
    fn into_events(self) -> Vec<ToolRuntimeEvent> {
        self.buffer
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .events
            .drain(..)
            .collect()
    }
}

/// Sink for the in-turn live event stream. Only receives events *after*
/// `commit_and_publish` has assigned sequences; preserves the commit-then-
/// publish invariant by forbidding sequence allocation outside the session
/// store.
#[derive(Clone)]
struct LiveTurnStream {
    tx: mpsc::UnboundedSender<anyhow::Result<SessionEvent>>,
    forwarded_event_cap: Option<u64>,
    forwarded_state: Arc<Mutex<ForwardedEventState>>,
}

#[derive(Debug, Default)]
struct ForwardedEventState {
    forwarded_events: u64,
    capped: bool,
}

impl LiveTurnStream {
    fn new(tx: mpsc::UnboundedSender<anyhow::Result<SessionEvent>>, cap: u64) -> Self {
        Self {
            tx,
            forwarded_event_cap: (cap > 0).then_some(cap),
            forwarded_state: Arc::new(Mutex::new(ForwardedEventState::default())),
        }
    }

    fn emit_committed(&self, event: SessionEvent) {
        let _ = self.tx.send(Ok(event));
    }

    fn emit_forwarded(&self, event: SessionEvent) {
        let should_send_lagged = {
            let mut state = self
                .forwarded_state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if state.capped {
                return;
            }
            if let Some(cap) = self.forwarded_event_cap
                && state.forwarded_events >= cap
            {
                state.capped = true;
                true
            } else {
                state.forwarded_events = state.forwarded_events.saturating_add(1);
                false
            }
        };

        if should_send_lagged {
            let _ = self.tx.send(Ok(forwarding_lagged_event()));
            return;
        }

        let _ = self.tx.send(Ok(event));
    }

    fn emit_error(&self, error: anyhow::Error) {
        let _ = self.tx.send(Err(error));
    }
}

fn forwarding_lagged_event() -> SessionEvent {
    PendingEvent::new(
        SessionId::from(crate::event_bus::BUS_SESSION_ID),
        Delivery::BestEffort,
        SessionEventPayload::Lagged { dropped_events: 1 },
    )
    .into_committed(0)
}

#[derive(Default)]
pub struct ParentStreamRegistry {
    active: Mutex<HashMap<SessionId, Vec<Weak<LiveTurnStream>>>>,
}

impl ParentStreamRegistry {
    fn register(
        self: &Arc<Self>,
        session_id: SessionId,
        stream: &Arc<LiveTurnStream>,
    ) -> ParentStreamRegistration {
        let weak = Arc::downgrade(stream);
        let mut active = self
            .active
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        active
            .entry(session_id.clone())
            .or_default()
            .push(weak.clone());
        ParentStreamRegistration {
            registry: self.clone(),
            session_id,
            stream: weak,
        }
    }

    fn forward_to_ancestors(&self, ancestors: &[SessionId], event: &SessionEvent) {
        if ancestors.is_empty() {
            return;
        }

        let streams = {
            let mut active = self
                .active
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let mut streams = Vec::new();
            let mut empty_keys = Vec::new();
            for ancestor in ancestors {
                if let Some(entries) = active.get_mut(ancestor) {
                    entries.retain(|entry| {
                        if let Some(stream) = entry.upgrade() {
                            streams.push(stream);
                            true
                        } else {
                            false
                        }
                    });
                    if entries.is_empty() {
                        empty_keys.push(ancestor.clone());
                    }
                }
            }
            for key in empty_keys {
                active.remove(&key);
            }
            streams
        };

        for stream in streams {
            stream.emit_forwarded(event.clone());
        }
    }

    fn deregister(&self, session_id: &SessionId, stream: &Weak<LiveTurnStream>) {
        let mut active = self
            .active
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(entries) = active.get_mut(session_id) {
            entries.retain(|entry| !Weak::ptr_eq(entry, stream) && entry.strong_count() > 0);
            if entries.is_empty() {
                active.remove(session_id);
            }
        }
    }
}

struct ParentStreamRegistration {
    registry: Arc<ParentStreamRegistry>,
    session_id: SessionId,
    stream: Weak<LiveTurnStream>,
}

impl Drop for ParentStreamRegistration {
    fn drop(&mut self) {
        self.registry.deregister(&self.session_id, &self.stream);
    }
}

fn track_fired_hook_ids(fired_hook_ids: &mut BTreeSet<String>, dispatch: &ExecutedHookDispatch) {
    for fired_hook_id in &dispatch.fired_hook_ids {
        fired_hook_ids.insert(fired_hook_id.clone());
    }
}

#[derive(Clone)]
pub struct SessionRuntime {
    services: Arc<RuntimeServices>,
    subagents: Arc<dyn SubagentControl>,
}

impl SessionRuntime {
    #[must_use]
    pub fn new(services: Arc<RuntimeServices>) -> Self {
        let subagents: Arc<dyn SubagentControl> = Arc::new(
            crate::subagents::RuntimeSubagentControl::new(services.clone()),
        );
        Self {
            services,
            subagents,
        }
    }

    #[must_use]
    pub fn subagent_control(&self) -> Arc<dyn SubagentControl> {
        self.subagents.clone()
    }

    pub async fn new_session(&self, init: SessionInit) -> anyhow::Result<HalterSession> {
        debug!(
            working_dir = %init.working_dir.display(),
            parent_session_id = ?init.parent_session_id,
            max_turns = ?init.max_turns,
            default_model = ?init.default_model,
            subagent_model = ?init.subagent_model,
            subagent_depth = init.subagent_depth,
            "creating session"
        );
        create_session_seeded(
            self.services.clone(),
            init,
            SessionState::default(),
            self.services.resources.snapshot(),
        )
        .await
    }

    pub async fn resume(&self, session_id: &SessionId) -> anyhow::Result<Option<HalterSession>> {
        let existing = self.services.sessions.load_session(session_id).await?;
        debug!(session_id = %session_id, found = existing.is_some(), "resuming session");
        if let Some(mut stored) = existing {
            let expected_state = stored.state.clone();
            stored.state.pending_session_start_source = Some(HookSessionStartSource::Resume);
            self.services
                .sessions
                .commit(
                    session_id,
                    None,
                    Some(expected_state),
                    Some(stored.state),
                    Vec::new(),
                )
                .await?;
            return Ok(Some(HalterSession::new(
                self.services.clone(),
                session_id.clone(),
            )?));
        }
        Ok(None)
    }

    pub async fn list_sessions(&self) -> anyhow::Result<Vec<SessionBlueprint>> {
        let sessions = self.services.sessions.list_sessions().await?;
        debug!(session_count = sessions.len(), "listed sessions");
        Ok(sessions)
    }

    pub fn replace_resources(
        &self,
        snapshot: ResourceSnapshot,
        hooks: Arc<Hooks>,
        hook_warnings: Vec<HookWarning>,
    ) {
        self.services
            .resources
            .replace(snapshot, hooks, hook_warnings);
    }

    /// Mark the runtime as shutting down, cancel every in-flight turn,
    /// and wait for the spawned task loops to settle (or be aborted)
    /// within `drain`. After this returns, subsequent `submit_turn`
    /// calls fail with a "runtime is shutting down" error.
    ///
    /// Idempotent: calling shutdown after the registry is already in
    /// the shutting-down state still drains any newly raced-in turns.
    pub async fn shutdown(&self, drain: std::time::Duration) -> crate::ShutdownReport {
        let report = self.services.turn_registry.shutdown(drain).await;
        info!(
            drained = report.turns_drained,
            aborted = report.turns_aborted,
            timed_out = report.timed_out,
            drain_ms = %drain.as_millis(),
            "runtime shutdown"
        );
        report
    }
}

impl SessionHandle {
    pub(crate) fn new(
        services: Arc<RuntimeServices>,
        session_id: SessionId,
    ) -> anyhow::Result<Self> {
        let session_hooks = lookup_or_create_session_hooks(&services, &session_id)?;
        let eviction = Arc::new(EvictionGuard {
            services: services.clone(),
            session_id: session_id.clone(),
        });
        Ok(Self {
            services,
            session_id,
            session_hooks,
            eviction,
        })
    }

    #[must_use]
    pub fn session_id(&self) -> &SessionId {
        &self.session_id
    }

    pub(crate) fn services(&self) -> &Arc<RuntimeServices> {
        &self.services
    }

    pub(crate) fn session_hooks(&self) -> &Arc<Hooks> {
        &self.session_hooks
    }

    pub async fn submit_turn(&self, turn: Turn) -> anyhow::Result<SessionEventStream> {
        self.submit_turn_with_cancel(turn, CancellationToken::new())
            .await
    }

    pub(crate) async fn submit_turn_with_cancel(
        &self,
        turn: Turn,
        turn_cancel: CancellationToken,
    ) -> anyhow::Result<SessionEventStream> {
        info!(
            session_id = %self.session_id,
            turn_id = %turn.id,
            user_part_count = turn.user_message.parts.len(),
            "submitting turn"
        );
        let stored = self
            .services
            .sessions
            .load_session(&self.session_id)
            .await?
            .with_context(|| {
                format!(
                    "failed to submit turn: unknown session '{}'",
                    self.session_id.0
                )
            })?;
        // Reject new turns once the runtime is shutting down so callers
        // get a structured error instead of a turn that gets aborted
        // mid-flight.
        if self.services.turn_registry.is_shutting_down() {
            anyhow::bail!(
                "failed to submit turn '{}': runtime is shutting down",
                turn.id
            );
        }

        let (tx, rx) = mpsc::unbounded_channel();
        let live = Arc::new(LiveTurnStream::new(
            tx,
            self.services.subagent_event_forwarding_cap,
        ));
        let parent_stream_registration = stored
            .blueprint
            .subagent_event_forwarding
            .is_enabled()
            .then(|| {
                self.services
                    .parent_streams
                    .register(self.session_id.clone(), &live)
            });
        let session = self.clone();

        let registry = session.services.turn_registry.clone();
        let turn_id_for_dereg = turn.id.clone();
        let turn_id_for_register = turn.id.clone();
        let task_cancel = turn_cancel.clone();
        let task_cancel_status = turn_cancel.clone();
        let handle = tokio::spawn(async move {
            // Always deregister, even if the turn body panics, so we don't
            // leak entries that block shutdown drain.
            struct DeregisterOnDrop {
                registry: Arc<TurnRegistry>,
                turn_id: TurnId,
            }
            impl Drop for DeregisterOnDrop {
                fn drop(&mut self) {
                    self.registry.deregister(&self.turn_id);
                }
            }
            let _guard = DeregisterOnDrop {
                registry: registry.clone(),
                turn_id: turn_id_for_dereg,
            };
            let _parent_stream_registration = parent_stream_registration;

            let expected_state = stored.state.clone();
            let started = session.make_event(SessionEventPayload::TurnStarted {
                turn_id: turn.id.clone(),
            });
            if let Err(error) = session
                .commit_and_publish(
                    None,
                    Some(expected_state),
                    None,
                    vec![started],
                    Some(live.as_ref()),
                )
                .await
            {
                error!(
                    session_id = %session.session_id,
                    turn_id = %turn.id,
                    error = %error,
                    "failed to commit turn start"
                );
                live.emit_error(error);
                return;
            }

            match session
                .run_turn(stored, turn.clone(), task_cancel, live.as_ref())
                .await
            {
                Ok(turn_commit) => {
                    if let Err(error) = session
                        .commit_and_publish(
                            Some(turn_commit.snapshot),
                            Some(turn_commit.expected_state),
                            Some(turn_commit.state),
                            turn_commit.events,
                            Some(live.as_ref()),
                        )
                        .await
                    {
                        error!(
                            session_id = %session.session_id,
                            turn_id = %turn.id,
                            error = %error,
                            "failed to commit successful turn"
                        );
                        live.emit_error(error);
                    }
                }
                Err(error) => {
                    let provider_error = error.downcast_ref::<ProviderError>();
                    let retryable = provider_error
                        .map(|provider_error| provider_error.retryable)
                        .unwrap_or(false);
                    let cancelled = task_cancel_status.is_cancelled()
                        || provider_error.is_some_and(ProviderError::is_cancelled);
                    error!(
                        session_id = %session.session_id,
                        turn_id = %turn.id,
                        error = %error,
                        retryable,
                        cancelled,
                        "turn failed before commit"
                    );
                    let failure_events =
                        vec![session.make_event(SessionEventPayload::TurnFailed {
                            turn_id: turn.id.clone(),
                            error: error.to_string(),
                            cancelled,
                            retryable,
                        })];
                    if let Err(commit_error) = session
                        .commit_turn_failure(failure_events, live.as_ref())
                        .await
                    {
                        error!(
                            session_id = %session.session_id,
                            turn_id = %turn.id,
                            error = %commit_error,
                            "failed to commit failed turn"
                        );
                        live.emit_error(commit_error);
                    }
                }
            }
        });

        if let Err(register_error) =
            self.services
                .turn_registry
                .register(turn_id_for_register, turn_cancel, handle)
        {
            // The registry already cancelled the token and aborted the
            // task before returning the error. Surface the same shutdown
            // error to the caller as the upfront check would have.
            anyhow::bail!("failed to register turn: {register_error}");
        }

        Ok(UnboundedReceiverStream::new(rx).boxed())
    }

    pub async fn replay(&self) -> anyhow::Result<Vec<SessionEvent>> {
        self.services.sessions.replay(&self.session_id).await
    }

    pub async fn shutdown(&self, reason: &str) -> anyhow::Result<()> {
        let stored = self
            .services
            .sessions
            .load_session(&self.session_id)
            .await?
            .with_context(|| {
                format!(
                    "failed to shut down session: unknown session '{}'",
                    self.session_id.0
                )
            })?;
        let expected_state = stored.state.clone();
        let mut state = stored.state;
        let mut events = Vec::new();
        let turn_id = TurnId::new();
        let hook_ctx = HookInvocationContext {
            turn_id: &turn_id,
            model: &stored.blueprint.default_model,
            working_dir: &stored.blueprint.working_dir,
        };
        let fired_hook_ids = state
            .fired_hook_ids
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        let dispatch = run_session_end(self, &fired_hook_ids, hook_ctx, reason).await?;
        self.record_hook_dispatch(&mut events, &dispatch);
        if dispatch.merged.block_reason.is_some() || dispatch.merged.stop_reason.is_some() {
            warn!(session_id = %self.session_id, reason, "hooks.ignored_block");
        }
        for message in apply_hook_side_effects(&mut state, &dispatch) {
            self.push_event(&mut events, SessionEventPayload::MessageItem { message });
        }
        self.push_event(&mut events, SessionEventPayload::SessionShutdownComplete);
        let _ = self
            .commit_and_publish(None, Some(expected_state), Some(state), events, None)
            .await?;
        evict_session_hooks(&self.services, &self.session_id);
        Ok(())
    }

    pub async fn notify(&self, notification_type: &str, message: &str) -> anyhow::Result<()> {
        let stored = self
            .services
            .sessions
            .load_session(&self.session_id)
            .await?
            .with_context(|| {
                format!(
                    "failed to emit notification: unknown session '{}'",
                    self.session_id.0
                )
            })?;
        let expected_state = stored.state.clone();
        let mut state = stored.state;
        let turn_id = TurnId::new();
        let hook_ctx = HookInvocationContext {
            turn_id: &turn_id,
            model: &stored.blueprint.default_model,
            working_dir: &stored.blueprint.working_dir,
        };
        let fired_hook_ids = state
            .fired_hook_ids
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        let dispatch =
            run_notification(self, &fired_hook_ids, hook_ctx, notification_type, message).await?;
        let mut events = Vec::new();
        self.record_hook_dispatch(&mut events, &dispatch);
        for message in apply_hook_side_effects(&mut state, &dispatch) {
            self.push_event(&mut events, SessionEventPayload::MessageItem { message });
        }
        let _ = self
            .commit_and_publish(None, Some(expected_state), Some(state), events, None)
            .await?;
        Ok(())
    }

    pub async fn compact(
        &self,
        trigger: &str,
        custom_instructions: Option<&str>,
    ) -> anyhow::Result<()> {
        let stored = self
            .services
            .sessions
            .load_session(&self.session_id)
            .await?
            .with_context(|| {
                format!(
                    "failed to compact session: unknown session '{}'",
                    self.session_id.0
                )
            })?;
        let expected_state = stored.state.clone();
        let mut state = stored.state;
        let mut events = Vec::new();
        let turn_id = TurnId::new();
        let hook_ctx = HookInvocationContext {
            turn_id: &turn_id,
            model: &stored.blueprint.default_model,
            working_dir: &stored.blueprint.working_dir,
        };
        let mut fired_hook_ids = state
            .fired_hook_ids
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        let pre_dispatch = run_pre_compact(
            self,
            &fired_hook_ids,
            hook_ctx,
            trigger,
            custom_instructions,
        )
        .await?;
        track_fired_hook_ids(&mut fired_hook_ids, &pre_dispatch);
        self.record_hook_dispatch(&mut events, &pre_dispatch);
        for message in apply_hook_side_effects(&mut state, &pre_dispatch) {
            self.push_event(&mut events, SessionEventPayload::MessageItem { message });
        }
        if pre_dispatch.merged.block_reason.is_some() {
            let _ = self
                .commit_and_publish(None, Some(expected_state), Some(state), events, None)
                .await?;
            return Ok(());
        }

        let observed = observe_state(stored.blueprint.working_dir.clone());
        let compaction_model = self
            .services
            .models
            .model(&stored.blueprint.default_model)?;
        let compaction_provider = self.services.models.provider(&compaction_model.provider)?;
        let outcome = self
            .services
            .context_manager
            .compact_now(
                &stored.blueprint,
                &state,
                &observed,
                stored.snapshot.as_ref(),
                &self.services.tools.specs(),
                &compaction_model,
                compaction_provider.as_ref(),
                custom_instructions,
            )
            .await?;
        let summary = match outcome.apply(&mut state) {
            Some(result) => result.summary,
            None => "No compaction needed.".to_owned(),
        };
        self.push_event(
            &mut events,
            SessionEventPayload::ContextCompacted {
                summary: summary.clone(),
            },
        );

        let post_dispatch =
            run_post_compact(self, &fired_hook_ids, hook_ctx, trigger, &summary).await?;
        self.record_hook_dispatch(&mut events, &post_dispatch);
        for message in apply_hook_side_effects(&mut state, &post_dispatch) {
            self.push_event(&mut events, SessionEventPayload::MessageItem { message });
        }

        let _ = self
            .commit_and_publish(None, Some(expected_state), Some(state), events, None)
            .await?;
        Ok(())
    }

    async fn run_turn(
        &self,
        stored: StoredSession,
        turn: Turn,
        turn_cancel: CancellationToken,
        live: &LiveTurnStream,
    ) -> anyhow::Result<TurnCommit> {
        let snapshot = self.services.resources.snapshot();
        let mut expected_state = stored.state.clone();
        let mut state = stored.state;
        let mut events = Vec::new();
        let mut turn_usage = Usage::default();
        let mut provider_iterations = 0u32;
        let mut fired_hook_ids = state
            .fired_hook_ids
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>();
        let hook_model = turn
            .default_model
            .clone()
            .unwrap_or_else(|| stored.blueprint.default_model.clone());
        let hook_ctx = HookInvocationContext {
            turn_id: &turn.id,
            model: &hook_model,
            working_dir: &stored.blueprint.working_dir,
        };

        for warning in std::mem::take(&mut state.pending_warning_messages) {
            self.push_event(
                &mut events,
                SessionEventPayload::Warning {
                    message: format_hook_warning(&warning),
                },
            );
        }

        if let Some(source) = state.pending_session_start_source.take() {
            let hook_dispatch = run_session_start(self, &fired_hook_ids, hook_ctx, source).await?;
            track_fired_hook_ids(&mut fired_hook_ids, &hook_dispatch);
            self.record_hook_dispatch(&mut events, &hook_dispatch);
            if hook_dispatch.merged.block_reason.is_some()
                || hook_dispatch.merged.stop_reason.is_some()
            {
                warn!(
                    session_id = %self.session_id,
                    turn_id = %turn.id,
                    "hooks.ignored_block"
                );
            }
            for message in apply_hook_side_effects(&mut state, &hook_dispatch) {
                self.push_event(&mut events, SessionEventPayload::MessageItem { message });
            }
        }

        let prompt_dispatch = run_user_prompt_submit(
            self,
            &fired_hook_ids,
            hook_ctx,
            &turn.user_message.plain_text(),
        )
        .await?;
        track_fired_hook_ids(&mut fired_hook_ids, &prompt_dispatch);
        self.record_hook_dispatch(&mut events, &prompt_dispatch);
        for message in apply_hook_side_effects(&mut state, &prompt_dispatch) {
            self.push_event(&mut events, SessionEventPayload::MessageItem { message });
        }

        if let Some(reason) = prompt_dispatch
            .merged
            .stop_reason
            .clone()
            .or_else(|| prompt_dispatch.merged.block_reason.clone())
        {
            let blocked = Message::System(SystemMessage {
                id: MessageId::new(),
                created_at: Utc::now(),
                text: reason,
            });
            state.messages.push(blocked.clone());
            self.push_event(
                &mut events,
                SessionEventPayload::MessageItem { message: blocked },
            );
            events.push(self.make_event(SessionEventPayload::TurnCompleted {
                turn_id: turn.id,
                usage: turn_usage,
            }));
            return Ok(TurnCommit {
                expected_state,
                snapshot,
                state,
                events,
            });
        }

        let user_message = Message::User(turn.user_message.clone());
        state.messages.push(user_message.clone());
        self.push_event(
            &mut events,
            SessionEventPayload::MessageItem {
                message: user_message,
            },
        );
        self.flush_turn_progress(
            snapshot.clone(),
            &mut expected_state,
            &state,
            &mut events,
            live,
        )
        .await?;

        loop {
            ensure_provider_iteration_allowed(stored.blueprint.max_turns, provider_iterations)?;
            provider_iterations = provider_iterations.saturating_add(1);

            let compaction_model = self
                .services
                .models
                .model(&stored.blueprint.default_model)?;
            let compaction_provider = self.services.models.provider(&compaction_model.provider)?;
            let observed = observe_state(stored.blueprint.working_dir.clone());
            let plan = self
                .services
                .context_manager
                .plan(
                    &stored.blueprint,
                    &state,
                    &observed,
                    snapshot.as_ref(),
                    &self.services.tools.specs(),
                    &compaction_model,
                    compaction_provider.as_ref(),
                )
                .await?;

            let plan_outcome = crate::CompactionOutcome {
                messages: plan.messages.clone(),
                compacted_prefix: plan.compacted_prefix.clone(),
                compaction: plan.compaction.clone(),
                session_start_latch: None,
            };
            if let Some(result) = plan_outcome.apply(&mut state) {
                self.push_event(
                    &mut events,
                    SessionEventPayload::ContextCompacted {
                        summary: result.summary,
                    },
                );
            }

            let prompt = self.services.prompt_assembler.assemble(&plan).await?;

            let selected_models = select_models(
                &stored.blueprint.default_model,
                &stored.blueprint.subagent_model,
                turn.default_model.as_ref(),
                turn.subagent_model.as_ref(),
            );
            let model = self.services.models.model(&selected_models.default_model)?;
            let subagent_model = self
                .services
                .models
                .model(&selected_models.subagent_model)?;
            let provider = self.services.models.provider(&model.provider)?;
            let request = ProviderRequest {
                session_id: self.session_id.clone(),
                turn_id: turn.id.clone(),
                model: model.clone(),
                prompt,
                compacted_prefix: plan.compacted_prefix.clone(),
                messages: plan.messages.clone(),
                tools: plan.tool_specs.clone(),
                previous_response_id: plan.previous_response_id.clone(),
                new_messages_start: plan.new_messages_start,
            };

            let provider_stream = provider.stream(request, turn_cancel.child_token()).await?;
            let mut materialized = materialize_assistant_message(provider_stream, &model).await?;
            accumulate_usage(&mut state.usage_so_far, &materialized.usage);
            accumulate_usage(&mut turn_usage, &materialized.usage);
            debug!(
                session_id = %self.session_id,
                turn_id = %turn.id,
                model_id = %model.id,
                subagent_model_id = %subagent_model.id,
                assistant_part_count = materialized.message.parts.len(),
                stop_reason = ?materialized.message.stop_reason,
                input_tokens = materialized.usage.input_tokens,
                output_tokens = materialized.usage.output_tokens,
                "materialized assistant message"
            );

            let (deduped_parts, duplicate_tool_calls) =
                dedupe_assistant_tool_call_parts(std::mem::take(&mut materialized.message.parts));
            if duplicate_tool_calls > 0 {
                warn!(
                    session_id = %self.session_id,
                    turn_id = %turn.id,
                    duplicate_tool_call_count = duplicate_tool_calls,
                    "deduped duplicate tool calls from provider output"
                );
                materialized.message.parts = deduped_parts;
            } else {
                materialized.message.parts = deduped_parts;
            }

            let assistant_message = Message::Assistant(materialized.message.clone());
            state.messages.push(assistant_message.clone());

            // Track response ID for previous_response_id chaining.
            if let Some(ref resp_id) = materialized.response_id {
                state.last_response_id = Some(resp_id.clone());
                state.messages_seen_by_provider = state.messages.len();
            }

            for payload in materialized.events {
                self.push_event(&mut events, payload);
            }
            self.push_event(
                &mut events,
                SessionEventPayload::MessageItem {
                    message: assistant_message,
                },
            );

            let tool_calls = assistant_tool_calls(&materialized.message);
            if tool_calls.is_empty() {
                let stop_dispatch = run_stop(
                    self,
                    &fired_hook_ids,
                    HookInvocationContext {
                        turn_id: &turn.id,
                        model: &model.id,
                        working_dir: &stored.blueprint.working_dir,
                    },
                    Some(&materialized.message),
                    true,
                )
                .await?;
                track_fired_hook_ids(&mut fired_hook_ids, &stop_dispatch);
                self.record_hook_dispatch(&mut events, &stop_dispatch);
                for message in apply_hook_side_effects(&mut state, &stop_dispatch) {
                    self.push_event(&mut events, SessionEventPayload::MessageItem { message });
                }
                if let Some(reason) = stop_dispatch.merged.block_reason.clone() {
                    let continuation = Message::User(halter_protocol::UserMessage::text(
                        if reason.trim().is_empty() {
                            "Continue."
                        } else {
                            &reason
                        },
                    ));
                    state.messages.push(continuation.clone());
                    self.push_event(
                        &mut events,
                        SessionEventPayload::MessageItem {
                            message: continuation,
                        },
                    );
                    self.flush_turn_progress(
                        snapshot.clone(),
                        &mut expected_state,
                        &state,
                        &mut events,
                        live,
                    )
                    .await?;
                    continue;
                }

                info!(
                    session_id = %self.session_id,
                    turn_id = %turn.id,
                    input_tokens = turn_usage.input_tokens,
                    output_tokens = turn_usage.output_tokens,
                    "turn completed without tool calls"
                );
                events.push(self.make_event(SessionEventPayload::TurnCompleted {
                    turn_id: turn.id,
                    usage: turn_usage,
                }));
                return Ok(TurnCommit {
                    expected_state,
                    snapshot,
                    state,
                    events,
                });
            }

            info!(
                session_id = %self.session_id,
                turn_id = %turn.id,
                tool_call_count = tool_calls.len(),
                "assistant requested tool calls"
            );

            let tool_events = self
                .execute_tool_calls(
                    &stored.blueprint,
                    snapshot.clone(),
                    turn_cancel.child_token(),
                    &selected_models.default_model,
                    &selected_models.subagent_model,
                    &turn.id,
                    &mut fired_hook_ids,
                    &mut state,
                    tool_calls,
                )
                .await?;
            events.extend(tool_events);
            self.flush_turn_progress(
                snapshot.clone(),
                &mut expected_state,
                &state,
                &mut events,
                live,
            )
            .await?;
        }
    }

    #[expect(clippy::too_many_arguments)]
    async fn execute_tool_calls(
        &self,
        blueprint: &SessionBlueprint,
        snapshot: Arc<ResourceSnapshot>,
        cancel: CancellationToken,
        effective_model: &ModelId,
        effective_subagent_model: &ModelId,
        turn_id: &halter_protocol::TurnId,
        fired_hook_ids: &mut BTreeSet<String>,
        state: &mut SessionState,
        tool_calls: Vec<ToolCall>,
    ) -> anyhow::Result<Vec<PendingEvent>> {
        let mut events = Vec::new();

        let tools = self.services.tools.clone();
        for batch in
            batch_tool_calls_by_concurrency(|name| tools.concurrency_for(&name.0), tool_calls)
        {
            // Phase A: pre-hook + block check + context build (sequential per call,
            // because each mutates `state` and may change `call.arguments`).
            let mut prepared: Vec<PreparedToolCall> = Vec::with_capacity(batch.len());
            for mut call in batch {
                let pre_dispatch = run_pre_tool_use(
                    self,
                    fired_hook_ids,
                    HookInvocationContext {
                        turn_id,
                        model: effective_model,
                        working_dir: &blueprint.working_dir,
                    },
                    &call,
                )
                .await?;
                track_fired_hook_ids(fired_hook_ids, &pre_dispatch);
                self.record_hook_dispatch(&mut events, &pre_dispatch);
                for message in apply_hook_side_effects(state, &pre_dispatch) {
                    self.push_event(&mut events, SessionEventPayload::MessageItem { message });
                }
                if let Some(updated_input) = pre_dispatch.merged.updated_input.clone() {
                    call.arguments = updated_input;
                }
                info!(
                    session_id = %self.session_id,
                    tool_call_id = %call.id,
                    tool_name = %call.name,
                    "executing tool call"
                );
                self.push_event(
                    &mut events,
                    SessionEventPayload::ToolExecutionStarted { call: call.clone() },
                );

                if let Some(reason) = pre_dispatch.merged.block_reason.clone() {
                    let error = ToolError::new(reason);
                    let outcome = ToolExecutionOutcome {
                        call: call.clone(),
                        result: Err(error.clone()),
                    };
                    let message = Message::Tool(ToolResultMessage {
                        id: MessageId::new(),
                        call_id: call.id.clone(),
                        content: ToolResult::Empty,
                        error: Some(error),
                        created_at: Utc::now(),
                    });
                    state.messages.push(message.clone());
                    self.push_event(
                        &mut events,
                        SessionEventPayload::ToolExecutionCompleted { outcome },
                    );
                    self.push_event(&mut events, SessionEventPayload::MessageItem { message });
                    continue;
                }

                let (emit, tool_event_drain) = self.spawn_tool_event_sink();
                let context = halter_tools::ToolContext {
                    session_id: self.session_id.clone(),
                    working_dir: blueprint.working_dir.clone(),
                    path_locks: self.services.path_locks.clone(),
                    tool_sessions: self.services.tool_sessions.clone(),
                    snapshot: snapshot.clone(),
                    cancel: cancel.child_token(),
                    emit,
                    policy: self.services.policy.clone(),
                    shell_timeout_secs: self.services.shell_timeout_secs,
                    subagent_parent: Some(Arc::new(SubagentParentContext {
                        blueprint: blueprint.clone(),
                        state: state.clone(),
                        snapshot: snapshot.clone(),
                        subagent_model: effective_subagent_model.clone(),
                    })),
                };
                state.pending_tool_calls.insert(
                    call.id.clone(),
                    PendingToolCall {
                        call: call.clone(),
                        submitted_at: Utc::now(),
                    },
                );

                prepared.push(PreparedToolCall {
                    call,
                    context,
                    tool_event_drain,
                });
            }

            // Phase B: dispatch execution. For Exclusive or single-call batches this
            // is serial; for ReadOnly/ParallelSafe batches of >1, runs concurrently.
            let tools = self.services.tools.clone();
            let executions: Vec<anyhow::Result<ToolResult>> =
                futures::future::join_all(prepared.iter().map(|p| {
                    let tools = tools.clone();
                    let context = p.context.clone();
                    let args = p.call.arguments.clone();
                    let name = p.call.name.0.clone();
                    async move { tools.execute(&name, context, args).await }
                }))
                .await;

            // Phase C: post-hook + state mutation (sequential, original order).
            for (prep, execution) in prepared.into_iter().zip(executions) {
                let PreparedToolCall {
                    call,
                    context,
                    tool_event_drain,
                } = prep;
                drop(context);
                for payload in tool_event_drain
                    .into_events()
                    .into_iter()
                    .filter_map(|event| tool_runtime_event_payload(&call.id, event))
                {
                    self.push_event(&mut events, payload);
                }

                let (mut content, error) = match execution {
                    Ok(result) => {
                        debug!(
                            session_id = %self.session_id,
                            tool_call_id = %call.id,
                            tool_name = %call.name,
                            result_kind = tool_result_kind(&result),
                            "tool call completed"
                        );
                        (result, None)
                    }
                    Err(error) => {
                        warn!(
                            session_id = %self.session_id,
                            tool_call_id = %call.id,
                            tool_name = %call.name,
                            error = %error,
                            "tool call failed"
                        );
                        (ToolResult::Empty, Some(ToolError::new(error.to_string())))
                    }
                };
                if error.is_none() {
                    let post_dispatch = run_post_tool_use(
                        self,
                        fired_hook_ids,
                        HookInvocationContext {
                            turn_id,
                            model: effective_model,
                            working_dir: &blueprint.working_dir,
                        },
                        &call,
                        &content,
                    )
                    .await?;
                    track_fired_hook_ids(fired_hook_ids, &post_dispatch);
                    self.record_hook_dispatch(&mut events, &post_dispatch);
                    for message in apply_hook_side_effects(state, &post_dispatch) {
                        self.push_event(&mut events, SessionEventPayload::MessageItem { message });
                    }
                    if let Some(updated_output) = post_dispatch.merged.updated_output {
                        content = tool_result_from_hook_value(updated_output);
                    }
                } else if let Some(tool_error) = error.as_ref() {
                    let post_dispatch = run_post_tool_use_failure(
                        self,
                        fired_hook_ids,
                        HookInvocationContext {
                            turn_id,
                            model: effective_model,
                            working_dir: &blueprint.working_dir,
                        },
                        &call,
                        tool_error,
                    )
                    .await?;
                    track_fired_hook_ids(fired_hook_ids, &post_dispatch);
                    self.record_hook_dispatch(&mut events, &post_dispatch);
                    for message in apply_hook_side_effects(state, &post_dispatch) {
                        self.push_event(&mut events, SessionEventPayload::MessageItem { message });
                    }
                }
                let outcome = ToolExecutionOutcome {
                    call: call.clone(),
                    result: error.clone().map_or_else(|| Ok(content.clone()), Err),
                };
                let message = Message::Tool(ToolResultMessage {
                    id: MessageId::new(),
                    call_id: call.id.clone(),
                    content,
                    error,
                    created_at: Utc::now(),
                });

                state.pending_tool_calls.shift_remove(&call.id);
                state.messages.push(message.clone());
                self.push_event(
                    &mut events,
                    SessionEventPayload::ToolExecutionCompleted { outcome },
                );
                self.push_event(&mut events, SessionEventPayload::MessageItem { message });
            }
        }

        Ok(events)
    }

    async fn commit_and_publish(
        &self,
        snapshot: Option<Arc<ResourceSnapshot>>,
        expected_state: Option<SessionState>,
        state: Option<SessionState>,
        events: Vec<PendingEvent>,
        live: Option<&LiveTurnStream>,
    ) -> anyhow::Result<Vec<SessionEvent>> {
        debug!(
            session_id = %self.session_id,
            event_count = events.len(),
            replace_snapshot = snapshot.is_some(),
            check_expected_state = expected_state.is_some(),
            replace_state = state.is_some(),
            "committing session events"
        );
        let committed = self
            .services
            .sessions
            .commit(&self.session_id, snapshot, expected_state, state, events)
            .await?;
        let forwarding_ancestors =
            forwarding_ancestors_for_session(&self.services, &self.session_id).await;
        // Single commit-then-publish point. Events fan out to both sinks, but
        // only *after* the store has assigned monotonic sequences; there is no
        // pre-commit emission path that could race with the real sequence or
        // deliver sentinel-sequenced events.
        for event in &committed {
            if let Some(live) = live {
                live.emit_committed(event.clone());
            }
            self.services
                .parent_streams
                .forward_to_ancestors(&forwarding_ancestors, event);
            self.services.event_bus.publish(event.clone());
            if let Some(recorder) = &self.services.trace_recorder {
                recorder.record(event);
            }
        }
        Ok(committed)
    }

    async fn flush_turn_progress(
        &self,
        snapshot: Arc<ResourceSnapshot>,
        expected_state: &mut SessionState,
        state: &SessionState,
        events: &mut Vec<PendingEvent>,
        live: &LiveTurnStream,
    ) -> anyhow::Result<()> {
        if events.is_empty() && state == expected_state {
            return Ok(());
        }

        self.commit_and_publish(
            Some(snapshot),
            Some(expected_state.clone()),
            Some(state.clone()),
            events.clone(),
            Some(live),
        )
        .await?;
        events.clear();
        *expected_state = state.clone();
        Ok(())
    }

    async fn commit_turn_failure(
        &self,
        failure_events: Vec<PendingEvent>,
        live: &LiveTurnStream,
    ) -> anyhow::Result<()> {
        let stored = self
            .services
            .sessions
            .load_session(&self.session_id)
            .await?
            .with_context(|| {
                format!(
                    "failed to commit failed turn: unknown session '{}'",
                    self.session_id.0
                )
            })?;
        self.commit_and_publish(None, Some(stored.state), None, failure_events, Some(live))
            .await?;
        Ok(())
    }

    fn spawn_tool_event_sink(&self) -> (Arc<dyn ToolEventSink>, ToolEventDrain) {
        let buffer = Arc::new(Mutex::new(ToolEventBuffer::default()));
        (
            Arc::new(SessionToolEventSink {
                buffer: buffer.clone(),
            }) as Arc<dyn ToolEventSink>,
            ToolEventDrain { buffer },
        )
    }

    fn push_event(&self, events: &mut Vec<PendingEvent>, payload: SessionEventPayload) {
        events.push(self.make_event(payload));
    }

    fn record_hook_dispatch(
        &self,
        events: &mut Vec<PendingEvent>,
        dispatch: &ExecutedHookDispatch,
    ) {
        for run in &dispatch.preview_runs {
            self.push_event(
                events,
                SessionEventPayload::HookStarted { run: run.clone() },
            );
        }
        for run in &dispatch.completed_runs {
            self.push_event(
                events,
                SessionEventPayload::HookCompleted { run: run.clone() },
            );
        }
    }

    fn make_event(&self, payload: SessionEventPayload) -> PendingEvent {
        let pending = PendingEvent::new(self.session_id.clone(), Delivery::Lossless, payload);
        // Mirror every event into the trace as soon as it's generated so that
        // long-running turns (many tool-call iterations under a single
        // `commit_and_publish`) still produce live trace output. The
        // committed counterpart arrives later via `record`.
        if let Some(recorder) = &self.services.trace_recorder {
            recorder.record_pending(&pending);
        }
        pending
    }
}

fn tool_runtime_event_payload(
    call_id: &halter_protocol::ToolCallId,
    event: ToolRuntimeEvent,
) -> Option<SessionEventPayload> {
    match event {
        ToolRuntimeEvent::ToolOutput { tool_name, chunk } => {
            Some(SessionEventPayload::ToolOutput {
                call_id: call_id.clone(),
                tool_name: tool_name.into(),
                chunk,
            })
        }
        ToolRuntimeEvent::Started { .. } | ToolRuntimeEvent::Completed { .. } => None,
    }
}

fn tool_runtime_event_tool_name(event: &ToolRuntimeEvent) -> &str {
    match event {
        ToolRuntimeEvent::Started { tool_name }
        | ToolRuntimeEvent::Completed { tool_name }
        | ToolRuntimeEvent::ToolOutput { tool_name, .. } => tool_name,
    }
}

fn tool_runtime_event_bytes(event: &ToolRuntimeEvent) -> usize {
    match event {
        ToolRuntimeEvent::Started { tool_name } | ToolRuntimeEvent::Completed { tool_name } => {
            tool_name.len()
        }
        ToolRuntimeEvent::ToolOutput { tool_name, chunk } => tool_name.len() + chunk.len(),
    }
}

#[derive(Debug)]
struct TurnCommit {
    expected_state: SessionState,
    snapshot: Arc<ResourceSnapshot>,
    state: SessionState,
    events: Vec<PendingEvent>,
}

#[derive(Debug, Default)]
struct PendingThinkingBlock {
    text: String,
    signature: Option<String>,
}

#[derive(Debug)]
struct PendingToolCallBlock {
    tool_call_id: halter_protocol::ToolCallId,
    name: halter_protocol::ToolName,
    arguments: String,
}

#[derive(Debug)]
pub(crate) struct MaterializedAssistantMessage {
    pub(crate) message: AssistantMessage,
    pub(crate) usage: Usage,
    pub(crate) events: Vec<SessionEventPayload>,
    /// Provider's response ID for `previous_response_id` chaining.
    pub(crate) response_id: Option<String>,
}

pub(crate) async fn materialize_assistant_message(
    mut provider_stream: BoxStream<'static, Result<StreamEvent, ProviderError>>,
    model: &halter_protocol::ResolvedModel,
) -> anyhow::Result<MaterializedAssistantMessage> {
    debug!(provider = %model.provider, model = %model.model, "materializing provider stream");
    let mut message_id = MessageId::new();
    let mut usage = Usage::default();
    let mut stop_reason = StopReason::EndTurn;
    let mut parts = Vec::new();
    let mut text_buffer = String::new();
    let mut delta_events = Vec::new();
    let mut accumulated_output_bytes = 0usize;
    let mut thinking_block: Option<PendingThinkingBlock> = None;
    let mut tool_call_blocks: std::collections::BTreeMap<BlockId, PendingToolCallBlock> =
        std::collections::BTreeMap::new();
    let mut captured_response_id: Option<String> = None;

    while let Some(item) = provider_stream.next().await {
        match item {
            Ok(StreamEvent::MessageStart { id }) => {
                message_id = id;
            }
            Ok(StreamEvent::TextStart { .. }) => {}
            Ok(StreamEvent::TextDelta { delta, .. }) => {
                append_provider_stream_chunk(
                    "text",
                    &mut text_buffer,
                    &delta,
                    &mut accumulated_output_bytes,
                )?;
                if delta_events.len() >= PROVIDER_STREAM_EVENT_CAP {
                    anyhow::bail!(
                        "provider stream text exceeded event cap: {} events",
                        PROVIDER_STREAM_EVENT_CAP
                    );
                }
                delta_events.push(SessionEventPayload::DeltaItem {
                    delta: halter_protocol::DeltaItem { text: delta },
                });
            }
            Ok(StreamEvent::TextEnd { .. }) => {
                flush_text_buffer(&mut parts, &mut text_buffer);
            }
            Ok(StreamEvent::ThinkingStart { .. }) => {
                thinking_block = Some(PendingThinkingBlock::default());
            }
            Ok(StreamEvent::ThinkingDelta { delta, .. }) => {
                let thinking = thinking_block.get_or_insert_with(PendingThinkingBlock::default);
                append_provider_stream_chunk(
                    "thinking",
                    &mut thinking.text,
                    &delta,
                    &mut accumulated_output_bytes,
                )?;
            }
            Ok(StreamEvent::ThinkingEnd { signature, .. }) => {
                if let Some(mut thinking) = thinking_block.take() {
                    thinking.signature = signature;
                    parts.push(AssistantPart::Thinking(halter_protocol::ThinkingBlock {
                        text: thinking.text,
                        signature: thinking.signature,
                    }));
                }
            }
            Ok(StreamEvent::ToolCallStart {
                id,
                tool_call_id,
                name,
            }) => {
                flush_text_buffer(&mut parts, &mut text_buffer);
                tool_call_blocks.insert(
                    id,
                    PendingToolCallBlock {
                        tool_call_id,
                        name,
                        arguments: String::new(),
                    },
                );
            }
            Ok(StreamEvent::ToolArgsDelta { id, delta }) => {
                let pending = tool_call_blocks.get_mut(&id).with_context(|| {
                    format!("failed to materialize tool call: missing block '{}'", id)
                })?;
                append_provider_stream_chunk(
                    "tool arguments",
                    &mut pending.arguments,
                    &delta,
                    &mut accumulated_output_bytes,
                )?;
            }
            Ok(StreamEvent::ToolCallEnd { id }) => {
                let pending = tool_call_blocks.remove(&id).with_context(|| {
                    format!("failed to materialize tool call: missing block '{}'", id)
                })?;
                let arguments = parse_tool_call_arguments(&pending.arguments)?;
                parts.push(AssistantPart::ToolCall(ToolCall {
                    id: pending.tool_call_id,
                    name: pending.name,
                    arguments,
                }));
            }
            Ok(StreamEvent::UsageUpdate { usage: updated }) => {
                usage = updated;
            }
            Ok(StreamEvent::MessageEnd {
                stop_reason: ended_reason,
                response_id: resp_id,
                ..
            }) => {
                stop_reason = ended_reason;
                if resp_id.is_some() {
                    captured_response_id = resp_id;
                }
            }
            Ok(StreamEvent::ProviderWarning { message }) => {
                warn!(provider = %model.provider, message = %message, "provider emitted warning");
            }
            Ok(StreamEvent::Error { error }) | Err(error) => {
                error!(provider = %model.provider, error = %error.message, "provider stream failed");
                return Err(anyhow::Error::new(error));
            }
        }
    }

    flush_text_buffer(&mut parts, &mut text_buffer);
    // Unterminated tool call blocks are recoverable: the upstream stream
    // ended without a `ToolCallEnd` (a misbehaving provider, a truncated
    // response, or an [DONE] frame that arrived before the codec closed
    // every output item). Treat each pending block as if a `ToolCallEnd`
    // had been received and push it as a synthetic `ToolCall` part using
    // the arguments we did accumulate. If those arguments fail to parse
    // (e.g. truncated mid-JSON), substitute `{}` so the tool runtime can
    // surface a tool-side error to the model rather than dropping the
    // entire turn. Previously this branch bailed with
    // "unterminated tool call block", which terminated the whole session
    // even though a single bad streaming event need not be fatal.
    for (block_id, pending) in std::mem::take(&mut tool_call_blocks) {
        let arguments = match parse_tool_call_arguments(&pending.arguments) {
            Ok(value) => value,
            Err(error) => {
                warn!(
                    provider = %model.provider,
                    model = %model.model,
                    tool_call_id = %pending.tool_call_id,
                    block_id = %block_id,
                    raw_arguments = %pending.arguments,
                    %error,
                    "stream ended with unterminated tool call whose arguments failed to parse; substituting empty object"
                );
                serde_json::json!({})
            }
        };
        warn!(
            provider = %model.provider,
            model = %model.model,
            tool_call_id = %pending.tool_call_id,
            block_id = %block_id,
            tool_name = %pending.name,
            "stream ended without ToolCallEnd; auto-closing tool call block"
        );
        parts.push(AssistantPart::ToolCall(ToolCall {
            id: pending.tool_call_id,
            name: pending.name,
            arguments,
        }));
    }
    debug!(
        provider = %model.provider,
        model = %model.model,
        message_id = %message_id,
        part_count = parts.len(),
        stop_reason = ?stop_reason,
        "finished materializing provider stream"
    );

    Ok(MaterializedAssistantMessage {
        message: AssistantMessage {
            id: message_id,
            created_at: Utc::now(),
            parts,
            stop_reason: Some(stop_reason),
            usage: Some(usage.clone()),
            replay_meta: ReplayMeta {
                provider_name: Some(model.provider.clone()),
                model: Some(model.id.clone()),
            },
        },
        usage,
        events: delta_events,
        response_id: captured_response_id,
    })
}

fn flush_text_buffer(parts: &mut Vec<AssistantPart>, text_buffer: &mut String) {
    if text_buffer.is_empty() {
        return;
    }

    parts.push(AssistantPart::Text {
        text: std::mem::take(text_buffer),
    });
}

fn parse_tool_call_arguments(arguments: &str) -> anyhow::Result<serde_json::Value> {
    if arguments.trim().is_empty() {
        return Ok(serde_json::json!({}));
    }

    serde_json::from_str(arguments)
        .with_context(|| "failed to materialize tool call: invalid json arguments")
}

fn append_provider_stream_chunk(
    label: &str,
    target: &mut String,
    chunk: &str,
    accumulated_output_bytes: &mut usize,
) -> anyhow::Result<()> {
    let observed = accumulated_output_bytes.saturating_add(chunk.len());
    if observed > PROVIDER_STREAM_OUTPUT_CAP_BYTES {
        anyhow::bail!(
            "provider stream {label} exceeded output cap: {observed} bytes (cap {PROVIDER_STREAM_OUTPUT_CAP_BYTES})"
        );
    }
    target.push_str(chunk);
    *accumulated_output_bytes = observed;
    Ok(())
}

fn assistant_tool_calls(message: &AssistantMessage) -> Vec<ToolCall> {
    message
        .parts
        .iter()
        .filter_map(|part| match part {
            AssistantPart::ToolCall(call) => Some(call.clone()),
            AssistantPart::Text { .. } | AssistantPart::Thinking(_) => None,
        })
        .collect()
}

fn dedupe_assistant_tool_call_parts(parts: Vec<AssistantPart>) -> (Vec<AssistantPart>, usize) {
    let mut deduped = Vec::with_capacity(parts.len());
    let mut seen_tool_call_ids = BTreeSet::new();
    let mut duplicate_count = 0;

    for part in parts {
        match &part {
            AssistantPart::ToolCall(call) => {
                if !seen_tool_call_ids.insert(call.id.clone()) {
                    duplicate_count += 1;
                    continue;
                }
            }
            AssistantPart::Text { .. } | AssistantPart::Thinking(_) => {}
        }
        deduped.push(part);
    }

    (deduped, duplicate_count)
}

fn ensure_provider_iteration_allowed(
    max_turns: Option<u32>,
    completed_iterations: u32,
) -> anyhow::Result<()> {
    if let Some(max_turns) = max_turns
        && completed_iterations >= max_turns
    {
        anyhow::bail!(
            "failed to run turn: max_turns {max_turns} exhausted before provider iteration {}",
            completed_iterations.saturating_add(1)
        );
    }
    Ok(())
}

fn accumulate_usage(total: &mut Usage, delta: &Usage) {
    total.input_tokens += delta.input_tokens;
    total.output_tokens += delta.output_tokens;
    total.cache_creation_input_tokens += delta.cache_creation_input_tokens;
    total.cache_read_input_tokens += delta.cache_read_input_tokens;
}

fn tool_result_kind(result: &ToolResult) -> &'static str {
    match result {
        ToolResult::Empty => "empty",
        ToolResult::Text { .. } => "text",
        ToolResult::Json { .. } => "json",
    }
}

pub(crate) fn apply_hook_side_effects(
    state: &mut SessionState,
    dispatch: &ExecutedHookDispatch,
) -> Vec<Message> {
    for fired_hook_id in &dispatch.fired_hook_ids {
        if !state
            .fired_hook_ids
            .iter()
            .any(|seen| seen == fired_hook_id)
        {
            state.fired_hook_ids.push(fired_hook_id.clone());
        }
    }

    for context in &dispatch.merged.additional_context {
        state
            .appended_prompt_segments
            .push(build_hook_prompt_segment(context));
    }

    let mut messages = Vec::new();
    for text in &dispatch.merged.system_messages {
        let message = Message::System(SystemMessage {
            id: MessageId::new(),
            created_at: Utc::now(),
            text: text.clone(),
        });
        state.messages.push(message.clone());
        messages.push(message);
    }

    messages
}

fn build_hook_prompt_segment(text: &str) -> PromptSegment {
    PromptSegment {
        id: PromptSegmentId::new(),
        text: text.to_owned(),
        volatility: Volatility::TurnDynamic,
        cache_scope: CacheScope::Dynamic,
        content_hash: hash_text(text),
        kind: PromptSegmentKind::Append,
    }
}

fn hash_text(text: &str) -> ContentHash {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn tool_result_from_hook_value(value: serde_json::Value) -> ToolResult {
    match value {
        serde_json::Value::Null => ToolResult::Empty,
        serde_json::Value::String(text) => ToolResult::Text { text },
        other => ToolResult::Json { value: other },
    }
}

async fn forwarding_ancestors_for_session(
    services: &Arc<RuntimeServices>,
    session_id: &SessionId,
) -> Vec<SessionId> {
    let stored = match services.sessions.load_session(session_id).await {
        Ok(Some(stored)) => stored,
        Ok(None) => return Vec::new(),
        Err(error) => {
            warn!(
                session_id = %session_id,
                error = %error,
                "failed to load session for subagent event forwarding"
            );
            return Vec::new();
        }
    };

    forwarding_ancestors_for_blueprint(services, &stored.blueprint).await
}

async fn forwarding_ancestors_for_blueprint(
    services: &Arc<RuntimeServices>,
    blueprint: &SessionBlueprint,
) -> Vec<SessionId> {
    if !blueprint.subagent_event_forwarding.is_enabled() {
        return Vec::new();
    }

    let mut ancestors = Vec::new();
    let mut seen = BTreeSet::new();
    let mut next = blueprint.parent_session_id.clone();
    while let Some(session_id) = next {
        if !seen.insert(session_id.clone()) {
            warn!(
                session_id = %blueprint.session_id,
                ancestor_session_id = %session_id,
                "stopped subagent event forwarding ancestor walk at cycle"
            );
            break;
        }

        ancestors.push(session_id.clone());
        next = match services.sessions.load_session(&session_id).await {
            Ok(Some(stored)) => stored.blueprint.parent_session_id,
            Ok(None) => {
                warn!(
                    session_id = %blueprint.session_id,
                    ancestor_session_id = %session_id,
                    "stopped subagent event forwarding ancestor walk at missing parent"
                );
                break;
            }
            Err(error) => {
                warn!(
                    session_id = %blueprint.session_id,
                    ancestor_session_id = %session_id,
                    error = %error,
                    "stopped subagent event forwarding ancestor walk after load failure"
                );
                break;
            }
        };
    }

    ancestors
}

pub(crate) async fn create_session_seeded(
    services: Arc<RuntimeServices>,
    init: SessionInit,
    mut initial_state: SessionState,
    snapshot: Arc<ResourceSnapshot>,
) -> anyhow::Result<HalterSession> {
    let default_registry_model = services.models.default_model()?;
    let subagent_registry_model = services.models.subagent_model()?;
    let selected_models = select_models(
        &default_registry_model.id,
        &subagent_registry_model.id,
        init.default_model.as_ref(),
        init.subagent_model.as_ref(),
    );
    services.models.model(&selected_models.default_model)?;
    services.models.model(&selected_models.subagent_model)?;
    let session_id = init.session_id.unwrap_or_default();
    let subagent_event_forwarding = init
        .subagent_event_forwarding
        .unwrap_or(services.subagent_event_forwarding);
    let blueprint = SessionBlueprint {
        session_id: session_id.clone(),
        parent_session_id: init.parent_session_id,
        default_model: selected_models.default_model,
        subagent_model: selected_models.subagent_model,
        subagent_event_forwarding,
        snapshot_revision: snapshot.revision.clone(),
        working_dir: init.working_dir,
        system_prompt_seed: init.system_prompt_seed,
        max_turns: init.max_turns,
        subagent_depth: init.subagent_depth,
    };
    info!(
        session_id = %session_id,
        default_model = %blueprint.default_model,
        subagent_model = %blueprint.subagent_model,
        subagent_event_forwarding = ?blueprint.subagent_event_forwarding,
        working_dir = %blueprint.working_dir.display(),
        snapshot_revision = %blueprint.snapshot_revision,
        "created session blueprint"
    );

    if initial_state.pending_session_start_source.is_none() {
        initial_state.pending_session_start_source = Some(HookSessionStartSource::Startup);
    }
    if initial_state.pending_warning_messages.is_empty() {
        initial_state.pending_warning_messages =
            services.resources.hook_warnings().as_ref().clone();
    }

    services
        .sessions
        .create_session(StoredSession {
            blueprint: blueprint.clone(),
            state: initial_state,
            snapshot,
        })
        .await?;

    if let Some(recorder) = &services.trace_recorder {
        recorder.open_session(
            &session_id,
            blueprint.parent_session_id.as_ref(),
            &blueprint,
        )?;
    }

    let started = PendingEvent::new(
        session_id.clone(),
        Delivery::Lossless,
        SessionEventPayload::SessionStarted,
    );
    let committed = services
        .sessions
        .commit(&session_id, None, None, None, vec![started])
        .await?;
    let forwarding_ancestors = forwarding_ancestors_for_blueprint(&services, &blueprint).await;
    for event in committed {
        if let Some(recorder) = &services.trace_recorder {
            recorder.record(&event);
        }
        services
            .parent_streams
            .forward_to_ancestors(&forwarding_ancestors, &event);
        services.event_bus.publish(event);
    }

    HalterSession::new(services, session_id)
}

fn lookup_or_create_session_hooks(
    services: &Arc<RuntimeServices>,
    session_id: &SessionId,
) -> anyhow::Result<Arc<Hooks>> {
    let hooks = Arc::new(services.registered_hooks.instantiate()?);
    match services.session_hook_store.lock() {
        Ok(mut store) => {
            if let Some(existing) = store.get(session_id) {
                return Ok(existing.clone());
            }
            store.insert(session_id.clone(), hooks.clone());
            Ok(hooks)
        }
        Err(_) => {
            error!(
                session_id = %session_id,
                "session hook store lock poisoned; rebuilding uncached session hooks"
            );
            Ok(hooks)
        }
    }
}

fn evict_session_hooks(services: &Arc<RuntimeServices>, session_id: &SessionId) {
    match services.session_hook_store.lock() {
        Ok(mut store) => {
            store.remove(session_id);
        }
        Err(_) => {
            error!(
                session_id = %session_id,
                "session hook store lock poisoned; skipping session hook eviction"
            );
        }
    }
}

fn format_hook_warning(warning: &HookWarning) -> String {
    let mut prefix = String::new();
    if let Some(plugin_name) = warning.plugin_name.as_deref() {
        prefix.push_str(&format!("plugin '{plugin_name}' "));
    }
    prefix.push_str("hook warning");
    if !warning.category.trim().is_empty() {
        prefix.push_str(&format!(" [{}]", warning.category));
    }
    if let Some(source_path) = warning.source_path.as_ref() {
        prefix.push_str(&format!(" at {}", source_path.display()));
    }
    format!("{prefix}: {}", warning.message)
}

fn observe_state(working_dir: PathBuf) -> ObservedState {
    let (git_branch, git_dirty) = probe_git(&working_dir);
    ObservedState {
        cwd: working_dir,
        git_branch,
        git_dirty,
        now_utc: Utc::now(),
        env_facts: Default::default(),
    }
}

struct PreparedToolCall {
    call: ToolCall,
    context: halter_tools::ToolContext,
    tool_event_drain: ToolEventDrain,
}

/// Partition `tool_calls` into concurrency-compatible batches per the
/// `ToolConcurrency` protocol variant declared for each tool:
///
/// - `Exclusive` tools are emitted in batches of size 1.
/// - Runs of `ReadOnly` and `ParallelSafe` tools are grouped into one batch
///   and may execute concurrently.
///
/// Concurrency is resolved via a caller-supplied closure. Tools the resolver
/// cannot classify are treated as `Exclusive` (safe default). Within a batch,
/// original order is preserved so hook and event sequencing remain
/// deterministic.
fn batch_tool_calls_by_concurrency(
    mut resolve: impl FnMut(&halter_protocol::ToolName) -> Option<halter_protocol::ToolConcurrency>,
    tool_calls: Vec<ToolCall>,
) -> Vec<Vec<ToolCall>> {
    use halter_protocol::ToolConcurrency;

    let mut concurrency_of = |call: &ToolCall| -> ToolConcurrency {
        resolve(&call.name).unwrap_or(ToolConcurrency::Exclusive)
    };

    let mut batches: Vec<Vec<ToolCall>> = Vec::new();
    let mut current: Vec<ToolCall> = Vec::new();

    for call in tool_calls {
        let concurrency = concurrency_of(&call);
        if matches!(concurrency, ToolConcurrency::Exclusive) {
            if !current.is_empty() {
                batches.push(std::mem::take(&mut current));
            }
            batches.push(vec![call]);
        } else {
            current.push(call);
        }
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

/// Probe the git branch and dirty flag for `working_dir`.
///
/// Returns `(None, None)` when the directory is not a git working tree or
/// `git` is not available. A detached HEAD returns the abbreviated commit as
/// the branch name. `git_dirty` is `Some(false)` for a clean tree, `Some(true)`
/// when any tracked or untracked change is present.
fn probe_git(working_dir: &std::path::Path) -> (Option<String>, Option<bool>) {
    use std::process::Command;

    let branch_output = Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .current_dir(working_dir)
        .output();
    let branch = match branch_output {
        Ok(out) if out.status.success() => {
            let name = String::from_utf8_lossy(&out.stdout).trim().to_owned();
            if name.is_empty() {
                return (None, None);
            }
            if name == "HEAD" {
                // detached HEAD — resolve to short commit
                Command::new("git")
                    .args(["rev-parse", "--short", "HEAD"])
                    .current_dir(working_dir)
                    .output()
                    .ok()
                    .filter(|out| out.status.success())
                    .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned())
                    .filter(|s| !s.is_empty())
            } else {
                Some(name)
            }
        }
        _ => return (None, None),
    };

    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(working_dir)
        .output()
        .ok()
        .filter(|out| out.status.success())
        .map(|out| !out.stdout.is_empty());

    (branch, dirty)
}

#[cfg(test)]
impl Default for RuntimeServices {
    fn default() -> Self {
        let snapshot = ResourceSnapshot::empty();
        Self {
            resources: Arc::new(ResourceHandle::new(
                snapshot,
                Arc::new(Hooks::default()),
                Vec::new(),
            )),
            registered_hooks: Arc::new(RegisteredHooks::default()),
            session_hook_store: Arc::new(Mutex::new(HashMap::new())),
            models: Arc::new(ModelRegistry::new()),
            tools: Arc::new(ToolRuntime::new()),
            path_locks: Arc::new(PathLockMap::default()),
            tool_sessions: Arc::new(ToolSessionStore::default()),
            sessions: Arc::new(halter_session::InMemorySessionStore::default()),
            policy: Arc::new(halter_tools::DefaultToolPolicy::new(Default::default())),
            prompt_assembler: Arc::new(crate::DefaultPromptAssembler),
            context_manager: Arc::new(DefaultContextManager::default()),
            event_bus: Arc::new(EventBus::default()),
            parent_streams: Arc::new(ParentStreamRegistry::default()),
            turn_registry: Arc::new(TurnRegistry::new()),
            subagent_event_forwarding: SubagentEventForwarding::Off,
            subagent_event_forwarding_cap: 100_000,
            shell_timeout_secs: 30,
            trace_recorder: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use async_trait::async_trait;
    use futures::stream::{self, BoxStream};
    use futures::{StreamExt, TryStreamExt};
    use halter_hooks::{
        Hook, HookEventName, HookResponse, HooksFile, RegisteredHookPriority, RegisteredHooks,
    };
    use halter_protocol::{
        ApiKind, BlockId, HookHandlerType, HookRunStatus, HookSessionStartSource, Message, ModelId,
        ModelRole, PluginId, ProviderCapabilities, ProviderError, ProviderKind, ProviderName,
        ProviderRequest, ResolvedModel, StopReason, StreamEvent, ToolCallId, ToolCapabilities,
        ToolConcurrency, ToolName, ToolResult, ToolSpec, Turn,
    };
    use halter_providers::{FakeProvider, Provider};
    use halter_tools::{
        DefaultToolPolicy, PolicySettings, Tool, ToolContext, register_builtin_tools,
        register_subagent_tools,
    };
    use serde_json::json;
    use tokio::sync::Notify;

    use super::*;
    use test_support::{
        configured_services, empty_hooks, install_file_hooks, new_session, resolved_test_model,
    };

    #[test]
    fn session_init_default_uses_embedded_system_prompt_seed() {
        let init = SessionInit::default();

        assert_eq!(init.system_prompt_seed.len(), 1);
        assert_eq!(
            init.system_prompt_seed[0].text,
            crate::prompt::default_system_prompt_text()
        );
    }

    /// Regression: dropping a short-lived `HalterSession` constructed for an
    /// already-live session must not close that session's trace writer.
    /// Previously, `EvictionGuard::drop` invoked `recorder.close_session(...)`,
    /// which removed the writer entry. Subagent hook dispatch builds
    /// temporary parent handles via `HalterSession::new`; when those
    /// temporaries dropped, the parent's trace was silently torn down and
    /// every subsequent event for that root session id was dropped on the
    /// floor.
    #[test]
    fn dropping_temporary_session_handle_does_not_close_trace_writer() {
        use halter_protocol::{
            Delivery, ModelId, PendingEvent, Revision, SessionBlueprint, SessionEventPayload,
            SessionId, SubagentEventForwarding,
        };

        let temp = tempfile::tempdir().expect("tempdir");
        let recorder =
            Arc::new(crate::TraceRecorder::open(temp.path().to_path_buf()).expect("recorder"));
        let services = RuntimeServices {
            trace_recorder: Some(recorder.clone()),
            ..RuntimeServices::default()
        };
        let services = Arc::new(services);

        let session_id = SessionId::from("regression-trace");
        let blueprint = SessionBlueprint {
            session_id: session_id.clone(),
            parent_session_id: None,
            default_model: ModelId::from("default"),
            subagent_model: ModelId::from("subagent"),
            subagent_event_forwarding: SubagentEventForwarding::Off,
            snapshot_revision: Revision::from("rev-1".to_owned()),
            working_dir: temp.path().to_path_buf(),
            system_prompt_seed: Vec::new(),
            max_turns: None,
            subagent_depth: 0,
        };
        recorder
            .open_session(&session_id, None, &blueprint)
            .expect("open session");

        // Construct two independent `HalterSession` handles for the same
        // session id — the same shape produced by subagent hook dispatch
        // (parent gets a primary handle, the start-hook code path then
        // builds a temporary handle for the same parent session id).
        let primary =
            HalterSession::new(services.clone(), session_id.clone()).expect("primary handle");
        {
            let _temporary =
                HalterSession::new(services.clone(), session_id.clone()).expect("temporary handle");
        } // drop the temporary here

        // Recording must still land in the trace file: the primary handle
        // is still alive, and the recorder must not have been closed by
        // the temporary's drop.
        let pending = PendingEvent::new(
            session_id.clone(),
            Delivery::Lossless,
            SessionEventPayload::Warning {
                message: "post-drop".to_owned(),
            },
        );
        recorder.record(&pending.into_committed(1));
        // Also keep the primary live until after the recording write.
        drop(primary);

        let path = temp.path().join(format!("{}.txt", session_id.0));
        let contents = std::fs::read_to_string(&path).expect("trace contents");
        assert!(
            contents.contains("post-drop"),
            "trace did not capture post-drop event:\n{contents}"
        );
    }

    /// Regression: a provider stream that ends without `ToolCallEnd` for an
    /// in-flight tool call must not abort the whole turn. We synthesize a
    /// `ToolCall` part using the accumulated arguments (or `{}` when those
    /// arguments are not valid JSON) and let the tool runtime surface any
    /// resulting tool-side error to the model. Previously this branch
    /// `bail!`ed with "unterminated tool call block", which terminated the
    /// session even though one truncated stream is recoverable.
    #[tokio::test]
    async fn materialize_handles_unterminated_tool_call_block() {
        let model = ResolvedModel {
            role: ModelRole::default(),
            id: ModelId::from("default"),
            provider: ProviderName::from("fake"),
            provider_kind: ProviderKind::Fake,
            api_kind: ApiKind::Fake,
            model: "halter/fake".to_owned(),
            max_input_tokens: Some(32_000),
            max_output_tokens: Some(4_096),
            reasoning: None,
            tokens_per_minute: None,
        };
        let message_id = halter_protocol::MessageId::new();
        let block_id = BlockId::new();
        let tool_call_id = ToolCallId::from("call-truncated");
        // Stream emits ToolCallStart + a partial argument delta, then
        // MessageEnd, then EOF — no ToolCallEnd.
        let stream: BoxStream<'static, Result<StreamEvent, ProviderError>> = stream::iter(vec![
            Ok(StreamEvent::MessageStart {
                id: message_id.clone(),
            }),
            Ok(StreamEvent::ToolCallStart {
                id: block_id.clone(),
                tool_call_id: tool_call_id.clone(),
                name: ToolName::from("write"),
            }),
            Ok(StreamEvent::ToolArgsDelta {
                id: block_id.clone(),
                delta: r#"{"path":"x.txt","content":"hi"}"#.to_owned(),
            }),
            Ok(StreamEvent::MessageEnd {
                id: message_id.clone(),
                stop_reason: StopReason::ToolUse,
                response_id: None,
            }),
        ])
        .boxed();

        let materialized = super::materialize_assistant_message(stream, &model)
            .await
            .expect("materialize must recover from unterminated tool call");
        let tool_calls: Vec<_> = materialized
            .message
            .parts
            .iter()
            .filter_map(|part| match part {
                AssistantPart::ToolCall(call) => Some(call.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(tool_calls.len(), 1, "one synthetic tool call expected");
        assert_eq!(tool_calls[0].id, tool_call_id);
        assert_eq!(tool_calls[0].name.0, "write");
        assert_eq!(
            tool_calls[0].arguments,
            serde_json::json!({"path": "x.txt", "content": "hi"})
        );
    }

    /// Same recovery path but the accumulated arguments are not valid JSON
    /// (truncated mid-string). The synthetic tool call carries `{}` so the
    /// tool runtime can produce a structured error rather than the whole
    /// turn aborting.
    #[tokio::test]
    async fn materialize_handles_unterminated_tool_call_with_invalid_json() {
        let model = ResolvedModel {
            role: ModelRole::default(),
            id: ModelId::from("default"),
            provider: ProviderName::from("fake"),
            provider_kind: ProviderKind::Fake,
            api_kind: ApiKind::Fake,
            model: "halter/fake".to_owned(),
            max_input_tokens: Some(32_000),
            max_output_tokens: Some(4_096),
            reasoning: None,
            tokens_per_minute: None,
        };
        let message_id = halter_protocol::MessageId::new();
        let block_id = BlockId::new();
        let tool_call_id = ToolCallId::from("call-bad-json");
        let stream: BoxStream<'static, Result<StreamEvent, ProviderError>> = stream::iter(vec![
            Ok(StreamEvent::MessageStart {
                id: message_id.clone(),
            }),
            Ok(StreamEvent::ToolCallStart {
                id: block_id.clone(),
                tool_call_id: tool_call_id.clone(),
                name: ToolName::from("write"),
            }),
            Ok(StreamEvent::ToolArgsDelta {
                id: block_id.clone(),
                // Truncated mid-string: serde_json::from_str will fail.
                delta: r#"{"path":"x.txt","content":"hel"#.to_owned(),
            }),
            Ok(StreamEvent::MessageEnd {
                id: message_id.clone(),
                stop_reason: StopReason::ToolUse,
                response_id: None,
            }),
        ])
        .boxed();

        let materialized = super::materialize_assistant_message(stream, &model)
            .await
            .expect("materialize must recover even with unparsable arguments");
        let tool_calls: Vec<_> = materialized
            .message
            .parts
            .iter()
            .filter_map(|part| match part {
                AssistantPart::ToolCall(call) => Some(call.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(tool_calls.len(), 1);
        assert_eq!(tool_calls[0].id, tool_call_id);
        assert_eq!(tool_calls[0].arguments, serde_json::json!({}));
    }

    #[tokio::test]
    async fn materialize_rejects_oversized_provider_text() {
        let model = resolved_test_model("default", "fake", "halter/fake");
        let block_id = BlockId::new();
        let oversized = "x".repeat(PROVIDER_STREAM_OUTPUT_CAP_BYTES + 1);
        let stream: BoxStream<'static, Result<StreamEvent, ProviderError>> = stream::iter(vec![
            Ok(StreamEvent::MessageStart {
                id: halter_protocol::MessageId::new(),
            }),
            Ok(StreamEvent::TextStart {
                id: block_id.clone(),
            }),
            Ok(StreamEvent::TextDelta {
                id: block_id,
                delta: oversized,
            }),
        ])
        .boxed();

        let error = super::materialize_assistant_message(stream, &model)
            .await
            .expect_err("oversized provider text should fail");
        assert!(error.to_string().contains("exceeded output cap"));
    }

    #[test]
    fn batch_tool_calls_isolates_exclusive_tools() {
        use halter_protocol::ToolCallId;

        let mut declared: HashMap<String, ToolConcurrency> = HashMap::new();
        for (name, concurrency) in [
            ("read_a", ToolConcurrency::ReadOnly),
            ("read_b", ToolConcurrency::ReadOnly),
            ("exclusive", ToolConcurrency::Exclusive),
            ("parallel", ToolConcurrency::ParallelSafe),
        ] {
            declared.insert(name.into(), concurrency);
        }

        let mk = |tool: &str, id: &str| ToolCall {
            id: ToolCallId::from(id),
            name: ToolName::from(tool),
            arguments: serde_json::Value::Null,
        };
        let batches = super::batch_tool_calls_by_concurrency(
            |name| declared.get(&name.0).copied(),
            vec![
                mk("read_a", "1"),
                mk("read_b", "2"),
                mk("exclusive", "3"),
                mk("parallel", "4"),
                mk("read_a", "5"),
            ],
        );
        assert_eq!(batches.len(), 3, "three batches: [r,r], [excl], [p,r]");
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[1].len(), 1);
        assert_eq!(batches[1][0].name.0, "exclusive");
        assert_eq!(batches[2].len(), 2);
    }

    #[test]
    fn batch_tool_calls_treats_unknown_tools_as_exclusive() {
        use halter_protocol::ToolCallId;

        let mk = |name: &str, id: &str| ToolCall {
            id: ToolCallId::from(id),
            name: ToolName::from(name),
            arguments: serde_json::Value::Null,
        };
        let batches = super::batch_tool_calls_by_concurrency(
            |_| None,
            vec![mk("mystery_a", "1"), mk("mystery_b", "2")],
        );
        assert_eq!(batches.len(), 2, "unknown tools must be exclusive");
    }

    #[test]
    fn probe_git_returns_none_outside_working_tree() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let (branch, dirty) = super::probe_git(tmp.path());
        assert_eq!(branch, None);
        assert_eq!(dirty, None);
    }

    #[test]
    fn probe_git_reports_branch_and_dirty_flag() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let root = tmp.path();
        let git = |args: &[&str]| {
            let status = std::process::Command::new("git")
                .args(args)
                .current_dir(root)
                .output()
                .expect("git command");
            assert!(status.status.success(), "git {:?} failed", args);
        };

        git(&["init", "--initial-branch=trunk"]);
        git(&["config", "user.email", "test@example.com"]);
        git(&["config", "user.name", "Test User"]);
        std::fs::write(root.join("seed.txt"), b"initial").expect("write seed");
        git(&["add", "seed.txt"]);
        git(&["commit", "-m", "seed"]);

        let (branch, dirty) = super::probe_git(root);
        assert_eq!(branch.as_deref(), Some("trunk"));
        assert_eq!(dirty, Some(false));

        std::fs::write(root.join("dirty.txt"), b"unstaged").expect("write dirty");
        let (branch, dirty) = super::probe_git(root);
        assert_eq!(branch.as_deref(), Some("trunk"));
        assert_eq!(dirty, Some(true));
    }

    mod test_support {
        use std::path::Path;
        use std::sync::Arc;

        use halter_hooks::{HookRegistrySource, Hooks, HooksFile};
        use halter_protocol::{
            ApiKind, ModelId, ModelRole, PluginId, ProviderKind, ProviderName, ResolvedModel,
            ResourceSnapshot, SubagentEventForwarding,
        };
        use halter_providers::Provider;
        use halter_tools::{DefaultToolPolicy, PolicySettings};

        use super::{HalterSession, ModelRegistry, RuntimeServices, SessionInit, SessionRuntime};

        pub(super) fn configured_services(
            provider: Arc<dyn Provider>,
            working_dir: &Path,
        ) -> Arc<RuntimeServices> {
            configured_services_with_runtime(
                provider,
                working_dir,
                SubagentEventForwarding::Off,
                100_000,
            )
        }

        pub(super) fn configured_services_with_runtime(
            provider: Arc<dyn Provider>,
            working_dir: &Path,
            subagent_event_forwarding: SubagentEventForwarding,
            subagent_event_forwarding_cap: u64,
        ) -> Arc<RuntimeServices> {
            configured_services_with_runtime_and_trace(
                provider,
                working_dir,
                subagent_event_forwarding,
                subagent_event_forwarding_cap,
                None,
            )
        }

        pub(super) fn configured_services_with_runtime_and_trace(
            provider: Arc<dyn Provider>,
            working_dir: &Path,
            subagent_event_forwarding: SubagentEventForwarding,
            subagent_event_forwarding_cap: u64,
            trace_recorder: Option<Arc<crate::TraceRecorder>>,
        ) -> Arc<RuntimeServices> {
            let mut services = RuntimeServices::default();
            let mut models = ModelRegistry::new();
            models.set_default_model(ResolvedModel {
                role: ModelRole::default(),
                id: ModelId::from("default"),
                provider: ProviderName::from("fake"),
                provider_kind: ProviderKind::Fake,
                api_kind: ApiKind::Fake,
                model: "halter/fake".to_owned(),
                max_input_tokens: Some(32_000),
                max_output_tokens: Some(4_096),
                reasoning: None,
                tokens_per_minute: None,
            });
            models.set_subagent_model(ResolvedModel {
                role: ModelRole::subagent(),
                id: ModelId::from("subagent"),
                provider: ProviderName::from("fake"),
                provider_kind: ProviderKind::Fake,
                api_kind: ApiKind::Fake,
                model: "halter/fake".to_owned(),
                max_input_tokens: Some(32_000),
                max_output_tokens: Some(4_096),
                reasoning: None,
                tokens_per_minute: None,
            });
            models.set_small_model(ResolvedModel {
                role: ModelRole::small(),
                id: ModelId::from("small"),
                provider: ProviderName::from("fake"),
                provider_kind: ProviderKind::Fake,
                api_kind: ApiKind::Fake,
                model: "halter/fake-small".to_owned(),
                max_input_tokens: Some(32_000),
                max_output_tokens: Some(4_096),
                reasoning: None,
                tokens_per_minute: None,
            });
            models.register_provider(ProviderName::from("fake"), provider);
            services.models = Arc::new(models);
            services.policy = Arc::new(DefaultToolPolicy::new(PolicySettings {
                allowed_write_roots: vec![working_dir.to_path_buf()],
                ..PolicySettings::default()
            }));
            services.subagent_event_forwarding = subagent_event_forwarding;
            services.subagent_event_forwarding_cap = subagent_event_forwarding_cap;
            services.trace_recorder = trace_recorder;
            Arc::new(services)
        }

        pub(super) async fn new_session(
            runtime: &SessionRuntime,
            working_dir: &Path,
        ) -> HalterSession {
            runtime
                .new_session(SessionInit {
                    working_dir: working_dir.to_path_buf(),
                    ..SessionInit::default()
                })
                .await
                .expect("session")
        }

        pub(super) fn install_file_hooks(
            services: &Arc<RuntimeServices>,
            working_dir: &Path,
            hooks_file: HooksFile,
        ) {
            services.resources.replace(
                ResourceSnapshot::empty(),
                Arc::new(Hooks::from_sources(vec![HookRegistrySource {
                    plugin_id: PluginId::from("test-plugin"),
                    plugin_root: working_dir.to_path_buf(),
                    source_path: working_dir.join("hooks/hooks.json"),
                    allowed_http_hosts: Vec::new(),
                    allowed_env_vars: Vec::new(),
                    file: hooks_file,
                }])),
                Vec::new(),
            );
        }

        pub(super) fn empty_hooks() -> Arc<Hooks> {
            Arc::new(Hooks::default())
        }

        pub(super) fn resolved_test_model(id: &str, provider: &str, model: &str) -> ResolvedModel {
            ResolvedModel {
                role: if id == "subagent" {
                    ModelRole::subagent()
                } else {
                    ModelRole::default()
                },
                id: ModelId::from(id),
                provider: ProviderName::from(provider),
                provider_kind: ProviderKind::Fake,
                api_kind: ApiKind::Fake,
                model: model.to_owned(),
                max_input_tokens: Some(32_000),
                max_output_tokens: Some(4_096),
                reasoning: None,
                tokens_per_minute: None,
            }
        }
    }

    #[tokio::test]
    async fn fake_provider_turn_produces_canonical_events() {
        let mut services = RuntimeServices::default();
        let mut models = ModelRegistry::new();
        models.set_default_model(ResolvedModel {
            role: ModelRole::default(),
            id: ModelId::from("default"),
            provider: ProviderName::from("fake"),
            provider_kind: ProviderKind::Fake,
            api_kind: ApiKind::Fake,
            model: "halter/fake".to_owned(),
            max_input_tokens: Some(32_000),
            max_output_tokens: Some(4_096),
            reasoning: None,
            tokens_per_minute: None,
        });
        models.set_subagent_model(ResolvedModel {
            role: ModelRole::subagent(),
            id: ModelId::from("subagent"),
            provider: ProviderName::from("fake"),
            provider_kind: ProviderKind::Fake,
            api_kind: ApiKind::Fake,
            model: "halter/fake".to_owned(),
            max_input_tokens: Some(32_000),
            max_output_tokens: Some(4_096),
            reasoning: None,
            tokens_per_minute: None,
        });
        models.register_provider(
            ProviderName::from("fake"),
            Arc::new(FakeProvider::default()),
        );
        services.models = Arc::new(models);

        let runtime = SessionRuntime::new(Arc::new(services));
        let session = runtime
            .new_session(SessionInit::default())
            .await
            .expect("session");
        let events = session
            .submit_turn(Turn::user("hello runtime"))
            .await
            .expect("submit turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("collect events");

        assert!(
            events
                .iter()
                .any(|event| matches!(event.payload, SessionEventPayload::TurnCompleted { .. }))
        );
    }

    #[tokio::test]
    async fn submit_turn_executes_tool_calls_until_completion() {
        let temp = tempfile::tempdir().expect("tempdir");
        let services = configured_services(Arc::new(ToolLoopProvider), temp.path());
        register_builtin_tools(&services.tools, &[]);
        let runtime = SessionRuntime::new(services.clone());
        let session = new_session(&runtime, temp.path()).await;

        let events = session
            .submit_turn(Turn::user("write a note"))
            .await
            .expect("submit turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("collect events");

        assert!(temp.path().join("note.txt").exists());
        assert!(events.iter().any(|event| matches!(
            event.payload,
            SessionEventPayload::ToolExecutionStarted { .. }
        )));
        assert!(events.iter().any(|event| matches!(
            event.payload,
            SessionEventPayload::ToolExecutionCompleted { .. }
        )));
        assert!(events.iter().any(|event| matches!(
            &event.payload,
            SessionEventPayload::MessageItem {
                message: Message::Assistant(assistant),
            } if assistant.parts.iter().any(|part| matches!(
                part,
                AssistantPart::Text { text } if text.contains("tool completed")
            ))
        )));
    }

    #[tokio::test]
    async fn submit_turn_dedupes_duplicate_tool_call_ids() {
        let temp = tempfile::tempdir().expect("tempdir");
        let executions = Arc::new(Mutex::new(0usize));
        let services = configured_services(Arc::new(DuplicateToolCallProvider), temp.path());
        services
            .tools
            .register(Arc::new(CountingTool::new(executions.clone())));
        let runtime = SessionRuntime::new(services.clone());
        let session = new_session(&runtime, temp.path()).await;

        let events = session
            .submit_turn(Turn::user("dedupe tool calls"))
            .await
            .expect("submit turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("collect events");

        assert_eq!(*executions.lock().expect("executions"), 1);
        assert!(events.iter().any(|event| matches!(
            &event.payload,
            SessionEventPayload::MessageItem {
                message: Message::Assistant(assistant),
            } if assistant.parts.iter().filter(|part| matches!(part, AssistantPart::ToolCall(_))).count() == 1
        )));
    }

    #[tokio::test]
    async fn pre_tool_use_hook_can_block_tool_execution() {
        let temp = tempfile::tempdir().expect("tempdir");
        let services = configured_services(Arc::new(ToolLoopProvider), temp.path());
        register_builtin_tools(&services.tools, &[]);
        let (hooks_file, warnings) = HooksFile::from_json_bytes(
            br#"{
                "hooks": {
                    "PreToolUse": [
                        {
                            "matcher": "write",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "echo blocked by hook >&2; exit 2"
                                }
                            ]
                        }
                    ]
                }
            }"#,
        )
        .expect("parse hooks");
        assert!(warnings.is_empty());
        install_file_hooks(&services, temp.path(), hooks_file);

        let runtime = SessionRuntime::new(services.clone());
        let session = new_session(&runtime, temp.path()).await;

        let events = session
            .submit_turn(Turn::user("write a note"))
            .await
            .expect("submit turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("collect events");

        assert!(!temp.path().join("note.txt").exists());
        assert!(
            events
                .iter()
                .any(|event| matches!(event.payload, SessionEventPayload::HookStarted { .. }))
        );
        assert!(events.iter().any(|event| matches!(
            &event.payload,
            SessionEventPayload::HookCompleted { run } if run.status == HookRunStatus::Blocked
        )));
        assert!(events.iter().any(|event| matches!(
            &event.payload,
            SessionEventPayload::MessageItem {
                message: Message::Tool(tool),
            } if tool
                .error
                .as_ref()
                .is_some_and(|error| error.message.contains("blocked by hook"))
        )));

        // C4 / AC2.6: a turn that the pre-tool hook blocks must not leak the
        // (uninvoked) call into the persisted `pending_tool_calls` map.
        let stored = services
            .sessions
            .load_session(session.session_id())
            .await
            .expect("load session")
            .expect("session present");
        assert!(
            stored.state.pending_tool_calls.is_empty(),
            "blocked-tool path left pending_tool_calls populated: {} entries",
            stored.state.pending_tool_calls.len()
        );
    }

    #[tokio::test]
    async fn ac2_1_dropping_a_clone_does_not_evict_hooks_from_the_original_handle() {
        let temp = tempfile::tempdir().expect("tempdir");
        let services = configured_services(Arc::new(ToolLoopProvider), temp.path());
        let runtime = SessionRuntime::new(services.clone());
        let session = new_session(&runtime, temp.path()).await;
        let session_id = session.session_id().clone();

        // Setup: the SessionHandle's hooks are registered in the runtime store.
        {
            let store = services
                .session_hook_store
                .lock()
                .expect("session hook store lock");
            assert!(
                store.contains_key(&session_id),
                "session hooks should be registered after new_session"
            );
        }

        // Before Phase 3, `HalterSession: Clone + Drop` evicted the hook entry
        // every time any clone was dropped (even an internal clone moved into
        // the spawned turn loop). The pinned PR re-introduces the footgun if
        // this assertion ever flips back to "absent".
        let cloned = session.clone();
        drop(cloned);

        {
            let store = services
                .session_hook_store
                .lock()
                .expect("session hook store lock");
            assert!(
                store.contains_key(&session_id),
                "dropping a clone evicted hooks; the original handle must keep them alive"
            );
        }

        drop(session);

        {
            let store = services
                .session_hook_store
                .lock()
                .expect("session hook store lock");
            assert!(
                !store.contains_key(&session_id),
                "session hooks should be evicted once the last handle is dropped"
            );
        }
    }

    #[tokio::test]
    async fn sdk_callback_hook_can_block_tool_execution() {
        let temp = tempfile::tempdir().expect("tempdir");
        let mut services = configured_services(Arc::new(ToolLoopProvider), temp.path());
        register_builtin_tools(&services.tools, &[]);

        let mut registered = RegisteredHooks::default();
        registered.register(
            PluginId::from("internal"),
            RegisteredHookPriority::AfterPlugins,
            Hook::callback(HookEventName::PreToolUse, |input| async move {
                if input.tool_name() == Some("write") {
                    HookResponse::block("blocked by callback hook")
                } else {
                    HookResponse::passthrough()
                }
            }),
        );
        Arc::get_mut(&mut services)
            .expect("unique services")
            .registered_hooks = Arc::new(registered);

        let runtime = SessionRuntime::new(services.clone());
        let session = new_session(&runtime, temp.path()).await;

        let events = session
            .submit_turn(Turn::user("write a note"))
            .await
            .expect("submit turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("collect events");

        assert!(!temp.path().join("note.txt").exists());
        assert!(events.iter().any(|event| matches!(
            &event.payload,
            SessionEventPayload::HookCompleted { run }
                if run.status == HookRunStatus::Blocked
                    && run.handler_type == HookHandlerType::Callback
        )));
        assert!(events.iter().any(|event| matches!(
            &event.payload,
            SessionEventPayload::MessageItem {
                message: Message::Tool(tool),
            } if tool
                .error
                .as_ref()
                .is_some_and(|error| error.message.contains("blocked by callback hook"))
        )));
    }

    #[tokio::test]
    async fn prompt_hook_uses_small_model_and_blocks_turn() {
        let temp = tempfile::tempdir().expect("tempdir");
        let requests = Arc::new(Mutex::new(Vec::<ProviderRequest>::new()));
        let mut services = RuntimeServices::default();
        let mut models = ModelRegistry::new();
        models.set_default_model(resolved_test_model("default", "fake", "default/model"));
        models.set_small_model(ResolvedModel {
            role: ModelRole::small(),
            id: ModelId::from("small"),
            provider: ProviderName::from("fake"),
            provider_kind: ProviderKind::Fake,
            api_kind: ApiKind::Fake,
            model: "small/model".to_owned(),
            max_input_tokens: Some(32_000),
            max_output_tokens: Some(4_096),
            reasoning: None,
            tokens_per_minute: None,
        });
        models.set_subagent_model(resolved_test_model("subagent", "fake", "subagent/model"));
        models.register_provider(
            ProviderName::from("fake"),
            Arc::new(JsonHookProvider::new(requests.clone())),
        );
        services.models = Arc::new(models);
        services.policy = Arc::new(DefaultToolPolicy::new(PolicySettings {
            allowed_write_roots: vec![temp.path().to_path_buf()],
            ..PolicySettings::default()
        }));

        let (hooks_file, warnings) = HooksFile::from_json_bytes(
            br#"{
                "hooks": {
                    "UserPromptSubmit": [
                        {
                            "hooks": [
                                {
                                    "type": "prompt",
                                    "prompt": "HOOK_PROMPT $ARGUMENTS"
                                }
                            ]
                        }
                    ]
                }
            }"#,
        )
        .expect("parse hooks");
        assert!(warnings.is_empty());
        let services = Arc::new(services);
        install_file_hooks(&services, temp.path(), hooks_file);

        let runtime = SessionRuntime::new(services);
        let session = new_session(&runtime, temp.path()).await;

        let events = session
            .submit_turn(Turn::user("blocked prompt"))
            .await
            .expect("submit turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("collect events");

        let requests = requests.lock().expect("requests");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].model.id, ModelId::from("small"));
        assert!(events.iter().any(|event| matches!(
            &event.payload,
            SessionEventPayload::HookCompleted { run } if run.status == HookRunStatus::Blocked
        )));
        assert!(events.iter().any(|event| matches!(
            &event.payload,
            SessionEventPayload::MessageItem {
                message: Message::System(system),
            } if system.text.contains("blocked by prompt hook")
        )));
    }

    #[tokio::test]
    async fn sdk_function_hooks_keep_state_per_session() {
        let temp = tempfile::tempdir().expect("tempdir");
        let mut services = configured_services(Arc::new(FakeProvider::default()), temp.path());
        let mut registered = RegisteredHooks::default();
        registered.register(
            PluginId::from("internal"),
            RegisteredHookPriority::AfterPlugins,
            Hook::function(HookEventName::UserPromptSubmit, || {
                let seen = Arc::new(Mutex::new(0usize));
                move |_input| {
                    let seen = seen.clone();
                    async move {
                        let mut seen = seen.lock().expect("seen");
                        *seen += 1;
                        HookResponse::passthrough()
                            .with_system_message(format!("function count {}", *seen))
                    }
                }
            }),
        );
        Arc::get_mut(&mut services)
            .expect("unique services")
            .registered_hooks = Arc::new(registered);

        let runtime = SessionRuntime::new(services.clone());
        let session_a = new_session(&runtime, temp.path()).await;
        let session_b = new_session(&runtime, temp.path()).await;

        let events_a1 = session_a
            .submit_turn(Turn::user("first"))
            .await
            .expect("submit turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("collect events");
        let events_a2 = session_a
            .submit_turn(Turn::user("second"))
            .await
            .expect("submit turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("collect events");
        let events_b1 = session_b
            .submit_turn(Turn::user("third"))
            .await
            .expect("submit turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("collect events");

        assert!(events_a1.iter().any(|event| matches!(
            &event.payload,
            SessionEventPayload::HookCompleted { run }
                if run.handler_type == HookHandlerType::Function
        )));
        assert!(events_a1.iter().any(|event| matches!(
            &event.payload,
            SessionEventPayload::MessageItem {
                message: Message::System(system),
            } if system.text.contains("function count 1")
        )));
        assert!(events_a2.iter().any(|event| matches!(
            &event.payload,
            SessionEventPayload::MessageItem {
                message: Message::System(system),
            } if system.text.contains("function count 2")
        )));
        assert!(events_b1.iter().any(|event| matches!(
            &event.payload,
            SessionEventPayload::MessageItem {
                message: Message::System(system),
            } if system.text.contains("function count 1")
        )));
    }

    #[tokio::test]
    async fn hook_warnings_emit_warning_events_on_next_turn() {
        let temp = tempfile::tempdir().expect("tempdir");
        let services = configured_services(Arc::new(FakeProvider::default()), temp.path());
        services.resources.replace(
            ResourceSnapshot::empty(),
            empty_hooks(),
            vec![halter_protocol::HookWarning {
                category: "test".to_owned(),
                message: "hook warning".to_owned(),
                ..halter_protocol::HookWarning::default()
            }],
        );
        let runtime = SessionRuntime::new(services);
        let session = new_session(&runtime, temp.path()).await;

        let events = session
            .submit_turn(Turn::user("hello"))
            .await
            .expect("submit turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("collect events");

        assert!(events.iter().any(|event| matches!(
            &event.payload,
            SessionEventPayload::Warning { message } if message.contains("hook warning")
        )));
    }

    #[tokio::test]
    async fn hook_events_delivered_after_turn_commits() {
        // Post-H6 single commit-then-publish: HookStarted/HookCompleted
        // arrive at turn commit, not intra-turn. This test only asserts they
        // are present in the post-commit stream.
        let temp = tempfile::tempdir().expect("tempdir");
        let mut services = RuntimeServices::default();
        let mut hooks = RegisteredHooks::default();
        hooks.register(
            PluginId::from("test-plugin"),
            RegisteredHookPriority::AfterPlugins,
            Hook::callback(HookEventName::UserPromptSubmit, move |_input| async move {
                HookResponse::passthrough()
            }),
        );
        services.registered_hooks = Arc::new(hooks);
        let mut models = ModelRegistry::new();
        models.set_default_model(resolved_test_model("default", "fake", "default/model"));
        models.set_subagent_model(resolved_test_model("subagent", "fake", "subagent/model"));
        models.register_provider(
            ProviderName::from("fake"),
            Arc::new(FakeProvider::default()),
        );
        services.models = Arc::new(models);
        services.policy = Arc::new(DefaultToolPolicy::new(PolicySettings {
            allowed_write_roots: vec![temp.path().to_path_buf()],
            ..PolicySettings::default()
        }));

        let runtime = SessionRuntime::new(Arc::new(services));
        let session = new_session(&runtime, temp.path()).await;

        let events = session
            .submit_turn(Turn::user("hello"))
            .await
            .expect("submit turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("collect events");

        assert!(
            events
                .iter()
                .any(|event| matches!(event.payload, SessionEventPayload::HookStarted { .. }))
        );
        assert!(
            events
                .iter()
                .any(|event| matches!(event.payload, SessionEventPayload::HookCompleted { .. }))
        );
    }

    #[tokio::test]
    async fn notify_runs_notification_hooks() {
        let temp = tempfile::tempdir().expect("tempdir");
        let services = configured_services(Arc::new(FakeProvider::default()), temp.path());
        let (hooks_file, warnings) = HooksFile::from_json_bytes(
            br#"{
                "hooks": {
                    "Notification": [
                        {
                            "matcher": "policy",
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "printf '{\"systemMessage\":\"notification seen\"}'"
                                }
                            ]
                        }
                    ]
                }
            }"#,
        )
        .expect("parse hooks");
        assert!(warnings.is_empty());
        install_file_hooks(&services, temp.path(), hooks_file);
        let runtime = SessionRuntime::new(services.clone());
        let session = new_session(&runtime, temp.path()).await;

        session
            .notify("policy", "denied")
            .await
            .expect("notify succeeds");

        let replay = session.replay().await.expect("replay");
        assert!(replay.iter().any(|event| matches!(
            &event.payload,
            SessionEventPayload::HookCompleted { run } if run.event_name == "Notification"
        )));
    }

    #[tokio::test]
    async fn compact_summarizes_older_messages_and_sets_session_start_latch() {
        let temp = tempfile::tempdir().expect("tempdir");
        let services = configured_services(Arc::new(FakeProvider::default()), temp.path());
        let runtime = SessionRuntime::new(services.clone());
        let session = new_session(&runtime, temp.path()).await;

        let mut stored = services
            .sessions
            .load_session(session.session_id())
            .await
            .expect("load session")
            .expect("session exists");
        stored.state.messages = (0..12)
            .map(|index| Message::User(halter_protocol::UserMessage::text(format!("msg {index}"))))
            .collect();
        let _ = services
            .sessions
            .commit(
                session.session_id(),
                None,
                None,
                Some(stored.state),
                Vec::new(),
            )
            .await
            .expect("commit state");

        session
            .compact("manual", Some("Focus on decisions"))
            .await
            .expect("compact succeeds");

        let stored = services
            .sessions
            .load_session(session.session_id())
            .await
            .expect("load compacted session")
            .expect("session exists");
        assert!(!stored.state.compacted_prefix.is_empty());
        assert!(stored.state.messages.is_empty());
        assert_eq!(
            stored.state.pending_session_start_source,
            Some(HookSessionStartSource::Compact)
        );

        let replay = session.replay().await.expect("replay");
        assert!(
            replay
                .iter()
                .any(|event| matches!(event.payload, SessionEventPayload::ContextCompacted { .. }))
        );
    }

    #[tokio::test]
    async fn submit_turn_compacts_immediately_after_response_when_threshold_is_reached() {
        let temp = tempfile::tempdir().expect("tempdir");
        let mut services = configured_services(Arc::new(FakeProvider::default()), temp.path());
        Arc::get_mut(&mut services)
            .expect("unique services")
            .context_manager = Arc::new(DefaultContextManager::new(
            150,
            0,
            halter_protocol::PruneSignalThreshold::VeryLow,
        ));
        let runtime = SessionRuntime::new(services.clone());
        let session = new_session(&runtime, temp.path()).await;

        session
            .submit_turn(Turn::user("x".repeat(150)))
            .await
            .expect("submit turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("collect events");

        let stored = services
            .sessions
            .load_session(session.session_id())
            .await
            .expect("load session")
            .expect("session exists");
        assert!(!stored.state.compacted_prefix.is_empty());
        assert_eq!(stored.state.messages.len(), 1);
        assert!(matches!(stored.state.messages[0], Message::Assistant(_)));
    }

    #[tokio::test]
    async fn submit_turn_failure_preserves_valid_transcript() {
        let temp = tempfile::tempdir().expect("tempdir");
        let services = configured_services(Arc::new(FailingProvider), temp.path());
        let runtime = SessionRuntime::new(services.clone());
        let session = new_session(&runtime, temp.path()).await;

        let events = session
            .submit_turn(Turn::user("will fail"))
            .await
            .expect("submit turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("collect events");

        let stored = services
            .sessions
            .load_session(session.session_id())
            .await
            .expect("load session")
            .expect("session exists");
        assert_eq!(stored.state.messages.len(), 1);
        assert!(matches!(
            &stored.state.messages[0],
            Message::User(user) if user.plain_text() == "will fail"
        ));
        assert!(
            events
                .iter()
                .any(|event| matches!(event.payload, SessionEventPayload::TurnFailed { .. }))
        );

        let events = services
            .sessions
            .replay(session.session_id())
            .await
            .expect("replay");
        assert!(
            events
                .iter()
                .any(|event| matches!(event.payload, SessionEventPayload::TurnFailed { .. }))
        );
        assert!(events.iter().any(|event| matches!(
            &event.payload,
            SessionEventPayload::MessageItem {
                message: Message::User(user),
            } if user.plain_text() == "will fail"
        )));
    }

    #[tokio::test]
    async fn later_turns_commit_latest_resource_snapshot() {
        let temp = tempfile::tempdir().expect("tempdir");
        let services = configured_services(Arc::new(FakeProvider::default()), temp.path());
        let runtime = SessionRuntime::new(services.clone());
        let session = new_session(&runtime, temp.path()).await;

        let mut reloaded = ResourceSnapshot::empty();
        reloaded.revision = halter_protocol::Revision::from("reloaded");
        runtime.replace_resources(reloaded, empty_hooks(), Vec::new());

        session
            .submit_turn(Turn::user("after reload"))
            .await
            .expect("submit turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("collect events");

        let stored = services
            .sessions
            .load_session(session.session_id())
            .await
            .expect("load session")
            .expect("session exists");
        assert_eq!(stored.snapshot.revision.0, "reloaded");
    }

    #[tokio::test]
    async fn session_init_can_override_subagent_model() {
        let temp = tempfile::tempdir().expect("tempdir");
        let services = configured_services(Arc::new(FakeProvider::default()), temp.path());
        let runtime = SessionRuntime::new(services.clone());
        let session = runtime
            .new_session(SessionInit::default().with_subagent_model("default"))
            .await
            .expect("session");

        let stored = services
            .sessions
            .load_session(session.session_id())
            .await
            .expect("load session")
            .expect("session exists");

        assert_eq!(stored.blueprint.default_model, ModelId::from("default"));
        assert_eq!(stored.blueprint.subagent_model, ModelId::from("default"));
    }

    #[tokio::test]
    async fn turn_default_model_override_selects_overridden_provider() {
        let temp = tempfile::tempdir().expect("tempdir");
        let default_requests = Arc::new(Mutex::new(Vec::<ProviderRequest>::new()));
        let subagent_requests = Arc::new(Mutex::new(Vec::<ProviderRequest>::new()));
        let mut services = RuntimeServices::default();
        let mut models = ModelRegistry::new();
        models.set_default_model(resolved_test_model(
            "default",
            "default-provider",
            "default/model",
        ));
        models.set_subagent_model(resolved_test_model(
            "subagent",
            "subagent-provider",
            "subagent/model",
        ));
        models.register_provider(
            ProviderName::from("default-provider"),
            Arc::new(RecordingProvider::new(
                default_requests.clone(),
                "default provider reply",
            )),
        );
        models.register_provider(
            ProviderName::from("subagent-provider"),
            Arc::new(RecordingProvider::new(
                subagent_requests.clone(),
                "subagent provider reply",
            )),
        );
        services.models = Arc::new(models);
        services.policy = Arc::new(DefaultToolPolicy::new(PolicySettings {
            allowed_write_roots: vec![temp.path().to_path_buf()],
            ..PolicySettings::default()
        }));

        let runtime = SessionRuntime::new(Arc::new(services));
        let session = new_session(&runtime, temp.path()).await;

        let events = session
            .submit_turn(Turn::user("hello").with_default_model("subagent"))
            .await
            .expect("submit turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("collect events");

        assert!(default_requests.lock().expect("requests").is_empty());
        let subagent_requests = subagent_requests.lock().expect("requests");
        assert_eq!(subagent_requests.len(), 1);
        assert_eq!(subagent_requests[0].model.id, ModelId::from("subagent"));
        assert_eq!(
            subagent_requests[0].model.provider,
            ProviderName::from("subagent-provider")
        );
        assert!(events.iter().any(|event| matches!(
            &event.payload,
            SessionEventPayload::MessageItem {
                message: Message::Assistant(assistant),
            } if assistant.parts.iter().any(|part| matches!(
                part,
                AssistantPart::Text { text } if text.contains("subagent provider reply")
            ))
        )));
    }

    #[tokio::test]
    async fn turn_rejects_unknown_subagent_model_override() {
        let temp = tempfile::tempdir().expect("tempdir");
        let services = configured_services(Arc::new(FakeProvider::default()), temp.path());
        let runtime = SessionRuntime::new(services.clone());
        let session = new_session(&runtime, temp.path()).await;

        let events = session
            .submit_turn(Turn::user("hello").with_subagent_model("missing"))
            .await
            .expect("submit turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("collect events");

        assert!(events.iter().any(|event| matches!(
            &event.payload,
            SessionEventPayload::TurnFailed { error, .. } if error.contains("unknown model 'missing'")
        )));
    }

    #[tokio::test]
    async fn submit_turn_delivers_tool_output_after_turn_commits() {
        // Post-H6: tool output chunks collected during tool execution are
        // flushed alongside other turn events at commit. This test asserts
        // the chunk is present in the committed stream, not that it arrives
        // mid-tool.
        let temp = tempfile::tempdir().expect("tempdir");
        let services = configured_services(Arc::new(StreamingToolProvider), temp.path());
        services.tools.register(Arc::new(StreamingTestTool));
        let runtime = SessionRuntime::new(services);
        let session = new_session(&runtime, temp.path()).await;

        let events = session
            .submit_turn(Turn::user("stream tool output"))
            .await
            .expect("submit turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("collect events");

        let tool_output = events
            .iter()
            .find_map(|event| match &event.payload {
                SessionEventPayload::ToolOutput { chunk, .. } => Some(chunk.clone()),
                _ => None,
            })
            .expect("tool output chunk present in committed events");
        assert_eq!(tool_output, "streamed chunk");
        assert!(
            events
                .iter()
                .any(|event| matches!(event.payload, SessionEventPayload::TurnCompleted { .. }))
        );
    }

    #[tokio::test]
    async fn mutating_tool_result_survives_later_provider_failure() {
        let temp = tempfile::tempdir().expect("tempdir");
        let services = configured_services(Arc::new(ToolThenFailProvider), temp.path());
        register_builtin_tools(&services.tools, &[]);
        let runtime = SessionRuntime::new(services.clone());
        let session = new_session(&runtime, temp.path()).await;

        let events = session
            .submit_turn(Turn::user("write then fail"))
            .await
            .expect("submit turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("collect events");

        let written = std::fs::read_to_string(temp.path().join("note.txt")).expect("written file");
        assert_eq!(written, "hello from tool");
        assert!(events.iter().any(|event| matches!(
            &event.payload,
            SessionEventPayload::ToolExecutionCompleted { outcome }
                if outcome.call.name.0 == "write" && outcome.result.is_ok()
        )));
        assert!(events.iter().any(|event| matches!(
            &event.payload,
            SessionEventPayload::TurnFailed { error, .. }
                if error.contains("provider failed after tool")
        )));

        let replayed = session.replay().await.expect("replay events");
        assert!(replayed.iter().any(|event| matches!(
            &event.payload,
            SessionEventPayload::ToolExecutionCompleted { outcome }
                if outcome.call.name.0 == "write" && outcome.result.is_ok()
        )));
        assert!(replayed.iter().any(|event| matches!(
            &event.payload,
            SessionEventPayload::TurnFailed { error, .. }
                if error.contains("provider failed after tool")
        )));

        let stored = services
            .sessions
            .load_session(session.session_id())
            .await
            .expect("load session")
            .expect("session exists");
        assert!(stored.state.messages.iter().any(|message| {
            matches!(
                message,
                Message::Tool(tool) if tool.error.is_none()
            )
        }));
    }

    #[tokio::test]
    async fn max_turns_caps_provider_iterations() {
        let temp = tempfile::tempdir().expect("tempdir");
        let services = configured_services(Arc::new(ToolLoopProvider), temp.path());
        register_builtin_tools(&services.tools, &[]);
        let runtime = SessionRuntime::new(services);
        let session = runtime
            .new_session(SessionInit {
                working_dir: temp.path().to_path_buf(),
                max_turns: Some(1),
                ..SessionInit::default()
            })
            .await
            .expect("session");

        let events = session
            .submit_turn(Turn::user("one iteration only"))
            .await
            .expect("submit turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("collect events");

        assert_eq!(
            std::fs::read_to_string(temp.path().join("note.txt")).expect("written file"),
            "hello from tool"
        );
        assert!(events.iter().any(|event| matches!(
            &event.payload,
            SessionEventPayload::TurnFailed { error, .. } if error.contains("max_turns 1")
        )));
        assert!(
            !events
                .iter()
                .any(|event| matches!(event.payload, SessionEventPayload::TurnCompleted { .. }))
        );
    }

    #[tokio::test]
    async fn subagent_event_forwarding_defaults_off() {
        let (parent_id, events) =
            run_subagent_firehose_turn(SubagentEventForwarding::Off, None, 100_000, "single").await;

        assert!(
            events.iter().all(|event| event.session_id == parent_id),
            "default-off parent stream should contain only parent session events: {events:?}"
        );
    }

    #[tokio::test]
    async fn traces_record_subagent_events_when_forwarding_is_off() {
        let temp = tempfile::tempdir().expect("tempdir");
        let traces_dir = temp.path().join("traces");
        let trace_recorder =
            Arc::new(crate::TraceRecorder::open(traces_dir.clone()).expect("trace recorder"));
        let provider = Arc::new(SubagentFirehoseProvider);
        let services = test_support::configured_services_with_runtime_and_trace(
            provider,
            temp.path(),
            SubagentEventForwarding::Off,
            100_000,
            Some(trace_recorder.clone()),
        );
        let runtime = SessionRuntime::new(services.clone());
        install_subagent_tools(&runtime, &services);
        let session = runtime
            .new_session(SessionInit {
                working_dir: temp.path().to_path_buf(),
                ..SessionInit::default()
            })
            .await
            .expect("session");
        let parent_id = session.session_id().clone();

        let events = session
            .submit_turn(Turn::user("single"))
            .await
            .expect("turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("events");
        assert!(
            events.iter().all(|event| event.session_id == parent_id),
            "forwarding is off, so live stream should stay parent-only: {events:?}"
        );

        trace_recorder.close_session(&parent_id);
        let contents = std::fs::read_to_string(traces_dir.join(format!("{}.txt", parent_id.0)))
            .expect("read trace");
        let lines = contents
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).expect("trace json"))
            .collect::<Vec<_>>();

        assert!(
            lines.iter().any(|line| {
                line.get("kind").and_then(serde_json::Value::as_str) == Some("subagent_header")
            }),
            "trace should include a subagent header even when forwarding is off:\n{contents}"
        );
        assert!(
            lines.iter().any(|line| {
                line.get("sequence").is_some()
                    && line.get("session_id").and_then(serde_json::Value::as_str)
                        != Some(parent_id.0.as_str())
                    && line
                        .pointer("/payload/kind")
                        .and_then(serde_json::Value::as_str)
                        == Some("delta_item")
                    && line
                        .pointer("/payload/delta/text")
                        .and_then(serde_json::Value::as_str)
                        .is_some_and(|text| text.contains("child done"))
            }),
            "trace should include committed subagent delta events even when forwarding is off:\n{contents}"
        );
    }

    #[tokio::test]
    async fn subagent_event_forwarding_includes_single_level_events() {
        let (parent_id, events) =
            run_subagent_firehose_turn(SubagentEventForwarding::All, None, 100_000, "single").await;

        assert!(
            events.iter().any(|event| event.session_id == parent_id),
            "parent events should still be present"
        );
        assert!(
            forwarded_events(&events, &parent_id)
                .iter()
                .any(|event| event_has_delta_text(event, "child done")),
            "parent stream should include child session deltas: {events:?}"
        );
    }

    #[tokio::test]
    async fn subagent_event_forwarding_includes_recursive_events() {
        let (parent_id, events) =
            run_subagent_firehose_turn(SubagentEventForwarding::All, None, 100_000, "recursive")
                .await;
        let forwarded = forwarded_events(&events, &parent_id);
        let forwarded_session_ids = forwarded
            .iter()
            .map(|event| event.session_id.clone())
            .collect::<BTreeSet<_>>();

        assert!(
            forwarded_session_ids.len() >= 2,
            "recursive forwarding should include child and grandchild sessions: {events:?}"
        );
        assert!(
            forwarded
                .iter()
                .any(|event| event_has_delta_text(event, "grandchild done")),
            "top-level stream should include grandchild deltas: {events:?}"
        );
    }

    #[tokio::test]
    async fn session_init_can_override_subagent_event_forwarding() {
        let temp = tempfile::tempdir().expect("tempdir");
        let provider = Arc::new(SubagentFirehoseProvider);
        let services = test_support::configured_services_with_runtime(
            provider,
            temp.path(),
            SubagentEventForwarding::Off,
            100_000,
        );
        let runtime = SessionRuntime::new(services.clone());
        install_subagent_tools(&runtime, &services);

        let enabled_session = runtime
            .new_session(SessionInit {
                working_dir: temp.path().to_path_buf(),
                subagent_event_forwarding: Some(SubagentEventForwarding::All),
                ..SessionInit::default()
            })
            .await
            .expect("enabled session");
        let enabled_parent_id = enabled_session.session_id().clone();
        let enabled_events = enabled_session
            .submit_turn(Turn::user("single"))
            .await
            .expect("enabled turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("enabled events");

        let default_session = runtime
            .new_session(SessionInit {
                working_dir: temp.path().to_path_buf(),
                ..SessionInit::default()
            })
            .await
            .expect("default session");
        let default_parent_id = default_session.session_id().clone();
        let default_events = default_session
            .submit_turn(Turn::user("single"))
            .await
            .expect("default turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("default events");

        assert!(
            !forwarded_events(&enabled_events, &enabled_parent_id).is_empty(),
            "per-session override should enable forwarding"
        );
        assert!(
            forwarded_events(&default_events, &default_parent_id).is_empty(),
            "harness default off should still apply to other sessions"
        );
    }

    #[tokio::test]
    async fn subagent_event_forwarding_cap_emits_lagged_and_stops_forwarding() {
        let (parent_id, events) =
            run_subagent_firehose_turn(SubagentEventForwarding::All, None, 2, "many child events")
                .await;

        assert!(
            events.iter().any(|event| {
                event.session_id.0 == crate::event_bus::BUS_SESSION_ID
                    && matches!(
                        event.payload,
                        SessionEventPayload::Lagged { dropped_events: 1 }
                    )
            }),
            "cap should emit a synthetic lagged event: {events:?}"
        );
        assert_eq!(
            forwarded_events(&events, &parent_id).len(),
            2,
            "forwarded child events should stop at the configured cap: {events:?}"
        );
    }

    async fn run_subagent_firehose_turn(
        default_forwarding: SubagentEventForwarding,
        session_forwarding: Option<SubagentEventForwarding>,
        cap: u64,
        prompt: &str,
    ) -> (SessionId, Vec<SessionEvent>) {
        let temp = tempfile::tempdir().expect("tempdir");
        let provider = Arc::new(SubagentFirehoseProvider);
        let services = test_support::configured_services_with_runtime(
            provider,
            temp.path(),
            default_forwarding,
            cap,
        );
        let runtime = SessionRuntime::new(services.clone());
        install_subagent_tools(&runtime, &services);
        let session = runtime
            .new_session(SessionInit {
                working_dir: temp.path().to_path_buf(),
                subagent_event_forwarding: session_forwarding,
                ..SessionInit::default()
            })
            .await
            .expect("session");
        let parent_id = session.session_id().clone();
        let events = session
            .submit_turn(Turn::user(prompt))
            .await
            .expect("turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("events");
        (parent_id, events)
    }

    fn install_subagent_tools(runtime: &SessionRuntime, services: &Arc<RuntimeServices>) {
        let snapshot = services.resources.snapshot();
        let available_model_ids = services
            .models
            .model_ids()
            .into_iter()
            .map(|model_id| model_id.0)
            .collect::<Vec<_>>();
        register_subagent_tools(
            &services.tools,
            runtime.subagent_control(),
            &[],
            snapshot.as_ref(),
            &available_model_ids,
        );
    }

    fn forwarded_events<'a>(
        events: &'a [SessionEvent],
        parent_id: &SessionId,
    ) -> Vec<&'a SessionEvent> {
        events
            .iter()
            .filter(|event| {
                &event.session_id != parent_id
                    && event.session_id.0 != crate::event_bus::BUS_SESSION_ID
            })
            .collect()
    }

    fn event_has_delta_text(event: &SessionEvent, needle: &str) -> bool {
        matches!(
            &event.payload,
            SessionEventPayload::DeltaItem { delta } if delta.text.contains(needle)
        )
    }

    #[derive(Debug)]
    struct ToolLoopProvider;

    #[derive(Debug)]
    struct DuplicateToolCallProvider;

    #[derive(Debug, Default)]
    struct SubagentFirehoseProvider;

    #[derive(Debug)]
    struct JsonHookProvider {
        requests: Arc<Mutex<Vec<ProviderRequest>>>,
    }

    impl JsonHookProvider {
        fn new(requests: Arc<Mutex<Vec<ProviderRequest>>>) -> Self {
            Self { requests }
        }
    }

    #[async_trait]
    impl Provider for JsonHookProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }

        async fn stream(
            &self,
            request: ProviderRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
            self.requests
                .lock()
                .expect("requests")
                .push(request.clone());
            let latest_user_text = request
                .messages
                .iter()
                .rev()
                .find_map(|message| match message {
                    Message::User(user) => Some(user.plain_text()),
                    Message::System(_) | Message::Assistant(_) | Message::Tool(_) => None,
                })
                .unwrap_or_default();
            let reply = if latest_user_text.starts_with("HOOK_PROMPT ") {
                "{\"decision\":\"block\",\"reason\":\"blocked by prompt hook\"}".to_owned()
            } else {
                "normal reply".to_owned()
            };
            let message_id = halter_protocol::MessageId::new();
            let block_id = BlockId::new();
            Ok(stream::iter(vec![
                Ok(StreamEvent::MessageStart {
                    id: message_id.clone(),
                }),
                Ok(StreamEvent::TextStart {
                    id: block_id.clone(),
                }),
                Ok(StreamEvent::TextDelta {
                    id: block_id.clone(),
                    delta: reply,
                }),
                Ok(StreamEvent::TextEnd { id: block_id }),
                Ok(StreamEvent::UsageUpdate {
                    usage: Usage {
                        input_tokens: 4,
                        output_tokens: 4,
                        cache_creation_input_tokens: 0,
                        cache_read_input_tokens: 0,
                    },
                }),
                Ok(StreamEvent::MessageEnd {
                    id: message_id,
                    stop_reason: StopReason::EndTurn,
                    response_id: None,
                }),
            ])
            .boxed())
        }
    }

    #[async_trait]
    impl Provider for SubagentFirehoseProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }

        async fn stream(
            &self,
            request: ProviderRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
            let latest_user_text = latest_user_text(&request.messages);
            let has_wait_response = has_wait_response(&request.messages);
            let latest_agent_id = latest_spawn_agent_id(&request.messages);

            if request.model.id.0 == "subagent" {
                if latest_user_text.contains("spawn grandchild") {
                    if has_wait_response {
                        return Ok(text_stream(vec!["child done"]));
                    }
                    if let Some(agent_id) = latest_agent_id {
                        return Ok(tool_call_stream(
                            "wait_agent",
                            json!({ "targets": [agent_id], "timeout_ms": 5_000 }),
                        ));
                    }
                    return Ok(tool_call_stream(
                        "spawn_agent",
                        json!({ "message": "grandchild task", "fork_context": false }),
                    ));
                }

                if latest_user_text.contains("grandchild") {
                    return Ok(text_stream(vec!["grandchild done"]));
                }
                if latest_user_text.contains("many child events") {
                    return Ok(text_stream(vec![
                        "child ", "done ", "with ", "many ", "events",
                    ]));
                }
                return Ok(text_stream(vec!["child done"]));
            }

            if has_wait_response {
                return Ok(text_stream(vec!["parent done"]));
            }
            if let Some(agent_id) = latest_agent_id {
                return Ok(tool_call_stream(
                    "wait_agent",
                    json!({ "targets": [agent_id], "timeout_ms": 5_000 }),
                ));
            }

            let child_task = if latest_user_text.contains("recursive") {
                "spawn grandchild"
            } else if latest_user_text.contains("many child events") {
                "many child events"
            } else {
                "child task"
            };
            Ok(tool_call_stream(
                "spawn_agent",
                json!({ "message": child_task, "fork_context": false }),
            ))
        }
    }

    fn latest_user_text(messages: &[Message]) -> String {
        messages
            .iter()
            .rev()
            .find_map(|message| match message {
                Message::User(user) => Some(user.plain_text()),
                Message::System(_) | Message::Assistant(_) | Message::Tool(_) => None,
            })
            .unwrap_or_default()
    }

    fn latest_spawn_agent_id(messages: &[Message]) -> Option<String> {
        messages.iter().rev().find_map(|message| match message {
            Message::Tool(tool) => match &tool.content {
                ToolResult::Json { value } => value
                    .get("agent_id")
                    .and_then(serde_json::Value::as_str)
                    .map(ToOwned::to_owned),
                ToolResult::Empty | ToolResult::Text { .. } => None,
            },
            Message::System(_) | Message::User(_) | Message::Assistant(_) => None,
        })
    }

    fn has_wait_response(messages: &[Message]) -> bool {
        messages.iter().any(|message| {
            matches!(
                message,
                Message::Tool(tool)
                    if matches!(&tool.content, ToolResult::Json { value } if value.get("timed_out").is_some())
            )
        })
    }

    fn text_stream(
        chunks: Vec<&'static str>,
    ) -> BoxStream<'static, Result<StreamEvent, ProviderError>> {
        let message_id = halter_protocol::MessageId::new();
        let block_id = BlockId::new();
        let mut events = vec![
            Ok(StreamEvent::MessageStart {
                id: message_id.clone(),
            }),
            Ok(StreamEvent::TextStart {
                id: block_id.clone(),
            }),
        ];
        for chunk in chunks {
            events.push(Ok(StreamEvent::TextDelta {
                id: block_id.clone(),
                delta: chunk.to_owned(),
            }));
        }
        events.extend([
            Ok(StreamEvent::TextEnd { id: block_id }),
            Ok(StreamEvent::UsageUpdate {
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 0,
                },
            }),
            Ok(StreamEvent::MessageEnd {
                id: message_id,
                stop_reason: StopReason::EndTurn,
                response_id: None,
            }),
        ]);
        stream::iter(events).boxed()
    }

    fn tool_call_stream(
        name: &'static str,
        arguments: serde_json::Value,
    ) -> BoxStream<'static, Result<StreamEvent, ProviderError>> {
        let message_id = halter_protocol::MessageId::new();
        let block_id = BlockId::new();
        let tool_call_id = ToolCallId::new();
        stream::iter(vec![
            Ok(StreamEvent::MessageStart {
                id: message_id.clone(),
            }),
            Ok(StreamEvent::ToolCallStart {
                id: block_id.clone(),
                tool_call_id,
                name: ToolName::from(name),
            }),
            Ok(StreamEvent::ToolArgsDelta {
                id: block_id.clone(),
                delta: arguments.to_string(),
            }),
            Ok(StreamEvent::ToolCallEnd { id: block_id }),
            Ok(StreamEvent::UsageUpdate {
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 0,
                },
            }),
            Ok(StreamEvent::MessageEnd {
                id: message_id,
                stop_reason: StopReason::ToolUse,
                response_id: None,
            }),
        ])
        .boxed()
    }

    #[async_trait]
    impl Provider for ToolLoopProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }

        async fn stream(
            &self,
            request: ProviderRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
            if request
                .messages
                .iter()
                .any(|message| matches!(message, Message::Tool(_)))
            {
                return Ok(stream::iter(vec![
                    Ok(StreamEvent::MessageStart {
                        id: halter_protocol::MessageId::new(),
                    }),
                    Ok(StreamEvent::TextStart { id: BlockId::new() }),
                    Ok(StreamEvent::TextDelta {
                        id: BlockId::new(),
                        delta: "tool completed".to_owned(),
                    }),
                    Ok(StreamEvent::TextEnd { id: BlockId::new() }),
                    Ok(StreamEvent::UsageUpdate {
                        usage: Usage {
                            input_tokens: 10,
                            output_tokens: 2,
                            cache_creation_input_tokens: 0,
                            cache_read_input_tokens: 0,
                        },
                    }),
                    Ok(StreamEvent::MessageEnd {
                        id: halter_protocol::MessageId::new(),
                        stop_reason: StopReason::EndTurn,
                        response_id: None,
                    }),
                ])
                .boxed());
            }

            let block_id = BlockId::new();
            let message_id = halter_protocol::MessageId::new();
            let tool_call_id = ToolCallId::new();
            Ok(stream::iter(vec![
                Ok(StreamEvent::MessageStart {
                    id: message_id.clone(),
                }),
                Ok(StreamEvent::ToolCallStart {
                    id: block_id.clone(),
                    tool_call_id,
                    name: halter_protocol::ToolName::from("write"),
                }),
                Ok(StreamEvent::ToolArgsDelta {
                    id: block_id.clone(),
                    delta: serde_json::json!({
                        "path": "note.txt",
                        "content": "hello from tool"
                    })
                    .to_string(),
                }),
                Ok(StreamEvent::ToolCallEnd { id: block_id }),
                Ok(StreamEvent::UsageUpdate {
                    usage: Usage {
                        input_tokens: 8,
                        output_tokens: 0,
                        cache_creation_input_tokens: 0,
                        cache_read_input_tokens: 0,
                    },
                }),
                Ok(StreamEvent::MessageEnd {
                    id: message_id,
                    stop_reason: StopReason::ToolUse,
                    response_id: None,
                }),
            ])
            .boxed())
        }
    }

    struct ToolThenFailProvider;

    #[async_trait]
    impl Provider for ToolThenFailProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }

        async fn stream(
            &self,
            request: ProviderRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
            if request
                .messages
                .iter()
                .any(|message| matches!(message, Message::Tool(_)))
            {
                return Ok(stream::iter(vec![Err(ProviderError::new(
                    "provider failed after tool",
                    false,
                ))])
                .boxed());
            }

            let block_id = BlockId::new();
            let message_id = halter_protocol::MessageId::new();
            Ok(stream::iter(vec![
                Ok(StreamEvent::MessageStart {
                    id: message_id.clone(),
                }),
                Ok(StreamEvent::ToolCallStart {
                    id: block_id.clone(),
                    tool_call_id: ToolCallId::new(),
                    name: ToolName::from("write"),
                }),
                Ok(StreamEvent::ToolArgsDelta {
                    id: block_id.clone(),
                    delta: serde_json::json!({
                        "path": "note.txt",
                        "content": "hello from tool"
                    })
                    .to_string(),
                }),
                Ok(StreamEvent::ToolCallEnd { id: block_id }),
                Ok(StreamEvent::MessageEnd {
                    id: message_id,
                    stop_reason: StopReason::ToolUse,
                    response_id: None,
                }),
            ])
            .boxed())
        }
    }

    #[async_trait]
    impl Provider for DuplicateToolCallProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }

        async fn stream(
            &self,
            request: ProviderRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
            if request
                .messages
                .iter()
                .any(|message| matches!(message, Message::Tool(_)))
            {
                return Ok(stream::iter(vec![
                    Ok(StreamEvent::MessageStart {
                        id: halter_protocol::MessageId::new(),
                    }),
                    Ok(StreamEvent::TextStart { id: BlockId::new() }),
                    Ok(StreamEvent::TextDelta {
                        id: BlockId::new(),
                        delta: "tool completed".to_owned(),
                    }),
                    Ok(StreamEvent::TextEnd { id: BlockId::new() }),
                    Ok(StreamEvent::MessageEnd {
                        id: halter_protocol::MessageId::new(),
                        stop_reason: StopReason::EndTurn,
                        response_id: None,
                    }),
                ])
                .boxed());
            }

            let first_block_id = BlockId::new();
            let second_block_id = BlockId::new();
            let tool_call_id = ToolCallId::from("call-dedupe");
            Ok(stream::iter(vec![
                Ok(StreamEvent::MessageStart {
                    id: halter_protocol::MessageId::new(),
                }),
                Ok(StreamEvent::ToolCallStart {
                    id: first_block_id.clone(),
                    tool_call_id: tool_call_id.clone(),
                    name: ToolName::from("count_test"),
                }),
                Ok(StreamEvent::ToolArgsDelta {
                    id: first_block_id.clone(),
                    delta: json!({ "value": 1 }).to_string(),
                }),
                Ok(StreamEvent::ToolCallEnd { id: first_block_id }),
                Ok(StreamEvent::ToolCallStart {
                    id: second_block_id.clone(),
                    tool_call_id,
                    name: ToolName::from("count_test"),
                }),
                Ok(StreamEvent::ToolArgsDelta {
                    id: second_block_id.clone(),
                    delta: json!({ "value": 1 }).to_string(),
                }),
                Ok(StreamEvent::ToolCallEnd {
                    id: second_block_id,
                }),
                Ok(StreamEvent::MessageEnd {
                    id: halter_protocol::MessageId::new(),
                    stop_reason: StopReason::ToolUse,
                    response_id: None,
                }),
            ])
            .boxed())
        }
    }

    #[derive(Debug)]
    struct FailingProvider;

    #[async_trait]
    impl Provider for FailingProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }

        async fn stream(
            &self,
            _request: ProviderRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
            Ok(stream::iter(vec![Err(ProviderError::new("provider exploded", false))]).boxed())
        }
    }

    /// Provider whose error is flagged retryable. Used to verify the
    /// runtime preserves the `retryable` bit on `TurnFailed`.
    #[derive(Debug)]
    struct RetryableFailingProvider;

    #[async_trait]
    impl Provider for RetryableFailingProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }

        async fn stream(
            &self,
            _request: ProviderRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
            Ok(stream::iter(vec![Err(ProviderError::new("rate limited", true))]).boxed())
        }
    }

    /// Provider that emits MessageStart and then blocks on its child
    /// cancel token. Models a long-running provider that responds to
    /// cancellation cooperatively — used for shutdown-drain tests.
    #[derive(Debug)]
    struct CancellableBlockingProvider {
        started: Arc<Notify>,
    }

    #[async_trait]
    impl Provider for CancellableBlockingProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }

        async fn stream(
            &self,
            _request: ProviderRequest,
            cancel: CancellationToken,
        ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
            let started = self.started.clone();
            let s = futures::stream::unfold(Some((cancel, started, false)), |state| async move {
                let (cancel, started, emitted) = state?;
                if !emitted {
                    started.notify_one();
                    return Some((
                        Ok(StreamEvent::MessageStart {
                            id: halter_protocol::MessageId::new(),
                        }),
                        Some((cancel, started, true)),
                    ));
                }
                cancel.cancelled().await;
                Some((Err(ProviderError::new("cancelled", false)), None))
            });
            Ok(s.boxed())
        }
    }

    /// Provider that emits MessageStart and then sleeps forever, ignoring
    /// the cancel token. Used to verify the shutdown drain deadline aborts
    /// uncooperative provider streams.
    #[derive(Debug)]
    struct UncancellableBlockingProvider {
        started: Arc<Notify>,
    }

    #[async_trait]
    impl Provider for UncancellableBlockingProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }

        async fn stream(
            &self,
            _request: ProviderRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
            let started = self.started.clone();
            let s = futures::stream::unfold(Some(started), |state| async move {
                let started = state?;
                started.notify_one();
                std::future::pending::<()>().await;
                #[allow(unreachable_code)]
                Some((
                    Ok(StreamEvent::MessageStart {
                        id: halter_protocol::MessageId::new(),
                    }),
                    None,
                ))
            });
            Ok(s.boxed())
        }
    }

    #[tokio::test]
    async fn turn_failed_carries_retryable_flag_from_provider_error() {
        // AC2.5: when a provider error is flagged retryable, TurnFailed
        // surfaces that flag rather than silently dropping it.
        let temp = tempfile::tempdir().expect("tempdir");
        let services = configured_services(Arc::new(RetryableFailingProvider), temp.path());
        let runtime = SessionRuntime::new(services.clone());
        let session = new_session(&runtime, temp.path()).await;

        let events = session
            .submit_turn(Turn::user("retryable failure"))
            .await
            .expect("submit turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("collect events");

        let live_failure = events
            .iter()
            .find_map(|event| match &event.payload {
                SessionEventPayload::TurnFailed { retryable, .. } => Some(*retryable),
                _ => None,
            })
            .expect("TurnFailed in live stream");
        assert!(live_failure, "live TurnFailed must preserve retryable=true");

        let replay = services
            .sessions
            .replay(session.session_id())
            .await
            .expect("replay");
        let persisted = replay
            .iter()
            .find_map(|event| match &event.payload {
                SessionEventPayload::TurnFailed { retryable, .. } => Some(*retryable),
                _ => None,
            })
            .expect("TurnFailed in persisted store");
        assert!(
            persisted,
            "persisted TurnFailed must preserve retryable=true"
        );
    }

    #[tokio::test]
    async fn turn_failed_non_retryable_default_path() {
        // AC2.5 negative: a provider error with retryable=false stays
        // false — guards against a misguided "default to true" patch.
        let temp = tempfile::tempdir().expect("tempdir");
        let services = configured_services(Arc::new(FailingProvider), temp.path());
        let runtime = SessionRuntime::new(services.clone());
        let session = new_session(&runtime, temp.path()).await;

        let events = session
            .submit_turn(Turn::user("non-retryable failure"))
            .await
            .expect("submit turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("collect events");

        let retryable = events
            .iter()
            .find_map(|event| match &event.payload {
                SessionEventPayload::TurnFailed { retryable, .. } => Some(*retryable),
                _ => None,
            })
            .expect("TurnFailed payload");
        assert!(!retryable);
    }

    #[tokio::test]
    async fn turn_failed_carries_originating_turn_id() {
        // AC2.4: the TurnId on the streamed and persisted TurnFailed must
        // match the TurnId of the submitted Turn.
        let temp = tempfile::tempdir().expect("tempdir");
        let services = configured_services(Arc::new(FailingProvider), temp.path());
        let runtime = SessionRuntime::new(services.clone());
        let session = new_session(&runtime, temp.path()).await;

        let turn = Turn::user("track id");
        let expected_id = turn.id.clone();
        let events = session
            .submit_turn(turn)
            .await
            .expect("submit turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("collect events");

        let live_id = events
            .iter()
            .find_map(|event| match &event.payload {
                SessionEventPayload::TurnFailed { turn_id, .. } => Some(turn_id.clone()),
                _ => None,
            })
            .expect("TurnFailed in stream");
        assert_eq!(live_id, expected_id, "stream turn_id must match submission");

        let replay = services
            .sessions
            .replay(session.session_id())
            .await
            .expect("replay");
        let persisted_id = replay
            .iter()
            .find_map(|event| match &event.payload {
                SessionEventPayload::TurnFailed { turn_id, .. } => Some(turn_id.clone()),
                _ => None,
            })
            .expect("TurnFailed in store");
        assert_eq!(
            persisted_id, expected_id,
            "persisted turn_id must match submission"
        );
    }

    #[tokio::test]
    async fn shutdown_drains_cooperative_in_flight_turn() {
        // AC2.3: shutdown cancels the provider stream's child token,
        // the provider exits, the spawned turn task settles, and the
        // drain reports completion within the deadline.
        let temp = tempfile::tempdir().expect("tempdir");
        let started = Arc::new(Notify::new());
        let services = configured_services(
            Arc::new(CancellableBlockingProvider {
                started: started.clone(),
            }),
            temp.path(),
        );
        let runtime = SessionRuntime::new(services.clone());
        let session = new_session(&runtime, temp.path()).await;

        let _stream = session
            .submit_turn(Turn::user("blocking turn"))
            .await
            .expect("submit turn");

        // Wait for the provider stream to actually start before signaling
        // shutdown — otherwise we race the spawn and never observe the
        // cooperative-cancellation path.
        started.notified().await;

        assert_eq!(services.turn_registry.in_flight_count(), 1);

        let report = runtime.shutdown(Duration::from_secs(2)).await;
        assert!(!report.timed_out, "cooperative drain must not time out");
        assert_eq!(report.turns_drained, 1);
        assert_eq!(report.turns_aborted, 0);
        assert!(services.turn_registry.is_shutting_down());
    }

    #[tokio::test]
    async fn shutdown_aborts_uncooperative_turn_after_deadline() {
        // AC2.8: a turn whose provider stream ignores cancellation must
        // still be aborted by the drain deadline rather than blocking
        // shutdown forever.
        let temp = tempfile::tempdir().expect("tempdir");
        let started = Arc::new(Notify::new());
        let services = configured_services(
            Arc::new(UncancellableBlockingProvider {
                started: started.clone(),
            }),
            temp.path(),
        );
        let runtime = SessionRuntime::new(services.clone());
        let session = new_session(&runtime, temp.path()).await;

        let _stream = session
            .submit_turn(Turn::user("stuck turn"))
            .await
            .expect("submit turn");
        started.notified().await;

        let report = runtime.shutdown(Duration::from_millis(100)).await;
        assert!(report.timed_out, "uncooperative drain must time out");
        assert!(
            report.turns_aborted >= 1,
            "at least one task must be aborted, got {report:?}"
        );
    }

    #[tokio::test]
    async fn submit_turn_after_shutdown_is_rejected() {
        // AC2.3 follow-on: the upfront shutdown check refuses to spawn
        // new turns once the registry is shutting down so callers fail
        // fast instead of seeing aborted-turn semantics.
        let temp = tempfile::tempdir().expect("tempdir");
        let services = configured_services(Arc::new(FakeProvider::default()), temp.path());
        let runtime = SessionRuntime::new(services.clone());
        let session = new_session(&runtime, temp.path()).await;

        let _ = runtime.shutdown(Duration::from_millis(0)).await;

        let err = match session.submit_turn(Turn::user("late submission")).await {
            Ok(_) => panic!("must reject post-shutdown submission"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("runtime is shutting down"),
            "unexpected error: {err}"
        );
    }

    #[derive(Debug)]
    struct RecordingProvider {
        requests: Arc<Mutex<Vec<ProviderRequest>>>,
        reply: &'static str,
    }

    impl RecordingProvider {
        fn new(requests: Arc<Mutex<Vec<ProviderRequest>>>, reply: &'static str) -> Self {
            Self { requests, reply }
        }
    }

    #[async_trait]
    impl Provider for RecordingProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }

        async fn stream(
            &self,
            request: ProviderRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
            self.requests.lock().expect("requests").push(request);
            let block_id = BlockId::new();
            let message_id = halter_protocol::MessageId::new();
            Ok(stream::iter(vec![
                Ok(StreamEvent::MessageStart {
                    id: message_id.clone(),
                }),
                Ok(StreamEvent::TextStart {
                    id: block_id.clone(),
                }),
                Ok(StreamEvent::TextDelta {
                    id: block_id.clone(),
                    delta: self.reply.to_owned(),
                }),
                Ok(StreamEvent::TextEnd { id: block_id }),
                Ok(StreamEvent::MessageEnd {
                    id: message_id,
                    stop_reason: StopReason::EndTurn,
                    response_id: None,
                }),
            ])
            .boxed())
        }
    }

    #[derive(Debug)]
    struct StreamingToolProvider;

    #[async_trait]
    impl Provider for StreamingToolProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }

        async fn stream(
            &self,
            request: ProviderRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
            if request
                .messages
                .iter()
                .any(|message| matches!(message, Message::Tool(_)))
            {
                return Ok(stream::iter(vec![
                    Ok(StreamEvent::MessageStart {
                        id: halter_protocol::MessageId::new(),
                    }),
                    Ok(StreamEvent::TextStart { id: BlockId::new() }),
                    Ok(StreamEvent::TextDelta {
                        id: BlockId::new(),
                        delta: "stream done".to_owned(),
                    }),
                    Ok(StreamEvent::TextEnd { id: BlockId::new() }),
                    Ok(StreamEvent::MessageEnd {
                        id: halter_protocol::MessageId::new(),
                        stop_reason: StopReason::EndTurn,
                        response_id: None,
                    }),
                ])
                .boxed());
            }

            let block_id = BlockId::new();
            Ok(stream::iter(vec![
                Ok(StreamEvent::MessageStart {
                    id: halter_protocol::MessageId::new(),
                }),
                Ok(StreamEvent::ToolCallStart {
                    id: block_id.clone(),
                    tool_call_id: ToolCallId::new(),
                    name: ToolName::from("stream_test"),
                }),
                Ok(StreamEvent::ToolArgsDelta {
                    id: block_id.clone(),
                    delta: json!({}).to_string(),
                }),
                Ok(StreamEvent::ToolCallEnd { id: block_id }),
                Ok(StreamEvent::MessageEnd {
                    id: halter_protocol::MessageId::new(),
                    stop_reason: StopReason::ToolUse,
                    response_id: None,
                }),
            ])
            .boxed())
        }
    }

    #[derive(Debug)]
    struct StreamingTestTool;

    #[derive(Debug)]
    struct CountingTool {
        executions: Arc<Mutex<usize>>,
    }

    impl CountingTool {
        fn new(executions: Arc<Mutex<usize>>) -> Self {
            Self { executions }
        }
    }

    #[async_trait]
    impl Tool for CountingTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: ToolName::from("count_test"),
                description: "Count tool executions".to_owned(),
                input_schema: json!({
                    "type": "object",
                    "properties": {}
                }),
                concurrency: ToolConcurrency::Exclusive,
                capabilities: ToolCapabilities::default(),
                provider_aliases: Default::default(),
            }
        }

        async fn execute(
            &self,
            _context: ToolContext,
            _input: serde_json::Value,
        ) -> anyhow::Result<ToolResult> {
            *self.executions.lock().expect("executions") += 1;
            Ok(ToolResult::Json {
                value: json!({ "ok": true }),
            })
        }
    }

    #[derive(Debug)]
    struct ParallelBatchProvider {
        tool_name: &'static str,
    }

    #[async_trait]
    impl Provider for ParallelBatchProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }

        async fn stream(
            &self,
            request: ProviderRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
            if request
                .messages
                .iter()
                .any(|message| matches!(message, Message::Tool(_)))
            {
                return Ok(stream::iter(vec![
                    Ok(StreamEvent::MessageStart {
                        id: halter_protocol::MessageId::new(),
                    }),
                    Ok(StreamEvent::TextStart { id: BlockId::new() }),
                    Ok(StreamEvent::TextDelta {
                        id: BlockId::new(),
                        delta: "done".to_owned(),
                    }),
                    Ok(StreamEvent::TextEnd { id: BlockId::new() }),
                    Ok(StreamEvent::MessageEnd {
                        id: halter_protocol::MessageId::new(),
                        stop_reason: StopReason::EndTurn,
                        response_id: None,
                    }),
                ])
                .boxed());
            }
            let first_block = BlockId::new();
            let second_block = BlockId::new();
            Ok(stream::iter(vec![
                Ok(StreamEvent::MessageStart {
                    id: halter_protocol::MessageId::new(),
                }),
                Ok(StreamEvent::ToolCallStart {
                    id: first_block.clone(),
                    tool_call_id: ToolCallId::new(),
                    name: ToolName::from(self.tool_name),
                }),
                Ok(StreamEvent::ToolArgsDelta {
                    id: first_block.clone(),
                    delta: json!({}).to_string(),
                }),
                Ok(StreamEvent::ToolCallEnd { id: first_block }),
                Ok(StreamEvent::ToolCallStart {
                    id: second_block.clone(),
                    tool_call_id: ToolCallId::new(),
                    name: ToolName::from(self.tool_name),
                }),
                Ok(StreamEvent::ToolArgsDelta {
                    id: second_block.clone(),
                    delta: json!({}).to_string(),
                }),
                Ok(StreamEvent::ToolCallEnd { id: second_block }),
                Ok(StreamEvent::MessageEnd {
                    id: halter_protocol::MessageId::new(),
                    stop_reason: StopReason::ToolUse,
                    response_id: None,
                }),
            ])
            .boxed())
        }
    }

    #[derive(Debug)]
    struct BarrierTool {
        barrier: Arc<tokio::sync::Barrier>,
        concurrency: ToolConcurrency,
        name: &'static str,
    }

    #[async_trait]
    impl Tool for BarrierTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: ToolName::from(self.name),
                description: "barrier-synchronized tool".to_owned(),
                input_schema: json!({ "type": "object", "properties": {} }),
                concurrency: self.concurrency,
                capabilities: ToolCapabilities::default(),
                provider_aliases: Default::default(),
            }
        }

        async fn execute(
            &self,
            _context: ToolContext,
            _input: serde_json::Value,
        ) -> anyhow::Result<ToolResult> {
            self.barrier.wait().await;
            Ok(ToolResult::Empty)
        }
    }

    #[tokio::test]
    async fn parallel_safe_tools_execute_concurrently() {
        // Two ParallelSafe tool calls in a single batch must run concurrently:
        // each awaits a 2-party barrier that only resolves when both are in
        // flight simultaneously. Serial execution would deadlock the barrier.
        let temp = tempfile::tempdir().expect("tempdir");
        let services = configured_services(
            Arc::new(ParallelBatchProvider {
                tool_name: "parallel_barrier",
            }),
            temp.path(),
        );
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        services.tools.register(Arc::new(BarrierTool {
            barrier: barrier.clone(),
            concurrency: ToolConcurrency::ParallelSafe,
            name: "parallel_barrier",
        }));
        let runtime = SessionRuntime::new(services.clone());
        let session = new_session(&runtime, temp.path()).await;

        let result = tokio::time::timeout(
            Duration::from_secs(5),
            session
                .submit_turn(Turn::user("run in parallel"))
                .await
                .expect("submit turn")
                .try_collect::<Vec<_>>(),
        )
        .await
        .expect("parallel-safe tools must not deadlock the barrier")
        .expect("collect events");

        let completed = result
            .iter()
            .filter(|event| {
                matches!(
                    event.payload,
                    SessionEventPayload::ToolExecutionCompleted { .. }
                )
            })
            .count();
        assert_eq!(
            completed, 2,
            "both parallel-safe tool executions must complete"
        );
    }

    #[tokio::test]
    async fn exclusive_tools_run_serially_in_batch() {
        // Two Exclusive tool calls in a single provider response must run in
        // distinct batches. Giving them a 2-party barrier means concurrent
        // execution would succeed and serial execution would hang — the
        // timeout branch catches the happy path and asserts serialization.
        let temp = tempfile::tempdir().expect("tempdir");
        let services = configured_services(
            Arc::new(ParallelBatchProvider {
                tool_name: "exclusive_barrier",
            }),
            temp.path(),
        );
        let barrier = Arc::new(tokio::sync::Barrier::new(2));
        services.tools.register(Arc::new(BarrierTool {
            barrier: barrier.clone(),
            concurrency: ToolConcurrency::Exclusive,
            name: "exclusive_barrier",
        }));
        let runtime = SessionRuntime::new(services.clone());
        let session = new_session(&runtime, temp.path()).await;

        let timed = tokio::time::timeout(
            Duration::from_millis(500),
            session
                .submit_turn(Turn::user("serialize exclusive"))
                .await
                .expect("submit turn")
                .try_collect::<Vec<_>>(),
        )
        .await;
        assert!(
            timed.is_err(),
            "exclusive tools must serialize; barrier should deadlock"
        );
    }

    #[async_trait]
    impl Tool for StreamingTestTool {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: ToolName::from("stream_test"),
                description: "Emit output before completing".to_owned(),
                input_schema: json!({
                    "type": "object",
                    "properties": {}
                }),
                concurrency: ToolConcurrency::Exclusive,
                capabilities: ToolCapabilities {
                    mutating: false,
                    requires_approval: false,
                    cancellable: false,
                    long_running: true,
                },
                provider_aliases: Default::default(),
            }
        }

        async fn execute(
            &self,
            context: ToolContext,
            _input: serde_json::Value,
        ) -> anyhow::Result<ToolResult> {
            context.emit.emit(ToolRuntimeEvent::ToolOutput {
                tool_name: "stream_test".to_owned(),
                chunk: "streamed chunk".to_owned(),
            });
            tokio::time::sleep(Duration::from_millis(300)).await;
            Ok(ToolResult::Json {
                value: json!({ "ok": true }),
            })
        }
    }
}
