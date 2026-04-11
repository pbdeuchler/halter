// pattern: Imperative Shell

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use anyhow::Context;
use arc_swap::ArcSwap;
use chrono::Utc;
use futures::stream::{BoxStream, StreamExt};
use halter_hooks::Hooks;
use halter_protocol::{
    AssistantMessage, AssistantPart, BlockId, CacheScope, ContentHash, Delivery,
    HookSessionStartSource, Message, MessageId, ModelId, ObservedState, PendingToolCall,
    PromptSegment, PromptSegmentId, ProviderError, ProviderRequest, ReplayMeta, ResourceSnapshot,
    SessionBlueprint, SessionEvent, SessionEventPayload, SessionId, SessionState, StopReason,
    StreamEvent, SystemMessage, ToolCall, ToolError, ToolExecutionOutcome, ToolResult,
    ToolResultMessage, Turn, TurnId, Usage, Volatility,
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
use crate::{
    ContextManager, EventBus, ExecutedHookDispatch, HookInvocationContext, PromptAssembler,
    run_notification, run_post_compact, run_post_tool_use, run_post_tool_use_failure,
    run_pre_compact, run_pre_tool_use, run_session_end, run_session_start, run_stop,
    run_user_prompt_submit,
};

pub type SessionEventStream = BoxStream<'static, anyhow::Result<SessionEvent>>;

pub struct RuntimeServices {
    pub resources: Arc<ResourceHandle>,
    pub models: Arc<ModelRegistry>,
    pub tools: Arc<ToolRuntime>,
    pub path_locks: Arc<PathLockMap>,
    pub tool_sessions: Arc<ToolSessionStore>,
    pub sessions: Arc<dyn SessionStore>,
    pub policy: Arc<dyn ToolPolicy>,
    pub prompt_assembler: Arc<dyn PromptAssembler>,
    pub context_manager: Arc<dyn ContextManager>,
    pub event_bus: Arc<EventBus>,
    pub max_tool_output_bytes: usize,
    pub shell_timeout_secs: u64,
}

#[derive(Debug, Clone)]
pub struct ResourceHandle {
    current: Arc<ArcSwap<ResourceState>>,
}

#[derive(Clone, Debug)]
struct ResourceState {
    snapshot: ResourceSnapshot,
    hooks: Arc<Hooks>,
    hook_warnings: Arc<Vec<String>>,
}

impl ResourceHandle {
    #[must_use]
    pub fn new(snapshot: ResourceSnapshot, hooks: Arc<Hooks>, hook_warnings: Vec<String>) -> Self {
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
    pub fn hook_warnings(&self) -> Arc<Vec<String>> {
        self.current.load().hook_warnings.clone()
    }

    pub fn replace(
        &self,
        snapshot: ResourceSnapshot,
        hooks: Arc<Hooks>,
        hook_warnings: Vec<String>,
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
    // pub max_tool_calls_per_turn: u32,
    pub subagent_depth: u32,
}

impl Default for SessionInit {
    fn default() -> Self {
        Self {
            session_id: None,
            parent_session_id: None,
            working_dir: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
            system_prompt_seed: Vec::new(),
            max_turns: None,
            default_model: None,
            subagent_model: None,
            // max_tool_calls_per_turn: 8,
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
}

#[derive(Clone)]
pub struct HalterSession {
    services: Arc<RuntimeServices>,
    session_id: SessionId,
}

struct SessionToolEventSink {
    call_id: halter_protocol::ToolCallId,
    events: Arc<Mutex<Vec<ToolRuntimeEvent>>>,
    live: Option<LiveTurnStream>,
}

impl ToolEventSink for SessionToolEventSink {
    fn emit(&self, event: ToolRuntimeEvent) {
        if let Some(live) = self.live.as_ref()
            && let Some(payload) = tool_runtime_event_payload(&self.call_id, event.clone())
        {
            live.emit_payload(payload);
        }
        self.events
            .lock()
            .expect("tool event lock poisoned")
            .push(event);
    }
}

struct ToolEventDrain {
    events: Arc<Mutex<Vec<ToolRuntimeEvent>>>,
}

impl ToolEventDrain {
    fn into_events(self) -> Vec<ToolRuntimeEvent> {
        self.events
            .lock()
            .expect("tool event lock poisoned")
            .drain(..)
            .collect()
    }
}

#[derive(Clone)]
struct LiveTurnStream {
    session_id: SessionId,
    tx: mpsc::UnboundedSender<anyhow::Result<SessionEvent>>,
    next_sequence: Arc<AtomicU64>,
}

impl LiveTurnStream {
    fn new(
        session_id: SessionId,
        tx: mpsc::UnboundedSender<anyhow::Result<SessionEvent>>,
        next_sequence: u64,
    ) -> Self {
        Self {
            session_id,
            tx,
            next_sequence: Arc::new(AtomicU64::new(next_sequence)),
        }
    }

    fn emit_payload(&self, payload: SessionEventPayload) {
        if !should_emit_live_payload(&payload) {
            return;
        }
        let event = SessionEvent {
            session_id: self.session_id.clone(),
            sequence: self.next_sequence.fetch_add(1, Ordering::Relaxed),
            delivery: Delivery::Lossless,
            payload,
        };
        let _ = self.tx.send(Ok(event));
    }

    fn emit_committed(&self, event: SessionEvent) {
        let _ = self.tx.send(Ok(event));
    }

    fn emit_error(&self, error: anyhow::Error) {
        let _ = self.tx.send(Err(error));
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
            stored.state.pending_session_start_source = Some(HookSessionStartSource::Resume);
            self.services
                .sessions
                .commit(session_id, None, Some(stored.state), Vec::new())
                .await?;
            return Ok(Some(HalterSession::new(
                self.services.clone(),
                session_id.clone(),
            )));
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
        hook_warnings: Vec<String>,
    ) {
        self.services
            .resources
            .replace(snapshot, hooks, hook_warnings);
    }
}

impl HalterSession {
    pub(crate) fn new(services: Arc<RuntimeServices>, session_id: SessionId) -> Self {
        Self {
            services,
            session_id,
        }
    }

    #[must_use]
    pub fn session_id(&self) -> &SessionId {
        &self.session_id
    }

    pub(crate) fn services(&self) -> &Arc<RuntimeServices> {
        &self.services
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
        let base_sequence = self.services.sessions.replay(&self.session_id).await?.len() as u64 + 1;
        let (tx, rx) = mpsc::unbounded_channel();
        let live = LiveTurnStream::new(self.session_id.clone(), tx, base_sequence);
        let session = self.clone();

        tokio::spawn(async move {
            live.emit_payload(SessionEventPayload::TurnStarted {
                turn_id: turn.id.clone(),
            });

            match session
                .run_turn(stored, turn.clone(), turn_cancel, Some(live.clone()))
                .await
            {
                Ok(turn_commit) => match session
                    .commit_and_publish(
                        Some(turn_commit.snapshot),
                        Some(turn_commit.state),
                        turn_commit.events,
                    )
                    .await
                {
                    Ok(committed) => {
                        for event in committed {
                            if matches!(event.payload, SessionEventPayload::TurnCompleted { .. }) {
                                live.emit_committed(event);
                            }
                        }
                    }
                    Err(error) => {
                        error!(
                            session_id = %session.session_id,
                            turn_id = %turn.id,
                            error = %error,
                            "failed to commit successful turn"
                        );
                        live.emit_error(error);
                    }
                },
                Err(error) => {
                    error!(
                        session_id = %session.session_id,
                        turn_id = %turn.id,
                        error = %error,
                        "turn failed before commit"
                    );
                    let turn_snapshot = session.services.resources.snapshot();
                    let failure_events = vec![
                        session.make_event(
                            0,
                            SessionEventPayload::TurnStarted {
                                turn_id: turn.id.clone(),
                            },
                        ),
                        session.make_event(
                            0,
                            SessionEventPayload::TurnFailed {
                                turn_id: turn.id.clone(),
                                error: error.to_string(),
                            },
                        ),
                    ];
                    match session
                        .commit_and_publish(Some(turn_snapshot), None, failure_events)
                        .await
                    {
                        Ok(committed) => {
                            for event in committed {
                                if matches!(event.payload, SessionEventPayload::TurnFailed { .. }) {
                                    live.emit_committed(event);
                                }
                            }
                        }
                        Err(commit_error) => {
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
            }
        });

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
        let mut state = stored.state;
        let mut events = Vec::new();
        let turn_id = TurnId::new();
        let hook_ctx = HookInvocationContext {
            turn_id: &turn_id,
            model: &stored.blueprint.default_model,
            working_dir: &stored.blueprint.working_dir,
        };
        let dispatch = run_session_end(self, &state, hook_ctx, reason).await?;
        self.record_hook_dispatch(&mut events, &dispatch, None);
        if dispatch.merged.block_reason.is_some() || dispatch.merged.stop_reason.is_some() {
            warn!(session_id = %self.session_id, reason, "hooks.ignored_block");
        }
        for message in apply_hook_side_effects(&mut state, &dispatch) {
            self.push_event(
                &mut events,
                SessionEventPayload::MessageItem { message },
                None,
            );
        }
        self.push_event(
            &mut events,
            SessionEventPayload::SessionShutdownComplete,
            None,
        );
        let _ = self.commit_and_publish(None, Some(state), events).await?;
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
        let mut state = stored.state;
        let turn_id = TurnId::new();
        let hook_ctx = HookInvocationContext {
            turn_id: &turn_id,
            model: &stored.blueprint.default_model,
            working_dir: &stored.blueprint.working_dir,
        };
        let dispatch = run_notification(self, &state, hook_ctx, notification_type, message).await?;
        let mut events = Vec::new();
        self.record_hook_dispatch(&mut events, &dispatch, None);
        for message in apply_hook_side_effects(&mut state, &dispatch) {
            self.push_event(
                &mut events,
                SessionEventPayload::MessageItem { message },
                None,
            );
        }
        let _ = self.commit_and_publish(None, Some(state), events).await?;
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
        let mut state = stored.state;
        let mut events = Vec::new();
        let turn_id = TurnId::new();
        let hook_ctx = HookInvocationContext {
            turn_id: &turn_id,
            model: &stored.blueprint.default_model,
            working_dir: &stored.blueprint.working_dir,
        };
        let pre_dispatch =
            run_pre_compact(self, &state, hook_ctx, trigger, custom_instructions).await?;
        self.record_hook_dispatch(&mut events, &pre_dispatch, None);
        for message in apply_hook_side_effects(&mut state, &pre_dispatch) {
            self.push_event(
                &mut events,
                SessionEventPayload::MessageItem { message },
                None,
            );
        }
        if pre_dispatch.merged.block_reason.is_some() {
            let _ = self.commit_and_publish(None, Some(state), events).await?;
            return Ok(());
        }

        let summary = compact_session_state(&mut state, custom_instructions);
        state.pending_session_start_source = Some(HookSessionStartSource::Compact);
        self.push_event(
            &mut events,
            SessionEventPayload::ContextCompacted {
                summary: summary.clone(),
            },
            None,
        );

        let post_dispatch = run_post_compact(self, &state, hook_ctx, trigger, &summary).await?;
        self.record_hook_dispatch(&mut events, &post_dispatch, None);
        for message in apply_hook_side_effects(&mut state, &post_dispatch) {
            self.push_event(
                &mut events,
                SessionEventPayload::MessageItem { message },
                None,
            );
        }

        let _ = self.commit_and_publish(None, Some(state), events).await?;
        Ok(())
    }

    async fn run_turn(
        &self,
        stored: StoredSession,
        turn: Turn,
        turn_cancel: CancellationToken,
        live: Option<LiveTurnStream>,
    ) -> anyhow::Result<TurnCommit> {
        let snapshot = self.services.resources.snapshot();
        let mut state = stored.state;
        let mut events = vec![self.make_event(
            0,
            SessionEventPayload::TurnStarted {
                turn_id: turn.id.clone(),
            },
        )];
        let mut turn_usage = Usage::default();
        let mut tool_calls_executed = 0u32;
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
                SessionEventPayload::Warning { message: warning },
                live.as_ref(),
            );
        }

        if let Some(source) = state.pending_session_start_source.take() {
            let hook_dispatch = run_session_start(self, &state, hook_ctx, source).await?;
            self.record_hook_dispatch(&mut events, &hook_dispatch, live.as_ref());
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
                self.push_event(
                    &mut events,
                    SessionEventPayload::MessageItem { message },
                    live.as_ref(),
                );
            }
        }

        let prompt_dispatch =
            run_user_prompt_submit(self, &state, hook_ctx, &turn.user_message.plain_text()).await?;
        self.record_hook_dispatch(&mut events, &prompt_dispatch, live.as_ref());
        for message in apply_hook_side_effects(&mut state, &prompt_dispatch) {
            self.push_event(
                &mut events,
                SessionEventPayload::MessageItem { message },
                live.as_ref(),
            );
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
                live.as_ref(),
            );
            events.push(self.make_event(
                0,
                SessionEventPayload::TurnCompleted {
                    turn_id: turn.id,
                    usage: turn_usage,
                },
            ));
            return Ok(TurnCommit {
                snapshot,
                state,
                events,
            });
        }

        state
            .messages
            .push(Message::User(turn.user_message.clone()));

        loop {
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
                )
                .await?;
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
                messages: plan.messages.clone(),
                tools: plan.tool_specs.clone(),
            };

            let provider_stream = provider.stream(request, turn_cancel.child_token()).await?;
            let materialized = materialize_assistant_message(provider_stream, &model).await?;
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

            let assistant_message = Message::Assistant(materialized.message.clone());
            state.messages.push(assistant_message.clone());
            for payload in materialized.events {
                self.push_event(&mut events, payload, live.as_ref());
            }
            self.push_event(
                &mut events,
                SessionEventPayload::MessageItem {
                    message: assistant_message,
                },
                live.as_ref(),
            );

            let tool_calls = assistant_tool_calls(&materialized.message);
            if tool_calls.is_empty() {
                let stop_dispatch = run_stop(
                    self,
                    &state,
                    HookInvocationContext {
                        turn_id: &turn.id,
                        model: &model.id,
                        working_dir: &stored.blueprint.working_dir,
                    },
                    Some(&materialized.message),
                    true,
                )
                .await?;
                self.record_hook_dispatch(&mut events, &stop_dispatch, live.as_ref());
                for message in apply_hook_side_effects(&mut state, &stop_dispatch) {
                    self.push_event(
                        &mut events,
                        SessionEventPayload::MessageItem { message },
                        live.as_ref(),
                    );
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
                        live.as_ref(),
                    );
                    continue;
                }

                info!(
                    session_id = %self.session_id,
                    turn_id = %turn.id,
                    input_tokens = turn_usage.input_tokens,
                    output_tokens = turn_usage.output_tokens,
                    "turn completed without tool calls"
                );
                events.push(self.make_event(
                    0,
                    SessionEventPayload::TurnCompleted {
                        turn_id: turn.id,
                        usage: turn_usage,
                    },
                ));
                return Ok(TurnCommit {
                    snapshot,
                    state,
                    events,
                });
            }

            tool_calls_executed += u32::try_from(tool_calls.len()).unwrap_or(u32::MAX);
            info!(
                session_id = %self.session_id,
                turn_id = %turn.id,
                tool_call_count = tool_calls.len(),
                tool_calls_executed,
                "assistant requested tool calls"
            );
            // if tool_calls_executed > stored.blueprint.max_tool_calls_per_turn {
            //     anyhow::bail!(
            //         "failed to submit turn: tool calls {} exceed max_tool_calls_per_turn {}",
            //         tool_calls_executed,
            //         stored.blueprint.max_tool_calls_per_turn
            //     );
            // }

            let tool_events = self
                .execute_tool_calls(
                    &stored.blueprint,
                    snapshot.clone(),
                    turn_cancel.child_token(),
                    &selected_models.default_model,
                    &selected_models.subagent_model,
                    &turn.id,
                    &mut state,
                    tool_calls,
                    live.clone(),
                )
                .await?;
            events.extend(tool_events);
        }
    }

    async fn execute_tool_calls(
        &self,
        blueprint: &SessionBlueprint,
        snapshot: Arc<ResourceSnapshot>,
        cancel: CancellationToken,
        effective_model: &ModelId,
        effective_subagent_model: &ModelId,
        turn_id: &halter_protocol::TurnId,
        state: &mut SessionState,
        tool_calls: Vec<ToolCall>,
        live: Option<LiveTurnStream>,
    ) -> anyhow::Result<Vec<SessionEvent>> {
        let mut events = Vec::new();

        for mut call in tool_calls {
            let pre_dispatch = run_pre_tool_use(
                self,
                state,
                HookInvocationContext {
                    turn_id,
                    model: effective_model,
                    working_dir: &blueprint.working_dir,
                },
                &call,
            )
            .await?;
            self.record_hook_dispatch(&mut events, &pre_dispatch, live.as_ref());
            for message in apply_hook_side_effects(state, &pre_dispatch) {
                self.push_event(
                    &mut events,
                    SessionEventPayload::MessageItem { message },
                    live.as_ref(),
                );
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
                live.as_ref(),
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
                    live.as_ref(),
                );
                self.push_event(
                    &mut events,
                    SessionEventPayload::MessageItem { message },
                    live.as_ref(),
                );
                continue;
            }

            let (emit, tool_event_drain) = self.spawn_tool_event_sink(&call, live.clone());
            let context = halter_tools::ToolContext {
                session_id: self.session_id.clone(),
                working_dir: blueprint.working_dir.clone(),
                path_locks: self.services.path_locks.clone(),
                tool_sessions: self.services.tool_sessions.clone(),
                file_view: Arc::new(state.file_view_cache.clone()),
                snapshot: snapshot.clone(),
                cancel: cancel.child_token(),
                emit,
                policy: self.services.policy.clone(),
                max_tool_output_bytes: self.services.max_tool_output_bytes,
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

            let execution = self
                .services
                .tools
                .execute(&call.name.0, context.clone(), call.arguments.clone())
                .await;
            drop(context);
            for payload in tool_event_drain
                .into_events()
                .into_iter()
                .filter_map(|event| tool_runtime_event_payload(&call.id, event))
            {
                self.push_event(&mut events, payload, live.as_ref());
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
                    state,
                    HookInvocationContext {
                        turn_id,
                        model: effective_model,
                        working_dir: &blueprint.working_dir,
                    },
                    &call,
                    &content,
                )
                .await?;
                self.record_hook_dispatch(&mut events, &post_dispatch, live.as_ref());
                for message in apply_hook_side_effects(state, &post_dispatch) {
                    self.push_event(
                        &mut events,
                        SessionEventPayload::MessageItem { message },
                        live.as_ref(),
                    );
                }
                if let Some(updated_output) = post_dispatch.merged.updated_output {
                    content = tool_result_from_hook_value(updated_output);
                }
            } else if let Some(tool_error) = error.as_ref() {
                let post_dispatch = run_post_tool_use_failure(
                    self,
                    state,
                    HookInvocationContext {
                        turn_id,
                        model: effective_model,
                        working_dir: &blueprint.working_dir,
                    },
                    &call,
                    tool_error,
                )
                .await?;
                self.record_hook_dispatch(&mut events, &post_dispatch, live.as_ref());
                for message in apply_hook_side_effects(state, &post_dispatch) {
                    self.push_event(
                        &mut events,
                        SessionEventPayload::MessageItem { message },
                        live.as_ref(),
                    );
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
                live.as_ref(),
            );
            self.push_event(
                &mut events,
                SessionEventPayload::MessageItem { message },
                live.as_ref(),
            );
        }

        Ok(events)
    }

    async fn commit_and_publish(
        &self,
        snapshot: Option<Arc<ResourceSnapshot>>,
        state: Option<SessionState>,
        events: Vec<SessionEvent>,
    ) -> anyhow::Result<Vec<SessionEvent>> {
        debug!(
            session_id = %self.session_id,
            event_count = events.len(),
            replace_snapshot = snapshot.is_some(),
            replace_state = state.is_some(),
            "committing session events"
        );
        let committed = self
            .services
            .sessions
            .commit(&self.session_id, snapshot, state, events)
            .await?;
        for event in &committed {
            self.services.event_bus.publish(event.clone());
        }
        Ok(committed)
    }

    fn spawn_tool_event_sink(
        &self,
        call: &ToolCall,
        live: Option<LiveTurnStream>,
    ) -> (Arc<dyn ToolEventSink>, ToolEventDrain) {
        let events = Arc::new(Mutex::new(Vec::new()));
        (
            Arc::new(SessionToolEventSink {
                call_id: call.id.clone(),
                events: events.clone(),
                live,
            }) as Arc<dyn ToolEventSink>,
            ToolEventDrain { events },
        )
    }

    fn push_event(
        &self,
        events: &mut Vec<SessionEvent>,
        payload: SessionEventPayload,
        live: Option<&LiveTurnStream>,
    ) {
        if let Some(live) = live {
            live.emit_payload(payload.clone());
        }
        events.push(self.make_event(0, payload));
    }

    fn record_hook_dispatch(
        &self,
        events: &mut Vec<SessionEvent>,
        dispatch: &ExecutedHookDispatch,
        live: Option<&LiveTurnStream>,
    ) {
        for run in &dispatch.preview_runs {
            self.push_event(
                events,
                SessionEventPayload::HookStarted { run: run.clone() },
                live,
            );
        }
        for run in &dispatch.completed_runs {
            self.push_event(
                events,
                SessionEventPayload::HookCompleted { run: run.clone() },
                live,
            );
        }
    }

    fn make_event(&self, sequence: u64, payload: SessionEventPayload) -> SessionEvent {
        SessionEvent {
            session_id: self.session_id.clone(),
            sequence,
            delivery: Delivery::Lossless,
            payload,
        }
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

fn should_emit_live_payload(payload: &SessionEventPayload) -> bool {
    !matches!(
        payload,
        SessionEventPayload::TurnStarted { .. }
            | SessionEventPayload::TurnCompleted { .. }
            | SessionEventPayload::TurnFailed { .. }
    )
}

#[derive(Debug)]
struct TurnCommit {
    snapshot: Arc<ResourceSnapshot>,
    state: SessionState,
    events: Vec<SessionEvent>,
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
    let mut thinking_block: Option<PendingThinkingBlock> = None;
    let mut tool_call_blocks: std::collections::BTreeMap<BlockId, PendingToolCallBlock> =
        std::collections::BTreeMap::new();

    while let Some(item) = provider_stream.next().await {
        match item {
            Ok(StreamEvent::MessageStart { id }) => {
                message_id = id;
            }
            Ok(StreamEvent::TextStart { .. }) => {}
            Ok(StreamEvent::TextDelta { delta, .. }) => {
                text_buffer.push_str(&delta);
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
                thinking.text.push_str(&delta);
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
                pending.arguments.push_str(&delta);
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
                ..
            }) => {
                stop_reason = ended_reason;
            }
            Ok(StreamEvent::ProviderWarning { message }) => {
                warn!(provider = %model.provider, message = %message, "provider emitted warning");
            }
            Ok(StreamEvent::Error { error }) | Err(error) => {
                error!(provider = %model.provider, error = %error.message, "provider stream failed");
                anyhow::bail!(error.message);
            }
        }
    }

    flush_text_buffer(&mut parts, &mut text_buffer);
    if !tool_call_blocks.is_empty() {
        anyhow::bail!("failed to materialize tool call: unterminated tool call block");
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

fn compact_session_state(state: &mut SessionState, custom_instructions: Option<&str>) -> String {
    const RETAINED_MESSAGES: usize = 8;

    if state.messages.len() <= RETAINED_MESSAGES {
        return "No compaction needed.".to_owned();
    }

    let split_index = state.messages.len() - RETAINED_MESSAGES;
    let summary_body = state.messages[..split_index]
        .iter()
        .map(render_message_for_summary)
        .filter(|line| !line.is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    let summary =
        if let Some(instructions) = custom_instructions.filter(|value| !value.trim().is_empty()) {
            format!("{instructions}\n\n{summary_body}")
        } else {
            summary_body
        };

    state.summaries.push(halter_protocol::SummarySlice {
        id: MessageId::new().0,
        text: summary.clone(),
    });
    state.messages = state.messages.split_off(split_index);

    summary
}

fn render_message_for_summary(message: &Message) -> String {
    match message {
        Message::System(message) => format!("system: {}", message.text),
        Message::User(message) => format!("user: {}", message.plain_text()),
        Message::Assistant(message) => {
            format!("assistant: {}", render_assistant_summary_text(message))
        }
        Message::Tool(message) => match &message.content {
            ToolResult::Empty => "tool: empty".to_owned(),
            ToolResult::Text { text } => format!("tool: {text}"),
            ToolResult::Json { value } => format!("tool: {value}"),
        },
    }
}

fn render_assistant_summary_text(message: &AssistantMessage) -> String {
    message
        .parts
        .iter()
        .filter_map(|part| match part {
            AssistantPart::Text { text } => Some(text.as_str()),
            AssistantPart::Thinking(_) | AssistantPart::ToolCall(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
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
    let blueprint = SessionBlueprint {
        session_id: session_id.clone(),
        parent_session_id: init.parent_session_id,
        default_model: selected_models.default_model,
        subagent_model: selected_models.subagent_model,
        snapshot_revision: snapshot.revision.clone(),
        working_dir: init.working_dir,
        system_prompt_seed: init.system_prompt_seed,
        max_turns: init.max_turns,
        // max_tool_calls_per_turn: init.max_tool_calls_per_turn,
        subagent_depth: init.subagent_depth,
    };
    info!(
        session_id = %session_id,
        default_model = %blueprint.default_model,
        subagent_model = %blueprint.subagent_model,
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

    let started = SessionEvent {
        session_id: session_id.clone(),
        sequence: 0,
        delivery: Delivery::Lossless,
        payload: SessionEventPayload::SessionStarted,
    };
    let committed = services
        .sessions
        .commit(&session_id, None, None, vec![started])
        .await?;
    for event in committed {
        services.event_bus.publish(event);
    }

    Ok(HalterSession::new(services, session_id))
}

fn observe_state(working_dir: PathBuf) -> ObservedState {
    ObservedState {
        cwd: working_dir,
        git_branch: None,
        git_dirty: None,
        now_utc: Utc::now(),
        env_facts: Default::default(),
    }
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
            models: Arc::new(ModelRegistry::new()),
            tools: Arc::new(ToolRuntime::new()),
            path_locks: Arc::new(PathLockMap::default()),
            tool_sessions: Arc::new(ToolSessionStore::default()),
            sessions: Arc::new(halter_session::InMemorySessionStore::default()),
            policy: Arc::new(halter_tools::DefaultToolPolicy::new(Default::default())),
            prompt_assembler: Arc::new(crate::DefaultPromptAssembler),
            context_manager: Arc::new(DefaultContextManager::default()),
            event_bus: Arc::new(EventBus::default()),
            max_tool_output_bytes: 262_144,
            shell_timeout_secs: 30,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use async_trait::async_trait;
    use futures::stream::{self, BoxStream};
    use futures::{StreamExt, TryStreamExt};
    use halter_hooks::{HookRegistrySource, HooksFile};
    use halter_protocol::{
        ApiKind, BlockId, HookRunStatus, HookSessionStartSource, Message, ModelId, ModelRole,
        PluginId, ProviderCapabilities, ProviderError, ProviderKind, ProviderName, ProviderRequest,
        ResolvedModel, StopReason, StreamEvent, ToolCallId, ToolCapabilities, ToolConcurrency,
        ToolName, ToolResult, ToolSpec, Turn,
    };
    use halter_providers::{FakeProvider, Provider};
    use halter_tools::{
        DefaultToolPolicy, PolicySettings, Tool, ToolContext, register_builtin_tools,
    };
    use serde_json::json;

    use super::*;

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
        let session = runtime
            .new_session(SessionInit {
                working_dir: temp.path().to_path_buf(),
                ..SessionInit::default()
            })
            .await
            .expect("session");

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
        services.resources.replace(
            ResourceSnapshot::empty(),
            Arc::new(Hooks::from_sources(vec![HookRegistrySource {
                plugin_id: PluginId::from("test-plugin"),
                plugin_root: temp.path().to_path_buf(),
                source_path: temp.path().join("hooks/hooks.json"),
                allowed_http_hosts: Vec::new(),
                allowed_env_vars: Vec::new(),
                file: hooks_file,
            }])),
            Vec::new(),
        );

        let runtime = SessionRuntime::new(services.clone());
        let session = runtime
            .new_session(SessionInit {
                working_dir: temp.path().to_path_buf(),
                ..SessionInit::default()
            })
            .await
            .expect("session");

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
        services.resources.replace(
            ResourceSnapshot::empty(),
            Arc::new(Hooks::from_sources(vec![HookRegistrySource {
                plugin_id: PluginId::from("test-plugin"),
                plugin_root: temp.path().to_path_buf(),
                source_path: temp.path().join("hooks/hooks.json"),
                allowed_http_hosts: Vec::new(),
                allowed_env_vars: Vec::new(),
                file: hooks_file,
            }])),
            Vec::new(),
        );

        let runtime = SessionRuntime::new(Arc::new(services));
        let session = runtime
            .new_session(SessionInit {
                working_dir: temp.path().to_path_buf(),
                ..SessionInit::default()
            })
            .await
            .expect("session");

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
    async fn hook_warnings_emit_warning_events_on_next_turn() {
        let temp = tempfile::tempdir().expect("tempdir");
        let services = configured_services(Arc::new(FakeProvider::default()), temp.path());
        services.resources.replace(
            ResourceSnapshot::empty(),
            Arc::new(Hooks::default()),
            vec!["hook warning".to_owned()],
        );
        let runtime = SessionRuntime::new(services);
        let session = runtime
            .new_session(SessionInit {
                working_dir: temp.path().to_path_buf(),
                ..SessionInit::default()
            })
            .await
            .expect("session");

        let events = session
            .submit_turn(Turn::user("hello"))
            .await
            .expect("submit turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("collect events");

        assert!(events.iter().any(|event| matches!(
            &event.payload,
            SessionEventPayload::Warning { message } if message == "hook warning"
        )));
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
        services.resources.replace(
            ResourceSnapshot::empty(),
            Arc::new(Hooks::from_sources(vec![HookRegistrySource {
                plugin_id: PluginId::from("test-plugin"),
                plugin_root: temp.path().to_path_buf(),
                source_path: temp.path().join("hooks/hooks.json"),
                allowed_http_hosts: Vec::new(),
                allowed_env_vars: Vec::new(),
                file: hooks_file,
            }])),
            Vec::new(),
        );
        let runtime = SessionRuntime::new(services.clone());
        let session = runtime
            .new_session(SessionInit {
                working_dir: temp.path().to_path_buf(),
                ..SessionInit::default()
            })
            .await
            .expect("session");

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
        let session = runtime
            .new_session(SessionInit {
                working_dir: temp.path().to_path_buf(),
                ..SessionInit::default()
            })
            .await
            .expect("session");

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
            .commit(session.session_id(), None, Some(stored.state), Vec::new())
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
        assert!(!stored.state.summaries.is_empty());
        assert!(stored.state.messages.len() <= 8);
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
    async fn submit_turn_failure_preserves_valid_transcript() {
        let temp = tempfile::tempdir().expect("tempdir");
        let services = configured_services(Arc::new(FailingProvider), temp.path());
        let runtime = SessionRuntime::new(services.clone());
        let session = runtime
            .new_session(SessionInit {
                working_dir: temp.path().to_path_buf(),
                ..SessionInit::default()
            })
            .await
            .expect("session");

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
        assert!(stored.state.messages.is_empty());
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
        assert!(
            !events
                .iter()
                .any(|event| matches!(event.payload, SessionEventPayload::MessageItem { .. }))
        );
    }

    #[tokio::test]
    async fn later_turns_commit_latest_resource_snapshot() {
        let temp = tempfile::tempdir().expect("tempdir");
        let services = configured_services(Arc::new(FakeProvider::default()), temp.path());
        let runtime = SessionRuntime::new(services.clone());
        let session = runtime
            .new_session(SessionInit {
                working_dir: temp.path().to_path_buf(),
                ..SessionInit::default()
            })
            .await
            .expect("session");

        let mut reloaded = ResourceSnapshot::empty();
        reloaded.revision = halter_protocol::Revision::from("reloaded");
        runtime.replace_resources(reloaded, Arc::new(Hooks::default()), Vec::new());

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
        let session = runtime
            .new_session(SessionInit {
                working_dir: temp.path().to_path_buf(),
                ..SessionInit::default()
            })
            .await
            .expect("session");

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
        let session = runtime
            .new_session(SessionInit {
                working_dir: temp.path().to_path_buf(),
                ..SessionInit::default()
            })
            .await
            .expect("session");

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
    async fn submit_turn_streams_tool_output_before_tool_completion() {
        let temp = tempfile::tempdir().expect("tempdir");
        let services = configured_services(Arc::new(StreamingToolProvider), temp.path());
        services.tools.register(Arc::new(StreamingTestTool));
        let runtime = SessionRuntime::new(services);
        let session = runtime
            .new_session(SessionInit {
                working_dir: temp.path().to_path_buf(),
                ..SessionInit::default()
            })
            .await
            .expect("session");

        let mut events = session
            .submit_turn(Turn::user("stream tool output"))
            .await
            .expect("submit turn");

        let tool_output = tokio::time::timeout(Duration::from_millis(150), async {
            while let Some(event) = events.next().await {
                let event = event.expect("stream event");
                if let SessionEventPayload::ToolOutput { chunk, .. } = event.payload {
                    return chunk;
                }
            }
            panic!("tool output never arrived");
        })
        .await
        .expect("tool output should stream before tool completion");

        assert_eq!(tool_output, "streamed chunk");

        let remaining = events
            .try_collect::<Vec<_>>()
            .await
            .expect("collect remaining events");
        assert!(
            remaining
                .iter()
                .any(|event| matches!(event.payload, SessionEventPayload::TurnCompleted { .. }))
        );
    }

    fn configured_services(
        provider: Arc<dyn Provider>,
        working_dir: &Path,
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
        });
        models.register_provider(ProviderName::from("fake"), provider);
        services.models = Arc::new(models);
        services.policy = Arc::new(DefaultToolPolicy::new(PolicySettings {
            allowed_write_roots: vec![working_dir.to_path_buf()],
            ..PolicySettings::default()
        }));
        Arc::new(services)
    }

    fn resolved_test_model(id: &str, provider: &str, model: &str) -> ResolvedModel {
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
        }
    }

    #[derive(Debug)]
    struct ToolLoopProvider;

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
                }),
            ])
            .boxed())
        }
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
                }),
            ])
            .boxed())
        }
    }

    #[derive(Debug)]
    struct StreamingTestTool;

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
