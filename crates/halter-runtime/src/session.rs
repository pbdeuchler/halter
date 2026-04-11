// pattern: Imperative Shell

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::Context;
use arc_swap::ArcSwap;
use chrono::Utc;
use futures::stream::{self, BoxStream, StreamExt};
use halter_protocol::{
    AssistantMessage, AssistantPart, BlockId, Delivery, Message, MessageId, ModelId, ObservedState,
    PendingToolCall, PromptSegment, ProviderError, ProviderRequest, ReplayMeta, ResourceSnapshot,
    SessionBlueprint, SessionEvent, SessionEventPayload, SessionId, SessionState, StopReason,
    StreamEvent, ToolCall, ToolError, ToolExecutionOutcome, ToolResult, ToolResultMessage, Turn,
    Usage,
};
use halter_providers::ModelRegistry;
use halter_session::{SessionStore, StoredSession};
use halter_tools::{
    PathLockMap, SubagentControl, SubagentParentContext, ToolEventSink, ToolPolicy,
    ToolRuntime, ToolRuntimeEvent, ToolSessionStore,
};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

#[cfg(test)]
use crate::DefaultContextManager;
use crate::model_selection::select_models;
use crate::{ContextManager, EventBus, PromptAssembler};

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
    current: Arc<ArcSwap<ResourceSnapshot>>,
}

impl ResourceHandle {
    #[must_use]
    pub fn new(snapshot: ResourceSnapshot) -> Self {
        Self {
            current: Arc::new(ArcSwap::from_pointee(snapshot)),
        }
    }

    #[must_use]
    pub fn snapshot(&self) -> Arc<ResourceSnapshot> {
        self.current.load_full()
    }

    pub fn replace(&self, snapshot: ResourceSnapshot) {
        info!(revision = %snapshot.revision, "replaced resource snapshot");
        self.current.store(Arc::new(snapshot));
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
    events: Arc<Mutex<Vec<ToolRuntimeEvent>>>,
}

impl ToolEventSink for SessionToolEventSink {
    fn emit(&self, event: ToolRuntimeEvent) {
        self.events.lock().expect("tool event lock poisoned").push(event);
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
        Ok(existing.map(|_| HalterSession::new(self.services.clone(), session_id.clone())))
    }

    pub async fn list_sessions(&self) -> anyhow::Result<Vec<SessionBlueprint>> {
        let sessions = self.services.sessions.list_sessions().await?;
        debug!(session_count = sessions.len(), "listed sessions");
        Ok(sessions)
    }

    pub fn replace_resources(&self, snapshot: ResourceSnapshot) {
        self.services.resources.replace(snapshot);
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

        match self.run_turn(stored, turn.clone(), turn_cancel).await {
            Ok(turn_commit) => {
                self.commit_and_stream(
                    Some(turn_commit.snapshot),
                    Some(turn_commit.state),
                    turn_commit.events,
                )
                .await
            }
            Err(error) => {
                error!(
                    session_id = %self.session_id,
                    turn_id = %turn.id,
                    error = %error,
                    "turn failed before commit"
                );
                let turn_snapshot = self.services.resources.snapshot();
                let failure_events = vec![
                    self.make_event(
                        0,
                        SessionEventPayload::TurnStarted {
                            turn_id: turn.id.clone(),
                        },
                    ),
                    self.make_event(
                        0,
                        SessionEventPayload::TurnFailed {
                            turn_id: turn.id,
                            error: error.to_string(),
                        },
                    ),
                ];
                self.commit_and_publish(Some(turn_snapshot), None, failure_events)
                    .await?;
                Err(error)
            }
        }
    }

    pub async fn replay(&self) -> anyhow::Result<Vec<SessionEvent>> {
        self.services.sessions.replay(&self.session_id).await
    }

    async fn run_turn(
        &self,
        stored: StoredSession,
        turn: Turn,
        turn_cancel: CancellationToken,
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
            events.extend(
                materialized
                    .events
                    .into_iter()
                    .map(|payload| self.make_event(0, payload)),
            );
            events.push(self.make_event(
                0,
                SessionEventPayload::MessageItem {
                    message: assistant_message,
                },
            ));

            let tool_calls = assistant_tool_calls(&materialized.message);
            if tool_calls.is_empty() {
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
                    &selected_models.subagent_model,
                    &mut state,
                    tool_calls,
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
        effective_subagent_model: &ModelId,
        state: &mut SessionState,
        tool_calls: Vec<ToolCall>,
    ) -> anyhow::Result<Vec<SessionEvent>> {
        let mut events = Vec::new();

        for call in tool_calls {
            let (emit, tool_event_drain) = self.spawn_tool_event_sink(&call);
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
            info!(
                session_id = %self.session_id,
                tool_call_id = %call.id,
                tool_name = %call.name,
                "executing tool call"
            );
            state.pending_tool_calls.insert(
                call.id.clone(),
                PendingToolCall {
                    call: call.clone(),
                    submitted_at: Utc::now(),
                },
            );
            events.push(self.make_event(
                0,
                SessionEventPayload::ToolExecutionStarted { call: call.clone() },
            ));

            let execution = self
                .services
                .tools
                .execute(&call.name.0, context.clone(), call.arguments.clone())
                .await;
            drop(context);
            events.extend(
                tool_event_drain
                    .into_events()
                    .into_iter()
                    .filter_map(|event| tool_runtime_event_payload(&call.id, event))
                    .map(|payload| self.make_event(0, payload)),
            );

            let (content, error) = match execution {
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
            events
                .push(self.make_event(0, SessionEventPayload::ToolExecutionCompleted { outcome }));
            events.push(self.make_event(0, SessionEventPayload::MessageItem { message }));
        }

        Ok(events)
    }

    async fn commit_and_stream(
        &self,
        snapshot: Option<Arc<ResourceSnapshot>>,
        state: Option<SessionState>,
        events: Vec<SessionEvent>,
    ) -> anyhow::Result<SessionEventStream> {
        let committed = self.commit_and_publish(snapshot, state, events).await?;
        Ok(stream::iter(committed.into_iter().map(Ok)).boxed())
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

    fn spawn_tool_event_sink(&self, call: &ToolCall) -> (Arc<dyn ToolEventSink>, ToolEventDrain) {
        let _ = call;
        let events = Arc::new(Mutex::new(Vec::new()));
        (
            Arc::new(SessionToolEventSink {
                events: events.clone(),
            }) as Arc<dyn ToolEventSink>,
            ToolEventDrain { events },
        )
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
        ToolRuntimeEvent::ToolOutput { tool_name, chunk } => Some(SessionEventPayload::ToolOutput {
            call_id: call_id.clone(),
            tool_name: tool_name.into(),
            chunk,
        }),
        ToolRuntimeEvent::Started { .. } | ToolRuntimeEvent::Completed { .. } => None,
    }
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
struct MaterializedAssistantMessage {
    message: AssistantMessage,
    usage: Usage,
    events: Vec<SessionEventPayload>,
}

async fn materialize_assistant_message(
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

pub(crate) async fn create_session_seeded(
    services: Arc<RuntimeServices>,
    init: SessionInit,
    initial_state: SessionState,
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
            resources: Arc::new(ResourceHandle::new(snapshot)),
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

    use async_trait::async_trait;
    use futures::TryStreamExt;
    use futures::stream::{self, BoxStream};
    use halter_protocol::{
        ApiKind, BlockId, Message, ModelId, ModelRole, ProviderCapabilities, ProviderError,
        ProviderKind, ProviderName, ProviderRequest, ResolvedModel, StopReason, StreamEvent,
        ToolCallId, Turn,
    };
    use halter_providers::{FakeProvider, Provider};
    use halter_tools::{DefaultToolPolicy, PolicySettings, register_builtin_tools};

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

        let error = match session.submit_turn(Turn::user("will fail")).await {
            Ok(_) => panic!("submit turn should fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("provider exploded"));

        let stored = services
            .sessions
            .load_session(session.session_id())
            .await
            .expect("load session")
            .expect("session exists");
        assert!(stored.state.messages.is_empty());

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
        runtime.replace_resources(reloaded);

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

        let error = match session
            .submit_turn(Turn::user("hello").with_subagent_model("missing"))
            .await
        {
            Ok(_) => panic!("submit turn should fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("unknown model 'missing'"));
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
}
