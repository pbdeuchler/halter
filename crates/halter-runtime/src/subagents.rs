// pattern: Imperative Shell

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::Context;
use async_trait::async_trait;
use futures::TryStreamExt;
use halter_protocol::{
    AgentId, AgentName, CloseSubagentRequest, CloseSubagentResponse, PendingEvent,
    SendSubagentInputRequest, SessionEventPayload, SessionId, SpawnSubagentRequest, SubagentState,
    SubagentStatus, Turn, TurnId, Usage, WaitSubagentRequest, WaitSubagentResponse,
};
use halter_session::SessionCommitConflict;
use halter_tools::{SubagentControl, SubagentParentContext};
use tokio::sync::{Mutex, Notify};
use tokio::task::JoinHandle;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::session::{apply_hook_side_effects, create_session_seeded};
use crate::subagent_session::{
    build_subagent_session_init, build_subagent_state, extract_subagent_output,
    extract_subagent_usage,
};
use crate::{
    HalterSession, HookInvocationContext, RuntimeServices, run_subagent_start, run_subagent_stop,
};

#[derive(Clone)]
pub struct RuntimeSubagentControl {
    inner: Arc<RuntimeSubagentState>,
}

struct RuntimeSubagentState {
    services: Arc<RuntimeServices>,
    registry: Mutex<SubagentRegistry>,
    changed: Notify,
    version: AtomicU64,
}

#[derive(Default)]
struct SubagentRegistry {
    entries: HashMap<String, RegisteredSubagent>,
}

struct RegisteredSubagent {
    status: SubagentStatus,
    generation: u64,
    running: Option<RunningTurn>,
}

struct RunningTurn {
    cancel: CancellationToken,
    join_handle: JoinHandle<()>,
}

struct TurnOutcome {
    state: SubagentState,
    last_message: Option<String>,
    usage: Option<Usage>,
    error: Option<String>,
}

const PARENT_HOOK_DISPATCH_MAX_RETRIES: usize = 3;

/// Upper bound on `SubagentStop`-hook-driven turn resubmissions per subagent
/// task. A hook that returns a `block_reason` resubmits the blocked turn with
/// that reason as input; a hook that *always* blocks would otherwise loop
/// forever (each resubmission is a full provider turn). Tripping the cap
/// fails the subagent with a descriptive error instead.
const SUBAGENT_STOP_RESUBMISSION_CAP: u32 = 8;

fn active_subagent_count(registry: &SubagentRegistry) -> usize {
    registry
        .entries
        .values()
        .filter(|entry| {
            entry.running.is_some() || matches!(entry.status.state, SubagentState::Running)
        })
        .count()
}

impl RuntimeSubagentControl {
    #[must_use]
    pub fn new(services: Arc<RuntimeServices>) -> Self {
        Self {
            inner: Arc::new(RuntimeSubagentState {
                services,
                registry: Mutex::new(SubagentRegistry::default()),
                changed: Notify::new(),
                version: AtomicU64::new(0),
            }),
        }
    }

    fn signal_change(&self) {
        self.inner.version.fetch_add(1, Ordering::SeqCst);
        self.inner.changed.notify_waiters();
    }

    async fn reserve_subagent_slot(
        &self,
        parent_depth: u32,
        agent_id: &AgentId,
        status: SubagentStatus,
    ) -> anyhow::Result<()> {
        loop {
            let active = {
                let registry = self.inner.registry.lock().await;
                active_subagent_count(&registry)
            };
            self.inner
                .services
                .policy
                .check_subagent_spawn_typed(parent_depth, active)
                .await?;

            let mut registry = self.inner.registry.lock().await;
            let active_now = active_subagent_count(&registry);
            if active_now != active {
                continue;
            }
            registry.entries.insert(
                agent_id.0.clone(),
                RegisteredSubagent {
                    status,
                    generation: 0,
                    running: None,
                },
            );
            drop(registry);
            self.signal_change();
            return Ok(());
        }
    }

    async fn remove_reserved_subagent(&self, agent_id: &AgentId) {
        let mut registry = self.inner.registry.lock().await;
        registry.entries.remove(&agent_id.0);
        drop(registry);
        self.signal_change();
    }

    async fn start_turn(
        &self,
        agent_id: &AgentId,
        session_id: &SessionId,
        agent_type: Option<AgentName>,
        message: String,
    ) -> anyhow::Result<SubagentStatus> {
        let parent_session_id = self
            .inner
            .services
            .sessions
            .load_session(session_id)
            .await?
            .and_then(|stored| stored.blueprint.parent_session_id);
        let cancel = CancellationToken::new();
        let (generation, status) = {
            let mut registry = self.inner.registry.lock().await;
            let entry = registry.entries.get_mut(&agent_id.0).with_context(|| {
                format!(
                    "failed to execute subagent request: unknown agent '{}'",
                    agent_id.0
                )
            })?;
            if matches!(entry.status.state, SubagentState::Closed) {
                anyhow::bail!(
                    "failed to execute subagent request: agent '{}' is closed",
                    agent_id.0
                );
            }
            if entry.running.is_some() {
                anyhow::bail!(
                    "failed to execute subagent request: agent '{}' is still running; use wait_agent to wait for completion, or close_agent to stop it",
                    agent_id.0
                );
            }
            entry.generation = entry.generation.saturating_add(1);
            entry.status.task = message.clone();
            if let Some(ref agent_type) = agent_type {
                entry.status.agent_type = Some(agent_type.clone());
            }
            entry.status.state = SubagentState::Running;
            entry.status.last_message = None;
            entry.status.usage = None;
            entry.status.error = None;
            (entry.generation, entry.status.clone())
        };

        let services = self.inner.services.clone();
        let task_agent_id = agent_id.clone();
        let task_session_id = session_id.clone();
        let task_message = message.clone();
        let task_cancel = cancel.clone();
        let controller = self.clone();
        let session = HalterSession::new(services.clone(), task_session_id.clone())?;
        let join_handle = tokio::spawn(async move {
            controller
                .run_turn_task(
                    task_agent_id,
                    task_session_id,
                    parent_session_id,
                    agent_type.clone(),
                    generation,
                    task_message,
                    task_cancel,
                    session,
                )
                .await;
        });

        let mut registry = self.inner.registry.lock().await;
        let orphan = register_running_turn(
            &mut registry,
            agent_id,
            generation,
            RunningTurn {
                cancel,
                join_handle,
            },
        );
        drop(registry);
        if let Some(orphan) = orphan {
            // `close` (or entry removal) raced the spawn before the turn was
            // registered, so its cancel/abort found nothing to stop. Stop the
            // freshly spawned task here instead of silently detaching it.
            orphan.cancel.cancel();
            orphan.join_handle.abort();
        }
        self.signal_change();
        info!(
            agent_id = %status.agent_id,
            session_id = %status.session_id,
            task = %status.task,
            "started subagent turn"
        );
        Ok(status)
    }

    #[expect(clippy::too_many_arguments)]
    async fn run_turn_task(
        &self,
        agent_id: AgentId,
        session_id: SessionId,
        parent_session_id: Option<SessionId>,
        agent_type: Option<AgentName>,
        generation: u64,
        message: String,
        cancel: CancellationToken,
        session: HalterSession,
    ) {
        let mut next_input = message;
        let mut resubmissions = 0u32;
        let outcome = loop {
            let turn_events = match session
                .submit_turn_with_cancel(Turn::user(next_input.clone()), cancel.clone())
                .await
            {
                Ok(events) => match events.try_collect::<Vec<_>>().await {
                    Ok(events) => events,
                    Err(error) => {
                        break TurnOutcome {
                            state: if cancel.is_cancelled() {
                                SubagentState::Cancelled
                            } else {
                                SubagentState::Failed
                            },
                            last_message: None,
                            usage: None,
                            error: Some(error.to_string()),
                        };
                    }
                },
                Err(error) => {
                    break TurnOutcome {
                        state: if cancel.is_cancelled() {
                            SubagentState::Cancelled
                        } else {
                            SubagentState::Failed
                        },
                        last_message: None,
                        usage: None,
                        error: Some(error.to_string()),
                    };
                }
            };

            if cancel.is_cancelled() {
                break TurnOutcome {
                    state: SubagentState::Cancelled,
                    last_message: None,
                    usage: None,
                    error: None,
                };
            }
            let own_turn_events = turn_events
                .iter()
                .filter(|event| event.session_id == session_id)
                .cloned()
                .collect::<Vec<_>>();

            let Some(parent_session_id) = parent_session_id.as_ref() else {
                break TurnOutcome {
                    state: SubagentState::Completed,
                    last_message: extract_subagent_output(&own_turn_events),
                    usage: extract_subagent_usage(&own_turn_events),
                    error: None,
                };
            };

            let continuation = match self
                .run_subagent_stop_hooks(
                    parent_session_id,
                    &agent_id,
                    agent_type.as_ref(),
                    &session_id,
                )
                .await
            {
                Ok(value) => value,
                Err(error) => {
                    break TurnOutcome {
                        state: SubagentState::Failed,
                        last_message: None,
                        usage: None,
                        error: Some(error.to_string()),
                    };
                }
            };

            if let Some(next_message) = continuation {
                // Bounded: a SubagentStop hook that always blocks must not
                // resubmit turns forever (each resubmission is a full
                // provider turn).
                if resubmissions >= SUBAGENT_STOP_RESUBMISSION_CAP {
                    break TurnOutcome {
                        state: SubagentState::Failed,
                        last_message: None,
                        usage: extract_subagent_usage(&own_turn_events),
                        error: Some(format!(
                            "subagent stop hooks kept blocking; resubmission cap of {SUBAGENT_STOP_RESUBMISSION_CAP} reached"
                        )),
                    };
                }
                resubmissions += 1;
                next_input = next_message;
                continue;
            }

            break TurnOutcome {
                state: SubagentState::Completed,
                last_message: extract_subagent_output(&own_turn_events),
                usage: extract_subagent_usage(&own_turn_events),
                error: None,
            };
        };

        self.finish_turn(agent_id, generation, outcome).await;
    }

    async fn finish_turn(&self, agent_id: AgentId, generation: u64, outcome: TurnOutcome) {
        let mut registry = self.inner.registry.lock().await;
        let Some(entry) = registry.entries.get_mut(&agent_id.0) else {
            return;
        };
        if entry.generation != generation {
            return;
        }
        if matches!(entry.status.state, SubagentState::Closed) {
            entry.running = None;
            return;
        }

        entry.running = None;
        entry.status.state = outcome.state;
        entry.status.last_message = outcome.last_message;
        entry.status.usage = outcome.usage;
        entry.status.error = outcome.error;
        debug!(
            agent_id = %entry.status.agent_id,
            state = ?entry.status.state,
            "completed subagent turn"
        );
        drop(registry);
        self.signal_change();
    }

    async fn run_parent_hook_dispatch(
        &self,
        parent_session_id: &SessionId,
        dispatch: crate::ExecutedHookDispatch,
        block_is_ignored: bool,
    ) -> anyhow::Result<Option<String>> {
        for attempt in 0..=PARENT_HOOK_DISPATCH_MAX_RETRIES {
            let Some(stored) = self
                .inner
                .services
                .sessions
                .load_session(parent_session_id)
                .await?
            else {
                return Ok(None);
            };

            let expected_state = stored.state.clone();
            let mut next_state = expected_state.clone();
            let mut events = Vec::new();
            events.extend(dispatch.preview_runs.iter().cloned().map(|run| {
                PendingEvent::new(
                    parent_session_id.clone(),
                    halter_protocol::Delivery::Lossless,
                    SessionEventPayload::HookStarted { run },
                )
            }));
            events.extend(dispatch.completed_runs.iter().cloned().map(|run| {
                PendingEvent::new(
                    parent_session_id.clone(),
                    halter_protocol::Delivery::Lossless,
                    SessionEventPayload::HookCompleted { run },
                )
            }));
            for message in apply_hook_side_effects(&mut next_state, &dispatch) {
                events.push(PendingEvent::new(
                    parent_session_id.clone(),
                    halter_protocol::Delivery::Lossless,
                    SessionEventPayload::MessageItem { message },
                ));
            }

            match self
                .inner
                .services
                .sessions
                .commit(
                    parent_session_id,
                    None,
                    Some(expected_state),
                    Some(next_state),
                    events,
                )
                .await
            {
                Ok(committed) => {
                    if block_is_ignored
                        && (dispatch.merged.block_reason.is_some()
                            || dispatch.merged.stop_reason.is_some())
                    {
                        warn!(
                            session_id = %parent_session_id,
                            "hooks.ignored_block"
                        );
                    }
                    for event in committed {
                        if let Some(recorder) = &self.inner.services.trace_recorder {
                            recorder.record(&event);
                        }
                        self.inner.services.event_bus.publish(event);
                    }
                    return Ok(dispatch.merged.block_reason.clone());
                }
                Err(error)
                    if error.downcast_ref::<SessionCommitConflict>().is_some()
                        && attempt < PARENT_HOOK_DISPATCH_MAX_RETRIES =>
                {
                    warn!(
                        session_id = %parent_session_id,
                        attempt = attempt + 1,
                        "hooks.parent_state_conflict_retry"
                    );
                    continue;
                }
                Err(error) => return Err(error),
            }
        }

        // The loop body always either returns or `continue`s. Reaching this
        // line means it fell through after exhausting retries; surface that as
        // an error rather than silently reporting success (the previous shape
        // returned `Ok(continuation)` and erased the conflict).
        Err(anyhow::anyhow!(
            "hook dispatch exhausted {} retries due to session commit conflict",
            PARENT_HOOK_DISPATCH_MAX_RETRIES
        ))
    }

    async fn run_subagent_start_hooks(
        &self,
        parent: &SubagentParentContext,
        status: &SubagentStatus,
    ) -> anyhow::Result<()> {
        let Some(stored) = self
            .inner
            .services
            .sessions
            .load_session(&parent.blueprint.session_id)
            .await?
        else {
            return Ok(());
        };
        let turn_id = TurnId::new();
        let session = HalterSession::new(
            self.inner.services.clone(),
            parent.blueprint.session_id.clone(),
        )?;
        let fired_hook_ids = stored
            .state
            .fired_hook_ids
            .iter()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        let dispatch = run_subagent_start(
            &session,
            &fired_hook_ids,
            HookInvocationContext {
                turn_id: &turn_id,
                model: &stored.blueprint.default_model,
                working_dir: &stored.blueprint.working_dir,
            },
            &status.agent_id,
            status
                .agent_type
                .as_ref()
                .map_or("default", |agent_type| agent_type.0.as_str()),
            &parent.blueprint.session_id,
        )
        .await?;
        let _ = self
            .run_parent_hook_dispatch(&parent.blueprint.session_id, dispatch, true)
            .await?;
        Ok(())
    }

    async fn run_subagent_stop_hooks(
        &self,
        parent_session_id: &SessionId,
        agent_id: &AgentId,
        agent_type: Option<&AgentName>,
        child_session_id: &SessionId,
    ) -> anyhow::Result<Option<String>> {
        let Some(stored) = self
            .inner
            .services
            .sessions
            .load_session(parent_session_id)
            .await?
        else {
            return Ok(None);
        };
        let turn_id = TurnId::new();
        let session = HalterSession::new(self.inner.services.clone(), parent_session_id.clone())?;
        let transcript_path = self
            .inner
            .services
            .sessions
            .transcript_path(child_session_id);
        let fired_hook_ids = stored
            .state
            .fired_hook_ids
            .iter()
            .cloned()
            .collect::<std::collections::BTreeSet<_>>();
        let dispatch = run_subagent_stop(
            &session,
            &fired_hook_ids,
            HookInvocationContext {
                turn_id: &turn_id,
                model: &stored.blueprint.default_model,
                working_dir: &stored.blueprint.working_dir,
            },
            agent_id,
            agent_type.map_or("default", |agent_type| agent_type.0.as_str()),
            transcript_path.as_deref(),
        )
        .await?;
        self.run_parent_hook_dispatch(parent_session_id, dispatch, false)
            .await
    }

    async fn terminal_status_for_targets(
        &self,
        targets: &[AgentId],
    ) -> anyhow::Result<Option<SubagentStatus>> {
        let registry = self.inner.registry.lock().await;
        let statuses = load_target_statuses(&registry, targets)?;
        Ok(statuses.into_iter().find(|status| status.is_terminal()))
    }
}

#[async_trait]
impl SubagentControl for RuntimeSubagentControl {
    async fn spawn(
        &self,
        parent: &SubagentParentContext,
        request: SpawnSubagentRequest,
    ) -> anyhow::Result<SubagentStatus> {
        if request.message.trim().is_empty() {
            anyhow::bail!("failed to execute spawn_agent tool: message cannot be empty");
        }

        let session_id = SessionId::new();
        let agent_id = AgentId::new();
        let init = build_subagent_session_init(parent, &session_id, &request)?;
        let state =
            build_subagent_state(parent, &session_id, &request.message, request.fork_context);
        let status = SubagentStatus {
            agent_id: agent_id.clone(),
            session_id: session_id.clone(),
            agent_type: request.agent_type.clone(),
            task: request.message.clone(),
            state: SubagentState::Running,
            last_message: None,
            usage: None,
            error: None,
        };

        self.reserve_subagent_slot(parent.blueprint.subagent_depth, &agent_id, status.clone())
            .await?;

        if let Err(error) = create_session_seeded(
            self.inner.services.clone(),
            init,
            state,
            parent.snapshot.clone(),
        )
        .await
        {
            self.remove_reserved_subagent(&agent_id).await;
            return Err(error);
        }

        if let Err(error) = self.run_subagent_start_hooks(parent, &status).await {
            self.remove_reserved_subagent(&agent_id).await;
            return Err(error);
        }

        self.start_turn(
            &agent_id,
            &session_id,
            request.agent_type.clone(),
            request.message,
        )
        .await
    }

    async fn send_input(
        &self,
        request: SendSubagentInputRequest,
    ) -> anyhow::Result<SubagentStatus> {
        if request.message.trim().is_empty() {
            anyhow::bail!("failed to execute send_input tool: message cannot be empty");
        }

        let (session_id, agent_type) = {
            let registry = self.inner.registry.lock().await;
            let entry = registry.entries.get(&request.target.0).with_context(|| {
                format!(
                    "failed to execute send_input tool: unknown agent '{}'",
                    request.target.0
                )
            })?;
            if entry.running.is_some() {
                anyhow::bail!(
                    "failed to execute send_input tool: agent '{}' is still running; use wait_agent to wait for completion, or close_agent to stop it",
                    request.target.0
                );
            }
            if matches!(entry.status.state, SubagentState::Closed) {
                anyhow::bail!(
                    "failed to execute send_input tool: agent '{}' is closed",
                    request.target.0
                );
            }
            (
                entry.status.session_id.clone(),
                entry.status.agent_type.clone(),
            )
        };

        self.start_turn(&request.target, &session_id, agent_type, request.message)
            .await
    }

    async fn wait(&self, request: WaitSubagentRequest) -> anyhow::Result<WaitSubagentResponse> {
        if request.targets.is_empty() {
            anyhow::bail!("failed to execute wait_agent tool: targets cannot be empty");
        }

        if let Some(status) = self.terminal_status_for_targets(&request.targets).await? {
            return Ok(WaitSubagentResponse {
                status: Some(status),
                timed_out: false,
                target_statuses: Vec::new(),
            });
        }

        let wait_for_status = async {
            loop {
                if let Some(status) = self.terminal_status_for_targets(&request.targets).await? {
                    return anyhow::Result::<SubagentStatus>::Ok(status);
                }
                let version = self.inner.version.load(Ordering::SeqCst);
                let notified = self.inner.changed.notified();
                if self.inner.version.load(Ordering::SeqCst) != version {
                    continue;
                }
                notified.await;
            }
        };

        match request.timeout_ms {
            Some(timeout_ms) => {
                match timeout(Duration::from_millis(timeout_ms), wait_for_status).await {
                    Ok(status) => Ok(WaitSubagentResponse {
                        status: Some(status?),
                        timed_out: false,
                        target_statuses: Vec::new(),
                    }),
                    Err(_) => {
                        let registry = self.inner.registry.lock().await;
                        let target_statuses = match load_target_statuses(
                            &registry,
                            &request.targets,
                        ) {
                            Ok(statuses) => statuses,
                            Err(error) => {
                                warn!(
                                    error = %error,
                                    "failed to snapshot subagent statuses after wait_agent timeout"
                                );
                                Vec::new()
                            }
                        };
                        Ok(WaitSubagentResponse {
                            status: None,
                            timed_out: true,
                            target_statuses,
                        })
                    }
                }
            }
            None => Ok(WaitSubagentResponse {
                status: Some(wait_for_status.await?),
                timed_out: false,
                target_statuses: Vec::new(),
            }),
        }
    }

    async fn close(&self, request: CloseSubagentRequest) -> anyhow::Result<CloseSubagentResponse> {
        let previous_status = {
            let mut registry = self.inner.registry.lock().await;
            let entry = registry
                .entries
                .get_mut(&request.target.0)
                .with_context(|| {
                    format!(
                        "failed to execute close_agent tool: unknown agent '{}'",
                        request.target.0
                    )
                })?;
            let previous = entry.status.clone();
            entry.generation = entry.generation.saturating_add(1);
            let closed_running_turn = entry.running.is_some();
            if let Some(running) = entry.running.take() {
                running.cancel.cancel();
                running.join_handle.abort();
            }
            entry.status.state = SubagentState::Closed;
            entry.status.error = closed_running_turn
                .then(|| "closed by close_agent (work was cancelled)".to_owned());
            previous
        };

        warn!(
            agent_id = %previous_status.agent_id,
            session_id = %previous_status.session_id,
            "closed subagent"
        );
        self.signal_change();
        Ok(CloseSubagentResponse { previous_status })
    }
}

/// Register `running` for `agent_id` when the entry is still at the spawning
/// generation and in the `Running` state; otherwise hand the turn back to the
/// caller as an orphan that must be cancelled and aborted. `close` can race
/// `start_turn` in the window between task spawn and registration: it bumps
/// the generation (and may mark the entry `Closed`) but finds no
/// `RunningTurn` to stop, so the raced spawn would otherwise run a full
/// uncancelled provider turn after the agent was closed.
fn register_running_turn(
    registry: &mut SubagentRegistry,
    agent_id: &AgentId,
    generation: u64,
    running: RunningTurn,
) -> Option<RunningTurn> {
    match registry.entries.get_mut(&agent_id.0) {
        Some(entry)
            if entry.generation == generation
                && matches!(entry.status.state, SubagentState::Running) =>
        {
            entry.running = Some(running);
            None
        }
        _ => Some(running),
    }
}

fn load_target_statuses(
    registry: &SubagentRegistry,
    targets: &[AgentId],
) -> anyhow::Result<Vec<SubagentStatus>> {
    targets
        .iter()
        .map(|target| {
            registry
                .entries
                .get(&target.0)
                .map(|entry| entry.status.clone())
                .with_context(|| {
                    format!(
                        "failed to execute wait_agent tool: unknown agent '{}'",
                        target.0
                    )
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use futures::StreamExt;
    use futures::stream::{self, BoxStream};
    use halter_protocol::{
        ApiKind, BlockId, Message, ModelId, ModelRole, ProviderCapabilities, ProviderError,
        ProviderKind, ProviderName, ProviderRequest, ResolvedModel, StopReason, StreamEvent,
    };
    use halter_providers::{ModelRegistry, Provider};
    use halter_session::InMemorySessionStore;
    use halter_tools::{
        DefaultToolPolicy, PathLockMap, PolicySettings, ToolRuntime, ToolSessionStore,
    };

    use super::*;
    use crate::{DefaultContextManager, DefaultPromptAssembler, EventBus, ResourceHandle};

    #[tokio::test]
    async fn spawn_and_wait_complete_child_session() {
        let provider_requests = Arc::new(Mutex::new(Vec::<ProviderRequest>::new()));
        let services = test_services(Arc::new(RecordingProvider::new(provider_requests.clone())));
        let control = RuntimeSubagentControl::new(services);
        let parent = parent_context();

        let status = control
            .spawn(
                &parent,
                SpawnSubagentRequest {
                    message: "delegate this".to_owned(),
                    agent_type: None,
                    fork_context: true,
                    model: None,
                },
            )
            .await
            .expect("spawn");

        assert_eq!(status.state, SubagentState::Running);
        let waited = control
            .wait(WaitSubagentRequest {
                targets: vec![status.agent_id.clone()],
                timeout_ms: Some(5_000),
            })
            .await
            .expect("wait");

        assert!(!waited.timed_out);
        assert!(waited.target_statuses.is_empty());
        let waited_status = waited.status.expect("completed status");
        assert_eq!(waited_status.state, SubagentState::Completed);
        assert_eq!(
            waited_status.last_message.as_deref(),
            Some("child reply [subagent/model]")
        );

        let requests = provider_requests.lock().expect("requests");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].model.id, ModelId::from("subagent"));
        assert_eq!(requests[0].messages.len(), parent.state.messages.len() + 1);
        assert_eq!(requests[0].messages[0], parent.state.messages[0]);
        assert!(matches!(
            &requests[0].messages[1],
            Message::User(user) if user.plain_text() == "delegate this"
        ));
    }

    #[tokio::test]
    async fn send_input_reuses_existing_child_session() {
        let provider_requests = Arc::new(Mutex::new(Vec::<ProviderRequest>::new()));
        let services = test_services(Arc::new(RecordingProvider::new(provider_requests.clone())));
        let control = RuntimeSubagentControl::new(services);
        let parent = parent_context();

        let spawned = control
            .spawn(
                &parent,
                SpawnSubagentRequest {
                    message: "first task".to_owned(),
                    agent_type: None,
                    fork_context: false,
                    model: None,
                },
            )
            .await
            .expect("spawn");
        control
            .wait(WaitSubagentRequest {
                targets: vec![spawned.agent_id.clone()],
                timeout_ms: Some(5_000),
            })
            .await
            .expect("wait");

        let restarted = control
            .send_input(SendSubagentInputRequest {
                target: spawned.agent_id.clone(),
                message: "follow up".to_owned(),
            })
            .await
            .expect("follow up");
        assert_eq!(restarted.session_id, spawned.session_id);

        let waited = control
            .wait(WaitSubagentRequest {
                targets: vec![spawned.agent_id.clone()],
                timeout_ms: Some(5_000),
            })
            .await
            .expect("wait");
        assert_eq!(
            waited.status.expect("status").last_message.as_deref(),
            Some("child reply [subagent/model]")
        );
        assert_eq!(provider_requests.lock().expect("requests").len(), 2);
    }

    #[tokio::test]
    async fn wait_timeout_returns_target_statuses() {
        let services = test_services(Arc::new(PendingProvider));
        let control = RuntimeSubagentControl::new(services);
        let parent = parent_context();

        let first = control
            .spawn(
                &parent,
                SpawnSubagentRequest {
                    message: "first task".to_owned(),
                    agent_type: None,
                    fork_context: false,
                    model: None,
                },
            )
            .await
            .expect("spawn first");
        let second = control
            .spawn(
                &parent,
                SpawnSubagentRequest {
                    message: "second task".to_owned(),
                    agent_type: None,
                    fork_context: false,
                    model: None,
                },
            )
            .await
            .expect("spawn second");

        let waited = control
            .wait(WaitSubagentRequest {
                targets: vec![first.agent_id.clone(), second.agent_id.clone()],
                timeout_ms: Some(5),
            })
            .await
            .expect("wait timeout");

        assert!(waited.timed_out);
        assert!(waited.status.is_none());
        assert_eq!(waited.target_statuses.len(), 2);
        assert!(
            waited
                .target_statuses
                .iter()
                .all(|status| status.state == SubagentState::Running),
            "target statuses should snapshot running agents: {:?}",
            waited.target_statuses
        );

        control
            .close(CloseSubagentRequest {
                target: first.agent_id,
            })
            .await
            .expect("close first");
        control
            .close(CloseSubagentRequest {
                target: second.agent_id,
            })
            .await
            .expect("close second");
    }

    #[tokio::test]
    async fn wait_unknown_target_errors_before_timeout() {
        let services = test_services(Arc::new(PendingProvider));
        let control = RuntimeSubagentControl::new(services);

        let error = control
            .wait(WaitSubagentRequest {
                targets: vec![AgentId::from("missing-agent")],
                timeout_ms: Some(5),
            })
            .await
            .expect_err("unknown target should error");

        assert!(
            error.to_string().contains("unknown agent 'missing-agent'"),
            "unexpected error: {error:#}"
        );
    }

    #[tokio::test]
    async fn send_input_running_agent_error_suggests_control_flow() {
        let services = test_services(Arc::new(PendingProvider));
        let control = RuntimeSubagentControl::new(services);
        let parent = parent_context();
        let spawned = control
            .spawn(
                &parent,
                SpawnSubagentRequest {
                    message: "running task".to_owned(),
                    agent_type: None,
                    fork_context: false,
                    model: None,
                },
            )
            .await
            .expect("spawn");

        let error = control
            .send_input(SendSubagentInputRequest {
                target: spawned.agent_id.clone(),
                message: "follow up too soon".to_owned(),
            })
            .await
            .expect_err("running send_input should fail");
        let message = error.to_string();
        assert!(
            message.contains("wait_agent"),
            "unexpected error: {message}"
        );
        assert!(
            message.contains("close_agent"),
            "unexpected error: {message}"
        );

        control
            .close(CloseSubagentRequest {
                target: spawned.agent_id,
            })
            .await
            .expect("close");
    }

    #[tokio::test]
    async fn close_running_agent_records_cancellation_status() {
        let services = test_services(Arc::new(PendingProvider));
        let control = RuntimeSubagentControl::new(services);
        let parent = parent_context();
        let spawned = control
            .spawn(
                &parent,
                SpawnSubagentRequest {
                    message: "running task".to_owned(),
                    agent_type: None,
                    fork_context: false,
                    model: None,
                },
            )
            .await
            .expect("spawn");

        let closed = control
            .close(CloseSubagentRequest {
                target: spawned.agent_id.clone(),
            })
            .await
            .expect("close");
        assert_eq!(closed.previous_status.state, SubagentState::Running);

        let waited = control
            .wait(WaitSubagentRequest {
                targets: vec![spawned.agent_id],
                timeout_ms: None,
            })
            .await
            .expect("wait");
        let status = waited.status.expect("closed status");
        assert_eq!(status.state, SubagentState::Closed);
        assert_eq!(
            status.error.as_deref(),
            Some("closed by close_agent (work was cancelled)")
        );
    }

    #[tokio::test]
    async fn spawn_respects_depth_policy() {
        let services = test_services(Arc::new(RecordingProvider::new(Arc::new(Mutex::new(
            Vec::new(),
        )))));
        let control = RuntimeSubagentControl::new(services);
        let mut parent = parent_context();
        parent.blueprint.subagent_depth = 3;

        let error = control
            .spawn(
                &parent,
                SpawnSubagentRequest {
                    message: "delegate this".to_owned(),
                    agent_type: None,
                    fork_context: true,
                    model: None,
                },
            )
            .await
            .expect_err("depth should fail");

        let message = error.to_string();
        assert!(
            message.contains("subagent limit reached: depth"),
            "expected typed SubagentLimit error, got: {message}"
        );
    }

    fn test_services(provider: Arc<dyn Provider>) -> Arc<RuntimeServices> {
        test_services_with_hooks(provider, halter_hooks::RegisteredHooks::default())
    }

    fn test_services_with_hooks(
        provider: Arc<dyn Provider>,
        registered_hooks: halter_hooks::RegisteredHooks,
    ) -> Arc<RuntimeServices> {
        let mut models = ModelRegistry::new();
        models.set_default_model(ResolvedModel {
            role: ModelRole::default(),
            id: ModelId::from("default"),
            provider: ProviderName::from("fake"),
            provider_kind: ProviderKind::Fake,
            api_kind: ApiKind::Fake,
            model: "default/model".to_owned(),
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
            model: "subagent/model".to_owned(),
            max_input_tokens: Some(32_000),
            max_output_tokens: Some(4_096),
            reasoning: None,
            tokens_per_minute: None,
        });
        models.register_provider(ProviderName::from("fake"), provider);

        Arc::new(RuntimeServices {
            resources: Arc::new(ResourceHandle::new(
                halter_protocol::ResourceSnapshot::empty(),
                Arc::new(halter_hooks::Hooks::default()),
                Vec::new(),
            )),
            registered_hooks: Arc::new(registered_hooks),
            session_hook_store: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            models: Arc::new(models),
            tools: Arc::new(ToolRuntime::new()),
            path_locks: Arc::new(PathLockMap::default()),
            tool_sessions: Arc::new(ToolSessionStore::default()),
            sessions: Arc::new(InMemorySessionStore::default()),
            policy: Arc::new(DefaultToolPolicy::new(PolicySettings::default())),
            prompt_assembler: Arc::new(DefaultPromptAssembler),
            context_manager: Arc::new(DefaultContextManager::default()),
            event_bus: Arc::new(EventBus::default()),
            parent_streams: Arc::new(crate::ParentStreamRegistry::default()),
            turn_registry: Arc::new(crate::TurnRegistry::new()),
            subagent_event_forwarding: halter_protocol::SubagentEventForwarding::Off,
            subagent_event_forwarding_cap: 100_000,
            shell_timeout_secs: 30,
            trace_recorder: None,
        })
    }

    /// Persist the parent session so `run_subagent_stop_hooks` can load it
    /// and dispatch `SubagentStop` hooks against real parent state.
    async fn store_parent_session(services: &Arc<RuntimeServices>, parent: &SubagentParentContext) {
        services
            .sessions
            .create_session(halter_session::StoredSession {
                blueprint: parent.blueprint.clone(),
                state: parent.state.clone(),
                snapshot: parent.snapshot.clone(),
            })
            .await
            .expect("store parent session");
    }

    /// Regression (M4): a `SubagentStop` hook that always blocks must not
    /// resubmit turns forever — the resubmission cap fails the subagent with
    /// a descriptive error after a bounded number of full provider turns.
    #[tokio::test]
    async fn always_blocking_stop_hook_trips_resubmission_cap() {
        let provider_requests = Arc::new(Mutex::new(Vec::<ProviderRequest>::new()));
        let mut registered = halter_hooks::RegisteredHooks::default();
        registered.register(
            halter_protocol::PluginId::from("internal"),
            halter_hooks::RegisteredHookPriority::AfterPlugins,
            halter_hooks::Hook::callback(
                halter_hooks::HookEventName::SubagentStop,
                |_input| async move { halter_hooks::HookResponse::block("do it again") },
            ),
        );
        let services = test_services_with_hooks(
            Arc::new(RecordingProvider::new(provider_requests.clone())),
            registered,
        );
        let control = RuntimeSubagentControl::new(services.clone());
        let parent = parent_context();
        store_parent_session(&services, &parent).await;
        // Keep a parent handle alive for the whole run, as a real parent
        // session would while its subagents are managed.
        let _parent_handle =
            HalterSession::new(services.clone(), parent.blueprint.session_id.clone())
                .expect("parent handle");

        let spawned = control
            .spawn(
                &parent,
                SpawnSubagentRequest {
                    message: "delegate this".to_owned(),
                    agent_type: None,
                    fork_context: false,
                    model: None,
                },
            )
            .await
            .expect("spawn");
        let waited = control
            .wait(WaitSubagentRequest {
                targets: vec![spawned.agent_id],
                timeout_ms: Some(30_000),
            })
            .await
            .expect("wait");

        let status = waited.status.expect("terminal status");
        assert_eq!(status.state, SubagentState::Failed);
        let error = status.error.expect("cap error");
        assert!(
            error.contains("resubmission cap of 8 reached"),
            "unexpected error: {error}"
        );
        // Initial turn + capped resubmissions, then the loop stops.
        assert_eq!(
            provider_requests.lock().expect("requests").len(),
            9,
            "provider turns must stop at the cap"
        );
    }

    /// Regression (H1): `SubagentStop` dispatch builds temporary parent
    /// handles; dropping one between dispatches used to evict the parent's
    /// hook-store entry, so the next dispatch saw freshly instantiated
    /// (stateless) hooks. A stateful hook that blocks only on its first
    /// invocation proves state now survives across dispatches: the subagent
    /// completes after exactly one resubmission instead of looping to the cap.
    #[tokio::test]
    async fn stop_hook_state_persists_across_subagent_dispatches() {
        let provider_requests = Arc::new(Mutex::new(Vec::<ProviderRequest>::new()));
        let mut registered = halter_hooks::RegisteredHooks::default();
        registered.register(
            halter_protocol::PluginId::from("internal"),
            halter_hooks::RegisteredHookPriority::AfterPlugins,
            halter_hooks::Hook::function(halter_hooks::HookEventName::SubagentStop, || {
                let calls = Arc::new(Mutex::new(0usize));
                move |_input| {
                    let calls = calls.clone();
                    async move {
                        let mut calls = calls.lock().expect("calls");
                        *calls += 1;
                        if *calls == 1 {
                            halter_hooks::HookResponse::block("one more pass")
                        } else {
                            halter_hooks::HookResponse::passthrough()
                        }
                    }
                }
            }),
        );
        let services = test_services_with_hooks(
            Arc::new(RecordingProvider::new(provider_requests.clone())),
            registered,
        );
        let control = RuntimeSubagentControl::new(services.clone());
        let parent = parent_context();
        store_parent_session(&services, &parent).await;
        let _parent_handle =
            HalterSession::new(services.clone(), parent.blueprint.session_id.clone())
                .expect("parent handle");

        let spawned = control
            .spawn(
                &parent,
                SpawnSubagentRequest {
                    message: "delegate this".to_owned(),
                    agent_type: None,
                    fork_context: false,
                    model: None,
                },
            )
            .await
            .expect("spawn");
        let waited = control
            .wait(WaitSubagentRequest {
                targets: vec![spawned.agent_id],
                timeout_ms: Some(30_000),
            })
            .await
            .expect("wait");

        let status = waited.status.expect("terminal status");
        assert_eq!(
            status.state,
            SubagentState::Completed,
            "hook state was lost between dispatches: {:?}",
            status.error
        );
        // First turn blocked once, second turn passed through.
        assert_eq!(provider_requests.lock().expect("requests").len(), 2);
    }

    /// Regression (M3): when `close` races `start_turn` in the window between
    /// task spawn and registration, the spawned turn must be handed back as
    /// an orphan for cancel+abort rather than registered or silently
    /// detached.
    #[tokio::test]
    async fn register_running_turn_registers_current_and_orphans_stale_turns() {
        fn registry_with(state: SubagentState, generation: u64) -> SubagentRegistry {
            let mut registry = SubagentRegistry::default();
            registry.entries.insert(
                "agent".to_owned(),
                RegisteredSubagent {
                    status: SubagentStatus {
                        agent_id: AgentId::from("agent"),
                        session_id: SessionId::from("child"),
                        agent_type: None,
                        task: "task".to_owned(),
                        state,
                        last_message: None,
                        usage: None,
                        error: None,
                    },
                    generation,
                    running: None,
                },
            );
            registry
        }

        let cases = [
            // (entry state, entry generation, should register)
            (SubagentState::Running, 1u64, true),
            // close bumped the generation before registration
            (SubagentState::Running, 2, false),
            // close marked the entry closed at the same generation
            (SubagentState::Closed, 1, false),
        ];
        for (state, entry_generation, should_register) in cases {
            let mut registry = registry_with(state, entry_generation);
            let cancel = CancellationToken::new();
            let join_handle = tokio::spawn(std::future::pending::<()>());
            let orphan = register_running_turn(
                &mut registry,
                &AgentId::from("agent"),
                1,
                RunningTurn {
                    cancel: cancel.clone(),
                    join_handle,
                },
            );
            let entry_running = registry
                .entries
                .get("agent")
                .expect("entry")
                .running
                .is_some();
            if should_register {
                assert!(orphan.is_none(), "current turn must register");
                assert!(entry_running, "registered turn must land on the entry");
                let running = registry
                    .entries
                    .get_mut("agent")
                    .expect("entry")
                    .running
                    .take()
                    .expect("running turn");
                running.join_handle.abort();
            } else {
                let orphan = orphan.expect("stale turn must be handed back");
                assert!(!entry_running, "stale turn must not be registered");
                // Caller contract: cancel and abort the orphan.
                orphan.cancel.cancel();
                orphan.join_handle.abort();
                assert!(cancel.is_cancelled());
            }
        }

        // Unknown agent: entry removed while spawning.
        let mut registry = SubagentRegistry::default();
        let join_handle = tokio::spawn(std::future::pending::<()>());
        let orphan = register_running_turn(
            &mut registry,
            &AgentId::from("missing"),
            1,
            RunningTurn {
                cancel: CancellationToken::new(),
                join_handle,
            },
        );
        let orphan = orphan.expect("turn for missing entry must be handed back");
        orphan.join_handle.abort();
        assert!(
            orphan
                .join_handle
                .await
                .expect_err("aborted task")
                .is_cancelled()
        );
    }

    fn parent_context() -> SubagentParentContext {
        SubagentParentContext {
            blueprint: halter_protocol::SessionBlueprint {
                session_id: SessionId::from("parent"),
                parent_session_id: None,
                default_model: ModelId::from("default"),
                subagent_model: ModelId::from("subagent"),
                subagent_event_forwarding: halter_protocol::SubagentEventForwarding::Off,
                snapshot_revision: halter_protocol::Revision::from("revision"),
                working_dir: ".".into(),
                system_prompt_seed: Vec::new(),
                max_turns: None,
                subagent_depth: 0,
            },
            state: halter_protocol::SessionState {
                messages: vec![Message::User(halter_protocol::UserMessage::text(
                    "root context",
                ))],
                ..halter_protocol::SessionState::default()
            },
            snapshot: Arc::new(halter_protocol::ResourceSnapshot::empty()),
            model: ModelId::from("default"),
            subagent_model: ModelId::from("subagent"),
        }
    }

    struct PendingProvider;

    #[async_trait]
    impl Provider for PendingProvider {
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

    struct RecordingProvider {
        requests: Arc<Mutex<Vec<ProviderRequest>>>,
    }

    impl RecordingProvider {
        fn new(requests: Arc<Mutex<Vec<ProviderRequest>>>) -> Self {
            Self { requests }
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
            self.requests
                .lock()
                .expect("requests")
                .push(request.clone());
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
                    delta: format!("child reply [{}]", request.model.model),
                }),
                Ok(StreamEvent::TextEnd {
                    id: block_id.clone(),
                }),
                Ok(StreamEvent::UsageUpdate {
                    usage: Usage {
                        input_tokens: 10,
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
}
