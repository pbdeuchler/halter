// pattern: Imperative Shell

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::Context;
use async_trait::async_trait;
use futures::TryStreamExt;
use halter_protocol::{
    AgentId, AgentName, CloseSubagentRequest, CloseSubagentResponse, SendSubagentInputRequest,
    SessionId, SpawnSubagentRequest, SubagentState, SubagentStatus, Turn, Usage,
    WaitSubagentRequest, WaitSubagentResponse,
};
use halter_tools::{SubagentControl, SubagentParentContext};
use tokio::sync::{Mutex, Notify};
use tokio::task::JoinHandle;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::session::create_session_seeded;
use crate::subagent_session::{
    build_subagent_session_init, build_subagent_state, extract_subagent_output,
    extract_subagent_usage,
};
use crate::{HalterSession, RuntimeServices};

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

    async fn start_turn(
        &self,
        agent_id: &AgentId,
        session_id: &SessionId,
        agent_type: Option<AgentName>,
        message: String,
    ) -> anyhow::Result<SubagentStatus> {
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
                    "failed to execute subagent request: agent '{}' is still running",
                    agent_id.0
                );
            }
            entry.generation = entry.generation.saturating_add(1);
            entry.status.task = message.clone();
            if let Some(agent_type) = agent_type {
                entry.status.agent_type = Some(agent_type);
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
        let mut join_handle = Some(tokio::spawn(async move {
            let session = HalterSession::new(services, task_session_id.clone());
            controller
                .run_turn_task(
                    task_agent_id,
                    task_session_id,
                    generation,
                    task_message,
                    task_cancel,
                    session,
                )
                .await;
        }));

        let mut registry = self.inner.registry.lock().await;
        let entry = registry.entries.get_mut(&agent_id.0).with_context(|| {
            format!(
                "failed to execute subagent request: unknown agent '{}'",
                agent_id.0
            )
        })?;
        if entry.generation == generation && matches!(entry.status.state, SubagentState::Running) {
            entry.running = Some(RunningTurn {
                cancel,
                join_handle: join_handle.take().expect("join handle set"),
            });
        }
        drop(registry);
        self.signal_change();
        info!(
            agent_id = %status.agent_id,
            session_id = %status.session_id,
            task = %status.task,
            "started subagent turn"
        );
        Ok(status)
    }

    async fn run_turn_task(
        &self,
        agent_id: AgentId,
        _session_id: SessionId,
        generation: u64,
        message: String,
        cancel: CancellationToken,
        session: HalterSession,
    ) {
        let outcome = match session
            .submit_turn_with_cancel(Turn::user(message.clone()), cancel.clone())
            .await
        {
            Ok(events) => match events.try_collect::<Vec<_>>().await {
                Ok(events) => TurnOutcome {
                    state: if cancel.is_cancelled() {
                        SubagentState::Cancelled
                    } else {
                        SubagentState::Completed
                    },
                    last_message: extract_subagent_output(&events),
                    usage: extract_subagent_usage(&events),
                    error: None,
                },
                Err(error) => TurnOutcome {
                    state: if cancel.is_cancelled() {
                        SubagentState::Cancelled
                    } else {
                        SubagentState::Failed
                    },
                    last_message: None,
                    usage: None,
                    error: Some(error.to_string()),
                },
            },
            Err(error) => TurnOutcome {
                state: if cancel.is_cancelled() {
                    SubagentState::Cancelled
                } else {
                    SubagentState::Failed
                },
                last_message: None,
                usage: None,
                error: Some(error.to_string()),
            },
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

        let active_subagents = {
            let registry = self.inner.registry.lock().await;
            registry
                .entries
                .values()
                .filter(|entry| entry.running.is_some())
                .count()
        };
        self.inner
            .services
            .policy
            .check_subagent_spawn(parent.blueprint.subagent_depth, active_subagents)
            .await?;

        let session_id = SessionId::new();
        let agent_id = AgentId::new();
        let init = build_subagent_session_init(parent, &session_id, &request)?;
        let state =
            build_subagent_state(parent, &session_id, &request.message, request.fork_context);
        create_session_seeded(
            self.inner.services.clone(),
            init,
            state,
            parent.snapshot.clone(),
        )
        .await?;

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
        {
            let mut registry = self.inner.registry.lock().await;
            registry.entries.insert(
                agent_id.0.clone(),
                RegisteredSubagent {
                    status: status.clone(),
                    generation: 0,
                    running: None,
                },
            );
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
                    "failed to execute send_input tool: agent '{}' is still running",
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
                    }),
                    Err(_) => Ok(WaitSubagentResponse {
                        status: None,
                        timed_out: true,
                    }),
                }
            }
            None => Ok(WaitSubagentResponse {
                status: Some(wait_for_status.await?),
                timed_out: false,
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
            if let Some(running) = entry.running.take() {
                running.cancel.cancel();
                running.join_handle.abort();
            }
            entry.status.state = SubagentState::Closed;
            entry.status.error = None;
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

        assert!(error.to_string().contains("max_subagent_depth"));
    }

    fn test_services(provider: Arc<dyn Provider>) -> Arc<RuntimeServices> {
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
        });
        models.register_provider(ProviderName::from("fake"), provider);

        Arc::new(RuntimeServices {
            resources: Arc::new(ResourceHandle::new(
                halter_protocol::ResourceSnapshot::empty(),
            )),
            models: Arc::new(models),
            tools: Arc::new(ToolRuntime::new()),
            path_locks: Arc::new(PathLockMap::default()),
            tool_sessions: Arc::new(ToolSessionStore::default()),
            sessions: Arc::new(InMemorySessionStore::default()),
            policy: Arc::new(DefaultToolPolicy::new(PolicySettings::default())),
            prompt_assembler: Arc::new(DefaultPromptAssembler),
            context_manager: Arc::new(DefaultContextManager::default()),
            event_bus: Arc::new(EventBus::default()),
            max_tool_output_bytes: 262_144,
            shell_timeout_secs: 30,
        })
    }

    fn parent_context() -> SubagentParentContext {
        SubagentParentContext {
            blueprint: halter_protocol::SessionBlueprint {
                session_id: SessionId::from("parent"),
                parent_session_id: None,
                default_model: ModelId::from("default"),
                subagent_model: ModelId::from("subagent"),
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
            subagent_model: ModelId::from("subagent"),
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
                }),
            ])
            .boxed())
        }
    }
}
