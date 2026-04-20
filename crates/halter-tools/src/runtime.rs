// pattern: Imperative Shell

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use async_trait::async_trait;
use halter_protocol::{
    CloseSubagentRequest, CloseSubagentResponse, ResourceSnapshot, SendSubagentInputRequest,
    SessionBlueprint, SessionId, SessionState, SpawnSubagentRequest, SubagentStatus, ToolResult,
    ToolSpec, WaitSubagentRequest, WaitSubagentResponse,
};
use serde_json::Value;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use crate::{PathLockMap, ToolPolicy, ToolSessionStore};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolRuntimeEvent {
    Started { tool_name: String },
    Completed { tool_name: String },
    ToolOutput { tool_name: String, chunk: String },
}

pub trait ToolEventSink: Send + Sync {
    fn emit(&self, event: ToolRuntimeEvent);
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct NoopToolEventSink;

impl ToolEventSink for NoopToolEventSink {
    fn emit(&self, _event: ToolRuntimeEvent) {}
}

#[derive(Debug, Clone)]
pub struct SubagentParentContext {
    pub blueprint: SessionBlueprint,
    pub state: SessionState,
    pub snapshot: Arc<ResourceSnapshot>,
    pub subagent_model: halter_protocol::ModelId,
}

#[async_trait]
pub trait SubagentControl: Send + Sync {
    async fn spawn(
        &self,
        parent: &SubagentParentContext,
        request: SpawnSubagentRequest,
    ) -> anyhow::Result<SubagentStatus>;
    async fn send_input(&self, request: SendSubagentInputRequest)
    -> anyhow::Result<SubagentStatus>;
    async fn wait(&self, request: WaitSubagentRequest) -> anyhow::Result<WaitSubagentResponse>;
    async fn close(&self, request: CloseSubagentRequest) -> anyhow::Result<CloseSubagentResponse>;
}

#[derive(Debug, Default)]
pub struct NoopSubagentControl;

#[async_trait]
impl SubagentControl for NoopSubagentControl {
    async fn spawn(
        &self,
        _parent: &SubagentParentContext,
        _request: SpawnSubagentRequest,
    ) -> anyhow::Result<SubagentStatus> {
        anyhow::bail!("failed to execute subagent tool: subagent control is unavailable")
    }

    async fn send_input(
        &self,
        _request: SendSubagentInputRequest,
    ) -> anyhow::Result<SubagentStatus> {
        anyhow::bail!("failed to execute subagent tool: subagent control is unavailable")
    }

    async fn wait(&self, _request: WaitSubagentRequest) -> anyhow::Result<WaitSubagentResponse> {
        anyhow::bail!("failed to execute subagent tool: subagent control is unavailable")
    }

    async fn close(&self, _request: CloseSubagentRequest) -> anyhow::Result<CloseSubagentResponse> {
        anyhow::bail!("failed to execute subagent tool: subagent control is unavailable")
    }
}

#[derive(Clone)]
pub struct ToolContext {
    pub session_id: SessionId,
    pub working_dir: PathBuf,
    pub path_locks: Arc<PathLockMap>,
    pub tool_sessions: Arc<ToolSessionStore>,
    pub snapshot: Arc<ResourceSnapshot>,
    pub cancel: CancellationToken,
    pub emit: Arc<dyn ToolEventSink>,
    pub policy: Arc<dyn ToolPolicy>,
    pub shell_timeout_secs: u64,
    pub subagent_parent: Option<Arc<SubagentParentContext>>,
}

#[async_trait]
pub trait Tool: Send + Sync {
    fn spec(&self) -> ToolSpec;
    async fn execute(&self, context: ToolContext, input: Value) -> anyhow::Result<ToolResult>;
}

#[derive(Default)]
pub struct ToolRuntime {
    tools: RwLock<HashMap<String, Arc<dyn Tool>>>,
}

impl ToolRuntime {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&self, tool: Arc<dyn Tool>) {
        let spec = tool.spec();
        debug!(tool_name = %spec.name, "registering tool");
        self.tools
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .insert(spec.name.0, tool);
    }

    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .values()
            .map(|tool| tool.spec())
            .collect()
    }

    /// Look up the declared [`ToolConcurrency`] for a registered tool by name.
    ///
    /// Returns `None` when the tool is not registered, letting the caller pick
    /// a conservative default (e.g. `Exclusive`) rather than guessing.
    #[must_use]
    pub fn concurrency_for(&self, name: &str) -> Option<halter_protocol::ToolConcurrency> {
        self.tools
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(name)
            .map(|tool| tool.spec().concurrency)
    }

    #[must_use]
    pub fn clone_filtered(&self, allowed: &[String]) -> Self {
        let allow_all = allowed.is_empty();
        let allowed = allowed.iter().collect::<std::collections::HashSet<_>>();
        let tools = self
            .tools
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .filter(|(name, _)| allow_all || allowed.contains(name))
            .map(|(name, tool)| (name.clone(), tool.clone()))
            .collect();
        Self {
            tools: RwLock::new(tools),
        }
    }

    pub async fn execute(
        &self,
        name: &str,
        context: ToolContext,
        input: Value,
    ) -> anyhow::Result<ToolResult> {
        let tool = self
            .tools
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(name)
            .cloned()
            .ok_or_else(|| {
                warn!(tool_name = name, "attempted to execute unknown tool");
                anyhow::anyhow!("failed to execute tool: unknown tool '{}'", name)
            })?;

        debug!(session_id = %context.session_id, tool_name = name, "dispatching tool execution");
        tool.execute(context, input).await
    }
}
