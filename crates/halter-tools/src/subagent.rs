// pattern: Imperative Shell

use std::sync::Arc;

use anyhow::Context;
use async_trait::async_trait;
use halter_protocol::{
    CloseSubagentRequest, CloseSubagentResponse, ResourceSnapshot, SendSubagentInputRequest,
    SpawnSubagentRequest, SubagentState, SubagentStatus, ToolCapabilities, ToolConcurrency,
    ToolName, ToolSpec, WaitSubagentRequest, WaitSubagentResponse,
};
use serde_json::{Value, json};

use crate::{
    SubagentControl, Tool, ToolContext, ToolOutput, ToolRuntime, ToolRuntimeEvent, TypedTool,
};

const DEFAULT_MODEL_ALIAS: &str = "default";
const SUBAGENT_MODEL_ALIAS: &str = "subagent";
const AUTO_RESOLVE_MODEL_ALIAS: &str = "auto_resolve";

pub fn register_subagent_tools(
    runtime: &ToolRuntime,
    control: Arc<dyn SubagentControl>,
    enabled: &[String],
    snapshot: &ResourceSnapshot,
    available_model_ids: &[String],
) {
    let register_all = enabled.is_empty();
    let mut available_agent_types = snapshot
        .agents
        .keys()
        .map(|name| name.0.clone())
        .collect::<Vec<_>>();
    available_agent_types.sort();
    for tool in [
        Arc::new(SpawnAgentTool::new(
            control.clone(),
            available_agent_types.clone(),
            available_model_ids.to_vec(),
        )) as Arc<dyn Tool>,
        Arc::new(SendInputTool::new(control.clone())),
        Arc::new(WaitAgentTool::new(control.clone())),
        Arc::new(CloseAgentTool::new(control.clone())),
    ] {
        let tool_name = tool.spec().name.0;
        if register_all || enabled.iter().any(|name| name == &tool_name) {
            runtime.register(tool);
        }
    }
}

#[derive(Clone)]
struct SpawnAgentTool {
    control: Arc<dyn SubagentControl>,
    available_agent_types: Vec<String>,
    available_model_ids: Vec<String>,
}

impl SpawnAgentTool {
    fn new(
        control: Arc<dyn SubagentControl>,
        available_agent_types: Vec<String>,
        available_model_ids: Vec<String>,
    ) -> Self {
        let available_model_ids = include_parent_local_model_aliases(available_model_ids);
        Self {
            control,
            available_agent_types,
            available_model_ids,
        }
    }

    fn input_schema(&self) -> Value {
        let model_schema = if self.available_model_ids.is_empty() {
            json!({
                "type": "string",
                "description": "Optional registered model id for the child session. Omit to use the configured subagent model. Built-in aliases default, subagent, and auto_resolve resolve relative to the parent session."
            })
        } else {
            json!({
                "type": "string",
                "enum": self.available_model_ids,
                "description": "Optional registered model id for the child session. Omit to use the configured subagent model. Built-in aliases default, subagent, and auto_resolve resolve relative to the parent session; use registered model ids, not provider model names."
            })
        };
        let mut properties = serde_json::Map::from_iter([
            ("message".to_owned(), json!({ "type": "string" })),
            (
                "fork_context".to_owned(),
                json!({
                    "type": "boolean",
                    "description": "When true, the child session inherits the parent's full message history, compacted context, and file view cache. Defaults to false: the child starts with a clean context and only the task message you provide. Leave false unless the child genuinely needs the parent's accumulated state."

                }),
            ),
            ("model".to_owned(), model_schema),
        ]);
        if !self.available_agent_types.is_empty() {
            properties.insert(
                "agent_type".to_owned(),
                json!({
                    "type": "string",
                    "enum": self.available_agent_types,
                    "description": "Optional named agent role. Omit to use the default child session."
                }),
            );
        }

        Value::Object(serde_json::Map::from_iter([
            ("type".to_owned(), json!("object")),
            ("properties".to_owned(), Value::Object(properties)),
            ("required".to_owned(), json!(["message"])),
        ]))
    }

    fn normalize_request(
        &self,
        request: SpawnSubagentRequest,
    ) -> anyhow::Result<SpawnSubagentRequest> {
        if let Some(agent_type) = request.agent_type.as_ref() {
            if self.available_agent_types.is_empty() {
                anyhow::bail!(
                    "failed to execute spawn_agent tool: agent_type '{}' requested but no named agent roles are registered; omit agent_type to spawn the default child session",
                    agent_type.0
                );
            } else if !self
                .available_agent_types
                .iter()
                .any(|name| name == &agent_type.0)
            {
                anyhow::bail!(
                    "failed to execute spawn_agent tool: unknown agent_type '{}'; available agent_type values: {}",
                    agent_type.0,
                    self.available_agent_types.join(", ")
                );
            }
        }

        if let Some(model) = request.model.as_ref()
            && !self.available_model_ids.is_empty()
            && !self.available_model_ids.iter().any(|name| name == &model.0)
        {
            anyhow::bail!(
                "failed to execute spawn_agent tool: unknown model '{}'; available model values: {}",
                model.0,
                self.available_model_ids.join(", ")
            );
        }

        Ok(request)
    }
}

#[async_trait]
impl TypedTool for SpawnAgentTool {
    type Output = SubagentStatus;

    fn describe(&self) -> ToolSpec {
        subagent_spec(
            "spawn_agent",
            "Spawn a child session to work on a delegated task. Omit agent_type to use the default child session. Omit model to use the configured subagent model. If you set model, use a registered model id; default, subagent, and auto_resolve resolve relative to the parent session.",
            self.input_schema(),
            ToolConcurrency::Exclusive,
            // long_running=true: spawning a child session can involve
            // provider handshakes and blueprint resolution that take
            // seconds, not milliseconds. Previously flagged false, which
            // caused the runtime to apply short-tool-timeout semantics to
            // spawn_agent. (finding L34)
            subagent_capabilities(true),
        )
    }

    async fn run(&self, context: ToolContext, input: Value) -> anyhow::Result<SubagentStatus> {
        emit_started(&context, "spawn_agent");
        let parent = context
            .subagent_parent
            .as_ref()
            .context("failed to execute spawn_agent tool: missing parent session context")?;
        let request: SpawnSubagentRequest = serde_json::from_value(input)
            .context("failed to execute spawn_agent tool: invalid input")?;
        let request = self.normalize_request(request)?;
        let status = self
            .control
            .spawn(parent, request, context.cancel.clone())
            .await?;
        emit_completed(&context, "spawn_agent");
        Ok(status)
    }
}

fn include_parent_local_model_aliases(mut available_model_ids: Vec<String>) -> Vec<String> {
    if available_model_ids.is_empty() {
        return available_model_ids;
    }

    for alias in [
        DEFAULT_MODEL_ALIAS,
        SUBAGENT_MODEL_ALIAS,
        AUTO_RESOLVE_MODEL_ALIAS,
    ] {
        if !available_model_ids.iter().any(|model_id| model_id == alias) {
            available_model_ids.push(alias.to_owned());
        }
    }
    available_model_ids.sort();
    available_model_ids
}

#[derive(Clone)]
struct SendInputTool {
    control: Arc<dyn SubagentControl>,
}

impl SendInputTool {
    fn new(control: Arc<dyn SubagentControl>) -> Self {
        Self { control }
    }
}

#[async_trait]
impl TypedTool for SendInputTool {
    type Output = SubagentStatus;

    fn describe(&self) -> ToolSpec {
        subagent_spec(
            "send_input",
            "Send a follow-up task to an existing child session after it reaches a terminal state (completed, failed, or cancelled). Closed children no longer accept input. Use wait_agent to wait for completion, or close_agent to stop a running agent.",
            json!({
                "type": "object",
                "properties": {
                    "target": { "type": "string" },
                    "message": { "type": "string" }
                },
                "required": ["target", "message"],
            }),
            ToolConcurrency::Exclusive,
            subagent_capabilities(false),
        )
    }

    async fn run(&self, context: ToolContext, input: Value) -> anyhow::Result<SubagentStatus> {
        emit_started(&context, "send_input");
        let request: SendSubagentInputRequest = serde_json::from_value(input)
            .context("failed to execute send_input tool: invalid input")?;
        let status = self.control.send_input(request).await?;
        emit_completed(&context, "send_input");
        Ok(status)
    }
}

#[derive(Clone)]
struct WaitAgentTool {
    control: Arc<dyn SubagentControl>,
}

impl WaitAgentTool {
    fn new(control: Arc<dyn SubagentControl>) -> Self {
        Self { control }
    }
}

#[async_trait]
impl TypedTool for WaitAgentTool {
    type Output = WaitSubagentResponse;

    fn describe(&self) -> ToolSpec {
        subagent_spec(
            "wait_agent",
            "Wait for one of the target child sessions to reach a terminal state (completed, failed, cancelled, or closed). Returns a status line for the agent that finished, `agent <id>: <state>`, followed by its final message. If timeout_ms expires before any target finishes, returns `timed out` followed by one status line for each requested agent.",
            json!({
                "type": "object",
                "properties": {
                    "targets": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "timeout_ms": { "type": "integer", "minimum": 0 }
                },
                "required": ["targets"],
            }),
            ToolConcurrency::ReadOnly,
            subagent_capabilities(true),
        )
    }

    async fn run(
        &self,
        context: ToolContext,
        input: Value,
    ) -> anyhow::Result<WaitSubagentResponse> {
        emit_started(&context, "wait_agent");
        let request: WaitSubagentRequest = serde_json::from_value(input)
            .context("failed to execute wait_agent tool: invalid input")?;
        let response = self.control.wait(request, context.cancel.clone()).await?;
        emit_completed(&context, "wait_agent");
        Ok(response)
    }
}

#[derive(Clone)]
struct CloseAgentTool {
    control: Arc<dyn SubagentControl>,
}

impl CloseAgentTool {
    fn new(control: Arc<dyn SubagentControl>) -> Self {
        Self { control }
    }
}

#[async_trait]
impl TypedTool for CloseAgentTool {
    type Output = CloseSubagentResponse;

    fn describe(&self) -> ToolSpec {
        subagent_spec(
            "close_agent",
            "Close an existing child session, stopping any in-progress work. Use this to cancel a running agent or clean up a finished one. Omit timeout_ms to wait for cleanup without a deadline; setting it bounds the wait and requests forced cleanup on timeout. The closed agent will no longer accept send_input calls.",
            json!({
                "type": "object",
                "properties": {
                    "target": { "type": "string" },
                    "timeout_ms": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Optional cancellation and cleanup timeout in milliseconds. Omit to wait without a deadline."
                    }
                },
                "required": ["target"],
            }),
            ToolConcurrency::Exclusive,
            ToolCapabilities {
                long_running: true,
                ..subagent_capabilities(false)
            },
        )
    }

    async fn run(
        &self,
        context: ToolContext,
        input: Value,
    ) -> anyhow::Result<CloseSubagentResponse> {
        emit_started(&context, "close_agent");
        let request: CloseSubagentRequest = serde_json::from_value(input)
            .context("failed to execute close_agent tool: invalid input")?;
        let response = self.control.close(request).await?;
        emit_completed(&context, "close_agent");
        Ok(response)
    }
}

/// One line naming the agent and its state, plus its error if it has one.
fn status_header(status: &SubagentStatus) -> String {
    let state = match status.state {
        SubagentState::Running => "running",
        SubagentState::Completed => "completed",
        SubagentState::Failed => "failed",
        SubagentState::Cancelled => "cancelled",
        SubagentState::Closed => "closed",
    };
    let error = status.error.iter().map(|error| format!("\nerror: {error}"));
    format!(
        "agent {}: {state}{}",
        status.agent_id,
        error.collect::<String>()
    )
}

impl ToolOutput for SubagentStatus {
    /// The status header, then the child's last message as it wrote it.
    fn to_model_string(&self) -> String {
        let message = self.last_message.iter();
        status_header(self)
            + &message
                .map(|text| format!("\n\n{text}"))
                .collect::<String>()
    }
}

impl ToolOutput for WaitSubagentResponse {
    /// The finished agent's status and final message, or on a timeout the
    /// current state of every target.
    fn to_model_string(&self) -> String {
        let Some(status) = &self.status else {
            let targets = self.target_statuses.iter().map(status_header);
            return format!("timed out\n{}", targets.collect::<Vec<_>>().join("\n"));
        };
        status.to_model_string()
    }
}

impl ToolOutput for CloseSubagentResponse {
    fn to_model_string(&self) -> String {
        format!("closed {}", status_header(&self.previous_status))
    }
}

/// Shared ToolSpec builder for the subagent tools. The four tools differ
/// only in name, description, input_schema, concurrency, and long_running,
/// so centralizing the scaffolding here (finding M45) removes four
/// near-identical ToolSpec blocks and keeps their shared flags
/// (mutating=false, approval=false) in one place.
fn subagent_spec(
    name: &'static str,
    description: &'static str,
    input_schema: Value,
    concurrency: ToolConcurrency,
    capabilities: ToolCapabilities,
) -> ToolSpec {
    ToolSpec {
        name: ToolName::from(name),
        description: description.to_owned(),
        input_schema,
        concurrency,
        capabilities,
        provider_aliases: Default::default(),
    }
}

fn subagent_capabilities(long_running: bool) -> ToolCapabilities {
    ToolCapabilities {
        mutating: false,
        requires_approval: false,
        // The long-running calls (spawn, wait) observe the tool's token.
        cancellable: long_running,
        long_running,
    }
}

fn emit_started(context: &ToolContext, tool_name: &str) {
    context.emit.emit(ToolRuntimeEvent::Started {
        tool_name: tool_name.to_owned(),
    });
}

fn emit_completed(context: &ToolContext, tool_name: &str) {
    context.emit.emit(ToolRuntimeEvent::Completed {
        tool_name: tool_name.to_owned(),
    });
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use halter_protocol::{
        AgentId, AgentName, CloseSubagentResponse, ModelId, ResourceSnapshot, SessionBlueprint,
        SessionId, SessionState, SpawnSubagentRequest, SubagentEventForwarding, SubagentState,
        SubagentStatus, WaitSubagentRequest, WaitSubagentResponse,
    };
    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    use crate::{
        DefaultToolPolicy, NoopToolEventSink, PolicySettings, SubagentParentContext, ToolPolicy,
    };

    use super::*;

    #[derive(Default)]
    struct RecordingSubagentControl {
        requests: Mutex<Vec<SpawnSubagentRequest>>,
        close_requests: Mutex<Vec<CloseSubagentRequest>>,
    }

    #[async_trait]
    impl SubagentControl for RecordingSubagentControl {
        async fn spawn(
            &self,
            _parent: &SubagentParentContext,
            request: SpawnSubagentRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<SubagentStatus> {
            self.requests.lock().expect("requests").push(request);
            Ok(SubagentStatus {
                agent_id: AgentId::from("agent-1"),
                session_id: SessionId::from("session-1"),
                agent_type: Some(AgentName::from("helper")),
                task: "delegate this".to_owned(),
                state: SubagentState::Running,
                last_message: None,
                usage: None,
                error: None,
            })
        }

        async fn send_input(
            &self,
            _request: SendSubagentInputRequest,
        ) -> anyhow::Result<SubagentStatus> {
            unreachable!(
                "RecordingSubagentControl::send_input was called, but this test does not exercise send_input"
            )
        }

        async fn wait(
            &self,
            _request: WaitSubagentRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<WaitSubagentResponse> {
            unreachable!(
                "RecordingSubagentControl::wait was called, but this test does not exercise wait_agent"
            )
        }

        async fn close(
            &self,
            request: CloseSubagentRequest,
        ) -> anyhow::Result<CloseSubagentResponse> {
            let previous_status = SubagentStatus {
                agent_id: request.target.clone(),
                session_id: SessionId::from("session-1"),
                agent_type: None,
                task: "delegate this".to_owned(),
                state: SubagentState::Completed,
                last_message: None,
                usage: None,
                error: None,
            };
            self.close_requests
                .lock()
                .expect("close requests")
                .push(request);
            Ok(CloseSubagentResponse { previous_status })
        }
    }

    #[tokio::test]
    async fn close_agent_forwards_optional_cleanup_deadlines() {
        for (input, timeout_ms) in [
            (json!({ "target": "agent-1" }), None),
            (json!({ "target": "agent-1", "timeout_ms": null }), None),
            (json!({ "target": "agent-1", "timeout_ms": 0 }), Some(0)),
            (json!({ "target": "agent-1", "timeout_ms": 25 }), Some(25)),
            (
                json!({ "target": "agent-1", "timeout_ms": u64::MAX }),
                Some(u64::MAX),
            ),
        ] {
            let control = Arc::new(RecordingSubagentControl::default());
            let tool = CloseAgentTool::new(control.clone());
            let context = ToolContext {
                session_id: SessionId::new(),
                working_dir: ".".into(),
                path_locks: Arc::new(crate::PathLockMap::default()),
                tool_sessions: Arc::new(crate::ToolSessionStore::default()),
                snapshot: Arc::new(ResourceSnapshot::empty()),
                cancel: CancellationToken::new(),
                emit: Arc::new(NoopToolEventSink),
                policy: Arc::new(DefaultToolPolicy::new(PolicySettings::default())),
                shell_timeout_secs: 30,
                subagent_parent: None,
            };
            let result = tool.execute(context, input).await.unwrap();
            assert_eq!(result.to_model_string(), "closed agent agent-1: completed");
            let value = result.structured().expect("structured");
            assert_eq!(value["previous_status"]["agent_id"], "agent-1");
            assert_eq!(
                *control.close_requests.lock().unwrap(),
                vec![CloseSubagentRequest {
                    target: AgentId::from("agent-1"),
                    timeout_ms
                }]
            );
            let spec = tool.spec();
            assert_eq!(spec.input_schema["required"], json!(["target"]));
            assert_eq!(
                spec.input_schema["properties"]["timeout_ms"]["type"],
                "integer"
            );
            assert_eq!(spec.input_schema["properties"]["timeout_ms"]["minimum"], 0);
        }
    }

    #[tokio::test]
    async fn spawn_agent_forwards_request() {
        let control = Arc::new(RecordingSubagentControl::default());
        let tool = SpawnAgentTool::new(
            control.clone(),
            vec!["helper".to_owned()],
            vec!["default".to_owned(), "subagent".to_owned()],
        );
        let context = ToolContext {
            session_id: SessionId::new(),
            working_dir: ".".into(),
            path_locks: Arc::new(crate::PathLockMap::default()),
            tool_sessions: Arc::new(crate::ToolSessionStore::default()),
            snapshot: Arc::new(ResourceSnapshot::empty()),
            cancel: CancellationToken::new(),
            emit: Arc::new(NoopToolEventSink),
            policy: Arc::new(DefaultToolPolicy::new(PolicySettings::default()))
                as Arc<dyn ToolPolicy>,
            shell_timeout_secs: 30,
            subagent_parent: Some(Arc::new(SubagentParentContext {
                blueprint: SessionBlueprint {
                    session_id: SessionId::from("parent-session"),
                    parent_session_id: None,
                    default_model: ModelId::from("default"),
                    subagent_model: ModelId::from("subagent"),
                    subagent_event_forwarding: SubagentEventForwarding::Off,
                    snapshot_revision: "revision".into(),
                    working_dir: ".".into(),
                    system_prompt_seed: Vec::new(),
                    max_turns: None,
                    subagent_depth: 0,
                },
                state: SessionState::default(),
                snapshot: Arc::new(ResourceSnapshot::empty()),
                model: ModelId::from("default"),
                subagent_model: ModelId::from("subagent"),
            })),
        };

        let result = tool
            .execute(
                context,
                json!({
                    "message": "delegate this",
                    "agent_type": "helper",
                    "fork_context": false,
                    "model": "default"
                }),
            )
            .await
            .expect("spawn succeeds");

        let value = result.structured().expect("structured");
        assert_eq!(value["agent_id"], "agent-1");
        let requests = control.requests.lock().expect("requests");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].message, "delegate this");
        assert_eq!(requests[0].agent_type, Some(AgentName::from("helper")));
        assert!(!requests[0].fork_context);
        assert_eq!(requests[0].model, Some(ModelId::from("default")));
    }

    #[test]
    fn wait_renders_the_final_message_as_text_under_a_status_header() {
        let status = |state, last_message: Option<&str>, error: Option<&str>| SubagentStatus {
            agent_id: AgentId::from("agent-1"),
            session_id: SessionId::from("session-1"),
            agent_type: None,
            task: "delegate this".to_owned(),
            state,
            last_message: last_message.map(str::to_owned),
            usage: None,
            error: error.map(str::to_owned),
        };
        let finished = status(
            SubagentState::Completed,
            Some("line \"one\"\nline two"),
            None,
        );
        let failed = status(SubagentState::Failed, None, Some("provider gave up"));
        let wait = |status: Option<SubagentStatus>, target_statuses| WaitSubagentResponse {
            timed_out: status.is_none(),
            status,
            target_statuses,
        };
        let cases = [
            (
                wait(Some(finished.clone()), Vec::new()),
                "agent agent-1: completed\n\nline \"one\"\nline two",
            ),
            (
                wait(Some(failed.clone()), Vec::new()),
                "agent agent-1: failed\nerror: provider gave up",
            ),
            (
                wait(
                    None,
                    vec![status(SubagentState::Running, Some("stale"), None), failed],
                ),
                "timed out\nagent agent-1: running\nagent agent-1: failed\nerror: provider gave up",
            ),
        ];

        for (response, text) in cases {
            let result = response.to_tool_result().expect("serializes");
            assert_eq!(result.to_model_string(), text);
            assert_eq!(
                result.structured(),
                Some(&serde_json::to_value(&response).expect("json"))
            );
        }
    }

    #[test]
    fn spawn_agent_model_schema_includes_parent_local_aliases() {
        let tool = SpawnAgentTool::new(
            Arc::new(RecordingSubagentControl::default()),
            Vec::new(),
            vec!["default".to_owned(), "small".to_owned()],
        );

        let spec = tool.spec();
        let model_enum = spec.input_schema["properties"]["model"]["enum"]
            .as_array()
            .expect("model enum");
        let model_values = model_enum
            .iter()
            .map(|value| value.as_str().expect("string enum value"))
            .collect::<Vec<_>>();

        assert!(model_values.contains(&"default"));
        assert!(model_values.contains(&"small"));
        assert!(model_values.contains(&"subagent"));
        assert!(model_values.contains(&"auto_resolve"));
    }

    #[test]
    fn subagent_tool_descriptions_explain_running_agent_control_flow() {
        let control = Arc::new(RecordingSubagentControl::default());

        let send_input = SendInputTool::new(control.clone()).spec().description;
        assert!(send_input.contains("terminal state"));
        assert!(send_input.contains("wait_agent"));
        assert!(send_input.contains("close_agent"));

        let wait_agent = WaitAgentTool::new(control.clone()).spec().description;
        assert!(wait_agent.contains("`agent <id>: <state>`"));
        assert!(wait_agent.contains("`timed out` followed by one status line"));
        assert!(!wait_agent.contains("target_statuses") && !wait_agent.contains("timed_out"));

        let close_agent = CloseAgentTool::new(control).spec().description;
        assert!(close_agent.contains("stopping any in-progress work"));
        assert!(close_agent.contains("cancel"));
        assert!(close_agent.contains("send_input"));
    }

    #[tokio::test]
    async fn spawn_agent_requires_parent_context() {
        let tool = SpawnAgentTool::new(
            Arc::new(RecordingSubagentControl::default()),
            Vec::new(),
            vec!["default".to_owned(), "subagent".to_owned()],
        );
        let context = ToolContext {
            session_id: SessionId::new(),
            working_dir: ".".into(),
            path_locks: Arc::new(crate::PathLockMap::default()),
            tool_sessions: Arc::new(crate::ToolSessionStore::default()),
            snapshot: Arc::new(ResourceSnapshot::empty()),
            cancel: CancellationToken::new(),
            emit: Arc::new(NoopToolEventSink),
            policy: Arc::new(DefaultToolPolicy::new(PolicySettings::default()))
                as Arc<dyn ToolPolicy>,
            shell_timeout_secs: 30,
            subagent_parent: None,
        };

        let error = tool
            .execute(context, json!({ "message": "delegate this" }))
            .await
            .expect_err("missing parent context should fail");

        assert!(error.to_string().contains("missing parent session context"));
    }

    #[tokio::test]
    async fn spawn_agent_rejects_agent_type_when_no_roles_are_loaded() {
        let control = Arc::new(RecordingSubagentControl::default());
        let tool = SpawnAgentTool::new(
            control.clone(),
            Vec::new(),
            vec!["default".to_owned(), "subagent".to_owned()],
        );
        let context = ToolContext {
            session_id: SessionId::new(),
            working_dir: ".".into(),
            path_locks: Arc::new(crate::PathLockMap::default()),
            tool_sessions: Arc::new(crate::ToolSessionStore::default()),
            snapshot: Arc::new(ResourceSnapshot::empty()),
            cancel: CancellationToken::new(),
            emit: Arc::new(NoopToolEventSink),
            policy: Arc::new(DefaultToolPolicy::new(PolicySettings::default()))
                as Arc<dyn ToolPolicy>,
            shell_timeout_secs: 30,
            subagent_parent: Some(Arc::new(SubagentParentContext {
                blueprint: SessionBlueprint {
                    session_id: SessionId::from("parent-session"),
                    parent_session_id: None,
                    default_model: ModelId::from("default"),
                    subagent_model: ModelId::from("subagent"),
                    subagent_event_forwarding: SubagentEventForwarding::Off,
                    snapshot_revision: "revision".into(),
                    working_dir: ".".into(),
                    system_prompt_seed: Vec::new(),
                    max_turns: None,
                    subagent_depth: 0,
                },
                state: SessionState::default(),
                snapshot: Arc::new(ResourceSnapshot::empty()),
                model: ModelId::from("default"),
                subagent_model: ModelId::from("subagent"),
            })),
        };

        let error = tool
            .execute(
                context,
                json!({
                    "message": "delegate this",
                    "agent_type": "general"
                }),
            )
            .await
            .expect_err("agent_type without registered roles should fail");

        assert!(
            error
                .to_string()
                .contains("no named agent roles are registered")
        );
        let requests = control.requests.lock().expect("requests");
        assert!(requests.is_empty());
    }

    #[tokio::test]
    async fn spawn_agent_rejects_unknown_model() {
        let tool = SpawnAgentTool::new(
            Arc::new(RecordingSubagentControl::default()),
            Vec::new(),
            vec!["default".to_owned(), "subagent".to_owned()],
        );
        let context = ToolContext {
            session_id: SessionId::new(),
            working_dir: ".".into(),
            path_locks: Arc::new(crate::PathLockMap::default()),
            tool_sessions: Arc::new(crate::ToolSessionStore::default()),
            snapshot: Arc::new(ResourceSnapshot::empty()),
            cancel: CancellationToken::new(),
            emit: Arc::new(NoopToolEventSink),
            policy: Arc::new(DefaultToolPolicy::new(PolicySettings::default()))
                as Arc<dyn ToolPolicy>,
            shell_timeout_secs: 30,
            subagent_parent: Some(Arc::new(SubagentParentContext {
                blueprint: SessionBlueprint {
                    session_id: SessionId::from("parent-session"),
                    parent_session_id: None,
                    default_model: ModelId::from("default"),
                    subagent_model: ModelId::from("subagent"),
                    subagent_event_forwarding: SubagentEventForwarding::Off,
                    snapshot_revision: "revision".into(),
                    working_dir: ".".into(),
                    system_prompt_seed: Vec::new(),
                    max_turns: None,
                    subagent_depth: 0,
                },
                state: SessionState::default(),
                snapshot: Arc::new(ResourceSnapshot::empty()),
                model: ModelId::from("default"),
                subagent_model: ModelId::from("subagent"),
            })),
        };

        let error = tool
            .execute(
                context,
                json!({
                    "message": "delegate this",
                    "model": "gpt-5"
                }),
            )
            .await
            .expect_err("unknown model should fail");

        assert!(error.to_string().contains("unknown model 'gpt-5'"));
        assert!(
            error
                .to_string()
                .contains("available model values: auto_resolve, default, subagent")
        );
    }

    #[test]
    fn spawn_agent_schema_omits_agent_type_without_loaded_roles() {
        let tool = SpawnAgentTool::new(
            Arc::new(RecordingSubagentControl::default()),
            Vec::new(),
            vec!["default".to_owned(), "subagent".to_owned()],
        );
        let spec = tool.spec();

        assert!(spec.input_schema["properties"].get("agent_type").is_none());
        assert_eq!(
            spec.input_schema["properties"]["model"]["enum"],
            json!(["auto_resolve", "default", "subagent"])
        );
    }

    #[test]
    fn spawn_agent_schema_lists_loaded_roles() {
        let tool = SpawnAgentTool::new(
            Arc::new(RecordingSubagentControl::default()),
            vec!["helper".to_owned(), "reviewer".to_owned()],
            vec![
                "default".to_owned(),
                "small".to_owned(),
                "subagent".to_owned(),
            ],
        );
        let spec = tool.spec();

        assert_eq!(
            spec.input_schema["properties"]["agent_type"]["enum"],
            json!(["helper", "reviewer"])
        );
        assert_eq!(
            spec.input_schema["properties"]["model"]["enum"],
            json!(["auto_resolve", "default", "small", "subagent"])
        );
    }
}
