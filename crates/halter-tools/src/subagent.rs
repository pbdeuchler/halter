// pattern: Imperative Shell

use std::sync::Arc;

use anyhow::Context;
use async_trait::async_trait;
use halter_protocol::{
    CloseSubagentRequest, ResourceSnapshot, SendSubagentInputRequest, SpawnSubagentRequest,
    ToolCapabilities, ToolConcurrency, ToolName, ToolResult, ToolSpec, WaitSubagentRequest,
};
use serde_json::{Value, json};

use crate::{SubagentControl, Tool, ToolContext, ToolRuntime, ToolRuntimeEvent};

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
                "description": "Optional registered model id for the child session. Omit to use the configured subagent model."
            })
        } else {
            json!({
                "type": "string",
                "enum": self.available_model_ids,
                "description": "Optional registered model id for the child session. Omit to use the configured subagent model. Use model ids such as default, small, or subagent, not provider model names."
            })
        };
        let mut properties = serde_json::Map::from_iter([
            ("message".to_owned(), json!({ "type": "string" })),
            (
                "fork_context".to_owned(),
                json!({
                    "type": "boolean",
                    "description": "When true, start from the parent session context. Defaults to true."
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
        mut request: SpawnSubagentRequest,
    ) -> anyhow::Result<SpawnSubagentRequest> {
        if let Some(agent_type) = request.agent_type.as_ref() {
            if self.available_agent_types.is_empty() {
                request.agent_type = None;
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
impl Tool for SpawnAgentTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("spawn_agent"),
            description:
                "Spawn a child session to work on a delegated task. Omit agent_type to use the default child session. If you set model, use a registered model id such as default, small, or subagent, not a provider model name."
                    .to_owned(),
            input_schema: self.input_schema(),
            concurrency: ToolConcurrency::Exclusive,
            capabilities: ToolCapabilities {
                mutating: false,
                requires_approval: false,
                cancellable: false,
                long_running: false,
            },
            provider_aliases: Default::default(),
        }
    }

    async fn execute(&self, context: ToolContext, input: Value) -> anyhow::Result<ToolResult> {
        emit_started(&context, "spawn_agent");
        let parent = context
            .subagent_parent
            .as_ref()
            .context("failed to execute spawn_agent tool: missing parent session context")?;
        let request: SpawnSubagentRequest = serde_json::from_value(input)
            .context("failed to execute spawn_agent tool: invalid input")?;
        let request = self.normalize_request(request)?;
        let status = self.control.spawn(parent, request).await?;
        emit_completed(&context, "spawn_agent");
        Ok(ToolResult::Json {
            value: serde_json::to_value(status)
                .context("failed to execute spawn_agent tool: invalid status payload")?,
        })
    }
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
impl Tool for SendInputTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("send_input"),
            description: "Send a follow-up task to an existing child session".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "target": { "type": "string" },
                    "message": { "type": "string" }
                },
                "required": ["target", "message"],
            }),
            concurrency: ToolConcurrency::Exclusive,
            capabilities: ToolCapabilities {
                mutating: false,
                requires_approval: false,
                cancellable: false,
                long_running: false,
            },
            provider_aliases: Default::default(),
        }
    }

    async fn execute(&self, context: ToolContext, input: Value) -> anyhow::Result<ToolResult> {
        emit_started(&context, "send_input");
        let request: SendSubagentInputRequest = serde_json::from_value(input)
            .context("failed to execute send_input tool: invalid input")?;
        let status = self.control.send_input(request).await?;
        emit_completed(&context, "send_input");
        Ok(ToolResult::Json {
            value: serde_json::to_value(status)
                .context("failed to execute send_input tool: invalid status payload")?,
        })
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
impl Tool for WaitAgentTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("wait_agent"),
            description: "Wait for one of the target child sessions to reach a terminal state"
                .to_owned(),
            input_schema: json!({
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
            concurrency: ToolConcurrency::ReadOnly,
            capabilities: ToolCapabilities {
                mutating: false,
                requires_approval: false,
                cancellable: false,
                long_running: true,
            },
            provider_aliases: Default::default(),
        }
    }

    async fn execute(&self, context: ToolContext, input: Value) -> anyhow::Result<ToolResult> {
        emit_started(&context, "wait_agent");
        let request: WaitSubagentRequest = serde_json::from_value(input)
            .context("failed to execute wait_agent tool: invalid input")?;
        let response = self.control.wait(request).await?;
        emit_completed(&context, "wait_agent");
        Ok(ToolResult::Json {
            value: serde_json::to_value(response)
                .context("failed to execute wait_agent tool: invalid response payload")?,
        })
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
impl Tool for CloseAgentTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("close_agent"),
            description: "Close an existing child session and stop accepting follow-up input"
                .to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "target": { "type": "string" }
                },
                "required": ["target"],
            }),
            concurrency: ToolConcurrency::Exclusive,
            capabilities: ToolCapabilities {
                mutating: false,
                requires_approval: false,
                cancellable: false,
                long_running: false,
            },
            provider_aliases: Default::default(),
        }
    }

    async fn execute(&self, context: ToolContext, input: Value) -> anyhow::Result<ToolResult> {
        emit_started(&context, "close_agent");
        let request: CloseSubagentRequest = serde_json::from_value(input)
            .context("failed to execute close_agent tool: invalid input")?;
        let response = self.control.close(request).await?;
        emit_completed(&context, "close_agent");
        Ok(ToolResult::Json {
            value: serde_json::to_value(response)
                .context("failed to execute close_agent tool: invalid response payload")?,
        })
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
        SessionId, SessionState, SpawnSubagentRequest, SubagentState, SubagentStatus,
        WaitSubagentRequest, WaitSubagentResponse,
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
    }

    #[async_trait]
    impl SubagentControl for RecordingSubagentControl {
        async fn spawn(
            &self,
            _parent: &SubagentParentContext,
            request: SpawnSubagentRequest,
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
            unreachable!("send_input not used in this test")
        }

        async fn wait(
            &self,
            _request: WaitSubagentRequest,
        ) -> anyhow::Result<WaitSubagentResponse> {
            unreachable!("wait not used in this test")
        }

        async fn close(
            &self,
            _request: CloseSubagentRequest,
        ) -> anyhow::Result<CloseSubagentResponse> {
            unreachable!("close not used in this test")
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
                    snapshot_revision: "revision".into(),
                    working_dir: ".".into(),
                    system_prompt_seed: Vec::new(),
                    max_turns: None,
                    subagent_depth: 0,
                },
                state: SessionState::default(),
                snapshot: Arc::new(ResourceSnapshot::empty()),
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

        let ToolResult::Json { value } = result else {
            panic!("expected json result");
        };
        assert_eq!(value["agent_id"], "agent-1");
        let requests = control.requests.lock().expect("requests");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].message, "delegate this");
        assert_eq!(requests[0].agent_type, Some(AgentName::from("helper")));
        assert!(!requests[0].fork_context);
        assert_eq!(requests[0].model, Some(ModelId::from("default")));
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
    async fn spawn_agent_ignores_agent_type_when_no_roles_are_loaded() {
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
                    snapshot_revision: "revision".into(),
                    working_dir: ".".into(),
                    system_prompt_seed: Vec::new(),
                    max_turns: None,
                    subagent_depth: 0,
                },
                state: SessionState::default(),
                snapshot: Arc::new(ResourceSnapshot::empty()),
                subagent_model: ModelId::from("subagent"),
            })),
        };

        tool.execute(
            context,
            json!({
                "message": "delegate this",
                "agent_type": "general"
            }),
        )
        .await
        .expect("spawn succeeds");

        let requests = control.requests.lock().expect("requests");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].agent_type, None);
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
                    snapshot_revision: "revision".into(),
                    working_dir: ".".into(),
                    system_prompt_seed: Vec::new(),
                    max_turns: None,
                    subagent_depth: 0,
                },
                state: SessionState::default(),
                snapshot: Arc::new(ResourceSnapshot::empty()),
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
                .contains("available model values: default, subagent")
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
            json!(["default", "subagent"])
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
            json!(["default", "small", "subagent"])
        );
    }
}
