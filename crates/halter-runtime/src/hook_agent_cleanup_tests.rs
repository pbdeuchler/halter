use super::*;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use futures::{StreamExt, stream, stream::BoxStream};
use halter_protocol::{
    ApiKind, BlockId, ModelRole, ProviderCapabilities, ProviderError, ProviderKind, ProviderName,
    ProviderRequest, ResolvedModel, StopReason, StreamEvent, ToolCallId, ToolCapabilities,
    ToolConcurrency, ToolName, ToolSpec,
};
use halter_providers::{ModelRegistry, Provider};
use halter_tools::{BackgroundTool, DefaultToolPolicy, PolicySettings, Tool, ToolContext};
use tokio::sync::oneshot;

struct CapturedAgent {
    context: ToolContext,
    pid: u32,
}

struct SignalDrop(Option<oneshot::Sender<()>>);

impl Drop for SignalDrop {
    fn drop(&mut self) {
        if let Some(sender) = self.0.take() {
            let _ = sender.send(());
        }
    }
}

struct CaptureAgentTool {
    captured: Mutex<Option<oneshot::Sender<CapturedAgent>>>,
    dropped: Mutex<Option<oneshot::Sender<()>>>,
    block: bool,
}

#[async_trait]
impl Tool for CaptureAgentTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("capture_agent"),
            description: "Capture isolated agent resources".to_owned(),
            input_schema: json!({"type": "object", "properties": {}}),
            concurrency: ToolConcurrency::Exclusive,
            capabilities: ToolCapabilities::default(),
            provider_aliases: Default::default(),
        }
    }

    async fn execute(&self, context: ToolContext, _input: Value) -> anyhow::Result<ToolResult> {
        let _dropped = SignalDrop(self.dropped.lock().unwrap().take());
        let ToolResult::Json { value } = BackgroundTool
            .execute(
                context.clone(),
                json!({"action": "spawn", "command": "sleep 3600"}),
            )
            .await?
        else {
            panic!("background spawn must return JSON");
        };
        let pid = value["pid"].as_u64().expect("background pid") as u32;
        self.captured
            .lock()
            .unwrap()
            .take()
            .unwrap()
            .send(CapturedAgent { context, pid })
            .unwrap_or_else(|_| panic!("capture receiver remains alive"));
        if self.block {
            // Deliberately ignore the tool's token. Cleanup must stop the actual
            // executor future rather than only cancelling the wrapper task.
            std::future::pending::<()>().await;
        }
        Ok(ToolResult::Json {
            value: json!({"ok": true}),
        })
    }
}

struct HookAgentProvider {
    calls: AtomicUsize,
    output: Option<&'static str>,
}

#[async_trait]
impl Provider for HookAgentProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::default()
    }

    async fn stream(
        &self,
        _request: ProviderRequest,
        _cancel: CancellationToken,
    ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
        let message = halter_protocol::MessageId::new();
        let block = BlockId::new();
        if self.calls.fetch_add(1, Ordering::SeqCst) == 0 {
            return Ok(stream::iter(vec![
                Ok(StreamEvent::MessageStart {
                    id: message.clone(),
                }),
                Ok(StreamEvent::ToolCallStart {
                    id: block.clone(),
                    tool_call_id: ToolCallId::new(),
                    name: ToolName::from("capture_agent"),
                }),
                Ok(StreamEvent::ToolArgsDelta {
                    id: block.clone(),
                    delta: "{}".to_owned(),
                }),
                Ok(StreamEvent::ToolCallEnd { id: block }),
                Ok(StreamEvent::MessageEnd {
                    id: message,
                    stop_reason: StopReason::ToolUse,
                    response_id: None,
                }),
            ])
            .boxed());
        }
        let Some(output) = self.output else {
            anyhow::bail!("provider failed after starting background work");
        };
        Ok(stream::iter(vec![
            Ok(StreamEvent::MessageStart {
                id: message.clone(),
            }),
            Ok(StreamEvent::TextStart { id: block.clone() }),
            Ok(StreamEvent::TextDelta {
                id: block.clone(),
                delta: output.to_owned(),
            }),
            Ok(StreamEvent::TextEnd { id: block }),
            Ok(StreamEvent::MessageEnd {
                id: message,
                stop_reason: StopReason::EndTurn,
                response_id: None,
            }),
        ])
        .boxed())
    }
}

fn hook_model(id: &str, role: ModelRole) -> ResolvedModel {
    ResolvedModel {
        role,
        id: ModelId::from(id),
        provider: ProviderName::from("cleanup-test"),
        provider_kind: ProviderKind::Fake,
        api_kind: ApiKind::Fake,
        model: "cleanup-test".to_owned(),
        max_input_tokens: Some(32_000),
        max_output_tokens: Some(4_096),
        reasoning: None,
        tokens_per_minute: None,
    }
}

async fn hook_fixture(
    root: &Path,
    output: Option<&'static str>,
    block: bool,
) -> (
    SessionExecutor,
    AgentHookConfig,
    HookDispatchRequest,
    oneshot::Receiver<CapturedAgent>,
    oneshot::Receiver<()>,
) {
    let (capture_tx, capture_rx) = oneshot::channel();
    let (drop_tx, drop_rx) = oneshot::channel();
    let mut services = RuntimeServices::default();
    let mut models = ModelRegistry::new();
    models.set_default_model(hook_model("default", ModelRole::default()));
    models.set_subagent_model(hook_model("subagent", ModelRole::subagent()));
    models.set_small_model(hook_model("small", ModelRole::small()));
    models.register_provider(
        ProviderName::from("cleanup-test"),
        Arc::new(HookAgentProvider {
            calls: AtomicUsize::new(0),
            output,
        }),
    );
    services.models = Arc::new(models);
    services.policy = Arc::new(DefaultToolPolicy::new(PolicySettings {
        allowed_read_roots: vec![root.to_owned()],
        allowed_shell_commands: vec!["sleep".to_owned()],
        ..PolicySettings::default()
    }));
    services.tools.register(Arc::new(CaptureAgentTool {
        captured: Mutex::new(Some(capture_tx)),
        dropped: Mutex::new(Some(drop_tx)),
        block,
    }));
    let resources = services.resources.snapshot();
    let parent = create_session_seeded(
        Arc::new(services),
        SessionInit {
            working_dir: root.to_owned(),
            ..SessionInit::default()
        },
        SessionState::default(),
        resources,
    )
    .await
    .expect("create parent session");
    let config = AgentHookConfig {
        prompt: "Use capture_agent, then return a hook response as JSON".to_owned(),
        model: None,
        allowed_tools: vec!["capture_agent".to_owned()],
        max_turns: None,
    };
    let request = HookDispatchRequest {
        event_name: HookEventName::UserPromptSubmit,
        matcher_value: None,
        payload: json!({"cwd": root}),
        fired_hook_ids: BTreeSet::new(),
    };
    (parent, config, request, capture_rx, drop_rx)
}

async fn assert_agent_closed(captured: CapturedAgent, dropped: oneshot::Receiver<()>) {
    let tool_dropped = matches!(timeout(Duration::from_secs(5), dropped).await, Ok(Ok(())));
    let store = captured.context.tool_sessions.clone();
    let slots_removed = timeout(Duration::from_secs(5), async {
        let mut poll = tokio::time::interval(Duration::from_millis(10));
        while store.has_process_state(&captured.context.session_id) {
            poll.tick().await;
        }
    })
    .await
    .is_ok();
    let jobs_remaining = store.has_running_jobs(&captured.context.session_id);
    let alive = std::process::Command::new("kill")
        .args(["-0", &captured.pid.to_string()])
        .stderr(Stdio::null())
        .status()
        .expect("probe owned background pid")
        .success();
    let mut stale_context = captured.context;
    stale_context.cancel = CancellationToken::new();
    let spawn = BackgroundTool
        .execute(
            stale_context.clone(),
            json!({"action": "spawn", "command": "sleep 3600"}),
        )
        .await;
    let stale_slots_created = store.has_process_state(&stale_context.session_id);
    // Always reap native processes even if a cleanup regression lets the stale
    // spawn through. Assertions below still inspect the state before this fixup.
    store
        .shutdown_session(&stale_context.session_id)
        .await
        .expect("test fallback cleanup");
    assert!(tool_dropped, "actual tool future was not dropped");
    assert!(slots_removed, "isolated process slots were not removed");
    assert!(!jobs_remaining, "isolated background job is still running");
    assert!(!alive, "owned background process survived hook completion");
    let error = spawn.expect_err("closed admission rejects stale spawn");
    assert!(error.to_string().contains("session is closed"), "{error:#}");
    assert!(!stale_slots_created, "stale getter recreated process slots");
}

#[tokio::test]
async fn hook_agent_closes_resources_on_success_and_output_failures() {
    for (case, output, success) in [
        ("success", Some("{}"), Some(true)),
        ("invalid JSON", Some("not JSON"), Some(false)),
        ("empty assistant output", Some(""), Some(true)),
        ("provider failure after tool use", None, None),
    ] {
        let root = tempfile::tempdir().unwrap();
        let (parent, config, request, captured, dropped) =
            hook_fixture(root.path(), output, false).await;
        let result = run_agent(
            &parent,
            Duration::from_secs(5),
            &config,
            &request,
            CancellationToken::new(),
        )
        .await;
        if let Some(success) = success {
            assert_eq!(result.is_ok(), success, "{case}");
        }
        assert_agent_closed(captured.await.expect("agent used capture tool"), dropped).await;
    }
}

#[tokio::test]
async fn hook_agent_closes_uncancellable_work_on_timeout_cancel_and_parent_drop() {
    for case in ["timeout", "cancel", "drop"] {
        let root = tempfile::tempdir().unwrap();
        let (parent, config, request, captured, dropped) =
            hook_fixture(root.path(), Some("{}"), true).await;
        let cancel = CancellationToken::new();
        let task_cancel = cancel.clone();
        let limit = if case == "timeout" {
            Duration::from_millis(250)
        } else {
            Duration::from_secs(60)
        };
        let task =
            tokio::spawn(
                async move { run_agent(&parent, limit, &config, &request, task_cancel).await },
            );
        let captured = timeout(Duration::from_secs(5), captured)
            .await
            .expect("agent started work")
            .expect("capture tool");
        assert!(
            captured
                .context
                .tool_sessions
                .has_running_jobs(&captured.context.session_id)
        );
        match case {
            "cancel" => cancel.cancel(),
            "drop" => task.abort(),
            _ => {}
        }
        let result = timeout(Duration::from_secs(5), task)
            .await
            .expect("hook wrapper settles");
        match case {
            "cancel" => assert!(matches!(
                result.unwrap().unwrap(),
                HandlerExecution::Cancelled
            )),
            "timeout" => assert!(
                result
                    .unwrap()
                    .expect_err("timeout error")
                    .to_string()
                    .contains("timed out")
            ),
            "drop" => assert!(result.expect_err("aborted parent").is_cancelled()),
            _ => unreachable!(),
        }
        assert_agent_closed(captured, dropped).await;
    }
}
