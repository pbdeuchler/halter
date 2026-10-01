use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::{self, BoxStream};
use halter_protocol::{
    ApiKind, AssembledPrompt, BlockId, Message, MessageId, ModelId, ModelRole, PanelIsolation,
    ProviderCapabilities, ProviderError, ProviderKind, ProviderName, ProviderRequest,
    ResolvedModel, ResourceSnapshot, StopReason, StreamEvent, ToolCallId, ToolCapabilities,
    ToolConcurrency, ToolName, ToolResult, ToolSpec, TurnId,
};
use halter_providers::{
    FakeProvider, FullTurnJudgePlan, FullTurnPanelist, ModelJudgeMember, ModelRegistry, Provider,
};
use halter_tools::{
    BackgroundTool, DefaultToolPolicy, NoopToolEventSink, PolicySettings, Tool, ToolContext,
};
use serde_json::{Value, json};
use tokio::sync::{Notify, mpsc};
use tokio_util::sync::CancellationToken;

use super::{FullTurnInputs, run_full_turn_deliberation, run_panel_turn, tests::test_blueprint};
use crate::RuntimeServices;

const TEST_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone, Copy, Debug)]
enum PanelOutcome {
    Success,
    Failure,
    PendingProvider,
    PendingTool,
}

struct CapturedPanel {
    context: ToolContext,
    pid: Option<u32>,
}

struct CaptureTool {
    captures: mpsc::UnboundedSender<CapturedPanel>,
    background: bool,
    outcome: PanelOutcome,
}

#[async_trait]
impl Tool for CaptureTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("capture_panel"),
            description: "Capture a panelist's tool context".into(),
            input_schema: json!({"type": "object"}),
            concurrency: ToolConcurrency::Exclusive,
            capabilities: ToolCapabilities::default(),
            provider_aliases: Default::default(),
        }
    }

    async fn execute(&self, mut context: ToolContext, _input: Value) -> anyhow::Result<ToolResult> {
        let pid = if self.background {
            let ToolResult::Json { value } = BackgroundTool
                .execute(
                    context.clone(),
                    json!({"action": "spawn", "command": "sleep 30"}),
                )
                .await?
            else {
                panic!("background spawn returns process metadata");
            };
            let pid = value["pid"].as_u64().expect("job pid") as u32;
            #[cfg(unix)]
            assert!(
                process_alive(pid).await,
                "captured background process is alive"
            );
            Some(pid)
        } else {
            None
        };
        // Retain process ownership for the assertions, without retaining the
        // execution's live event sender and preventing its stream from ending.
        context.emit = Arc::new(NoopToolEventSink);
        self.captures
            .send(CapturedPanel { context, pid })
            .expect("capture receiver");
        if matches!(self.outcome, PanelOutcome::PendingTool) {
            // Custom tools need not cooperate with their cancellation token.
            return std::future::pending().await;
        }
        Ok(ToolResult::Json {
            value: json!({"captured": true}),
        })
    }
}

struct PanelProvider {
    outcome: PanelOutcome,
    pending: Arc<Notify>,
}

#[async_trait]
impl Provider for PanelProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::default()
    }

    async fn stream(
        &self,
        request: ProviderRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
        if request
            .messages
            .iter()
            .any(|message| matches!(message, Message::Tool(_)))
        {
            return match self.outcome {
                PanelOutcome::Failure => {
                    anyhow::bail!("panel provider failed after tool execution")
                }
                PanelOutcome::PendingProvider => {
                    self.pending.notify_one();
                    // Intentionally ignore the token to exercise owned task cancellation.
                    std::future::pending().await
                }
                PanelOutcome::Success | PanelOutcome::PendingTool => {
                    FakeProvider::default().stream(request, cancel).await
                }
            };
        }
        let block_id = BlockId::new();
        let message_id = MessageId::new();
        Ok(stream::iter([
            Ok(StreamEvent::MessageStart {
                id: message_id.clone(),
            }),
            Ok(StreamEvent::ToolCallStart {
                id: block_id.clone(),
                tool_call_id: ToolCallId::new(),
                name: ToolName::from("capture_panel"),
            }),
            Ok(StreamEvent::ToolArgsDelta {
                id: block_id.clone(),
                delta: "{}".to_owned(),
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

fn panel_model() -> ResolvedModel {
    ResolvedModel {
        role: ModelRole::default(),
        id: ModelId::from("panel"),
        provider: ProviderName::from("panel-provider"),
        provider_kind: ProviderKind::Fake,
        api_kind: ApiKind::Fake,
        model: "panel".to_owned(),
        max_input_tokens: Some(32_000),
        max_output_tokens: Some(4_096),
        reasoning: None,
        tokens_per_minute: None,
    }
}

fn panelist() -> FullTurnPanelist {
    FullTurnPanelist {
        model_id: ModelId::from("panel"),
        label: "panel".to_owned(),
    }
}

fn services(
    working_dir: &Path,
    outcome: PanelOutcome,
    background: bool,
) -> (
    Arc<RuntimeServices>,
    mpsc::UnboundedReceiver<CapturedPanel>,
    Arc<Notify>,
) {
    let pending = Arc::new(Notify::new());
    let (captures, receiver) = mpsc::unbounded_channel();
    let mut models = ModelRegistry::new();
    models.set_default_model(panel_model());
    models.register_provider(
        ProviderName::from("panel-provider"),
        Arc::new(PanelProvider {
            outcome,
            pending: pending.clone(),
        }),
    );
    let services = RuntimeServices {
        models: Arc::new(models),
        policy: Arc::new(DefaultToolPolicy::new(PolicySettings {
            allowed_read_roots: vec![working_dir.to_owned()],
            allowed_shell_commands: ["sleep".to_owned()].into_iter().collect(),
            ..PolicySettings::default()
        })),
        ..Default::default()
    };
    services.tools.register(Arc::new(CaptureTool {
        captures,
        background,
        outcome,
    }));
    (Arc::new(services), receiver, pending)
}

fn start_panel(
    services: Arc<RuntimeServices>,
    working_dir: &Path,
    cancel: CancellationToken,
) -> tokio::task::JoinHandle<Option<halter_providers::Candidate>> {
    tokio::spawn(run_panel_turn(
        services,
        Arc::new(ResourceSnapshot::empty()),
        test_blueprint(working_dir.to_owned()),
        panelist(),
        working_dir.to_owned(),
        None,
        Arc::new(Vec::new()),
        Arc::new("investigate".to_owned()),
        cancel,
    ))
}

async fn assert_panel_closed(captured: CapturedPanel) {
    let CapturedPanel { mut context, pid } = captured;
    let store = Arc::downgrade(&context.tool_sessions);
    let process_state = context.tool_sessions.has_process_state(&context.session_id);
    let running_jobs = context.tool_sessions.has_running_jobs(&context.session_id);
    #[cfg(unix)]
    let process_alive = if let Some(pid) = pid {
        process_alive(pid).await
    } else {
        false
    };
    #[cfg(not(unix))]
    let _ = pid;
    context.cancel = CancellationToken::new();
    let spawn = BackgroundTool
        .execute(
            context.clone(),
            json!({
                "action": "spawn", "command": "sleep 30"
            }),
        )
        .await;
    // Also clean up on a regression, before assertions can unwind the test.
    context
        .tool_sessions
        .shutdown_session(&context.session_id)
        .await
        .expect("test cleanup");
    assert!(!process_state, "completed panel retains process resources");
    assert!(!running_jobs, "completed panel retains live jobs");
    #[cfg(unix)]
    assert!(!process_alive, "completed panel leaves its process alive");
    assert!(
        spawn
            .expect_err("completed panel must reject new process work")
            .to_string()
            .contains("session is closed")
    );
    drop(context);
    // Even admission-only panels must release the isolated store, rather than
    // leave their lifetime entry owned by a detached execution or cleanup task.
    tokio::time::timeout(TEST_TIMEOUT, async {
        let mut poll = tokio::time::interval(Duration::from_millis(10));
        while store.upgrade().is_some() {
            poll.tick().await;
        }
    })
    .await
    .expect("completed panel releases its isolated tool store");
}

#[cfg(unix)]
async fn process_alive(pid: u32) -> bool {
    tokio::process::Command::new("sh")
        .args(["-c", "kill -0 \"$1\"", "halter-panel-pid", &pid.to_string()])
        .output()
        .await
        .expect("check owned process existence")
        .status
        .success()
}

#[tokio::test]
async fn repeated_panel_outcomes_close_their_tool_sessions() {
    for (outcome, background) in [
        (PanelOutcome::Success, false),
        (PanelOutcome::Failure, false),
        #[cfg(unix)]
        (PanelOutcome::Success, true),
        #[cfg(unix)]
        (PanelOutcome::Failure, true),
    ] {
        let temp = tempfile::tempdir().expect("tempdir");
        let (services, mut captures, _) = services(temp.path(), outcome, background);
        for _ in 0..3 {
            let candidate = tokio::time::timeout(
                TEST_TIMEOUT,
                start_panel(services.clone(), temp.path(), CancellationToken::new()),
            )
            .await
            .expect("panel finishes")
            .expect("panel task");
            assert_eq!(
                candidate.is_some(),
                matches!(outcome, PanelOutcome::Success),
                "{outcome:?}"
            );
            let captured = tokio::time::timeout(TEST_TIMEOUT, captures.recv())
                .await
                .expect("panel used capture tool")
                .expect("captured panel");
            assert_panel_closed(captured).await;
        }
    }
}

#[tokio::test]
async fn cancelled_panel_closes_jobs_even_when_provider_or_tool_ignores_cancellation() {
    for outcome in [PanelOutcome::PendingProvider, PanelOutcome::PendingTool] {
        let temp = tempfile::tempdir().expect("tempdir");
        let (services, mut captures, pending) = services(temp.path(), outcome, cfg!(unix));
        let cancel = CancellationToken::new();
        let panel = start_panel(services, temp.path(), cancel.clone());
        let captured = tokio::time::timeout(TEST_TIMEOUT, captures.recv())
            .await
            .expect("panel tool starts")
            .expect("captured panel");
        if matches!(outcome, PanelOutcome::PendingProvider) {
            tokio::time::timeout(TEST_TIMEOUT, pending.notified())
                .await
                .expect("provider blocks");
        }
        cancel.cancel();
        let result = tokio::time::timeout(TEST_TIMEOUT, panel).await;
        // Make the negative test failure clean up the real background job too.
        if result.is_err() {
            captured
                .context
                .tool_sessions
                .shutdown_session(&captured.context.session_id)
                .await
                .expect("test cleanup");
        }
        assert!(
            result
                .expect("cancelled panel finishes")
                .expect("panel task")
                .is_none()
        );
        assert_panel_closed(captured).await;
    }
}

#[tokio::test]
async fn dropping_deliberation_closes_detached_panel_work() {
    let temp = tempfile::tempdir().expect("tempdir");
    let (services, mut captures, pending) =
        services(temp.path(), PanelOutcome::PendingProvider, cfg!(unix));
    let inputs = FullTurnInputs {
        services,
        blueprint: test_blueprint(temp.path().to_owned()),
        snapshot: Arc::new(ResourceSnapshot::empty()),
        fork_messages: Vec::new(),
        judge_messages: Vec::new(),
        session_id: "parent".into(),
        turn_id: TurnId::new(),
        user_text: "investigate".to_owned(),
        prompt: AssembledPrompt {
            segments: Vec::new(),
            transcript: Vec::new(),
            ordered_segments: Vec::new(),
            prefix_cache_key: String::new(),
            rendered_prefix: String::new(),
            rendered_transcript: String::new(),
            rendered: String::new(),
            cache_breakpoints: Default::default(),
            system_segment_count: 0,
            skill_segment_count: 0,
        },
    };
    let plan = Arc::new(FullTurnJudgePlan {
        synthesis: ModelJudgeMember {
            provider: Arc::new(FakeProvider::default()),
            model: panel_model(),
        },
        panel: vec![panelist()],
        isolation: PanelIsolation::SharedFull,
    });
    let deliberation = tokio::spawn(run_full_turn_deliberation(
        inputs,
        plan,
        CancellationToken::new(),
    ));
    let captured = tokio::time::timeout(TEST_TIMEOUT, captures.recv())
        .await
        .expect("panel tool starts")
        .expect("captured panel");
    tokio::time::timeout(TEST_TIMEOUT, pending.notified())
        .await
        .expect("provider blocks");
    let store = captured.context.tool_sessions.clone();
    deliberation.abort();
    assert!(
        deliberation
            .await
            .expect_err("deliberation is dropped")
            .is_cancelled()
    );
    let settled = tokio::time::timeout(TEST_TIMEOUT, async {
        let mut poll = tokio::time::interval(Duration::from_millis(10));
        while store.has_process_state(&captured.context.session_id) {
            poll.tick().await;
        }
    })
    .await;
    if settled.is_err() {
        store
            .shutdown_session(&captured.context.session_id)
            .await
            .expect("test cleanup");
    }
    settled.expect("dropped deliberation closes its panel tool sessions");
    drop(store);
    assert_panel_closed(captured).await;
}
