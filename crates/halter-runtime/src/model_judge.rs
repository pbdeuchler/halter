// FullTurn model-judge deliberation. Where the OneShot judge lives at the
// `Provider` seam (one inference per panelist, every step), the FullTurn judge
// lives here at the turn seam: once per user message, each panelist runs a
// *complete agentic turn* (inference + tool loop) in its own forked sub-session,
// and the shared synthesis core (`halter_providers::run_panel_synthesis`) judges
// their outcomes. The resulting guidance is handed back to `run_turn`, which
// injects it into the default model's opening inference.
//
// Panelists are advisory: the default model owns the real, user-visible
// execution. How aggressively a panelist may touch the workspace is governed by
// `PanelIsolation` (read-only tools by default; shared full tools; or a per-panel
// git worktree).

use std::future::Future;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;

use futures::TryStreamExt;
use halter_protocol::{
    AssembledPrompt, Message, PanelIsolation, ProviderRequest, ResourceSnapshot, SessionBlueprint,
    SessionId, SubagentEventForwarding, Turn, TurnId,
};
use halter_providers::{
    Candidate, FullTurnJudgePlan, FullTurnPanelist, MODEL_JUDGE_TRACE_TARGET, run_panel_synthesis,
};
use halter_tools::{ToolRuntime, ToolSessionStore};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::session::create_session_seeded;
use crate::subagent_session::extract_subagent_output;
use crate::{ParentStreamRegistry, ResourceHandle, RuntimeServices, SessionInit, TurnRegistry};

/// Framing prepended (as a system segment) to every FullTurn panelist's session.
/// It tells the panelist it is one advisory voice running a real, tool-using turn
/// whose *outcome* a synthesis judge will weigh — so it should investigate and
/// conclude decisively, not assume its work ships.
const FULL_TURN_PANEL_PREFIX: &str = "You are one of several expert models on a \
    review panel, each running this turn independently with your own tools. Work \
    the task as you see fit — read, search, reason, and use tools to reach a \
    well-founded conclusion. You are advising, not shipping: an independent \
    synthesis judge will read your final outcome alongside the other panelists' \
    and distill guidance for the model that actually commits the work. Treat any \
    changes you make as a scratch exploration, and finish with a clear, \
    self-contained summary of what you found, what you would do next and why, \
    the alternatives you weighed, and the risks or unknowns that would change \
    your mind. Be specific and decisive — hedged guidance is hard to act on.";

/// Everything `run_turn` hands to the deliberation. Owned (not borrowed) so the
/// resulting future captures only `Send` data and can be awaited inside the
/// parent turn's spawned task. `run_turn` builds this once, on the opening
/// inference of a FullTurn-judged turn.
pub(crate) struct FullTurnInputs {
    pub services: Arc<RuntimeServices>,
    pub blueprint: SessionBlueprint,
    pub snapshot: Arc<ResourceSnapshot>,
    /// Parent context as it stood *before* this turn's user message. Each panel
    /// sub-session is seeded with it, then submits the user message as its turn.
    pub fork_messages: Vec<Message>,
    /// The default model's assembled view of this turn, used as the synthesis
    /// judge's context.
    pub judge_messages: Vec<Message>,
    pub prompt: AssembledPrompt,
    pub session_id: SessionId,
    pub turn_id: TurnId,
    /// Plain text of the user's message; the panelists' turn input.
    pub user_text: String,
}

/// Run the FullTurn panel and synthesis, returning advisory guidance for the
/// default model. Returns `None` when no panelist produced a usable outcome, in
/// which case `run_turn` proceeds with the plain default model.
///
/// Returns a boxed `Send` future rather than an `async fn`: deliberation runs
/// panel turns, which run turns, which (via the parent's turn spawn) would make
/// this future's `Send` inference recurse into itself. The explicit boxed `Send`
/// boundary breaks that auto-trait cycle.
pub(crate) fn run_full_turn_deliberation(
    inputs: FullTurnInputs,
    plan: Arc<FullTurnJudgePlan>,
    cancel: CancellationToken,
) -> Pin<Box<dyn Future<Output = Option<String>> + Send>> {
    Box::pin(async move {
        if plan.panel.is_empty() {
            return None;
        }

        let FullTurnInputs {
            services,
            blueprint,
            snapshot,
            fork_messages,
            judge_messages,
            prompt,
            session_id,
            turn_id,
            user_text,
        } = inputs;
        let fork_messages = Arc::new(fork_messages);
        let user_text = Arc::new(user_text);
        let read_only_tools = (plan.isolation == PanelIsolation::ReadOnly)
            .then(|| Arc::new(read_only_tool_names(&services.tools)));
        let synthesis_base = ProviderRequest {
            session_id,
            turn_id: turn_id.clone(),
            model: plan.synthesis.model.clone(),
            prompt,
            compacted_prefix: Vec::new(),
            messages: judge_messages,
            tools: Vec::new(),
            previous_response_id: None,
            new_messages_start: 0,
        };

        let workspaces =
            provision_workspaces(&blueprint, plan.isolation, &turn_id.0, plan.panel.len()).await;

        // Each panel turn runs as its own task so the parent turn's future stays
        // `Send` and the panels genuinely run in parallel.
        let mut handles = Vec::with_capacity(plan.panel.len());
        for (panelist, workspace) in plan.panel.iter().zip(workspaces.iter()) {
            handles.push(tokio::spawn(run_panel_turn(
                services.clone(),
                snapshot.clone(),
                blueprint.clone(),
                panelist.clone(),
                workspace.dir(&blueprint),
                read_only_tools.clone(),
                fork_messages.clone(),
                user_text.clone(),
                cancel.child_token(),
            )));
        }

        let mut candidates: Vec<Candidate> = Vec::new();
        for handle in handles {
            if let Ok(Some(candidate)) = handle.await {
                candidates.push(candidate);
            }
        }

        for workspace in &workspaces {
            workspace.cleanup(&blueprint).await;
        }
        if candidates.is_empty() {
            warn!(
                target: MODEL_JUDGE_TRACE_TARGET,
                "full-turn model-judge produced no panel outcomes; falling back to the default model alone"
            );
            return None;
        }

        match run_panel_synthesis(&plan.synthesis, &synthesis_base, &candidates, &cancel).await {
            Ok(synthesis) => Some(synthesis),
            Err(error) => {
                warn!(
                    target: MODEL_JUDGE_TRACE_TARGET,
                    %error,
                    "full-turn model-judge synthesis failed; falling back to the default model alone"
                );
                None
            }
        }
    })
}

/// Run one panelist as a forked sub-session and collect its final outcome as a
/// [`Candidate`]. Returns `None` if the panel turn failed or produced no output.
#[allow(clippy::too_many_arguments)]
async fn run_panel_turn(
    parent_services: Arc<RuntimeServices>,
    snapshot: Arc<ResourceSnapshot>,
    blueprint: SessionBlueprint,
    panelist: FullTurnPanelist,
    working_dir: PathBuf,
    read_only_tools: Option<Arc<Vec<String>>>,
    fork_messages: Arc<Vec<Message>>,
    user_text: Arc<String>,
    cancel: CancellationToken,
) -> Option<Candidate> {
    let panel_session_id = SessionId::new();
    let services = panel_services(
        &parent_services,
        &snapshot,
        read_only_tools.as_deref().map(Vec::as_slice),
    );
    let init = panel_session_init(&blueprint, &panel_session_id, &panelist, working_dir);
    let state = halter_protocol::SessionState {
        messages: fork_messages.as_ref().clone(),
        ..Default::default()
    };

    let session = match create_session_seeded(services, init, state, snapshot.clone()).await {
        Ok(session) => session,
        Err(error) => {
            warn!(
                target: MODEL_JUDGE_TRACE_TARGET,
                event = "panel_error",
                candidate_id = %panelist.label,
                %error,
                "full-turn model-judge failed to create panel session"
            );
            return None;
        }
    };

    let events = match session
        .submit_turn_with_cancel(Turn::user(user_text.as_str().to_owned()), cancel.clone())
        .await
    {
        Ok(stream) => match stream.try_collect::<Vec<_>>().await {
            Ok(events) => events,
            Err(error) => {
                warn!(
                    target: MODEL_JUDGE_TRACE_TARGET,
                    event = "panel_error",
                    candidate_id = %panelist.label,
                    %error,
                    "full-turn model-judge panel turn failed"
                );
                return None;
            }
        },
        Err(error) => {
            warn!(
                target: MODEL_JUDGE_TRACE_TARGET,
                event = "panel_error",
                candidate_id = %panelist.label,
                %error,
                "full-turn model-judge panel turn could not start"
            );
            return None;
        }
    };

    let own_events: Vec<_> = events
        .into_iter()
        .filter(|event| event.session_id == panel_session_id)
        .collect();
    let body = extract_subagent_output(&own_events)
        .map(|text| text.trim().to_owned())
        .filter(|text| !text.is_empty())?;

    info!(
        target: MODEL_JUDGE_TRACE_TARGET,
        event = "panel_full_turn_response",
        candidate_id = %panelist.label,
        model = %panelist.label,
        response = %body,
        "full-turn model-judge panel outcome"
    );
    Some(Candidate::new(
        panelist.label.clone(),
        panelist.label.clone(),
        body,
    ))
}

/// Build the `SessionInit` for a panel sub-session: the panelist model, the
/// chosen working directory, and the parent's system prompt plus the FullTurn
/// framing. Nested subagents use the plain panelist model so deliberation never
/// recurses into another judge.
fn panel_session_init(
    blueprint: &SessionBlueprint,
    panel_session_id: &SessionId,
    panelist: &FullTurnPanelist,
    working_dir: PathBuf,
) -> SessionInit {
    let mut system_prompt_seed = blueprint.system_prompt_seed.clone();
    system_prompt_seed.push(crate::prompt::system_prompt_segment(FULL_TURN_PANEL_PREFIX));
    SessionInit {
        session_id: Some(panel_session_id.clone()),
        parent_session_id: Some(blueprint.session_id.clone()),
        working_dir,
        system_prompt_seed,
        max_turns: blueprint.max_turns,
        default_model: Some(panelist.model_id.clone()),
        subagent_model: Some(panelist.model_id.clone()),
        subagent_event_forwarding: Some(SubagentEventForwarding::Off),
        subagent_depth: blueprint.subagent_depth + 1,
    }
}

/// Build a per-panel `RuntimeServices` that shares the parent's models, policy,
/// and stores but runs hooks-free with its own tool sessions and turn registry —
/// mirroring the agent-hook spawn path. When `read_only_tools` is `Some`, the
/// tool set is filtered to those names.
fn panel_services(
    parent: &Arc<RuntimeServices>,
    snapshot: &Arc<ResourceSnapshot>,
    read_only_tools: Option<&[String]>,
) -> Arc<RuntimeServices> {
    let tools = match read_only_tools {
        Some(allowed) => Arc::new(parent.tools.clone_filtered(allowed)),
        None => Arc::new(parent.tools.clone_filtered(&[])),
    };
    Arc::new(RuntimeServices {
        resources: Arc::new(ResourceHandle::new(
            snapshot.as_ref().clone(),
            Arc::new(halter_hooks::Hooks::default()),
            Vec::new(),
        )),
        registered_hooks: Arc::new(halter_hooks::RegisteredHooks::default()),
        session_hook_store: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        models: parent.models.clone(),
        tools,
        path_locks: parent.path_locks.clone(),
        tool_sessions: Arc::new(ToolSessionStore::default()),
        sessions: parent.sessions.clone(),
        policy: parent.policy.clone(),
        prompt_assembler: parent.prompt_assembler.clone(),
        context_manager: parent.context_manager.clone(),
        event_bus: parent.event_bus.clone(),
        parent_streams: Arc::new(ParentStreamRegistry::default()),
        turn_registry: Arc::new(TurnRegistry::new()),
        subagent_event_forwarding: SubagentEventForwarding::Off,
        subagent_event_forwarding_cap: parent.subagent_event_forwarding_cap,
        shell_timeout_secs: parent.shell_timeout_secs,
        trace_recorder: parent.trace_recorder.clone(),
    })
}

/// Names of registered tools that do not mutate the workspace (read/glob/grep
/// and friends), used to build the read-only panel tool set.
fn read_only_tool_names(tools: &ToolRuntime) -> Vec<String> {
    tools
        .specs()
        .into_iter()
        .filter(|spec| !spec.capabilities.mutating)
        .map(|spec| spec.name.0)
        .collect()
}

/// Where a single panelist runs. `Shared` reuses the parent working directory;
/// `Worktree` holds a provisioned git worktree path to be removed afterward.
enum PanelWorkspace {
    Shared,
    Worktree(PathBuf),
}

impl PanelWorkspace {
    fn dir(&self, blueprint: &SessionBlueprint) -> PathBuf {
        match self {
            Self::Shared => blueprint.working_dir.clone(),
            Self::Worktree(path) => path.clone(),
        }
    }

    async fn cleanup(&self, blueprint: &SessionBlueprint) {
        if let Self::Worktree(path) = self {
            remove_worktree(&blueprint.working_dir, path).await;
        }
    }
}

/// Decide each panelist's workspace. For `Worktree` isolation in a git repo, a
/// detached worktree is provisioned per panel under `.halter-panels/<turn>`;
/// anything else (or any git failure) falls back to the shared directory.
async fn provision_workspaces(
    blueprint: &SessionBlueprint,
    isolation: PanelIsolation,
    turn_id: &str,
    count: usize,
) -> Vec<PanelWorkspace> {
    if isolation != PanelIsolation::Worktree {
        return (0..count).map(|_| PanelWorkspace::Shared).collect();
    }

    let repo = &blueprint.working_dir;
    if !is_git_repo(repo).await {
        warn!(
            target: MODEL_JUDGE_TRACE_TARGET,
            working_dir = %repo.display(),
            "full-turn model-judge worktree isolation requested but working dir is not a git repo; using shared workspace"
        );
        return (0..count).map(|_| PanelWorkspace::Shared).collect();
    }

    let base = repo.join(".halter-panels").join(turn_id);
    if let Err(error) = tokio::fs::create_dir_all(&base).await {
        warn!(
            target: MODEL_JUDGE_TRACE_TARGET,
            %error,
            "full-turn model-judge could not create worktree base; using shared workspace"
        );
        return (0..count).map(|_| PanelWorkspace::Shared).collect();
    }

    let mut workspaces = Vec::with_capacity(count);
    for index in 0..count {
        let path = base.join(index.to_string());
        match add_worktree(repo, &path).await {
            Ok(()) => workspaces.push(PanelWorkspace::Worktree(path)),
            Err(error) => {
                warn!(
                    target: MODEL_JUDGE_TRACE_TARGET,
                    %error,
                    panel = index,
                    "full-turn model-judge worktree add failed; using shared workspace for this panel"
                );
                workspaces.push(PanelWorkspace::Shared);
            }
        }
    }
    workspaces
}

async fn is_git_repo(dir: &Path) -> bool {
    run_git(dir, &["rev-parse", "--is-inside-work-tree"])
        .await
        .is_ok()
}

async fn add_worktree(repo: &Path, path: &Path) -> anyhow::Result<()> {
    let path = path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("worktree path is not valid UTF-8"))?;
    run_git(repo, &["worktree", "add", "--detach", path, "HEAD"]).await
}

async fn remove_worktree(repo: &Path, path: &Path) {
    let Some(path) = path.to_str() else {
        return;
    };
    let _ = run_git(repo, &["worktree", "remove", "--force", path]).await;
}

async fn run_git(dir: &Path, args: &[&str]) -> anyhow::Result<()> {
    let output = tokio::process::Command::new("git")
        .current_dir(dir)
        .args(args)
        .output()
        .await?;
    if output.status.success() {
        Ok(())
    } else {
        anyhow::bail!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        )
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use halter_protocol::{
        PanelIsolation, Revision, SessionBlueprint, SessionId, SubagentEventForwarding,
    };
    use halter_tools::{ToolRuntime, register_builtin_tools};

    use super::{PanelWorkspace, provision_workspaces, read_only_tool_names};

    fn test_blueprint(working_dir: PathBuf) -> SessionBlueprint {
        SessionBlueprint {
            session_id: SessionId::from("test"),
            parent_session_id: None,
            default_model: "default".into(),
            subagent_model: "subagent".into(),
            subagent_event_forwarding: SubagentEventForwarding::Off,
            snapshot_revision: Revision::from("rev"),
            working_dir,
            system_prompt_seed: Vec::new(),
            max_turns: None,
            subagent_depth: 0,
        }
    }

    #[test]
    fn read_only_tool_names_excludes_mutating_tools() {
        let tools = ToolRuntime::new();
        register_builtin_tools(
            &tools,
            &[
                "read".to_owned(),
                "glob".to_owned(),
                "grep".to_owned(),
                "write".to_owned(),
                "edit".to_owned(),
                "shell".to_owned(),
            ],
        );

        let names = read_only_tool_names(&tools);
        for kept in ["read", "glob", "grep"] {
            assert!(names.iter().any(|name| name == kept), "{kept} is read-only");
        }
        for dropped in ["write", "edit", "shell"] {
            assert!(
                !names.iter().any(|name| name == dropped),
                "{dropped} mutates and must be excluded"
            );
        }
    }

    #[tokio::test]
    async fn provision_workspaces_shared_for_non_worktree_isolation() {
        let blueprint = test_blueprint(PathBuf::from("/tmp"));
        let workspaces =
            provision_workspaces(&blueprint, PanelIsolation::ReadOnly, "turn", 3).await;
        assert_eq!(workspaces.len(), 3);
        assert!(
            workspaces
                .iter()
                .all(|workspace| matches!(workspace, PanelWorkspace::Shared))
        );
    }

    #[tokio::test]
    async fn provision_workspaces_falls_back_to_shared_outside_a_git_repo() {
        let temp = tempfile::tempdir().expect("tempdir");
        let blueprint = test_blueprint(temp.path().to_path_buf());
        let workspaces =
            provision_workspaces(&blueprint, PanelIsolation::Worktree, "turn", 2).await;
        assert_eq!(workspaces.len(), 2);
        assert!(
            workspaces
                .iter()
                .all(|workspace| matches!(workspace, PanelWorkspace::Shared)),
            "a non-git working dir must fall back to the shared workspace"
        );
    }
}
