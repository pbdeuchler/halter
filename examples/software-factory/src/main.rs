// pattern: Imperative Shell

mod core;

use std::collections::{HashMap, HashSet};
use std::env;
use std::error::Error as StdError;
use std::fmt;
use std::io::ErrorKind;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, bail};
use async_trait::async_trait;
use chrono::Utc;
use clap::Parser;
use futures::StreamExt;
use halter::prelude::*;
use halter_config::{
    ConfiguredProvider, ContextConfig, HarnessConfig, ModelConfig, ModelJudgeConfig,
    ModelJudgeMode, ModelSlot, ModelSlotRef, ModelsConfig, NetworkPolicyConfig, PanelIsolation,
    PolicyConfig, ProviderConfig, ProvidersConfig, ResourcesConfig, RuntimeConfig, SearchRoots,
    SessionsConfig, ShellPolicyConfig, ToolsConfig,
};
use halter_protocol::{
    AssistantPart, CacheScope, Message, PromptSegment, PromptSegmentId, PromptSegmentKind,
    PruneSignalThreshold, ReasoningEffort, SessionEventPayload, ToolCapabilities, ToolConcurrency,
    ToolName, ToolResult, ToolSpec, Turn, Usage, Volatility,
};
use halter_tools::{Tool, ToolContext};
use reqwest::header::{ACCEPT, AUTHORIZATION, HeaderMap, HeaderName, HeaderValue, USER_AGENT};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::{process::Command, sync::RwLock};
use tracing::{debug, info, warn};
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

use crate::core::{
    CHECKPOINT_PATH, CandidateSet, CodeReview, FACTORY_WORKTREE_TMP_DIR, IMPLEMENTATION_PLAN_PATH,
    IssueComment, IssueDoc, JudgeSelection, ModelSpec, MonitorAction, PLANNING_CORPUS_BODY_CHARS,
    PROJECT_GUIDANCE_FILENAMES, PROJECT_GUIDANCE_MAX_BYTES, ProjectGuidanceDoc, PullRequestDraft,
    RECENT_OPEN_ISSUE_LIMIT, RepoSlug, branch_name, candidate_set_for_issue,
    dirty_status_excluding, ensure_requested_issue_selection, factory_worktree_dir_name,
    format_project_system_prompt, is_maintainer_author_association, issue_corpus, monitor_action,
    parse_github_remote_url, parse_issue_number_input, parse_json_response, selected_issue_numbers,
    validate_issue_number, validate_recent_issue_limit,
};

#[derive(Debug, Parser)]
#[command(name = "software-factory")]
#[command(about = "Example Halter workflow that turns GitHub issues into an implementation PR")]
struct Cli {
    #[arg(
        long,
        default_value = "origin",
        help = "Git remote whose GitHub URL identifies the repository"
    )]
    remote: String,
    #[arg(long, help = "Base branch; defaults to the repository default branch")]
    base: Option<String>,
    #[arg(
        long,
        help = "Branch name to create; defaults to a generated factory branch"
    )]
    branch: Option<String>,
    #[arg(
        long,
        help = "Create and run inside a dedicated git worktree under /tmp"
    )]
    worktree: bool,
    #[arg(
        long,
        help = "Poll the opened PR for reviews and /plsfix comments until it merges"
    )]
    monitor: bool,
    #[arg(long, help = "Allow starting from a dirty worktree")]
    allow_dirty: bool,
    #[arg(
        long,
        help = "Include the generated implementation plan file in commits"
    )]
    commit_impl_plan: bool,
    #[arg(
        long,
        conflicts_with = "reset_checkpoint",
        help = "Resume from the factory checkpoint file for this worktree"
    )]
    resume: bool,
    #[arg(
        long,
        help = "Delete any existing factory checkpoint before starting a fresh run"
    )]
    reset_checkpoint: bool,
    #[arg(long, help = "Work on one specific open GitHub issue number")]
    issue: Option<u64>,
    #[arg(
        long,
        default_value_t = 5,
        help = "Maximum Kimi/GPT review-repair iterations"
    )]
    max_review_iterations: usize,
    #[arg(long, default_value_t = 60, help = "Seconds between PR monitor polls")]
    poll_seconds: u64,
    #[arg(
        long,
        default_value = "openrouter/z-ai/glm-5.2",
        help = "Provider/model for issue grouping and /plsfix refinement"
    )]
    glm_model: String,
    #[arg(
        long,
        default_value = "openrouter/moonshotai/kimi-k2.7-code",
        help = "Provider/model for implementation"
    )]
    implementer_model: String,
    #[arg(
        long,
        default_value = "openai/gpt-5.5",
        help = "Provider/model for branch-diff code review"
    )]
    reviewer_model: String,
    #[arg(
        long,
        default_value = "openrouter/google/gemma-4-31b-it",
        help = "Provider/model for PR title and body drafting"
    )]
    pr_model: String,
}

/// Third-party targets that become too noisy when users set `RUST_LOG=debug`.
const NOISY_TARGET_SUPPRESSIONS: &str = "tokenize=warn,parse=warn,expansion=warn,commands=warn,\
     pattern=warn,completion=warn,jobs=warn,unimplemented=warn,\
     hyper_util=warn,hyper=warn,reqwest=warn,h2=warn,rustls=warn";
const FACTORY_LOCAL_STATE_PATHS: [&str; 2] = [IMPLEMENTATION_PLAN_PATH, CHECKPOINT_PATH];

fn logging_filter_spec(user_directives: Option<&str>) -> String {
    let directives = user_directives
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("info");
    format!("{directives},{NOISY_TARGET_SUPPRESSIONS}")
}

fn logging_filter_from_spec(spec: &str) -> anyhow::Result<EnvFilter> {
    EnvFilter::try_new(spec).context("invalid RUST_LOG filter")
}

fn init_logging() -> anyhow::Result<()> {
    let user_directives = match env::var(EnvFilter::DEFAULT_ENV) {
        Ok(value) => Some(value),
        Err(env::VarError::NotPresent) => None,
        Err(env::VarError::NotUnicode(_)) => bail!("invalid utf-8 in RUST_LOG"),
    };
    let filter = logging_filter_from_spec(&logging_filter_spec(user_directives.as_deref()))?;
    tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer().with_target(true).compact())
        .try_init()
        .context("failed to initialize logging")
}

fn excluded_commit_paths(commit_impl_plan: bool) -> Vec<&'static str> {
    if commit_impl_plan {
        vec![CHECKPOINT_PATH]
    } else {
        vec![IMPLEMENTATION_PLAN_PATH, CHECKPOINT_PATH]
    }
}

const CHECKPOINT_VERSION: u8 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct FactoryCheckpoint {
    version: u8,
    repo: String,
    base_branch: String,
    requested_issue: Option<u64>,
    commit_impl_plan: bool,
    issues: Option<Vec<IssueDoc>>,
    candidates: Option<CandidateSet>,
    selection: Option<JudgeSelection>,
    implementation_plan: Option<String>,
    branch: Option<String>,
    base_ref: Option<String>,
    implemented: bool,
    reviewed: bool,
    committed: bool,
    commit_sha: Option<String>,
    pushed: bool,
    pr_draft: Option<PullRequestDraft>,
    pr: Option<CheckpointPullRequest>,
    completed: bool,
}

impl FactoryCheckpoint {
    fn new(
        repo: &RepoSlug,
        base_branch: &str,
        requested_issue: Option<u64>,
        commit_impl_plan: bool,
    ) -> Self {
        Self {
            version: CHECKPOINT_VERSION,
            repo: repo.to_string(),
            base_branch: base_branch.to_owned(),
            requested_issue,
            commit_impl_plan,
            issues: None,
            candidates: None,
            selection: None,
            implementation_plan: None,
            branch: None,
            base_ref: None,
            implemented: false,
            reviewed: false,
            committed: false,
            commit_sha: None,
            pushed: false,
            pr_draft: None,
            pr: None,
            completed: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
struct CheckpointPullRequest {
    number: u64,
    html_url: String,
}

impl From<&GitHubPullRequest> for CheckpointPullRequest {
    fn from(pr: &GitHubPullRequest) -> Self {
        Self {
            number: pr.number,
            html_url: pr.html_url.clone(),
        }
    }
}

fn validate_checkpoint_for_run(
    checkpoint: &FactoryCheckpoint,
    repo: &RepoSlug,
    base_branch: &str,
    requested_issue: Option<u64>,
    commit_impl_plan: bool,
) -> Result<(), String> {
    if checkpoint.version != CHECKPOINT_VERSION {
        return Err(format!(
            "checkpoint version {} is not supported by this binary version {}",
            checkpoint.version, CHECKPOINT_VERSION
        ));
    }
    if checkpoint.repo != repo.to_string() {
        return Err(format!(
            "checkpoint repo {} does not match current repo {repo}",
            checkpoint.repo
        ));
    }
    if checkpoint.base_branch != base_branch {
        return Err(format!(
            "checkpoint base branch {} does not match current base branch {base_branch}",
            checkpoint.base_branch
        ));
    }
    if checkpoint.requested_issue != requested_issue {
        return Err(format!(
            "checkpoint requested issue {:?} does not match current requested issue {:?}",
            checkpoint.requested_issue, requested_issue
        ));
    }
    if checkpoint.commit_impl_plan != commit_impl_plan {
        return Err(format!(
            "checkpoint commit_impl_plan={} does not match current commit_impl_plan={commit_impl_plan}",
            checkpoint.commit_impl_plan
        ));
    }
    validate_checkpoint_stage_state(checkpoint)
}

fn validate_checkpoint_stage_state(checkpoint: &FactoryCheckpoint) -> Result<(), String> {
    if checkpoint.selection.is_some() && checkpoint.candidates.is_none() {
        return Err("checkpoint has a judge selection but no candidate set".to_owned());
    }
    if checkpoint.implementation_plan.is_some() && checkpoint.selection.is_none() {
        return Err("checkpoint has an implementation plan but no judge selection".to_owned());
    }
    if checkpoint.branch.is_some() && checkpoint.selection.is_none() {
        return Err("checkpoint has a branch but no judge selection".to_owned());
    }
    if checkpoint.implemented && checkpoint.branch.is_none() {
        return Err("checkpoint marks implementation complete but has no branch".to_owned());
    }
    if checkpoint.reviewed && !checkpoint.implemented {
        return Err("checkpoint marks review complete before implementation".to_owned());
    }
    if checkpoint.committed && !checkpoint.reviewed {
        return Err("checkpoint marks commit complete before review".to_owned());
    }
    if checkpoint.pushed && !checkpoint.committed {
        return Err("checkpoint marks push complete before commit".to_owned());
    }
    if checkpoint.pr_draft.is_some() && !checkpoint.pushed {
        return Err("checkpoint has a PR draft before push".to_owned());
    }
    if checkpoint.pr.is_some() && checkpoint.pr_draft.is_none() {
        return Err("checkpoint has a PR before PR draft".to_owned());
    }
    if checkpoint.completed && checkpoint.pr.is_none() {
        return Err("checkpoint marks run complete before PR creation".to_owned());
    }
    Ok(())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    init_logging()?;

    let requested_issue = cli
        .issue
        .map(validate_issue_number)
        .transpose()
        .map_err(anyhow::Error::msg)?;
    info!(
        remote = %cli.remote,
        base = ?cli.base,
        branch = ?cli.branch,
        worktree = cli.worktree,
        monitor = cli.monitor,
        allow_dirty = cli.allow_dirty,
        commit_impl_plan = cli.commit_impl_plan,
        resume = cli.resume,
        reset_checkpoint = cli.reset_checkpoint,
        requested_issue = ?requested_issue,
        max_review_iterations = cli.max_review_iterations,
        poll_seconds = cli.poll_seconds,
        "starting software factory run"
    );
    let cwd = std::env::current_dir().context("failed to read current directory")?;
    let launch_worktree = git_worktree_root(&cwd).await?;
    let repo = github_repo_from_git_remote(&launch_worktree, &cli.remote).await?;
    let github = GitHubClient::from_local_credentials(&launch_worktree).await?;
    let repo_info = github.fetch_repo(&repo).await?;
    let base_branch = cli.base.clone().unwrap_or(repo_info.default_branch);
    let run_id = Utc::now().format("%Y%m%d%H%M%S").to_string();
    let worktree = resolve_execution_worktree(
        &launch_worktree,
        cli.worktree,
        cli.resume,
        &repo,
        &base_branch,
        &run_id,
    )
    .await?;
    let project_system_prompt = read_project_system_prompt(&worktree).await?;
    let mut base_config = default_factory_config();
    add_worktree_policy(&mut base_config, &worktree);
    info!(
        cwd = %cwd.display(),
        launch_worktree = %launch_worktree.display(),
        worktree = %worktree.display(),
        worktree_mode = cli.worktree,
        repo = %repo,
        base_branch = %base_branch,
        "resolved repository context"
    );
    if let Some(prompt) = project_system_prompt.as_deref() {
        info!(bytes = prompt.len(), "loaded project guidance");
    } else {
        info!("no project guidance files found");
    }
    let checkpoint_path = worktree.join(CHECKPOINT_PATH);
    let mut checkpoint = initialize_checkpoint(
        &checkpoint_path,
        cli.resume,
        cli.reset_checkpoint,
        &repo,
        &base_branch,
        requested_issue,
        cli.commit_impl_plan,
    )
    .await?;

    let issues = if let Some(issues) = checkpoint.issues.clone() {
        info!(
            count = issues.len(),
            "using issue corpus from factory checkpoint"
        );
        issues
    } else {
        let issues = if let Some(number) = requested_issue {
            info!(repo = %repo, issue = number, "fetching requested open issue");
            vec![github.fetch_open_issue(&repo, number).await?]
        } else {
            info!(repo = %repo, limit = RECENT_OPEN_ISSUE_LIMIT, "fetching recent open issues");
            github
                .fetch_recent_open_issues(&repo, RECENT_OPEN_ISSUE_LIMIT)
                .await?
        };
        if issues.is_empty() {
            bail!("failed to select work: {repo} has no open non-PR issues");
        }
        checkpoint.issues = Some(issues.clone());
        write_checkpoint(&checkpoint_path, &checkpoint).await?;
        issues
    };
    if issues.is_empty() {
        bail!("failed to select work: {repo} has no open non-PR issues");
    }
    info!(issue_count = issues.len(), "loaded issue corpus");
    let issue_cache = issue_cache_from_docs(&issues);
    let allowed_issue_numbers = issues
        .iter()
        .map(|issue| issue.number)
        .collect::<HashSet<_>>();
    let corpus = issue_corpus(&repo, &issues, None);
    let planning_corpus = issue_corpus(&repo, &issues, Some(PLANNING_CORPUS_BODY_CHARS));
    let implementation_plan_path = worktree.join(IMPLEMENTATION_PLAN_PATH);
    if let Some(parent) = implementation_plan_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let glm = build_model_harness(
        &base_config,
        "issue grouping and feedback refinement",
        ModelSpec::parse(&cli.glm_model).map_err(anyhow::Error::msg)?,
        ReasoningEffort::Xhigh,
        &worktree,
    )
    .await?;
    let judge = build_judge_harness(
        &base_config,
        &worktree,
        Arc::new(GitHubIssueTool::new(
            github.clone(),
            repo.clone(),
            issue_cache.clone(),
            allowed_issue_numbers,
        )),
    )
    .await?;
    let implementer = build_model_harness(
        &base_config,
        "implementation",
        ModelSpec::parse(&cli.implementer_model).map_err(anyhow::Error::msg)?,
        ReasoningEffort::Xhigh,
        &worktree,
    )
    .await?;
    let reviewer = build_model_harness(
        &base_config,
        "code review",
        ModelSpec::parse(&cli.reviewer_model).map_err(anyhow::Error::msg)?,
        ReasoningEffort::Xhigh,
        &worktree,
    )
    .await?;
    let pr_writer = build_model_harness(
        &base_config,
        "pr drafting",
        ModelSpec::parse(&cli.pr_model).map_err(anyhow::Error::msg)?,
        ReasoningEffort::Medium,
        &worktree,
    )
    .await?;

    let candidates = if let Some(candidates) = checkpoint.candidates.clone() {
        info!("using issue candidates from factory checkpoint");
        candidates
    } else {
        let candidates = if let Some(number) = requested_issue {
            let issue = issues
                .iter()
                .find(|issue| issue.number == number)
                .with_context(|| format!("failed to find requested issue #{number} after fetch"))?;
            info!(issue = number, "using requested issue as the candidate set");
            candidate_set_for_issue(issue)
        } else {
            propose_issue_candidates(
                &glm,
                &worktree,
                &repo,
                &corpus,
                project_system_prompt.as_deref(),
            )
            .await?
        };
        checkpoint.candidates = Some(candidates.clone());
        write_checkpoint(&checkpoint_path, &checkpoint).await?;
        candidates
    };
    let selection = if let Some(selection) = checkpoint.selection.clone() {
        info!("using judge selection from factory checkpoint");
        selection
    } else {
        let selection = judge_issue_plan(
            &judge,
            &worktree,
            &repo,
            &planning_corpus,
            &candidates,
            IMPLEMENTATION_PLAN_PATH,
            project_system_prompt.as_deref(),
        )
        .await?;
        checkpoint.selection = Some(selection.clone());
        let implementation_plan = read_implementation_plan(&implementation_plan_path).await?;
        checkpoint.implementation_plan = Some(implementation_plan);
        write_checkpoint(&checkpoint_path, &checkpoint).await?;
        selection
    };
    let implementation_plan =
        if let Some(implementation_plan) = checkpoint.implementation_plan.clone() {
            if implementation_plan.trim().is_empty() {
                bail!("factory checkpoint implementation plan is empty");
            }
            restore_implementation_plan(&implementation_plan_path, &implementation_plan).await?;
            implementation_plan
        } else {
            let implementation_plan = read_implementation_plan(&implementation_plan_path).await?;
            checkpoint.implementation_plan = Some(implementation_plan.clone());
            write_checkpoint(&checkpoint_path, &checkpoint).await?;
            implementation_plan
        };
    let issue_numbers = selected_issue_numbers(&selection);
    if issue_numbers.is_empty() {
        bail!("failed to select work: judge did not return issue numbers");
    }
    ensure_requested_issue_selection(&selection, requested_issue).map_err(anyhow::Error::msg)?;
    ensure_selected_issues_are_open(&issues, &issue_numbers)?;

    let (current_branch, base_ref) = if let Some(branch) = checkpoint.branch.clone() {
        info!(branch = %branch, "checking out checkpoint branch");
        checkout_branch(&worktree, &branch).await?;
        let base_ref = checkpoint
            .base_ref
            .clone()
            .unwrap_or_else(|| format!("origin/{base_branch}"));
        (branch, base_ref)
    } else {
        let branch = prepare_branch(
            &worktree,
            &base_branch,
            cli.branch.as_deref(),
            cli.allow_dirty,
            &repo,
            &selection,
            &FACTORY_LOCAL_STATE_PATHS,
            &run_id,
        )
        .await?;
        let base_ref = format!("origin/{base_branch}");
        checkpoint.branch = Some(branch.clone());
        checkpoint.base_ref = Some(base_ref.clone());
        write_checkpoint(&checkpoint_path, &checkpoint).await?;
        (branch, base_ref)
    };
    info!(branch = %current_branch, base_ref = %base_ref, "prepared branch");
    let commit_excluded_paths = excluded_commit_paths(cli.commit_impl_plan);

    if checkpoint.implemented {
        info!("skipping implementation; factory checkpoint marks it complete");
    } else {
        implement_plan(
            &implementer,
            &worktree,
            &repo,
            &selection,
            &implementation_plan,
            &issues,
            project_system_prompt.as_deref(),
        )
        .await?;
        checkpoint.implemented = true;
        write_checkpoint(&checkpoint_path, &checkpoint).await?;
    }

    if checkpoint.reviewed {
        info!("skipping review loop; factory checkpoint marks it complete");
    } else {
        run_review_loop(
            &implementer,
            &reviewer,
            &worktree,
            &base_ref,
            &implementation_plan,
            cli.max_review_iterations,
            project_system_prompt.as_deref(),
        )
        .await?;
        checkpoint.reviewed = true;
        write_checkpoint(&checkpoint_path, &checkpoint).await?;
        info!("implementation review loop completed");
    }

    if !checkpoint.committed {
        if !branch_has_diff(&worktree, &base_ref).await? {
            bail!("failed to create PR: implementation produced no diff against {base_ref}");
        }
        let committed = commit_if_dirty(
            &worktree,
            &format!("Implement {}", selection.title),
            &commit_excluded_paths,
        )
        .await?;
        checkpoint.committed = true;
        checkpoint.commit_sha = Some(current_commit(&worktree).await?);
        write_checkpoint(&checkpoint_path, &checkpoint).await?;
        info!(committed, "commit step completed");
    } else {
        info!("skipping commit; factory checkpoint marks it complete");
    }

    if checkpoint.pushed {
        info!("skipping push; factory checkpoint marks it complete");
    } else {
        run_cmd(&worktree, "git", &["push", "-u", "origin", &current_branch]).await?;
        checkpoint.pushed = true;
        write_checkpoint(&checkpoint_path, &checkpoint).await?;
    }

    let pr_draft = if let Some(pr_draft) = checkpoint.pr_draft.clone() {
        info!("using PR draft from factory checkpoint");
        pr_draft
    } else {
        let final_diff = branch_diff(&worktree, &base_ref).await?;
        let pr_draft = draft_pr(
            &pr_writer,
            &worktree,
            &repo,
            &selection,
            &implementation_plan,
            &issue_numbers,
            &final_diff,
            project_system_prompt.as_deref(),
        )
        .await?;
        checkpoint.pr_draft = Some(pr_draft.clone());
        write_checkpoint(&checkpoint_path, &checkpoint).await?;
        pr_draft
    };
    let pr = if let Some(pr) = checkpoint.pr.clone() {
        info!(pr_number = pr.number, url = %pr.html_url, "using PR from factory checkpoint");
        pr
    } else {
        let pr = github
            .create_pull_request(&repo, &current_branch, &base_branch, &pr_draft)
            .await?;
        let pr = CheckpointPullRequest::from(&pr);
        checkpoint.pr = Some(pr.clone());
        write_checkpoint(&checkpoint_path, &checkpoint).await?;
        info!(pr_number = pr.number, url = %pr.html_url, "created pull request");
        pr
    };

    println!("created PR: {}", pr.html_url);

    if cli.monitor {
        monitor_pr(MonitorContext {
            github: &github,
            glm: &glm,
            implementer: &implementer,
            reviewer: &reviewer,
            worktree: &worktree,
            repo: &repo,
            pr_number: pr.number,
            branch: &current_branch,
            base_ref: &base_ref,
            selection: &selection,
            implementation_plan: &implementation_plan,
            project_system_prompt: project_system_prompt.as_deref(),
            excluded_commit_paths: &commit_excluded_paths,
            max_review_iterations: cli.max_review_iterations,
            poll_seconds: cli.poll_seconds,
        })
        .await?;
    }
    checkpoint.completed = true;
    write_checkpoint(&checkpoint_path, &checkpoint).await?;

    info!("shutting down harnesses");
    shutdown_all([&glm, &judge, &implementer, &reviewer, &pr_writer]).await;
    info!("software factory run complete");
    Ok(())
}

async fn canonicalize_existing(path: impl AsRef<Path>) -> anyhow::Result<PathBuf> {
    tokio::fs::canonicalize(path.as_ref())
        .await
        .with_context(|| format!("failed to canonicalize {}", path.as_ref().display()))
}

async fn resolve_execution_worktree(
    launch_worktree: &Path,
    use_tmp_worktree: bool,
    resume: bool,
    repo: &RepoSlug,
    base_branch: &str,
    run_id: &str,
) -> anyhow::Result<PathBuf> {
    if !use_tmp_worktree {
        return Ok(launch_worktree.to_path_buf());
    }

    if resume {
        if path_is_factory_tmp_worktree(launch_worktree) {
            info!(
                worktree = %launch_worktree.display(),
                "resuming existing factory git worktree"
            );
            return Ok(launch_worktree.to_path_buf());
        }
        bail!(
            "failed to resume --worktree: cd into the existing factory worktree under {} and run --resume",
            factory_worktree_tmp_root().display()
        );
    }

    create_factory_worktree(launch_worktree, repo, base_branch, run_id).await
}

fn factory_worktree_tmp_root() -> PathBuf {
    PathBuf::from("/tmp").join(FACTORY_WORKTREE_TMP_DIR)
}

fn path_is_factory_tmp_worktree(path: &Path) -> bool {
    path.starts_with(factory_worktree_tmp_root())
        || path.starts_with(Path::new("/private/tmp").join(FACTORY_WORKTREE_TMP_DIR))
}

async fn create_factory_worktree(
    source_worktree: &Path,
    repo: &RepoSlug,
    base_branch: &str,
    run_id: &str,
) -> anyhow::Result<PathBuf> {
    let parent = factory_worktree_tmp_root();
    tokio::fs::create_dir_all(&parent).await.with_context(|| {
        format!(
            "failed to create factory worktree directory {}",
            parent.display()
        )
    })?;

    let worktree = parent.join(factory_worktree_dir_name(repo, run_id));
    if tokio::fs::try_exists(&worktree)
        .await
        .with_context(|| format!("failed to inspect factory worktree {}", worktree.display()))?
    {
        bail!(
            "factory git worktree path already exists: {}; use --resume from that worktree or choose a different run",
            worktree.display()
        );
    }

    run_cmd(source_worktree, "git", &["fetch", "origin", base_branch]).await?;
    let worktree_arg = worktree.to_str().with_context(|| {
        format!(
            "factory worktree path is not valid UTF-8: {}",
            worktree.display()
        )
    })?;
    let base_ref = format!("origin/{base_branch}");
    info!(
        source_worktree = %source_worktree.display(),
        worktree = %worktree.display(),
        base_ref = %base_ref,
        "creating factory git worktree"
    );
    run_cmd(
        source_worktree,
        "git",
        &["worktree", "add", "--detach", worktree_arg, &base_ref],
    )
    .await?;
    canonicalize_existing(&worktree).await
}

async fn initialize_checkpoint(
    path: &Path,
    resume: bool,
    reset: bool,
    repo: &RepoSlug,
    base_branch: &str,
    requested_issue: Option<u64>,
    commit_impl_plan: bool,
) -> anyhow::Result<FactoryCheckpoint> {
    if reset {
        remove_checkpoint_if_exists(path).await?;
    }

    if resume {
        let checkpoint = read_checkpoint(path).await?;
        validate_checkpoint_for_run(
            &checkpoint,
            repo,
            base_branch,
            requested_issue,
            commit_impl_plan,
        )
        .map_err(anyhow::Error::msg)?;
        info!(path = %path.display(), "loaded factory checkpoint");
        return Ok(checkpoint);
    }

    if checkpoint_exists(path).await? {
        bail!(
            "factory checkpoint already exists at {}; pass --resume to continue it or --reset-checkpoint to start over",
            path.display()
        );
    }

    let checkpoint = FactoryCheckpoint::new(repo, base_branch, requested_issue, commit_impl_plan);
    write_checkpoint(path, &checkpoint).await?;
    Ok(checkpoint)
}

async fn checkpoint_exists(path: &Path) -> anyhow::Result<bool> {
    match tokio::fs::metadata(path).await {
        Ok(metadata) if metadata.is_file() => Ok(true),
        Ok(_) => bail!(
            "factory checkpoint path exists but is not a regular file: {}",
            path.display()
        ),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error)
            .with_context(|| format!("failed to inspect factory checkpoint {}", path.display())),
    }
}

async fn read_checkpoint(path: &Path) -> anyhow::Result<FactoryCheckpoint> {
    let raw = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("failed to read factory checkpoint {}", path.display()))?;
    serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse factory checkpoint {}", path.display()))
}

async fn write_checkpoint(path: &Path, checkpoint: &FactoryCheckpoint) -> anyhow::Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("checkpoint path {} has no parent", path.display()))?;
    tokio::fs::create_dir_all(parent)
        .await
        .with_context(|| format!("failed to create checkpoint directory {}", parent.display()))?;

    let tmp = path.with_extension("json.tmp");
    let data =
        serde_json::to_vec_pretty(checkpoint).context("failed to serialize factory checkpoint")?;
    tokio::fs::write(&tmp, data)
        .await
        .with_context(|| format!("failed to write temporary checkpoint {}", tmp.display()))?;
    match tokio::fs::rename(&tmp, path).await {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::AlreadyExists => {
            tokio::fs::remove_file(path).await.with_context(|| {
                format!("failed to replace factory checkpoint {}", path.display())
            })?;
            tokio::fs::rename(&tmp, path).await.with_context(|| {
                format!("failed to install factory checkpoint {}", path.display())
            })?;
        }
        Err(error) => {
            return Err(error).with_context(|| {
                format!("failed to install factory checkpoint {}", path.display())
            });
        }
    }
    info!(path = %path.display(), "wrote factory checkpoint");
    Ok(())
}

async fn remove_checkpoint_if_exists(path: &Path) -> anyhow::Result<()> {
    match tokio::fs::remove_file(path).await {
        Ok(()) => {
            info!(path = %path.display(), "removed factory checkpoint");
            Ok(())
        }
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(()),
        Err(error) => Err(error)
            .with_context(|| format!("failed to remove factory checkpoint {}", path.display())),
    }
}

async fn restore_implementation_plan(path: &Path, plan: &str) -> anyhow::Result<()> {
    let parent = path
        .parent()
        .with_context(|| format!("implementation plan path {} has no parent", path.display()))?;
    tokio::fs::create_dir_all(parent)
        .await
        .with_context(|| format!("failed to create {}", parent.display()))?;
    tokio::fs::write(path, plan)
        .await
        .with_context(|| format!("failed to restore implementation plan {}", path.display()))
}

async fn read_project_system_prompt(worktree: &Path) -> anyhow::Result<Option<String>> {
    let mut docs = Vec::new();
    for filename in PROJECT_GUIDANCE_FILENAMES {
        let path = worktree.join(filename);
        let metadata = match tokio::fs::metadata(&path).await {
            Ok(metadata) => metadata,
            Err(error) if error.kind() == ErrorKind::NotFound => continue,
            Err(error) => {
                return Err(error).with_context(|| {
                    format!("failed to inspect project guidance {}", path.display())
                });
            }
        };
        if !metadata.is_file() {
            warn!(
                path = %path.display(),
                "skipping project guidance path because it is not a regular file"
            );
            continue;
        }
        if metadata.len() > PROJECT_GUIDANCE_MAX_BYTES {
            bail!(
                "failed to read project guidance {}: file is {} bytes, above the {} byte limit",
                path.display(),
                metadata.len(),
                PROJECT_GUIDANCE_MAX_BYTES
            );
        }

        let text = tokio::fs::read_to_string(&path)
            .await
            .with_context(|| format!("failed to read project guidance {}", path.display()))?;
        docs.push(ProjectGuidanceDoc {
            filename: filename.to_owned(),
            text,
        });
    }
    Ok(format_project_system_prompt(&docs))
}

fn default_factory_config() -> HarnessConfig {
    HarnessConfig {
        version: 1,
        providers: ProvidersConfig {
            openai: Some(ProviderConfig::default()),
            anthropic: None,
            openrouter: Some(ProviderConfig::default()),
        },
        models: ModelsConfig {
            default: Some(ModelSlot::Reference(ModelSlotRef::ModelJudge)),
            subagent: Some(ModelSlot::Reference(ModelSlotRef::AutoResolve)),
            small: Some(model_config(
                ConfiguredProvider::OpenAi,
                "gpt-5.5",
                ReasoningEffort::Medium,
            )),
            model_judge: Some(ModelJudgeConfig {
                mode: ModelJudgeMode::FullTurn,
                default: model_config(
                    ConfiguredProvider::OpenRouter,
                    "z-ai/glm-5.2",
                    ReasoningEffort::Xhigh,
                ),
                synthesis: model_config(
                    ConfiguredProvider::OpenRouter,
                    "google/gemma-4-31b-it",
                    ReasoningEffort::High,
                ),
                panel: vec![
                    model_config(
                        ConfiguredProvider::OpenRouter,
                        "minimax/minimax-m3",
                        ReasoningEffort::Xhigh,
                    ),
                    // model_config(
                    //     ConfiguredProvider::OpenRouter,
                    //     "nvidia/nemotron-3-ultra-550b-a55b",
                    //     ReasoningEffort::Xhigh,
                    // ),
                    model_config(
                        ConfiguredProvider::OpenRouter,
                        "moonshotai/kimi-k2.6",
                        ReasoningEffort::Xhigh,
                    ),
                    model_config(
                        ConfiguredProvider::OpenRouter,
                        "qwen/qwen3.6-27b",
                        ReasoningEffort::Xhigh,
                    ),
                ],
                panel_isolation: PanelIsolation::ReadOnly,
            }),
        },
        resources: ResourcesConfig {
            skills: SearchRoots {
                roots: vec![PathBuf::from("./.agent/skills")],
            },
            plugins: SearchRoots {
                roots: vec![PathBuf::from("./.agent/plugins")],
            },
        },
        context: ContextConfig {
            compaction_threshold: 230_000,
            pre_compaction_target: 150_000,
            prune_signal_threshold: PruneSignalThreshold::Low,
        },
        tools: ToolsConfig {
            enabled: judge_example_tools()
                .into_iter()
                .map(ToOwned::to_owned)
                .collect(),
        },
        policy: PolicyConfig {
            allowed_write_roots: vec![PathBuf::from("./"), PathBuf::from("/tmp/halter")],
            max_read_bytes: 1_048_576,
            max_subagent_depth: 3,
            max_concurrent_subagents: 8,
            shell: ShellPolicyConfig {
                enabled: true,
                allow: judge_example_shell_allowlist()
                    .into_iter()
                    .map(ToOwned::to_owned)
                    .collect(),
                timeout_secs: 30,
            },
            network: NetworkPolicyConfig {
                enabled: true,
                ..NetworkPolicyConfig::default()
            },
        },
        sessions: SessionsConfig::default(),
        runtime: RuntimeConfig {
            traces_dir: Some(PathBuf::from("~/.halter/traces/")),
            ..RuntimeConfig::default()
        },
        ..HarnessConfig::default()
    }
}

fn model_config(
    provider: ConfiguredProvider,
    model: impl Into<String>,
    reasoning: ReasoningEffort,
) -> ModelConfig {
    ModelConfig {
        provider,
        model: model.into(),
        max_input_tokens: None,
        max_output_tokens: None,
        reasoning: Some(reasoning),
        tokens_per_minute: Some(500_000),
    }
}

fn judge_example_tools() -> [&'static str; 17] {
    [
        "read",
        "glob",
        "grep",
        "profile",
        "write",
        "edit",
        "shell",
        "process",
        "task",
        "pty",
        "ast_grep",
        "image",
        "wait_agent",
        "spawn_agent",
        "send_input",
        "close_agent",
        "browser",
    ]
}

fn judge_example_shell_allowlist() -> [&'static str; 17] {
    [
        "git", "cargo", "rg", "ls", "find", "python", "python3", "pwd", "echo", "date", "gh",
        "which", "sort", "nl", "sed", "wc", "head",
    ]
}

async fn git_worktree_root(cwd: &Path) -> anyhow::Result<PathBuf> {
    let root = run_cmd(cwd, "git", &["rev-parse", "--show-toplevel"])
        .await
        .context("failed to locate git worktree root; run software-factory inside a git repo")?;
    canonicalize_existing(root.trim()).await
}

async fn github_repo_from_git_remote(worktree: &Path, remote: &str) -> anyhow::Result<RepoSlug> {
    let remote_url = run_cmd(
        worktree,
        "git",
        &["config", "--get", &format!("remote.{remote}.url")],
    )
    .await
    .with_context(|| format!("failed to read git remote URL for remote '{remote}'"))?;
    parse_github_remote_url(&remote_url).map_err(anyhow::Error::msg)
}

type IssueCache = Arc<RwLock<HashMap<u64, IssueDoc>>>;

fn issue_cache_from_docs(issues: &[IssueDoc]) -> IssueCache {
    Arc::new(RwLock::new(
        issues
            .iter()
            .cloned()
            .map(|issue| (issue.number, issue))
            .collect(),
    ))
}

#[derive(Clone)]
struct GitHubIssueTool {
    github: GitHubClient,
    repo: RepoSlug,
    cache: IssueCache,
    allowed_numbers: HashSet<u64>,
}

impl GitHubIssueTool {
    fn new(
        github: GitHubClient,
        repo: RepoSlug,
        cache: IssueCache,
        allowed_numbers: HashSet<u64>,
    ) -> Self {
        Self {
            github,
            repo,
            cache,
            allowed_numbers,
        }
    }

    async fn cached_or_fetch(&self, number: u64) -> anyhow::Result<(IssueDoc, &'static str)> {
        if !self.allowed_numbers.contains(&number) {
            bail!("failed to fetch issue #{number}: issue is outside the current issue corpus");
        }
        if let Some(issue) = self.cache.read().await.get(&number).cloned() {
            info!(issue = number, "github_issue tool cache hit");
            return Ok((issue, "cache"));
        }

        info!(issue = number, repo = %self.repo, "github_issue tool fetching issue");
        let issue = self.github.fetch_open_issue(&self.repo, number).await?;
        self.cache.write().await.insert(number, issue.clone());
        Ok((issue, "github"))
    }
}

#[async_trait]
impl Tool for GitHubIssueTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("github_issue"),
            description: "Fetch full text for an open GitHub issue in the current factory corpus. Returns cached issue text when available.".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "number": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "GitHub issue number"
                    }
                },
                "required": ["number"]
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

    async fn execute(&self, _context: ToolContext, input: Value) -> anyhow::Result<ToolResult> {
        let number = parse_issue_number_input(&input).map_err(anyhow::Error::msg)?;
        let (issue, source) = self.cached_or_fetch(number).await?;
        Ok(ToolResult::Json {
            value: json!({
                "source": source,
                "issue": issue,
            }),
        })
    }
}

fn add_worktree_policy(config: &mut HarnessConfig, worktree: &Path) {
    absolutize_relative_roots(&mut config.policy.allowed_write_roots, worktree);
    if !config
        .policy
        .allowed_write_roots
        .iter()
        .any(|root| root == worktree)
    {
        config
            .policy
            .allowed_write_roots
            .push(worktree.to_path_buf());
    }
    absolutize_relative_roots(&mut config.resources.skills.roots, worktree);
    absolutize_relative_roots(&mut config.resources.plugins.roots, worktree);
}

fn absolutize_relative_roots(roots: &mut [PathBuf], worktree: &Path) {
    for root in roots {
        if root.is_relative() && !path_starts_with_tilde(root) {
            *root = if root == Path::new(".") || root == Path::new("./") {
                worktree.to_path_buf()
            } else {
                worktree.join(&root)
            };
        }
    }
}

fn path_starts_with_tilde(path: &Path) -> bool {
    path.components()
        .next()
        .is_some_and(|component| matches!(component, Component::Normal(value) if value == "~"))
}

async fn build_judge_harness(
    config: &HarnessConfig,
    worktree: &Path,
    issue_tool: Arc<dyn Tool>,
) -> anyhow::Result<Halter> {
    info!("building model judge harness");
    let mut config = config.clone();
    add_worktree_policy(&mut config, worktree);
    if !config
        .tools
        .enabled
        .iter()
        .any(|tool| tool == "github_issue")
    {
        config.tools.enabled.push("github_issue".to_owned());
    }
    let resources = ResourceCompiler::from_config(&config).compile().await?;
    let harness = Halter::builder()
        .with_config(config)
        .with_compiled_resources(resources)
        .with_tool(issue_tool)
        .build()
        .await?;
    info!("built model judge harness");
    Ok(harness)
}

async fn build_model_harness(
    config: &HarnessConfig,
    role: &str,
    model: ModelSpec,
    reasoning: ReasoningEffort,
    worktree: &Path,
) -> anyhow::Result<Halter> {
    info!(
        role,
        provider = ?model.provider,
        model = %model.model,
        reasoning = ?reasoning,
        "building model harness"
    );
    let mut config = config.clone();
    add_worktree_policy(&mut config, worktree);
    let model = model.into_model_config(reasoning, Some(230_000), Some(16_384));
    config.models.default = Some(ModelSlot::Inline(model.clone()));
    config.models.small = Some(model.clone());
    config.models.subagent = Some(ModelSlot::Inline(model));
    let resources = ResourceCompiler::from_config(&config).compile().await?;
    let harness = Halter::from_compiled_resources(config, resources).await?;
    info!(role, "built model harness");
    Ok(harness)
}

async fn shutdown_all<'a>(harnesses: impl IntoIterator<Item = &'a Halter>) {
    for harness in harnesses {
        let _ = harness.shutdown(Duration::from_secs(10)).await;
    }
}

#[derive(Debug, PartialEq, Eq)]
struct AgentRun {
    text: String,
}

const FACTORY_TURN_USER_MESSAGE: &str =
    "Execute the appended turn-specific instructions for this software factory stage.";

/// Shared closing instruction for stages whose response is parsed as JSON.
const JSON_ONLY_OUTPUT_RULE: &str = "Return ONLY the JSON object as your final message — no markdown code fences and no surrounding prose.";

/// Shared rule for coding stages that run cargo, whose builds exceed the
/// 30-second default shell timeout when no explicit timeout is supplied.
const CARGO_TIMEOUT_RULE: &str = "When running builds, tests, lints, or other checks through the shell tool, pass an explicit timeout_ms of at least 120000; these commands routinely exceed the 30-second default.";
const CODE_REVIEW_MAX_TURNS: u32 = 10;

fn json_preview(value: &Value, max_chars: usize) -> String {
    single_line_preview(&value.to_string(), max_chars)
}

fn single_line_preview(text: &str, max_chars: usize) -> String {
    let normalized = text.replace('\n', "\\n").replace('\r', "\\r");
    let mut preview = String::new();
    let mut truncated = false;
    for (index, ch) in normalized.chars().enumerate() {
        if index == max_chars {
            truncated = true;
            break;
        }
        preview.push(ch);
    }
    if truncated {
        preview.push_str("...");
    }
    preview
}

fn tool_result_kind(result: &ToolResult) -> &'static str {
    match result {
        ToolResult::Empty => "empty",
        ToolResult::Text { .. } => "text",
        ToolResult::Json { .. } => "json",
    }
}

fn tool_result_size(result: &ToolResult) -> usize {
    match result {
        ToolResult::Empty => 0,
        ToolResult::Text { text } => text.len(),
        ToolResult::Json { value } => value.to_string().len(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AgentTextRequirement {
    Required,
    Optional,
}

const CODING_STAGE_RETRY_POLICY: AgentStageRetryPolicy = AgentStageRetryPolicy {
    max_attempts: 3,
    base_backoff: Duration::from_secs(5),
    max_backoff: Duration::from_secs(30),
};
const INFERRED_AGENT_STAGE_CAPACITY_BACKOFF: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AgentStageRetryPolicy {
    max_attempts: u32,
    base_backoff: Duration,
    max_backoff: Duration,
}

impl AgentStageRetryPolicy {
    fn delay_after_failure(self, failed_attempt: u32, error: &str) -> Option<Duration> {
        if failed_attempt >= self.max_attempts {
            return None;
        }
        let delay = inferred_agent_stage_backoff_hint(error)
            .unwrap_or_else(|| exponential_agent_stage_backoff(self, failed_attempt));
        Some(delay.min(self.max_backoff))
    }
}

fn exponential_agent_stage_backoff(policy: AgentStageRetryPolicy, failed_attempt: u32) -> Duration {
    let exponent = failed_attempt.saturating_sub(1).min(31);
    let multiplier = 1u128 << exponent;
    let millis = policy
        .base_backoff
        .as_millis()
        .saturating_mul(multiplier)
        .min(policy.max_backoff.as_millis())
        .min(u128::from(u64::MAX));
    Duration::from_millis(millis as u64)
}

fn inferred_agent_stage_backoff_hint(error: &str) -> Option<Duration> {
    let lower = error.to_ascii_lowercase();
    (lower.contains("overloaded")
        || lower.contains("capacity")
        || lower.contains("rate limit")
        || lower.contains("rate_limit")
        || lower.contains("too many requests"))
    .then_some(INFERRED_AGENT_STAGE_CAPACITY_BACKOFF)
}

fn agent_stage_failure_is_retryable(retryable: bool, cancelled: bool, error: &str) -> bool {
    !cancelled && (retryable || inferred_agent_stage_backoff_hint(error).is_some())
}

#[derive(Debug)]
struct AgentStageTurnFailure {
    label: String,
    error: String,
    retryable: bool,
    cancelled: bool,
}

impl AgentStageTurnFailure {
    fn should_retry(&self) -> bool {
        agent_stage_failure_is_retryable(self.retryable, self.cancelled, &self.error)
    }
}

impl fmt::Display for AgentStageTurnFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "agent stage {} failed: {}",
            self.label, self.error
        )
    }
}

impl StdError for AgentStageTurnFailure {}

fn agent_stage_error_is_retryable(error: &anyhow::Error) -> bool {
    error.downcast_ref::<AgentStageTurnFailure>().map_or_else(
        || inferred_agent_stage_backoff_hint(&error.to_string()).is_some(),
        AgentStageTurnFailure::should_retry,
    )
}

#[derive(Debug, Clone, Copy)]
enum FactorySystemPrompt {
    General,
    Coding,
}

impl FactorySystemPrompt {
    fn segment(self) -> PromptSegment {
        match self {
            Self::General => prompts::default_system_prompt_segment(),
            Self::Coding => prompts::coding_agent_prompt_segment(),
        }
    }
}

fn session_init_with_appended_context(
    worktree: &Path,
    system_prompt: FactorySystemPrompt,
    turn_instructions: &str,
    project_system_prompt: Option<&str>,
    max_turns: Option<u32>,
) -> anyhow::Result<SessionInit> {
    let mut init = SessionInit {
        working_dir: worktree.to_path_buf(),
        system_prompt_seed: vec![system_prompt.segment()],
        max_turns,
        ..SessionInit::default()
    };
    if let Some(segment) = project_guidance_prompt_segment(project_system_prompt) {
        init.system_prompt_seed.push(segment);
    }
    init.system_prompt_seed
        .push(turn_instructions_prompt_segment(turn_instructions)?);
    Ok(init)
}

fn project_guidance_prompt_segment(project_system_prompt: Option<&str>) -> Option<PromptSegment> {
    let text = project_system_prompt?.trim();
    if text.is_empty() {
        return None;
    }
    Some(append_prompt_segment(text))
}

fn turn_instructions_prompt_segment(turn_instructions: &str) -> anyhow::Result<PromptSegment> {
    let turn_instructions = turn_instructions.trim();
    if turn_instructions.is_empty() {
        bail!("failed to start agent turn: turn-specific instructions are empty");
    }
    Ok(append_prompt_segment(&format!(
        "# Turn-Specific Instructions\n\n{turn_instructions}"
    )))
}

fn append_prompt_segment(text: &str) -> PromptSegment {
    let text = text.trim().to_owned();
    PromptSegment {
        id: PromptSegmentId::new(),
        content_hash: hash_prompt_text(&text),
        text,
        volatility: Volatility::TurnDynamic,
        cache_scope: CacheScope::Dynamic,
        kind: PromptSegmentKind::Append,
    }
}

fn hash_prompt_text(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

async fn run_code_review_agent_with_system_prompt(
    harness: &Halter,
    worktree: &Path,
    label: &str,
    prompt: impl Into<String>,
    project_system_prompt: Option<&str>,
) -> anyhow::Result<AgentRun> {
    run_agent_with_prompt_kind(
        harness,
        worktree,
        label,
        prompt,
        FactorySystemPrompt::Coding,
        project_system_prompt,
        AgentTextRequirement::Required,
        Some(CODE_REVIEW_MAX_TURNS),
    )
    .await
}

async fn run_coding_action_with_system_prompt(
    harness: &Halter,
    worktree: &Path,
    label: &str,
    prompt: impl Into<String>,
    project_system_prompt: Option<&str>,
) -> anyhow::Result<()> {
    run_agent_with_prompt_kind_with_retry(
        harness,
        worktree,
        label,
        prompt,
        FactorySystemPrompt::Coding,
        project_system_prompt,
        AgentTextRequirement::Optional,
        CODING_STAGE_RETRY_POLICY,
    )
    .await?;
    Ok(())
}

async fn run_agent_with_system_prompt(
    harness: &Halter,
    worktree: &Path,
    label: &str,
    prompt: impl Into<String>,
    project_system_prompt: Option<&str>,
) -> anyhow::Result<AgentRun> {
    run_agent_with_prompt_kind(
        harness,
        worktree,
        label,
        prompt,
        FactorySystemPrompt::General,
        project_system_prompt,
        AgentTextRequirement::Required,
        None,
    )
    .await
}

async fn run_agent_with_prompt_kind_with_retry(
    harness: &Halter,
    worktree: &Path,
    label: &str,
    prompt: impl Into<String>,
    system_prompt: FactorySystemPrompt,
    project_system_prompt: Option<&str>,
    text_requirement: AgentTextRequirement,
    retry_policy: AgentStageRetryPolicy,
) -> anyhow::Result<AgentRun> {
    let prompt = prompt.into();
    let mut attempt = 1;

    loop {
        match run_agent_with_prompt_kind(
            harness,
            worktree,
            label,
            prompt.clone(),
            system_prompt,
            project_system_prompt,
            text_requirement,
            None,
        )
        .await
        {
            Ok(run) => return Ok(run),
            Err(error) if agent_stage_error_is_retryable(&error) => {
                let message = error.to_string();
                let Some(delay) = retry_policy.delay_after_failure(attempt, &message) else {
                    warn!(
                        stage = label,
                        attempt,
                        max_attempts = retry_policy.max_attempts,
                        error = %message,
                        "agent stage retry budget exhausted"
                    );
                    return Err(error);
                };
                warn!(
                    stage = label,
                    attempt,
                    max_attempts = retry_policy.max_attempts,
                    retry_in_ms = delay.as_millis() as u64,
                    error = %message,
                    "retrying agent stage after transient failure"
                );
                tokio::time::sleep(delay).await;
                attempt += 1;
            }
            Err(error) => return Err(error),
        }
    }
}

async fn run_agent_with_prompt_kind(
    harness: &Halter,
    worktree: &Path,
    label: &str,
    prompt: impl Into<String>,
    system_prompt: FactorySystemPrompt,
    project_system_prompt: Option<&str>,
    text_requirement: AgentTextRequirement,
    max_turns: Option<u32>,
) -> anyhow::Result<AgentRun> {
    let turn_instructions = prompt.into();
    info!(
        stage = label,
        system_prompt = ?system_prompt,
        max_turns = ?max_turns,
        prompt_bytes = turn_instructions.len(),
        project_guidance = project_system_prompt.is_some_and(|prompt| !prompt.trim().is_empty()),
        "starting agent turn"
    );
    let session = harness
        .new_session(session_init_with_appended_context(
            worktree,
            system_prompt,
            &turn_instructions,
            project_system_prompt,
            max_turns,
        )?)
        .await?;
    let mut events = session
        .submit_turn(Turn::user(FACTORY_TURN_USER_MESSAGE))
        .await?;
    let mut latest_text = None;
    let mut delta_text = String::new();
    let mut usage = Usage::default();

    while let Some(event) = events.next().await {
        let event =
            event.with_context(|| format!("failed to read event for agent stage {label}"))?;
        match event.payload {
            SessionEventPayload::SessionStarted => {
                info!(stage = label, "agent session started");
            }
            SessionEventPayload::Warning { message } => {
                warn!(stage = label, warning = %message, "agent warning");
            }
            SessionEventPayload::TurnStarted { turn_id } => {
                info!(stage = label, turn_id = %turn_id, "agent turn started");
            }
            SessionEventPayload::DeltaItem { delta } => {
                debug!(stage = label, bytes = delta.text.len(), "assistant delta");
                delta_text.push_str(&delta.text);
            }
            SessionEventPayload::MessageItem {
                message: Message::Assistant(message),
            } => {
                latest_text = Some(
                    message
                        .parts
                        .iter()
                        .filter_map(|part| match part {
                            AssistantPart::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<String>(),
                );
            }
            SessionEventPayload::ToolExecutionStarted { call } => {
                info!(
                    stage = label,
                    tool = %call.name,
                    call_id = %call.id,
                    arguments = %json_preview(&call.arguments, 500),
                    "tool started"
                );
            }
            SessionEventPayload::ToolOutput {
                call_id,
                tool_name,
                chunk,
            } => {
                debug!(
                    stage = label,
                    tool = %tool_name,
                    call_id = %call_id,
                    bytes = chunk.len(),
                    preview = %single_line_preview(&chunk, 300),
                    "tool output"
                );
            }
            SessionEventPayload::HookStarted { run } => {
                info!(
                    stage = label,
                    hook = %run.event_name,
                    plugin = %run.plugin_id,
                    handler = ?run.handler_type,
                    "hook started"
                );
            }
            SessionEventPayload::HookCompleted { run } => {
                info!(
                    stage = label,
                    hook = %run.event_name,
                    plugin = %run.plugin_id,
                    status = ?run.status,
                    duration_ms = ?run.duration_ms,
                    entries = run.entries.len(),
                    message = ?run.status_message,
                    "hook completed"
                );
            }
            SessionEventPayload::ToolExecutionCompleted { outcome } => {
                let tool = outcome.call.name;
                let call_id = outcome.call.id;
                match outcome.result {
                    Ok(result) => {
                        info!(
                            stage = label,
                            tool = %tool,
                            call_id = %call_id,
                            result_kind = tool_result_kind(&result),
                            result_bytes = tool_result_size(&result),
                            "tool completed"
                        );
                    }
                    Err(error) => {
                        warn!(
                            stage = label,
                            tool = %tool,
                            call_id = %call_id,
                            error = %error,
                            "tool failed"
                        );
                    }
                }
            }
            SessionEventPayload::ApprovalRequested { tool_name, reason } => {
                warn!(
                    stage = label,
                    tool = %tool_name,
                    reason = %reason,
                    "tool approval requested"
                );
            }
            SessionEventPayload::ContextCompacted { summary } => {
                info!(
                    stage = label,
                    summary_bytes = summary.len(),
                    "context compacted"
                );
            }
            SessionEventPayload::TurnCompleted {
                turn_id,
                usage: turn_usage,
            } => {
                usage = turn_usage;
                info!(
                    stage = label,
                    turn_id = %turn_id,
                    input_tokens = usage.input_tokens,
                    output_tokens = usage.output_tokens,
                    cache_creation_input_tokens = usage.cache_creation_input_tokens,
                    cache_read_input_tokens = usage.cache_read_input_tokens,
                    "agent turn completed"
                );
                break;
            }
            SessionEventPayload::TurnFailed {
                turn_id,
                error,
                cancelled,
                retryable,
                ..
            } => {
                warn!(
                    stage = label,
                    turn_id = %turn_id,
                    cancelled,
                    retryable,
                    error = %error,
                    "agent turn failed"
                );
                let _ = session.shutdown("turn_failed").await;
                return Err(anyhow::Error::new(AgentStageTurnFailure {
                    label: label.to_owned(),
                    error,
                    retryable,
                    cancelled,
                }));
            }
            SessionEventPayload::Lagged { dropped_events } => {
                warn!(stage = label, dropped_events, "agent event stream lagged");
            }
            SessionEventPayload::SessionShutdownComplete => {
                info!(stage = label, "agent session shutdown complete");
            }
            _ => {}
        }
    }

    info!(stage = label, "shutting down agent session");
    session.shutdown(label).await?;
    let run = agent_run_from_text(label, latest_text, delta_text, text_requirement)?;
    if text_requirement == AgentTextRequirement::Optional && run.text.is_empty() {
        info!(
            stage = label,
            "agent turn produced no assistant text; continuing because text is optional"
        );
    }
    info!(
        stage = label,
        input_tokens = usage.input_tokens,
        output_tokens = usage.output_tokens,
        "completed agent turn"
    );
    Ok(run)
}

fn agent_run_from_text(
    label: &str,
    latest_text: Option<String>,
    delta_text: String,
    text_requirement: AgentTextRequirement,
) -> anyhow::Result<AgentRun> {
    if let Some(text) = latest_text
        .filter(|text| !text.trim().is_empty())
        .or_else(|| (!delta_text.trim().is_empty()).then_some(delta_text))
    {
        return Ok(AgentRun { text });
    }

    match text_requirement {
        AgentTextRequirement::Required => {
            bail!("agent stage {label} produced no assistant text");
        }
        AgentTextRequirement::Optional => Ok(AgentRun {
            text: String::new(),
        }),
    }
}

async fn propose_issue_candidates(
    glm: &Halter,
    worktree: &Path,
    repo: &RepoSlug,
    corpus: &str,
    project_system_prompt: Option<&str>,
) -> anyhow::Result<CandidateSet> {
    let prompt = format!(
        r#"You are triaging open GitHub issues for {repo}.

Read the entire issue corpus and identify up to 3 excellent groups of open issues that can be solved holistically with one elegant pull request. Do not design the implementation. Prefer groups that:
- clearly share one root cause or cohesive code path
- can be solved without more maintainer input
- are likely contained enough for one PR
- avoid speculative feature work

If there are not enough good groups, choose individual open issues that are straightforward, contained, and do not need maintainer input.

{JSON_ONLY_OUTPUT_RULE} Use this shape:
{{
  "candidates": [
    {{
      "title": "short candidate name",
      "issue_numbers": [123],
      "rationale": "why this is cohesive and suitable",
      "maintainer_input_risk": "why this does not need maintainer input"
    }}
  ]
}}

ISSUE CORPUS:
{corpus}
"#
    );
    let raw = run_agent_with_system_prompt(
        glm,
        worktree,
        "glm issue grouping",
        prompt,
        project_system_prompt,
    )
    .await?
    .text;
    let candidates: CandidateSet = parse_json_response(&raw).map_err(anyhow::Error::msg)?;
    if candidates.candidates.is_empty() {
        bail!("GLM did not return any candidate issue groups");
    }
    Ok(candidates)
}

async fn judge_issue_plan(
    judge: &Halter,
    worktree: &Path,
    repo: &RepoSlug,
    corpus: &str,
    candidates: &CandidateSet,
    implementation_plan_path: &str,
    project_system_prompt: Option<&str>,
) -> anyhow::Result<JudgeSelection> {
    let candidates_json = serde_json::to_string_pretty(candidates)?;
    let prompt = format!(
        r#"You are a model-judge planning group for a software factory workflow targeting {repo}.

The full open-issue corpus is provided below (each issue body is capped at roughly 1k tokens). A prior model also proposed candidate issue groups as a first pass. You have everything you need to decide immediately. Work through these steps in order:

1. Group alike issues. Use the corpus and the candidate proposals to cluster issues that share one root cause or cohesive code path; refine or discard the proposals as needed.
2. Narrow the groups and individual issues down to the strongest, most contained candidates.
3. Agree on a shortlist of at most 3 groups/issues worth working on.
4. Select exactly one — the smallest cohesive PR you have high confidence in — and design its implementation plan.

Selection rules:
- prefer the smallest cohesive PR with high confidence
- select only issues whose corpus state is open
- reject any candidate that needs maintainer clarification
- only maintainer comments are included; non-maintainer comments are intentionally omitted
- corpus bodies are truncated, so use the `github_issue` tool to fetch the complete untruncated text of any issue before you commit to selecting it

The implementation plan you write must:
- be saved as markdown to {implementation_plan_path}
- include selected issue numbers, scope, concrete files/modules to inspect, step-by-step changes, happy-path and sad-path tests, verification commands, and risks
- not contain code

Output protocol — follow in this exact order:
1. Write the full plan to {implementation_plan_path} using the write tool.
2. Confirm the file is non-empty.
3. Only then, as your FINAL message, return ONLY the JSON object below — no markdown code fences, no surrounding prose, and no plan text (the plan belongs only in the file).

JSON shape:
{{
  "title": "PR-sized implementation title",
  "issue_numbers": [123],
  "notes": "judge rationale and constraints"
}}

CANDIDATE GROUPINGS (first-pass proposals to refine):
{candidates_json}

OPEN ISSUE CORPUS:
{corpus}
"#
    );
    let raw = run_agent_with_system_prompt(
        judge,
        worktree,
        "model judge planning",
        prompt,
        project_system_prompt,
    )
    .await?
    .text;
    parse_json_response(&raw).map_err(anyhow::Error::msg)
}

async fn read_implementation_plan(path: &Path) -> anyhow::Result<String> {
    let plan = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("failed to read implementation plan {}", path.display()))?;
    if plan.trim().is_empty() {
        bail!(
            "failed to read implementation plan: {} is empty",
            path.display()
        );
    }
    Ok(plan)
}

async fn prepare_branch(
    worktree: &Path,
    base_branch: &str,
    requested_branch: Option<&str>,
    allow_dirty: bool,
    repo: &RepoSlug,
    selection: &JudgeSelection,
    excluded_dirty_paths: &[&str],
    run_id: &str,
) -> anyhow::Result<String> {
    info!(
        base_branch,
        requested_branch = ?requested_branch,
        allow_dirty,
        "preparing factory branch"
    );
    if !allow_dirty && git_is_dirty(worktree, excluded_dirty_paths).await? {
        bail!("failed to prepare branch: worktree is dirty; commit/stash or pass --allow-dirty");
    }
    run_cmd(worktree, "git", &["fetch", "origin", base_branch]).await?;
    let branch = requested_branch
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| branch_name(repo, &selection.title, run_id));
    let base_ref = format!("origin/{base_branch}");
    info!(
        branch = %branch,
        base_ref = %base_ref,
        title = %selection.title,
        repo = %repo,
        "checking out factory branch"
    );
    run_cmd(worktree, "git", &["checkout", "-b", &branch, &base_ref]).await?;
    Ok(branch)
}

async fn implement_plan(
    implementer: &Halter,
    worktree: &Path,
    repo: &RepoSlug,
    selection: &JudgeSelection,
    implementation_plan: &str,
    issues: &[IssueDoc],
    project_system_prompt: Option<&str>,
) -> anyhow::Result<()> {
    let selected = selected_issue_details(selection, issues);
    let prompt = format!(
        r#"You are the implementation agent for a Halter software factory run in repo {repo}.

Implement the plan below in the current local worktree.

Rules:
- Do not create, switch, commit, push, or open branches/PRs. The orchestrator owns git branch and PR operations.
- Keep the diff scoped to the selected issues.
- Add or update tests for happy paths and sad paths described in the plan.
- Run the narrowest meaningful verification commands.
- {CARGO_TIMEOUT_RULE}
- If the plan proves impossible without maintainer input, stop and explain exactly why.

SELECTED ISSUES:
{selected}

IMPLEMENTATION PLAN:
"#,
    );
    let prompt = format!("{prompt}{implementation_plan}\n");
    run_coding_action_with_system_prompt(
        implementer,
        worktree,
        "kimi implementation",
        prompt,
        project_system_prompt,
    )
    .await?;
    Ok(())
}

fn selected_issue_details(selection: &JudgeSelection, issues: &[IssueDoc]) -> String {
    let selected: HashSet<u64> = selected_issue_numbers(selection).into_iter().collect();
    let selected_issues = issues
        .iter()
        .filter(|issue| selected.contains(&issue.number))
        .cloned()
        .collect::<Vec<_>>();
    let repo = RepoSlug {
        owner: "selected".to_owned(),
        name: "issues".to_owned(),
    };
    issue_corpus(&repo, &selected_issues, None)
}

fn ensure_selected_issues_are_open(
    issues: &[IssueDoc],
    issue_numbers: &[u64],
) -> anyhow::Result<()> {
    let unknown = issue_numbers
        .iter()
        .filter(|number| !issues.iter().any(|issue| issue.number == **number))
        .map(|number| format!("#{number}"))
        .collect::<Vec<_>>();
    if !unknown.is_empty() {
        bail!(
            "failed to select work: judge selected issue(s) not present in the corpus: {}",
            unknown.join(", ")
        );
    }

    let closed = issue_numbers
        .iter()
        .filter_map(|number| {
            issues
                .iter()
                .find(|issue| issue.number == *number)
                .filter(|issue| issue.state != "open")
                .map(|issue| format!("#{} ({})", issue.number, issue.state))
        })
        .collect::<Vec<_>>();
    if !closed.is_empty() {
        bail!(
            "failed to select work: judge selected non-open issue(s): {}",
            closed.join(", ")
        );
    }
    Ok(())
}

async fn run_review_loop(
    implementer: &Halter,
    reviewer: &Halter,
    worktree: &Path,
    base_ref: &str,
    implementation_plan: &str,
    max_iterations: usize,
    project_system_prompt: Option<&str>,
) -> anyhow::Result<CodeReview> {
    for iteration in 1..=max_iterations {
        let diff = branch_diff(worktree, base_ref).await?;
        if diff.trim().is_empty() {
            bail!("review loop cannot continue: branch diff is empty");
        }
        info!(
            iteration,
            max_iterations,
            diff_bytes = diff.len(),
            "starting review iteration"
        );
        let review = review_diff(
            reviewer,
            worktree,
            base_ref,
            &diff,
            ReviewIteration {
                current: iteration,
                max: max_iterations,
            },
            project_system_prompt,
        )
        .await?;
        if review.clean && review.findings.is_empty() {
            info!(iteration, "review loop is clean");
            return Ok(review);
        }
        warn!(
            iteration,
            findings = review.findings.len(),
            "review requested changes"
        );
        let review_json = serde_json::to_string_pretty(&review)?;
        let prompt = review_repair_prompt(
            implementation_plan,
            &review_json,
            ReviewIteration {
                current: iteration,
                max: max_iterations,
            },
        );
        run_coding_action_with_system_prompt(
            implementer,
            worktree,
            "kimi review repair",
            prompt,
            project_system_prompt,
        )
        .await?;
    }
    bail!("review loop exhausted after {max_iterations} iterations without a clean review")
}

async fn review_diff(
    reviewer: &Halter,
    worktree: &Path,
    base_ref: &str,
    diff: &str,
    iteration: ReviewIteration,
    project_system_prompt: Option<&str>,
) -> anyhow::Result<CodeReview> {
    let prompt = code_review_prompt(base_ref, diff, iteration);
    let raw = run_code_review_agent_with_system_prompt(
        reviewer,
        worktree,
        "gpt code review",
        prompt,
        project_system_prompt,
    )
    .await?
    .text;
    parse_json_response(&raw).map_err(anyhow::Error::msg)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ReviewIteration {
    current: usize,
    max: usize,
}

impl ReviewIteration {
    fn is_first(self) -> bool {
        self.current == 1
    }

    fn is_final(self) -> bool {
        self.max > 0 && self.current == self.max
    }
}

fn code_review_prompt(base_ref: &str, diff: &str, iteration: ReviewIteration) -> String {
    let intro = if iteration.is_first() {
        format!("You are reviewing a branch diff against {base_ref}.")
    } else {
        format!(
            "Your previous code review has been addressed. Thoroughly re-review the branch diff against {base_ref} and ensure all findings have been addressed and there are no new ones."
        )
    };
    let final_instruction = final_review_iteration_instruction(iteration, "review");
    format!(
        r#"{intro}

Review stance:
- Prioritize correctness bugs, regressions, missing tests, unsafe behavior, and broken edge cases.
- Include but do not block on style nits unless they create real maintenance risk.
- Mark clean=true only when there are no required fixes.
- {CARGO_TIMEOUT_RULE}
{final_instruction}
{JSON_ONLY_OUTPUT_RULE} Use this shape:
{{
  "clean": false,
  "summary": "short review summary",
  "findings": [
    {{
      "severity": "high|medium|low",
      "file": "path/to/file",
      "line": 123,
      "problem": "specific problem",
      "recommendation": "specific fix"
    }}
  ]
}}

BRANCH DIFF:
{diff}
"#
    )
}

fn review_repair_prompt(
    implementation_plan: &str,
    review_json: &str,
    iteration: ReviewIteration,
) -> String {
    let final_instruction = final_review_iteration_instruction(iteration, "implementation");
    format!(
        r#"The code review for the current branch found issues. Fix every finding in the current worktree.

Rules:
- Do not create, switch, commit, push, or open branches/PRs.
- Keep the fix scoped to the implementation plan and review findings.
- Run focused tests or checks that cover the changed behavior.
- {CARGO_TIMEOUT_RULE}
{final_instruction}- Return a concise summary and the verification commands you ran.

IMPLEMENTATION PLAN:
{implementation_plan}

REVIEW RESULT:
{review_json}
"#
    )
}

fn final_review_iteration_instruction(iteration: ReviewIteration, participant: &str) -> String {
    if !iteration.is_final() {
        return String::new();
    }
    match participant {
        "review" => "- This is the last code review iteration. Ensure every previous finding has been addressed and no new required fixes remain. If there are repeated issues or miscommunications, think deeply and take a different review approach before returning findings.\n\n".to_owned(),
        "implementation" => "- This is the last review-repair iteration. Ensure every finding is fully addressed. If there are repeated issues or miscommunications, think deeply and take a different implementation approach before finishing.\n".to_owned(),
        _ => String::new(),
    }
}

async fn draft_pr(
    pr_writer: &Halter,
    worktree: &Path,
    repo: &RepoSlug,
    selection: &JudgeSelection,
    implementation_plan: &str,
    issue_numbers: &[u64],
    diff: &str,
    project_system_prompt: Option<&str>,
) -> anyhow::Result<PullRequestDraft> {
    let closing_lines = issue_numbers
        .iter()
        .map(|number| format!("Closes #{number}"))
        .collect::<Vec<_>>()
        .join("\n");
    let prompt = format!(
        r#"Create a detailed GitHub pull request for {repo}.

Return ONLY the JSON object as your final message — no surrounding prose and not wrapped in a code fence (the body value itself may contain markdown). Use this shape:
{{
  "title": "concise PR title",
  "body": "detailed PR body in markdown"
}}

The body must include:
- summary of user-facing behavior
- implementation notes
- tests/verification section
- these exact issue closing references:
{closing_lines}

SELECTED WORK:
{}

IMPLEMENTATION PLAN:
{implementation_plan}

FINAL BRANCH DIFF:
{diff}
"#,
        serde_json::to_string_pretty(selection)?
    );
    let raw = run_agent_with_system_prompt(
        pr_writer,
        worktree,
        "gemma pr draft",
        prompt,
        project_system_prompt,
    )
    .await?
    .text;
    parse_json_response(&raw).map_err(anyhow::Error::msg)
}

struct MonitorContext<'a> {
    github: &'a GitHubClient,
    glm: &'a Halter,
    implementer: &'a Halter,
    reviewer: &'a Halter,
    worktree: &'a Path,
    repo: &'a RepoSlug,
    pr_number: u64,
    branch: &'a str,
    base_ref: &'a str,
    selection: &'a JudgeSelection,
    implementation_plan: &'a str,
    project_system_prompt: Option<&'a str>,
    excluded_commit_paths: &'a [&'a str],
    max_review_iterations: usize,
    poll_seconds: u64,
}

async fn monitor_pr(ctx: MonitorContext<'_>) -> anyhow::Result<()> {
    let mut seen = ctx
        .github
        .initial_seen_pr_activity(ctx.repo, ctx.pr_number)
        .await?;
    info!(
        pr_number = ctx.pr_number,
        seen_activity = seen.len(),
        poll_seconds = ctx.poll_seconds,
        "starting PR monitor"
    );
    loop {
        info!(pr_number = ctx.pr_number, "polling PR state");
        let pr = ctx
            .github
            .fetch_pull_request(ctx.repo, ctx.pr_number)
            .await?;
        if pr.merged.unwrap_or(false) {
            info!(pr_number = ctx.pr_number, url = %pr.html_url, "PR merged");
            println!("PR merged: {}", pr.html_url);
            return Ok(());
        }
        if pr.state != "open" {
            bail!(
                "monitor stopped: PR #{} is {} but not merged",
                ctx.pr_number,
                pr.state
            );
        }

        let action = ctx
            .github
            .fetch_new_pr_activity(ctx.repo, ctx.pr_number, &mut seen)
            .await?;
        if action.is_empty() {
            info!(
                pr_number = ctx.pr_number,
                poll_seconds = ctx.poll_seconds,
                "no new PR activity"
            );
            tokio::time::sleep(Duration::from_secs(ctx.poll_seconds)).await;
            continue;
        }
        info!(
            pr_number = ctx.pr_number,
            review_feedback = action.code_review_feedback.len(),
            plsfix_comments = action.plsfix_comments.len(),
            "new PR activity"
        );

        if !action.code_review_feedback.is_empty() {
            let feedback = action.code_review_feedback.join("\n\n---\n\n");
            apply_feedback(
                ctx.implementer,
                ctx.reviewer,
                ctx.worktree,
                ctx.base_ref,
                ctx.implementation_plan,
                &format!("Address this GitHub code review feedback:\n\n{feedback}"),
                ctx.max_review_iterations,
                ctx.project_system_prompt,
            )
            .await?;
            commit_if_dirty(
                ctx.worktree,
                "Address PR code review feedback",
                ctx.excluded_commit_paths,
            )
            .await?;
            run_cmd(ctx.worktree, "git", &["push", "origin", ctx.branch]).await?;
        }

        if !action.plsfix_comments.is_empty() {
            let comments = action.plsfix_comments.join("\n\n---\n\n");
            let instruction = refine_plsfix_comments(
                ctx.glm,
                ctx.worktree,
                ctx.selection,
                ctx.implementation_plan,
                &comments,
                ctx.project_system_prompt,
            )
            .await?;
            apply_feedback(
                ctx.implementer,
                ctx.reviewer,
                ctx.worktree,
                ctx.base_ref,
                ctx.implementation_plan,
                &instruction,
                ctx.max_review_iterations,
                ctx.project_system_prompt,
            )
            .await?;
            commit_if_dirty(
                ctx.worktree,
                "Address /plsfix PR feedback",
                ctx.excluded_commit_paths,
            )
            .await?;
            run_cmd(ctx.worktree, "git", &["push", "origin", ctx.branch]).await?;
        }

        tokio::time::sleep(Duration::from_secs(ctx.poll_seconds)).await;
    }
}

async fn refine_plsfix_comments(
    glm: &Halter,
    worktree: &Path,
    selection: &JudgeSelection,
    implementation_plan: &str,
    comments: &str,
    project_system_prompt: Option<&str>,
) -> anyhow::Result<String> {
    let prompt = format!(
        r#"A maintainer left /plsfix comments on the PR. Convert them into a precise implementation instruction for the coding agent.

Do not design a full new plan. Preserve the maintainer's requested fix, call out any ambiguity, and keep the instruction scoped to the selected issues.

SELECTED WORK:
{}

IMPLEMENTATION PLAN:
{implementation_plan}

/plsfix COMMENTS:
{comments}
"#,
        serde_json::to_string_pretty(selection)?
    );
    Ok(run_agent_with_system_prompt(
        glm,
        worktree,
        "glm plsfix refinement",
        prompt,
        project_system_prompt,
    )
    .await?
    .text)
}

async fn apply_feedback(
    implementer: &Halter,
    reviewer: &Halter,
    worktree: &Path,
    base_ref: &str,
    implementation_plan: &str,
    feedback: &str,
    max_review_iterations: usize,
    project_system_prompt: Option<&str>,
) -> anyhow::Result<()> {
    let prompt = format!(
        r#"Apply this PR feedback in the current worktree.

Rules:
- Do not create, switch, commit, push, or open branches/PRs.
- Keep changes scoped to the selected issues and feedback.
- Run focused verification for the changed behavior.
- {CARGO_TIMEOUT_RULE}

IMPLEMENTATION PLAN:
{implementation_plan}

FEEDBACK:
{feedback}
"#
    );
    run_coding_action_with_system_prompt(
        implementer,
        worktree,
        "kimi pr feedback",
        prompt,
        project_system_prompt,
    )
    .await?;
    run_review_loop(
        implementer,
        reviewer,
        worktree,
        base_ref,
        implementation_plan,
        max_review_iterations,
        project_system_prompt,
    )
    .await?;
    Ok(())
}

async fn git_is_dirty(worktree: &Path, excluded_paths: &[&str]) -> anyhow::Result<bool> {
    let status = run_cmd(worktree, "git", &["status", "--porcelain"]).await?;
    Ok(dirty_status_excluding(&status, excluded_paths))
}

async fn current_branch(worktree: &Path) -> anyhow::Result<String> {
    Ok(run_cmd(worktree, "git", &["branch", "--show-current"])
        .await?
        .trim()
        .to_owned())
}

async fn checkout_branch(worktree: &Path, branch: &str) -> anyhow::Result<()> {
    if branch.trim().is_empty() {
        bail!("failed to resume branch: checkpoint branch is empty");
    }
    let current = current_branch(worktree).await?;
    if current == branch {
        return Ok(());
    }
    run_cmd(worktree, "git", &["checkout", branch]).await?;
    Ok(())
}

async fn current_commit(worktree: &Path) -> anyhow::Result<String> {
    Ok(run_cmd(worktree, "git", &["rev-parse", "HEAD"])
        .await?
        .trim()
        .to_owned())
}

async fn branch_has_diff(worktree: &Path, base_ref: &str) -> anyhow::Result<bool> {
    Ok(!branch_diff(worktree, base_ref).await?.trim().is_empty())
}

async fn branch_diff(worktree: &Path, base_ref: &str) -> anyhow::Result<String> {
    run_cmd(worktree, "git", &["diff", "--find-renames", base_ref]).await
}

async fn commit_if_dirty(
    worktree: &Path,
    message: &str,
    excluded_paths: &[&str],
) -> anyhow::Result<bool> {
    if !git_is_dirty(worktree, excluded_paths).await? {
        info!(message, excluded_paths = ?excluded_paths, "worktree is clean");
        return Ok(false);
    }
    info!(message, excluded_paths = ?excluded_paths, "staging worktree changes");
    run_cmd(worktree, "git", &["add", "-A"]).await?;
    for path in excluded_paths {
        run_cmd(worktree, "git", &["reset", "--", path]).await?;
    }
    if run_cmd(worktree, "git", &["diff", "--cached", "--quiet"])
        .await
        .is_ok()
    {
        info!(message, "no staged changes to commit");
        return Ok(false);
    }
    info!(message, "committing worktree changes");
    run_cmd(worktree, "git", &["commit", "-m", message]).await?;
    Ok(true)
}

async fn run_cmd(worktree: &Path, program: &str, args: &[&str]) -> anyhow::Result<String> {
    let command = args.join(" ");
    debug!(
        cwd = %worktree.display(),
        program,
        args = %command,
        "running command"
    );
    let output = Command::new(program)
        .args(args)
        .current_dir(worktree)
        .output()
        .await
        .with_context(|| format!("failed to run command: {program} {command}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        warn!(
            cwd = %worktree.display(),
            program,
            args = %command,
            status = %output.status,
            stdout_bytes = output.stdout.len(),
            stderr_bytes = output.stderr.len(),
            stderr = %single_line_preview(stderr.trim(), 500),
            "command failed"
        );
        bail!(
            "command failed: {program} {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
            command
        );
    }
    debug!(
        cwd = %worktree.display(),
        program,
        args = %command,
        status = %output.status,
        stdout_bytes = output.stdout.len(),
        stderr_bytes = output.stderr.len(),
        "command completed"
    );
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

#[derive(Clone)]
struct GitHubClient {
    client: reqwest::Client,
}

impl GitHubClient {
    async fn from_local_credentials(worktree: &Path) -> anyhow::Result<Self> {
        let token = match github_token_from_env() {
            Some(token) => {
                info!("using GitHub token from environment");
                Some(token)
            }
            None => github_token_from_gh_cli(worktree).await,
        };
        let authenticated = token.is_some();
        let mut headers = HeaderMap::new();
        headers.insert(
            USER_AGENT,
            HeaderValue::from_static("halter-software-factory-example"),
        );
        headers.insert(
            ACCEPT,
            HeaderValue::from_static("application/vnd.github+json"),
        );
        headers.insert(
            HeaderName::from_static("x-github-api-version"),
            HeaderValue::from_static("2022-11-28"),
        );
        if let Some(token) = token {
            headers.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {token}"))
                    .context("failed to build GitHub authorization header")?,
            );
        }
        let client = reqwest::Client::builder()
            .default_headers(headers)
            .build()
            .context("failed to build GitHub client")?;
        info!(authenticated, "built GitHub client");
        Ok(Self { client })
    }

    async fn get<T: for<'de> Deserialize<'de>>(&self, url: &str) -> anyhow::Result<T> {
        debug!(method = "GET", url, "GitHub request");
        let response = self.client.get(url).send().await?;
        decode_response(response, "GET", url).await
    }

    async fn post_json<T: for<'de> Deserialize<'de>, B: Serialize>(
        &self,
        url: &str,
        body: &B,
    ) -> anyhow::Result<T> {
        info!(method = "POST", url, "GitHub request");
        let response = self.client.post(url).json(body).send().await?;
        decode_response(response, "POST", url).await
    }

    async fn fetch_repo(&self, repo: &RepoSlug) -> anyhow::Result<GitHubRepo> {
        info!(repo = %repo, "fetching GitHub repository metadata");
        self.get(&repo.api_base()).await
    }

    async fn fetch_recent_open_issues(
        &self,
        repo: &RepoSlug,
        limit: usize,
    ) -> anyhow::Result<Vec<IssueDoc>> {
        let limit = validate_recent_issue_limit(limit).map_err(anyhow::Error::msg)?;
        info!(repo = %repo, limit, "searching recent open GitHub issues");
        let mut url = reqwest::Url::parse("https://api.github.com/search/issues")
            .context("failed to build GitHub issue search URL")?;
        url.query_pairs_mut()
            .append_pair("q", &format!("repo:{repo} is:issue is:open"))
            .append_pair("sort", "created")
            .append_pair("order", "desc")
            .append_pair("per_page", &limit.to_string())
            .append_pair("page", "1");
        let search: GitHubIssueSearchResponse = self.get(url.as_str()).await?;
        info!(
            repo = %repo,
            search_items = search.items.len(),
            "GitHub issue search returned"
        );
        let mut docs = Vec::with_capacity(search.items.len());
        for issue in search
            .items
            .into_iter()
            .filter(|issue| issue.pull_request.is_none())
            .filter(|issue| issue.state == "open")
        {
            info!(repo = %repo, issue = issue.number, "fetching issue comments");
            let comments = self.fetch_issue_comments(repo, issue.number).await?;
            docs.push(issue.into_doc(comments));
        }
        info!(repo = %repo, issue_count = docs.len(), "loaded open issue docs");
        Ok(docs)
    }

    async fn fetch_open_issue(&self, repo: &RepoSlug, number: u64) -> anyhow::Result<IssueDoc> {
        info!(repo = %repo, issue = number, "fetching open GitHub issue");
        let url = format!("{}/issues/{number}", repo.api_base());
        let issue: GitHubIssue = self.get(&url).await?;
        if issue.pull_request.is_some() {
            bail!("failed to fetch issue #{number}: GitHub item is a pull request");
        }
        if issue.state != "open" {
            bail!(
                "failed to fetch issue #{}: issue is {}",
                issue.number,
                issue.state
            );
        }
        let comments = self.fetch_issue_comments(repo, issue.number).await?;
        Ok(issue.into_doc(comments))
    }

    async fn fetch_issue_comments(
        &self,
        repo: &RepoSlug,
        issue_number: u64,
    ) -> anyhow::Result<Vec<IssueComment>> {
        let mut page = 1;
        let mut comments = Vec::new();
        loop {
            debug!(
                repo = %repo,
                issue = issue_number,
                page,
                "fetching issue comments page"
            );
            let url = format!(
                "{}/issues/{issue_number}/comments?per_page=100&page={page}",
                repo.api_base()
            );
            let batch: Vec<GitHubIssueComment> = self.get(&url).await?;
            let count = batch.len();
            comments.extend(
                batch
                    .into_iter()
                    .filter(|comment| is_maintainer_author_association(&comment.author_association))
                    .map(GitHubIssueComment::into_issue_comment),
            );
            if count < 100 {
                break;
            }
            page += 1;
        }
        info!(
            repo = %repo,
            issue = issue_number,
            maintainer_comments = comments.len(),
            "loaded issue comments"
        );
        Ok(comments)
    }

    async fn create_pull_request(
        &self,
        repo: &RepoSlug,
        branch: &str,
        base: &str,
        draft: &PullRequestDraft,
    ) -> anyhow::Result<GitHubPullRequest> {
        info!(
            repo = %repo,
            branch,
            base,
            title = %draft.title,
            "creating GitHub pull request"
        );
        let url = format!("{}/pulls", repo.api_base());
        let request = CreatePullRequest {
            title: draft.title.clone(),
            head: branch.to_owned(),
            base: base.to_owned(),
            body: draft.body.clone(),
        };
        self.post_json(&url, &request).await
    }

    async fn fetch_pull_request(
        &self,
        repo: &RepoSlug,
        number: u64,
    ) -> anyhow::Result<GitHubPullRequest> {
        debug!(repo = %repo, pr_number = number, "fetching pull request");
        self.get(&format!("{}/pulls/{number}", repo.api_base()))
            .await
    }

    async fn initial_seen_pr_activity(
        &self,
        repo: &RepoSlug,
        pr_number: u64,
    ) -> anyhow::Result<HashSet<String>> {
        let mut seen = HashSet::new();
        let _ = self
            .fetch_new_pr_activity(repo, pr_number, &mut seen)
            .await?;
        Ok(seen)
    }

    async fn fetch_new_pr_activity(
        &self,
        repo: &RepoSlug,
        pr_number: u64,
        seen: &mut HashSet<String>,
    ) -> anyhow::Result<MonitorAction> {
        debug!(repo = %repo, pr_number, "fetching new PR activity");
        let issue_comments: Vec<GitHubIssueComment> = self
            .get_paginated(&format!("{}/issues/{pr_number}/comments", repo.api_base()))
            .await?;
        let reviews: Vec<GitHubReview> = self
            .get_paginated(&format!("{}/pulls/{pr_number}/reviews", repo.api_base()))
            .await?;
        let review_comments: Vec<GitHubReviewComment> = self
            .get_paginated(&format!("{}/pulls/{pr_number}/comments", repo.api_base()))
            .await?;

        let mut review_feedback = Vec::new();
        let mut issue_comment_bodies = Vec::new();

        for comment in issue_comments {
            let key = format!("issue-comment:{}", comment.id);
            if seen.insert(key) && is_maintainer_author_association(&comment.author_association) {
                issue_comment_bodies.push(comment.body.unwrap_or_default());
            }
        }
        for review in reviews {
            let key = format!("review:{}", review.id);
            if seen.insert(key) && is_maintainer_author_association(&review.author_association) {
                if let Some(body) = review.body.filter(|body| !body.trim().is_empty()) {
                    review_feedback.push(format!(
                        "Review {} by {}:\n{}",
                        review.state,
                        review.user.map(|user| user.login).unwrap_or_default(),
                        body
                    ));
                }
            }
        }
        for comment in review_comments {
            let key = format!("review-comment:{}", comment.id);
            if seen.insert(key) && is_maintainer_author_association(&comment.author_association) {
                review_feedback.push(format!(
                    "Review comment on {}:{} by {}:\n{}",
                    comment.path,
                    comment.line.unwrap_or_default(),
                    comment.user.login,
                    comment.body
                ));
            }
        }

        let action = monitor_action(review_feedback, issue_comment_bodies);
        debug!(
            repo = %repo,
            pr_number,
            review_feedback = action.code_review_feedback.len(),
            plsfix_comments = action.plsfix_comments.len(),
            "classified PR activity"
        );
        Ok(action)
    }

    async fn get_paginated<T: for<'de> Deserialize<'de>>(
        &self,
        base_url: &str,
    ) -> anyhow::Result<Vec<T>> {
        let mut page = 1;
        let mut all = Vec::new();
        loop {
            let separator = if base_url.contains('?') { '&' } else { '?' };
            let url = format!("{base_url}{separator}per_page=100&page={page}");
            debug!(url = %url, page, "fetching GitHub page");
            let batch: Vec<T> = self.get(&url).await?;
            let count = batch.len();
            debug!(url = %url, page, count, "loaded GitHub page");
            all.extend(batch);
            if count < 100 {
                break;
            }
            page += 1;
        }
        Ok(all)
    }
}

fn github_token_from_env() -> Option<String> {
    std::env::var("GITHUB_TOKEN")
        .ok()
        .or_else(|| std::env::var("GH_TOKEN").ok())
        .map(|token| token.trim().to_owned())
        .filter(|token| !token.is_empty())
}

async fn github_token_from_gh_cli(worktree: &Path) -> Option<String> {
    let output = Command::new("gh")
        .args(["auth", "token"])
        .current_dir(worktree)
        .output()
        .await;
    match output {
        Ok(output) if output.status.success() => {
            let token = String::from_utf8_lossy(&output.stdout).trim().to_owned();
            if token.is_empty() {
                warn!("gh auth token returned an empty token; continuing without GitHub auth");
                None
            } else {
                info!("using GitHub token from gh auth token");
                Some(token)
            }
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            warn!(
                status = %output.status,
                stderr = %stderr.trim(),
                "failed to read GitHub token from gh; continuing without GitHub auth"
            );
            None
        }
        Err(error) => {
            warn!(
                error = %error,
                "failed to execute gh auth token; continuing without GitHub auth"
            );
            None
        }
    }
}

async fn decode_response<T: for<'de> Deserialize<'de>>(
    response: reqwest::Response,
    method: &str,
    url: &str,
) -> anyhow::Result<T> {
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        warn!(
            method,
            url,
            status = %status,
            body = %single_line_preview(&body, 500),
            "GitHub API request failed"
        );
        bail!("GitHub API request failed: {method} {url} returned {status}: {body}");
    }
    debug!(method, url, status = %status, "GitHub response");
    response
        .json::<T>()
        .await
        .with_context(|| format!("failed to decode GitHub response for {method} {url}"))
}

#[derive(Debug, Deserialize)]
struct GitHubRepo {
    default_branch: String,
}

#[derive(Debug, Deserialize)]
struct GitHubUser {
    login: String,
}

#[derive(Debug, Deserialize)]
struct GitHubLabel {
    name: String,
}

#[derive(Debug, Deserialize)]
struct GitHubIssueSearchResponse {
    items: Vec<GitHubIssue>,
}

#[derive(Debug, Deserialize)]
struct GitHubIssue {
    number: u64,
    state: String,
    title: String,
    body: Option<String>,
    labels: Vec<GitHubLabel>,
    user: GitHubUser,
    html_url: String,
    pull_request: Option<serde_json::Value>,
}

impl GitHubIssue {
    fn into_doc(self, comments: Vec<IssueComment>) -> IssueDoc {
        IssueDoc {
            number: self.number,
            state: self.state,
            title: self.title,
            body: self.body.unwrap_or_default(),
            labels: self.labels.into_iter().map(|label| label.name).collect(),
            author: self.user.login,
            url: self.html_url,
            comments,
        }
    }
}

#[derive(Debug, Deserialize)]
struct GitHubIssueComment {
    id: u64,
    body: Option<String>,
    user: GitHubUser,
    created_at: String,
    #[serde(default)]
    author_association: String,
}

impl GitHubIssueComment {
    fn into_issue_comment(self) -> IssueComment {
        IssueComment {
            author: self.user.login,
            body: self.body.unwrap_or_default(),
            created_at: self.created_at,
        }
    }
}

#[derive(Debug, Deserialize)]
struct GitHubPullRequest {
    number: u64,
    html_url: String,
    state: String,
    merged: Option<bool>,
}

#[derive(Debug, Serialize)]
struct CreatePullRequest {
    title: String,
    head: String,
    base: String,
    body: String,
}

#[derive(Debug, Deserialize)]
struct GitHubReview {
    id: u64,
    state: String,
    body: Option<String>,
    #[serde(default)]
    author_association: String,
    user: Option<GitHubUser>,
}

#[derive(Debug, Deserialize)]
struct GitHubReviewComment {
    id: u64,
    body: String,
    path: String,
    line: Option<u64>,
    #[serde(default)]
    author_association: String,
    user: GitHubUser,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_factory_config_matches_judge_example_tool_and_shell_lists() {
        let config = default_factory_config();
        assert_eq!(config.tools.enabled, judge_example_tools());
        assert_eq!(config.policy.shell.allow, judge_example_shell_allowlist());
        assert!(config.policy.network.enabled);
        assert!(
            config.models.subagent.is_some_and(|slot| matches!(
                slot,
                ModelSlot::Reference(ModelSlotRef::AutoResolve)
            ))
        );
    }

    #[test]
    fn add_worktree_policy_absolutizes_relative_resource_roots_idempotently() {
        let mut config = default_factory_config();
        let worktree = Path::new("/tmp/factory-project");

        add_worktree_policy(&mut config, worktree);
        add_worktree_policy(&mut config, worktree);

        assert_eq!(
            config.policy.allowed_write_roots,
            vec![worktree.to_path_buf(), PathBuf::from("/tmp/halter"),]
        );
        assert_eq!(
            config.resources.skills.roots,
            vec![worktree.join(".agent/skills")]
        );
        assert_eq!(
            config.resources.plugins.roots,
            vec![worktree.join(".agent/plugins")]
        );

        let mut config = default_factory_config();
        config.resources.skills.roots = vec![PathBuf::from("~/skills")];
        add_worktree_policy(&mut config, worktree);

        assert_eq!(
            config.resources.skills.roots,
            vec![PathBuf::from("~/skills")]
        );
    }

    #[test]
    fn excluded_commit_paths_cover_checkpoint_and_optional_plan() {
        assert_eq!(excluded_commit_paths(true), vec![CHECKPOINT_PATH]);
        assert_eq!(
            excluded_commit_paths(false),
            vec![IMPLEMENTATION_PLAN_PATH, CHECKPOINT_PATH]
        );
    }

    #[test]
    fn logging_filter_spec_defaults_or_uses_rust_log_and_appends_suppressions() {
        struct Case {
            name: &'static str,
            user_directives: Option<&'static str>,
            expected_prefix: &'static str,
        }

        let cases = [
            Case {
                name: "missing rust log defaults to info",
                user_directives: None,
                expected_prefix: "info,",
            },
            Case {
                name: "blank rust log defaults to info",
                user_directives: Some(" \n"),
                expected_prefix: "info,",
            },
            Case {
                name: "configured rust log is preserved",
                user_directives: Some("debug,halter=trace"),
                expected_prefix: "debug,halter=trace,",
            },
        ];

        for case in cases {
            let spec = logging_filter_spec(case.user_directives);

            assert!(
                spec.starts_with(case.expected_prefix),
                "{}: {spec}",
                case.name
            );
            assert!(spec.contains(NOISY_TARGET_SUPPRESSIONS), "{}", case.name);
            logging_filter_from_spec(&spec).expect(case.name);
        }
    }

    #[test]
    fn logging_filter_from_spec_covers_valid_and_invalid_specs() {
        logging_filter_from_spec("info,halter=debug").expect("valid filter");

        let error =
            logging_filter_from_spec("halter=not-a-level").expect_err("invalid level should fail");

        assert!(error.to_string().contains("invalid RUST_LOG filter"));
    }

    #[test]
    fn single_line_preview_covers_normalized_truncated_and_empty_text() {
        struct Case {
            name: &'static str,
            text: &'static str,
            max_chars: usize,
            expected: &'static str,
        }

        let cases = [
            Case {
                name: "empty",
                text: "",
                max_chars: 10,
                expected: "",
            },
            Case {
                name: "short unchanged",
                text: "abc",
                max_chars: 10,
                expected: "abc",
            },
            Case {
                name: "newlines are escaped",
                text: "a\nb\rc",
                max_chars: 20,
                expected: "a\\nb\\rc",
            },
            Case {
                name: "truncated at character boundary",
                text: "abéde",
                max_chars: 3,
                expected: "abé...",
            },
            Case {
                name: "zero limit truncates non-empty text",
                text: "abc",
                max_chars: 0,
                expected: "...",
            },
        ];

        for case in cases {
            assert_eq!(
                single_line_preview(case.text, case.max_chars),
                case.expected,
                "{}",
                case.name
            );
        }
    }

    #[test]
    fn json_preview_serializes_before_previewing() {
        let value = json!({
            "message": "hello\nworld",
            "ok": true,
        });

        let preview = json_preview(&value, 16);

        assert!(preview.starts_with('{'));
        assert!(preview.ends_with("..."));
        assert!(!preview.contains('\n'));
    }

    #[test]
    fn tool_result_logging_helpers_cover_each_result_kind() {
        let json_value = json!({"a": 1});
        let cases = [
            (ToolResult::Empty, "empty", 0),
            (
                ToolResult::Text {
                    text: "hello".to_owned(),
                },
                "text",
                5,
            ),
            (
                ToolResult::Json {
                    value: json_value.clone(),
                },
                "json",
                json_value.to_string().len(),
            ),
        ];

        for (result, expected_kind, expected_size) in cases {
            assert_eq!(tool_result_kind(&result), expected_kind);
            assert_eq!(tool_result_size(&result), expected_size);
        }
    }

    #[tokio::test]
    async fn read_project_system_prompt_returns_none_when_no_guidance_files_exist() {
        let dir = tempfile::tempdir().expect("tempdir");

        let prompt = read_project_system_prompt(dir.path())
            .await
            .expect("guidance read");

        assert_eq!(prompt, None);
    }

    #[tokio::test]
    async fn read_project_system_prompt_reads_top_level_guidance_in_fixed_order() {
        let dir = tempfile::tempdir().expect("tempdir");
        tokio::fs::write(dir.path().join("SOUL.md"), "soul rules")
            .await
            .expect("write soul");
        tokio::fs::write(dir.path().join("CLAUDE.md"), "claude rules")
            .await
            .expect("write claude");
        tokio::fs::create_dir_all(dir.path().join("nested"))
            .await
            .expect("create nested");
        tokio::fs::write(dir.path().join("nested").join("AGENTS.md"), "ignored")
            .await
            .expect("write nested agents");

        let prompt = read_project_system_prompt(dir.path())
            .await
            .expect("guidance read")
            .expect("guidance prompt");

        let claude = prompt.find("## CLAUDE.md").expect("claude section");
        let soul = prompt.find("## SOUL.md").expect("soul section");
        assert!(claude < soul);
        assert!(prompt.contains("claude rules"));
        assert!(prompt.contains("soul rules"));
        assert!(!prompt.contains("ignored"));
    }

    #[tokio::test]
    async fn read_project_system_prompt_rejects_oversized_guidance_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        tokio::fs::write(
            dir.path().join("CLAUDE.md"),
            vec![b'x'; PROJECT_GUIDANCE_MAX_BYTES as usize + 1],
        )
        .await
        .expect("write oversized claude");

        let error = read_project_system_prompt(dir.path())
            .await
            .expect_err("oversized guidance should fail");

        assert!(error.to_string().contains("above the"));
    }

    #[test]
    fn factory_system_prompt_segments_use_built_in_defaults() {
        let general = FactorySystemPrompt::General.segment();
        assert_eq!(general.text, prompts::default_system_prompt());
        assert_eq!(general.kind, PromptSegmentKind::System);

        let coding = FactorySystemPrompt::Coding.segment();
        assert_eq!(coding.text, prompts::default_coding_agent_prompt());
        assert_eq!(coding.kind, PromptSegmentKind::System);
    }

    #[test]
    fn project_guidance_prompt_segment_covers_empty_and_non_empty_inputs() {
        assert!(project_guidance_prompt_segment(None).is_none());
        assert!(project_guidance_prompt_segment(Some(" \n")).is_none());

        let segment =
            project_guidance_prompt_segment(Some("Follow project rules.")).expect("segment");

        assert_eq!(segment.text, "Follow project rules.");
        assert_eq!(segment.kind, PromptSegmentKind::Append);
        assert_eq!(segment.volatility, Volatility::TurnDynamic);
        assert_eq!(segment.cache_scope, CacheScope::Dynamic);
        assert_eq!(segment.content_hash.len(), 64);
    }

    #[test]
    fn turn_instructions_prompt_segment_covers_non_empty_and_empty_inputs() {
        let segment =
            turn_instructions_prompt_segment("  Run the focused tests.  ").expect("segment");

        assert_eq!(segment.kind, PromptSegmentKind::Append);
        assert_eq!(segment.volatility, Volatility::TurnDynamic);
        assert_eq!(segment.cache_scope, CacheScope::Dynamic);
        assert!(segment.text.contains("# Turn-Specific Instructions"));
        assert!(segment.text.contains("Run the focused tests."));

        let error = turn_instructions_prompt_segment(" \n").expect_err("empty should fail");
        assert!(
            error
                .to_string()
                .contains("turn-specific instructions are empty")
        );
    }

    #[test]
    fn agent_run_from_text_covers_required_and_optional_outputs() {
        struct Case {
            name: &'static str,
            latest_text: Option<&'static str>,
            delta_text: &'static str,
            requirement: AgentTextRequirement,
            expected_text: Option<&'static str>,
            expected_error: Option<&'static str>,
        }

        let cases = [
            Case {
                name: "required latest assistant text",
                latest_text: Some("final message"),
                delta_text: "partial delta",
                requirement: AgentTextRequirement::Required,
                expected_text: Some("final message"),
                expected_error: None,
            },
            Case {
                name: "required delta fallback",
                latest_text: None,
                delta_text: "streamed message",
                requirement: AgentTextRequirement::Required,
                expected_text: Some("streamed message"),
                expected_error: None,
            },
            Case {
                name: "required ignores blank latest and uses delta",
                latest_text: Some(" \n"),
                delta_text: "streamed message",
                requirement: AgentTextRequirement::Required,
                expected_text: Some("streamed message"),
                expected_error: None,
            },
            Case {
                name: "required empty output fails",
                latest_text: None,
                delta_text: " \n",
                requirement: AgentTextRequirement::Required,
                expected_text: None,
                expected_error: Some("produced no assistant text"),
            },
            Case {
                name: "optional empty output succeeds",
                latest_text: None,
                delta_text: " \n",
                requirement: AgentTextRequirement::Optional,
                expected_text: Some(""),
                expected_error: None,
            },
        ];

        for case in cases {
            let result = agent_run_from_text(
                "test stage",
                case.latest_text.map(ToOwned::to_owned),
                case.delta_text.to_owned(),
                case.requirement,
            );

            match (case.expected_text, case.expected_error, result) {
                (Some(expected), None, Ok(run)) => {
                    assert_eq!(run.text, expected, "{}", case.name);
                }
                (None, Some(expected), Err(error)) => {
                    assert!(
                        error.to_string().contains(expected),
                        "{}: {error}",
                        case.name
                    );
                }
                (_, _, other) => panic!("{}: unexpected result {other:?}", case.name),
            }
        }
    }

    #[test]
    fn agent_stage_failure_retry_detection_covers_flags_and_transient_text() {
        struct Case {
            name: &'static str,
            retryable: bool,
            cancelled: bool,
            error: &'static str,
            expected: bool,
        }

        let cases = [
            Case {
                name: "provider retryable flag",
                retryable: true,
                cancelled: false,
                error: "provider asked for retry",
                expected: true,
            },
            Case {
                name: "cancelled provider retryable flag",
                retryable: true,
                cancelled: true,
                error: "provider asked for retry",
                expected: false,
            },
            Case {
                name: "overloaded text fallback",
                retryable: false,
                cancelled: false,
                error: r#"Provider returned error (Parasail: {"error":{"message":"The engine is currently overloaded. Please try again later."}})"#,
                expected: true,
            },
            Case {
                name: "rate limit text fallback",
                retryable: false,
                cancelled: false,
                error: "provider returned 429 Too Many Requests",
                expected: true,
            },
            Case {
                name: "fatal validation error",
                retryable: false,
                cancelled: false,
                error: "invalid request: missing required field",
                expected: false,
            },
        ];

        for case in cases {
            assert_eq!(
                agent_stage_failure_is_retryable(case.retryable, case.cancelled, case.error),
                case.expected,
                "{}",
                case.name
            );
        }
    }

    #[test]
    fn agent_stage_retry_policy_covers_hint_exponential_cap_and_exhaustion() {
        struct Case {
            name: &'static str,
            policy: AgentStageRetryPolicy,
            failed_attempt: u32,
            error: &'static str,
            expected: Option<Duration>,
        }

        let policy = AgentStageRetryPolicy {
            max_attempts: 3,
            base_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_millis(250),
        };
        let cases = [
            Case {
                name: "first retry uses base backoff",
                policy,
                failed_attempt: 1,
                error: "retryable provider failure",
                expected: Some(Duration::from_millis(100)),
            },
            Case {
                name: "second retry doubles backoff",
                policy,
                failed_attempt: 2,
                error: "retryable provider failure",
                expected: Some(Duration::from_millis(200)),
            },
            Case {
                name: "hint is capped to max backoff",
                policy,
                failed_attempt: 1,
                error: "upstream model is overloaded",
                expected: Some(Duration::from_millis(250)),
            },
            Case {
                name: "budget exhaustion returns none",
                policy,
                failed_attempt: 3,
                error: "retryable provider failure",
                expected: None,
            },
        ];

        for case in cases {
            assert_eq!(
                case.policy
                    .delay_after_failure(case.failed_attempt, case.error),
                case.expected,
                "{}",
                case.name
            );
        }
    }

    #[test]
    fn agent_stage_turn_failure_display_and_retry_metadata_match() {
        let retryable = AgentStageTurnFailure {
            label: "kimi implementation".to_owned(),
            error: "provider overloaded".to_owned(),
            retryable: false,
            cancelled: false,
        };
        assert_eq!(
            retryable.to_string(),
            "agent stage kimi implementation failed: provider overloaded"
        );
        assert!(retryable.should_retry());

        let cancelled = AgentStageTurnFailure {
            cancelled: true,
            ..retryable
        };
        assert!(!cancelled.should_retry());
    }

    #[test]
    fn checkpoint_validation_covers_context_and_stage_errors() {
        let repo = RepoSlug::parse("pbdeuchler/halter").expect("valid repo");
        let valid = FactoryCheckpoint::new(&repo, "main", Some(7), false);
        validate_checkpoint_for_run(&valid, &repo, "main", Some(7), false)
            .expect("valid checkpoint");
        let mut completed = valid.clone();
        completed.candidates = Some(sample_candidates());
        completed.selection = Some(sample_selection());
        completed.implementation_plan = Some("plan".to_owned());
        completed.branch = Some("factory/test".to_owned());
        completed.base_ref = Some("origin/main".to_owned());
        completed.implemented = true;
        completed.reviewed = true;
        completed.committed = true;
        completed.commit_sha = Some("abc123".to_owned());
        completed.pushed = true;
        completed.pr_draft = Some(sample_pr_draft());
        completed.pr = Some(sample_checkpoint_pr());
        completed.completed = true;
        validate_checkpoint_for_run(&completed, &repo, "main", Some(7), false)
            .expect("completed checkpoint");

        let mut wrong_version = valid.clone();
        wrong_version.version = CHECKPOINT_VERSION + 1;
        assert!(
            validate_checkpoint_for_run(&wrong_version, &repo, "main", Some(7), false)
                .expect_err("wrong version")
                .contains("version")
        );

        let wrong_repo = FactoryCheckpoint {
            repo: "other/repo".to_owned(),
            ..valid.clone()
        };
        assert!(
            validate_checkpoint_for_run(&wrong_repo, &repo, "main", Some(7), false)
                .expect_err("wrong repo")
                .contains("repo")
        );

        assert!(
            validate_checkpoint_for_run(&valid, &repo, "release", Some(7), false)
                .expect_err("wrong base")
                .contains("base branch")
        );
        assert!(
            validate_checkpoint_for_run(&valid, &repo, "main", Some(8), false)
                .expect_err("wrong issue")
                .contains("requested issue")
        );
        assert!(
            validate_checkpoint_for_run(&valid, &repo, "main", Some(7), true)
                .expect_err("wrong commit_impl_plan")
                .contains("commit_impl_plan")
        );

        struct Case {
            name: &'static str,
            mutate: Box<dyn Fn(&mut FactoryCheckpoint)>,
            expected: &'static str,
        }

        let cases = [
            Case {
                name: "selection without candidates",
                mutate: Box::new(|checkpoint| {
                    checkpoint.selection = Some(sample_selection());
                }),
                expected: "selection",
            },
            Case {
                name: "plan without selection",
                mutate: Box::new(|checkpoint| {
                    checkpoint.implementation_plan = Some("plan".to_owned());
                }),
                expected: "implementation plan",
            },
            Case {
                name: "branch without selection",
                mutate: Box::new(|checkpoint| {
                    checkpoint.branch = Some("factory/test".to_owned());
                }),
                expected: "branch",
            },
            Case {
                name: "implemented without branch",
                mutate: Box::new(|checkpoint| {
                    checkpoint.implemented = true;
                }),
                expected: "implementation",
            },
            Case {
                name: "reviewed before implemented",
                mutate: Box::new(|checkpoint| {
                    checkpoint.reviewed = true;
                }),
                expected: "review",
            },
            Case {
                name: "committed before reviewed",
                mutate: Box::new(|checkpoint| {
                    checkpoint.committed = true;
                }),
                expected: "commit",
            },
            Case {
                name: "pushed before committed",
                mutate: Box::new(|checkpoint| {
                    checkpoint.pushed = true;
                }),
                expected: "push",
            },
            Case {
                name: "draft before push",
                mutate: Box::new(|checkpoint| {
                    checkpoint.pr_draft = Some(sample_pr_draft());
                }),
                expected: "PR draft",
            },
            Case {
                name: "pr before draft",
                mutate: Box::new(|checkpoint| {
                    checkpoint.pr = Some(sample_checkpoint_pr());
                }),
                expected: "PR",
            },
            Case {
                name: "completed before pr",
                mutate: Box::new(|checkpoint| {
                    checkpoint.completed = true;
                }),
                expected: "complete",
            },
        ];

        for case in cases {
            let mut checkpoint = valid.clone();
            (case.mutate)(&mut checkpoint);
            let error = validate_checkpoint_stage_state(&checkpoint).expect_err(case.name);
            assert!(
                error.contains(case.expected),
                "{}: expected {error:?} to contain {:?}",
                case.name,
                case.expected
            );
        }
    }

    #[tokio::test]
    async fn checkpoint_file_io_covers_write_read_remove_and_missing() {
        let repo = RepoSlug::parse("pbdeuchler/halter").expect("valid repo");
        let checkpoint = FactoryCheckpoint::new(&repo, "main", Some(7), false);
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(".halter/software-factory/checkpoint.json");

        assert!(!checkpoint_exists(&path).await.expect("exists check"));
        remove_checkpoint_if_exists(&path)
            .await
            .expect("remove missing");

        write_checkpoint(&path, &checkpoint)
            .await
            .expect("write checkpoint");
        assert!(checkpoint_exists(&path).await.expect("exists check"));

        let loaded = read_checkpoint(&path).await.expect("read checkpoint");
        assert_eq!(loaded, checkpoint);

        remove_checkpoint_if_exists(&path)
            .await
            .expect("remove checkpoint");
        assert!(!checkpoint_exists(&path).await.expect("exists check"));
        let error = read_checkpoint(&path)
            .await
            .expect_err("missing checkpoint should fail");
        assert!(error.to_string().contains("failed to read"));

        tokio::fs::create_dir_all(&path)
            .await
            .expect("create directory at checkpoint path");
        let error = checkpoint_exists(&path)
            .await
            .expect_err("directory checkpoint path should fail");
        assert!(error.to_string().contains("not a regular file"));
    }

    #[tokio::test]
    async fn initialize_checkpoint_covers_fresh_resume_existing_and_reset() {
        let repo = RepoSlug::parse("pbdeuchler/halter").expect("valid repo");
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join(".halter/software-factory/checkpoint.json");

        let fresh = initialize_checkpoint(&path, false, false, &repo, "main", None, false)
            .await
            .expect("fresh checkpoint");
        assert_eq!(fresh, FactoryCheckpoint::new(&repo, "main", None, false));

        let existing_error = initialize_checkpoint(&path, false, false, &repo, "main", None, false)
            .await
            .expect_err("existing checkpoint should block fresh run");
        assert!(existing_error.to_string().contains("--resume"));

        let resumed = initialize_checkpoint(&path, true, false, &repo, "main", None, false)
            .await
            .expect("resume checkpoint");
        assert_eq!(resumed, fresh);

        let reset = initialize_checkpoint(&path, false, true, &repo, "main", Some(9), true)
            .await
            .expect("reset checkpoint");
        assert_eq!(reset, FactoryCheckpoint::new(&repo, "main", Some(9), true));

        let mismatch = initialize_checkpoint(&path, true, false, &repo, "main", None, true)
            .await
            .expect_err("resume should validate requested issue");
        assert!(mismatch.to_string().contains("requested issue"));
    }

    #[tokio::test]
    async fn resolve_execution_worktree_covers_current_tmp_resume_and_bad_resume() {
        let repo = RepoSlug::parse("pbdeuchler/halter").expect("valid repo");
        let dir = tempfile::tempdir().expect("tempdir");

        let current =
            resolve_execution_worktree(dir.path(), false, false, &repo, "main", "20260617")
                .await
                .expect("current worktree mode");
        assert_eq!(current, dir.path());

        let tmp = factory_worktree_tmp_root().join("resume-target");
        let resumed = resolve_execution_worktree(&tmp, true, true, &repo, "main", "20260617")
            .await
            .expect("resume from tmp factory worktree");
        assert_eq!(resumed, tmp);

        let error = resolve_execution_worktree(dir.path(), true, true, &repo, "main", "20260617")
            .await
            .expect_err("resume outside tmp factory worktree should fail");
        assert!(
            error
                .to_string()
                .contains("cd into the existing factory worktree")
        );
    }

    #[tokio::test]
    async fn create_factory_worktree_rejects_existing_path() {
        let repo = RepoSlug::parse("pbdeuchler/halter").expect("valid repo");
        let run_id = unique_test_run_id("existing");
        let path = factory_worktree_tmp_root().join(factory_worktree_dir_name(&repo, &run_id));
        remove_dir_if_exists(&path).await;
        tokio::fs::create_dir_all(&path)
            .await
            .expect("create existing path");

        let error = create_factory_worktree(Path::new("/does/not/matter"), &repo, "main", &run_id)
            .await
            .expect_err("existing worktree path should fail");

        assert!(error.to_string().contains("already exists"));
        remove_dir_if_exists(&path).await;
    }

    #[tokio::test]
    async fn create_factory_worktree_adds_detached_tmp_worktree() {
        let (_dir, source) = init_git_repo_with_origin().await;
        let repo = RepoSlug::parse("pbdeuchler/halter").expect("valid repo");
        let run_id = unique_test_run_id("create");
        let expected_path =
            factory_worktree_tmp_root().join(factory_worktree_dir_name(&repo, &run_id));
        remove_dir_if_exists(&expected_path).await;

        let worktree = create_factory_worktree(&source, &repo, "main", &run_id)
            .await
            .expect("create factory worktree");

        assert!(path_is_factory_tmp_worktree(&worktree));
        assert_eq!(
            run_cmd(&worktree, "git", &["rev-parse", "--is-inside-work-tree"])
                .await
                .expect("inside worktree")
                .trim(),
            "true"
        );
        assert_eq!(
            run_cmd(&worktree, "git", &["branch", "--show-current"])
                .await
                .expect("detached branch")
                .trim(),
            ""
        );
        assert!(worktree.join("README.md").exists());

        remove_git_worktree(&source, &worktree).await;
    }

    #[tokio::test]
    async fn prepare_branch_covers_generated_branch_and_dirty_rejection() {
        let (_dir, source) = init_git_repo_with_origin().await;
        let repo = RepoSlug::parse("pbdeuchler/halter").expect("valid repo");
        let selection = sample_selection();

        let branch = prepare_branch(
            &source,
            "main",
            None,
            false,
            &repo,
            &selection,
            &[],
            "20260617",
        )
        .await
        .expect("prepare generated branch");
        assert_eq!(branch, "halter-factory/halter-20260617-fix-issue");
        assert_eq!(
            current_branch(&source).await.expect("current branch"),
            branch
        );

        tokio::fs::write(source.join("dirty.txt"), "dirty\n")
            .await
            .expect("write dirty file");
        let error = prepare_branch(
            &source,
            "main",
            Some("factory/other"),
            false,
            &repo,
            &selection,
            &[],
            "20260618",
        )
        .await
        .expect_err("dirty worktree should fail");
        assert!(error.to_string().contains("worktree is dirty"));
    }

    #[tokio::test]
    async fn commit_if_dirty_excludes_multiple_factory_state_paths() {
        let (_dir, source) = init_git_repo_with_origin().await;
        let state_dir = source.join(".halter/software-factory");
        tokio::fs::create_dir_all(&state_dir)
            .await
            .expect("create factory state dir");
        tokio::fs::write(source.join("tracked.txt"), "tracked\n")
            .await
            .expect("write tracked change");
        tokio::fs::write(source.join(IMPLEMENTATION_PLAN_PATH), "plan\n")
            .await
            .expect("write implementation plan");
        tokio::fs::write(source.join(CHECKPOINT_PATH), "{}\n")
            .await
            .expect("write checkpoint");

        let committed = commit_if_dirty(
            &source,
            "Commit tracked change",
            &[IMPLEMENTATION_PLAN_PATH, CHECKPOINT_PATH],
        )
        .await
        .expect("commit tracked change");
        assert!(committed);

        let committed_paths = run_cmd(
            &source,
            "git",
            &["show", "--name-only", "--format=", "HEAD"],
        )
        .await
        .expect("show committed paths");
        assert!(committed_paths.lines().any(|line| line == "tracked.txt"));
        assert!(!committed_paths.contains(".halter/software-factory"));

        let committed = commit_if_dirty(
            &source,
            "Skip local state only",
            &[IMPLEMENTATION_PLAN_PATH, CHECKPOINT_PATH],
        )
        .await
        .expect("skip local state only");
        assert!(!committed);
    }

    fn sample_candidates() -> CandidateSet {
        CandidateSet {
            candidates: vec![crate::core::IssueCandidate {
                title: "Fix issue".to_owned(),
                issue_numbers: vec![7],
                rationale: "selected".to_owned(),
                maintainer_input_risk: "low".to_owned(),
            }],
        }
    }

    fn sample_selection() -> JudgeSelection {
        JudgeSelection {
            title: "Fix issue".to_owned(),
            issue_numbers: vec![7],
            notes: "notes".to_owned(),
        }
    }

    fn sample_pr_draft() -> PullRequestDraft {
        PullRequestDraft {
            title: "Fix issue".to_owned(),
            body: "Body".to_owned(),
        }
    }

    fn sample_checkpoint_pr() -> CheckpointPullRequest {
        CheckpointPullRequest {
            number: 42,
            html_url: "https://github.com/pbdeuchler/halter/pull/42".to_owned(),
        }
    }

    fn unique_test_run_id(prefix: &str) -> String {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system time after epoch")
            .as_nanos();
        format!("{prefix}-{}-{nanos}", std::process::id())
    }

    async fn remove_dir_if_exists(path: &Path) {
        match tokio::fs::remove_dir_all(path).await {
            Ok(()) => {}
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            Err(error) => panic!("failed to remove {}: {error}", path.display()),
        }
    }

    async fn init_git_repo_with_origin() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let origin = dir.path().join("origin.git");
        let source = dir.path().join("source");
        let origin_arg = origin.to_str().expect("utf-8 origin path");
        let source_arg = source.to_str().expect("utf-8 source path");

        run_cmd(dir.path(), "git", &["init", "--bare", origin_arg])
            .await
            .expect("init bare origin");
        run_cmd(dir.path(), "git", &["init", source_arg])
            .await
            .expect("init source repo");
        run_cmd(&source, "git", &["checkout", "-b", "main"])
            .await
            .expect("create main branch");
        run_cmd(
            &source,
            "git",
            &["config", "user.name", "Software Factory Test"],
        )
        .await
        .expect("set git user name");
        run_cmd(
            &source,
            "git",
            &["config", "user.email", "software-factory-test@example.com"],
        )
        .await
        .expect("set git user email");
        tokio::fs::write(source.join("README.md"), "hello\n")
            .await
            .expect("write readme");
        run_cmd(&source, "git", &["add", "README.md"])
            .await
            .expect("stage readme");
        run_cmd(&source, "git", &["commit", "-m", "Initial commit"])
            .await
            .expect("initial commit");
        run_cmd(&source, "git", &["remote", "add", "origin", origin_arg])
            .await
            .expect("add origin");
        run_cmd(&source, "git", &["push", "-u", "origin", "main"])
            .await
            .expect("push main");

        (dir, source)
    }

    async fn remove_git_worktree(source: &Path, worktree: &Path) {
        let worktree_arg = worktree.to_str().expect("utf-8 worktree path");
        run_cmd(
            source,
            "git",
            &["worktree", "remove", "--force", worktree_arg],
        )
        .await
        .expect("remove git worktree");
    }

    #[test]
    fn session_init_with_appended_context_uses_coding_prompt_and_append_segments() {
        let init = session_init_with_appended_context(
            Path::new("/tmp/project"),
            FactorySystemPrompt::Coding,
            "do the work",
            Some("rules"),
            None,
        )
        .expect("session init");

        assert_eq!(init.working_dir, PathBuf::from("/tmp/project"));
        assert_eq!(init.system_prompt_seed.len(), 3);
        assert_eq!(
            init.system_prompt_seed[0].text,
            prompts::default_coding_agent_prompt()
        );
        assert_eq!(init.system_prompt_seed[0].kind, PromptSegmentKind::System);
        assert_eq!(init.system_prompt_seed[1].text, "rules");
        assert_eq!(init.system_prompt_seed[1].kind, PromptSegmentKind::Append);
        assert!(init.system_prompt_seed[2].text.contains("do the work"));
        assert_eq!(init.system_prompt_seed[2].kind, PromptSegmentKind::Append);
    }

    #[test]
    fn session_init_with_appended_context_rejects_empty_turn_instructions() {
        let error = session_init_with_appended_context(
            Path::new("/tmp/project"),
            FactorySystemPrompt::General,
            " \n",
            None,
            None,
        )
        .expect_err("empty turn instructions should fail");

        assert!(
            error
                .to_string()
                .contains("turn-specific instructions are empty")
        );
    }

    #[test]
    fn session_init_with_appended_context_applies_optional_max_turns() {
        let init = session_init_with_appended_context(
            Path::new("/tmp/project"),
            FactorySystemPrompt::Coding,
            "review the branch",
            None,
            Some(CODE_REVIEW_MAX_TURNS),
        )
        .expect("session init");

        assert_eq!(init.max_turns, Some(10));
    }

    #[test]
    fn code_review_prompt_covers_initial_follow_up_and_final_iterations() {
        struct Case {
            name: &'static str,
            iteration: ReviewIteration,
            want_initial: bool,
            want_follow_up: bool,
            want_final: bool,
        }

        let cases = [
            Case {
                name: "initial_review",
                iteration: ReviewIteration { current: 1, max: 5 },
                want_initial: true,
                want_follow_up: false,
                want_final: false,
            },
            Case {
                name: "follow_up_review",
                iteration: ReviewIteration { current: 2, max: 5 },
                want_initial: false,
                want_follow_up: true,
                want_final: false,
            },
            Case {
                name: "final_review",
                iteration: ReviewIteration { current: 5, max: 5 },
                want_initial: false,
                want_follow_up: true,
                want_final: true,
            },
        ];

        for case in cases {
            let prompt = code_review_prompt("origin/master", "diff --git a/x b/x", case.iteration);

            assert_eq!(
                prompt.contains("You are reviewing a branch diff against origin/master."),
                case.want_initial,
                "{} initial prompt mismatch",
                case.name
            );
            assert_eq!(
                prompt.contains("Your previous code review has been addressed."),
                case.want_follow_up,
                "{} follow-up prompt mismatch",
                case.name
            );
            assert_eq!(
                prompt.contains("last code review iteration"),
                case.want_final,
                "{} final prompt mismatch",
                case.name
            );
            assert!(prompt.contains(JSON_ONLY_OUTPUT_RULE), "{}", case.name);
            assert!(prompt.contains("BRANCH DIFF:\ndiff --git"), "{}", case.name);
        }
    }

    #[test]
    fn review_repair_prompt_covers_regular_and_final_iterations() {
        struct Case {
            name: &'static str,
            iteration: ReviewIteration,
            want_final: bool,
        }

        let cases = [
            Case {
                name: "regular_repair",
                iteration: ReviewIteration { current: 4, max: 5 },
                want_final: false,
            },
            Case {
                name: "final_repair",
                iteration: ReviewIteration { current: 5, max: 5 },
                want_final: true,
            },
        ];

        for case in cases {
            let prompt = review_repair_prompt("plan", r#"{"clean":false}"#, case.iteration);

            assert!(
                prompt.contains("IMPLEMENTATION PLAN:\nplan"),
                "{}",
                case.name
            );
            assert!(
                prompt.contains(
                    r#"REVIEW RESULT:
{"clean":false}"#
                ),
                "{}",
                case.name
            );
            assert_eq!(
                prompt.contains("last review-repair iteration"),
                case.want_final,
                "{} final repair mismatch",
                case.name
            );
            assert_eq!(
                prompt.contains("different implementation approach"),
                case.want_final,
                "{} final approach mismatch",
                case.name
            );
        }
    }

    #[tokio::test]
    async fn github_issue_tool_returns_cached_issue_without_fetching() {
        let issue = IssueDoc {
            number: 7,
            state: "open".to_owned(),
            title: "cached issue".to_owned(),
            body: "body".to_owned(),
            labels: vec![],
            author: "maintainer".to_owned(),
            url: "https://example.test/issues/7".to_owned(),
            comments: vec![],
        };
        let cache = issue_cache_from_docs(&[issue]);
        let tool = GitHubIssueTool::new(
            GitHubClient {
                client: reqwest::Client::new(),
            },
            RepoSlug::parse("pbdeuchler/halter").expect("valid repo"),
            cache,
            HashSet::from([7]),
        );

        let (got, source) = tool.cached_or_fetch(7).await.expect("cached issue");

        assert_eq!(source, "cache");
        assert_eq!(got.number, 7);
        assert_eq!(got.title, "cached issue");
    }

    #[tokio::test]
    async fn github_issue_tool_rejects_issue_outside_current_corpus() {
        let cache = issue_cache_from_docs(&[]);
        let tool = GitHubIssueTool::new(
            GitHubClient {
                client: reqwest::Client::new(),
            },
            RepoSlug::parse("pbdeuchler/halter").expect("valid repo"),
            cache,
            HashSet::from([7]),
        );

        let error = tool
            .cached_or_fetch(8)
            .await
            .expect_err("outside issue should be rejected before fetching");

        assert!(
            error
                .to_string()
                .contains("outside the current issue corpus")
        );
    }
}
