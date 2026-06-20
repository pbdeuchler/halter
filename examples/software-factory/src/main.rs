mod agent;
mod checkpoint;
mod cli;
mod config;
mod core;
mod git;
mod github;
mod harness;
mod logging;
mod prompts;
mod stages;
#[cfg(test)]
mod test_support;
mod worktree;

use std::collections::HashSet;
use std::sync::Arc;

use anyhow::{Context, bail};
use chrono::Utc;
use clap::Parser;
use halter_protocol::ReasoningEffort;
use tracing::info;

use crate::checkpoint::{
    CheckpointPullRequest, FACTORY_LOCAL_STATE_PATHS, excluded_commit_paths, initialize_checkpoint,
    restore_implementation_plan, write_checkpoint,
};
use crate::cli::Cli;
use crate::config::{add_worktree_policy, default_factory_config};
use crate::core::{
    CHECKPOINT_PATH, IMPLEMENTATION_PLAN_PATH, ModelSpec, PLANNING_CORPUS_BODY_CHARS,
    RECENT_OPEN_ISSUE_LIMIT, candidate_set_for_issue, ensure_requested_issue_selection,
    issue_corpus, selected_issue_numbers, validate_issue_number,
};
use crate::git::{
    branch_diff, branch_has_diff, checkout_branch, commit_if_dirty, current_commit, run_cmd,
};
use crate::github::{
    GitHubClient, GitHubIssueTool, github_repo_from_git_remote, issue_cache_from_docs,
};
use crate::harness::{build_judge_harness, build_model_harness, shutdown_all};
use crate::logging::init_logging;
use crate::prompts::read_project_system_prompt;
use crate::stages::{
    MonitorContext, draft_pr, ensure_selected_issues_are_open, implement_plan, judge_issue_plan,
    monitor_pr, prepare_branch, propose_issue_candidates, read_implementation_plan,
    run_review_loop,
};
use crate::worktree::{git_worktree_root, resolve_execution_worktree};

#[tokio::main]
pub(crate) async fn main() -> anyhow::Result<()> {
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
