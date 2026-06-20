use std::collections::HashSet;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, bail};
use halter::prelude::*;
use tracing::{info, warn};

use crate::agent::{
    run_agent_with_system_prompt, run_code_review_agent_with_system_prompt,
    run_coding_action_with_system_prompt,
};
use crate::core::{
    CandidateSet, CodeReview, IssueDoc, JudgeSelection, PullRequestDraft, RepoSlug, branch_name,
    issue_corpus, parse_json_response, selected_issue_numbers,
};
use crate::git::{branch_diff, commit_if_dirty, git_is_dirty, run_cmd};
use crate::github::GitHubClient;
use crate::prompts::{
    CARGO_TIMEOUT_RULE, JSON_ONLY_OUTPUT_RULE, ReviewIteration, code_review_prompt,
    review_repair_prompt,
};

pub(crate) async fn propose_issue_candidates(
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

pub(crate) async fn judge_issue_plan(
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

pub(crate) async fn read_implementation_plan(path: &Path) -> anyhow::Result<String> {
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

pub(crate) async fn prepare_branch(
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

pub(crate) async fn implement_plan(
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

pub(crate) fn selected_issue_details(selection: &JudgeSelection, issues: &[IssueDoc]) -> String {
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

pub(crate) fn ensure_selected_issues_are_open(
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

pub(crate) async fn run_review_loop(
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

pub(crate) async fn review_diff(
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

pub(crate) async fn draft_pr(
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

pub(crate) struct MonitorContext<'a> {
    pub(crate) github: &'a GitHubClient,
    pub(crate) glm: &'a Halter,
    pub(crate) implementer: &'a Halter,
    pub(crate) reviewer: &'a Halter,
    pub(crate) worktree: &'a Path,
    pub(crate) repo: &'a RepoSlug,
    pub(crate) pr_number: u64,
    pub(crate) branch: &'a str,
    pub(crate) base_ref: &'a str,
    pub(crate) selection: &'a JudgeSelection,
    pub(crate) implementation_plan: &'a str,
    pub(crate) project_system_prompt: Option<&'a str>,
    pub(crate) excluded_commit_paths: &'a [&'a str],
    pub(crate) max_review_iterations: usize,
    pub(crate) poll_seconds: u64,
}

pub(crate) async fn monitor_pr(ctx: MonitorContext<'_>) -> anyhow::Result<()> {
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

pub(crate) async fn refine_plsfix_comments(
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

pub(crate) async fn apply_feedback(
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::*;

    use crate::core::RepoSlug;
    use crate::git::current_branch;

    #[tokio::test]
    pub(crate) async fn prepare_branch_covers_generated_branch_and_dirty_rejection() {
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
}
