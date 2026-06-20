use std::path::Path;

use anyhow::{Context, bail};
use tokio::process::Command;
use tracing::{debug, info, warn};

use crate::agent::single_line_preview;
use crate::core::dirty_status_excluding;

pub(crate) async fn git_is_dirty(worktree: &Path, excluded_paths: &[&str]) -> anyhow::Result<bool> {
    let status = run_cmd(worktree, "git", &["status", "--porcelain"]).await?;
    Ok(dirty_status_excluding(&status, excluded_paths))
}

pub(crate) async fn current_branch(worktree: &Path) -> anyhow::Result<String> {
    Ok(run_cmd(worktree, "git", &["branch", "--show-current"])
        .await?
        .trim()
        .to_owned())
}

pub(crate) async fn checkout_branch(worktree: &Path, branch: &str) -> anyhow::Result<()> {
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

pub(crate) async fn current_commit(worktree: &Path) -> anyhow::Result<String> {
    Ok(run_cmd(worktree, "git", &["rev-parse", "HEAD"])
        .await?
        .trim()
        .to_owned())
}

pub(crate) async fn branch_has_diff(worktree: &Path, base_ref: &str) -> anyhow::Result<bool> {
    Ok(!branch_diff(worktree, base_ref).await?.trim().is_empty())
}

pub(crate) async fn branch_diff(worktree: &Path, base_ref: &str) -> anyhow::Result<String> {
    run_cmd(worktree, "git", &["diff", "--find-renames", base_ref]).await
}

pub(crate) async fn commit_if_dirty(
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

pub(crate) async fn run_cmd(
    worktree: &Path,
    program: &str,
    args: &[&str],
) -> anyhow::Result<String> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::*;

    use crate::core::{CHECKPOINT_PATH, IMPLEMENTATION_PLAN_PATH};

    #[tokio::test]
    pub(crate) async fn commit_if_dirty_excludes_multiple_factory_state_paths() {
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
}
