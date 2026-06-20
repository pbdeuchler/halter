use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use crate::checkpoint::CheckpointPullRequest;
use crate::core::{CandidateSet, IssueCandidate, JudgeSelection, PullRequestDraft};
use crate::git::run_cmd;

pub(crate) fn sample_candidates() -> CandidateSet {
    CandidateSet {
        candidates: vec![IssueCandidate {
            title: "Fix issue".to_owned(),
            issue_numbers: vec![7],
            rationale: "selected".to_owned(),
            maintainer_input_risk: "low".to_owned(),
        }],
    }
}

pub(crate) fn sample_selection() -> JudgeSelection {
    JudgeSelection {
        title: "Fix issue".to_owned(),
        issue_numbers: vec![7],
        notes: "notes".to_owned(),
    }
}

pub(crate) fn sample_pr_draft() -> PullRequestDraft {
    PullRequestDraft {
        title: "Fix issue".to_owned(),
        body: "Body".to_owned(),
    }
}

pub(crate) fn sample_checkpoint_pr() -> CheckpointPullRequest {
    CheckpointPullRequest {
        number: 42,
        html_url: "https://github.com/pbdeuchler/halter/pull/42".to_owned(),
    }
}

pub(crate) fn unique_test_run_id(prefix: &str) -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system time after epoch")
        .as_nanos();
    format!("{prefix}-{}-{nanos}", std::process::id())
}

pub(crate) async fn remove_dir_if_exists(path: &Path) {
    match tokio::fs::remove_dir_all(path).await {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::NotFound => {}
        Err(error) => panic!("failed to remove {}: {error}", path.display()),
    }
}

pub(crate) async fn init_git_repo_with_origin() -> (tempfile::TempDir, PathBuf) {
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

pub(crate) async fn remove_git_worktree(source: &Path, worktree: &Path) {
    let worktree_arg = worktree.to_str().expect("utf-8 worktree path");
    run_cmd(
        source,
        "git",
        &["worktree", "remove", "--force", worktree_arg],
    )
    .await
    .expect("remove git worktree");
}
