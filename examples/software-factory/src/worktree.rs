use std::path::{Path, PathBuf};

use anyhow::{Context, bail};
use tracing::info;

use crate::core::{FACTORY_WORKTREE_TMP_DIR, RepoSlug, factory_worktree_dir_name};
use crate::git::run_cmd;

pub(crate) async fn canonicalize_existing(path: impl AsRef<Path>) -> anyhow::Result<PathBuf> {
    tokio::fs::canonicalize(path.as_ref())
        .await
        .with_context(|| format!("failed to canonicalize {}", path.as_ref().display()))
}

pub(crate) async fn resolve_execution_worktree(
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

pub(crate) fn factory_worktree_tmp_root() -> PathBuf {
    PathBuf::from("/tmp").join(FACTORY_WORKTREE_TMP_DIR)
}

pub(crate) fn path_is_factory_tmp_worktree(path: &Path) -> bool {
    path.starts_with(factory_worktree_tmp_root())
        || path.starts_with(Path::new("/private/tmp").join(FACTORY_WORKTREE_TMP_DIR))
}

pub(crate) async fn create_factory_worktree(
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

pub(crate) async fn git_worktree_root(cwd: &Path) -> anyhow::Result<PathBuf> {
    let root = run_cmd(cwd, "git", &["rev-parse", "--show-toplevel"])
        .await
        .context("failed to locate git worktree root; run software-factory inside a git repo")?;
    canonicalize_existing(root.trim()).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::*;

    use std::path::Path;

    use crate::core::{RepoSlug, factory_worktree_dir_name};

    #[tokio::test]
    pub(crate) async fn resolve_execution_worktree_covers_current_tmp_resume_and_bad_resume() {
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
    pub(crate) async fn create_factory_worktree_rejects_existing_path() {
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
    pub(crate) async fn create_factory_worktree_adds_detached_tmp_worktree() {
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
}
