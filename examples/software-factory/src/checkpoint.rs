use std::io::ErrorKind;
use std::path::Path;

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use tracing::info;

use crate::core::{
    CHECKPOINT_PATH, CandidateSet, IMPLEMENTATION_PLAN_PATH, IssueDoc, JudgeSelection,
    PullRequestDraft, RepoSlug,
};
use crate::github::GitHubPullRequest;

pub(crate) fn excluded_commit_paths(commit_impl_plan: bool) -> Vec<&'static str> {
    if commit_impl_plan {
        vec![CHECKPOINT_PATH]
    } else {
        vec![IMPLEMENTATION_PLAN_PATH, CHECKPOINT_PATH]
    }
}

pub(crate) const FACTORY_LOCAL_STATE_PATHS: [&str; 2] = [IMPLEMENTATION_PLAN_PATH, CHECKPOINT_PATH];

pub(crate) const CHECKPOINT_VERSION: u8 = 1;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct FactoryCheckpoint {
    pub(crate) version: u8,
    pub(crate) repo: String,
    pub(crate) base_branch: String,
    pub(crate) requested_issue: Option<u64>,
    pub(crate) commit_impl_plan: bool,
    pub(crate) issues: Option<Vec<IssueDoc>>,
    pub(crate) candidates: Option<CandidateSet>,
    pub(crate) selection: Option<JudgeSelection>,
    pub(crate) implementation_plan: Option<String>,
    pub(crate) branch: Option<String>,
    pub(crate) base_ref: Option<String>,
    pub(crate) implemented: bool,
    pub(crate) reviewed: bool,
    pub(crate) committed: bool,
    pub(crate) commit_sha: Option<String>,
    pub(crate) pushed: bool,
    pub(crate) pr_draft: Option<PullRequestDraft>,
    pub(crate) pr: Option<CheckpointPullRequest>,
    pub(crate) completed: bool,
}

impl FactoryCheckpoint {
    pub(crate) fn new(
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
pub(crate) struct CheckpointPullRequest {
    pub(crate) number: u64,
    pub(crate) html_url: String,
}

impl From<&GitHubPullRequest> for CheckpointPullRequest {
    fn from(pr: &GitHubPullRequest) -> Self {
        Self {
            number: pr.number,
            html_url: pr.html_url.clone(),
        }
    }
}

pub(crate) fn validate_checkpoint_for_run(
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

pub(crate) fn validate_checkpoint_stage_state(
    checkpoint: &FactoryCheckpoint,
) -> Result<(), String> {
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
pub(crate) async fn initialize_checkpoint(
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

pub(crate) async fn checkpoint_exists(path: &Path) -> anyhow::Result<bool> {
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

pub(crate) async fn read_checkpoint(path: &Path) -> anyhow::Result<FactoryCheckpoint> {
    let raw = tokio::fs::read_to_string(path)
        .await
        .with_context(|| format!("failed to read factory checkpoint {}", path.display()))?;
    serde_json::from_str(&raw)
        .with_context(|| format!("failed to parse factory checkpoint {}", path.display()))
}

pub(crate) async fn write_checkpoint(
    path: &Path,
    checkpoint: &FactoryCheckpoint,
) -> anyhow::Result<()> {
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

pub(crate) async fn remove_checkpoint_if_exists(path: &Path) -> anyhow::Result<()> {
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

pub(crate) async fn restore_implementation_plan(path: &Path, plan: &str) -> anyhow::Result<()> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::*;

    use crate::core::{CHECKPOINT_PATH, IMPLEMENTATION_PLAN_PATH, RepoSlug};

    #[test]
    pub(crate) fn excluded_commit_paths_cover_checkpoint_and_optional_plan() {
        assert_eq!(excluded_commit_paths(true), vec![CHECKPOINT_PATH]);
        assert_eq!(
            excluded_commit_paths(false),
            vec![IMPLEMENTATION_PLAN_PATH, CHECKPOINT_PATH]
        );
    }

    #[test]
    pub(crate) fn checkpoint_validation_covers_context_and_stage_errors() {
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

        pub(crate) struct Case {
            pub(crate) name: &'static str,
            pub(crate) mutate: Box<dyn Fn(&mut FactoryCheckpoint)>,
            pub(crate) expected: &'static str,
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
    pub(crate) async fn checkpoint_file_io_covers_write_read_remove_and_missing() {
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
    pub(crate) async fn initialize_checkpoint_covers_fresh_resume_existing_and_reset() {
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
}
