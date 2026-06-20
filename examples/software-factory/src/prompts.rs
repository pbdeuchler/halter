use std::io::ErrorKind;
use std::path::Path;

use anyhow::{Context, bail};
use halter::prelude::*;
use halter_protocol::{CacheScope, PromptSegment, PromptSegmentId, PromptSegmentKind, Volatility};
use sha2::{Digest, Sha256};
use tracing::warn;

use crate::core::{
    PROJECT_GUIDANCE_FILENAMES, PROJECT_GUIDANCE_MAX_BYTES, ProjectGuidanceDoc,
    format_project_system_prompt,
};

pub(crate) async fn read_project_system_prompt(worktree: &Path) -> anyhow::Result<Option<String>> {
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

pub(crate) const FACTORY_TURN_USER_MESSAGE: &str =
    "Execute the appended turn-specific instructions for this software factory stage.";

/// Shared closing instruction for stages whose response is parsed as JSON.
pub(crate) const JSON_ONLY_OUTPUT_RULE: &str = "Return ONLY the JSON object as your final message — no markdown code fences and no surrounding prose.";

/// Shared rule for coding stages that run cargo, whose builds exceed the
/// 30-second default shell timeout when no explicit timeout is supplied.
pub(crate) const CARGO_TIMEOUT_RULE: &str = "When running builds, tests, lints, or other checks through the shell tool, pass an explicit timeout_ms of at least 120000; these commands routinely exceed the 30-second default.";
pub(crate) const CODE_REVIEW_MAX_TURNS: u32 = 100;

#[derive(Debug, Clone, Copy)]
pub(crate) enum FactorySystemPrompt {
    General,
    Coding,
}

impl FactorySystemPrompt {
    pub(crate) fn segment(self) -> PromptSegment {
        match self {
            Self::General => prompts::default_system_prompt_segment(),
            Self::Coding => prompts::coding_agent_prompt_segment(),
        }
    }
}

pub(crate) fn session_init_with_appended_context(
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

pub(crate) fn project_guidance_prompt_segment(
    project_system_prompt: Option<&str>,
) -> Option<PromptSegment> {
    let text = project_system_prompt?.trim();
    if text.is_empty() {
        return None;
    }
    Some(append_prompt_segment(text))
}

pub(crate) fn turn_instructions_prompt_segment(
    turn_instructions: &str,
) -> anyhow::Result<PromptSegment> {
    let turn_instructions = turn_instructions.trim();
    if turn_instructions.is_empty() {
        bail!("failed to start agent turn: turn-specific instructions are empty");
    }
    Ok(append_prompt_segment(&format!(
        "# Turn-Specific Instructions\n\n{turn_instructions}"
    )))
}

pub(crate) fn append_prompt_segment(text: &str) -> PromptSegment {
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

pub(crate) fn hash_prompt_text(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReviewIteration {
    pub(crate) current: usize,
    pub(crate) max: usize,
}

impl ReviewIteration {
    pub(crate) fn is_first(self) -> bool {
        self.current == 1
    }

    pub(crate) fn is_final(self) -> bool {
        self.max > 0 && self.current == self.max
    }
}

pub(crate) fn code_review_prompt(base_ref: &str, diff: &str, iteration: ReviewIteration) -> String {
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

pub(crate) fn review_repair_prompt(
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

pub(crate) fn final_review_iteration_instruction(
    iteration: ReviewIteration,
    participant: &str,
) -> String {
    if !iteration.is_final() {
        return String::new();
    }
    match participant {
        "review" => "- This is the last code review iteration. Ensure every previous finding has been addressed and no new required fixes remain. If there are repeated issues or miscommunications, think deeply and take a different review approach before returning findings.\n\n".to_owned(),
        "implementation" => "- This is the last review-repair iteration. Ensure every finding is fully addressed. If there are repeated issues or miscommunications, think deeply and take a different implementation approach before finishing.\n".to_owned(),
        _ => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::{Path, PathBuf};

    use halter_protocol::{CacheScope, PromptSegmentKind, Volatility};

    use crate::core::PROJECT_GUIDANCE_MAX_BYTES;

    #[tokio::test]
    pub(crate) async fn read_project_system_prompt_returns_none_when_no_guidance_files_exist() {
        let dir = tempfile::tempdir().expect("tempdir");

        let prompt = read_project_system_prompt(dir.path())
            .await
            .expect("guidance read");

        assert_eq!(prompt, None);
    }

    #[tokio::test]
    pub(crate) async fn read_project_system_prompt_reads_top_level_guidance_in_fixed_order() {
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
    pub(crate) async fn read_project_system_prompt_rejects_oversized_guidance_file() {
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
    pub(crate) fn factory_system_prompt_segments_use_built_in_defaults() {
        let general = FactorySystemPrompt::General.segment();
        assert_eq!(general.text, prompts::default_system_prompt());
        assert_eq!(general.kind, PromptSegmentKind::System);

        let coding = FactorySystemPrompt::Coding.segment();
        assert_eq!(coding.text, prompts::default_coding_agent_prompt());
        assert_eq!(coding.kind, PromptSegmentKind::System);
    }

    #[test]
    pub(crate) fn project_guidance_prompt_segment_covers_empty_and_non_empty_inputs() {
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
    pub(crate) fn turn_instructions_prompt_segment_covers_non_empty_and_empty_inputs() {
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
    pub(crate) fn session_init_with_appended_context_uses_coding_prompt_and_append_segments() {
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
    pub(crate) fn session_init_with_appended_context_rejects_empty_turn_instructions() {
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
    pub(crate) fn session_init_with_appended_context_applies_optional_max_turns() {
        let init = session_init_with_appended_context(
            Path::new("/tmp/project"),
            FactorySystemPrompt::Coding,
            "review the branch",
            None,
            Some(CODE_REVIEW_MAX_TURNS),
        )
        .expect("session init");

        assert_eq!(init.max_turns, Some(CODE_REVIEW_MAX_TURNS));
    }

    #[test]
    pub(crate) fn code_review_prompt_covers_initial_follow_up_and_final_iterations() {
        pub(crate) struct Case {
            pub(crate) name: &'static str,
            pub(crate) iteration: ReviewIteration,
            pub(crate) want_initial: bool,
            pub(crate) want_follow_up: bool,
            pub(crate) want_final: bool,
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
    pub(crate) fn review_repair_prompt_covers_regular_and_final_iterations() {
        pub(crate) struct Case {
            pub(crate) name: &'static str,
            pub(crate) iteration: ReviewIteration,
            pub(crate) want_final: bool,
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
}
