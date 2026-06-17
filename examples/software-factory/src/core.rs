// pattern: Functional Core

use std::fmt;

use halter_config::{ConfiguredProvider, ModelConfig};
use halter_protocol::ReasoningEffort;
use serde::{Deserialize, Serialize, de::DeserializeOwned};
use serde_json::Value;

pub const IMPLEMENTATION_PLAN_PATH: &str = ".halter/software-factory/implementation-plan.md";
pub const CHECKPOINT_PATH: &str = ".halter/software-factory/checkpoint.json";
pub const RECENT_OPEN_ISSUE_LIMIT: usize = 100;
pub const PROJECT_GUIDANCE_FILENAMES: [&str; 3] = ["CLAUDE.md", "AGENTS.md", "SOUL.md"];
pub const PROJECT_GUIDANCE_MAX_BYTES: u64 = 1_048_576;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RepoSlug {
    pub owner: String,
    pub name: String,
}

impl RepoSlug {
    pub fn parse(raw: &str) -> Result<Self, String> {
        let trimmed = raw.trim();
        let (owner, name) = trimmed
            .split_once('/')
            .ok_or_else(|| "repo must use owner/name form".to_owned())?;
        if name.contains('/') {
            return Err("repo must use owner/name form".to_owned());
        }
        let owner = owner.trim();
        let name = name.trim();
        if owner.is_empty() || name.is_empty() {
            return Err("repo owner and name must be non-empty".to_owned());
        }
        if owner.chars().any(char::is_whitespace) || name.chars().any(char::is_whitespace) {
            return Err("repo owner and name must not contain whitespace".to_owned());
        }
        Ok(Self {
            owner: owner.to_owned(),
            name: name.to_owned(),
        })
    }

    pub fn api_base(&self) -> String {
        format!("https://api.github.com/repos/{}/{}", self.owner, self.name)
    }
}

impl fmt::Display for RepoSlug {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.owner, self.name)
    }
}

pub fn parse_github_remote_url(raw: &str) -> Result<RepoSlug, String> {
    let trimmed = raw.trim().trim_end_matches('/');
    let path = if let Some(path) = trimmed.strip_prefix("git@github.com:") {
        path
    } else if let Some(path) = trimmed.strip_prefix("ssh://git@github.com/") {
        path
    } else if let Some(path) = trimmed.strip_prefix("https://github.com/") {
        path
    } else if let Some(path) = trimmed.strip_prefix("http://github.com/") {
        path
    } else {
        return Err("git remote must point at github.com".to_owned());
    };
    let without_git_suffix = path.strip_suffix(".git").unwrap_or(path);
    RepoSlug::parse(without_git_suffix)
}

pub fn is_maintainer_author_association(value: &str) -> bool {
    matches!(
        value,
        "OWNER" | "MEMBER" | "COLLABORATOR" | "owner" | "member" | "collaborator"
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSpec {
    pub provider: ConfiguredProvider,
    pub model: String,
}

impl ModelSpec {
    pub fn parse(raw: &str) -> Result<Self, String> {
        let trimmed = raw.trim();
        let (provider, model) = trimmed
            .split_once('/')
            .ok_or_else(|| "model must use provider/model form".to_owned())?;
        let provider = match provider {
            "openai" => ConfiguredProvider::OpenAi,
            "anthropic" => ConfiguredProvider::Anthropic,
            "openrouter" => ConfiguredProvider::OpenRouter,
            _ => {
                return Err(
                    "model provider must be one of openai, anthropic, or openrouter".to_owned(),
                );
            }
        };
        if model.trim().is_empty() {
            return Err("model name must be non-empty".to_owned());
        }
        Ok(Self {
            provider,
            model: model.to_owned(),
        })
    }

    pub fn into_model_config(
        self,
        reasoning: ReasoningEffort,
        max_input_tokens: Option<u32>,
        max_output_tokens: Option<u32>,
    ) -> ModelConfig {
        ModelConfig {
            provider: self.provider,
            model: self.model,
            max_input_tokens,
            max_output_tokens,
            reasoning: Some(reasoning),
            tokens_per_minute: Some(500_000),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IssueComment {
    pub author: String,
    pub body: String,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IssueDoc {
    pub number: u64,
    pub state: String,
    pub title: String,
    pub body: String,
    pub labels: Vec<String>,
    pub author: String,
    pub url: String,
    pub comments: Vec<IssueComment>,
}

pub fn issue_corpus(repo: &RepoSlug, issues: &[IssueDoc]) -> String {
    let mut out = String::new();
    out.push_str(&format!(
        "# Open GitHub issues for {repo}\n\nOnly maintainer comments are included.\n\n"
    ));
    for issue in issues {
        out.push_str(&format!(
            "## #{}: {}\nURL: {}\nState: {}\nAuthor: {}\nLabels: {}\n\n{}\n",
            issue.number,
            issue.title,
            issue.url,
            issue.state,
            issue.author,
            if issue.labels.is_empty() {
                "(none)".to_owned()
            } else {
                issue.labels.join(", ")
            },
            empty_as_placeholder(&issue.body)
        ));
        if !issue.comments.is_empty() {
            out.push_str("\nComments:\n");
            for comment in &issue.comments {
                out.push_str(&format!(
                    "- {} at {}:\n{}\n",
                    comment.author,
                    comment.created_at,
                    indent(&comment.body)
                ));
            }
        }
        out.push('\n');
    }
    out
}

pub fn issue_index(repo: &RepoSlug, issues: &[IssueDoc]) -> String {
    let mut out = String::new();
    out.push_str(&format!("# Open GitHub issue index for {repo}\n\n"));
    for issue in issues {
        out.push_str(&format!(
            "- #{} [{}] {} | labels: {} | maintainer_comments: {}\n",
            issue.number,
            issue.state,
            issue.title,
            if issue.labels.is_empty() {
                "(none)".to_owned()
            } else {
                issue.labels.join(", ")
            },
            issue.comments.len()
        ));
    }
    out
}

fn empty_as_placeholder(value: &str) -> &str {
    if value.trim().is_empty() {
        "(no body)"
    } else {
        value
    }
}

fn indent(value: &str) -> String {
    value
        .lines()
        .map(|line| format!("  {line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IssueCandidate {
    pub title: String,
    pub issue_numbers: Vec<u64>,
    pub rationale: String,
    #[serde(default)]
    pub maintainer_input_risk: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CandidateSet {
    pub candidates: Vec<IssueCandidate>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct JudgeSelection {
    pub title: String,
    pub issue_numbers: Vec<u64>,
    #[serde(default)]
    pub notes: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReviewFinding {
    pub severity: String,
    pub file: String,
    #[serde(default)]
    pub line: Option<u64>,
    pub problem: String,
    pub recommendation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CodeReview {
    pub clean: bool,
    pub summary: String,
    #[serde(default)]
    pub findings: Vec<ReviewFinding>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PullRequestDraft {
    pub title: String,
    pub body: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProjectGuidanceDoc {
    pub filename: String,
    pub text: String,
}

pub fn format_project_system_prompt(docs: &[ProjectGuidanceDoc]) -> Option<String> {
    let non_empty_docs = docs
        .iter()
        .filter(|doc| !doc.text.trim().is_empty())
        .collect::<Vec<_>>();
    if non_empty_docs.is_empty() {
        return None;
    }

    let mut out = String::from(
        "# Project Guidance\n\nThe following top-level project files are authoritative for this repository. Follow them while designing, implementing, and reviewing changes.\n",
    );
    for doc in non_empty_docs {
        out.push_str(&format!("\n## {}\n\n{}\n", doc.filename, doc.text.trim()));
    }
    Some(out)
}

pub fn parse_json_response<T: DeserializeOwned>(raw: &str) -> Result<T, String> {
    let json = json_slice(raw).ok_or_else(|| "agent response did not contain JSON".to_owned())?;
    serde_json::from_str(json).map_err(|error| format!("agent response JSON was invalid: {error}"))
}

pub fn json_slice(raw: &str) -> Option<&str> {
    let start = raw.find(['{', '['])?;
    let end = raw.rfind(['}', ']'])?;
    (end >= start).then_some(raw[start..=end].trim())
}

pub fn selected_issue_numbers(selection: &JudgeSelection) -> Vec<u64> {
    let mut numbers = selection.issue_numbers.clone();
    numbers.sort_unstable();
    numbers.dedup();
    numbers
}

pub fn validate_issue_number(number: u64) -> Result<u64, String> {
    if number == 0 {
        return Err("issue number must be a positive integer".to_owned());
    }
    Ok(number)
}

pub fn validate_recent_issue_limit(limit: usize) -> Result<usize, String> {
    if !(1..=100).contains(&limit) {
        return Err("recent issue limit must be between 1 and 100".to_owned());
    }
    Ok(limit)
}

pub fn candidate_set_for_issue(issue: &IssueDoc) -> CandidateSet {
    CandidateSet {
        candidates: vec![IssueCandidate {
            title: format!("Issue #{}: {}", issue.number, issue.title),
            issue_numbers: vec![issue.number],
            rationale: format!("The user explicitly requested issue #{}.", issue.number),
            maintainer_input_risk:
                "Judge must reject this issue if its text indicates maintainer input is needed."
                    .to_owned(),
        }],
    }
}

pub fn ensure_requested_issue_selection(
    selection: &JudgeSelection,
    requested_issue: Option<u64>,
) -> Result<(), String> {
    let Some(requested_issue) = requested_issue else {
        return Ok(());
    };
    let selected = selected_issue_numbers(selection);
    if selected == [requested_issue] {
        return Ok(());
    }
    let selected = selected
        .into_iter()
        .map(|number| format!("#{number}"))
        .collect::<Vec<_>>()
        .join(", ");
    Err(format!(
        "judge selected {selected} but --issue requested #{requested_issue}"
    ))
}

pub fn is_plsfix_comment(body: &str) -> bool {
    body.trim_start().starts_with("/plsfix")
}

pub fn strip_plsfix_prefix(body: &str) -> &str {
    body.trim_start()
        .strip_prefix("/plsfix")
        .map(str::trim_start)
        .unwrap_or_else(|| body.trim_start())
}

pub fn sanitize_branch_component(raw: &str) -> String {
    let mut out = String::new();
    let mut previous_dash = false;
    for ch in raw.chars() {
        let normalized = if ch.is_ascii_alphanumeric() {
            Some(ch.to_ascii_lowercase())
        } else if ch == '-' || ch == '_' || ch == '/' || ch.is_whitespace() {
            Some('-')
        } else {
            None
        };
        if let Some(ch) = normalized {
            if ch == '-' {
                if !previous_dash && !out.is_empty() {
                    out.push(ch);
                    previous_dash = true;
                }
            } else {
                out.push(ch);
                previous_dash = false;
            }
        }
        if out.len() >= 64 {
            break;
        }
    }
    out.trim_matches('-').to_owned()
}

pub fn branch_name(repo: &RepoSlug, title: &str, run_id: &str) -> String {
    let repo = sanitize_branch_component(&repo.name);
    let title = sanitize_branch_component(title);
    let run_id = sanitize_branch_component(run_id);
    let title = if title.is_empty() {
        "issues".to_owned()
    } else {
        title
    };
    format!("halter-factory/{repo}-{run_id}-{title}")
}

pub fn parse_issue_number_input(input: &Value) -> Result<u64, String> {
    let number = input
        .get("number")
        .and_then(Value::as_u64)
        .ok_or_else(|| "field 'number' must be a positive integer".to_owned())?;
    validate_issue_number(number)
        .map_err(|_| "field 'number' must be a positive integer".to_owned())
}

pub fn dirty_status_excluding(status: &str, excluded_path: Option<&str>) -> bool {
    status
        .lines()
        .filter(|line| !line.trim().is_empty())
        .any(|line| !excluded_path.is_some_and(|path| status_line_mentions_path(line, path)))
}

fn status_line_mentions_path(line: &str, path: &str) -> bool {
    let trimmed = line.trim();
    let normalized_path = path.trim_start_matches("./");
    trimmed.ends_with(normalized_path)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MonitorAction {
    pub code_review_feedback: Vec<String>,
    pub plsfix_comments: Vec<String>,
}

impl MonitorAction {
    pub fn is_empty(&self) -> bool {
        self.code_review_feedback.is_empty() && self.plsfix_comments.is_empty()
    }
}

pub fn monitor_action(review_feedback: Vec<String>, issue_comments: Vec<String>) -> MonitorAction {
    let plsfix_comments = issue_comments
        .into_iter()
        .filter(|body| is_plsfix_comment(body))
        .map(|body| strip_plsfix_prefix(&body).to_owned())
        .filter(|body| !body.trim().is_empty())
        .collect();
    MonitorAction {
        code_review_feedback: review_feedback
            .into_iter()
            .filter(|body| !body.trim().is_empty())
            .collect(),
        plsfix_comments,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn repo_slug_parse_covers_valid_and_invalid_inputs() {
        let valid = RepoSlug::parse("pbdeuchler/halter").expect("valid repo");
        assert_eq!(
            valid,
            RepoSlug {
                owner: "pbdeuchler".to_owned(),
                name: "halter".to_owned()
            }
        );

        for raw in [
            "",
            "halter",
            "/halter",
            "pbdeuchler/",
            "pbdeuchler/hal ter",
            "a/b/c",
        ] {
            assert!(
                RepoSlug::parse(raw).is_err(),
                "{raw:?} should not parse as repo slug"
            );
        }
    }

    #[test]
    fn model_spec_parse_preserves_openrouter_model_slashes() {
        let spec = ModelSpec::parse("openrouter/moonshotai/kimi-k2.7-code").expect("valid model");
        assert_eq!(spec.provider, ConfiguredProvider::OpenRouter);
        assert_eq!(spec.model, "moonshotai/kimi-k2.7-code");

        let openai = ModelSpec::parse("openai/gpt-5.5").expect("valid openai model");
        assert_eq!(openai.provider, ConfiguredProvider::OpenAi);
        assert_eq!(openai.model, "gpt-5.5");

        for raw in ["", "gpt-5.5", "bogus/model", "openai/"] {
            assert!(
                ModelSpec::parse(raw).is_err(),
                "{raw:?} should not parse as a model spec"
            );
        }
    }

    #[test]
    fn parse_github_remote_url_accepts_common_github_forms() {
        let cases = [
            (
                "https://github.com/pbdeuchler/halter.git",
                "pbdeuchler",
                "halter",
            ),
            (
                "https://github.com/pbdeuchler/halter",
                "pbdeuchler",
                "halter",
            ),
            (
                "git@github.com:pbdeuchler/halter.git",
                "pbdeuchler",
                "halter",
            ),
            (
                "ssh://git@github.com/pbdeuchler/halter.git",
                "pbdeuchler",
                "halter",
            ),
        ];

        for (raw, owner, name) in cases {
            let got = parse_github_remote_url(raw).expect("valid GitHub remote");
            assert_eq!(got.owner, owner);
            assert_eq!(got.name, name);
        }
    }

    #[test]
    fn parse_github_remote_url_rejects_non_github_or_malformed_remotes() {
        for raw in [
            "",
            "https://gitlab.com/pbdeuchler/halter.git",
            "git@github.com:pbdeuchler",
            "https://github.com/pbdeuchler/hal ter.git",
        ] {
            assert!(
                parse_github_remote_url(raw).is_err(),
                "{raw:?} should not parse as a GitHub remote"
            );
        }
    }

    #[test]
    fn maintainer_author_association_matches_github_maintainer_roles() {
        for value in ["OWNER", "MEMBER", "COLLABORATOR", "owner"] {
            assert!(
                is_maintainer_author_association(value),
                "{value:?} should count as maintainer"
            );
        }

        for value in ["CONTRIBUTOR", "FIRST_TIMER", "NONE", ""] {
            assert!(
                !is_maintainer_author_association(value),
                "{value:?} should not count as maintainer"
            );
        }
    }

    #[test]
    fn json_slice_accepts_fenced_and_rejects_missing_json() {
        assert_eq!(
            json_slice("```json\n{\"clean\":true}\n```"),
            Some("{\"clean\":true}")
        );
        assert_eq!(json_slice("prefix [1,2] suffix"), Some("[1,2]"));
        assert_eq!(json_slice("no json here"), None);
    }

    #[test]
    fn parse_json_response_covers_success_and_invalid_json() {
        let review: CodeReview =
            parse_json_response("{\"clean\":true,\"summary\":\"ok\"}").expect("valid review");
        assert!(review.clean);
        assert_eq!(review.summary, "ok");

        assert!(parse_json_response::<CodeReview>("no json here").is_err());
        assert!(parse_json_response::<CodeReview>("{\"clean\":true}").is_err());
    }

    #[test]
    fn selected_issue_numbers_are_sorted_and_deduped() {
        let selection = JudgeSelection {
            title: "fix things".to_owned(),
            issue_numbers: vec![42, 7, 42],
            notes: String::new(),
        };

        assert_eq!(selected_issue_numbers(&selection), vec![7, 42]);
    }

    #[test]
    fn format_project_system_prompt_covers_empty_and_whitespace_docs() {
        assert_eq!(format_project_system_prompt(&[]), None);
        assert_eq!(
            format_project_system_prompt(&[ProjectGuidanceDoc {
                filename: "CLAUDE.md".to_owned(),
                text: "  \n".to_owned(),
            }]),
            None
        );
    }

    #[test]
    fn format_project_system_prompt_preserves_guidance_in_order() {
        let prompt = format_project_system_prompt(&[
            ProjectGuidanceDoc {
                filename: "CLAUDE.md".to_owned(),
                text: "Use cargo test.".to_owned(),
            },
            ProjectGuidanceDoc {
                filename: "AGENTS.md".to_owned(),
                text: "\nPrefer small diffs.\n".to_owned(),
            },
        ])
        .expect("guidance prompt");

        let claude = prompt.find("## CLAUDE.md").expect("claude section");
        let agents = prompt.find("## AGENTS.md").expect("agents section");
        assert!(claude < agents);
        assert!(prompt.contains("Use cargo test."));
        assert!(prompt.contains("Prefer small diffs."));
        assert!(prompt.contains("authoritative for this repository"));
    }

    #[test]
    fn issue_number_validation_covers_success_and_error_cases() {
        assert_eq!(validate_issue_number(1).expect("valid issue"), 1);
        assert_eq!(validate_issue_number(42).expect("valid issue"), 42);
        assert!(validate_issue_number(0).is_err());
    }

    #[test]
    fn recent_issue_limit_validation_covers_boundaries() {
        for limit in [1, RECENT_OPEN_ISSUE_LIMIT] {
            assert_eq!(
                validate_recent_issue_limit(limit).expect("valid limit"),
                limit
            );
        }

        for limit in [0, RECENT_OPEN_ISSUE_LIMIT + 1] {
            assert!(
                validate_recent_issue_limit(limit).is_err(),
                "{limit} should not be accepted as an issue limit"
            );
        }
    }

    #[test]
    fn candidate_set_for_issue_builds_single_forced_candidate() {
        let issue = IssueDoc {
            number: 13,
            state: "open".to_owned(),
            title: "make logs quieter".to_owned(),
            body: "too much output".to_owned(),
            labels: vec!["bug".to_owned()],
            author: "octo".to_owned(),
            url: "https://example.test/issues/13".to_owned(),
            comments: vec![],
        };

        let candidates = candidate_set_for_issue(&issue);

        assert_eq!(candidates.candidates.len(), 1);
        assert_eq!(candidates.candidates[0].issue_numbers, vec![13]);
        assert!(candidates.candidates[0].title.contains("#13"));
        assert!(
            candidates.candidates[0]
                .rationale
                .contains("explicitly requested")
        );
    }

    #[test]
    fn requested_issue_selection_guard_allows_exact_match_only() {
        let matching = JudgeSelection {
            title: "fix requested issue".to_owned(),
            issue_numbers: vec![7, 7],
            notes: String::new(),
        };
        assert!(ensure_requested_issue_selection(&matching, None).is_ok());
        assert!(ensure_requested_issue_selection(&matching, Some(7)).is_ok());

        for selection in [
            JudgeSelection {
                title: "wrong".to_owned(),
                issue_numbers: vec![8],
                notes: String::new(),
            },
            JudgeSelection {
                title: "extra".to_owned(),
                issue_numbers: vec![7, 8],
                notes: String::new(),
            },
            JudgeSelection {
                title: "empty".to_owned(),
                issue_numbers: vec![],
                notes: String::new(),
            },
        ] {
            assert!(
                ensure_requested_issue_selection(&selection, Some(7)).is_err(),
                "{selection:?} should not satisfy --issue 7"
            );
        }
    }

    #[test]
    fn issue_corpus_includes_state_and_empty_body_placeholder() {
        let repo = RepoSlug::parse("pbdeuchler/halter").expect("valid repo");
        let corpus = issue_corpus(
            &repo,
            &[IssueDoc {
                number: 9,
                state: "closed".to_owned(),
                title: "old behavior".to_owned(),
                body: String::new(),
                labels: vec!["bug".to_owned()],
                author: "octo".to_owned(),
                url: "https://example.test/issues/9".to_owned(),
                comments: vec![],
            }],
        );

        assert!(corpus.contains("State: closed"));
        assert!(corpus.contains("(no body)"));
        assert!(corpus.contains("Only maintainer comments are included."));
    }

    #[test]
    fn issue_index_excludes_body_and_comment_text() {
        let repo = RepoSlug::parse("pbdeuchler/halter").expect("valid repo");
        let index = issue_index(
            &repo,
            &[IssueDoc {
                number: 9,
                state: "open".to_owned(),
                title: "old behavior".to_owned(),
                body: "secret body details".to_owned(),
                labels: vec!["bug".to_owned()],
                author: "octo".to_owned(),
                url: "https://example.test/issues/9".to_owned(),
                comments: vec![IssueComment {
                    author: "maintainer".to_owned(),
                    body: "secret comment details".to_owned(),
                    created_at: "2026-06-17T00:00:00Z".to_owned(),
                }],
            }],
        );

        assert!(index.contains("#9 [open] old behavior"));
        assert!(index.contains("maintainer_comments: 1"));
        assert!(!index.contains("secret body details"));
        assert!(!index.contains("secret comment details"));
    }

    #[test]
    fn plsfix_detection_uses_trimmed_prefix_only() {
        assert!(is_plsfix_comment("  /plsfix please update docs"));
        assert_eq!(
            strip_plsfix_prefix("  /plsfix please update docs"),
            "please update docs"
        );
        assert!(!is_plsfix_comment("please /plsfix this"));
    }

    #[test]
    fn branch_component_sanitizes_and_collapses_separators() {
        assert_eq!(
            sanitize_branch_component("Fix: API / CLI_state!"),
            "fix-api-cli-state"
        );
        assert_eq!(sanitize_branch_component("..."), "");

        let repo = RepoSlug::parse("pbdeuchler/halter").expect("valid repo");
        assert_eq!(
            branch_name(&repo, "Fix: API / CLI_state!", "20260617"),
            "halter-factory/halter-20260617-fix-api-cli-state"
        );
    }

    #[test]
    fn parse_issue_number_input_covers_success_and_error_cases() {
        let valid = serde_json::json!({ "number": 42 });
        assert_eq!(parse_issue_number_input(&valid).expect("valid number"), 42);

        for input in [
            serde_json::json!({}),
            serde_json::json!({ "number": 0 }),
            serde_json::json!({ "number": "42" }),
        ] {
            assert!(
                parse_issue_number_input(&input).is_err(),
                "{input:?} should not parse as an issue number"
            );
        }
    }

    #[test]
    fn dirty_status_excluding_ignores_only_the_requested_file() {
        assert!(!dirty_status_excluding("", None));
        assert!(dirty_status_excluding(
            "?? .halter/software-factory/implementation-plan.md\n",
            None
        ));
        assert!(!dirty_status_excluding(
            "?? .halter/software-factory/implementation-plan.md\n",
            Some(IMPLEMENTATION_PLAN_PATH)
        ));
        assert!(dirty_status_excluding(
            "?? .halter/software-factory/implementation-plan.md\n M src/lib.rs\n",
            Some(IMPLEMENTATION_PLAN_PATH)
        ));
    }

    #[test]
    fn monitor_action_routes_review_feedback_and_plsfix_comments() {
        let action = monitor_action(
            vec!["review requested changes".to_owned(), "  ".to_owned()],
            vec![
                "/plsfix handle windows".to_owned(),
                "ordinary comment".to_owned(),
            ],
        );

        assert_eq!(
            action.code_review_feedback,
            vec!["review requested changes"]
        );
        assert_eq!(action.plsfix_comments, vec!["handle windows"]);
        assert!(!action.is_empty());

        assert!(monitor_action(vec![], vec![]).is_empty());
    }
}
