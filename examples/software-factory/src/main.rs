// pattern: Imperative Shell

mod core;

use std::collections::{HashMap, HashSet};
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, bail};
use async_trait::async_trait;
use chrono::Utc;
use clap::Parser;
use futures::StreamExt;
use halter::prelude::*;
use halter_config::{
    ConfiguredProvider, ContextConfig, HarnessConfig, ModelConfig, ModelJudgeConfig, ModelSlot,
    ModelSlotRef, ModelsConfig, NetworkPolicyConfig, PolicyConfig, ProviderConfig, ProvidersConfig,
    ResourcesConfig, RuntimeConfig, SearchRoots, SessionsConfig, ShellPolicyConfig, ToolsConfig,
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
use tracing::{info, warn};

use crate::core::{
    CandidateSet, CodeReview, IMPLEMENTATION_PLAN_PATH, IssueComment, IssueDoc, JudgeSelection,
    ModelSpec, MonitorAction, PROJECT_GUIDANCE_FILENAMES, PROJECT_GUIDANCE_MAX_BYTES,
    ProjectGuidanceDoc, PullRequestDraft, RECENT_OPEN_ISSUE_LIMIT, RepoSlug, branch_name,
    candidate_set_for_issue, dirty_status_excluding, ensure_requested_issue_selection,
    format_project_system_prompt, is_maintainer_author_association, issue_corpus, issue_index,
    monitor_action, parse_github_remote_url, parse_issue_number_input, parse_json_response,
    selected_issue_numbers, validate_issue_number, validate_recent_issue_limit,
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt::init();

    let cli = Cli::parse();
    let requested_issue = cli
        .issue
        .map(validate_issue_number)
        .transpose()
        .map_err(anyhow::Error::msg)?;
    let cwd = std::env::current_dir().context("failed to read current directory")?;
    let worktree = git_worktree_root(&cwd).await?;
    let project_system_prompt = read_project_system_prompt(&worktree).await?;
    let repo = github_repo_from_git_remote(&worktree, &cli.remote).await?;
    let mut base_config = default_factory_config();
    add_worktree_policy(&mut base_config, &worktree);

    let github = GitHubClient::from_local_credentials(&worktree).await?;
    let repo_info = github.fetch_repo(&repo).await?;
    let base_branch = cli.base.clone().unwrap_or(repo_info.default_branch);

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
    let issue_cache = issue_cache_from_docs(&issues);
    let allowed_issue_numbers = issues
        .iter()
        .map(|issue| issue.number)
        .collect::<HashSet<_>>();
    let corpus = issue_corpus(&repo, &issues);
    let index = issue_index(&repo, &issues);
    let implementation_plan_path = worktree.join(IMPLEMENTATION_PLAN_PATH);
    if let Some(parent) = implementation_plan_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let glm = build_model_harness(
        &base_config,
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
        ModelSpec::parse(&cli.implementer_model).map_err(anyhow::Error::msg)?,
        ReasoningEffort::Xhigh,
        &worktree,
    )
    .await?;
    let reviewer = build_model_harness(
        &base_config,
        ModelSpec::parse(&cli.reviewer_model).map_err(anyhow::Error::msg)?,
        ReasoningEffort::Xhigh,
        &worktree,
    )
    .await?;
    let pr_writer = build_model_harness(
        &base_config,
        ModelSpec::parse(&cli.pr_model).map_err(anyhow::Error::msg)?,
        ReasoningEffort::Medium,
        &worktree,
    )
    .await?;

    let candidates = if let Some(number) = requested_issue {
        let issue = issues
            .iter()
            .find(|issue| issue.number == number)
            .with_context(|| format!("failed to find requested issue #{number} after fetch"))?;
        candidate_set_for_issue(issue)
    } else {
        propose_issue_candidates(&glm, &worktree, &repo, &corpus).await?
    };
    let selection = judge_issue_plan(
        &judge,
        &worktree,
        &repo,
        &index,
        &candidates,
        IMPLEMENTATION_PLAN_PATH,
        project_system_prompt.as_deref(),
    )
    .await?;
    let implementation_plan = read_implementation_plan(&implementation_plan_path).await?;
    let issue_numbers = selected_issue_numbers(&selection);
    if issue_numbers.is_empty() {
        bail!("failed to select work: judge did not return issue numbers");
    }
    ensure_requested_issue_selection(&selection, requested_issue).map_err(anyhow::Error::msg)?;
    ensure_selected_issues_are_open(&issues, &issue_numbers)?;

    prepare_branch(
        &worktree,
        &base_branch,
        cli.branch.as_deref(),
        cli.allow_dirty,
        &repo,
        &selection,
        Some(IMPLEMENTATION_PLAN_PATH),
    )
    .await?;
    let current_branch = current_branch(&worktree).await?;
    let base_ref = format!("origin/{base_branch}");

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

    if !branch_has_diff(&worktree, &base_ref).await? {
        bail!("failed to create PR: implementation produced no diff against {base_ref}");
    }
    commit_if_dirty(
        &worktree,
        &format!("Implement {}", selection.title),
        (!cli.commit_impl_plan).then_some(IMPLEMENTATION_PLAN_PATH),
    )
    .await?;
    run_cmd(&worktree, "git", &["push", "-u", "origin", &current_branch]).await?;

    let final_diff = branch_diff(&worktree, &base_ref).await?;
    let pr_draft = draft_pr(
        &pr_writer,
        &worktree,
        &repo,
        &selection,
        &implementation_plan,
        &issue_numbers,
        &final_diff,
    )
    .await?;
    let pr = github
        .create_pull_request(&repo, &current_branch, &base_branch, &pr_draft)
        .await?;

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
            excluded_commit_path: (!cli.commit_impl_plan).then_some(IMPLEMENTATION_PLAN_PATH),
            max_review_iterations: cli.max_review_iterations,
            poll_seconds: cli.poll_seconds,
        })
        .await?;
    }

    shutdown_all([&glm, &judge, &implementer, &reviewer, &pr_writer]).await;
    Ok(())
}

async fn canonicalize_existing(path: impl AsRef<Path>) -> anyhow::Result<PathBuf> {
    tokio::fs::canonicalize(path.as_ref())
        .await
        .with_context(|| format!("failed to canonicalize {}", path.as_ref().display()))
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
            subagent: Some(ModelSlot::Reference(ModelSlotRef::ModelJudge)),
            small: Some(model_config(
                ConfiguredProvider::OpenAi,
                "gpt-5.5",
                ReasoningEffort::Medium,
            )),
            model_judge: Some(ModelJudgeConfig {
                default: model_config(
                    ConfiguredProvider::OpenRouter,
                    "z-ai/glm-5.2",
                    ReasoningEffort::Xhigh,
                ),
                synthesis: model_config(
                    ConfiguredProvider::OpenRouter,
                    "google/gemma-4-31b-it",
                    ReasoningEffort::Medium,
                ),
                panel: vec![
                    model_config(
                        ConfiguredProvider::OpenRouter,
                        "minimax/minimax-m3",
                        ReasoningEffort::Xhigh,
                    ),
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
            return Ok((issue, "cache"));
        }

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
}

async fn build_judge_harness(
    config: &HarnessConfig,
    worktree: &Path,
    issue_tool: Arc<dyn Tool>,
) -> anyhow::Result<Halter> {
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
    Halter::builder()
        .with_config(config)
        .with_compiled_resources(resources)
        .with_tool(issue_tool)
        .build()
        .await
}

async fn build_model_harness(
    config: &HarnessConfig,
    model: ModelSpec,
    reasoning: ReasoningEffort,
    worktree: &Path,
) -> anyhow::Result<Halter> {
    let mut config = config.clone();
    add_worktree_policy(&mut config, worktree);
    let model = model.into_model_config(reasoning, Some(230_000), Some(16_384));
    config.models.default = Some(ModelSlot::Inline(model.clone()));
    config.models.small = Some(model.clone());
    config.models.subagent = Some(ModelSlot::Inline(model));
    let resources = ResourceCompiler::from_config(&config).compile().await?;
    Halter::from_compiled_resources(config, resources).await
}

async fn shutdown_all<'a>(harnesses: impl IntoIterator<Item = &'a Halter>) {
    for harness in harnesses {
        let _ = harness.shutdown(Duration::from_secs(10)).await;
    }
}

struct AgentRun {
    text: String,
}

fn session_init_with_project_guidance(
    worktree: &Path,
    project_system_prompt: Option<&str>,
) -> SessionInit {
    let mut init = SessionInit {
        working_dir: worktree.to_path_buf(),
        ..SessionInit::default()
    };
    if let Some(segment) = project_guidance_prompt_segment(project_system_prompt) {
        init.system_prompt_seed.push(segment);
    }
    init
}

fn project_guidance_prompt_segment(project_system_prompt: Option<&str>) -> Option<PromptSegment> {
    let text = project_system_prompt?.trim();
    if text.is_empty() {
        return None;
    }
    Some(PromptSegment {
        id: PromptSegmentId::new(),
        text: text.to_owned(),
        volatility: Volatility::SessionStable,
        cache_scope: CacheScope::PrefixCacheable,
        content_hash: hash_prompt_text(text),
        kind: PromptSegmentKind::System,
    })
}

fn hash_prompt_text(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

async fn run_agent(
    harness: &Halter,
    worktree: &Path,
    label: &str,
    prompt: impl Into<String>,
) -> anyhow::Result<AgentRun> {
    run_agent_with_system_prompt(harness, worktree, label, prompt, None).await
}

async fn run_agent_with_system_prompt(
    harness: &Halter,
    worktree: &Path,
    label: &str,
    prompt: impl Into<String>,
    project_system_prompt: Option<&str>,
) -> anyhow::Result<AgentRun> {
    info!(stage = label, "starting agent turn");
    let session = harness
        .new_session(session_init_with_project_guidance(
            worktree,
            project_system_prompt,
        ))
        .await?;
    let mut events = session.submit_turn(Turn::user(prompt.into())).await?;
    let mut latest_text = None;
    let mut delta_text = String::new();
    let mut usage = Usage::default();

    while let Some(event) = events.next().await {
        match event?.payload {
            SessionEventPayload::DeltaItem { delta } => {
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
            SessionEventPayload::TurnCompleted {
                usage: turn_usage, ..
            } => {
                usage = turn_usage;
                break;
            }
            SessionEventPayload::TurnFailed { error, .. } => {
                let _ = session.shutdown("turn_failed").await;
                bail!("agent stage {label} failed: {error}");
            }
            _ => {}
        }
    }

    session.shutdown(label).await?;
    let text = latest_text
        .filter(|text| !text.trim().is_empty())
        .or_else(|| (!delta_text.trim().is_empty()).then_some(delta_text))
        .with_context(|| format!("agent stage {label} produced no assistant text"))?;
    info!(
        stage = label,
        input_tokens = usage.input_tokens,
        output_tokens = usage.output_tokens,
        "completed agent turn"
    );
    Ok(AgentRun { text })
}

async fn propose_issue_candidates(
    glm: &Halter,
    worktree: &Path,
    repo: &RepoSlug,
    corpus: &str,
) -> anyhow::Result<CandidateSet> {
    let prompt = format!(
        r#"You are triaging open GitHub issues for {repo}.

Read the entire issue corpus and identify up to 3 excellent groups of open issues that can be solved holistically with one elegant pull request. Do not design the implementation. Prefer groups that:
- clearly share one root cause or cohesive code path
- can be solved without more maintainer input
- are likely contained enough for one PR
- avoid speculative feature work

If there are not enough good groups, choose individual open issues that are straightforward, contained, and do not need maintainer input.

Return ONLY JSON with this shape:
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
    let raw = run_agent(glm, worktree, "glm issue grouping", prompt)
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
    index: &str,
    candidates: &CandidateSet,
    implementation_plan_path: &str,
    project_system_prompt: Option<&str>,
) -> anyhow::Result<JudgeSelection> {
    let candidates_json = serde_json::to_string_pretty(candidates)?;
    let prompt = format!(
        r#"You are a model-judge planning group for a software factory workflow targeting {repo}.

The prior model proposed candidate issue groups. Pick exactly one candidate group or individual issue. Then write a detailed implementation plan to this file:
{implementation_plan_path}

Use the `github_issue` tool to fetch the full text for any issue you seriously consider before selecting it. The issue index below is intentionally compact and does not include issue bodies or comment text.

Selection rules:
- prefer the smallest cohesive PR with high confidence
- select only issues whose issue index state is open
- reject any candidate that needs maintainer clarification
- treat comments returned by `github_issue` as maintainer comments; comments from non-maintainers are intentionally omitted
- include concrete files/modules to inspect when inferable
- include happy-path and sad-path tests expected for the implementation
- include risks and verification commands

Implementation file rules:
- write the full detailed implementation plan as markdown to {implementation_plan_path}
- include selected issue numbers, scope, concrete steps, test plan, verification commands, and risks
- do not write code
- do not include the implementation plan text in the JSON response

Return ONLY JSON with this shape:
{{
  "title": "PR-sized implementation title",
  "issue_numbers": [123],
  "notes": "judge rationale and constraints"
}}

CANDIDATES:
{candidates_json}

ISSUE INDEX:
{index}
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
    excluded_dirty_path: Option<&str>,
) -> anyhow::Result<()> {
    if !allow_dirty && git_is_dirty(worktree, excluded_dirty_path).await? {
        bail!("failed to prepare branch: worktree is dirty; commit/stash or pass --allow-dirty");
    }
    run_cmd(worktree, "git", &["fetch", "origin", base_branch]).await?;
    let branch = requested_branch.map(ToOwned::to_owned).unwrap_or_else(|| {
        branch_name(
            repo,
            &selection.title,
            &Utc::now().format("%Y%m%d%H%M%S").to_string(),
        )
    });
    let base_ref = format!("origin/{base_branch}");
    run_cmd(worktree, "git", &["checkout", "-b", &branch, &base_ref]).await?;
    Ok(())
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
- If the plan proves impossible without maintainer input, stop and explain exactly why.

SELECTED ISSUES:
{selected}

IMPLEMENTATION PLAN:
"#,
    );
    let prompt = format!("{prompt}{implementation_plan}\n");
    run_agent_with_system_prompt(
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
    issue_corpus(&repo, &selected_issues)
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
        let review =
            review_diff(reviewer, worktree, base_ref, &diff, project_system_prompt).await?;
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
        let prompt = format!(
            r#"The code review for the current branch found issues. Fix every finding in the current worktree.

Rules:
- Do not create, switch, commit, push, or open branches/PRs.
- Keep the fix scoped to the implementation plan and review findings.
- Run focused tests or checks that cover the changed behavior.
- Return a concise summary and the verification commands you ran.

IMPLEMENTATION PLAN:
{implementation_plan}

REVIEW RESULT:
{review_json}
"#
        );
        run_agent_with_system_prompt(
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
    project_system_prompt: Option<&str>,
) -> anyhow::Result<CodeReview> {
    let prompt = format!(
        r#"You are reviewing a branch diff against {base_ref}.

Review stance:
- Prioritize correctness bugs, regressions, missing tests, unsafe behavior, and broken edge cases.
- Do not block on style nits unless they create real maintenance risk.
- Mark clean=true only when there are no required fixes.

Return ONLY JSON with this shape:
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
    );
    let raw = run_agent_with_system_prompt(
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

async fn draft_pr(
    pr_writer: &Halter,
    worktree: &Path,
    repo: &RepoSlug,
    selection: &JudgeSelection,
    implementation_plan: &str,
    issue_numbers: &[u64],
    diff: &str,
) -> anyhow::Result<PullRequestDraft> {
    let closing_lines = issue_numbers
        .iter()
        .map(|number| format!("Closes #{number}"))
        .collect::<Vec<_>>()
        .join("\n");
    let prompt = format!(
        r#"Create a detailed GitHub pull request for {repo}.

Return ONLY JSON with this shape:
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
    let raw = run_agent(pr_writer, worktree, "gemma pr draft", prompt)
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
    excluded_commit_path: Option<&'a str>,
    max_review_iterations: usize,
    poll_seconds: u64,
}

async fn monitor_pr(ctx: MonitorContext<'_>) -> anyhow::Result<()> {
    let mut seen = ctx
        .github
        .initial_seen_pr_activity(ctx.repo, ctx.pr_number)
        .await?;
    loop {
        let pr = ctx
            .github
            .fetch_pull_request(ctx.repo, ctx.pr_number)
            .await?;
        if pr.merged.unwrap_or(false) {
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
            tokio::time::sleep(Duration::from_secs(ctx.poll_seconds)).await;
            continue;
        }

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
                ctx.excluded_commit_path,
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
                ctx.excluded_commit_path,
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
    Ok(run_agent(glm, worktree, "glm plsfix refinement", prompt)
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

IMPLEMENTATION PLAN:
{implementation_plan}

FEEDBACK:
{feedback}
"#
    );
    run_agent_with_system_prompt(
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

async fn git_is_dirty(worktree: &Path, excluded_path: Option<&str>) -> anyhow::Result<bool> {
    let status = run_cmd(worktree, "git", &["status", "--porcelain"]).await?;
    Ok(dirty_status_excluding(&status, excluded_path))
}

async fn current_branch(worktree: &Path) -> anyhow::Result<String> {
    Ok(run_cmd(worktree, "git", &["branch", "--show-current"])
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
    excluded_path: Option<&str>,
) -> anyhow::Result<bool> {
    if !git_is_dirty(worktree, excluded_path).await? {
        return Ok(false);
    }
    run_cmd(worktree, "git", &["add", "-A"]).await?;
    if let Some(path) = excluded_path {
        run_cmd(worktree, "git", &["reset", "--", path]).await?;
    }
    if run_cmd(worktree, "git", &["diff", "--cached", "--quiet"])
        .await
        .is_ok()
    {
        return Ok(false);
    }
    run_cmd(worktree, "git", &["commit", "-m", message]).await?;
    Ok(true)
}

async fn run_cmd(worktree: &Path, program: &str, args: &[&str]) -> anyhow::Result<String> {
    let output = Command::new(program)
        .args(args)
        .current_dir(worktree)
        .output()
        .await
        .with_context(|| format!("failed to run command: {program} {}", args.join(" ")))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        bail!(
            "command failed: {program} {}\nstdout:\n{stdout}\nstderr:\n{stderr}",
            args.join(" ")
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

#[derive(Clone)]
struct GitHubClient {
    client: reqwest::Client,
}

impl GitHubClient {
    async fn from_local_credentials(worktree: &Path) -> anyhow::Result<Self> {
        let token = match github_token_from_env() {
            Some(token) => Some(token),
            None => github_token_from_gh_cli(worktree).await,
        };
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
        Ok(Self { client })
    }

    async fn get<T: for<'de> Deserialize<'de>>(&self, url: &str) -> anyhow::Result<T> {
        let response = self.client.get(url).send().await?;
        decode_response(response, "GET", url).await
    }

    async fn post_json<T: for<'de> Deserialize<'de>, B: Serialize>(
        &self,
        url: &str,
        body: &B,
    ) -> anyhow::Result<T> {
        let response = self.client.post(url).json(body).send().await?;
        decode_response(response, "POST", url).await
    }

    async fn fetch_repo(&self, repo: &RepoSlug) -> anyhow::Result<GitHubRepo> {
        self.get(&repo.api_base()).await
    }

    async fn fetch_recent_open_issues(
        &self,
        repo: &RepoSlug,
        limit: usize,
    ) -> anyhow::Result<Vec<IssueDoc>> {
        let limit = validate_recent_issue_limit(limit).map_err(anyhow::Error::msg)?;
        let mut url = reqwest::Url::parse("https://api.github.com/search/issues")
            .context("failed to build GitHub issue search URL")?;
        url.query_pairs_mut()
            .append_pair("q", &format!("repo:{repo} is:issue is:open"))
            .append_pair("sort", "created")
            .append_pair("order", "desc")
            .append_pair("per_page", &limit.to_string())
            .append_pair("page", "1");
        let search: GitHubIssueSearchResponse = self.get(url.as_str()).await?;
        let mut docs = Vec::with_capacity(search.items.len());
        for issue in search
            .items
            .into_iter()
            .filter(|issue| issue.pull_request.is_none())
            .filter(|issue| issue.state == "open")
        {
            let comments = self.fetch_issue_comments(repo, issue.number).await?;
            docs.push(issue.into_doc(comments));
        }
        Ok(docs)
    }

    async fn fetch_open_issue(&self, repo: &RepoSlug, number: u64) -> anyhow::Result<IssueDoc> {
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
        Ok(comments)
    }

    async fn create_pull_request(
        &self,
        repo: &RepoSlug,
        branch: &str,
        base: &str,
        draft: &PullRequestDraft,
    ) -> anyhow::Result<GitHubPullRequest> {
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

        Ok(monitor_action(review_feedback, issue_comment_bodies))
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
            let batch: Vec<T> = self.get(&url).await?;
            let count = batch.len();
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
        bail!("GitHub API request failed: {method} {url} returned {status}: {body}");
    }
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
    fn project_guidance_prompt_segment_covers_empty_and_non_empty_inputs() {
        assert!(project_guidance_prompt_segment(None).is_none());
        assert!(project_guidance_prompt_segment(Some(" \n")).is_none());

        let segment =
            project_guidance_prompt_segment(Some("Follow project rules.")).expect("segment");

        assert_eq!(segment.text, "Follow project rules.");
        assert_eq!(segment.kind, PromptSegmentKind::System);
        assert_eq!(segment.volatility, Volatility::SessionStable);
        assert_eq!(segment.cache_scope, CacheScope::PrefixCacheable);
        assert_eq!(segment.content_hash.len(), 64);
    }

    #[test]
    fn session_init_with_project_guidance_appends_system_prompt_seed() {
        let init = session_init_with_project_guidance(Path::new("/tmp/project"), Some("rules"));

        assert_eq!(init.working_dir, PathBuf::from("/tmp/project"));
        assert!(init.system_prompt_seed.len() >= 2);
        let last = init.system_prompt_seed.last().expect("last system segment");
        assert_eq!(last.text, "rules");
        assert_eq!(last.kind, PromptSegmentKind::System);
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
