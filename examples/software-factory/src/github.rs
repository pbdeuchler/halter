use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, bail};
use async_trait::async_trait;
use halter_protocol::{ToolCapabilities, ToolConcurrency, ToolName, ToolResult, ToolSpec};
use halter_tools::{Tool, ToolContext};
use reqwest::header::{ACCEPT, AUTHORIZATION, HeaderMap, HeaderName, HeaderValue, USER_AGENT};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::{process::Command, sync::RwLock};
use tracing::{debug, info, warn};

use crate::agent::single_line_preview;
use crate::core::{
    IssueComment, IssueDoc, MonitorAction, PullRequestDraft, RepoSlug,
    is_maintainer_author_association, monitor_action, parse_github_remote_url,
    parse_issue_number_input, validate_recent_issue_limit,
};
use crate::git::run_cmd;

pub(crate) async fn github_repo_from_git_remote(
    worktree: &Path,
    remote: &str,
) -> anyhow::Result<RepoSlug> {
    let remote_url = run_cmd(
        worktree,
        "git",
        &["config", "--get", &format!("remote.{remote}.url")],
    )
    .await
    .with_context(|| format!("failed to read git remote URL for remote '{remote}'"))?;
    parse_github_remote_url(&remote_url).map_err(anyhow::Error::msg)
}

pub(crate) type IssueCache = Arc<RwLock<HashMap<u64, IssueDoc>>>;

pub(crate) fn issue_cache_from_docs(issues: &[IssueDoc]) -> IssueCache {
    Arc::new(RwLock::new(
        issues
            .iter()
            .cloned()
            .map(|issue| (issue.number, issue))
            .collect(),
    ))
}

#[derive(Clone)]
pub(crate) struct GitHubIssueTool {
    pub(crate) github: GitHubClient,
    pub(crate) repo: RepoSlug,
    pub(crate) cache: IssueCache,
    pub(crate) allowed_numbers: HashSet<u64>,
}

impl GitHubIssueTool {
    pub(crate) fn new(
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

    pub(crate) async fn cached_or_fetch(
        &self,
        number: u64,
    ) -> anyhow::Result<(IssueDoc, &'static str)> {
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
#[derive(Clone)]
pub(crate) struct GitHubClient {
    pub(crate) client: reqwest::Client,
}

impl GitHubClient {
    pub(crate) async fn from_local_credentials(worktree: &Path) -> anyhow::Result<Self> {
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

    pub(crate) async fn get<T: for<'de> Deserialize<'de>>(&self, url: &str) -> anyhow::Result<T> {
        debug!(method = "GET", url, "GitHub request");
        let response = self.client.get(url).send().await?;
        decode_response(response, "GET", url).await
    }

    pub(crate) async fn post_json<T: for<'de> Deserialize<'de>, B: Serialize>(
        &self,
        url: &str,
        body: &B,
    ) -> anyhow::Result<T> {
        info!(method = "POST", url, "GitHub request");
        let response = self.client.post(url).json(body).send().await?;
        decode_response(response, "POST", url).await
    }

    pub(crate) async fn fetch_repo(&self, repo: &RepoSlug) -> anyhow::Result<GitHubRepo> {
        info!(repo = %repo, "fetching GitHub repository metadata");
        self.get(&repo.api_base()).await
    }

    pub(crate) async fn fetch_recent_open_issues(
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

    pub(crate) async fn fetch_open_issue(
        &self,
        repo: &RepoSlug,
        number: u64,
    ) -> anyhow::Result<IssueDoc> {
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

    pub(crate) async fn fetch_issue_comments(
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

    pub(crate) async fn create_pull_request(
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

    pub(crate) async fn fetch_pull_request(
        &self,
        repo: &RepoSlug,
        number: u64,
    ) -> anyhow::Result<GitHubPullRequest> {
        debug!(repo = %repo, pr_number = number, "fetching pull request");
        self.get(&format!("{}/pulls/{number}", repo.api_base()))
            .await
    }

    pub(crate) async fn initial_seen_pr_activity(
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

    pub(crate) async fn fetch_new_pr_activity(
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

    pub(crate) async fn get_paginated<T: for<'de> Deserialize<'de>>(
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

pub(crate) fn github_token_from_env() -> Option<String> {
    std::env::var("GITHUB_TOKEN")
        .ok()
        .or_else(|| std::env::var("GH_TOKEN").ok())
        .map(|token| token.trim().to_owned())
        .filter(|token| !token.is_empty())
}

pub(crate) async fn github_token_from_gh_cli(worktree: &Path) -> Option<String> {
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

pub(crate) async fn decode_response<T: for<'de> Deserialize<'de>>(
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
pub(crate) struct GitHubRepo {
    pub(crate) default_branch: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GitHubUser {
    pub(crate) login: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GitHubLabel {
    pub(crate) name: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GitHubIssueSearchResponse {
    pub(crate) items: Vec<GitHubIssue>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GitHubIssue {
    pub(crate) number: u64,
    pub(crate) state: String,
    pub(crate) title: String,
    pub(crate) body: Option<String>,
    pub(crate) labels: Vec<GitHubLabel>,
    pub(crate) user: GitHubUser,
    pub(crate) html_url: String,
    pub(crate) pull_request: Option<serde_json::Value>,
}

impl GitHubIssue {
    pub(crate) fn into_doc(self, comments: Vec<IssueComment>) -> IssueDoc {
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
pub(crate) struct GitHubIssueComment {
    pub(crate) id: u64,
    pub(crate) body: Option<String>,
    pub(crate) user: GitHubUser,
    pub(crate) created_at: String,
    #[serde(default)]
    pub(crate) author_association: String,
}

impl GitHubIssueComment {
    pub(crate) fn into_issue_comment(self) -> IssueComment {
        IssueComment {
            author: self.user.login,
            body: self.body.unwrap_or_default(),
            created_at: self.created_at,
        }
    }
}

#[derive(Debug, Deserialize)]
pub(crate) struct GitHubPullRequest {
    pub(crate) number: u64,
    pub(crate) html_url: String,
    pub(crate) state: String,
    pub(crate) merged: Option<bool>,
}

#[derive(Debug, Serialize)]
pub(crate) struct CreatePullRequest {
    pub(crate) title: String,
    pub(crate) head: String,
    pub(crate) base: String,
    pub(crate) body: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GitHubReview {
    pub(crate) id: u64,
    pub(crate) state: String,
    pub(crate) body: Option<String>,
    #[serde(default)]
    pub(crate) author_association: String,
    pub(crate) user: Option<GitHubUser>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct GitHubReviewComment {
    pub(crate) id: u64,
    pub(crate) body: String,
    pub(crate) path: String,
    pub(crate) line: Option<u64>,
    #[serde(default)]
    pub(crate) author_association: String,
    pub(crate) user: GitHubUser,
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashSet;

    use crate::core::{IssueDoc, RepoSlug};

    #[tokio::test]
    pub(crate) async fn github_issue_tool_returns_cached_issue_without_fetching() {
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
    pub(crate) async fn github_issue_tool_rejects_issue_outside_current_corpus() {
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
