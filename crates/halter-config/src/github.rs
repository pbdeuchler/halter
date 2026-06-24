// pattern: Imperative Shell

use std::collections::{BTreeMap, BTreeSet};
use std::io::{Cursor, Read};
use std::path::{Component, Path, PathBuf};

use anyhow::Context;
use flate2::read::GzDecoder;
use futures::future::try_join_all;
use reqwest::header::{ACCEPT, AUTHORIZATION, HeaderMap, HeaderValue, USER_AGENT};
use serde_json::Value;

use crate::LoadedPlugin;
use crate::resources::{
    MemTree, PluginLoadOptions, PluginTree, join_relative_path, load_plugin_tree,
    normalize_relative_path,
};

const GITHUB_HOST: &str = "github.com";
const GITHUB_API: &str = "https://api.github.com/repos";
const CODEX_MARKETPLACE_MANIFEST: &str = ".agents/plugins/marketplace.json";
const CLAUDE_MARKETPLACE_MANIFEST: &str = ".claude-plugin/marketplace.json";

/// Which plugins to take from a marketplace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PluginSelection {
    /// Load every plugin listed by the marketplace.
    All,
    /// Load only the named plugins.
    Only(Vec<String>),
}

impl Default for PluginSelection {
    fn default() -> Self {
        Self::All
    }
}

/// Repo coordinates shared by both source kinds.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GithubSource {
    pub owner: String,
    pub repo: String,
    /// `None` means GitHub's default branch. Otherwise this may be a branch,
    /// tag, or commit SHA.
    pub reference: Option<String>,
    /// `None` means repo root.
    pub path: Option<PathBuf>,
}

impl GithubSource {
    /// Build a source from structured fields. Use this for refs containing `/`.
    pub fn new(owner: impl Into<String>, repo: impl Into<String>) -> Self {
        Self {
            owner: owner.into(),
            repo: repo.into(),
            reference: None,
            path: None,
        }
    }

    /// Set the branch, tag, or commit SHA.
    #[must_use]
    pub fn with_reference(mut self, reference: impl Into<String>) -> Self {
        self.reference = Some(reference.into());
        self
    }

    /// Set the path within the repo.
    #[must_use]
    pub fn with_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.path = Some(path.into());
        self
    }
}

/// A declarative GitHub plugin source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GithubPlugins {
    /// The source path is one plugin directory.
    Plugin(GithubSource),
    /// The source path is a marketplace directory.
    Marketplace {
        source: GithubSource,
        select: PluginSelection,
    },
}

impl GithubPlugins {
    /// Parse `https://github.com/{owner}/{repo}` or
    /// `https://github.com/{owner}/{repo}/tree/{ref}/{path}` as one plugin.
    pub fn plugin(url: &str) -> anyhow::Result<Self> {
        Ok(Self::Plugin(parse_github_url(url)?))
    }

    /// Parse the supported GitHub URL forms as a marketplace.
    pub fn marketplace(url: &str) -> anyhow::Result<Self> {
        Ok(Self::Marketplace {
            source: parse_github_url(url)?,
            select: PluginSelection::All,
        })
    }

    /// Structured constructor for one plugin.
    pub fn from_source(source: GithubSource) -> Self {
        Self::Plugin(source)
    }

    /// Structured constructor for one marketplace.
    pub fn marketplace_from_source(source: GithubSource) -> Self {
        Self::Marketplace {
            source,
            select: PluginSelection::All,
        }
    }

    /// Restrict a marketplace to an allow-list of plugin names. This is a no-op
    /// for a standalone plugin source.
    pub fn only<I, S>(self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        match self {
            Self::Marketplace { source, .. } => Self::Marketplace {
                source,
                select: PluginSelection::Only(names.into_iter().map(Into::into).collect()),
            },
            plugin => plugin,
        }
    }
}

/// Fetches GitHub tarballs and loads plugins from them in memory.
#[derive(Debug, Clone)]
pub struct GithubFetcher {
    client: reqwest::Client,
    token: Option<String>,
}

impl Default for GithubFetcher {
    fn default() -> Self {
        Self::new()
    }
}

impl GithubFetcher {
    /// Create a fetcher using `GITHUB_TOKEN` or `GH_TOKEN` when either exists.
    pub fn new() -> Self {
        Self {
            client: reqwest::Client::new(),
            token: std::env::var("GITHUB_TOKEN")
                .ok()
                .or_else(|| std::env::var("GH_TOKEN").ok()),
        }
    }

    /// Replace the HTTP client.
    #[must_use]
    pub fn with_client(mut self, client: reqwest::Client) -> Self {
        self.client = client;
        self
    }

    /// Set an explicit GitHub bearer token.
    #[must_use]
    pub fn with_token(mut self, token: impl Into<String>) -> Self {
        self.token = Some(token.into());
        self
    }

    /// Fetch and load one source into in-memory plugins.
    pub async fn load(&self, source: &GithubPlugins) -> anyhow::Result<Vec<LoadedPlugin>> {
        Ok(self
            .load_with_keys(source)
            .await?
            .into_iter()
            .map(|(_, plugin)| plugin)
            .collect())
    }

    /// Fetch and load many sources concurrently, flatten, and dedupe by
    /// `{marketplace}:{plugin-name}`. Standalone plugins use an empty
    /// marketplace prefix, e.g. `:rust-tools`.
    pub async fn load_all<I>(&self, sources: I) -> anyhow::Result<Vec<LoadedPlugin>>
    where
        I: IntoIterator<Item = GithubPlugins>,
    {
        let sources = sources.into_iter().collect::<Vec<_>>();
        let batches =
            try_join_all(sources.iter().map(|source| self.load_with_keys(source))).await?;
        Ok(dedupe_loaded_plugins(batches))
    }

    async fn load_with_keys(
        &self,
        source: &GithubPlugins,
    ) -> anyhow::Result<Vec<(String, LoadedPlugin)>> {
        match source {
            GithubPlugins::Plugin(source) => {
                let repo = self.fetch_repo(source).await?;
                Ok(vec![load_plugin_from_repo(source, &repo, None)?])
            }
            GithubPlugins::Marketplace { source, select } => {
                let repo = self.fetch_repo(source).await?;
                self.load_marketplace_from_repo(source, &repo, select).await
            }
        }
    }

    async fn fetch_repo(&self, source: &GithubSource) -> anyhow::Result<FetchedRepo> {
        validate_source(source)?;
        let url = tarball_url(source);
        let mut headers = HeaderMap::new();
        headers.insert(USER_AGENT, HeaderValue::from_static("halter-config"));
        headers.insert(
            ACCEPT,
            HeaderValue::from_static("application/vnd.github+json"),
        );
        if let Some(token) = self.token.as_deref() {
            headers.insert(
                AUTHORIZATION,
                HeaderValue::from_str(&format!("Bearer {token}"))
                    .context("invalid GitHub token header value")?,
            );
        }

        let response = self
            .client
            .get(&url)
            .headers(headers)
            .send()
            .await
            .with_context(|| {
                format!(
                    "failed to fetch GitHub tarball for {}",
                    source_label(source)
                )
            })?
            .error_for_status()
            .with_context(|| {
                format!("GitHub tarball request failed for {}", source_label(source))
            })?;
        let bytes = response.bytes().await.with_context(|| {
            format!("failed to read GitHub tarball for {}", source_label(source))
        })?;
        let fallback_ref = source
            .reference
            .as_deref()
            .unwrap_or("default-branch")
            .to_owned();
        let (files, resolved_revision) = extract_github_tarball(&bytes, &fallback_ref)?;
        let root = repo_synthetic_root(source, &resolved_revision, Path::new(""));
        Ok(FetchedRepo {
            tree: MemTree::new(root, files),
            resolved_revision,
        })
    }

    async fn load_marketplace_from_repo(
        &self,
        source: &GithubSource,
        repo: &FetchedRepo,
        select: &PluginSelection,
    ) -> anyhow::Result<Vec<(String, LoadedPlugin)>> {
        let base = normalized_source_path(source)?;
        let codex_manifest = join_relative_path(&base, Path::new(CODEX_MARKETPLACE_MANIFEST));
        let claude_manifest = join_relative_path(&base, Path::new(CLAUDE_MARKETPLACE_MANIFEST));
        let (manifest_path, kind) = if repo.tree.is_file(&codex_manifest) {
            (codex_manifest, MarketplaceKind::Codex)
        } else if repo.tree.is_file(&claude_manifest) {
            (claude_manifest, MarketplaceKind::Claude)
        } else {
            anyhow::bail!(
                "marketplace at '{}' is missing a supported manifest: expected {} or {}",
                repo_synthetic_root(source, &repo.resolved_revision, &base).display(),
                CODEX_MARKETPLACE_MANIFEST,
                CLAUDE_MARKETPLACE_MANIFEST
            );
        };
        let manifest: Value = serde_json::from_slice(&repo.tree.read(&manifest_path)?)
            .with_context(|| {
                format!(
                    "failed to parse marketplace manifest at {}",
                    repo_synthetic_root(source, &repo.resolved_revision, &manifest_path).display()
                )
            })?;
        let marketplace_name = required_string(&manifest, "name", "marketplace manifest")?;
        let entries = manifest
            .get("plugins")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                anyhow::anyhow!("marketplace '{marketplace_name}' is missing array field 'plugins'")
            })?;

        let allow = match select {
            PluginSelection::All => None,
            PluginSelection::Only(names) => Some(names.iter().cloned().collect::<BTreeSet<_>>()),
        };
        let mut seen_names = BTreeSet::new();
        let mut loaded_names = BTreeSet::new();
        let mut plugins = Vec::new();

        for entry in entries {
            let plugin_name = required_string(entry, "name", "marketplace plugin entry")?;
            if !seen_names.insert(plugin_name.clone()) {
                anyhow::bail!(
                    "marketplace '{marketplace_name}' contains duplicate plugin name '{plugin_name}'"
                );
            }
            if allow
                .as_ref()
                .is_some_and(|allowed| !allowed.contains(&plugin_name))
            {
                continue;
            }
            let source_value = entry.get("source").ok_or_else(|| {
                anyhow::anyhow!(
                    "marketplace plugin '{plugin_name}' is missing required field 'source'"
                )
            })?;
            match parse_marketplace_source(source_value, kind, &base, &plugin_name)? {
                MarketplacePluginSource::Local(path) => {
                    let plugin_source = GithubSource {
                        owner: source.owner.clone(),
                        repo: source.repo.clone(),
                        reference: source.reference.clone(),
                        path: Some(path),
                    };
                    plugins.push(load_plugin_from_repo(
                        &plugin_source,
                        repo,
                        Some(&marketplace_name),
                    )?);
                    loaded_names.insert(plugin_name);
                }
                MarketplacePluginSource::Github(github_source) => {
                    let fetched = self.fetch_repo(&github_source).await?;
                    plugins.push(load_plugin_from_repo(
                        &github_source,
                        &fetched,
                        Some(&marketplace_name),
                    )?);
                    loaded_names.insert(plugin_name);
                }
            }
        }

        if let Some(allowed) = allow {
            for requested in allowed {
                if !loaded_names.contains(&requested) {
                    anyhow::bail!(
                        "marketplace '{marketplace_name}' does not contain selected plugin '{requested}'"
                    );
                }
            }
        }

        Ok(plugins)
    }
}

/// Top-level convenience for loading many GitHub plugin sources.
pub async fn load_plugins<I>(sources: I) -> anyhow::Result<Vec<LoadedPlugin>>
where
    I: IntoIterator<Item = GithubPlugins>,
{
    GithubFetcher::new().load_all(sources).await
}

#[derive(Debug)]
struct FetchedRepo {
    tree: MemTree,
    resolved_revision: String,
}

#[derive(Debug, Clone, Copy)]
enum MarketplaceKind {
    Codex,
    Claude,
}

#[derive(Debug)]
enum MarketplacePluginSource {
    Local(PathBuf),
    Github(GithubSource),
}

fn parse_github_url(url: &str) -> anyhow::Result<GithubSource> {
    let parsed = reqwest::Url::parse(url).with_context(|| {
        format!("invalid GitHub URL '{url}': expected https://github.com/{{owner}}/{{repo}}")
    })?;
    if parsed.scheme() != "https" || parsed.host_str() != Some(GITHUB_HOST) {
        anyhow::bail!(
            "unsupported GitHub URL '{url}': expected https://github.com/{{owner}}/{{repo}}"
        );
    }
    if parsed.query().is_some() || parsed.fragment().is_some() {
        anyhow::bail!(
            "unsupported GitHub URL '{url}': query strings and fragments are not supported"
        );
    }

    let segments = parsed
        .path_segments()
        .map(|segments| {
            segments
                .filter(|segment| !segment.is_empty())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if segments.len() < 2 {
        anyhow::bail!(
            "unsupported GitHub URL '{url}': expected https://github.com/{{owner}}/{{repo}}"
        );
    }
    let owner = segments[0].to_owned();
    let repo = trim_git_suffix(segments[1]);
    if owner.is_empty() || repo.is_empty() {
        anyhow::bail!("unsupported GitHub URL '{url}': owner and repo must be non-empty");
    }

    match segments.as_slice() {
        [_, _] => Ok(GithubSource {
            owner,
            repo,
            reference: None,
            path: None,
        }),
        [_, _, "tree", reference, rest @ ..] if !reference.is_empty() => {
            let path = if rest.is_empty() {
                None
            } else {
                Some(normalize_url_path(rest, url)?)
            };
            Ok(GithubSource {
                owner,
                repo,
                reference: Some((*reference).to_owned()),
                path,
            })
        }
        _ => anyhow::bail!(
            "unsupported GitHub URL '{url}': only repo roots and /tree/{{ref}}/{{path}} URLs are supported"
        ),
    }
}

fn normalize_url_path(segments: &[&str], original: &str) -> anyhow::Result<PathBuf> {
    normalize_relative_path(&segments.iter().collect::<PathBuf>(), original)
}

fn trim_git_suffix(repo: &str) -> String {
    repo.strip_suffix(".git").unwrap_or(repo).to_owned()
}

fn validate_source(source: &GithubSource) -> anyhow::Result<()> {
    if source.owner.trim().is_empty() {
        anyhow::bail!("GitHub source owner must be non-empty");
    }
    if source.repo.trim().is_empty() {
        anyhow::bail!("GitHub source repo must be non-empty");
    }
    if let Some(path) = &source.path {
        normalize_relative_path(path, &path.display().to_string())?;
    }
    Ok(())
}

fn normalized_source_path(source: &GithubSource) -> anyhow::Result<PathBuf> {
    match source.path.as_deref() {
        Some(path) => normalize_relative_path(path, &path.display().to_string()),
        None => Ok(PathBuf::new()),
    }
}

fn tarball_url(source: &GithubSource) -> String {
    match source.reference.as_deref() {
        Some(reference) => format!(
            "{}/{}/{}/tarball/{}",
            GITHUB_API, source.owner, source.repo, reference
        ),
        None => format!("{}/{}/{}/tarball", GITHUB_API, source.owner, source.repo),
    }
}

fn source_label(source: &GithubSource) -> String {
    let mut label = format!("{}/{}", source.owner, source.repo);
    if let Some(reference) = &source.reference {
        label.push('@');
        label.push_str(reference);
    }
    if let Some(path) = &source.path {
        label.push('/');
        label.push_str(&path_to_slash(path));
    }
    label
}

fn repo_synthetic_root(source: &GithubSource, revision: &str, path: &Path) -> PathBuf {
    let mut root = format!("github:{}/{}@{}", source.owner, source.repo, revision);
    if !path.as_os_str().is_empty() {
        root.push('/');
        root.push_str(&path_to_slash(path));
    }
    PathBuf::from(root)
}

fn path_to_slash(path: &Path) -> String {
    path.components()
        .filter_map(|component| match component {
            Component::Normal(part) => Some(part.to_string_lossy().to_string()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("/")
}

fn load_plugin_from_repo(
    source: &GithubSource,
    repo: &FetchedRepo,
    marketplace: Option<&str>,
) -> anyhow::Result<(String, LoadedPlugin)> {
    let path = normalized_source_path(source)?;
    let root = repo_synthetic_root(source, &repo.resolved_revision, &path);
    let tree = repo.tree.scoped(&path, root)?;
    let plugin = load_plugin_tree(&tree, PluginLoadOptions::remote_install())?
        .ok_or_else(|| anyhow::anyhow!("remote plugin unexpectedly did not load"))?;
    let key = match marketplace {
        Some(marketplace) => format!("{marketplace}:{}", plugin.manifest.name),
        None => format!(":{}", plugin.manifest.name),
    };
    Ok((key, plugin))
}

fn parse_marketplace_source(
    value: &Value,
    kind: MarketplaceKind,
    marketplace_root: &Path,
    plugin_name: &str,
) -> anyhow::Result<MarketplacePluginSource> {
    match value {
        Value::String(raw) => local_marketplace_source(raw, marketplace_root, plugin_name),
        Value::Object(object) => {
            let source_kind = object
                .get("source")
                .and_then(Value::as_str)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "marketplace plugin '{plugin_name}' has object source missing string field 'source'"
                    )
                })?;
            match source_kind {
                "local" => {
                    let path = object.get("path").and_then(Value::as_str).ok_or_else(|| {
                        anyhow::anyhow!(
                            "marketplace plugin '{plugin_name}' local source is missing string field 'path'"
                        )
                    })?;
                    local_marketplace_source(path, marketplace_root, plugin_name)
                }
                "github" => parse_github_marketplace_source(object, plugin_name),
                other => anyhow::bail!(
                    "marketplace plugin '{plugin_name}' has unsupported source kind '{other}'"
                ),
            }
        }
        _ => anyhow::bail!(
            "marketplace plugin '{plugin_name}' source must be a {} source",
            match kind {
                MarketplaceKind::Codex => "Codex",
                MarketplaceKind::Claude => "Claude",
            }
        ),
    }
}

fn local_marketplace_source(
    raw: &str,
    marketplace_root: &Path,
    plugin_name: &str,
) -> anyhow::Result<MarketplacePluginSource> {
    if !raw.starts_with("./") {
        anyhow::bail!(
            "marketplace plugin '{plugin_name}' local source '{raw}' must start with './'"
        );
    }
    let rel = normalize_relative_path(Path::new(raw), raw)?;
    Ok(MarketplacePluginSource::Local(join_relative_path(
        marketplace_root,
        &rel,
    )))
}

fn parse_github_marketplace_source(
    object: &serde_json::Map<String, Value>,
    plugin_name: &str,
) -> anyhow::Result<MarketplacePluginSource> {
    let repo = object.get("repo").and_then(Value::as_str).ok_or_else(|| {
        anyhow::anyhow!(
            "marketplace plugin '{plugin_name}' github source is missing string field 'repo'"
        )
    })?;
    let (owner, repo_name) = repo.split_once('/').ok_or_else(|| {
        anyhow::anyhow!(
            "marketplace plugin '{plugin_name}' github source repo must be 'owner/repo'"
        )
    })?;
    let reference = object
        .get("sha")
        .and_then(Value::as_str)
        .or_else(|| object.get("ref").and_then(Value::as_str))
        .map(ToOwned::to_owned);
    let path = object
        .get("path")
        .and_then(Value::as_str)
        .map(|raw| normalize_relative_path(Path::new(raw), raw))
        .transpose()?;
    Ok(MarketplacePluginSource::Github(GithubSource {
        owner: owner.to_owned(),
        repo: trim_git_suffix(repo_name),
        reference,
        path,
    }))
}

fn required_string(value: &Value, key: &str, context: &str) -> anyhow::Result<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| anyhow::anyhow!("{context} is missing required string field '{key}'"))
}

fn dedupe_loaded_plugins(batches: Vec<Vec<(String, LoadedPlugin)>>) -> Vec<LoadedPlugin> {
    let mut seen = BTreeSet::new();
    let mut plugins = Vec::new();
    for (key, plugin) in batches.into_iter().flatten() {
        if seen.insert(key) {
            plugins.push(plugin);
        }
    }
    plugins
}

fn extract_github_tarball(
    bytes: &[u8],
    fallback_ref: &str,
) -> anyhow::Result<(BTreeMap<PathBuf, Vec<u8>>, String)> {
    let decoder = GzDecoder::new(Cursor::new(bytes));
    let mut archive = tar::Archive::new(decoder);
    let mut prefix = None::<PathBuf>;
    let mut files = BTreeMap::new();

    for entry in archive.entries().context("failed to read GitHub tarball")? {
        let mut entry = entry.context("failed to read GitHub tar entry")?;
        if !entry.header().entry_type().is_file() {
            continue;
        }
        let path = entry
            .path()
            .context("failed to read GitHub tar entry path")?
            .into_owned();
        let mut components = path.components();
        let Some(Component::Normal(first)) = components.next() else {
            anyhow::bail!(
                "GitHub tarball entry '{}' is not under a repo prefix",
                path.display()
            );
        };
        let current_prefix = PathBuf::from(first);
        if let Some(existing) = &prefix {
            if existing != &current_prefix {
                anyhow::bail!(
                    "GitHub tarball contains multiple top-level prefixes: '{}' and '{}'",
                    existing.display(),
                    current_prefix.display()
                );
            }
        } else {
            prefix = Some(current_prefix);
        }
        let rest = components.as_path().to_path_buf();
        if rest.as_os_str().is_empty() {
            continue;
        }
        let rel = normalize_relative_path(&rest, &path.display().to_string())?;
        let mut body = Vec::new();
        entry
            .read_to_end(&mut body)
            .context("failed to read GitHub tar entry body")?;
        files.insert(rel, body);
    }

    if files.is_empty() {
        anyhow::bail!("GitHub tarball did not contain any files");
    }
    let resolved_revision = prefix
        .and_then(|prefix| {
            prefix
                .file_name()
                .map(|name| name.to_string_lossy().to_string())
        })
        .and_then(|prefix| {
            prefix
                .rsplit_once('-')
                .map(|(_, revision)| revision.to_owned())
        })
        .filter(|revision| !revision.is_empty())
        .unwrap_or_else(|| fallback_ref.to_owned());
    Ok((files, resolved_revision))
}

#[cfg(test)]
mod tests {
    use super::*;
    use halter_protocol::{PluginId, PluginManifest};

    fn plugin(name: &str, root: &str) -> LoadedPlugin {
        LoadedPlugin {
            id: PluginId::from(format!("plugin-{name}-{root}")),
            root: PathBuf::from(root),
            manifest: PluginManifest {
                name: name.to_owned(),
                version: "0.1.0".to_owned(),
                skills: Vec::new(),
                agents: Vec::new(),
                hooks: None,
                mcp_servers: None,
                lsp_servers: None,
                allowed_http_hosts: Vec::new(),
                allowed_env_vars: Vec::new(),
            },
            skills: Vec::new(),
            agents: Vec::new(),
            hooks: Vec::new(),
            mcp_servers: Vec::new(),
            lsp_servers: Vec::new(),
            output_styles: Vec::new(),
            bin_paths: Vec::new(),
            defaults: crate::PluginDefaults::default(),
        }
    }

    #[test]
    fn parses_supported_github_urls() {
        struct Case {
            name: &'static str,
            url: &'static str,
            owner: &'static str,
            repo: &'static str,
            reference: Option<&'static str>,
            path: Option<&'static str>,
        }

        let cases = [
            Case {
                name: "repo_root",
                url: "https://github.com/acme/tools",
                owner: "acme",
                repo: "tools",
                reference: None,
                path: None,
            },
            Case {
                name: "repo_root_git_suffix_trailing_slash",
                url: "https://github.com/acme/tools.git/",
                owner: "acme",
                repo: "tools",
                reference: None,
                path: None,
            },
            Case {
                name: "tree_ref_path",
                url: "https://github.com/acme/mono/tree/v1.2.3/plugins/sql",
                owner: "acme",
                repo: "mono",
                reference: Some("v1.2.3"),
                path: Some("plugins/sql"),
            },
            Case {
                name: "tree_sha",
                url: "https://github.com/acme/mono/tree/0123456789012345678901234567890123456789/plugin",
                owner: "acme",
                repo: "mono",
                reference: Some("0123456789012345678901234567890123456789"),
                path: Some("plugin"),
            },
        ];

        for case in cases {
            let got = parse_github_url(case.url).unwrap_or_else(|error| {
                panic!("{} should parse: {error:?}", case.name);
            });
            assert_eq!(got.owner, case.owner, "{}", case.name);
            assert_eq!(got.repo, case.repo, "{}", case.name);
            assert_eq!(got.reference.as_deref(), case.reference, "{}", case.name);
            assert_eq!(
                got.path.as_ref().map(|path| path_to_slash(path)),
                case.path.map(str::to_owned),
                "{}",
                case.name
            );
        }
    }

    #[test]
    fn rejects_unsupported_github_urls() {
        let urls = [
            "http://github.com/acme/tools",
            "https://example.com/acme/tools",
            "https://github.com/acme/tools/blob/main/src/lib.rs",
            "https://github.com/acme/tools/pull/1",
            "https://gist.github.com/acme/123",
            "https://github.com/acme",
            "https://github.com/acme/tools?tab=readme",
        ];

        for url in urls {
            assert!(parse_github_url(url).is_err(), "{url} should fail");
        }
    }

    #[tokio::test]
    async fn marketplace_prefers_codex_manifest_and_honors_allow_list() {
        let mut files = BTreeMap::new();
        files.insert(
            PathBuf::from(CODEX_MARKETPLACE_MANIFEST),
            br#"{
  "name": "acme-tools",
  "plugins": [
    {"name": "one", "source": {"source": "local", "path": "./plugins/one"}},
    {"name": "two", "source": {"source": "local", "path": "./plugins/two"}}
  ]
}"#
            .to_vec(),
        );
        files.insert(
            PathBuf::from(CLAUDE_MARKETPLACE_MANIFEST),
            br#"{"name":"wrong","plugins":[]}"#.to_vec(),
        );
        files.insert(
            PathBuf::from("plugins/two/.codex-plugin/plugin.json"),
            br#"{"name":"two","version":"0.1.0","skills":"./skills"}"#.to_vec(),
        );
        files.insert(
            PathBuf::from("plugins/two/skills/review/SKILL.md"),
            b"---\nname: review\n---\n\nReview.\n".to_vec(),
        );
        let source = GithubSource::new("acme", "market");
        let repo = FetchedRepo {
            tree: MemTree::new(PathBuf::from("github:acme/market@abc123"), files),
            resolved_revision: "abc123".to_owned(),
        };

        let loaded = GithubFetcher::new()
            .load_marketplace_from_repo(
                &source,
                &repo,
                &PluginSelection::Only(vec!["two".to_owned()]),
            )
            .await
            .expect("load marketplace");

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, "acme-tools:two");
        assert_eq!(loaded[0].1.manifest.name, "two");
        assert_eq!(loaded[0].1.skills[0].name, "review");
    }

    #[tokio::test]
    async fn marketplace_falls_back_to_claude_manifest() {
        let mut files = BTreeMap::new();
        files.insert(
            PathBuf::from(CLAUDE_MARKETPLACE_MANIFEST),
            br#"{
  "name": "claude-tools",
  "plugins": [
    {"name": "one", "source": "./plugins/one"}
  ]
}"#
            .to_vec(),
        );
        files.insert(
            PathBuf::from("plugins/one/.claude-plugin/plugin.json"),
            br#"{"name":"one","version":"0.1.0"}"#.to_vec(),
        );
        let source = GithubSource::new("acme", "market");
        let repo = FetchedRepo {
            tree: MemTree::new(PathBuf::from("github:acme/market@abc123"), files),
            resolved_revision: "abc123".to_owned(),
        };

        let loaded = GithubFetcher::new()
            .load_marketplace_from_repo(&source, &repo, &PluginSelection::All)
            .await
            .expect("load marketplace");

        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].0, "claude-tools:one");
    }

    #[tokio::test]
    async fn marketplace_errors_without_codex_or_claude_manifest() {
        let source = GithubSource::new("acme", "market");
        let repo = FetchedRepo {
            tree: MemTree::new(PathBuf::from("github:acme/market@abc123"), BTreeMap::new()),
            resolved_revision: "abc123".to_owned(),
        };

        let error = GithubFetcher::new()
            .load_marketplace_from_repo(&source, &repo, &PluginSelection::All)
            .await
            .expect_err("missing marketplace manifest should fail");

        assert!(
            error.to_string().contains("missing a supported manifest"),
            "unexpected error: {error:?}"
        );
    }

    #[tokio::test]
    async fn marketplace_errors_on_duplicate_names() {
        let mut files = BTreeMap::new();
        files.insert(
            PathBuf::from(CLAUDE_MARKETPLACE_MANIFEST),
            br#"{"name":"dupes","plugins":[{"name":"one","source":"./a"},{"name":"one","source":"./b"}]}"#.to_vec(),
        );
        files.insert(
            PathBuf::from("a/.claude-plugin/plugin.json"),
            br#"{"name":"one","version":"0.1.0"}"#.to_vec(),
        );
        let source = GithubSource::new("acme", "market");
        let repo = FetchedRepo {
            tree: MemTree::new(PathBuf::from("github:acme/market@abc123"), files),
            resolved_revision: "abc123".to_owned(),
        };

        let error = GithubFetcher::new()
            .load_marketplace_from_repo(&source, &repo, &PluginSelection::All)
            .await
            .expect_err("duplicate plugin names should fail");

        assert!(error.to_string().contains("duplicate plugin name"));
    }

    #[test]
    fn dedupe_collapses_exact_qualified_keys_only() {
        let deduped = dedupe_loaded_plugins(vec![
            vec![
                (":tool".to_owned(), plugin("standalone-a", "a")),
                (":tool".to_owned(), plugin("standalone-b", "b")),
            ],
            vec![
                ("market:tool".to_owned(), plugin("market-a", "c")),
                ("other:tool".to_owned(), plugin("other-a", "d")),
            ],
        ]);

        assert_eq!(deduped.len(), 3);
        assert_eq!(deduped[0].manifest.name, "standalone-a");
        assert_eq!(deduped[1].manifest.name, "market-a");
        assert_eq!(deduped[2].manifest.name, "other-a");
    }

    #[test]
    fn extracts_github_tarball_in_memory() {
        let bytes = build_test_tarball([
            (
                "repo-0123456789abcdef/.claude-plugin/plugin.json",
                br#"{"name":"remote","version":"0.1.0"}"#.as_slice(),
            ),
            (
                "repo-0123456789abcdef/skills/a/SKILL.md",
                b"body".as_slice(),
            ),
        ]);

        let (files, revision) = extract_github_tarball(&bytes, "main").expect("extract");

        assert_eq!(revision, "0123456789abcdef");
        assert!(files.contains_key(Path::new(".claude-plugin/plugin.json")));
        assert!(files.contains_key(Path::new("skills/a/SKILL.md")));
    }

    fn build_test_tarball<const N: usize>(files: [(&str, &[u8]); N]) -> Vec<u8> {
        let mut tar_bytes = Vec::new();
        {
            let mut archive = tar::Builder::new(&mut tar_bytes);
            for (path, body) in files {
                let mut header = tar::Header::new_gnu();
                header.set_size(body.len() as u64);
                header.set_mode(0o644);
                header.set_cksum();
                archive
                    .append_data(&mut header, path, body)
                    .expect("append");
            }
            archive.finish().expect("finish tar");
        }
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        std::io::Write::write_all(&mut encoder, &tar_bytes).expect("gzip write");
        encoder.finish().expect("gzip finish")
    }
}
