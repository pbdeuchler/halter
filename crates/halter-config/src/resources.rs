// pattern: Imperative Shell

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io;
#[cfg(feature = "remote-plugins")]
use std::io::ErrorKind;
use std::path::{Component, Path, PathBuf};
#[cfg(feature = "remote-plugins")]
use std::sync::Arc;

use anyhow::Context;
#[cfg(feature = "remote-plugins")]
use halter_hooks::HookHandlerConfig;
use halter_hooks::{HooksFile, HooksLoadWarning};
use halter_protocol::{AgentId, ContentHash, PluginId, PluginManifest, SkillId};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tracing::{debug, info};

#[derive(Debug, Clone)]
/// File loaded as part of a skill or plugin resource.
pub struct LoadedResourceFile {
    pub path: PathBuf,
    pub body: String,
    pub revision: ContentHash,
}

#[derive(Debug, Clone)]
/// Executable path discovered under a resource root.
pub struct LoadedExecutable {
    pub path: PathBuf,
}

#[derive(Debug, Clone)]
/// Fully loaded skill directory, including its `SKILL.md` body.
pub struct LoadedSkill {
    pub id: SkillId,
    pub name: String,
    pub description: String,
    pub root: PathBuf,
    pub body: String,
    pub supporting_files: Vec<LoadedResourceFile>,
    pub scripts: Vec<LoadedExecutable>,
    pub revision: ContentHash,
}

#[derive(Debug, Clone)]
/// Loaded agent definition from a plugin.
pub struct LoadedAgent {
    pub id: AgentId,
    pub name: String,
    pub prompt: String,
    pub revision: ContentHash,
}

#[derive(Debug, Clone)]
/// Parsed hook file plus load warnings from one plugin.
pub struct LoadedHooksFile {
    pub plugin_id: PluginId,
    pub plugin_root: PathBuf,
    pub source_path: PathBuf,
    pub revision: ContentHash,
    pub parsed: HooksFile,
    pub warnings: Vec<HooksLoadWarning>,
}

#[derive(Debug, Clone)]
/// Raw MCP server definition loaded from a plugin.
pub struct LoadedMcpServer {
    pub path: PathBuf,
    pub body: Value,
}

#[derive(Debug, Clone)]
/// Raw LSP server definition loaded from a plugin.
pub struct LoadedLspServer {
    pub path: PathBuf,
    pub body: Value,
}

#[derive(Debug, Clone)]
/// Output-style resource discovered in a plugin.
pub struct LoadedOutputStyle {
    pub path: PathBuf,
}

#[derive(Debug, Clone, Default)]
/// Plugin default settings preserved for future consumers.
pub struct PluginDefaults {
    pub settings: BTreeMap<String, Value>,
}

#[derive(Debug, Clone)]
/// Fully loaded plugin root and all supported plugin resources.
pub struct LoadedPlugin {
    pub id: PluginId,
    pub root: PathBuf,
    pub manifest: PluginManifest,
    pub skills: Vec<LoadedSkill>,
    pub agents: Vec<LoadedAgent>,
    pub hooks: Vec<LoadedHooksFile>,
    pub mcp_servers: Vec<LoadedMcpServer>,
    pub lsp_servers: Vec<LoadedLspServer>,
    pub output_styles: Vec<LoadedOutputStyle>,
    pub bin_paths: Vec<PathBuf>,
    pub defaults: PluginDefaults,
}

#[derive(Debug, Default, Clone)]
/// Loader for standalone skill roots.
pub struct SkillLoader;

impl SkillLoader {
    /// Load all skills under the configured roots.
    pub fn load_roots(&self, roots: &[PathBuf]) -> anyhow::Result<Vec<LoadedSkill>> {
        debug!(root_count = roots.len(), "loading skill roots");
        let mut skills = Vec::new();
        let mut visited = BTreeSet::new();
        for root in normalized_roots(roots) {
            if !root.exists() {
                debug!(root = %root.display(), "skipping missing skill root");
                continue;
            }
            collect_skills(&root, &mut visited, &mut skills)?;
        }
        for skill in &mut skills {
            skill.body = render_plugin_vars(&skill.root, &skill.body);
            skill.revision = hash_bytes(skill.body.as_bytes());
        }
        info!(skill_count = skills.len(), "loaded skills");
        Ok(skills)
    }
}

#[derive(Debug, Default, Clone)]
/// Loader for plugin roots.
pub struct PluginLoader;

impl PluginLoader {
    /// Load every plugin directory found under the configured roots.
    pub fn load_roots(&self, roots: &[PathBuf]) -> anyhow::Result<Vec<LoadedPlugin>> {
        debug!(root_count = roots.len(), "loading plugin roots");
        let mut plugins = Vec::new();
        for root in normalized_roots(roots) {
            if !root.exists() {
                debug!(root = %root.display(), "skipping missing plugin root");
                continue;
            }
            let mut entries = fs::read_dir(&root)?.collect::<Result<Vec<_>, std::io::Error>>()?;
            // Preserve path order so plugin_load_order stays deterministic across reloads.
            entries.sort_by_key(|entry| entry.path());
            for entry in entries {
                if entry.file_type()?.is_dir() {
                    let plugin_root = entry.path();
                    let tree = DiskTree::new(plugin_root.clone())?;
                    if let Some(plugin) = load_plugin_tree(&tree, PluginLoadOptions::disk_scan())? {
                        plugins.push(plugin);
                    }
                }
            }
        }
        info!(plugin_count = plugins.len(), "loaded plugins");
        Ok(plugins)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ManifestSearch {
    DiskCompatible,
    #[cfg(feature = "remote-plugins")]
    RemoteCodexClaude,
}

#[cfg(feature = "remote-plugins")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MissingManifest {
    Skip,
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ExecComponentPolicy {
    Allow,
    #[cfg(feature = "remote-plugins")]
    WarnAndSkip,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct PluginLoadOptions {
    manifest_search: ManifestSearch,
    #[cfg(feature = "remote-plugins")]
    missing_manifest: MissingManifest,
    exec_policy: ExecComponentPolicy,
}

impl PluginLoadOptions {
    fn disk_scan() -> Self {
        Self {
            manifest_search: ManifestSearch::DiskCompatible,
            #[cfg(feature = "remote-plugins")]
            missing_manifest: MissingManifest::Skip,
            exec_policy: ExecComponentPolicy::Allow,
        }
    }

    #[cfg(feature = "remote-plugins")]
    pub(crate) fn remote_install() -> Self {
        Self {
            manifest_search: ManifestSearch::RemoteCodexClaude,
            missing_manifest: MissingManifest::Error,
            exec_policy: ExecComponentPolicy::WarnAndSkip,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TreeEntryKind {
    File,
    Dir,
}

#[derive(Debug, Clone)]
pub(crate) struct TreeEntry {
    rel: PathBuf,
    kind: TreeEntryKind,
}

pub(crate) trait PluginTree {
    fn root_display(&self) -> &Path;
    fn stable_root(&self) -> &Path;
    fn display_path(&self, rel: &Path) -> PathBuf {
        join_display_path(self.root_display(), rel)
    }
    fn stable_path(&self, rel: &Path) -> PathBuf {
        join_display_path(self.stable_root(), rel)
    }
    fn read(&self, rel: &Path) -> io::Result<Vec<u8>>;
    fn read_dir(&self, rel: &Path) -> io::Result<Vec<TreeEntry>>;
    fn is_file(&self, rel: &Path) -> bool;
    fn is_dir(&self, rel: &Path) -> bool;
}

#[derive(Debug, Clone)]
struct DiskTree {
    root_display: PathBuf,
    stable_root: PathBuf,
}

impl DiskTree {
    fn new(root: PathBuf) -> anyhow::Result<Self> {
        let stable_root = fs::canonicalize(&root).with_context(|| {
            format!(
                "failed to canonicalize plugin root '{}' while building disk tree",
                root.display()
            )
        })?;
        Ok(Self {
            root_display: root,
            stable_root,
        })
    }
}

impl PluginTree for DiskTree {
    fn root_display(&self) -> &Path {
        &self.root_display
    }

    fn stable_root(&self) -> &Path {
        &self.stable_root
    }

    fn read(&self, rel: &Path) -> io::Result<Vec<u8>> {
        fs::read(self.stable_path(rel))
    }

    fn read_dir(&self, rel: &Path) -> io::Result<Vec<TreeEntry>> {
        let mut entries = Vec::new();
        for entry in fs::read_dir(self.stable_path(rel))? {
            let entry = entry?;
            let kind = if entry.file_type()?.is_dir() {
                TreeEntryKind::Dir
            } else {
                TreeEntryKind::File
            };
            entries.push(TreeEntry {
                rel: join_relative_path(rel, &PathBuf::from(entry.file_name())),
                kind,
            });
        }
        entries.sort_by(|a, b| a.rel.cmp(&b.rel));
        Ok(entries)
    }

    fn is_file(&self, rel: &Path) -> bool {
        self.stable_path(rel).is_file()
    }

    fn is_dir(&self, rel: &Path) -> bool {
        self.stable_path(rel).is_dir()
    }
}

#[cfg(feature = "remote-plugins")]
#[derive(Debug, Clone)]
pub(crate) struct MemTree {
    files: Arc<BTreeMap<PathBuf, Vec<u8>>>,
    base: PathBuf,
    root: PathBuf,
}

#[cfg(feature = "remote-plugins")]
impl MemTree {
    pub(crate) fn new(root: PathBuf, files: BTreeMap<PathBuf, Vec<u8>>) -> Self {
        Self {
            files: Arc::new(files),
            base: PathBuf::new(),
            root,
        }
    }

    pub(crate) fn scoped(&self, rel: &Path, root: PathBuf) -> anyhow::Result<Self> {
        let rel = normalize_relative_path(rel, &rel.display().to_string())?;
        if !rel.as_os_str().is_empty() && !self.is_dir(&rel) {
            anyhow::bail!(
                "remote plugin path '{}' does not exist or is not a directory",
                rel.display()
            );
        }
        Ok(Self {
            files: Arc::clone(&self.files),
            base: join_relative_path(&self.base, &rel),
            root,
        })
    }

    fn storage_path(&self, rel: &Path) -> PathBuf {
        join_relative_path(&self.base, rel)
    }
}

#[cfg(feature = "remote-plugins")]
impl PluginTree for MemTree {
    fn root_display(&self) -> &Path {
        &self.root
    }

    fn stable_root(&self) -> &Path {
        &self.root
    }

    fn read(&self, rel: &Path) -> io::Result<Vec<u8>> {
        let storage = self.storage_path(rel);
        self.files.get(&storage).cloned().ok_or_else(|| {
            io::Error::new(
                ErrorKind::NotFound,
                format!("remote tree file '{}' not found", rel.display()),
            )
        })
    }

    fn read_dir(&self, rel: &Path) -> io::Result<Vec<TreeEntry>> {
        let storage = self.storage_path(rel);
        if !self.is_dir(rel) {
            return Err(io::Error::new(
                ErrorKind::NotFound,
                format!("remote tree directory '{}' not found", rel.display()),
            ));
        }

        let mut entries = BTreeMap::new();
        for file in self.files.keys() {
            if !path_has_prefix(file, &storage) {
                continue;
            }
            let Ok(rest) = strip_prefix_components(file, &storage) else {
                continue;
            };
            let mut components = rest.components();
            let Some(first) = components.next() else {
                continue;
            };
            let child_rel = join_relative_path(rel, Path::new(first.as_os_str()));
            let kind = if components.next().is_some() {
                TreeEntryKind::Dir
            } else {
                TreeEntryKind::File
            };
            entries
                .entry(child_rel)
                .and_modify(|existing| {
                    if kind == TreeEntryKind::Dir {
                        *existing = TreeEntryKind::Dir;
                    }
                })
                .or_insert(kind);
        }

        Ok(entries
            .into_iter()
            .map(|(rel, kind)| TreeEntry { rel, kind })
            .collect())
    }

    fn is_file(&self, rel: &Path) -> bool {
        self.files.contains_key(&self.storage_path(rel))
    }

    fn is_dir(&self, rel: &Path) -> bool {
        let storage = self.storage_path(rel);
        if storage.as_os_str().is_empty() {
            return !self.files.is_empty();
        }
        self.files.keys().any(|file| {
            path_has_prefix(file, &storage)
                && strip_prefix_components(file, &storage)
                    .map(|rest| rest.components().next().is_some())
                    .unwrap_or(false)
        })
    }
}

pub(crate) fn load_plugin_tree<T: PluginTree>(
    tree: &T,
    options: PluginLoadOptions,
) -> anyhow::Result<Option<LoadedPlugin>> {
    let Some(manifest_path) = manifest_candidates(options.manifest_search)
        .iter()
        .map(PathBuf::from)
        .find(|path| tree.is_file(path))
    else {
        #[cfg(feature = "remote-plugins")]
        if options.missing_manifest == MissingManifest::Error {
            anyhow::bail!(
                "plugin at '{}' is missing a supported manifest: expected .codex-plugin/plugin.json or .claude-plugin/plugin.json",
                tree.root_display().display()
            );
        }
        return Ok(None);
    };

    let manifest_bytes = tree.read(&manifest_path).with_context(|| {
        format!(
            "failed to read plugin manifest at {}",
            tree.display_path(&manifest_path).display()
        )
    })?;
    let manifest_value: Value = serde_json::from_slice(&manifest_bytes).with_context(|| {
        format!(
            "failed to parse plugin manifest at {}",
            tree.display_path(&manifest_path).display()
        )
    })?;

    let manifest = parse_plugin_manifest(&manifest_value, &tree.display_path(&manifest_path))?;
    let plugin_id = stable_plugin_id(tree.stable_root(), &manifest);

    let mut skills = Vec::new();
    let mut visited_skill_dirs = BTreeSet::new();
    for skill_path in &manifest.skills {
        let resolved = resolve_plugin_path(skill_path)?;
        if tree.is_file(&join_relative_path(&resolved, Path::new("SKILL.md"))) {
            skills.push(load_skill_from_tree(tree, &resolved)?);
        } else if tree.is_dir(&resolved) {
            collect_skills_in_tree(tree, &resolved, &mut visited_skill_dirs, &mut skills)?;
        } else {
            anyhow::bail!(
                "invalid plugin path '{}': component does not exist under '{}'",
                skill_path,
                tree.root_display().display()
            );
        }
    }

    skills.iter_mut().for_each(|skill| {
        skill.body = render_plugin_vars(tree.root_display(), &skill.body);
        skill.revision = hash_bytes(skill.body.as_bytes());
    });

    let mut agents = Vec::new();
    for agent_path in &manifest.agents {
        let resolved = resolve_plugin_path(agent_path)?;
        if tree.is_file(&resolved) {
            agents.push(load_agent_file_from_tree(tree, &resolved)?);
        } else if tree.is_dir(&resolved) {
            load_agent_dir_from_tree(tree, &resolved, &mut agents)?;
        } else {
            anyhow::bail!(
                "invalid plugin path '{}': component does not exist under '{}'",
                agent_path,
                tree.root_display().display()
            );
        }
    }

    for agent in &mut agents {
        agent.prompt = render_plugin_vars(tree.root_display(), &agent.prompt);
        agent.revision = hash_bytes(agent.prompt.as_bytes());
    }

    let hooks = load_plugin_hooks(tree, &plugin_id, &manifest, options.exec_policy)?;

    Ok(Some(LoadedPlugin {
        id: plugin_id,
        root: tree.root_display().to_path_buf(),
        manifest,
        skills,
        agents,
        hooks,
        mcp_servers: Vec::new(),
        lsp_servers: Vec::new(),
        output_styles: Vec::new(),
        bin_paths: Vec::new(),
        defaults: PluginDefaults::default(),
    }))
}

fn collect_skills(
    root: &Path,
    visited: &mut BTreeSet<PathBuf>,
    sink: &mut Vec<LoadedSkill>,
) -> anyhow::Result<()> {
    let canonical_root = fs::canonicalize(root).with_context(|| {
        format!(
            "failed to canonicalize skill collection root '{}'",
            root.display()
        )
    })?;
    if !visited.insert(canonical_root) {
        return Ok(());
    }

    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            let skill_md = path.join("SKILL.md");
            if skill_md.exists() {
                sink.push(load_skill_root(&path)?);
            } else {
                collect_skills(&path, visited, sink)?;
            }
        }
    }
    Ok(())
}

fn load_skill_root(root: &Path) -> anyhow::Result<LoadedSkill> {
    let body = fs::read_to_string(root.join("SKILL.md"))
        .with_context(|| format!("failed to read {}", root.join("SKILL.md").display()))?;
    let frontmatter = parse_frontmatter(&body);
    let name = frontmatter.get("name").cloned().unwrap_or_else(|| {
        root.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string()
    });
    let description = frontmatter.get("description").cloned().unwrap_or_default();
    Ok(LoadedSkill {
        id: stable_skill_id(root)?,
        name,
        description,
        root: root.to_path_buf(),
        revision: hash_bytes(body.as_bytes()),
        body,
        supporting_files: Vec::new(),
        scripts: load_scripts(root)?,
    })
}

fn stable_skill_id(root: &Path) -> anyhow::Result<SkillId> {
    let canonical_root = fs::canonicalize(root).with_context(|| {
        format!(
            "failed to canonicalize skill root '{}' while computing a stable id",
            root.display()
        )
    })?;
    Ok(stable_skill_id_from_path(&canonical_root))
}

fn stable_skill_id_from_path(path: &Path) -> SkillId {
    SkillId::from(format!(
        "skill-{}",
        hash_bytes(path.display().to_string().as_bytes())
    ))
}

fn load_scripts(root: &Path) -> anyhow::Result<Vec<LoadedExecutable>> {
    let scripts_dir = root.join("scripts");
    if !scripts_dir.exists() {
        return Ok(Vec::new());
    }

    let mut scripts = Vec::new();
    for entry in fs::read_dir(scripts_dir)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            scripts.push(LoadedExecutable { path: entry.path() });
        }
    }
    Ok(scripts)
}

fn collect_skills_in_tree<T: PluginTree>(
    tree: &T,
    rel: &Path,
    visited: &mut BTreeSet<PathBuf>,
    sink: &mut Vec<LoadedSkill>,
) -> anyhow::Result<()> {
    let stable = tree.stable_path(rel);
    if !visited.insert(stable) {
        return Ok(());
    }

    for entry in tree.read_dir(rel)? {
        if entry.kind == TreeEntryKind::Dir {
            let skill_md = join_relative_path(&entry.rel, Path::new("SKILL.md"));
            if tree.is_file(&skill_md) {
                sink.push(load_skill_from_tree(tree, &entry.rel)?);
            } else {
                collect_skills_in_tree(tree, &entry.rel, visited, sink)?;
            }
        }
    }
    Ok(())
}

fn load_skill_from_tree<T: PluginTree>(tree: &T, rel: &Path) -> anyhow::Result<LoadedSkill> {
    let skill_md = join_relative_path(rel, Path::new("SKILL.md"));
    let body =
        String::from_utf8(tree.read(&skill_md).with_context(|| {
            format!("failed to read {}", tree.display_path(&skill_md).display())
        })?)
        .with_context(|| {
            format!(
                "skill file at {} is not UTF-8",
                tree.display_path(&skill_md).display()
            )
        })?;
    let frontmatter = parse_frontmatter(&body);
    let name = frontmatter.get("name").cloned().unwrap_or_else(|| {
        rel.file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string()
    });
    let description = frontmatter.get("description").cloned().unwrap_or_default();
    Ok(LoadedSkill {
        id: stable_skill_id_from_path(&tree.stable_path(rel)),
        name,
        description,
        root: tree.display_path(rel),
        revision: hash_bytes(body.as_bytes()),
        body,
        supporting_files: Vec::new(),
        scripts: Vec::new(),
    })
}

fn load_agent_dir_from_tree<T: PluginTree>(
    tree: &T,
    rel: &Path,
    sink: &mut Vec<LoadedAgent>,
) -> anyhow::Result<()> {
    for entry in tree.read_dir(rel)? {
        if entry.kind == TreeEntryKind::File {
            sink.push(load_agent_file_from_tree(tree, &entry.rel)?);
        }
    }
    Ok(())
}

fn load_agent_file_from_tree<T: PluginTree>(tree: &T, rel: &Path) -> anyhow::Result<LoadedAgent> {
    let prompt = String::from_utf8(tree.read(rel).with_context(|| {
        format!(
            "failed to read agent prompt at {}",
            tree.display_path(rel).display()
        )
    })?)
    .with_context(|| {
        format!(
            "agent prompt at {} is not UTF-8",
            tree.display_path(rel).display()
        )
    })?;
    let revision = hash_bytes(prompt.as_bytes());
    Ok(LoadedAgent {
        id: AgentId::new(),
        name: rel
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string(),
        prompt,
        revision,
    })
}

fn load_plugin_hooks<T: PluginTree>(
    tree: &T,
    plugin_id: &PluginId,
    manifest: &PluginManifest,
    exec_policy: ExecComponentPolicy,
) -> anyhow::Result<Vec<LoadedHooksFile>> {
    let hooks_rel = match manifest.hooks.as_deref() {
        Some(component) => Some(resolve_plugin_path(component)?),
        None => {
            let candidate = PathBuf::from("hooks/hooks.json");
            tree.is_file(&candidate).then_some(candidate)
        }
    };
    let Some(source_rel) = hooks_rel else {
        return Ok(Vec::new());
    };

    let body = tree.read(&source_rel).with_context(|| {
        format!(
            "failed to read hooks file at {}",
            tree.display_path(&source_rel).display()
        )
    })?;
    let revision = hash_bytes(&body);
    let (parsed, warnings) = match HooksFile::from_json_bytes(&body) {
        Ok(result) => result,
        Err(error) => (
            HooksFile::default(),
            vec![HooksLoadWarning::new("parse_error", error.to_string())],
        ),
    };
    #[cfg(not(feature = "remote-plugins"))]
    let _ = exec_policy;
    #[cfg(feature = "remote-plugins")]
    let (parsed, warnings) = {
        let mut parsed = parsed;
        let mut warnings = warnings;
        if exec_policy == ExecComponentPolicy::WarnAndSkip {
            skip_exec_backed_hooks(&mut parsed, &mut warnings);
        }
        (parsed, warnings)
    };

    Ok(vec![LoadedHooksFile {
        plugin_id: plugin_id.clone(),
        plugin_root: tree.root_display().to_path_buf(),
        source_path: tree.display_path(&source_rel),
        revision,
        parsed,
        warnings,
    }])
}

#[cfg(feature = "remote-plugins")]
fn skip_exec_backed_hooks(parsed: &mut HooksFile, warnings: &mut Vec<HooksLoadWarning>) {
    let mut skipped = 0usize;
    parsed.hooks.retain(|_, groups| {
        groups.retain_mut(|group| {
            let before = group.hooks.len();
            group
                .hooks
                .retain(|hook| !matches!(hook.config, HookHandlerConfig::Command(_)));
            skipped += before.saturating_sub(group.hooks.len());
            !group.hooks.is_empty()
        });
        !groups.is_empty()
    });

    if skipped > 0 {
        warnings.push(HooksLoadWarning::new(
            "remote_exec_component",
            format!("skipped {skipped} command hook(s) from remote in-memory plugin"),
        ));
    }
}

fn parse_frontmatter(body: &str) -> BTreeMap<String, String> {
    let mut fields = BTreeMap::new();
    let mut lines = body.lines();
    if lines.next() != Some("---") {
        return fields;
    }

    for line in lines {
        if line.trim() == "---" {
            break;
        }
        if let Some((key, value)) = line.split_once(':') {
            fields.insert(key.trim().to_owned(), value.trim().to_owned());
        }
    }
    fields
}

fn parse_plugin_manifest(value: &Value, path: &Path) -> anyhow::Result<PluginManifest> {
    Ok(PluginManifest {
        name: value
            .get("name")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "plugin manifest at {} is missing required string field 'name'",
                    path.display()
                )
            })?
            .to_owned(),
        version: value
            .get("version")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "plugin manifest at {} is missing required string field 'version'",
                    path.display()
                )
            })?
            .to_owned(),
        skills: read_string_or_array(value, "skills"),
        agents: read_string_or_array(value, "agents"),
        hooks: value
            .get("hooks")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        mcp_servers: value
            .get("mcpServers")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        lsp_servers: value
            .get("lspServers")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        allowed_http_hosts: read_string_or_array(value, "allowedHttpHosts"),
        allowed_env_vars: read_string_or_array(value, "allowedEnvVars"),
    })
}

fn read_string_or_array(value: &Value, key: &str) -> Vec<String> {
    match value.get(key) {
        Some(Value::String(raw)) => trimmed_non_empty(raw).into_iter().collect(),
        Some(Value::Array(values)) => values
            .iter()
            .filter_map(Value::as_str)
            .filter_map(trimmed_non_empty)
            .collect(),
        _ => Vec::new(),
    }
}

fn trimmed_non_empty(value: &str) -> Option<String> {
    let trimmed = value.trim();
    (!trimmed.is_empty()).then(|| trimmed.to_owned())
}

fn manifest_candidates(search: ManifestSearch) -> &'static [&'static str] {
    match search {
        ManifestSearch::DiskCompatible => &[
            ".codex-plugin/plugin.json",
            ".claude-plugin/plugin.json",
            ".agent-plugin/plugin.json",
            ".halter-plugin/plugin.json",
            "plugin.json",
        ],
        #[cfg(feature = "remote-plugins")]
        ManifestSearch::RemoteCodexClaude => {
            &[".codex-plugin/plugin.json", ".claude-plugin/plugin.json"]
        }
    }
}

fn resolve_plugin_path(component: &str) -> anyhow::Result<PathBuf> {
    let uses_plugin_alias = component.contains("${CLAUDE_PLUGIN_ROOT}")
        || component.contains("${PLUGIN_ROOT}")
        || component.contains("${CLAUDE_PLUGIN_DATA}")
        || component.contains("${PLUGIN_DATA}");

    if !component.starts_with("./") && !uses_plugin_alias {
        anyhow::bail!(
            "invalid plugin path '{}': relative component paths must start with './'",
            component
        );
    }

    let expanded = component
        .replace("${CLAUDE_PLUGIN_ROOT}", ".")
        .replace("${PLUGIN_ROOT}", ".")
        .replace("${CLAUDE_PLUGIN_DATA}", "./.data")
        .replace("${PLUGIN_DATA}", "./.data");
    normalize_relative_path(Path::new(&expanded), component)
}

pub(crate) fn normalize_relative_path(path: &Path, original: &str) -> anyhow::Result<PathBuf> {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::Normal(part) => normalized.push(part),
            Component::ParentDir => {
                if !normalized.pop() {
                    anyhow::bail!(
                        "invalid plugin path '{}': component resolves outside the plugin root",
                        original
                    );
                }
            }
            Component::RootDir | Component::Prefix(_) => {
                anyhow::bail!(
                    "invalid plugin path '{}': component resolves outside the plugin root",
                    original
                );
            }
        }
    }
    Ok(normalized)
}

pub(crate) fn join_relative_path(base: &Path, rel: &Path) -> PathBuf {
    if base.as_os_str().is_empty() {
        rel.to_path_buf()
    } else if rel.as_os_str().is_empty() {
        base.to_path_buf()
    } else {
        base.join(rel)
    }
}

fn join_display_path(base: &Path, rel: &Path) -> PathBuf {
    if rel.as_os_str().is_empty() {
        base.to_path_buf()
    } else {
        base.join(rel)
    }
}

#[cfg(feature = "remote-plugins")]
fn path_has_prefix(path: &Path, prefix: &Path) -> bool {
    prefix.as_os_str().is_empty() || path.starts_with(prefix)
}

#[cfg(feature = "remote-plugins")]
fn strip_prefix_components(
    path: &Path,
    prefix: &Path,
) -> Result<PathBuf, std::path::StripPrefixError> {
    if prefix.as_os_str().is_empty() {
        Ok(path.to_path_buf())
    } else {
        path.strip_prefix(prefix).map(Path::to_path_buf)
    }
}

fn hash_bytes(bytes: &[u8]) -> ContentHash {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn stable_plugin_id(root: &Path, manifest: &PluginManifest) -> PluginId {
    let fingerprint = format!(
        "{}\0{}\0{}",
        manifest.name,
        manifest.version,
        root.display()
    );
    PluginId::from(format!("plugin-{}", hash_bytes(fingerprint.as_bytes())))
}

/// Expand plugin template variables in `text`.
///
/// Supported aliases:
/// - `${CLAUDE_PLUGIN_ROOT}` / `${PLUGIN_ROOT}` -> the plugin (or skill) root.
/// - `${CLAUDE_PLUGIN_DATA}` / `${PLUGIN_DATA}` -> `<root>/.data`.
///
/// Unknown `${...}` tokens are left literal. For standalone skills, `root` is
/// the skill directory itself, so `PLUGIN_DATA` resolves to `<skill_root>/.data`.
fn render_plugin_vars(root: &Path, text: &str) -> String {
    let plugin_root = root.to_string_lossy().to_string();
    let plugin_data = root.join(".data").to_string_lossy().to_string();
    text.replace("${CLAUDE_PLUGIN_ROOT}", &plugin_root)
        .replace("${PLUGIN_ROOT}", &plugin_root)
        .replace("${CLAUDE_PLUGIN_DATA}", &plugin_data)
        .replace("${PLUGIN_DATA}", &plugin_data)
}

fn normalized_roots(roots: &[PathBuf]) -> Vec<PathBuf> {
    let mut seen = BTreeSet::new();
    let mut normalized = Vec::new();
    for root in roots {
        let expanded = crate::expand_path(root);
        let key = expanded.to_string_lossy().to_string();
        if seen.insert(key) {
            normalized.push(expanded);
        }
    }
    normalized
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "remote-plugins")]
    use halter_hooks::{HookEventName, HookHandlerConfig};

    fn write_plugin(root: &Path, manifest_dir: &str, manifest: &str, skill_body: &str) -> PathBuf {
        let plugin_root = root.join("plugin");
        let skill_dir = plugin_root.join("skills/reviewer");
        let manifest_dir = plugin_root.join(manifest_dir);
        fs::create_dir_all(&skill_dir).expect("create skill dir");
        fs::create_dir_all(&manifest_dir).expect("create manifest dir");
        fs::write(skill_dir.join("SKILL.md"), skill_body).expect("write skill");
        fs::write(manifest_dir.join("plugin.json"), manifest).expect("write manifest");
        plugin_root
    }

    #[test]
    fn plugin_loader_prefers_codex_manifest_and_supports_string_paths() {
        let temp = tempfile::tempdir().expect("tempdir");
        let plugin_root = temp.path().join("plugin");
        let codex_dir = plugin_root.join(".codex-plugin");
        let claude_dir = plugin_root.join(".claude-plugin");
        let skill_dir = plugin_root.join("skills/codex");
        fs::create_dir_all(&codex_dir).expect("codex dir");
        fs::create_dir_all(&claude_dir).expect("claude dir");
        fs::create_dir_all(&skill_dir).expect("skill dir");
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: codex-skill\n---\n\nUse ${PLUGIN_ROOT}.\n",
        )
        .expect("skill");
        fs::write(
            codex_dir.join("plugin.json"),
            r#"{"name":"codex-plugin","version":"1.0.0","skills":"./skills/"}"#,
        )
        .expect("codex manifest");
        fs::write(
            claude_dir.join("plugin.json"),
            r#"{"name":"claude-plugin","version":"1.0.0"}"#,
        )
        .expect("claude manifest");

        let tree = DiskTree::new(plugin_root).expect("tree");
        let plugin = load_plugin_tree(&tree, PluginLoadOptions::disk_scan())
            .expect("load")
            .expect("plugin");

        assert_eq!(plugin.manifest.name, "codex-plugin");
        assert_eq!(plugin.skills.len(), 1);
        assert_eq!(plugin.skills[0].name, "codex-skill");
    }

    #[test]
    #[cfg(feature = "remote-plugins")]
    fn remote_plugin_requires_codex_or_claude_manifest() {
        let tree = MemTree::new(PathBuf::from("github:acme/missing@sha"), BTreeMap::new());
        let error = load_plugin_tree(&tree, PluginLoadOptions::remote_install())
            .expect_err("missing manifest should fail");

        assert!(
            error.to_string().contains("missing a supported manifest"),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    #[cfg(feature = "remote-plugins")]
    fn remote_mem_tree_loads_content_plugin() {
        let mut files = BTreeMap::new();
        files.insert(
            PathBuf::from(".claude-plugin/plugin.json"),
            br#"{"name":"remote","version":"0.1.0","skills":["./skills/reviewer"],"agents":["./agents"]}"#.to_vec(),
        );
        files.insert(
            PathBuf::from("skills/reviewer/SKILL.md"),
            b"---\nname: reviewer\n---\n\nReview from ${PLUGIN_ROOT}.\n".to_vec(),
        );
        files.insert(
            PathBuf::from("agents/helper.md"),
            b"Help from ${PLUGIN_ROOT}.".to_vec(),
        );
        let tree = MemTree::new(PathBuf::from("github:acme/remote@abc123"), files);

        let plugin = load_plugin_tree(&tree, PluginLoadOptions::remote_install())
            .expect("load")
            .expect("plugin");

        assert_eq!(plugin.manifest.name, "remote");
        assert_eq!(plugin.skills[0].name, "reviewer");
        assert!(plugin.skills[0].body.contains("github:acme/remote@abc123"));
        assert_eq!(plugin.agents[0].name, "helper");
    }

    #[test]
    #[cfg(feature = "remote-plugins")]
    fn mem_tree_rejects_parent_traversal() {
        let mut files = BTreeMap::new();
        files.insert(
            PathBuf::from(".claude-plugin/plugin.json"),
            br#"{"name":"unsafe","version":"0.1.0","skills":["./../outside"]}"#.to_vec(),
        );
        let tree = MemTree::new(PathBuf::from("github:acme/unsafe@abc123"), files);

        let error = load_plugin_tree(&tree, PluginLoadOptions::remote_install())
            .expect_err("traversal should fail");

        assert!(
            error
                .to_string()
                .contains("component resolves outside the plugin root"),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    #[cfg(feature = "remote-plugins")]
    fn remote_command_hooks_are_warned_and_skipped() {
        let mut files = BTreeMap::new();
        files.insert(
            PathBuf::from(".claude-plugin/plugin.json"),
            br#"{"name":"hooks","version":"0.1.0","hooks":"./hooks/hooks.json"}"#.to_vec(),
        );
        files.insert(
            PathBuf::from("hooks/hooks.json"),
            br#"{"hooks":{"PreToolUse":[{"hooks":[{"type":"command","command":"./run.sh"},{"type":"prompt","prompt":"check"}]}]}}"#.to_vec(),
        );
        let tree = MemTree::new(PathBuf::from("github:acme/hooks@abc123"), files);

        let plugin = load_plugin_tree(&tree, PluginLoadOptions::remote_install())
            .expect("load")
            .expect("plugin");

        let hooks = plugin.hooks.first().expect("hooks file");
        assert!(
            hooks
                .warnings
                .iter()
                .any(|warning| warning.category == "remote_exec_component")
        );
        let groups = hooks
            .parsed
            .hooks
            .get(&HookEventName::PreToolUse)
            .expect("pre tool hooks");
        assert_eq!(groups[0].hooks.len(), 1);
        assert!(matches!(
            groups[0].hooks[0].config,
            HookHandlerConfig::Prompt(_)
        ));
    }

    #[test]
    fn render_plugin_vars_leaves_unknown_tokens_literal() {
        let temp = tempfile::tempdir().expect("tempdir");
        let input = "token=${UNKNOWN_VAR}/foo unchanged";
        let output = render_plugin_vars(temp.path(), input);
        assert_eq!(output, input);
    }

    #[test]
    fn stable_skill_id_errors_on_nonexistent_path() {
        let temp = tempfile::tempdir().expect("tempdir");
        let missing = temp.path().join("does_not_exist");
        let error = stable_skill_id(&missing).expect_err("canonicalize should fail");
        assert!(
            error
                .to_string()
                .contains("failed to canonicalize skill root"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn claude_manifest_fallback_loads_disk_plugin() {
        let temp = tempfile::tempdir().expect("tempdir");
        let plugin_root = write_plugin(
            temp.path(),
            ".claude-plugin",
            r#"{"name":"example","version":"0.1.0","skills":["./skills"]}"#,
            "---\nname: reviewer\n---\n\n# Reviewer\n",
        );
        let tree = DiskTree::new(plugin_root).expect("tree");

        let plugin = load_plugin_tree(&tree, PluginLoadOptions::disk_scan())
            .expect("load")
            .expect("plugin");

        assert_eq!(plugin.manifest.name, "example");
        assert_eq!(plugin.skills[0].name, "reviewer");
    }
}
