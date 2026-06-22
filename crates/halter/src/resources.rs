// pattern: Imperative Shell

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use halter_config::HarnessConfig;
use halter_hooks::{HookRegistrySource, Hooks, HooksFile, HooksLoadWarning};
use halter_protocol::{
    AgentDef, AgentId, AgentName, ContentHash, HookWarning, HookWarningSeverity, InstructionFile,
    PluginId, PluginManifest, PromptRegistry, ResourceSnapshot, Revision, SkillDef, SkillId,
    SkillName,
};
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

#[derive(Clone, Debug)]
/// Resource snapshot plus compiled hooks ready for runtime use.
pub struct CompiledResources {
    pub snapshot: ResourceSnapshot,
    pub hooks: Arc<Hooks>,
    pub hook_warnings: Vec<HookWarning>,
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
        for root in normalized_roots(roots)? {
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
        for root in normalized_roots(roots)? {
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
                    if let Some(plugin) = load_plugin_root(&plugin_root)? {
                        plugins.push(plugin);
                    }
                }
            }
        }
        info!(plugin_count = plugins.len(), "loaded plugins");
        Ok(plugins)
    }
}

#[derive(Debug, Clone)]
/// Compiles loaded or on-disk resources into a runtime snapshot.
pub struct ResourceCompiler {
    config: HarnessConfig,
    skill_roots: Option<Vec<PathBuf>>,
    plugin_roots: Option<Vec<PathBuf>>,
    loaded_skills: Vec<LoadedSkill>,
    loaded_plugins: Vec<LoadedPlugin>,
}

impl ResourceCompiler {
    /// Start from harness config defaults.
    #[must_use]
    pub fn from_config(config: &HarnessConfig) -> Self {
        Self {
            config: config.clone(),
            skill_roots: None,
            plugin_roots: None,
            loaded_skills: Vec::new(),
            loaded_plugins: Vec::new(),
        }
    }

    /// Override skill search roots.
    #[must_use]
    pub fn with_skill_roots(mut self, roots: Vec<PathBuf>) -> Self {
        self.skill_roots = Some(roots);
        self
    }

    /// Override plugin search roots.
    #[must_use]
    pub fn with_plugin_roots(mut self, roots: Vec<PathBuf>) -> Self {
        self.plugin_roots = Some(roots);
        self
    }

    /// Use preloaded skills instead of reading skill roots.
    #[must_use]
    pub fn with_loaded_skills(mut self, skills: Vec<LoadedSkill>) -> Self {
        self.loaded_skills = skills;
        self
    }

    /// Use preloaded plugins instead of reading plugin roots.
    #[must_use]
    pub fn with_loaded_plugins(mut self, plugins: Vec<LoadedPlugin>) -> Self {
        self.loaded_plugins = plugins;
        self
    }

    /// Compile resources on a blocking worker.
    pub async fn compile(self) -> anyhow::Result<CompiledResources> {
        tokio::task::spawn_blocking(move || compile_resources(self))
            .await
            .context("failed to join resource compiler task")?
    }
}

fn compile_resources(compiler: ResourceCompiler) -> anyhow::Result<CompiledResources> {
    let ResourceCompiler {
        config,
        skill_roots,
        plugin_roots,
        loaded_skills,
        loaded_plugins,
    } = compiler;
    let skill_loader = SkillLoader;
    let plugin_loader = PluginLoader;
    debug!("compiling resource snapshot");

    let mut skills = if loaded_skills.is_empty() {
        skill_loader.load_roots(
            skill_roots
                .as_deref()
                .unwrap_or(&config.resources.skills.roots),
        )?
    } else {
        loaded_skills
    };

    let plugins = if loaded_plugins.is_empty() {
        plugin_loader.load_roots(
            plugin_roots
                .as_deref()
                .unwrap_or(&config.resources.plugins.roots),
        )?
    } else {
        loaded_plugins
    };

    for plugin in &plugins {
        skills.extend(plugin.skills.clone());
    }
    info!(
        skill_count = skills.len(),
        plugin_count = plugins.len(),
        "assembled resource inputs"
    );

    let mut snapshot = ResourceSnapshot::empty();
    let mut revision_hasher = Sha256::new();

    for skill in skills {
        revision_hasher.update(skill.revision.as_bytes());
        snapshot.skills.insert(
            SkillName::from(skill.name.clone()),
            SkillDef {
                id: skill.id,
                name: skill.name,
                description: skill.description,
                body: skill.body,
            },
        );
    }

    let hook_sources = plugins
        .iter()
        .flat_map(|plugin| {
            plugin
                .hooks
                .iter()
                .cloned()
                .map(|hooks_file| HookRegistrySource {
                    plugin_id: hooks_file.plugin_id,
                    plugin_root: hooks_file.plugin_root,
                    source_path: hooks_file.source_path,
                    allowed_http_hosts: plugin.manifest.allowed_http_hosts.clone(),
                    allowed_env_vars: plugin.manifest.allowed_env_vars.clone(),
                    file: hooks_file.parsed,
                })
        })
        .collect::<Vec<_>>();
    let hook_warnings = plugins
        .iter()
        .flat_map(|plugin| {
            plugin.hooks.iter().flat_map(move |hooks_file| {
                hooks_file.warnings.iter().map(move |warning| HookWarning {
                    severity: HookWarningSeverity::Warning,
                    category: warning.category.clone(),
                    plugin_id: Some(plugin.id.clone()),
                    plugin_name: Some(plugin.manifest.name.clone()),
                    source_path: Some(hooks_file.source_path.clone()),
                    message: warning.message.clone(),
                })
            })
        })
        .collect::<Vec<_>>();

    for plugin in plugins {
        revision_hasher.update(plugin.manifest.name.as_bytes());
        revision_hasher.update(plugin.manifest.version.as_bytes());
        for hooks_file in &plugin.hooks {
            revision_hasher.update(hooks_file.revision.as_bytes());
        }
        for agent in &plugin.agents {
            revision_hasher.update(agent.revision.as_bytes());
            snapshot.agents.insert(
                AgentName::from(agent.name.clone()),
                AgentDef {
                    id: agent.id.clone(),
                    name: agent.name.clone(),
                    prompt: agent.prompt.clone(),
                },
            );
        }
        snapshot.plugins.insert(plugin.id, plugin.manifest);
    }

    snapshot.prompts = PromptRegistry::default();
    snapshot.instruction_files.push(InstructionFile {
        path: PathBuf::from("generated://resource-compiler"),
        body: "Resources were compiled before runtime instantiation.".to_owned(),
    });
    snapshot.revision = Revision::from(format!("{:x}", revision_hasher.finalize()));
    info!(revision = %snapshot.revision, "compiled resource snapshot");
    Ok(CompiledResources {
        snapshot,
        hooks: Arc::new(Hooks::from_sources(hook_sources)),
        hook_warnings,
    })
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
    let fingerprint = canonical_root.display().to_string();
    Ok(SkillId::from(format!(
        "skill-{}",
        hash_bytes(fingerprint.as_bytes())
    )))
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

fn load_plugin_root(root: &Path) -> anyhow::Result<Option<LoadedPlugin>> {
    let Some(manifest_path) = [
        root.join(".claude-plugin/plugin.json"),
        root.join(".agent-plugin/plugin.json"),
        root.join(".halter-plugin/plugin.json"),
        root.join("plugin.json"),
    ]
    .into_iter()
    .find(|path| path.exists()) else {
        return Ok(None);
    };

    let manifest_value: Value = serde_json::from_str(&fs::read_to_string(&manifest_path)?)
        .with_context(|| {
            format!(
                "failed to parse plugin manifest at {}",
                manifest_path.display()
            )
        })?;

    let manifest = PluginManifest {
        name: manifest_value
            .get("name")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "plugin manifest at {} is missing required string field 'name'",
                    manifest_path.display()
                )
            })?
            .to_owned(),
        version: manifest_value
            .get("version")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "plugin manifest at {} is missing required string field 'version'",
                    manifest_path.display()
                )
            })?
            .to_owned(),
        skills: read_string_array(&manifest_value, "skills"),
        agents: read_string_array(&manifest_value, "agents"),
        hooks: manifest_value
            .get("hooks")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        mcp_servers: manifest_value
            .get("mcpServers")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        lsp_servers: manifest_value
            .get("lspServers")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        allowed_http_hosts: read_string_array(&manifest_value, "allowedHttpHosts"),
        allowed_env_vars: read_string_array(&manifest_value, "allowedEnvVars"),
    };
    let plugin_id = stable_plugin_id(root, &manifest)?;

    let mut skills = Vec::new();
    let mut visited_skill_dirs = BTreeSet::new();
    for skill_path in &manifest.skills {
        let resolved = resolve_plugin_path(root, skill_path)?;
        if resolved.join("SKILL.md").exists() {
            skills.push(load_skill_root(&resolved)?);
        } else if resolved.is_dir() {
            collect_skills(&resolved, &mut visited_skill_dirs, &mut skills)?;
        }
    }

    skills.iter_mut().for_each(|skill| {
        skill.body = render_plugin_vars(root, &skill.body);
        skill.revision = hash_bytes(skill.body.as_bytes());
    });

    let mut agents = Vec::new();
    for agent_path in &manifest.agents {
        let resolved = resolve_plugin_path(root, agent_path)?;
        if resolved.is_file() {
            agents.push(load_agent_file(&resolved)?);
        } else if resolved.is_dir() {
            load_agent_dir(&resolved, &mut agents)?;
        }
    }

    for agent in &mut agents {
        agent.prompt = render_plugin_vars(root, &agent.prompt);
        agent.revision = hash_bytes(agent.prompt.as_bytes());
    }

    let hooks = load_plugin_hooks(root, &plugin_id, &manifest)?;

    Ok(Some(LoadedPlugin {
        id: plugin_id,
        root: root.to_path_buf(),
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

fn resolve_plugin_path(root: &Path, component: &str) -> anyhow::Result<PathBuf> {
    let expanded = expand_plugin_component_path(root, component);
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

    let candidate = PathBuf::from(expanded);
    reject_parent_components(&candidate, component)?;

    let canonical_root = fs::canonicalize(root)
        .with_context(|| format!("failed to canonicalize plugin root '{}'", root.display()))?;
    let resolved = if candidate.is_absolute() {
        candidate
    } else {
        canonical_root.join(candidate)
    };
    let canonical_resolved = fs::canonicalize(&resolved).with_context(|| {
        format!(
            "invalid plugin path '{}': component does not exist under '{}'",
            component,
            root.display()
        )
    })?;

    if !canonical_resolved.starts_with(&canonical_root) {
        anyhow::bail!(
            "invalid plugin path '{}': component resolves outside the plugin root",
            component
        );
    }
    Ok(canonical_resolved)
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

fn read_string_array(value: &Value, key: &str) -> Vec<String> {
    value
        .get(key)
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|value| value.as_str().map(ToOwned::to_owned))
        .collect()
}

fn hash_bytes(bytes: &[u8]) -> ContentHash {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

fn stable_plugin_id(root: &Path, manifest: &PluginManifest) -> anyhow::Result<PluginId> {
    let canonical_root = fs::canonicalize(root).with_context(|| {
        format!(
            "failed to canonicalize plugin root '{}' while computing a stable id",
            root.display()
        )
    })?;
    let fingerprint = format!(
        "{}\0{}\0{}",
        manifest.name,
        manifest.version,
        canonical_root.display()
    );
    Ok(PluginId::from(format!(
        "plugin-{}",
        hash_bytes(fingerprint.as_bytes())
    )))
}

fn normalized_roots(roots: &[PathBuf]) -> anyhow::Result<Vec<PathBuf>> {
    let mut seen = BTreeSet::new();
    let mut normalized = Vec::new();
    for root in roots {
        let expanded = halter_config::expand_path(root);
        let key = expanded.to_string_lossy().to_string();
        if seen.insert(key) {
            normalized.push(expanded);
        }
    }
    Ok(normalized)
}

/// Expand plugin template variables in `text`.
///
/// Supported aliases:
/// - `${CLAUDE_PLUGIN_ROOT}` / `${PLUGIN_ROOT}` → the plugin (or skill) root.
/// - `${CLAUDE_PLUGIN_DATA}` / `${PLUGIN_DATA}` → `<root>/.data`.
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

fn expand_plugin_component_path(root: &Path, component: &str) -> String {
    render_plugin_vars(root, component)
}

fn reject_parent_components(path: &Path, component: &str) -> anyhow::Result<()> {
    if path
        .components()
        .any(|part| matches!(part, Component::ParentDir))
    {
        anyhow::bail!(
            "invalid plugin path '{}': component resolves outside the plugin root",
            component
        );
    }
    Ok(())
}

fn load_agent_dir(root: &Path, sink: &mut Vec<LoadedAgent>) -> anyhow::Result<()> {
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            sink.push(load_agent_file(&entry.path())?);
        }
    }
    Ok(())
}

fn load_agent_file(path: &Path) -> anyhow::Result<LoadedAgent> {
    let prompt = fs::read_to_string(path)
        .with_context(|| format!("failed to read agent prompt at {}", path.display()))?;
    let revision = hash_bytes(prompt.as_bytes());
    Ok(LoadedAgent {
        id: AgentId::new(),
        name: path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string(),
        prompt,
        revision,
    })
}

fn load_plugin_hooks(
    root: &Path,
    plugin_id: &PluginId,
    manifest: &PluginManifest,
) -> anyhow::Result<Vec<LoadedHooksFile>> {
    let hooks_path = match manifest.hooks.as_deref() {
        Some(component) => Some(resolve_plugin_path(root, component)?),
        None => {
            let candidate = root.join("hooks/hooks.json");
            candidate.exists().then_some(candidate)
        }
    };
    let Some(source_path) = hooks_path else {
        return Ok(Vec::new());
    };

    let body = fs::read(&source_path)
        .with_context(|| format!("failed to read hooks file at {}", source_path.display()))?;
    let revision = hash_bytes(&body);
    let (parsed, warnings) = match HooksFile::from_json_bytes(&body) {
        Ok(result) => result,
        Err(error) => (
            HooksFile::default(),
            vec![HooksLoadWarning::new("parse_error", error.to_string())],
        ),
    };

    Ok(vec![LoadedHooksFile {
        plugin_id: plugin_id.clone(),
        plugin_root: root.to_path_buf(),
        source_path,
        revision,
        parsed,
        warnings,
    }])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn resource_compiler_loads_skill_roots() {
        let temp = tempfile::tempdir().expect("tempdir");
        let skill_dir = temp.path().join("skills/hello");
        fs::create_dir_all(&skill_dir).expect("create skill dir");
        fs::write(
            skill_dir.join("SKILL.md"),
            r#"---
name: hello
description: says hello
---

# Hello
"#,
        )
        .expect("write skill");

        let mut config = HarnessConfig::default();
        config.resources.skills.roots = vec![temp.path().join("skills")];
        let resources = ResourceCompiler::from_config(&config)
            .compile()
            .await
            .expect("compile resources");

        assert!(
            resources
                .snapshot
                .skills
                .contains_key(&SkillName::from("hello"))
        );
    }

    #[tokio::test]
    async fn resource_compiler_loads_claude_manifest_directory_components() {
        let temp = tempfile::tempdir().expect("tempdir");
        let plugin_root = temp.path().join("plugin");
        let skill_dir = plugin_root.join("skills/reviewer");
        let agents_dir = plugin_root.join("agents");
        let manifest_dir = plugin_root.join(".claude-plugin");

        fs::create_dir_all(&skill_dir).expect("create skill dir");
        fs::create_dir_all(&agents_dir).expect("create agents dir");
        fs::create_dir_all(&manifest_dir).expect("create manifest dir");
        fs::write(
            skill_dir.join("SKILL.md"),
            r#"---
name: reviewer
description: reviews code
---

# Reviewer
"#,
        )
        .expect("write skill");
        fs::write(agents_dir.join("helper.md"), "You are a helper.").expect("write agent prompt");
        fs::write(
            manifest_dir.join("plugin.json"),
            r#"{
  "name": "example",
  "version": "0.1.0",
  "skills": ["${PLUGIN_ROOT}/skills"],
  "agents": ["./agents"]
}"#,
        )
        .expect("write manifest");

        let mut config = HarnessConfig::default();
        config.resources.plugins.roots = vec![temp.path().to_path_buf()];
        let resources = ResourceCompiler::from_config(&config)
            .compile()
            .await
            .expect("compile resources");

        assert!(
            resources
                .snapshot
                .skills
                .contains_key(&SkillName::from("reviewer"))
        );
        assert!(
            resources
                .snapshot
                .agents
                .contains_key(&AgentName::from("helper"))
        );
    }

    #[tokio::test]
    async fn resource_compiler_rejects_plugin_path_traversal() {
        let temp = tempfile::tempdir().expect("tempdir");
        let plugin_root = temp.path().join("plugin");
        let manifest_dir = plugin_root.join(".agent-plugin");

        fs::create_dir_all(&manifest_dir).expect("create manifest dir");
        fs::write(
            manifest_dir.join("plugin.json"),
            r#"{
  "name": "unsafe",
  "version": "0.1.0",
  "skills": ["./../../outside"]
}"#,
        )
        .expect("write manifest");

        let mut config = HarnessConfig::default();
        config.resources.plugins.roots = vec![temp.path().to_path_buf()];
        let error = ResourceCompiler::from_config(&config)
            .compile()
            .await
            .expect_err("compile should fail");

        assert!(
            error
                .to_string()
                .contains("component resolves outside the plugin root")
        );
    }

    #[tokio::test]
    async fn resource_compiler_uses_stable_plugin_ids() {
        let temp = tempfile::tempdir().expect("tempdir");
        let plugin_root = temp.path().join("plugin");
        let manifest_dir = plugin_root.join(".halter-plugin");

        fs::create_dir_all(&manifest_dir).expect("create manifest dir");
        fs::write(
            manifest_dir.join("plugin.json"),
            r#"{
  "name": "stable-plugin",
  "version": "1.2.3"
}"#,
        )
        .expect("write manifest");

        let mut config = HarnessConfig::default();
        config.resources.plugins.roots = vec![temp.path().to_path_buf()];

        let first = ResourceCompiler::from_config(&config)
            .compile()
            .await
            .expect("compile resources");
        let second = ResourceCompiler::from_config(&config)
            .compile()
            .await
            .expect("compile resources again");

        let first_id = first
            .snapshot
            .plugins
            .keys()
            .next()
            .cloned()
            .expect("plugin id");
        let second_id = second
            .snapshot
            .plugins
            .keys()
            .next()
            .cloned()
            .expect("plugin id");

        assert_eq!(first_id, second_id);
    }

    #[tokio::test]
    async fn resource_compiler_uses_stable_skill_ids() {
        let temp = tempfile::tempdir().expect("tempdir");
        let skill_dir = temp.path().join("skills/hello");
        fs::create_dir_all(&skill_dir).expect("create skill dir");
        fs::write(
            skill_dir.join("SKILL.md"),
            r#"---
name: hello
description: says hello
---

# Hello
"#,
        )
        .expect("write skill");

        let mut config = HarnessConfig::default();
        config.resources.skills.roots = vec![temp.path().join("skills")];

        let first = ResourceCompiler::from_config(&config)
            .compile()
            .await
            .expect("compile resources");
        let second = ResourceCompiler::from_config(&config)
            .compile()
            .await
            .expect("compile resources again");

        let first_id = first
            .snapshot
            .skills
            .get(&SkillName::from("hello"))
            .map(|s| s.id.clone())
            .expect("first skill id");
        let second_id = second
            .snapshot
            .skills
            .get(&SkillName::from("hello"))
            .map(|s| s.id.clone())
            .expect("second skill id");

        assert_eq!(first_id, second_id);
        assert!(
            first_id.0.starts_with("skill-"),
            "skill id should be content-addressed: {}",
            first_id.0
        );
    }

    #[tokio::test]
    async fn m8_plugin_manifest_missing_name_fails_closed() {
        let temp = tempfile::tempdir().expect("tempdir");
        let plugin_root = temp.path().join("plugin");
        let manifest_dir = plugin_root.join(".halter-plugin");

        fs::create_dir_all(&manifest_dir).expect("create manifest dir");
        fs::write(manifest_dir.join("plugin.json"), r#"{"version": "0.1.0"}"#)
            .expect("write manifest");

        let mut config = HarnessConfig::default();
        config.resources.plugins.roots = vec![temp.path().to_path_buf()];
        let error = ResourceCompiler::from_config(&config)
            .compile()
            .await
            .expect_err("compile should fail");

        assert!(
            error
                .to_string()
                .contains("missing required string field 'name'")
                || error.chain().any(|e| e
                    .to_string()
                    .contains("missing required string field 'name'")),
            "unexpected error: {error:?}"
        );
    }

    #[tokio::test]
    async fn m8_plugin_manifest_missing_version_fails_closed() {
        let temp = tempfile::tempdir().expect("tempdir");
        let plugin_root = temp.path().join("plugin");
        let manifest_dir = plugin_root.join(".halter-plugin");

        fs::create_dir_all(&manifest_dir).expect("create manifest dir");
        fs::write(manifest_dir.join("plugin.json"), r#"{"name": "example"}"#)
            .expect("write manifest");

        let mut config = HarnessConfig::default();
        config.resources.plugins.roots = vec![temp.path().to_path_buf()];
        let error = ResourceCompiler::from_config(&config)
            .compile()
            .await
            .expect_err("compile should fail");

        assert!(
            error
                .to_string()
                .contains("missing required string field 'version'")
                || error.chain().any(|e| e
                    .to_string()
                    .contains("missing required string field 'version'")),
            "unexpected error: {error:?}"
        );
    }

    #[tokio::test]
    async fn m8_plugin_manifest_blank_name_fails_closed() {
        let temp = tempfile::tempdir().expect("tempdir");
        let plugin_root = temp.path().join("plugin");
        let manifest_dir = plugin_root.join(".halter-plugin");

        fs::create_dir_all(&manifest_dir).expect("create manifest dir");
        fs::write(
            manifest_dir.join("plugin.json"),
            r#"{"name": "  ", "version": "0.1.0"}"#,
        )
        .expect("write manifest");

        let mut config = HarnessConfig::default();
        config.resources.plugins.roots = vec![temp.path().to_path_buf()];
        let error = ResourceCompiler::from_config(&config)
            .compile()
            .await
            .expect_err("compile should fail");

        assert!(
            error.chain().any(|e| e
                .to_string()
                .contains("missing required string field 'name'")),
            "unexpected error: {error:?}"
        );
    }

    #[test]
    fn m9_stable_skill_id_errors_on_nonexistent_path() {
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
    fn m9_stable_plugin_id_errors_on_nonexistent_path() {
        let temp = tempfile::tempdir().expect("tempdir");
        let missing = temp.path().join("does_not_exist");
        let manifest = PluginManifest {
            name: "x".to_owned(),
            version: "0.0.1".to_owned(),
            skills: Vec::new(),
            agents: Vec::new(),
            hooks: None,
            mcp_servers: None,
            lsp_servers: None,
            allowed_http_hosts: Vec::new(),
            allowed_env_vars: Vec::new(),
        };
        let error = stable_plugin_id(&missing, &manifest).expect_err("canonicalize should fail");
        assert!(
            error
                .to_string()
                .contains("failed to canonicalize plugin root"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn render_plugin_vars_replaces_all_four_aliases() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().to_path_buf();
        let data = root.join(".data");
        let input =
            "root=${CLAUDE_PLUGIN_ROOT}|${PLUGIN_ROOT} data=${CLAUDE_PLUGIN_DATA}|${PLUGIN_DATA}";
        let output = render_plugin_vars(&root, input);
        let expected_root = root.to_string_lossy().to_string();
        let expected_data = data.to_string_lossy().to_string();
        let expected =
            format!("root={expected_root}|{expected_root} data={expected_data}|{expected_data}");
        assert_eq!(output, expected);
    }

    #[test]
    fn render_plugin_vars_leaves_unknown_tokens_literal() {
        let temp = tempfile::tempdir().expect("tempdir");
        let input = "token=${UNKNOWN_VAR}/foo unchanged";
        let output = render_plugin_vars(temp.path(), input);
        assert_eq!(output, input);
    }

    #[tokio::test]
    async fn plugin_skill_body_renders_plugin_vars() {
        let temp = tempfile::tempdir().expect("tempdir");
        let plugin_root = temp.path().join("plugin");
        let skill_dir = plugin_root.join("skills/reviewer");
        let manifest_dir = plugin_root.join(".claude-plugin");

        fs::create_dir_all(&skill_dir).expect("create skill dir");
        fs::create_dir_all(&manifest_dir).expect("create manifest dir");
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: reviewer\n---\n\nUse ${PLUGIN_ROOT}/assets/foo.md and cache at ${CLAUDE_PLUGIN_DATA}.\n",
        )
        .expect("write skill");
        fs::write(
            manifest_dir.join("plugin.json"),
            r#"{
  "name": "example",
  "version": "0.1.0",
  "skills": ["./skills/reviewer"]
}"#,
        )
        .expect("write manifest");

        let mut config = HarnessConfig::default();
        config.resources.plugins.roots = vec![temp.path().to_path_buf()];
        let resources = ResourceCompiler::from_config(&config)
            .compile()
            .await
            .expect("compile resources");

        let skill = resources
            .snapshot
            .skills
            .get(&SkillName::from("reviewer"))
            .expect("reviewer skill");
        assert!(
            !skill.body.contains("${PLUGIN_ROOT}"),
            "body still contains template: {}",
            skill.body
        );
        assert!(
            !skill.body.contains("${CLAUDE_PLUGIN_DATA}"),
            "body still contains template: {}",
            skill.body
        );
        assert!(
            skill.body.contains("/plugin/.data"),
            "body should contain plugin data dir: {}",
            skill.body
        );
    }

    #[tokio::test]
    async fn plugin_skill_directory_tree_renders_plugin_vars() {
        let temp = tempfile::tempdir().expect("tempdir");
        let plugin_root = temp.path().join("plugin");
        let skill_dir = plugin_root.join("skills/reviewer");
        let manifest_dir = plugin_root.join(".claude-plugin");

        fs::create_dir_all(&skill_dir).expect("create skill dir");
        fs::create_dir_all(&manifest_dir).expect("create manifest dir");
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: reviewer\n---\n\nLoad ${PLUGIN_ROOT}/assets/foo.md.\n",
        )
        .expect("write skill");
        fs::write(
            manifest_dir.join("plugin.json"),
            r#"{
  "name": "example",
  "version": "0.1.0",
  "skills": ["./skills"]
}"#,
        )
        .expect("write manifest");

        let mut config = HarnessConfig::default();
        config.resources.plugins.roots = vec![temp.path().to_path_buf()];
        let resources = ResourceCompiler::from_config(&config)
            .compile()
            .await
            .expect("compile resources");

        let skill = resources
            .snapshot
            .skills
            .get(&SkillName::from("reviewer"))
            .expect("reviewer skill");
        assert!(
            !skill.body.contains("${PLUGIN_ROOT}"),
            "body still contains template: {}",
            skill.body
        );
        assert!(
            skill.body.contains("/plugin/assets/foo.md"),
            "body should resolve plugin root: {}",
            skill.body
        );
        assert!(
            !skill.body.contains("/plugin/skills/assets/foo.md"),
            "body should not resolve to skills subdirectory: {}",
            skill.body
        );
    }

    #[tokio::test]
    async fn plugin_agent_prompt_renders_plugin_vars() {
        let temp = tempfile::tempdir().expect("tempdir");
        let plugin_root = temp.path().join("plugin");
        let agents_dir = plugin_root.join("agents");
        let manifest_dir = plugin_root.join(".claude-plugin");

        fs::create_dir_all(&agents_dir).expect("create agents dir");
        fs::create_dir_all(&manifest_dir).expect("create manifest dir");
        fs::write(
            agents_dir.join("helper.md"),
            "Tools are at ${PLUGIN_ROOT}/tools.",
        )
        .expect("write agent prompt");
        fs::write(
            manifest_dir.join("plugin.json"),
            r#"{
  "name": "example",
  "version": "0.1.0",
  "agents": ["./agents"]
}"#,
        )
        .expect("write manifest");

        let mut config = HarnessConfig::default();
        config.resources.plugins.roots = vec![temp.path().to_path_buf()];
        let resources = ResourceCompiler::from_config(&config)
            .compile()
            .await
            .expect("compile resources");

        let agent = resources
            .snapshot
            .agents
            .get(&AgentName::from("helper"))
            .expect("helper agent");
        assert!(
            !agent.prompt.contains("${PLUGIN_ROOT}"),
            "prompt still contains template: {}",
            agent.prompt
        );
        assert!(
            agent.prompt.contains("/plugin/tools"),
            "prompt should contain plugin tools path: {}",
            agent.prompt
        );
    }

    #[tokio::test]
    async fn standalone_skill_body_renders_plugin_vars() {
        let temp = tempfile::tempdir().expect("tempdir");
        let skill_dir = temp.path().join("skills/hello");
        fs::create_dir_all(&skill_dir).expect("create skill dir");
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: hello\n---\n\nRun ${PLUGIN_ROOT}/scripts/x.sh and cache in ${PLUGIN_DATA}.\n",
        )
        .expect("write skill");

        let mut config = HarnessConfig::default();
        config.resources.skills.roots = vec![temp.path().join("skills")];
        let resources = ResourceCompiler::from_config(&config)
            .compile()
            .await
            .expect("compile resources");

        let skill = resources
            .snapshot
            .skills
            .get(&SkillName::from("hello"))
            .expect("hello skill");
        assert!(
            !skill.body.contains("${PLUGIN_ROOT}"),
            "body still contains template: {}",
            skill.body
        );
        assert!(
            skill.body.contains("/skills/hello/scripts/x.sh"),
            "body should contain skill scripts path: {}",
            skill.body
        );
        assert!(
            skill.body.contains("/skills/hello/.data"),
            "body should contain skill data dir: {}",
            skill.body
        );
    }

    #[tokio::test]
    async fn removed_halter_plugin_root_alias_is_rejected() {
        let temp = tempfile::tempdir().expect("tempdir");
        let plugin_root = temp.path().join("plugin");
        let manifest_dir = plugin_root.join(".agent-plugin");

        fs::create_dir_all(&manifest_dir).expect("create manifest dir");
        fs::write(
            manifest_dir.join("plugin.json"),
            r#"{
  "name": "legacy",
  "version": "0.1.0",
  "skills": ["${HALTER_PLUGIN_ROOT}/skills"]
}"#,
        )
        .expect("write manifest");

        let mut config = HarnessConfig::default();
        config.resources.plugins.roots = vec![temp.path().to_path_buf()];
        let error = ResourceCompiler::from_config(&config)
            .compile()
            .await
            .expect_err("compile should fail");
        assert!(
            error
                .to_string()
                .contains("relative component paths must start with './'"),
            "unexpected error: {error:?}"
        );
    }

    #[tokio::test]
    async fn removed_alias_in_body_left_literal() {
        let temp = tempfile::tempdir().expect("tempdir");
        let plugin_root = temp.path().join("plugin");
        let skill_dir = plugin_root.join("skills/legacy");
        let manifest_dir = plugin_root.join(".claude-plugin");

        fs::create_dir_all(&skill_dir).expect("create skill dir");
        fs::create_dir_all(&manifest_dir).expect("create manifest dir");
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: legacy\n---\n\nPath ${HALTER_PLUGIN_ROOT}/foo stays literal.\n",
        )
        .expect("write skill");
        fs::write(
            manifest_dir.join("plugin.json"),
            r#"{
  "name": "legacy",
  "version": "0.1.0",
  "skills": ["./skills/legacy"]
}"#,
        )
        .expect("write manifest");

        let mut config = HarnessConfig::default();
        config.resources.plugins.roots = vec![temp.path().to_path_buf()];
        let resources = ResourceCompiler::from_config(&config)
            .compile()
            .await
            .expect("compile resources");

        let skill = resources
            .snapshot
            .skills
            .get(&SkillName::from("legacy"))
            .expect("legacy skill");
        assert!(
            skill.body.contains("${HALTER_PLUGIN_ROOT}/foo"),
            "body should leave removed alias literal: {}",
            skill.body
        );
    }

    #[tokio::test]
    async fn rendered_alias_affects_snapshot_revision() {
        use std::collections::HashMap;

        fn copy_dir_all(src: &Path, dst: &Path) -> std::io::Result<()> {
            fs::create_dir_all(dst)?;
            for entry in fs::read_dir(src)? {
                let entry = entry?;
                let src_path = entry.path();
                let dst_path = dst.join(entry.file_name());
                if entry.file_type()?.is_dir() {
                    copy_dir_all(&src_path, &dst_path)?;
                } else {
                    fs::copy(&src_path, &dst_path)?;
                }
            }
            Ok(())
        }

        let temp = tempfile::tempdir().expect("tempdir");
        let plugin_root = temp.path().join("plugin");
        let skill_dir = plugin_root.join("skills/rev");
        let agents_dir = plugin_root.join("agents");
        let manifest_dir = plugin_root.join(".claude-plugin");

        fs::create_dir_all(&skill_dir).expect("create skill dir");
        fs::create_dir_all(&agents_dir).expect("create agents dir");
        fs::create_dir_all(&manifest_dir).expect("create manifest dir");
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: revskill\n---\n\nUse ${PLUGIN_ROOT}/a.md.\n",
        )
        .expect("write skill");
        fs::write(agents_dir.join("agent.md"), "Prompt ${PLUGIN_ROOT}/a.md.")
            .expect("write agent prompt");
        fs::write(
            manifest_dir.join("plugin.json"),
            r#"{
  "name": "revplugin",
  "version": "0.1.0",
  "skills": ["./skills/rev"],
  "agents": ["./agents"]
}"#,
        )
        .expect("write manifest");

        // Two copies of the same plugin at different absolute roots. The rendered
        // ${PLUGIN_ROOT} values differ, so the snapshot revisions must differ.
        let parent_a = temp.path().join("parent-a");
        let parent_b = temp.path().join("parent-b");
        copy_dir_all(&plugin_root, &parent_a.join("plugin")).expect("copy plugin a");
        copy_dir_all(&plugin_root, &parent_b.join("plugin")).expect("copy plugin b");
        let root_a = parent_a.join("plugin");

        let roots_a: Vec<PathBuf> = vec![parent_a.clone()];
        let roots_b: Vec<PathBuf> = vec![parent_b.clone()];

        let config = HarnessConfig::default();
        let first = ResourceCompiler::from_config(&config)
            .with_plugin_roots(roots_a)
            .compile()
            .await
            .expect("compile first");
        let second = ResourceCompiler::from_config(&config)
            .with_plugin_roots(roots_b)
            .compile()
            .await
            .expect("compile second");

        assert_ne!(
            first.snapshot.revision, second.snapshot.revision,
            "different root spellings should yield different snapshot revisions"
        );

        let by_name: HashMap<_, _> = first
            .snapshot
            .skills
            .iter()
            .map(|(name, skill)| (name.clone(), skill))
            .collect();
        let skill_a = by_name.get(&SkillName::from("revskill")).expect("revskill");
        let agent_a = first
            .snapshot
            .agents
            .get(&AgentName::from("agent"))
            .expect("agent");
        assert!(
            skill_a.body.contains(root_a.to_string_lossy().as_ref()),
            "skill body should contain rendered root path: {}",
            skill_a.body
        );
        assert!(
            agent_a.prompt.contains(root_a.to_string_lossy().as_ref()),
            "agent prompt should contain rendered root path: {}",
            agent_a.prompt
        );
    }
}
