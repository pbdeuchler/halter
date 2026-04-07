// pattern: Imperative Shell

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};
use std::{env, fs};

use anyhow::Context;
use halter_config::HarnessConfig;
use halter_protocol::{
    AgentDef, AgentId, AgentName, ContentHash, InstructionFile, PluginId, PluginManifest,
    PromptRegistry, ResourceSnapshot, Revision, SkillDef, SkillId, SkillName,
};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tracing::{debug, info};

#[derive(Debug, Clone)]
pub struct LoadedResourceFile {
    pub path: PathBuf,
    pub body: String,
    pub revision: ContentHash,
}

#[derive(Debug, Clone)]
pub struct LoadedExecutable {
    pub path: PathBuf,
}

#[derive(Debug, Clone)]
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
pub struct LoadedAgent {
    pub id: AgentId,
    pub name: String,
    pub prompt: String,
}

#[derive(Debug, Clone)]
pub struct LoadedHook {
    pub path: PathBuf,
}

#[derive(Debug, Clone)]
pub struct LoadedMcpServer {
    pub path: PathBuf,
    pub body: Value,
}

#[derive(Debug, Clone)]
pub struct LoadedLspServer {
    pub path: PathBuf,
    pub body: Value,
}

#[derive(Debug, Clone)]
pub struct LoadedOutputStyle {
    pub path: PathBuf,
}

#[derive(Debug, Clone, Default)]
pub struct PluginDefaults {
    pub settings: BTreeMap<String, Value>,
}

#[derive(Debug, Clone)]
pub struct LoadedPlugin {
    pub id: PluginId,
    pub root: PathBuf,
    pub manifest: PluginManifest,
    pub skills: Vec<LoadedSkill>,
    pub agents: Vec<LoadedAgent>,
    pub hooks: Vec<LoadedHook>,
    pub mcp_servers: Vec<LoadedMcpServer>,
    pub lsp_servers: Vec<LoadedLspServer>,
    pub output_styles: Vec<LoadedOutputStyle>,
    pub bin_paths: Vec<PathBuf>,
    pub defaults: PluginDefaults,
}

#[derive(Debug, Default, Clone)]
pub struct SkillLoader;

impl SkillLoader {
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
        info!(skill_count = skills.len(), "loaded skills");
        Ok(skills)
    }
}

#[derive(Debug, Default, Clone)]
pub struct PluginLoader;

impl PluginLoader {
    pub fn load_roots(&self, roots: &[PathBuf]) -> anyhow::Result<Vec<LoadedPlugin>> {
        debug!(root_count = roots.len(), "loading plugin roots");
        let mut plugins = Vec::new();
        for root in normalized_roots(roots)? {
            if !root.exists() {
                debug!(root = %root.display(), "skipping missing plugin root");
                continue;
            }
            for entry in fs::read_dir(&root)? {
                let entry = entry?;
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
pub struct ResourceCompiler {
    config: HarnessConfig,
    skill_roots: Option<Vec<PathBuf>>,
    plugin_roots: Option<Vec<PathBuf>>,
    loaded_skills: Vec<LoadedSkill>,
    loaded_plugins: Vec<LoadedPlugin>,
}

impl ResourceCompiler {
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

    #[must_use]
    pub fn with_skill_roots(mut self, roots: Vec<PathBuf>) -> Self {
        self.skill_roots = Some(roots);
        self
    }

    #[must_use]
    pub fn with_plugin_roots(mut self, roots: Vec<PathBuf>) -> Self {
        self.plugin_roots = Some(roots);
        self
    }

    #[must_use]
    pub fn with_loaded_skills(mut self, skills: Vec<LoadedSkill>) -> Self {
        self.loaded_skills = skills;
        self
    }

    #[must_use]
    pub fn with_loaded_plugins(mut self, plugins: Vec<LoadedPlugin>) -> Self {
        self.loaded_plugins = plugins;
        self
    }

    pub async fn compile(self) -> anyhow::Result<ResourceSnapshot> {
        tokio::task::spawn_blocking(move || compile_resources(self))
            .await
            .context("failed to join resource compiler task")?
    }
}

fn compile_resources(compiler: ResourceCompiler) -> anyhow::Result<ResourceSnapshot> {
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

    for plugin in plugins {
        revision_hasher.update(plugin.manifest.name.as_bytes());
        revision_hasher.update(plugin.manifest.version.as_bytes());
        for agent in &plugin.agents {
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
    Ok(snapshot)
}

fn collect_skills(
    root: &Path,
    visited: &mut BTreeSet<PathBuf>,
    sink: &mut Vec<LoadedSkill>,
) -> anyhow::Result<()> {
    let canonical_root = fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
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
        id: SkillId::new(),
        name,
        description,
        root: root.to_path_buf(),
        revision: hash_bytes(body.as_bytes()),
        body,
        supporting_files: Vec::new(),
        scripts: load_scripts(root)?,
    })
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
            .unwrap_or("unnamed-plugin")
            .to_owned(),
        version: manifest_value
            .get("version")
            .and_then(Value::as_str)
            .unwrap_or("0.0.0")
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
    };

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

    let mut agents = Vec::new();
    for agent_path in &manifest.agents {
        let resolved = resolve_plugin_path(root, agent_path)?;
        if resolved.is_file() {
            agents.push(load_agent_file(&resolved)?);
        } else if resolved.is_dir() {
            load_agent_dir(&resolved, &mut agents)?;
        }
    }

    Ok(Some(LoadedPlugin {
        id: PluginId::new(),
        root: root.to_path_buf(),
        manifest,
        skills,
        agents,
        hooks: Vec::new(),
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
        || component.contains("${HALTER_PLUGIN_ROOT}")
        || component.contains("${CLAUDE_PLUGIN_DATA}")
        || component.contains("${PLUGIN_DATA}")
        || component.contains("${HALTER_PLUGIN_DATA}");

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

fn normalized_roots(roots: &[PathBuf]) -> anyhow::Result<Vec<PathBuf>> {
    let mut seen = BTreeSet::new();
    let mut normalized = Vec::new();
    for root in roots {
        let expanded = expand_path(root);
        let key = expanded.to_string_lossy().to_string();
        if seen.insert(key) {
            normalized.push(expanded);
        }
    }
    Ok(normalized)
}

fn expand_path(path: &Path) -> PathBuf {
    let raw = path.to_string_lossy();
    if let Some(stripped) = raw.strip_prefix("~/") {
        if let Ok(home) = env::var("HOME") {
            return PathBuf::from(home).join(stripped);
        }
    }
    PathBuf::from(raw.as_ref())
}

fn expand_plugin_component_path(root: &Path, component: &str) -> String {
    let plugin_root = root.to_string_lossy().to_string();
    let plugin_data = root.join(".data").to_string_lossy().to_string();
    component
        .replace("${CLAUDE_PLUGIN_ROOT}", &plugin_root)
        .replace("${PLUGIN_ROOT}", &plugin_root)
        .replace("${HALTER_PLUGIN_ROOT}", &plugin_root)
        .replace("${CLAUDE_PLUGIN_DATA}", &plugin_data)
        .replace("${PLUGIN_DATA}", &plugin_data)
        .replace("${HALTER_PLUGIN_DATA}", &plugin_data)
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
    Ok(LoadedAgent {
        id: AgentId::new(),
        name: path
            .file_stem()
            .unwrap_or_default()
            .to_string_lossy()
            .to_string(),
        prompt,
    })
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
        let snapshot = ResourceCompiler::from_config(&config)
            .compile()
            .await
            .expect("compile resources");

        assert!(snapshot.skills.contains_key(&SkillName::from("hello")));
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
        let snapshot = ResourceCompiler::from_config(&config)
            .compile()
            .await
            .expect("compile resources");

        assert!(snapshot.skills.contains_key(&SkillName::from("reviewer")));
        assert!(snapshot.agents.contains_key(&AgentName::from("helper")));
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
}
