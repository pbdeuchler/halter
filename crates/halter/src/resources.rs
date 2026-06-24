// pattern: Imperative Shell

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use halter_config::{HarnessConfig, LoadedPlugin, LoadedSkill, PluginLoader, SkillLoader};
use halter_hooks::{HookRegistrySource, Hooks};
use halter_protocol::{
    AgentDef, AgentName, HookWarning, HookWarningSeverity, InstructionFile, PromptRegistry,
    ResourceSnapshot, Revision, SkillDef, SkillName,
};
use sha2::{Digest, Sha256};
use tracing::{debug, info};

#[derive(Clone, Debug)]
/// Resource snapshot plus compiled hooks ready for runtime use.
pub struct CompiledResources {
    pub snapshot: ResourceSnapshot,
    pub hooks: Arc<Hooks>,
    pub hook_warnings: Vec<HookWarning>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::Path;

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
        let manifest_dir = plugin_root.join(".claude-plugin");

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
        let manifest_dir = plugin_root.join(".claude-plugin");

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
        let manifest_dir = plugin_root.join(".claude-plugin");

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
        let manifest_dir = plugin_root.join(".claude-plugin");

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
