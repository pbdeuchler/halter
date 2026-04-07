// pattern: Functional Core

use std::path::{Component, Path, PathBuf};

use async_trait::async_trait;

#[derive(Debug, Clone)]
pub struct PolicySettings {
    pub allowed_write_roots: Vec<PathBuf>,
    pub max_read_bytes: usize,
    pub max_tool_output_bytes: usize,
    pub shell_enabled: bool,
    pub allowed_shell_commands: Vec<String>,
    pub shell_timeout_secs: u64,
    pub network_enabled: bool,
    pub allowed_hosts: Vec<String>,
    pub max_subagent_depth: u32,
    pub max_concurrent_subagents: usize,
}

impl Default for PolicySettings {
    fn default() -> Self {
        Self {
            allowed_write_roots: vec![PathBuf::from("."), PathBuf::from("/tmp/halter")],
            max_read_bytes: 1_048_576,
            max_tool_output_bytes: 262_144,
            shell_enabled: true,
            allowed_shell_commands: vec![
                "git".to_owned(),
                "cargo".to_owned(),
                "rg".to_owned(),
                "ls".to_owned(),
                "find".to_owned(),
            ],
            shell_timeout_secs: 30,
            network_enabled: false,
            allowed_hosts: Vec::new(),
            max_subagent_depth: 3,
            max_concurrent_subagents: 8,
        }
    }
}

#[async_trait]
pub trait ToolPolicy: Send + Sync {
    async fn check_read(&self, path: &Path, bytes: usize) -> anyhow::Result<()>;
    async fn check_write(&self, path: &Path) -> anyhow::Result<()>;
    async fn check_shell(&self, program: &str) -> anyhow::Result<()>;
    async fn check_subagent_spawn(
        &self,
        parent_depth: u32,
        active_subagents: usize,
    ) -> anyhow::Result<()>;
}

#[derive(Debug, Clone)]
pub struct DefaultToolPolicy {
    settings: PolicySettings,
}

impl DefaultToolPolicy {
    #[must_use]
    pub fn new(settings: PolicySettings) -> Self {
        Self { settings }
    }

    #[must_use]
    pub fn settings(&self) -> &PolicySettings {
        &self.settings
    }
}

#[async_trait]
impl ToolPolicy for DefaultToolPolicy {
    async fn check_read(&self, path: &Path, bytes: usize) -> anyhow::Result<()> {
        let _ = normalize_existing_or_absolute(path)?;
        if bytes > self.settings.max_read_bytes {
            anyhow::bail!(
                "failed to execute read tool: '{}' requested {} bytes exceeds max_read_bytes {}",
                path.display(),
                bytes,
                self.settings.max_read_bytes
            );
        }
        Ok(())
    }

    async fn check_write(&self, path: &Path) -> anyhow::Result<()> {
        let candidate = canonicalize_write_target(path)?;
        let allowed = self
            .settings
            .allowed_write_roots
            .iter()
            .filter_map(|root| canonicalize_allowed_root(root).ok())
            .any(|root| candidate.starts_with(&root));

        if !allowed {
            anyhow::bail!(
                "failed to execute write tool: path '{}' is outside allowed_write_roots",
                path.display()
            );
        }
        Ok(())
    }

    async fn check_shell(&self, program: &str) -> anyhow::Result<()> {
        if !self.settings.shell_enabled {
            anyhow::bail!("failed to execute shell tool: shell usage is disabled by policy");
        }

        if !self
            .settings
            .allowed_shell_commands
            .iter()
            .any(|allowed| allowed == program)
        {
            anyhow::bail!(
                "failed to execute shell tool: program '{}' is not in the allowlist",
                program
            );
        }
        Ok(())
    }

    async fn check_subagent_spawn(
        &self,
        parent_depth: u32,
        active_subagents: usize,
    ) -> anyhow::Result<()> {
        if parent_depth >= self.settings.max_subagent_depth {
            anyhow::bail!(
                "failed to execute spawn_agent tool: subagent depth {} exceeds max_subagent_depth {}",
                parent_depth + 1,
                self.settings.max_subagent_depth
            );
        }

        if active_subagents >= self.settings.max_concurrent_subagents {
            anyhow::bail!(
                "failed to execute spawn_agent tool: active subagents {} exceed max_concurrent_subagents {}",
                active_subagents,
                self.settings.max_concurrent_subagents
            );
        }

        Ok(())
    }
}

fn canonicalize_allowed_root(root: &Path) -> anyhow::Result<PathBuf> {
    normalize_existing_or_absolute(root)
}

fn canonicalize_write_target(path: &Path) -> anyhow::Result<PathBuf> {
    let absolute = absolute_path(path)?;
    reject_parent_traversal(&absolute)?;

    let mut existing_ancestor = absolute.as_path();
    while !existing_ancestor.exists() {
        existing_ancestor = existing_ancestor.parent().ok_or_else(|| {
            anyhow::anyhow!(
                "failed to execute write tool: path '{}' has no valid parent",
                path.display()
            )
        })?;
    }

    let resolved_ancestor = normalize_existing_or_absolute(existing_ancestor)?;
    let suffix = absolute
        .strip_prefix(existing_ancestor)
        .expect("existing ancestor must prefix absolute path");

    Ok(resolved_ancestor.join(suffix))
}

fn normalize_existing_or_absolute(path: &Path) -> anyhow::Result<PathBuf> {
    let absolute = absolute_path(path)?;
    if absolute.exists() {
        return std::fs::canonicalize(&absolute).map_err(Into::into);
    }

    reject_parent_traversal(&absolute)?;
    Ok(absolute)
}

fn absolute_path(path: &Path) -> anyhow::Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn reject_parent_traversal(path: &Path) -> anyhow::Result<()> {
    if path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        anyhow::bail!(
            "failed to execute write tool: path '{}' traverses outside its root",
            path.display()
        );
    }
    Ok(())
}
