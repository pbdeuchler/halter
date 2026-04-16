// pattern: Functional Core

use std::io::Cursor;
use std::path::{Component, Path, PathBuf};

use async_trait::async_trait;
use brush_parser::{Parser, ParserOptions, SourceInfo, ast};

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

impl PolicySettings {
    /// Permissive settings used only in tests. Production code must construct
    /// `PolicySettings` from a user-supplied `PolicyConfig` via the builder so
    /// policy behavior is never silently inherited from a hard-coded default.
    #[must_use]
    pub fn permissive() -> Self {
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
    async fn check_shell_command(&self, command: &str) -> anyhow::Result<()>;
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

    fn check_shell_enabled(&self) -> anyhow::Result<()> {
        if !self.settings.shell_enabled {
            anyhow::bail!("failed to execute shell tool: shell usage is disabled by policy");
        }
        Ok(())
    }

    fn ensure_allowed_shell_program(&self, program: &str) -> anyhow::Result<()> {
        if program == "shell" {
            return Ok(());
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
        self.check_shell_enabled()?;
        self.ensure_allowed_shell_program(program)
    }

    async fn check_shell_command(&self, command: &str) -> anyhow::Result<()> {
        self.check_shell_enabled()?;
        for program in shell_command_programs(command)? {
            self.ensure_allowed_shell_program(&program)?;
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

fn shell_command_programs(command: &str) -> anyhow::Result<Vec<String>> {
    if command.trim().is_empty() {
        return Ok(Vec::new());
    }

    // Walk the parsed shell AST rather than tokenizing text so pipelines,
    // subshells, process substitutions, and nested command lists are checked too.
    let mut parser = Parser::new(
        Cursor::new(command),
        &ParserOptions::default(),
        &SourceInfo::default(),
    );
    let program = parser.parse_program().map_err(|error| {
        anyhow::anyhow!("failed to execute shell tool: invalid shell command: {error}")
    })?;
    let mut programs = Vec::new();
    collect_program_commands(&program, &mut programs);
    Ok(programs)
}

fn collect_program_commands(program: &ast::Program, programs: &mut Vec<String>) {
    for command in &program.complete_commands {
        collect_compound_list_commands(command, programs);
    }
}

fn collect_compound_list_commands(list: &ast::CompoundList, programs: &mut Vec<String>) {
    for item in &list.0 {
        collect_and_or_list_commands(&item.0, programs);
    }
}

fn collect_and_or_list_commands(list: &ast::AndOrList, programs: &mut Vec<String>) {
    collect_pipeline_commands(&list.first, programs);
    for item in &list.additional {
        match item {
            ast::AndOr::And(pipeline) | ast::AndOr::Or(pipeline) => {
                collect_pipeline_commands(pipeline, programs);
            }
        }
    }
}

fn collect_pipeline_commands(pipeline: &ast::Pipeline, programs: &mut Vec<String>) {
    for command in &pipeline.seq {
        collect_command_programs(command, programs);
    }
}

fn collect_command_programs(command: &ast::Command, programs: &mut Vec<String>) {
    match command {
        ast::Command::Simple(simple) => collect_simple_command_programs(simple, programs),
        ast::Command::Compound(compound, redirects) => {
            collect_compound_command_programs(compound, programs);
            if let Some(redirects) = redirects {
                collect_redirect_list_programs(redirects, programs);
            }
        }
        ast::Command::Function(function) => collect_function_programs(function, programs),
        ast::Command::ExtendedTest(_) => {}
    }
}

fn collect_simple_command_programs(command: &ast::SimpleCommand, programs: &mut Vec<String>) {
    if let Some(word_or_name) = &command.word_or_name {
        let program = word_or_name.value.trim();
        if !program.is_empty() {
            programs.push(program.to_owned());
        }
    }
    if let Some(prefix) = &command.prefix {
        collect_prefix_or_suffix_items(&prefix.0, programs);
    }
    if let Some(suffix) = &command.suffix {
        collect_prefix_or_suffix_items(&suffix.0, programs);
    }
}

fn collect_prefix_or_suffix_items(
    items: &[ast::CommandPrefixOrSuffixItem],
    programs: &mut Vec<String>,
) {
    for item in items {
        match item {
            ast::CommandPrefixOrSuffixItem::IoRedirect(redirect) => {
                collect_io_redirect_programs(redirect, programs);
            }
            ast::CommandPrefixOrSuffixItem::ProcessSubstitution(_, subshell) => {
                collect_subshell_programs(subshell, programs);
            }
            ast::CommandPrefixOrSuffixItem::Word(_)
            | ast::CommandPrefixOrSuffixItem::AssignmentWord(_, _) => {}
        }
    }
}

fn collect_compound_command_programs(command: &ast::CompoundCommand, programs: &mut Vec<String>) {
    match command {
        ast::CompoundCommand::Arithmetic(_) => {}
        ast::CompoundCommand::ArithmeticForClause(command) => {
            collect_do_group_programs(&command.body, programs);
        }
        ast::CompoundCommand::BraceGroup(command) => {
            collect_compound_list_commands(&command.list, programs);
        }
        ast::CompoundCommand::Subshell(command) => collect_subshell_programs(command, programs),
        ast::CompoundCommand::ForClause(command) => {
            collect_do_group_programs(&command.body, programs)
        }
        ast::CompoundCommand::CaseClause(command) => {
            for case in &command.cases {
                if let Some(list) = &case.cmd {
                    collect_compound_list_commands(list, programs);
                }
            }
        }
        ast::CompoundCommand::IfClause(command) => {
            collect_compound_list_commands(&command.condition, programs);
            collect_compound_list_commands(&command.then, programs);
            if let Some(elses) = &command.elses {
                for else_clause in elses {
                    if let Some(condition) = &else_clause.condition {
                        collect_compound_list_commands(condition, programs);
                    }
                    collect_compound_list_commands(&else_clause.body, programs);
                }
            }
        }
        ast::CompoundCommand::WhileClause(command) | ast::CompoundCommand::UntilClause(command) => {
            collect_compound_list_commands(&command.0, programs);
            collect_do_group_programs(&command.1, programs);
        }
    }
}

fn collect_function_programs(function: &ast::FunctionDefinition, programs: &mut Vec<String>) {
    collect_compound_command_programs(&function.body.0, programs);
    if let Some(redirects) = &function.body.1 {
        collect_redirect_list_programs(redirects, programs);
    }
}

fn collect_do_group_programs(command: &ast::DoGroupCommand, programs: &mut Vec<String>) {
    collect_compound_list_commands(&command.list, programs);
}

fn collect_subshell_programs(command: &ast::SubshellCommand, programs: &mut Vec<String>) {
    collect_compound_list_commands(&command.list, programs);
}

fn collect_redirect_list_programs(list: &ast::RedirectList, programs: &mut Vec<String>) {
    for redirect in &list.0 {
        collect_io_redirect_programs(redirect, programs);
    }
}

fn collect_io_redirect_programs(redirect: &ast::IoRedirect, programs: &mut Vec<String>) {
    match redirect {
        ast::IoRedirect::File(_, _, target) => collect_io_target_programs(target, programs),
        ast::IoRedirect::HereDocument(_, _)
        | ast::IoRedirect::HereString(_, _)
        | ast::IoRedirect::OutputAndError(_, _) => {}
    }
}

fn collect_io_target_programs(target: &ast::IoFileRedirectTarget, programs: &mut Vec<String>) {
    match target {
        ast::IoFileRedirectTarget::ProcessSubstitution(_, subshell) => {
            collect_subshell_programs(subshell, programs);
        }
        ast::IoFileRedirectTarget::Filename(_)
        | ast::IoFileRedirectTarget::Fd(_)
        | ast::IoFileRedirectTarget::Duplicate(_) => {}
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn check_shell_command_rejects_disallowed_program() {
        let policy = DefaultToolPolicy::new(PolicySettings {
            allowed_shell_commands: vec!["git".to_owned()],
            ..PolicySettings::permissive()
        });

        let error = policy
            .check_shell_command("git status && rg TODO .")
            .await
            .expect_err("disallowed program should fail");

        assert!(
            error
                .to_string()
                .contains("program 'rg' is not in the allowlist")
        );
    }

    #[tokio::test]
    async fn check_shell_command_rejects_mixed_pipeline() {
        let policy = DefaultToolPolicy::new(PolicySettings {
            allowed_shell_commands: vec!["rg".to_owned()],
            ..PolicySettings::permissive()
        });

        let error = policy
            .check_shell_command("rg TODO . | sort")
            .await
            .expect_err("mixed pipeline should fail");

        assert!(
            error
                .to_string()
                .contains("program 'sort' is not in the allowlist")
        );
    }

    #[tokio::test]
    async fn check_shell_command_allows_allowlisted_commands_and_pipelines() {
        let policy = DefaultToolPolicy::new(PolicySettings {
            allowed_shell_commands: vec!["git".to_owned(), "rg".to_owned()],
            ..PolicySettings::permissive()
        });

        policy
            .check_shell_command("git status | rg modified")
            .await
            .expect("pipeline should be allowed");
    }
}
