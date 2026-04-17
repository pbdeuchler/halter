// pattern: Functional Core
//
// Capability-oriented tool policy.
//
// Phase 1 of the review remediation (see
// `docs/design-plans/2026-04-17-review-remediation-core.md`) introduces a new
// capability-oriented method set on `ToolPolicy` — `check_read_path`,
// `check_write_path`, `check_process_signal`, `check_shell_enabled`,
// `check_shell_command_strict`, `check_network`, `check_subagent_spawn_typed`
// — and the supporting `CanonicalPath` and `PolicyError` types.
//
// The pre-existing name-based methods (`check_read`, `check_write`,
// `check_shell`, `check_shell_command`, `check_subagent_spawn`) are retained
// as `#[deprecated]` compatibility shims so that the builtin call sites in
// `builtin/*.rs` and `halter-runtime/src/subagents.rs` keep compiling. Phase 2
// of the remediation migrates those call sites to the new surface and removes
// the deprecated methods.

use std::io::Cursor;
use std::net::IpAddr;
use std::path::{Component, Path, PathBuf};

use async_trait::async_trait;
use brush_parser::{Parser, ParserOptions, SourceInfo, ast};
use globset::{Glob, GlobSet, GlobSetBuilder};

mod canonical;
mod errors;

#[cfg(test)]
mod security_tests;

pub use canonical::CanonicalPath;
pub use errors::{Pid, PolicyError};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum ShellMode {
    /// Reject function definitions and `eval`/`exec`/`source`/`.` commands.
    /// Default. The brush-parser AST walk is the enforcement point; this is a
    /// defensive rejection layer, not a complete isolation boundary.
    #[default]
    Strict,
    /// Retain the prior program-token allowlist behavior. Documented as *not*
    /// a security boundary; available for workflows that need function
    /// definitions or `eval`.
    Relaxed,
}

/// A permitted sidecar at a loopback address (for e.g. a local
/// development proxy or a container-bound service). Presence in
/// `PolicySettings::allowed_loopback_services` is the only way loopback IPs
/// reach the network after Phase 1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopbackAllow {
    pub host: String,
    pub port: Option<u16>,
}

#[derive(Debug, Clone)]
pub struct PolicySettings {
    pub allowed_write_roots: Vec<PathBuf>,
    pub allowed_read_roots: Vec<PathBuf>,
    pub sensitive_path_patterns: Vec<String>,
    pub max_read_bytes: usize,
    pub max_tool_output_bytes: usize,
    pub shell_enabled: bool,
    pub shell_mode: ShellMode,
    pub allowed_shell_commands: Vec<String>,
    pub shell_timeout_secs: u64,
    pub network_enabled: bool,
    pub allowed_hosts: Vec<String>,
    pub allowed_loopback_services: Vec<LoopbackAllow>,
    pub process_tree_root: Option<Pid>,
    pub max_subagent_depth: u32,
    pub max_concurrent_subagents: usize,
}

impl Default for PolicySettings {
    fn default() -> Self {
        Self {
            allowed_write_roots: vec![PathBuf::from("."), PathBuf::from("/tmp/halter")],
            allowed_read_roots: default_read_roots(),
            sensitive_path_patterns: default_sensitive_patterns(),
            max_read_bytes: 1_048_576,
            max_tool_output_bytes: 262_144,
            shell_enabled: true,
            shell_mode: ShellMode::Strict,
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
            allowed_loopback_services: Vec::new(),
            process_tree_root: None,
            max_subagent_depth: 3,
            max_concurrent_subagents: 8,
        }
    }
}

fn default_read_roots() -> Vec<PathBuf> {
    let mut roots = vec![PathBuf::from(".")];
    if let Some(tmp) = std::env::var_os("TMPDIR") {
        roots.push(PathBuf::from(tmp));
    } else {
        roots.push(PathBuf::from("/tmp"));
    }
    roots
}

fn default_sensitive_patterns() -> Vec<String> {
    vec![
        "**/.ssh/**".to_owned(),
        "**/.aws/**".to_owned(),
        "**/.secrets".to_owned(),
        "/etc/shadow".to_owned(),
        "/etc/shadow.*".to_owned(),
    ]
}

#[async_trait]
pub trait ToolPolicy: Send + Sync {
    // ---------- deprecated compatibility shims (to be removed in Phase 2)

    #[deprecated(note = "use check_read_path; removed in Phase 2")]
    async fn check_read(&self, path: &Path, bytes: usize) -> anyhow::Result<()>;

    #[deprecated(note = "use check_write_path; removed in Phase 2")]
    async fn check_write(&self, path: &Path) -> anyhow::Result<()>;

    #[deprecated(note = "use check_shell_enabled; removed in Phase 2")]
    async fn check_shell(&self, program: &str) -> anyhow::Result<()>;

    #[deprecated(note = "use check_shell_command_strict; removed in Phase 2")]
    async fn check_shell_command(&self, command: &str) -> anyhow::Result<()>;

    #[deprecated(note = "use check_subagent_spawn_typed; removed in Phase 2")]
    async fn check_subagent_spawn(
        &self,
        parent_depth: u32,
        active_subagents: usize,
    ) -> anyhow::Result<()>;

    // ---------- new capability-oriented surface

    /// Resolve and authorize `path` for a read that will read at most `bytes`
    /// bytes. Returns a `CanonicalPath` pinning the parent directory fd so
    /// that the actual read via `openat` cannot be redirected by a concurrent
    /// symlink swap.
    async fn check_read_path(
        &self,
        path: &Path,
        bytes: usize,
    ) -> Result<CanonicalPath, PolicyError>;

    /// Resolve and authorize `path` for a write. Only the parent directory
    /// needs to exist; the leaf may be created. Returned `CanonicalPath`
    /// holds the parent fd.
    async fn check_write_path(&self, path: &Path) -> Result<CanonicalPath, PolicyError>;

    /// Authorize sending a process signal to `pid`. Enforces the session's
    /// process-tree boundary when configured.
    async fn check_process_signal(&self, pid: Pid) -> Result<(), PolicyError>;

    /// Authorize the shell capability generically (e.g. spawning a PTY).
    async fn check_shell_enabled(&self) -> Result<(), PolicyError>;

    /// Authorize `command` under a specific `ShellMode`. Strict mode rejects
    /// function definitions and `eval`/`exec`/`source`/`.` invocations via
    /// the brush-parser AST walk.
    async fn check_shell_command_strict(
        &self,
        command: &str,
        mode: ShellMode,
    ) -> Result<(), PolicyError>;

    /// Authorize a network request to `url`. Loopback addresses are denied
    /// unless an entry in `allowed_loopback_services` matches.
    async fn check_network(&self, url: &str) -> Result<(), PolicyError>;

    /// Capability-typed subagent spawn check. Parallel to the deprecated
    /// `check_subagent_spawn` but returns `PolicyError`.
    async fn check_subagent_spawn_typed(
        &self,
        parent_depth: u32,
        active: usize,
    ) -> Result<(), PolicyError>;
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

    fn check_shell_enabled_inner(&self) -> Result<(), PolicyError> {
        if self.settings.shell_enabled {
            Ok(())
        } else {
            Err(PolicyError::ShellDisabled)
        }
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

    fn sensitive_glob_set(&self) -> Result<GlobSet, PolicyError> {
        let mut builder = GlobSetBuilder::new();
        for pattern in &self.settings.sensitive_path_patterns {
            let glob = Glob::new(pattern).map_err(|_| PolicyError::SensitivePathDenied {
                attempted: PathBuf::from(pattern),
                rule: "invalid_pattern",
            })?;
            builder.add(glob);
        }
        builder
            .build()
            .map_err(|_| PolicyError::SensitivePathDenied {
                attempted: PathBuf::new(),
                rule: "invalid_pattern_set",
            })
    }

    fn canonical_read_roots(&self) -> Result<Vec<PathBuf>, PolicyError> {
        canonicalize_root_list(&self.settings.allowed_read_roots)
    }

    fn canonical_write_roots(&self) -> Result<Vec<PathBuf>, PolicyError> {
        canonicalize_root_list(&self.settings.allowed_write_roots)
    }

    fn verify_under_any_root(candidate: &Path, roots: &[PathBuf]) -> Result<(), PolicyError> {
        if roots.iter().any(|root| candidate.starts_with(root)) {
            Ok(())
        } else {
            Err(PolicyError::NotInRoot {
                attempted: candidate.to_path_buf(),
                roots: roots.to_vec(),
            })
        }
    }

    fn verify_not_sensitive(&self, candidate: &Path) -> Result<(), PolicyError> {
        let set = self.sensitive_glob_set()?;
        if set.is_match(candidate) {
            return Err(PolicyError::SensitivePathDenied {
                attempted: candidate.to_path_buf(),
                rule: "sensitive_path",
            });
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
            .filter_map(|root| canonicalize_allowed_root_legacy(root).ok())
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
        self.ensure_allowed_shell_program(program)
    }

    async fn check_shell_command(&self, command: &str) -> anyhow::Result<()> {
        if !self.settings.shell_enabled {
            anyhow::bail!("failed to execute shell tool: shell usage is disabled by policy");
        }
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

    async fn check_read_path(
        &self,
        path: &Path,
        bytes: usize,
    ) -> Result<CanonicalPath, PolicyError> {
        if bytes > self.settings.max_read_bytes {
            return Err(PolicyError::ReadTooLarge {
                bytes,
                max: self.settings.max_read_bytes,
            });
        }
        let absolute = absolute_path_typed(path)?;
        reject_parent_traversal_typed(&absolute)?;
        self.verify_not_sensitive(&absolute)?;
        let canonical = CanonicalPath::for_existing(&absolute)?;
        self.verify_not_sensitive(canonical.path())?;
        let roots = self.canonical_read_roots()?;
        Self::verify_under_any_root(canonical.path(), &roots)?;
        Ok(canonical)
    }

    async fn check_write_path(&self, path: &Path) -> Result<CanonicalPath, PolicyError> {
        let absolute = absolute_path_typed(path)?;
        reject_parent_traversal_typed(&absolute)?;
        self.verify_not_sensitive(&absolute)?;
        let canonical = resolve_write_target_typed(&absolute)?;
        self.verify_not_sensitive(canonical.path())?;
        let roots = self.canonical_write_roots()?;
        Self::verify_under_any_root(canonical.path(), &roots)?;
        Ok(canonical)
    }

    async fn check_process_signal(&self, pid: Pid) -> Result<(), PolicyError> {
        if pid <= 1 {
            return Err(PolicyError::ProcessOutsideTree { pid });
        }
        // Phase 2 threads `process_tree_root` into the live session and walks
        // the descendant tree. For Phase 1 we enforce only the init-PID floor
        // so that AC1.6 holds; AC1.7 lands with Phase 2.
        let _ = self.settings.process_tree_root;
        Ok(())
    }

    async fn check_shell_enabled(&self) -> Result<(), PolicyError> {
        self.check_shell_enabled_inner()
    }

    async fn check_shell_command_strict(
        &self,
        command: &str,
        mode: ShellMode,
    ) -> Result<(), PolicyError> {
        self.check_shell_enabled_inner()?;
        if command.trim().is_empty() {
            return Ok(());
        }
        let program = parse_shell_program(command)?;
        if matches!(mode, ShellMode::Strict) {
            reject_strict_mode_constructs(&program)?;
        }
        Ok(())
    }

    async fn check_network(&self, url: &str) -> Result<(), PolicyError> {
        let host = extract_host(url).ok_or_else(|| PolicyError::NetworkDenied {
            url: url.to_owned(),
            rule: "unparseable_url",
        })?;
        if let Ok(addr) = host.parse::<IpAddr>()
            && addr.is_loopback()
        {
            return self.allow_loopback(url, &host, None);
        }
        if host.eq_ignore_ascii_case("localhost") {
            return self.allow_loopback(url, &host, None);
        }
        if !self.settings.network_enabled {
            return Err(PolicyError::NetworkDenied {
                url: url.to_owned(),
                rule: "network_disabled",
            });
        }
        if !self.settings.allowed_hosts.is_empty()
            && !self
                .settings
                .allowed_hosts
                .iter()
                .any(|h| h.eq_ignore_ascii_case(&host))
        {
            return Err(PolicyError::NetworkDenied {
                url: url.to_owned(),
                rule: "host_not_allowed",
            });
        }
        Ok(())
    }

    async fn check_subagent_spawn_typed(
        &self,
        parent_depth: u32,
        active: usize,
    ) -> Result<(), PolicyError> {
        if parent_depth >= self.settings.max_subagent_depth {
            return Err(PolicyError::SubagentLimit {
                kind: "depth",
                current: parent_depth + 1,
                limit: self.settings.max_subagent_depth,
            });
        }
        if active >= self.settings.max_concurrent_subagents {
            return Err(PolicyError::SubagentLimit {
                kind: "concurrent",
                current: u32::try_from(active).unwrap_or(u32::MAX),
                limit: u32::try_from(self.settings.max_concurrent_subagents).unwrap_or(u32::MAX),
            });
        }
        Ok(())
    }
}

impl DefaultToolPolicy {
    fn allow_loopback(&self, url: &str, host: &str, port: Option<u16>) -> Result<(), PolicyError> {
        let matches = self.settings.allowed_loopback_services.iter().any(|entry| {
            entry.host.eq_ignore_ascii_case(host) && (entry.port.is_none() || entry.port == port)
        });
        if matches {
            Ok(())
        } else {
            Err(PolicyError::NetworkDenied {
                url: url.to_owned(),
                rule: "loopback_not_allowlisted",
            })
        }
    }
}

fn canonicalize_allowed_root_legacy(root: &Path) -> anyhow::Result<PathBuf> {
    normalize_existing_or_absolute(root)
}

/// Canonicalize a list of configured roots. A nonexistent root produces a
/// typed `NonexistentRoot` error instead of being silently dropped (AC1.14).
fn canonicalize_root_list(roots: &[PathBuf]) -> Result<Vec<PathBuf>, PolicyError> {
    let mut out = Vec::with_capacity(roots.len());
    for root in roots {
        let absolute = absolute_path_typed(root)?;
        let canonical =
            std::fs::canonicalize(&absolute).map_err(|_| PolicyError::NonexistentRoot {
                root: absolute.clone(),
            })?;
        out.push(canonical);
    }
    Ok(out)
}

fn absolute_path_typed(path: &Path) -> Result<PathBuf, PolicyError> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        let cwd = std::env::current_dir().map_err(|e| PolicyError::io(path.to_path_buf(), e))?;
        Ok(cwd.join(path))
    }
}

fn reject_parent_traversal_typed(path: &Path) -> Result<(), PolicyError> {
    if path
        .components()
        .any(|component| matches!(component, Component::ParentDir))
    {
        return Err(PolicyError::ParentTraversal {
            attempted: path.to_path_buf(),
        });
    }
    Ok(())
}

fn resolve_write_target_typed(absolute: &Path) -> Result<CanonicalPath, PolicyError> {
    let mut existing = absolute;
    while !existing.exists() {
        existing = existing
            .parent()
            .ok_or_else(|| PolicyError::ParentTraversal {
                attempted: absolute.to_path_buf(),
            })?;
    }
    if existing == absolute {
        return CanonicalPath::for_existing(absolute);
    }
    // The leaf (or intermediate dirs) don't yet exist. Canonicalize the
    // closest existing ancestor and rejoin the non-existent suffix.
    let canonical_ancestor =
        std::fs::canonicalize(existing).map_err(|e| PolicyError::io(existing.to_path_buf(), e))?;
    let suffix = absolute
        .strip_prefix(existing)
        .expect("existing ancestor must prefix absolute path");
    let full = canonical_ancestor.join(suffix);
    CanonicalPath::for_target(&full)
}

fn extract_host(url: &str) -> Option<String> {
    let (_scheme, rest) = url.split_once("://")?;
    let host_part = rest.split(['/', '?', '#']).next()?;
    // Strip user:pass@
    let after_auth = match host_part.rsplit_once('@') {
        Some((_, hp)) => hp,
        None => host_part,
    };
    // Strip :port
    let host = match after_auth.rsplit_once(':') {
        Some((h, _)) => h,
        None => after_auth,
    };
    if host.is_empty() {
        None
    } else {
        Some(host.to_owned())
    }
}

fn parse_shell_program(command: &str) -> Result<ast::Program, PolicyError> {
    let mut parser = Parser::new(
        Cursor::new(command),
        &ParserOptions::default(),
        &SourceInfo::default(),
    );
    parser
        .parse_program()
        .map_err(|e| PolicyError::ShellCommandRejected {
            reason: "unparseable",
            fragment: e.to_string(),
        })
}

fn reject_strict_mode_constructs(program: &ast::Program) -> Result<(), PolicyError> {
    for command in &program.complete_commands {
        visit_compound_list_strict(command)?;
    }
    Ok(())
}

fn visit_compound_list_strict(list: &ast::CompoundList) -> Result<(), PolicyError> {
    for item in &list.0 {
        visit_and_or_list_strict(&item.0)?;
    }
    Ok(())
}

fn visit_and_or_list_strict(list: &ast::AndOrList) -> Result<(), PolicyError> {
    visit_pipeline_strict(&list.first)?;
    for item in &list.additional {
        match item {
            ast::AndOr::And(pipeline) | ast::AndOr::Or(pipeline) => {
                visit_pipeline_strict(pipeline)?;
            }
        }
    }
    Ok(())
}

fn visit_pipeline_strict(pipeline: &ast::Pipeline) -> Result<(), PolicyError> {
    for command in &pipeline.seq {
        visit_command_strict(command)?;
    }
    Ok(())
}

fn visit_command_strict(command: &ast::Command) -> Result<(), PolicyError> {
    match command {
        ast::Command::Simple(simple) => visit_simple_strict(simple),
        ast::Command::Compound(compound, _) => visit_compound_strict(compound),
        ast::Command::Function(_) => Err(PolicyError::ShellCommandRejected {
            reason: "function_definition",
            fragment: "fn() {...}".to_owned(),
        }),
        ast::Command::ExtendedTest(_) => Ok(()),
    }
}

fn visit_compound_strict(command: &ast::CompoundCommand) -> Result<(), PolicyError> {
    match command {
        ast::CompoundCommand::Arithmetic(_) => Ok(()),
        ast::CompoundCommand::ArithmeticForClause(cmd) => {
            visit_compound_list_strict(&cmd.body.list)
        }
        ast::CompoundCommand::BraceGroup(cmd) => visit_compound_list_strict(&cmd.list),
        ast::CompoundCommand::Subshell(cmd) => visit_compound_list_strict(&cmd.list),
        ast::CompoundCommand::ForClause(cmd) => visit_compound_list_strict(&cmd.body.list),
        ast::CompoundCommand::CaseClause(cmd) => {
            for case in &cmd.cases {
                if let Some(list) = &case.cmd {
                    visit_compound_list_strict(list)?;
                }
            }
            Ok(())
        }
        ast::CompoundCommand::IfClause(cmd) => {
            visit_compound_list_strict(&cmd.condition)?;
            visit_compound_list_strict(&cmd.then)?;
            if let Some(elses) = &cmd.elses {
                for else_clause in elses {
                    if let Some(condition) = &else_clause.condition {
                        visit_compound_list_strict(condition)?;
                    }
                    visit_compound_list_strict(&else_clause.body)?;
                }
            }
            Ok(())
        }
        ast::CompoundCommand::WhileClause(cmd) | ast::CompoundCommand::UntilClause(cmd) => {
            visit_compound_list_strict(&cmd.0)?;
            visit_compound_list_strict(&cmd.1.list)
        }
    }
}

fn visit_simple_strict(simple: &ast::SimpleCommand) -> Result<(), PolicyError> {
    let Some(program) = simple.word_or_name.as_ref() else {
        return Ok(());
    };
    let program = program.value.trim();
    match program {
        "eval" => Err(PolicyError::ShellCommandRejected {
            reason: "eval",
            fragment: program.to_owned(),
        }),
        "exec" => Err(PolicyError::ShellCommandRejected {
            reason: "exec",
            fragment: program.to_owned(),
        }),
        "source" => Err(PolicyError::ShellCommandRejected {
            reason: "source",
            fragment: program.to_owned(),
        }),
        "." => Err(PolicyError::ShellCommandRejected {
            reason: "dot_source",
            fragment: program.to_owned(),
        }),
        _ => Ok(()),
    }
}

// ---------- legacy helpers retained for the deprecated trait methods

fn shell_command_programs(command: &str) -> anyhow::Result<Vec<String>> {
    if command.trim().is_empty() {
        return Ok(Vec::new());
    }

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
#[allow(deprecated)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn check_shell_command_rejects_disallowed_program() {
        let policy = DefaultToolPolicy::new(PolicySettings {
            allowed_shell_commands: vec!["git".to_owned()],
            ..PolicySettings::default()
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
            ..PolicySettings::default()
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
            ..PolicySettings::default()
        });

        policy
            .check_shell_command("git status | rg modified")
            .await
            .expect("pipeline should be allowed");
    }
}
