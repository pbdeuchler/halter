// pattern: Functional Core
//
// Capability-oriented tool policy. See
// `docs/design-plans/2026-04-17-review-remediation-core.md`.
//
// The trait surface is intentionally small and capability-typed:
// `check_read_path`, `check_write_path`, `check_process_signal`,
// `check_shell_enabled`, `check_shell_command_strict`, `check_network`,
// `check_subagent_spawn_typed`. Each call returns either `()` or a
// `CanonicalPath` (a path + parent dir fd), and errors are `PolicyError`
// variants — no name-based bypass surface and no anyhow erasure.

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
/// `PolicySettings::allowed_loopback` is the only way loopback IPs reach the
/// network after the Phase 1 `NetworkPolicy` refactor.
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
    /// Carried for forward compatibility with config files. The current
    /// `Strict`/`Relaxed` mode split does not consult this list — the
    /// brush-parser AST walk in `check_shell_command_strict` enforces shell
    /// rules without a per-program allowlist. Reintroducing allowlisting on
    /// top of strict mode is a Phase 3 design decision.
    pub allowed_shell_commands: Vec<String>,
    pub shell_timeout_secs: u64,
    pub network_enabled: bool,
    /// Remote-host allowlist. Default is `vec!["*"]` (allow any non-loopback
    /// host when `network_enabled` is true). The `*` entry is short-circuited
    /// to allow; any other entry must match the request host case-insensitively
    /// as a literal.
    pub allowed_hosts: Vec<String>,
    /// Loopback sidecar allowlist. Default is `Vec::new()` (deny). Loopback
    /// hosts (`localhost`, `127/8`, `::1`) flow through this gate exclusively,
    /// even when `allowed_hosts` contains `*`.
    pub allowed_loopback: Vec<LoopbackAllow>,
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
            allowed_hosts: vec!["*".to_owned()],
            allowed_loopback: Vec::new(),
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

    /// Synchronous mirror of [`check_write_path`]. Callers running on a
    /// blocking worker (e.g. `spawn_blocking`) use this to move the
    /// authorization check next to the actual filesystem write, closing the
    /// TOCTOU window between check and open (finding H33). Default
    /// implementations must perform identical policy enforcement to the
    /// async variant.
    fn check_write_path_blocking(&self, path: &Path) -> Result<CanonicalPath, PolicyError>;

    /// Authorize sending a process signal to `pid`. Enforces the session's
    /// process-tree boundary when configured.
    async fn check_process_signal(&self, pid: Pid) -> Result<(), PolicyError>;

    /// Authorize the shell capability generically (e.g. spawning a PTY).
    async fn check_shell_enabled(&self) -> Result<(), PolicyError>;

    /// The shell mode this policy is configured to enforce. Tools that aren't
    /// hand-picking a mode (i.e. production builtins) should pull this and
    /// pass it to `check_shell_command_strict`. Tests pick a mode explicitly.
    fn shell_mode(&self) -> ShellMode;

    /// Authorize `command` under a specific `ShellMode`. Strict mode rejects
    /// function definitions and `eval`/`exec`/`source`/`.` invocations via
    /// the brush-parser AST walk.
    async fn check_shell_command_strict(
        &self,
        command: &str,
        mode: ShellMode,
    ) -> Result<(), PolicyError>;

    /// Authorize a network request to `url`. Loopback addresses are denied
    /// unless an entry in `allowed_loopback` matches; other hosts must match
    /// `allowed_hosts` (the `*` wildcard short-circuits to allow). A
    /// `network_enabled == false` kill switch denies every request.
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

    fn check_write_path_sync(&self, path: &Path) -> Result<CanonicalPath, PolicyError> {
        let absolute = absolute_path_typed(path)?;
        reject_parent_traversal_typed(&absolute)?;
        self.verify_not_sensitive(&absolute)?;
        let canonical = resolve_write_target_typed(&absolute)?;
        self.verify_not_sensitive(canonical.path())?;
        let roots = self.canonical_write_roots()?;
        Self::verify_under_any_root(canonical.path(), &roots)?;
        Ok(canonical)
    }
}

#[async_trait]
impl ToolPolicy for DefaultToolPolicy {
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
        self.check_write_path_sync(path)
    }

    fn check_write_path_blocking(&self, path: &Path) -> Result<CanonicalPath, PolicyError> {
        self.check_write_path_sync(path)
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

    fn shell_mode(&self) -> ShellMode {
        self.settings.shell_mode
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
        if !self.settings.network_enabled {
            return Err(PolicyError::NetworkDenied {
                url: url.to_owned(),
                rule: "network_disabled",
            });
        }
        let (host, port) =
            extract_host_and_port(url).ok_or_else(|| PolicyError::NetworkDenied {
                url: url.to_owned(),
                rule: "unparseable_url",
            })?;
        if is_loopback_host(&host) {
            return self.allow_loopback(url, &host, port);
        }
        if self
            .settings
            .allowed_hosts
            .iter()
            .any(|entry| entry.trim() == "*")
        {
            return Ok(());
        }
        if !self
            .settings
            .allowed_hosts
            .iter()
            .any(|entry| entry.eq_ignore_ascii_case(&host))
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
        let matches = self.settings.allowed_loopback.iter().any(|entry| {
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

fn is_loopback_host(host: &str) -> bool {
    if host.eq_ignore_ascii_case("localhost") {
        return true;
    }
    if let Ok(addr) = host.parse::<IpAddr>() {
        return addr.is_loopback();
    }
    false
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

fn extract_host_and_port(url: &str) -> Option<(String, Option<u16>)> {
    let (_scheme, rest) = url.split_once("://")?;
    let host_part = rest.split(['/', '?', '#']).next()?;
    let after_auth = match host_part.rsplit_once('@') {
        Some((_, hp)) => hp,
        None => host_part,
    };
    if let Some(after_bracket) = after_auth.strip_prefix('[') {
        let end = after_bracket.find(']')?;
        let host = &after_bracket[..end];
        let tail = &after_bracket[end + 1..];
        let port = tail
            .strip_prefix(':')
            .and_then(|value| value.parse::<u16>().ok());
        if host.is_empty() {
            return None;
        }
        return Some((host.to_owned(), port));
    }
    if let Some((host, port_str)) = after_auth.rsplit_once(':') {
        if host.is_empty() {
            return None;
        }
        let port = port_str.parse::<u16>().ok();
        return Some((host.to_owned(), port));
    }
    if after_auth.is_empty() {
        None
    } else {
        Some((after_auth.to_owned(), None))
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
