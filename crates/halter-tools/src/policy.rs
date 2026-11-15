// pattern: Functional Core
//
// Capability-oriented tool policy. See
// `docs/design-plans/2026-04-17-review-remediation-core.md`.
//
// The trait surface is intentionally small and capability-typed:
// `check_read_path`, `check_write_path`, `check_process_signal`,
// `check_shell_enabled`, `check_network`,
// `check_subagent_spawn_typed`. Each call returns either `()` or a
// `CanonicalPath` (a path + parent dir fd), and errors are `PolicyError`
// variants — no name-based bypass surface and no anyhow erasure.

use std::net::IpAddr;
use std::path::{Component, Path, PathBuf};
use std::sync::OnceLock;

use async_trait::async_trait;
use globset::{Glob, GlobSet, GlobSetBuilder};

mod canonical;
mod errors;

#[cfg(test)]
mod security_tests;

pub use canonical::CanonicalPath;
pub use errors::{Pid, PolicyError};

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
/// Concrete settings used by [`DefaultToolPolicy`].
pub struct PolicySettings {
    pub allowed_write_roots: Vec<PathBuf>,
    pub allowed_read_roots: Vec<PathBuf>,
    pub sensitive_path_patterns: Vec<String>,
    pub max_read_bytes: usize,
    pub shell_enabled: bool,
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
            allowed_write_roots: vec![PathBuf::from("."), default_write_root()],
            allowed_read_roots: default_read_roots(),
            sensitive_path_patterns: default_sensitive_patterns(),
            max_read_bytes: 1_048_576,
            shell_enabled: true,
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

/// Built-in scratch write root, `<temp dir>/halter`. Unlike configured roots,
/// it is created on first use rather than reported missing.
#[must_use]
pub fn default_write_root() -> PathBuf {
    std::env::temp_dir().join("halter")
}

fn default_read_roots() -> Vec<PathBuf> {
    vec![PathBuf::from("."), std::env::temp_dir()]
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
/// Capability checks enforced before built-in tools perform side effects.
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

    /// Synchronous mirror of [`Self::check_write_path`]. Callers running on a
    /// blocking worker (e.g. `spawn_blocking`) use this to move the
    /// authorization check next to the actual filesystem write, closing the
    /// TOCTOU window between check and open (finding H33). Default
    /// implementations must perform identical policy enforcement to the
    /// async variant.
    fn check_write_path_blocking(&self, path: &Path) -> Result<CanonicalPath, PolicyError>;

    /// Authorize sending a process signal to `pid`. Enforces the session's
    /// process-tree boundary when configured.
    async fn check_process_signal(&self, pid: Pid) -> Result<(), PolicyError>;

    /// Authorize running shell commands (the `shell`, `background` and `pty`
    /// tools). This is a kill switch only; commands are not inspected.
    async fn check_shell_enabled(&self) -> Result<(), PolicyError>;

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
/// Default tool policy implementation.
pub struct DefaultToolPolicy {
    settings: PolicySettings,
    sensitive_glob_set: OnceLock<GlobSet>,
}

impl DefaultToolPolicy {
    /// Build a policy from explicit settings.
    ///
    /// The sensitive-path glob set is compiled lazily on the first check that
    /// needs it and then cached for the lifetime of this policy.
    #[must_use]
    pub fn new(settings: PolicySettings) -> Self {
        Self {
            settings,
            sensitive_glob_set: OnceLock::new(),
        }
    }

    /// Borrow the underlying policy settings.
    #[must_use]
    pub fn settings(&self) -> &PolicySettings {
        &self.settings
    }

    fn sensitive_glob_set(&self) -> Result<&GlobSet, PolicyError> {
        if let Some(set) = self.sensitive_glob_set.get() {
            return Ok(set);
        }

        let mut builder = GlobSetBuilder::new();
        for pattern in &self.settings.sensitive_path_patterns {
            let glob = Glob::new(pattern).map_err(|_| PolicyError::SensitivePathDenied {
                attempted: PathBuf::from(pattern),
                rule: "invalid_pattern",
            })?;
            builder.add(glob);
        }
        let set = builder
            .build()
            .map_err(|_| PolicyError::SensitivePathDenied {
                attempted: PathBuf::new(),
                rule: "invalid_pattern_set",
            })?;

        // If multiple threads race here, all builds use the same inputs and
        // produce the same set; dropping a loser is harmless.
        let _ = self.sensitive_glob_set.set(set);
        Ok(self.sensitive_glob_set.get().expect("just initialized"))
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
        if let Some(root) = self.settings.process_tree_root {
            let descendants = crate::builtin::process::list_descendants(root);
            if !descendants.contains(&pid) {
                return Err(PolicyError::ProcessOutsideTree { pid });
            }
        }
        Ok(())
    }

    async fn check_shell_enabled(&self) -> Result<(), PolicyError> {
        if self.settings.shell_enabled {
            Ok(())
        } else {
            Err(PolicyError::ShellDisabled)
        }
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
/// typed `NonexistentRoot` error instead of being silently dropped (AC1.14),
/// except the built-in [`default_write_root`], which is created on demand.
fn canonicalize_root_list(roots: &[PathBuf]) -> Result<Vec<PathBuf>, PolicyError> {
    let mut out = Vec::with_capacity(roots.len());
    for root in roots {
        let absolute = absolute_path_typed(root)?;
        let canonical = std::fs::canonicalize(&absolute)
            .or_else(|error| {
                if absolute != default_write_root() {
                    return Err(error);
                }
                std::fs::create_dir_all(&absolute)?;
                std::fs::canonicalize(&absolute)
            })
            .map_err(|_| PolicyError::NonexistentRoot {
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
