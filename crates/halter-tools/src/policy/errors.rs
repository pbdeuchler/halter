// pattern: Functional Core
//
// Capability-oriented policy errors. See
// `docs/design-plans/2026-04-17-review-remediation-core.md` §Architecture.1.

use std::path::PathBuf;

use thiserror::Error;

/// Process identifier used by policy checks.
pub type Pid = i32;

#[derive(Debug, Error)]
/// Typed error returned by tool policy checks.
pub enum PolicyError {
    #[error("path '{}' is not under any allowed root (roots: {roots:?})", attempted.display())]
    NotInRoot {
        attempted: PathBuf,
        roots: Vec<PathBuf>,
    },

    #[error("path '{}' denied by sensitive-path rule '{rule}'", attempted.display())]
    SensitivePathDenied {
        attempted: PathBuf,
        rule: &'static str,
    },

    #[error("path '{}' traverses a symlink out of its allowed root", attempted.display())]
    SymlinkEscape { attempted: PathBuf },

    #[error("pid {pid} is outside the session's tracked process tree")]
    ProcessOutsideTree { pid: Pid },

    #[error("shell usage is disabled by policy")]
    ShellDisabled,

    #[error("shell command rejected: {reason} (fragment: {fragment:?})")]
    ShellCommandRejected {
        reason: &'static str,
        fragment: String,
    },

    #[error("network request denied for '{url}' (rule: {rule})")]
    NetworkDenied { url: String, rule: &'static str },

    #[error("subagent limit reached: {kind} {current} exceeds max {limit}")]
    SubagentLimit {
        kind: &'static str,
        current: u32,
        limit: u32,
    },

    #[error("allowed root '{}' does not exist", root.display())]
    NonexistentRoot { root: PathBuf },

    #[error("path '{}' traverses outside its root via '..'", attempted.display())]
    ParentTraversal { attempted: PathBuf },

    #[error("read size {bytes} exceeds max_read_bytes {max}")]
    ReadTooLarge { bytes: usize, max: usize },

    #[error("io error resolving '{}': {source}", path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

impl PolicyError {
    pub(crate) fn io(path: PathBuf, source: std::io::Error) -> Self {
        Self::Io { path, source }
    }
}
