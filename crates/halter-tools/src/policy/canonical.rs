// pattern: Functional Core (with a tiny, well-contained unsafe shell)
//
// `CanonicalPath` pairs an already-canonicalized path with an `OwnedFd` /
// `OwnedHandle` on its parent directory, so that subsequent `openat` /
// `renameat` tool operations close the symlink-swap TOCTOU window between
// policy check and side-effect. See
// `docs/design-plans/2026-04-17-review-remediation-core.md` §Glossary
// ("FD-bound path") and Phase 1.

use std::path::{Path, PathBuf};

use super::errors::PolicyError;

#[derive(Debug)]
pub struct CanonicalPath {
    path: PathBuf,
    #[cfg(unix)]
    parent_dir_fd: std::os::fd::OwnedFd,
    #[cfg(windows)]
    parent_dir: PathBuf,
}

impl CanonicalPath {
    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn into_path(self) -> PathBuf {
        self.path
    }

    #[cfg(unix)]
    pub fn parent_dir_fd(&self) -> std::os::fd::BorrowedFd<'_> {
        use std::os::fd::AsFd;
        self.parent_dir_fd.as_fd()
    }

    /// Resolve an already-existing path. Canonicalizes through symlinks and
    /// opens the canonical parent directory without following further symlinks.
    ///
    /// Intended for read paths.
    pub(crate) fn for_existing(absolute: &Path) -> Result<Self, PolicyError> {
        let canonical = std::fs::canonicalize(absolute)
            .map_err(|e| PolicyError::io(absolute.to_path_buf(), e))?;
        let parent = canonical
            .parent()
            .ok_or_else(|| PolicyError::ParentTraversal {
                attempted: canonical.clone(),
            })?
            .to_path_buf();
        let fd = open_dir_nofollow(&parent)?;
        Ok(Self::from_parts(canonical, fd, parent))
    }

    /// Resolve a path whose leaf may not yet exist. Canonicalizes the parent
    /// (which *must* exist) and preserves the requested leaf name.
    ///
    /// Intended for write paths.
    pub(crate) fn for_target(absolute: &Path) -> Result<Self, PolicyError> {
        let parent = absolute
            .parent()
            .ok_or_else(|| PolicyError::ParentTraversal {
                attempted: absolute.to_path_buf(),
            })?;
        let leaf = absolute
            .file_name()
            .ok_or_else(|| PolicyError::ParentTraversal {
                attempted: absolute.to_path_buf(),
            })?;
        let canonical_parent = std::fs::canonicalize(parent)
            .map_err(|e| PolicyError::io(parent.to_path_buf(), e))?;
        let fd = open_dir_nofollow(&canonical_parent)?;
        let canonical = canonical_parent.join(leaf);
        Ok(Self::from_parts(canonical, fd, canonical_parent))
    }

    #[cfg(unix)]
    fn from_parts(
        path: PathBuf,
        fd: std::os::fd::OwnedFd,
        _parent: PathBuf,
    ) -> Self {
        Self {
            path,
            parent_dir_fd: fd,
        }
    }

    #[cfg(windows)]
    fn from_parts(path: PathBuf, _fd: (), parent: PathBuf) -> Self {
        Self {
            path,
            parent_dir: parent,
        }
    }
}

#[cfg(unix)]
fn open_dir_nofollow(path: &Path) -> Result<std::os::fd::OwnedFd, PolicyError> {
    use std::ffi::CString;
    use std::os::fd::FromRawFd;
    use std::os::unix::ffi::OsStrExt;

    let bytes = path.as_os_str().as_bytes();
    let c_path = CString::new(bytes).map_err(|_| PolicyError::ParentTraversal {
        attempted: path.to_path_buf(),
    })?;

    // O_NOFOLLOW: refuse if `path` itself is a symlink. Combined with the
    // caller's prior `canonicalize` this rules out a post-canonicalize swap
    // of the parent entry itself.
    //
    // O_DIRECTORY: fail if it isn't a directory.
    //
    // O_CLOEXEC: don't leak the fd into children.
    //
    // `O_RDONLY` is sufficient for use as an `openat` relative root.
    let flags = libc::O_NOFOLLOW | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_RDONLY;
    let raw = unsafe { libc::open(c_path.as_ptr(), flags) };
    if raw < 0 {
        let err = std::io::Error::last_os_error();
        return Err(PolicyError::io(path.to_path_buf(), err));
    }
    // SAFETY: `raw` is a fresh fd just returned by `open`; we exclusively own
    // it until `OwnedFd::drop` closes it.
    Ok(unsafe { std::os::fd::OwnedFd::from_raw_fd(raw) })
}

#[cfg(windows)]
#[allow(clippy::unnecessary_wraps)]
fn open_dir_nofollow(_path: &Path) -> Result<(), PolicyError> {
    // Advisory on Windows per design doc; full reparse-point rejection is
    // follow-up work (§Additional Considerations / Windows caveat).
    Ok(())
}
