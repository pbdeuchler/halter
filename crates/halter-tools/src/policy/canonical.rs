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
/// Canonicalized path plus parent-directory handle for safe open/write.
pub struct CanonicalPath {
    path: PathBuf,
    #[cfg(unix)]
    parent_dir_fd: std::os::fd::OwnedFd,
    #[cfg(windows)]
    parent_dir: PathBuf,
}

impl CanonicalPath {
    /// Canonical absolute path authorized by policy.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Consume the wrapper and return the canonical path.
    pub fn into_path(self) -> PathBuf {
        self.path
    }

    /// Open the authorized leaf for reading without following a new symlink.
    pub fn open_read_blocking(&self) -> Result<std::fs::File, PolicyError> {
        open_leaf_read(self)
    }

    /// Atomically replace the authorized leaf with `bytes`.
    pub fn atomic_write_blocking(&self, bytes: &[u8]) -> Result<(), PolicyError> {
        write_leaf_atomic(self, bytes)
    }

    /// Borrow the pinned parent directory file descriptor.
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
        let canonical_parent =
            std::fs::canonicalize(parent).map_err(|e| PolicyError::io(parent.to_path_buf(), e))?;
        let fd = open_dir_nofollow(&canonical_parent)?;
        let canonical = canonical_parent.join(leaf);
        Ok(Self::from_parts(canonical, fd, canonical_parent))
    }

    #[cfg(unix)]
    fn from_parts(path: PathBuf, fd: std::os::fd::OwnedFd, _parent: PathBuf) -> Self {
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
fn open_leaf_read(path: &CanonicalPath) -> Result<std::fs::File, PolicyError> {
    use std::os::fd::{AsRawFd, FromRawFd};

    let leaf = leaf_c_string(path)?;
    let flags = libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW;
    let raw = unsafe { libc::openat(path.parent_dir_fd.as_raw_fd(), leaf.as_ptr(), flags) };
    if raw < 0 {
        return Err(PolicyError::io(
            path.path.clone(),
            std::io::Error::last_os_error(),
        ));
    }
    // SAFETY: `raw` is a fresh fd returned by `openat`; `File` takes ownership
    // and closes it on drop.
    Ok(unsafe { std::fs::File::from_raw_fd(raw) })
}

#[cfg(windows)]
fn open_leaf_read(path: &CanonicalPath) -> Result<std::fs::File, PolicyError> {
    std::fs::File::open(&path.path).map_err(|error| PolicyError::io(path.path.clone(), error))
}

#[cfg(unix)]
fn write_leaf_atomic(path: &CanonicalPath, bytes: &[u8]) -> Result<(), PolicyError> {
    use std::ffi::CString;
    use std::io::Write;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

    let leaf = leaf_c_string(path)?;
    let dirfd = path.parent_dir_fd.as_raw_fd();
    let temp_name = CString::new(format!(
        ".halter-tmp-{}-{}",
        std::process::id(),
        TEMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    ))
    .map_err(|error| PolicyError::io(path.path.clone(), std::io::Error::other(error)))?;

    let flags = libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_CLOEXEC;
    let raw = unsafe { libc::openat(dirfd, temp_name.as_ptr(), flags, 0o600) };
    if raw < 0 {
        return Err(PolicyError::io(
            path.path.clone(),
            std::io::Error::last_os_error(),
        ));
    }

    // SAFETY: `raw` is a fresh fd returned by `openat`; `File` takes ownership
    // and closes it on every return path.
    let mut file = unsafe { std::fs::File::from_raw_fd(raw) };
    if let Err(error) = file.write_all(bytes).and_then(|()| file.flush()) {
        let _ = unsafe { libc::unlinkat(dirfd, temp_name.as_ptr(), 0) };
        return Err(PolicyError::io(path.path.clone(), error));
    }
    drop(file);

    let renamed = unsafe { libc::renameat(dirfd, temp_name.as_ptr(), dirfd, leaf.as_ptr()) };
    if renamed < 0 {
        let error = std::io::Error::last_os_error();
        let _ = unsafe { libc::unlinkat(dirfd, temp_name.as_ptr(), 0) };
        return Err(PolicyError::io(path.path.clone(), error));
    }

    Ok(())
}

#[cfg(windows)]
fn write_leaf_atomic(path: &CanonicalPath, bytes: &[u8]) -> Result<(), PolicyError> {
    use std::io::Write;

    let parent = path
        .path
        .parent()
        .ok_or_else(|| PolicyError::ParentTraversal {
            attempted: path.path.clone(),
        })?;
    let mut temp = tempfile::NamedTempFile::new_in(parent)
        .map_err(|error| PolicyError::io(parent.to_path_buf(), error))?;
    temp.write_all(bytes)
        .and_then(|()| temp.flush())
        .map_err(|error| PolicyError::io(path.path.clone(), error))?;
    temp.persist(&path.path)
        .map(|_| ())
        .map_err(|error| PolicyError::io(path.path.clone(), error.error))
}

#[cfg(unix)]
fn leaf_c_string(path: &CanonicalPath) -> Result<std::ffi::CString, PolicyError> {
    use std::os::unix::ffi::OsStrExt;

    let leaf = path
        .path
        .file_name()
        .ok_or_else(|| PolicyError::ParentTraversal {
            attempted: path.path.clone(),
        })?;
    std::ffi::CString::new(leaf.as_bytes()).map_err(|_| PolicyError::ParentTraversal {
        attempted: path.path.clone(),
    })
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
