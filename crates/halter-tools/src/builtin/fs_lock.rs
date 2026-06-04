// pattern: Imperative Shell

use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use dashmap::DashMap;
use parking_lot::{ArcRwLockReadGuard, ArcRwLockWriteGuard, RawRwLock, RwLock};

type PathGuardMap = Arc<DashMap<PathBuf, Arc<RwLock<()>>>>;

#[derive(Debug, Clone)]
/// Per-path read/write lock map shared by file-mutating tools.
pub struct PathLockMap {
    entries: PathGuardMap,
}

impl Default for PathLockMap {
    fn default() -> Self {
        Self {
            entries: Arc::new(DashMap::new()),
        }
    }
}

impl PathLockMap {
    /// Acquire a shared read lock for a path.
    pub fn acquire_read(&self, path: &Path) -> anyhow::Result<PathReadGuard> {
        let key = canonical_lock_path(path)?;
        let entry = self.entry_for(&key);
        Ok(PathReadGuard {
            _guard: entry.read_arc(),
        })
    }

    /// Acquire an exclusive write lock for a path.
    pub fn acquire_write(&self, path: &Path) -> anyhow::Result<PathWriteGuard> {
        let key = canonical_lock_path(path)?;
        let entry = self.entry_for(&key);
        Ok(PathWriteGuard {
            _guard: entry.write_arc(),
        })
    }

    fn entry_for(&self, key: &Path) -> Arc<RwLock<()>> {
        self.entries
            .entry(key.to_path_buf())
            .or_insert_with(|| Arc::new(RwLock::new(())))
            .clone()
    }
}

/// Read guard returned by [`PathLockMap::acquire_read`].
pub struct PathReadGuard {
    _guard: ArcRwLockReadGuard<RawRwLock, ()>,
}

/// Write guard returned by [`PathLockMap::acquire_write`].
pub struct PathWriteGuard {
    _guard: ArcRwLockWriteGuard<RawRwLock, ()>,
}

fn canonical_lock_path(path: &Path) -> anyhow::Result<PathBuf> {
    let absolute = absolute_path(path)?;
    if absolute.exists() {
        return Ok(std::fs::canonicalize(&absolute)?);
    }

    let normalized = normalize_components(&absolute);
    let mut existing_ancestor = normalized.as_path();
    while !existing_ancestor.exists() {
        existing_ancestor = existing_ancestor.parent().ok_or_else(|| {
            anyhow::anyhow!(
                "failed to canonicalize lock path '{}': missing parent",
                path.display()
            )
        })?;
    }

    let resolved_ancestor = std::fs::canonicalize(existing_ancestor)?;
    let suffix = normalized
        .strip_prefix(existing_ancestor)
        .expect("existing ancestor must prefix normalized path");
    Ok(resolved_ancestor.join(suffix))
}

fn absolute_path(path: &Path) -> anyhow::Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn normalize_components(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::RootDir | Component::Prefix(_) | Component::Normal(_) => {
                normalized.push(component);
            }
        }
    }
    normalized
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use super::*;

    #[test]
    fn allows_concurrent_reads() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("file.txt");
        std::fs::write(&path, "hello").expect("write");
        let locks = PathLockMap::default();
        let _first = locks.acquire_read(&path).expect("first read lock");
        let _second = locks.acquire_read(&path).expect("second read lock");
    }

    #[test]
    fn blocks_write_while_read_is_held() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("file.txt");
        std::fs::write(&path, "hello").expect("write");
        let locks = Arc::new(PathLockMap::default());
        let read_guard = locks.acquire_read(&path).expect("read lock");
        let (tx, rx) = mpsc::channel();
        let path_for_thread = path.clone();
        let locks_for_thread = locks.clone();

        let handle = thread::spawn(move || {
            let _write = locks_for_thread
                .acquire_write(&path_for_thread)
                .expect("write lock");
            tx.send(()).expect("send");
        });

        assert!(rx.recv_timeout(Duration::from_millis(50)).is_err());
        drop(read_guard);
        rx.recv_timeout(Duration::from_secs(1))
            .expect("write lock should unblock");
        handle.join().expect("join");
    }
}
