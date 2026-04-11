// pattern: Imperative Shell

use std::sync::Arc;

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::PathLockMap;

use super::types::{SearchConfig, run_advanced_search};

pub fn run(
    working_dir: std::path::PathBuf,
    path_locks: Arc<PathLockMap>,
    cancel: CancellationToken,
    config: SearchConfig,
) -> anyhow::Result<Value> {
    run_advanced_search(working_dir, path_locks, cancel, config)
}
