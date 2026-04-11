// pattern: Imperative Shell

use std::sync::Arc;

use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::PathLockMap;

use super::types::{
    AggregateResult, FileSearchResult, SearchConfig, build_matcher, build_response,
    collect_entries, decode_searchable_bytes, resolve_search_root, search_text,
};

#[cfg_attr(feature = "advanced-tools", allow(dead_code))]
pub fn run(
    working_dir: std::path::PathBuf,
    path_locks: Arc<PathLockMap>,
    cancel: CancellationToken,
    config: SearchConfig,
) -> anyhow::Result<Value> {
    let search_root = resolve_search_root(&working_dir, &config.path);
    let entries = collect_entries(&search_root, config.glob.as_deref(), config.type_filter.as_deref())?;
    let matcher = build_matcher(&config.pattern, config.ignore_case, config.multiline)?;
    let mut files = Vec::new();
    let mut files_searched = 0u64;
    let mut files_with_matches = 0u64;
    let mut total_matches = 0u64;

    for entry in entries {
        if cancel.is_cancelled() {
            anyhow::bail!("failed to execute grep tool: cancelled");
        }

        let _lock = path_locks.acquire_read(&entry.absolute_path)?;
        let bytes = std::fs::read(&entry.absolute_path)?;
        let Some(text) = decode_searchable_bytes(&bytes) else {
            continue;
        };
        files_searched += 1;
        let mut file = search_text(&text, &matcher, &config);
        file.path = entry.display_path;
        if file.total_matches > 0 {
            files_with_matches += 1;
            total_matches += file.total_matches;
            files.push(match config.output_mode {
                super::types::OutputMode::Content => file,
                _ => FileSearchResult {
                    path: file.path,
                    matches: Vec::new(),
                    total_matches: file.total_matches,
                },
            });
        }
    }

    Ok(build_response(
        AggregateResult {
            files,
            files_searched,
            files_with_matches,
            total_matches,
        },
        &config,
    ))
}
