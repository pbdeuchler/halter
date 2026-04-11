// pattern: Imperative Shell

use std::fs::File;
use std::sync::Arc;

use memmap2::Mmap;
use rayon::prelude::*;
use serde_json::Value;
use tokio_util::sync::CancellationToken;

use crate::PathLockMap;

use super::types::{
    AggregateResult, FileSearchResult, SearchConfig, build_matcher, build_response,
    collect_entries, decode_searchable_bytes, resolve_search_root, search_text,
};

pub fn run(
    working_dir: std::path::PathBuf,
    path_locks: Arc<PathLockMap>,
    cancel: CancellationToken,
    config: SearchConfig,
) -> anyhow::Result<Value> {
    let search_root = resolve_search_root(&working_dir, &config.path);
    let entries = collect_entries(&search_root, config.glob.as_deref(), config.type_filter.as_deref())?;
    let matcher = build_matcher(&config.pattern, config.ignore_case, config.multiline)?;
    let files_searched = entries.len() as u64;

    let mut files: Vec<FileSearchResult> = entries
        .par_iter()
        .filter_map(|entry| {
            if cancel.is_cancelled() {
                return None;
            }

            let _lock = path_locks.acquire_read(&entry.absolute_path).ok()?;
            let bytes = read_bytes(&entry.absolute_path).ok()?;
            let text = decode_searchable_bytes(&bytes)?;
            let mut result = search_text(&text, &matcher, &config);
            if result.total_matches == 0 {
                return None;
            }
            result.path = entry.display_path.clone();
            Some(match config.output_mode {
                super::types::OutputMode::Content => result,
                _ => FileSearchResult {
                    path: result.path,
                    matches: Vec::new(),
                    total_matches: result.total_matches,
                },
            })
        })
        .collect();

    files.sort_by(|left, right| left.path.cmp(&right.path));
    let total_matches = files.iter().map(|file| file.total_matches).sum();
    Ok(build_response(
        AggregateResult {
            files_searched,
            files_with_matches: files.len() as u64,
            total_matches,
            files,
        },
        &config,
    ))
}

fn read_bytes(path: &std::path::Path) -> anyhow::Result<Vec<u8>> {
    let file = File::open(path)?;
    let mapped = unsafe { Mmap::map(&file) };
    match mapped {
        Ok(mapped) => Ok(mapped.as_ref().to_vec()),
        Err(_) => Ok(std::fs::read(path)?),
    }
}
