// pattern: Imperative Shell

use std::path::PathBuf;
use std::time::UNIX_EPOCH;

use async_trait::async_trait;
use globset::{GlobBuilder, GlobSet};
use halter_protocol::{ToolCapabilities, ToolConcurrency, ToolName, ToolSpec};
use ignore::WalkBuilder;
use serde::Serialize;
use serde_json::{Value, json};

use crate::{ToolContext, ToolOutput, TypedTool};

use super::common::{
    ToolScope, ensure_not_cancelled, optional_bool, optional_string, optional_u64, required_string,
};

const MAX_SORTED_CANDIDATES: usize = 10_000;

#[derive(Debug)]
/// Built-in tool for filesystem glob expansion.
pub struct GlobTool;

#[async_trait]
impl TypedTool for GlobTool {
    type Output = GlobOutput;

    fn describe(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("glob"),
            description: "Find files by glob pattern (e.g. `**/*.rs`, `src/**/*.ts`) relative \
                to the working directory, honoring .gitignore. Results are sorted by path; set \
                `sort_by_mtime` to order by most-recently-modified instead. Filter with \
                `file_type` (file/dir/symlink). For open-ended searches that need several \
                glob/grep rounds, delegate to a subagent instead."
                .to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string" },
                    "max_results": { "type": "integer", "minimum": 1 },
                    "file_type": { "type": "string", "enum": ["file", "dir", "symlink"] },
                    "sort_by_mtime": { "type": "boolean" }
                },
                "required": ["pattern"],
            }),
            concurrency: ToolConcurrency::ReadOnly,
            capabilities: ToolCapabilities::default(),
            provider_aliases: Default::default(),
        }
    }

    async fn run(&self, context: ToolContext, input: Value) -> anyhow::Result<Self::Output> {
        let _scope = ToolScope::new(&context, "glob");
        ensure_not_cancelled(&context.cancel)?;
        let pattern = required_string(&input, "pattern")?;
        let max_results = optional_u64(&input, "max_results")?;
        let file_type = optional_string(&input, "file_type");
        let sort_by_mtime = optional_bool(&input, "sort_by_mtime")?.unwrap_or(false);
        let matcher = build_glob_matcher(pattern)?;
        let mut builder = WalkBuilder::new(&context.working_dir);
        builder
            .standard_filters(true)
            .require_git(false)
            .follow_links(false)
            .sort_by_file_path(|left, right| left.cmp(right));

        let mut matches = Vec::new();
        for entry in builder.build() {
            ensure_not_cancelled(&context.cancel)?;
            let entry = entry?;
            let path = entry.path();
            if path == context.working_dir {
                continue;
            }

            let relative = path.strip_prefix(&context.working_dir).unwrap_or(path);
            if !matcher.is_match(relative) {
                continue;
            }

            let metadata = entry.metadata()?;
            if !matches_file_type(metadata.file_type(), file_type) {
                continue;
            }

            let mtime = metadata
                .modified()
                .ok()
                .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
                .map(|duration| duration.as_secs());
            matches.push(GlobMatch {
                path: path.to_path_buf(),
                file_type: file_type_name(metadata.file_type()),
                mtime,
            });
            if sort_by_mtime && matches.len() >= MAX_SORTED_CANDIDATES {
                break;
            }

            if !sort_by_mtime
                && max_results.is_some_and(|max_results| matches.len() as u64 >= max_results)
            {
                break;
            }
        }

        if sort_by_mtime {
            matches.sort_by(|left, right| {
                right
                    .mtime
                    .unwrap_or(0)
                    .cmp(&left.mtime.unwrap_or(0))
                    .then_with(|| left.path.as_os_str().cmp(right.path.as_os_str()))
            });
            if let Some(max_results) = max_results {
                matches.truncate(usize::try_from(max_results).unwrap_or(usize::MAX));
            }
        }

        Ok(GlobOutput {
            total_matches: matches.len(),
            matches,
        })
    }
}

#[derive(Debug, Serialize)]
/// Result of a `glob` call.
pub struct GlobOutput {
    matches: Vec<GlobMatch>,
    total_matches: usize,
}

#[derive(Debug, Serialize)]
struct GlobMatch {
    path: PathBuf,
    file_type: &'static str,
    mtime: Option<u64>,
}

impl ToolOutput for GlobOutput {
    /// One path per line; directories carry a trailing `/`.
    fn to_model_string(&self) -> String {
        if self.matches.is_empty() {
            return "No matches".to_owned();
        }
        self.matches
            .iter()
            .map(|entry| {
                let suffix = if entry.file_type == "dir" { "/" } else { "" };
                format!("{}{suffix}\n", entry.path.display())
            })
            .collect()
    }
}

fn build_glob_matcher(pattern: &str) -> anyhow::Result<GlobSet> {
    let glob = GlobBuilder::new(pattern)
        .literal_separator(false)
        .build()
        .map_err(|error| anyhow::anyhow!("invalid glob pattern '{pattern}': {error}"))?;
    let mut builder = globset::GlobSetBuilder::new();
    builder.add(glob);
    builder.build().map_err(Into::into)
}

fn matches_file_type(file_type: std::fs::FileType, expected: Option<&str>) -> bool {
    // Unknown variants are rejected (return false) rather than panicking on
    // `unreachable!`. Schema validation should have rejected the input
    // upstream, but treating this as a soft rejection means a schema
    // regression produces an empty result set instead of a runtime abort.
    // (finding L32)
    match expected {
        None => true,
        Some("file") => file_type.is_file(),
        Some("dir") => file_type.is_dir(),
        Some("symlink") => file_type.is_symlink(),
        Some(_) => false,
    }
}

fn file_type_name(file_type: std::fs::FileType) -> &'static str {
    if file_type.is_dir() {
        "dir"
    } else if file_type.is_symlink() {
        "symlink"
    } else {
        "file"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn glob_output_renders_one_path_per_line_and_keeps_the_json_shape() {
        let output = GlobOutput {
            matches: vec![
                GlobMatch {
                    path: PathBuf::from("/w/a \"é\".rs"),
                    file_type: "file",
                    mtime: Some(5),
                },
                GlobMatch {
                    path: PathBuf::from("/w/src"),
                    file_type: "dir",
                    mtime: None,
                },
            ],
            total_matches: 2,
        };

        assert_eq!(output.to_model_string(), "/w/a \"é\".rs\n/w/src/\n");
        assert_eq!(
            serde_json::to_value(&output).expect("serialize"),
            json!({
                "matches": [
                    { "path": "/w/a \"é\".rs", "file_type": "file", "mtime": 5 },
                    { "path": "/w/src", "file_type": "dir", "mtime": null },
                ],
                "total_matches": 2,
            })
        );
        let empty = GlobOutput {
            matches: Vec::new(),
            total_matches: 0,
        };
        assert_eq!(empty.to_model_string(), "No matches");
    }
}
