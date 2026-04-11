// pattern: Imperative Shell

use std::time::UNIX_EPOCH;

use async_trait::async_trait;
use globset::{GlobBuilder, GlobSet};
use halter_protocol::{ToolCapabilities, ToolConcurrency, ToolName, ToolResult, ToolSpec};
use ignore::WalkBuilder;
use serde_json::{Value, json};

use crate::{Tool, ToolContext};

use super::common::{ToolScope, ensure_not_cancelled, optional_bool, optional_string, optional_u64, required_string};

#[derive(Debug)]
pub struct GlobTool;

#[async_trait]
impl Tool for GlobTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("glob"),
            description: "Expand a glob pattern relative to the working directory".to_owned(),
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

    async fn execute(&self, context: ToolContext, input: Value) -> anyhow::Result<ToolResult> {
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
            matches.push(json!({
                "path": path,
                "file_type": file_type_name(metadata.file_type()),
                "mtime": mtime,
            }));

            if !sort_by_mtime
                && max_results.is_some_and(|max_results| matches.len() as u64 >= max_results)
            {
                break;
            }
        }

        if sort_by_mtime {
            matches.sort_by(|left, right| {
                let left_mtime = left["mtime"].as_u64().unwrap_or(0);
                let right_mtime = right["mtime"].as_u64().unwrap_or(0);
                right_mtime
                    .cmp(&left_mtime)
                    .then_with(|| left["path"].as_str().cmp(&right["path"].as_str()))
            });
            if let Some(max_results) = max_results {
                matches.truncate(max_results as usize);
            }
        }

        Ok(ToolResult::Json {
            value: json!({
                "matches": matches,
                "total_matches": matches.len(),
            }),
        })
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
    match expected {
        None => true,
        Some("file") => file_type.is_file(),
        Some("dir") => file_type.is_dir(),
        Some("symlink") => file_type.is_symlink(),
        Some(other) => unreachable!("schema already constrained file_type: {other}"),
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
