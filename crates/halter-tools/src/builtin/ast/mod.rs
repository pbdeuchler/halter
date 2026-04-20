// pattern: Imperative Shell

mod find;
mod language;
mod replace;

use std::path::{Path, PathBuf};

use async_trait::async_trait;
use globset::{GlobBuilder, GlobSet};
use halter_protocol::{ToolCapabilities, ToolConcurrency, ToolName, ToolResult, ToolSpec};
use ignore::WalkBuilder;
use serde_json::{Value, json};

use crate::{Tool, ToolContext};

use super::common::{ToolScope, ensure_not_cancelled, optional_string};

#[derive(Debug)]
pub struct AstGrepTool;

#[derive(Debug, Clone)]
pub(super) struct FileCandidate {
    pub absolute_path: PathBuf,
    pub display_path: String,
}

#[async_trait]
impl Tool for AstGrepTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("ast_grep"),
            description: "Search or rewrite source files with structural AST patterns".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "action": { "type": "string", "enum": ["find", "replace"] },
                    "patterns": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "rewrites": {
                        "type": "object",
                        "additionalProperties": { "type": "string" }
                    },
                    "path": { "type": "string" },
                    "glob": { "type": "string" },
                    "lang": { "type": "string" },
                    "selector": { "type": "string" },
                    "strictness": {
                        "type": "string",
                        "enum": ["cst", "smart", "ast", "relaxed", "signature", "template"]
                    },
                    "limit": { "type": "integer", "minimum": 1 },
                    "offset": { "type": "integer", "minimum": 0 },
                    "include_meta": { "type": "boolean" },
                    "dry_run": { "type": "boolean" },
                    "max_replacements": { "type": "integer", "minimum": 1 },
                    "max_files": { "type": "integer", "minimum": 1 },
                    "fail_on_parse_error": { "type": "boolean" }
                },
                "required": ["action"],
            }),
            concurrency: ToolConcurrency::Exclusive,
            capabilities: ToolCapabilities {
                mutating: true,
                requires_approval: false,
                cancellable: true,
                long_running: true,
            },
            provider_aliases: Default::default(),
        }
    }

    async fn execute(&self, context: ToolContext, input: Value) -> anyhow::Result<ToolResult> {
        let _scope = ToolScope::new(&context, "ast_grep");
        ensure_not_cancelled(&context.cancel)?;
        let action = optional_string(&input, "action")
            .ok_or_else(|| anyhow::anyhow!("invalid tool input: missing string field 'action'"))?;

        let value = match action {
            "find" => {
                let config = find::FindConfig::parse(&input)?;
                let working_dir = context.working_dir.clone();
                let path_locks = context.path_locks.clone();
                let cancel = context.cancel.clone();
                tokio::task::spawn_blocking(move || {
                    find::run(config, working_dir, path_locks, cancel)
                })
                .await??
            }
            "replace" => {
                let config = replace::ReplaceConfig::parse(&input)?;
                let candidates = collect_candidates(
                    &context.working_dir,
                    config.path.as_deref(),
                    config.glob.as_deref(),
                    config.lang.as_deref(),
                )?;
                let path_locks = context.path_locks.clone();
                let cancel = context.cancel.clone();
                let policy = context.policy.clone();
                tokio::task::spawn_blocking(move || {
                    replace::run(config, candidates, path_locks, cancel, policy)
                })
                .await??
            }
            other => anyhow::bail!("failed to execute ast_grep tool: unknown action '{other}'"),
        };

        Ok(ToolResult::Json { value })
    }
}

pub(super) fn collect_candidates(
    working_dir: &Path,
    path: Option<&str>,
    glob: Option<&str>,
    explicit_lang: Option<&str>,
) -> anyhow::Result<Vec<FileCandidate>> {
    let search_root = resolve_search_root(working_dir, path);
    let metadata = std::fs::metadata(&search_root).map_err(|error| {
        anyhow::anyhow!(
            "failed to execute ast_grep tool: search path '{}' does not exist: {error}",
            search_root.display()
        )
    })?;

    if metadata.is_file() {
        if explicit_lang.is_none() && language::infer_language_from_path(&search_root).is_none() {
            return Ok(Vec::new());
        }
        return Ok(vec![FileCandidate {
            display_path: display_path(working_dir, &search_root),
            absolute_path: search_root,
        }]);
    }
    if !metadata.is_dir() {
        anyhow::bail!(
            "failed to execute ast_grep tool: search path '{}' is not a file or directory",
            search_root.display()
        );
    }

    let matcher = build_glob_set(glob)?;
    let mut builder = WalkBuilder::new(&search_root);
    builder
        .standard_filters(true)
        .require_git(false)
        .follow_links(false)
        .sort_by_file_path(|left, right| left.cmp(right));

    let mut candidates = Vec::new();
    for entry in builder.build() {
        let entry = entry?;
        let path = entry.path();
        if !entry.file_type().is_some_and(|kind| kind.is_file()) {
            continue;
        }

        let relative = path.strip_prefix(&search_root).unwrap_or(path);
        if let Some(matcher) = matcher.as_ref()
            && !matcher.is_match(relative)
        {
            continue;
        }
        if explicit_lang.is_none() && language::infer_language_from_path(path).is_none() {
            continue;
        }

        candidates.push(FileCandidate {
            absolute_path: path.to_path_buf(),
            display_path: relative.to_string_lossy().into_owned(),
        });
    }

    Ok(candidates)
}

fn resolve_search_root(working_dir: &Path, path: Option<&str>) -> PathBuf {
    match path.map(str::trim).filter(|path| !path.is_empty()) {
        None => working_dir.to_path_buf(),
        Some(path) => {
            let candidate = PathBuf::from(path);
            if candidate.is_absolute() {
                candidate
            } else {
                working_dir.join(candidate)
            }
        }
    }
}

fn display_path(working_dir: &Path, path: &Path) -> String {
    path.strip_prefix(working_dir)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
}

fn build_glob_set(glob: Option<&str>) -> anyhow::Result<Option<GlobSet>> {
    let Some(glob) = glob.map(str::trim).filter(|glob| !glob.is_empty()) else {
        return Ok(None);
    };
    let glob = GlobBuilder::new(glob)
        .literal_separator(false)
        .build()
        .map_err(|error| anyhow::anyhow!("invalid ast_grep glob '{glob}': {error}"))?;
    let mut builder = globset::GlobSetBuilder::new();
    builder.add(glob);
    Ok(Some(builder.build()?))
}
