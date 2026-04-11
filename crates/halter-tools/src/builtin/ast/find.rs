// pattern: Functional Core

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use ast_grep_core::{MatchStrictness, matcher::Pattern, tree_sitter::LanguageExt};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::PathLockMap;

use super::language::{canonical_name, parse_strictness, resolve_language};
use super::collect_candidates;
use crate::builtin::common::{ensure_not_cancelled, optional_bool, optional_string, optional_u64};

const DEFAULT_FIND_LIMIT: u64 = 50;

#[derive(Clone)]
pub(super) struct FindConfig {
    pub path: Option<String>,
    pub glob: Option<String>,
    pub lang: Option<String>,
    pub selector: Option<String>,
    pub patterns: Vec<String>,
    pub strictness: MatchStrictness,
    pub limit: u64,
    pub offset: u64,
    pub include_meta: bool,
}

impl FindConfig {
    pub(super) fn parse(input: &Value) -> anyhow::Result<Self> {
        let patterns = input
            .get("patterns")
            .and_then(Value::as_array)
            .ok_or_else(|| {
                anyhow::anyhow!("invalid tool input: missing array field 'patterns'")
            })?
            .iter()
            .map(|value| {
                value
                    .as_str()
                    .map(str::trim)
                    .filter(|pattern| !pattern.is_empty())
                    .map(ToOwned::to_owned)
                    .ok_or_else(|| {
                        anyhow::anyhow!(
                            "failed to execute ast_grep tool: patterns must contain non-empty strings"
                        )
                    })
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        if patterns.is_empty() {
            anyhow::bail!("failed to execute ast_grep tool: patterns must not be empty");
        }

        Ok(Self {
            path: optional_string(input, "path").map(ToOwned::to_owned),
            glob: optional_string(input, "glob").map(ToOwned::to_owned),
            lang: optional_string(input, "lang").map(ToOwned::to_owned),
            selector: optional_string(input, "selector").map(ToOwned::to_owned),
            patterns,
            strictness: parse_strictness(optional_string(input, "strictness"))?,
            limit: optional_u64(input, "limit")?.unwrap_or(DEFAULT_FIND_LIMIT),
            offset: optional_u64(input, "offset")?.unwrap_or(0),
            include_meta: optional_bool(input, "include_meta")?.unwrap_or(false),
        })
    }
}

#[derive(Debug, Clone)]
struct FindMatch {
    path: String,
    text: String,
    byte_start: u64,
    byte_end: u64,
    start_line: u64,
    start_column: u64,
    end_line: u64,
    end_column: u64,
    meta_variables: Option<HashMap<String, String>>,
}

pub(super) fn run(
    config: FindConfig,
    working_dir: std::path::PathBuf,
    path_locks: Arc<PathLockMap>,
    cancel: CancellationToken,
) -> anyhow::Result<Value> {
    let candidates = collect_candidates(
        &working_dir,
        config.path.as_deref(),
        config.glob.as_deref(),
        config.lang.as_deref(),
    )?;

    let mut matches = Vec::new();
    let mut files_with_matches = BTreeSet::new();
    let mut parse_errors = Vec::new();
    let mut total_matches = 0u64;

    for candidate in &candidates {
        ensure_not_cancelled(&cancel)?;
        let language = match resolve_language(config.lang.as_deref(), &candidate.absolute_path) {
            Ok(language) => language,
            Err(error) => {
                parse_errors.push(format!("{}: {error}", candidate.display_path));
                continue;
            }
        };
        let text = {
            let _lock = path_locks.acquire_read(&candidate.absolute_path)?;
            std::fs::read_to_string(&candidate.absolute_path)?
        };
        let ast = language.ast_grep(text.as_str());
        if ast.root().dfs().any(|node| node.is_error()) {
            parse_errors.push(format!(
                "{}: parse error (syntax tree contains error nodes)",
                candidate.display_path
            ));
        }

        for pattern in &config.patterns {
            ensure_not_cancelled(&cancel)?;
            let compiled = match compile_pattern(
                pattern,
                config.selector.as_deref(),
                config.strictness.clone(),
                language,
            ) {
                Ok(compiled) => compiled,
                Err(error) => {
                    parse_errors.push(format!(
                        "{}: {}: {error}",
                        candidate.display_path, pattern
                    ));
                    continue;
                }
            };

            for matched in ast.root().find_all(compiled.clone()) {
                ensure_not_cancelled(&cancel)?;
                let range = matched.range();
                let start = matched.start_pos();
                let end = matched.end_pos();
                total_matches = total_matches.saturating_add(1);
                matches.push(FindMatch {
                    path: candidate.display_path.clone(),
                    text: matched.text().into_owned(),
                    byte_start: range.start as u64,
                    byte_end: range.end as u64,
                    start_line: start.line().saturating_add(1) as u64,
                    start_column: start.column(matched.get_node()).saturating_add(1) as u64,
                    end_line: end.line().saturating_add(1) as u64,
                    end_column: end.column(matched.get_node()).saturating_add(1) as u64,
                    meta_variables: config
                        .include_meta
                        .then(|| HashMap::<String, String>::from(matched.get_env().clone())),
                });
                files_with_matches.insert(candidate.display_path.clone());
            }
        }
    }

    matches.sort_by(|left, right| {
        left.path
            .cmp(&right.path)
            .then(left.start_line.cmp(&right.start_line))
            .then(left.start_column.cmp(&right.start_column))
            .then(left.byte_start.cmp(&right.byte_start))
            .then(left.byte_end.cmp(&right.byte_end))
    });

    let offset = config.offset as usize;
    let limit = config.limit as usize;
    let limit_reached = matches.len().saturating_sub(offset) > limit;
    let visible = matches.into_iter().skip(offset).take(limit).collect::<Vec<_>>();

    Ok(json!({
        "matches": visible.iter().map(|matched| json!({
            "path": matched.path,
            "text": matched.text,
            "byte_start": matched.byte_start,
            "byte_end": matched.byte_end,
            "start_line": matched.start_line,
            "start_column": matched.start_column,
            "end_line": matched.end_line,
            "end_column": matched.end_column,
            "meta_variables": matched.meta_variables,
        })).collect::<Vec<_>>(),
        "total_matches": total_matches,
        "files_searched": candidates.len(),
        "files_with_matches": files_with_matches.len(),
        "limit_reached": limit_reached,
        "parse_errors": (!parse_errors.is_empty()).then_some(parse_errors),
    }))
}

fn compile_pattern(
    pattern: &str,
    selector: Option<&str>,
    strictness: MatchStrictness,
    language: ast_grep_language::SupportLang,
) -> anyhow::Result<Pattern> {
    let mut compiled = if let Some(selector) = selector.map(str::trim).filter(|selector| !selector.is_empty()) {
        Pattern::contextual(pattern, selector, language)
    } else {
        Pattern::try_new(pattern, language)
    }
    .map_err(|error| {
        anyhow::anyhow!(
            "failed to compile ast_grep pattern for {}: {error}",
            canonical_name(language)
        )
    })?;
    compiled.strictness = strictness;
    Ok(compiled)
}

#[cfg(all(test, feature = "ast-tools"))]
mod tests {
    use super::*;

    #[test]
    fn finds_rust_function_matches() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("main.rs"), "fn alpha() {}\nfn beta() {}\n").expect("write");

        let value = run(
            FindConfig {
                path: Some(temp.path().to_string_lossy().into_owned()),
                glob: None,
                lang: None,
                selector: None,
                patterns: vec!["fn $NAME() {}".to_owned()],
                strictness: MatchStrictness::Smart,
                limit: 50,
                offset: 0,
                include_meta: true,
            },
            temp.path().to_path_buf(),
            Arc::new(PathLockMap::default()),
            CancellationToken::new(),
        )
        .expect("find should succeed");

        assert_eq!(value["total_matches"], 2);
        assert_eq!(value["files_with_matches"], 1);
        assert_eq!(value["matches"][0]["meta_variables"]["NAME"], "alpha");
    }
}
