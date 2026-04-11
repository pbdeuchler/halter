// pattern: Functional Core

use std::collections::BTreeMap;
use std::sync::Arc;

use ast_grep_core::{MatchStrictness, matcher::Pattern, source::Edit, tree_sitter::LanguageExt};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use crate::PathLockMap;

use super::FileCandidate;
use super::language::{parse_strictness, resolve_language};
use crate::builtin::common::{
    atomic_write_blocking, ensure_not_cancelled, optional_bool, optional_string, optional_u64,
};

#[derive(Clone)]
pub(super) struct ReplaceConfig {
    pub path: Option<String>,
    pub glob: Option<String>,
    pub lang: Option<String>,
    pub selector: Option<String>,
    pub rewrites: Vec<(String, String)>,
    pub strictness: MatchStrictness,
    pub dry_run: bool,
    pub max_replacements: u64,
    pub max_files: u64,
    pub fail_on_parse_error: bool,
}

impl ReplaceConfig {
    pub(super) fn parse(input: &Value) -> anyhow::Result<Self> {
        let rewrites = input
            .get("rewrites")
            .and_then(Value::as_object)
            .ok_or_else(|| anyhow::anyhow!("invalid tool input: missing object field 'rewrites'"))?
            .iter()
            .map(|(pattern, replacement)| {
                let replacement = replacement.as_str().ok_or_else(|| {
                    anyhow::anyhow!(
                        "failed to execute ast_grep tool: rewrite values must be strings"
                    )
                })?;
                if pattern.trim().is_empty() {
                    anyhow::bail!(
                        "failed to execute ast_grep tool: rewrite patterns must not be empty"
                    );
                }
                Ok((pattern.clone(), replacement.to_owned()))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        if rewrites.is_empty() {
            anyhow::bail!("failed to execute ast_grep tool: rewrites must not be empty");
        }

        Ok(Self {
            path: optional_string(input, "path").map(ToOwned::to_owned),
            glob: optional_string(input, "glob").map(ToOwned::to_owned),
            lang: optional_string(input, "lang").map(ToOwned::to_owned),
            selector: optional_string(input, "selector").map(ToOwned::to_owned),
            rewrites,
            strictness: parse_strictness(optional_string(input, "strictness"))?,
            dry_run: optional_bool(input, "dry_run")?.unwrap_or(true),
            max_replacements: optional_u64(input, "max_replacements")?.unwrap_or(u64::MAX),
            max_files: optional_u64(input, "max_files")?.unwrap_or(u64::MAX),
            fail_on_parse_error: optional_bool(input, "fail_on_parse_error")?.unwrap_or(false),
        })
    }
}

#[derive(Debug, Clone)]
struct ReplaceChange {
    path: String,
    before: String,
    after: String,
    start_line: u64,
    end_line: u64,
}

pub(super) fn run(
    config: ReplaceConfig,
    candidates: Vec<FileCandidate>,
    path_locks: Arc<PathLockMap>,
    cancel: CancellationToken,
) -> anyhow::Result<Value> {
    let mut changes = Vec::new();
    let mut file_counts = BTreeMap::<String, u64>::new();
    let mut parse_errors = Vec::new();
    let mut files_touched = 0u64;
    let mut limit_reached = false;

    for candidate in &candidates {
        ensure_not_cancelled(&cancel)?;
        if files_touched >= config.max_files {
            limit_reached = true;
            break;
        }

        let language = match resolve_language(config.lang.as_deref(), &candidate.absolute_path) {
            Ok(language) => language,
            Err(error) => {
                if config.fail_on_parse_error {
                    return Err(error);
                }
                parse_errors.push(format!("{}: {error}", candidate.display_path));
                continue;
            }
        };
        let guard = if config.dry_run {
            PathGuard::Read(path_locks.acquire_read(&candidate.absolute_path)?)
        } else {
            PathGuard::Write(path_locks.acquire_write(&candidate.absolute_path)?)
        };
        let source = std::fs::read_to_string(&candidate.absolute_path)?;
        let ast = language.ast_grep(source.as_str());
        if ast.root().dfs().any(|node| node.is_error()) {
            let message = format!(
                "{}: parse error (syntax tree contains error nodes)",
                candidate.display_path
            );
            if config.fail_on_parse_error {
                drop(guard);
                anyhow::bail!("failed to execute ast_grep tool: {message}");
            }
            parse_errors.push(message);
            drop(guard);
            continue;
        }

        let mut file_edits = Vec::new();
        let mut file_changes = Vec::new();

        'rules: for (pattern, replacement) in &config.rewrites {
            let compiled = match compile_pattern(
                pattern,
                config.selector.as_deref(),
                config.strictness.clone(),
                language,
            ) {
                Ok(compiled) => compiled,
                Err(error) => {
                    if config.fail_on_parse_error {
                        drop(guard);
                        return Err(error);
                    }
                    parse_errors.push(format!("{}: {}: {error}", candidate.display_path, pattern));
                    continue;
                }
            };

            for matched in ast.root().find_all(compiled.clone()) {
                ensure_not_cancelled(&cancel)?;
                if changes.len() as u64 + file_changes.len() as u64 == config.max_replacements {
                    limit_reached = true;
                    break 'rules;
                }
                let edit = matched.replace_by(replacement.as_str());
                let start = matched.start_pos();
                let end = matched.end_pos();
                let after = String::from_utf8(edit.inserted_text.clone()).map_err(|error| {
                    anyhow::anyhow!(
                        "failed to execute ast_grep tool: replacement text is not valid utf-8: {error}"
                    )
                })?;
                file_changes.push(ReplaceChange {
                    path: candidate.display_path.clone(),
                    before: matched.text().into_owned(),
                    after,
                    start_line: start.line().saturating_add(1) as u64,
                    end_line: end.line().saturating_add(1) as u64,
                });
                file_edits.push(edit);
            }
        }

        if file_changes.is_empty() {
            drop(guard);
            continue;
        }
        files_touched = files_touched.saturating_add(1);
        file_counts.insert(candidate.display_path.clone(), file_changes.len() as u64);

        if !config.dry_run {
            let output = apply_edits(&source, &file_edits)?;
            if output != source {
                atomic_write_blocking(&candidate.absolute_path, output.as_bytes())?;
            }
        }
        drop(guard);

        changes.extend(file_changes);
        if limit_reached {
            break;
        }
    }

    Ok(json!({
        "changes": changes.iter().map(|change| json!({
            "path": change.path,
            "before": change.before,
            "after": change.after,
            "start_line": change.start_line,
            "end_line": change.end_line,
        })).collect::<Vec<_>>(),
        "file_changes": file_counts.iter().map(|(path, count)| json!({
            "path": path,
            "count": count,
        })).collect::<Vec<_>>(),
        "total_replacements": changes.len(),
        "files_touched": files_touched,
        "files_searched": candidates.len(),
        "applied": !config.dry_run,
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
    let mut compiled = if let Some(selector) = selector
        .map(str::trim)
        .filter(|selector| !selector.is_empty())
    {
        Pattern::contextual(pattern, selector, language)
    } else {
        Pattern::try_new(pattern, language)
    }
    .map_err(|error| anyhow::anyhow!("failed to compile ast_grep rewrite pattern: {error}"))?;
    compiled.strictness = strictness;
    Ok(compiled)
}

fn apply_edits(content: &str, edits: &[Edit<String>]) -> anyhow::Result<String> {
    let mut edits = edits.iter().collect::<Vec<_>>();
    edits.sort_by_key(|edit| edit.position);

    let mut previous_end = 0usize;
    for edit in &edits {
        if edit.position < previous_end {
            anyhow::bail!("failed to execute ast_grep tool: overlapping replacements detected");
        }
        previous_end = edit.position.saturating_add(edit.deleted_length);
    }

    let mut output = content.to_owned();
    for edit in edits.into_iter().rev() {
        let start = edit.position;
        let end = start.saturating_add(edit.deleted_length);
        if end > output.len() {
            anyhow::bail!("failed to execute ast_grep tool: computed edit range is out of bounds");
        }
        let replacement = String::from_utf8(edit.inserted_text.clone()).map_err(|error| {
            anyhow::anyhow!(
                "failed to execute ast_grep tool: replacement text is not valid utf-8: {error}"
            )
        })?;
        output.replace_range(start..end, &replacement);
    }

    Ok(output)
}

#[allow(dead_code)]
enum PathGuard {
    Read(crate::builtin::fs_lock::PathReadGuard),
    Write(crate::builtin::fs_lock::PathWriteGuard),
}

#[cfg(all(test, feature = "ast-tools"))]
mod tests {
    use super::*;

    #[test]
    fn rejects_overlapping_edits() {
        let error = apply_edits(
            "abcdef",
            &[
                Edit::<String> {
                    position: 1,
                    deleted_length: 3,
                    inserted_text: b"x".to_vec(),
                },
                Edit::<String> {
                    position: 2,
                    deleted_length: 1,
                    inserted_text: b"y".to_vec(),
                },
            ],
        )
        .expect_err("overlap should fail");

        assert!(error.to_string().contains("overlapping replacements"));
    }

    #[test]
    fn dry_run_reports_changes_without_writing() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("main.rs");
        std::fs::write(&path, "fn alpha() {}\n").expect("write");

        let value = run(
            ReplaceConfig {
                path: Some(temp.path().to_string_lossy().into_owned()),
                glob: None,
                lang: None,
                selector: None,
                rewrites: vec![(
                    "fn $NAME() {}".to_owned(),
                    "fn $NAME() -> i32 { 0 }".to_owned(),
                )],
                strictness: MatchStrictness::Smart,
                dry_run: true,
                max_replacements: u64::MAX,
                max_files: u64::MAX,
                fail_on_parse_error: false,
            },
            vec![FileCandidate {
                absolute_path: path.clone(),
                display_path: "main.rs".to_owned(),
            }],
            Arc::new(PathLockMap::default()),
            CancellationToken::new(),
        )
        .expect("replace should succeed");

        assert_eq!(value["total_replacements"], 1);
        assert_eq!(
            std::fs::read_to_string(path).expect("read"),
            "fn alpha() {}\n"
        );
    }

    #[test]
    fn max_replacements_allows_exact_limit() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("main.rs");
        std::fs::write(&path, "fn alpha() {}\nfn beta() {}\n").expect("write");

        let value = run(
            ReplaceConfig {
                path: Some(temp.path().to_string_lossy().into_owned()),
                glob: None,
                lang: None,
                selector: None,
                rewrites: vec![(
                    "fn $NAME() {}".to_owned(),
                    "fn $NAME() -> i32 { 0 }".to_owned(),
                )],
                strictness: MatchStrictness::Smart,
                dry_run: true,
                max_replacements: 2,
                max_files: u64::MAX,
                fail_on_parse_error: false,
            },
            vec![FileCandidate {
                absolute_path: path,
                display_path: "main.rs".to_owned(),
            }],
            Arc::new(PathLockMap::default()),
            CancellationToken::new(),
        )
        .expect("replace should succeed");

        assert_eq!(value["total_replacements"], 2);
    }
}
