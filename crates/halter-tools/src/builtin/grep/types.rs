// pattern: Functional Core

use std::borrow::Cow;
use std::path::{Path, PathBuf};

use globset::{GlobBuilder, GlobSet};
use ignore::WalkBuilder;
use regex::{Regex, RegexBuilder};
use serde_json::{Value, json};

use super::super::common::resolve_path;
use super::super::text::{truncate_to_width, visible_width};

pub const DEFAULT_MAX_MATCHES: u64 = 100;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputMode {
    Content,
    Count,
    FilesWithMatches,
}

impl OutputMode {
    #[must_use]
    pub fn from_str(value: Option<&str>) -> Self {
        match value {
            Some("count") => Self::Count,
            Some("files_with_matches") => Self::FilesWithMatches,
            _ => Self::Content,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchConfig {
    pub pattern: String,
    pub path: String,
    pub glob: Option<String>,
    pub type_filter: Option<String>,
    pub ignore_case: bool,
    pub multiline: bool,
    pub context_before: usize,
    pub context_after: usize,
    pub max_matches: u64,
    pub offset: u64,
    pub max_columns: Option<usize>,
    pub output_mode: OutputMode,
}

#[derive(Debug, Clone)]
pub struct FileEntry {
    pub absolute_path: PathBuf,
    pub display_path: String,
}

#[derive(Debug, Clone)]
pub struct ContextLine {
    pub line_number: u64,
    pub line: String,
}

#[derive(Debug, Clone)]
pub struct MatchRecord {
    pub line_number: u64,
    pub line: String,
    pub context_before: Vec<ContextLine>,
    pub context_after: Vec<ContextLine>,
    pub truncated: bool,
}

#[derive(Debug, Clone)]
pub struct FileSearchResult {
    pub path: String,
    pub matches: Vec<MatchRecord>,
    pub total_matches: u64,
}

#[derive(Debug, Clone)]
pub struct AggregateResult {
    pub files: Vec<FileSearchResult>,
    pub files_searched: u64,
    pub files_with_matches: u64,
    pub total_matches: u64,
}

#[derive(Debug, Clone)]
struct IndexedLine {
    start: usize,
    end_content: usize,
}

pub fn resolve_search_root(working_dir: &Path, path: &str) -> PathBuf {
    resolve_path(working_dir, path)
}

pub fn collect_entries(
    search_root: &Path,
    glob: Option<&str>,
    type_filter: Option<&str>,
) -> anyhow::Result<Vec<FileEntry>> {
    let metadata = std::fs::metadata(search_root)?;
    if metadata.is_file() {
        if !matches_type_filter(search_root, type_filter) {
            return Ok(Vec::new());
        }
        let display_path = search_root
            .file_name()
            .and_then(|name| name.to_str())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| search_root.to_string_lossy().into_owned());
        return Ok(vec![FileEntry {
            absolute_path: search_root.to_path_buf(),
            display_path,
        }]);
    }

    if !metadata.is_dir() {
        return Ok(Vec::new());
    }

    let matcher = compile_glob(glob)?;
    let mut entries = Vec::new();
    let mut builder = WalkBuilder::new(search_root);
    builder
        .standard_filters(true)
        .require_git(false)
        .follow_links(false)
        .sort_by_file_path(|left, right| left.cmp(right));

    for entry in builder.build() {
        let entry = entry?;
        let path = entry.path();
        if path == search_root || !entry.file_type().is_some_and(|kind| kind.is_file()) {
            continue;
        }
        let relative = path.strip_prefix(search_root).unwrap_or(path);
        if let Some(matcher) = matcher.as_ref() {
            if !matcher.is_match(relative) {
                continue;
            }
        }
        if !matches_type_filter(path, type_filter) {
            continue;
        }
        entries.push(FileEntry {
            absolute_path: path.to_path_buf(),
            display_path: relative.to_string_lossy().into_owned(),
        });
    }

    Ok(entries)
}

pub fn build_matcher(pattern: &str, ignore_case: bool, multiline: bool) -> anyhow::Result<Regex> {
    let sanitized = sanitize_braces(pattern);
    build_regex(sanitized.as_ref(), ignore_case, multiline).or_else(|error| {
        if error.to_string().contains("unclosed group")
            || error.to_string().contains("unopened group")
        {
            let escaped = escape_unescaped_parentheses(sanitized.as_ref());
            if escaped.as_ref() != sanitized.as_ref() {
                return build_regex(escaped.as_ref(), ignore_case, multiline);
            }
        }
        Err(error)
    })
}

pub fn build_response(result: AggregateResult, config: &SearchConfig) -> Value {
    match config.output_mode {
        OutputMode::Content => build_content_response(result, config),
        OutputMode::Count => build_count_response(result),
        OutputMode::FilesWithMatches => build_files_with_matches_response(result),
    }
}

pub fn search_text(text: &str, matcher: &Regex, config: &SearchConfig) -> FileSearchResult {
    let indexed_lines = index_lines(text);
    let mut matches = Vec::new();
    let mut total_matches = 0u64;

    for matched in matcher.find_iter(text) {
        total_matches += 1;
        if config.output_mode != OutputMode::Content {
            continue;
        }

        if indexed_lines.is_empty() {
            continue;
        }

        let start_line_index = line_index_for_position(&indexed_lines, matched.start());
        let end_position = matched.end().saturating_sub(1);
        let end_line_index = line_index_for_position(&indexed_lines, end_position);
        let line_number = start_line_index as u64 + 1;
        let line = render_line(
            &text[indexed_lines[start_line_index].start..indexed_lines[end_line_index].end_content],
            config.max_columns,
        );
        let context_before = render_context_before(
            text,
            &indexed_lines,
            start_line_index,
            config.context_before,
            config.max_columns,
        );
        let context_after = render_context_after(
            text,
            &indexed_lines,
            end_line_index,
            config.context_after,
            config.max_columns,
        );

        matches.push(MatchRecord {
            line_number,
            truncated: line.1,
            line: line.0,
            context_before,
            context_after,
        });
    }

    FileSearchResult {
        path: String::new(),
        matches,
        total_matches,
    }
}

pub fn decode_searchable_bytes(bytes: &[u8]) -> Option<String> {
    if bytes.contains(&0) {
        return None;
    }
    Some(String::from_utf8_lossy(bytes).into_owned())
}

fn build_content_response(result: AggregateResult, config: &SearchConfig) -> Value {
    let mut skipped = 0u64;
    let mut emitted = 0u64;
    let mut matches = Vec::new();

    'files: for file in &result.files {
        for matched in &file.matches {
            if skipped < config.offset {
                skipped += 1;
                continue;
            }
            if emitted >= config.max_matches {
                break 'files;
            }
            matches.push(json!({
                "path": file.path,
                "line_number": matched.line_number,
                "line": matched.line,
                "context_before": if matched.context_before.is_empty() { Value::Null } else { json!(matched.context_before.iter().map(|line| json!({"line_number": line.line_number, "line": line.line})).collect::<Vec<_>>()) },
                "context_after": if matched.context_after.is_empty() { Value::Null } else { json!(matched.context_after.iter().map(|line| json!({"line_number": line.line_number, "line": line.line})).collect::<Vec<_>>()) },
                "truncated": matched.truncated,
            }));
            emitted += 1;
        }
    }

    json!({
        "matches": matches,
        "total_matches": result.total_matches,
        "files_searched": result.files_searched,
        "files_with_matches": result.files_with_matches,
        "truncated": result.total_matches > config.offset.saturating_add(emitted),
    })
}

fn build_count_response(result: AggregateResult) -> Value {
    json!({
        "matches": result.files.iter().map(|file| json!({
            "path": file.path,
            "line_number": 0,
            "line": "",
            "match_count": file.total_matches,
        })).collect::<Vec<_>>(),
        "total_matches": result.total_matches,
        "files_searched": result.files_searched,
        "files_with_matches": result.files_with_matches,
        "truncated": false,
    })
}

fn build_files_with_matches_response(result: AggregateResult) -> Value {
    json!({
        "matches": result.files.iter().map(|file| json!({
            "path": file.path,
            "line_number": 0,
            "line": "",
        })).collect::<Vec<_>>(),
        "total_matches": result.total_matches,
        "files_searched": result.files_searched,
        "files_with_matches": result.files_with_matches,
        "truncated": false,
    })
}

fn build_regex(pattern: &str, ignore_case: bool, multiline: bool) -> anyhow::Result<Regex> {
    RegexBuilder::new(pattern)
        .case_insensitive(ignore_case)
        .multi_line(multiline)
        .dot_matches_new_line(multiline)
        .size_limit(10 * 1024 * 1024)
        .dfa_size_limit(10 * 1024 * 1024)
        .build()
        .map_err(|error| anyhow::anyhow!("failed to compile grep pattern: {error}"))
}

fn compile_glob(glob: Option<&str>) -> anyhow::Result<Option<GlobSet>> {
    let Some(glob) = glob else {
        return Ok(None);
    };
    let pattern = GlobBuilder::new(glob)
        .literal_separator(false)
        .build()
        .map_err(|error| anyhow::anyhow!("invalid grep glob '{glob}': {error}"))?;
    let mut builder = globset::GlobSetBuilder::new();
    builder.add(pattern);
    Ok(Some(builder.build()?))
}

fn matches_type_filter(path: &Path, type_filter: Option<&str>) -> bool {
    let Some(type_filter) = type_filter.map(str::trim).filter(|value| !value.is_empty()) else {
        return true;
    };
    let normalized = type_filter.trim_start_matches('.').to_ascii_lowercase();
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();
    let extension = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    match normalized.as_str() {
        "js" | "javascript" => matches!(extension.as_str(), "js" | "jsx" | "mjs" | "cjs"),
        "ts" | "typescript" => matches!(extension.as_str(), "ts" | "tsx" | "mts" | "cts"),
        "json" => matches!(extension.as_str(), "json" | "jsonc" | "json5"),
        "yaml" | "yml" => matches!(extension.as_str(), "yaml" | "yml"),
        "md" | "markdown" => matches!(extension.as_str(), "md" | "markdown" | "mdx"),
        "py" | "python" => matches!(extension.as_str(), "py" | "pyi"),
        "rs" | "rust" => extension == "rs",
        "go" => extension == "go",
        "java" => extension == "java",
        "sh" | "bash" => matches!(extension.as_str(), "sh" | "bash" | "zsh"),
        "docker" | "dockerfile" => file_name == "dockerfile",
        "make" | "makefile" => file_name == "makefile",
        other => extension == other || file_name == other,
    }
}

fn index_lines(text: &str) -> Vec<IndexedLine> {
    if text.is_empty() {
        return Vec::new();
    }

    let mut lines = Vec::new();
    let mut start = 0usize;
    for (index, byte) in text.bytes().enumerate() {
        if byte == b'\n' {
            let end_content = if index > start && text.as_bytes()[index - 1] == b'\r' {
                index - 1
            } else {
                index
            };
            lines.push(IndexedLine { start, end_content });
            start = index + 1;
        }
    }
    if start < text.len() || lines.is_empty() {
        lines.push(IndexedLine {
            start,
            end_content: text.len(),
        });
    }
    lines
}

fn line_index_for_position(lines: &[IndexedLine], position: usize) -> usize {
    match lines.binary_search_by_key(&position, |line| line.start) {
        Ok(index) => index,
        Err(index) => index.saturating_sub(1),
    }
}

fn render_line(line: &str, max_columns: Option<usize>) -> (String, bool) {
    let trimmed = line.trim_end_matches(['\r', '\n']);
    match max_columns {
        Some(max_columns) if visible_width(trimmed) > max_columns => {
            (truncate_to_width(trimmed, max_columns), true)
        }
        _ => (trimmed.to_owned(), false),
    }
}

fn render_context_before(
    text: &str,
    lines: &[IndexedLine],
    start_line_index: usize,
    count: usize,
    max_columns: Option<usize>,
) -> Vec<ContextLine> {
    let start = start_line_index.saturating_sub(count);
    (start..start_line_index)
        .map(|index| {
            let line = &text[lines[index].start..lines[index].end_content];
            let (line, _) = render_line(line, max_columns);
            ContextLine {
                line_number: index as u64 + 1,
                line,
            }
        })
        .collect()
}

fn render_context_after(
    text: &str,
    lines: &[IndexedLine],
    end_line_index: usize,
    count: usize,
    max_columns: Option<usize>,
) -> Vec<ContextLine> {
    let end = (end_line_index + count + 1).min(lines.len());
    ((end_line_index + 1)..end)
        .map(|index| {
            let line = &text[lines[index].start..lines[index].end_content];
            let (line, _) = render_line(line, max_columns);
            ContextLine {
                line_number: index as u64 + 1,
                line,
            }
        })
        .collect()
}

fn sanitize_braces(pattern: &str) -> Cow<'_, str> {
    let bytes = pattern.as_bytes();
    if !bytes.contains(&b'{') && !bytes.contains(&b'}') {
        return Cow::Borrowed(pattern);
    }

    let mut result = String::with_capacity(pattern.len() + 8);
    let mut modified = false;
    let mut index = 0usize;

    while index < bytes.len() {
        if bytes[index] == b'\\' && index + 1 < bytes.len() {
            result.push('\\');
            index += 1;
            let character = pattern[index..].chars().next().expect("character");
            result.push(character);
            index += character.len_utf8();
            continue;
        }

        if bytes[index] == b'{' {
            if let Some(end) = valid_repetition_end(bytes, index) {
                result.push_str(&pattern[index..=end]);
                index = end + 1;
                continue;
            }
            result.push_str("\\{");
            modified = true;
            index += 1;
            continue;
        }

        if bytes[index] == b'}' {
            result.push_str("\\}");
            modified = true;
            index += 1;
            continue;
        }

        let character = pattern[index..].chars().next().expect("character");
        result.push(character);
        index += character.len_utf8();
    }

    if modified {
        Cow::Owned(result)
    } else {
        Cow::Borrowed(pattern)
    }
}

fn valid_repetition_end(bytes: &[u8], start: usize) -> Option<usize> {
    let mut index = start + 1;
    if index >= bytes.len() || !bytes[index].is_ascii_digit() {
        return None;
    }
    while index < bytes.len() && bytes[index].is_ascii_digit() {
        index += 1;
    }
    if index >= bytes.len() {
        return None;
    }
    if bytes[index] == b'}' {
        return Some(index);
    }
    if bytes[index] != b',' {
        return None;
    }
    index += 1;
    while index < bytes.len() && bytes[index].is_ascii_digit() {
        index += 1;
    }
    if index < bytes.len() && bytes[index] == b'}' {
        Some(index)
    } else {
        None
    }
}

fn escape_unescaped_parentheses(pattern: &str) -> Cow<'_, str> {
    let bytes = pattern.as_bytes();
    if !bytes.contains(&b'(') && !bytes.contains(&b')') {
        return Cow::Borrowed(pattern);
    }

    let mut result = String::with_capacity(pattern.len() + 4);
    let mut modified = false;
    let mut index = 0usize;
    while index < bytes.len() {
        if bytes[index] == b'\\' && index + 1 < bytes.len() {
            result.push('\\');
            index += 1;
            let character = pattern[index..].chars().next().expect("character");
            result.push(character);
            index += character.len_utf8();
            continue;
        }

        let character = pattern[index..].chars().next().expect("character");
        if matches!(character, '(' | ')') {
            result.push('\\');
            modified = true;
        }
        result.push(character);
        index += character.len_utf8();
    }

    if modified {
        Cow::Owned(result)
    } else {
        Cow::Borrowed(pattern)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_braces_escapes_non_quantifiers() {
        assert_eq!(sanitize_braces("${platform}").as_ref(), "$\\{platform\\}");
    }

    #[test]
    fn escape_unescaped_parentheses_handles_literal_suffix() {
        assert_eq!(
            escape_unescaped_parentheses("fetchAnthropicProvider(").as_ref(),
            r"fetchAnthropicProvider\("
        );
    }

    #[test]
    fn search_text_returns_context() {
        let config = SearchConfig {
            pattern: "beta".to_owned(),
            path: ".".to_owned(),
            glob: None,
            type_filter: None,
            ignore_case: false,
            multiline: false,
            context_before: 1,
            context_after: 1,
            max_matches: DEFAULT_MAX_MATCHES,
            offset: 0,
            max_columns: None,
            output_mode: OutputMode::Content,
        };
        let matcher = build_matcher("beta", false, false).expect("matcher");
        let result = search_text("alpha\nbeta\ngamma\n", &matcher, &config);
        assert_eq!(result.total_matches, 1);
        assert_eq!(result.matches[0].context_before[0].line, "alpha");
        assert_eq!(result.matches[0].context_after[0].line, "gamma");
    }
}
