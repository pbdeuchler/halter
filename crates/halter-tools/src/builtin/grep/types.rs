// pattern: Functional Core

use std::borrow::Cow;
use std::fs::File;
use std::io;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use globset::{GlobBuilder, GlobSet};
use grep_regex::{RegexMatcher, RegexMatcherBuilder};
use grep_searcher::{Searcher, SearcherBuilder, Sink, SinkMatch};
use ignore::WalkBuilder;
use serde_json::{Value, json};
use smallvec::SmallVec;
use tokio_util::sync::CancellationToken;

use crate::PathLockMap;

use super::super::common::resolve_path;
use super::super::text::{truncate_to_width, visible_width};

pub const DEFAULT_MAX_MATCHES: u64 = 100;
const KNOWN_TEXT_BINARY_PROBE_BYTES: usize = 512;

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
pub struct ContextLine {
    pub line_number: u64,
    pub line: String,
}

#[derive(Debug, Clone)]
struct CollectedMatch {
    line_number: u64,
    line: String,
    context_before: SmallVec<[ContextLine; 8]>,
    context_after: SmallVec<[ContextLine; 8]>,
    truncated: bool,
}

#[derive(Debug, Clone)]
struct SearchResultInternal {
    matches: Vec<CollectedMatch>,
    match_count: u64,
    collected: u64,
    limit_reached: bool,
}

#[derive(Debug, Clone)]
struct FileSearchResult {
    path: String,
    matches: Vec<CollectedMatch>,
    total_matches: u64,
}

#[derive(Debug, Clone)]
struct AggregateResult {
    files: Vec<FileSearchResult>,
    files_searched: u64,
    files_with_matches: u64,
    total_matches: u64,
    limit_reached: bool,
}

#[derive(Debug, Clone)]
struct FileEntry {
    absolute_path: PathBuf,
    display_path: String,
    prefer_text_fast_path: bool,
}

#[derive(Clone, Copy)]
struct SearchParams {
    context_before: usize,
    context_after: usize,
    max_columns: Option<usize>,
    mode: OutputMode,
    max_matches: Option<u64>,
    offset: u64,
    multiline: bool,
}

enum TypeFilter {
    Known {
        exts: &'static [&'static str],
        names: &'static [&'static str],
    },
    Custom(String),
}

impl TypeFilter {
    fn match_ext(&self, ext: &str) -> bool {
        match self {
            Self::Known { exts, .. } => exts
                .iter()
                .any(|candidate| ext.eq_ignore_ascii_case(candidate)),
            Self::Custom(custom_ext) => ext.eq_ignore_ascii_case(custom_ext),
        }
    }

    fn match_name(&self, name: &str) -> bool {
        match self {
            Self::Known { names, .. } => names
                .iter()
                .any(|candidate| name.eq_ignore_ascii_case(candidate)),
            Self::Custom(custom) => name.eq_ignore_ascii_case(custom),
        }
    }
}

struct MatchCollector {
    matches: Vec<CollectedMatch>,
    match_count: u64,
    collected_count: u64,
    max_matches: Option<u64>,
    offset: u64,
    skipped: u64,
    limit_reached: bool,
    max_columns: Option<usize>,
    collect_matches: bool,
    before_count: usize,
    after_count: usize,
}

impl MatchCollector {
    const fn new(
        max_matches: Option<u64>,
        offset: u64,
        max_columns: Option<usize>,
        collect_matches: bool,
        before_count: usize,
        after_count: usize,
    ) -> Self {
        Self {
            matches: Vec::new(),
            match_count: 0,
            collected_count: 0,
            max_matches,
            offset,
            skipped: 0,
            limit_reached: false,
            max_columns,
            collect_matches,
            before_count,
            after_count,
        }
    }
}

impl Sink for MatchCollector {
    type Error = io::Error;

    fn matched(
        &mut self,
        _searcher: &Searcher,
        matched: &SinkMatch<'_>,
    ) -> Result<bool, Self::Error> {
        self.match_count = self.match_count.saturating_add(1);

        if self.limit_reached {
            return Ok(false);
        }

        if self.skipped < self.offset {
            self.skipped = self.skipped.saturating_add(1);
            return Ok(true);
        }

        if self.collect_matches {
            let raw_line = bytes_to_trimmed_string(matched.bytes());
            let (line, truncated) = truncate_line(&raw_line, self.max_columns);
            let line_number = matched.line_number().unwrap_or(0);

            let (context_before, context_after) = if self.before_count > 0 || self.after_count > 0 {
                extract_context_lines(
                    matched.buffer(),
                    matched.bytes_range_in_buffer(),
                    self.before_count,
                    self.after_count,
                    line_number,
                    self.max_columns,
                )
            } else {
                (SmallVec::new(), SmallVec::new())
            };

            self.matches.push(CollectedMatch {
                line_number,
                line,
                context_before,
                context_after,
                truncated,
            });
        }

        self.collected_count = self.collected_count.saturating_add(1);
        if let Some(max_matches) = self.max_matches
            && self.collected_count >= max_matches
        {
            self.limit_reached = true;
        }

        Ok(true)
    }
}

enum FileBytes {
    #[cfg(feature = "advanced-tools")]
    Mapped(memmap2::Mmap),
    Owned(Vec<u8>),
}

impl FileBytes {
    fn as_slice(&self) -> &[u8] {
        match self {
            #[cfg(feature = "advanced-tools")]
            Self::Mapped(mapped) => mapped.as_ref(),
            Self::Owned(bytes) => bytes.as_slice(),
        }
    }
}

#[cfg(not(feature = "advanced-tools"))]
pub fn run_basic_search(
    working_dir: PathBuf,
    path_locks: Arc<PathLockMap>,
    cancel: CancellationToken,
    config: SearchConfig,
) -> anyhow::Result<Value> {
    execute_search(working_dir, path_locks, cancel, config, false)
}

#[cfg(feature = "advanced-tools")]
pub fn run_advanced_search(
    working_dir: PathBuf,
    path_locks: Arc<PathLockMap>,
    cancel: CancellationToken,
    config: SearchConfig,
) -> anyhow::Result<Value> {
    execute_search(working_dir, path_locks, cancel, config, true)
}

fn execute_search(
    working_dir: PathBuf,
    path_locks: Arc<PathLockMap>,
    cancel: CancellationToken,
    config: SearchConfig,
    allow_parallel: bool,
) -> anyhow::Result<Value> {
    let search_root = resolve_search_root(&working_dir, &config.path);
    let entries = collect_entries(
        &search_root,
        config.glob.as_deref(),
        config.type_filter.as_deref(),
    )?;
    let matcher = build_matcher(&config.pattern, config.ignore_case, config.multiline)?;

    let result =
        if allow_parallel && config.offset == 0 && config.output_mode == OutputMode::Content {
            run_parallel_search(&entries, &matcher, &config, path_locks, cancel)?
        } else {
            run_sequential_search(&entries, &matcher, &config, path_locks, cancel)?
        };

    Ok(build_response(result, &config))
}

fn run_parallel_search(
    entries: &[FileEntry],
    matcher: &RegexMatcher,
    config: &SearchConfig,
    path_locks: Arc<PathLockMap>,
    cancel: CancellationToken,
) -> anyhow::Result<AggregateResult> {
    #[cfg(feature = "advanced-tools")]
    {
        use rayon::prelude::*;

        #[derive(Default)]
        struct ParallelFileResult {
            file: Option<FileSearchResult>,
            searched: bool,
        }

        let params = SearchParams {
            context_before: config.context_before,
            context_after: config.context_after,
            max_columns: config.max_columns,
            mode: config.output_mode,
            max_matches: None,
            offset: 0,
            multiline: config.multiline,
        };

        let mut results: Vec<ParallelFileResult> = entries
            .par_iter()
            .map_init(
                || build_searcher(params.multiline),
                |searcher, entry| {
                    if cancel.is_cancelled() {
                        return ParallelFileResult::default();
                    }

                    let Ok(_lock) = path_locks.acquire_read(&entry.absolute_path) else {
                        return ParallelFileResult::default();
                    };
                    let Ok(Some(bytes)) =
                        read_file_bytes(&entry.absolute_path, entry.prefer_text_fast_path)
                    else {
                        return ParallelFileResult::default();
                    };
                    let Ok(search) = run_search(searcher, matcher, bytes.as_slice(), params) else {
                        return ParallelFileResult::default();
                    };

                    ParallelFileResult {
                        file: Some(FileSearchResult {
                            path: entry.display_path.clone(),
                            matches: if config.output_mode == OutputMode::Content {
                                search.matches
                            } else {
                                Vec::new()
                            },
                            total_matches: search.match_count,
                        }),
                        searched: true,
                    }
                },
            )
            .collect();

        if cancel.is_cancelled() {
            anyhow::bail!("failed to execute grep tool: cancelled");
        }

        results.sort_by(|left, right| {
            let left_path = left
                .file
                .as_ref()
                .map(|file| file.path.as_str())
                .unwrap_or("");
            let right_path = right
                .file
                .as_ref()
                .map(|file| file.path.as_str())
                .unwrap_or("");
            left_path.cmp(right_path)
        });

        let files_searched = results.iter().filter(|result| result.searched).count() as u64;
        let mut files = Vec::new();
        let mut files_with_matches = 0u64;
        let mut total_matches = 0u64;

        for result in results {
            let Some(file) = result.file else {
                continue;
            };
            if file.total_matches == 0 {
                continue;
            }
            files_with_matches = files_with_matches.saturating_add(1);
            total_matches = total_matches.saturating_add(file.total_matches);
            files.push(file);
        }

        Ok(AggregateResult {
            files,
            files_searched,
            files_with_matches,
            total_matches,
            limit_reached: false,
        })
    }

    #[cfg(not(feature = "advanced-tools"))]
    {
        let _ = (entries, matcher, config, path_locks, cancel);
        unreachable!("advanced search is only available with the advanced-tools feature");
    }
}

fn run_sequential_search(
    entries: &[FileEntry],
    matcher: &RegexMatcher,
    config: &SearchConfig,
    path_locks: Arc<PathLockMap>,
    cancel: CancellationToken,
) -> anyhow::Result<AggregateResult> {
    let mut files = Vec::new();
    let mut files_searched = 0u64;
    let mut files_with_matches = 0u64;
    let mut total_matches = 0u64;
    let mut collected = 0u64;
    let mut limit_reached = false;
    let mut searcher = build_searcher(config.multiline);

    let base_params = SearchParams {
        context_before: config.context_before,
        context_after: config.context_after,
        max_columns: config.max_columns,
        mode: config.output_mode,
        max_matches: Some(config.max_matches),
        offset: config.offset,
        multiline: config.multiline,
    };

    for entry in entries {
        if cancel.is_cancelled() {
            anyhow::bail!("failed to execute grep tool: cancelled");
        }
        if base_params.max_matches == Some(0) || collected >= config.max_matches {
            limit_reached = true;
            break;
        }

        let file_offset = config.offset.saturating_sub(total_matches);
        let remaining = config.max_matches.saturating_sub(collected);
        if remaining == 0 {
            limit_reached = true;
            break;
        }

        let _lock = path_locks.acquire_read(&entry.absolute_path)?;
        let Some(bytes) = read_file_bytes(&entry.absolute_path, entry.prefer_text_fast_path)?
        else {
            continue;
        };
        files_searched = files_searched.saturating_add(1);

        let params = SearchParams {
            max_matches: Some(remaining),
            offset: file_offset,
            ..base_params
        };
        let search = run_search(&mut searcher, matcher, bytes.as_slice(), params)?;
        if search.match_count == 0 {
            continue;
        }

        files_with_matches = files_with_matches.saturating_add(1);
        total_matches = total_matches.saturating_add(search.match_count);
        collected = collected.saturating_add(search.collected);
        files.push(FileSearchResult {
            path: entry.display_path.clone(),
            matches: if config.output_mode == OutputMode::Content {
                search.matches
            } else {
                Vec::new()
            },
            total_matches: search.match_count,
        });

        if search.limit_reached || collected >= config.max_matches {
            limit_reached = true;
            break;
        }
    }

    Ok(AggregateResult {
        files,
        files_searched,
        files_with_matches,
        total_matches,
        limit_reached,
    })
}

fn run_search(
    searcher: &mut Searcher,
    matcher: &RegexMatcher,
    content: &[u8],
    params: SearchParams,
) -> io::Result<SearchResultInternal> {
    let collect_matches = matches!(params.mode, OutputMode::Content);
    let (before_count, after_count) = if collect_matches {
        (params.context_before, params.context_after)
    } else {
        (0, 0)
    };

    let mut collector = MatchCollector::new(
        params.max_matches,
        params.offset,
        params.max_columns,
        collect_matches,
        before_count,
        after_count,
    );
    searcher.search_slice(matcher, content, &mut collector)?;

    Ok(SearchResultInternal {
        matches: collector.matches,
        match_count: collector.match_count,
        collected: collector.collected_count,
        limit_reached: collector.limit_reached,
    })
}

fn build_searcher(multiline: bool) -> Searcher {
    SearcherBuilder::new()
        .line_number(true)
        .multi_line(multiline)
        .build()
}

fn build_regex_matcher(
    pattern: &str,
    ignore_case: bool,
    multiline: bool,
) -> Result<RegexMatcher, grep_regex::Error> {
    RegexMatcherBuilder::new()
        .case_insensitive(ignore_case)
        .multi_line(multiline)
        .build(pattern)
}

pub fn build_matcher(
    pattern: &str,
    ignore_case: bool,
    multiline: bool,
) -> anyhow::Result<RegexMatcher> {
    let sanitized = sanitize_braces(pattern);
    build_regex_matcher(sanitized.as_ref(), ignore_case, multiline)
        .or_else(|error| {
            let message = error.to_string();
            if message.contains("unclosed group") || message.contains("unopened group") {
                let escaped = escape_unescaped_parentheses(sanitized.as_ref());
                if escaped.as_ref() != sanitized.as_ref() {
                    return build_regex_matcher(escaped.as_ref(), ignore_case, multiline);
                }
            }
            Err(error)
        })
        .map_err(|error| anyhow::anyhow!("failed to compile grep pattern: {error}"))
}

pub fn resolve_search_root(working_dir: &Path, path: &str) -> PathBuf {
    resolve_path(working_dir, path)
}

fn collect_entries(
    search_root: &Path,
    glob: Option<&str>,
    type_filter: Option<&str>,
) -> anyhow::Result<Vec<FileEntry>> {
    let metadata = std::fs::symlink_metadata(search_root)?;
    let type_filter = resolve_type_filter(type_filter);
    let matcher = compile_glob(glob)?;

    if metadata.file_type().is_symlink() {
        let resolved = std::fs::metadata(search_root)?;
        if !resolved.is_file() {
            return Ok(Vec::new());
        }
    } else if !metadata.is_file() && !metadata.is_dir() {
        return Ok(Vec::new());
    }

    if metadata.is_file() {
        if let Some(type_filter) = type_filter.as_ref()
            && !matches_type_filter(search_root, type_filter)
        {
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
            prefer_text_fast_path: is_known_text_path(search_root),
        }]);
    }

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
        if let Some(matcher) = matcher.as_ref()
            && !matcher.is_match(relative)
        {
            continue;
        }
        if let Some(type_filter) = type_filter.as_ref()
            && !matches_type_filter(path, type_filter)
        {
            continue;
        }

        entries.push(FileEntry {
            absolute_path: path.to_path_buf(),
            display_path: relative.to_string_lossy().into_owned(),
            prefer_text_fast_path: is_known_text_path(path),
        });
    }

    Ok(entries)
}

fn compile_glob(glob: Option<&str>) -> anyhow::Result<Option<GlobSet>> {
    let Some(glob) = glob.map(str::trim).filter(|glob| !glob.is_empty()) else {
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

fn resolve_type_filter(type_name: Option<&str>) -> Option<TypeFilter> {
    let normalized = type_name
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| value.trim_start_matches('.').to_ascii_lowercase())?;

    let (exts, names): (&[&str], &[&str]) = match normalized.as_str() {
        "js" | "javascript" => (&["js", "jsx", "mjs", "cjs"], &[]),
        "ts" | "typescript" => (&["ts", "tsx", "mts", "cts"], &[]),
        "json" => (&["json", "jsonc", "json5"], &[]),
        "yaml" | "yml" => (&["yaml", "yml"], &[]),
        "toml" => (&["toml"], &[]),
        "md" | "markdown" => (&["md", "markdown", "mdx"], &[]),
        "py" | "python" => (&["py", "pyi"], &[]),
        "rs" | "rust" => (&["rs"], &[]),
        "go" => (&["go"], &[]),
        "java" => (&["java"], &[]),
        "kt" | "kotlin" => (&["kt", "kts"], &[]),
        "c" => (&["c", "h"], &[]),
        "cpp" | "cxx" => (&["cpp", "cc", "cxx", "hpp", "hxx", "hh"], &[]),
        "cs" | "csharp" => (&["cs", "csx"], &[]),
        "php" => (&["php", "phtml"], &[]),
        "rb" | "ruby" => (&["rb", "rake", "gemspec"], &[]),
        "sh" | "bash" => (&["sh", "bash", "zsh"], &[]),
        "zsh" => (&["zsh"], &[]),
        "fish" => (&["fish"], &[]),
        "html" => (&["html", "htm"], &[]),
        "css" => (&["css"], &[]),
        "scss" => (&["scss"], &[]),
        "sass" => (&["sass"], &[]),
        "less" => (&["less"], &[]),
        "xml" => (&["xml"], &[]),
        "docker" | "dockerfile" => (&[], &["dockerfile"]),
        "make" | "makefile" => (&[], &["makefile"]),
        other => return Some(TypeFilter::Custom(other.to_owned())),
    };

    Some(TypeFilter::Known { exts, names })
}

fn matches_type_filter(path: &Path, filter: &TypeFilter) -> bool {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    if filter.match_name(file_name) {
        return true;
    }
    let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
    !extension.is_empty() && filter.match_ext(extension)
}

const KNOWN_TEXT_EXTENSIONS: &[&str] = &[
    "js", "jsx", "mjs", "cjs", "ts", "tsx", "mts", "cts", "json", "jsonc", "json5", "yaml", "yml",
    "toml", "md", "markdown", "mdx", "py", "pyi", "rs", "go", "java", "kt", "kts", "c", "h", "cpp",
    "cc", "cxx", "hpp", "hxx", "hh", "cs", "csx", "php", "phtml", "rb", "rake", "gemspec", "sh",
    "bash", "zsh", "fish", "html", "htm", "css", "scss", "sass", "less", "xml",
];

fn is_known_text_path(path: &Path) -> bool {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("");
    if file_name.eq_ignore_ascii_case("dockerfile") || file_name.eq_ignore_ascii_case("makefile") {
        return true;
    }

    let extension = path.extension().and_then(|ext| ext.to_str()).unwrap_or("");
    !extension.is_empty()
        && KNOWN_TEXT_EXTENSIONS
            .iter()
            .any(|candidate| extension.eq_ignore_ascii_case(candidate))
}

fn read_file_bytes(path: &Path, prefer_text_fast_path: bool) -> io::Result<Option<FileBytes>> {
    let metadata = std::fs::symlink_metadata(path)?;
    let resolved_metadata = if metadata.file_type().is_symlink() {
        let target_metadata = std::fs::metadata(path)?;
        if !target_metadata.is_file() {
            return Ok(None);
        }
        target_metadata
    } else if metadata.is_file() {
        metadata
    } else {
        return Ok(None);
    };

    if resolved_metadata.len() == 0 {
        return Ok(Some(FileBytes::Owned(Vec::new())));
    }

    let file = File::open(path)?;
    #[cfg(feature = "advanced-tools")]
    let bytes = match unsafe { memmap2::Mmap::map(&file) } {
        Ok(mapped) => FileBytes::Mapped(mapped),
        Err(_) => FileBytes::Owned(std::fs::read(path)?),
    };
    #[cfg(not(feature = "advanced-tools"))]
    let bytes = {
        let _ = file;
        FileBytes::Owned(std::fs::read(path)?)
    };

    if prefer_text_fast_path && is_known_text_path(path) {
        let slice = bytes.as_slice();
        let probe_len = slice.len().min(KNOWN_TEXT_BINARY_PROBE_BYTES);
        if slice[..probe_len].contains(&0) {
            return Ok(None);
        }
    } else if bytes.as_slice().contains(&0) {
        return Ok(None);
    }

    Ok(Some(bytes))
}

fn truncate_line(line: &str, max_columns: Option<usize>) -> (String, bool) {
    match max_columns {
        Some(max_columns) if visible_width(line) > max_columns => {
            (truncate_to_width(line, max_columns), true)
        }
        _ => (line.to_owned(), false),
    }
}

fn bytes_to_trimmed_string(bytes: &[u8]) -> String {
    match std::str::from_utf8(bytes) {
        Ok(text) => text.trim_end().to_owned(),
        Err(_) => String::from_utf8_lossy(bytes).trim_end().to_owned(),
    }
}

fn extract_context_lines(
    buffer: &[u8],
    match_range: Range<usize>,
    before: usize,
    after: usize,
    match_line_number: u64,
    max_columns: Option<usize>,
) -> (SmallVec<[ContextLine; 8]>, SmallVec<[ContextLine; 8]>) {
    let mut before_lines = SmallVec::new();
    let mut after_lines = SmallVec::new();

    if before > 0 && match_range.start > 0 {
        let mut end = match_range.start;
        let mut line_number = match_line_number;

        for _ in 0..before {
            if end == 0 || line_number == 0 {
                break;
            }
            let content_end = if buffer[end - 1] == b'\n' {
                end - 1
            } else {
                end
            };
            let start = match buffer[..content_end]
                .iter()
                .rposition(|&byte| byte == b'\n')
            {
                Some(position) => position + 1,
                None => 0,
            };
            line_number = line_number.saturating_sub(1);
            let raw = bytes_to_trimmed_string(&buffer[start..content_end]);
            let (line, _) = truncate_line(&raw, max_columns);
            before_lines.push(ContextLine { line_number, line });
            end = start;
        }
        before_lines.reverse();
    }

    if after > 0 && match_range.end < buffer.len() {
        let newline_count = buffer[match_range.clone()]
            .iter()
            .filter(|&&byte| byte == b'\n')
            .count() as u64;
        let mut start = match_range.end;
        for line_number in
            (match_line_number + newline_count)..(match_line_number + newline_count + after as u64)
        {
            if start >= buffer.len() {
                break;
            }
            let end = match buffer[start..].iter().position(|&byte| byte == b'\n') {
                Some(position) => start + position,
                None => buffer.len(),
            };
            let raw = bytes_to_trimmed_string(&buffer[start..end]);
            let (line, _) = truncate_line(&raw, max_columns);
            after_lines.push(ContextLine { line_number, line });
            start = end.saturating_add(1);
        }
    }

    (before_lines, after_lines)
}

fn build_response(result: AggregateResult, config: &SearchConfig) -> Value {
    match config.output_mode {
        OutputMode::Content => build_content_response(result, config),
        OutputMode::Count => build_count_response(result),
        OutputMode::FilesWithMatches => build_files_with_matches_response(result),
    }
}

fn build_content_response(result: AggregateResult, config: &SearchConfig) -> Value {
    let mut skipped = 0u64;
    let mut emitted = 0u64;
    let mut matches = Vec::new();

    'files: for file in &result.files {
        for matched in &file.matches {
            if skipped < config.offset {
                skipped = skipped.saturating_add(1);
                continue;
            }
            if emitted >= config.max_matches {
                break 'files;
            }
            matches.push(json!({
                "path": file.path,
                "line_number": matched.line_number,
                "line": matched.line,
                "context_before": if matched.context_before.is_empty() {
                    Value::Null
                } else {
                    json!(matched.context_before.iter().map(|line| json!({
                        "line_number": line.line_number,
                        "line": line.line,
                    })).collect::<Vec<_>>())
                },
                "context_after": if matched.context_after.is_empty() {
                    Value::Null
                } else {
                    json!(matched.context_after.iter().map(|line| json!({
                        "line_number": line.line_number,
                        "line": line.line,
                    })).collect::<Vec<_>>())
                },
                "truncated": matched.truncated,
            }));
            emitted = emitted.saturating_add(1);
        }
    }

    json!({
        "matches": matches,
        "total_matches": result.total_matches,
        "files_searched": result.files_searched,
        "files_with_matches": result.files_with_matches,
        "truncated": result.limit_reached
            || result.total_matches > config.offset.saturating_add(emitted),
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
        "truncated": result.limit_reached,
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
        "truncated": result.limit_reached,
    })
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
    fn extract_context_lines_returns_expected_lines() {
        let buffer = b"alpha\nbeta\ngamma\ndelta\n";
        let match_start = buffer
            .windows(6)
            .position(|window| window == b"gamma\n")
            .unwrap();
        let match_end = match_start + 6;
        let (before, after) = extract_context_lines(buffer, match_start..match_end, 1, 1, 3, None);

        assert_eq!(before.len(), 1);
        assert_eq!(before[0].line_number, 2);
        assert_eq!(before[0].line, "beta");
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].line_number, 4);
        assert_eq!(after[0].line, "delta");
    }
}
