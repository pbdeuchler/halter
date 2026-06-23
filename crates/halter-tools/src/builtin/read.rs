// pattern: Imperative Shell

use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use halter_protocol::{ToolCapabilities, ToolConcurrency, ToolName, ToolResult, ToolSpec};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tracing::debug;

use crate::{Tool, ToolContext};

use super::common::{ToolScope, ensure_not_cancelled, optional_bool, optional_u64, required_string, resolve_path};

const MAX_READ_LIMIT: u64 = 500;
const DEFAULT_READ_LIMIT: u64 = MAX_READ_LIMIT;
const DEFAULT_READ_TIMEOUT_SECS: u64 = 10;

#[derive(Debug)]
/// Built-in tool for reading UTF-8 file windows.
pub struct ReadTool;

#[derive(Debug, PartialEq, Eq)]
struct ReadWindow {
    content: String,
    sha256: String,
    total_lines: u64,
    /// `(line_number, line_text)` pairs for every line in the requested window.
    /// Only populated when `line_numbers` is requested; empty otherwise.
    lines: Vec<(u64, String)>,
}

#[async_trait]
impl Tool for ReadTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("read"),
            description: format!(
                "Read a UTF-8 text file, up to {MAX_READ_LIMIT} lines per call. `path` may \
                 be absolute or relative to the working directory. Page through larger files \
                 with `offset` (1-based start line) and `limit`; the response includes \
                 `total_lines`, so a returned count below it means the file continues past \
                 the window. Returns the raw file text plus its sha256. When reviewing or \
                 debugging code, set `line_numbers = true` to receive a `lines` array with \
                 per-line `{{\"line_number\", \"line\"}}` entries instead of using shell \
                 commands such as `nl`, `sed -n`, `awk`, or `wc -l`."
            ),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "offset": { "type": "integer", "minimum": 1, "default": 1 },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": MAX_READ_LIMIT,
                        "default": DEFAULT_READ_LIMIT
                    },
                    "timeout_secs": {
                        "type": "integer",
                        "minimum": 1,
                        "default": DEFAULT_READ_TIMEOUT_SECS
                    },
                    "line_numbers": {
                        "type": "boolean",
                        "default": false,
                        "description": "When true, return a structured `lines` array of `{\"line_number\": 1, \"line\": \"...\"}` objects in addition to the legacy `content` string."
                    }
                },
                "required": ["path"],
            }),
            concurrency: ToolConcurrency::ReadOnly,
            capabilities: ToolCapabilities::default(),
            provider_aliases: Default::default(),
        }
    }

    async fn execute(&self, context: ToolContext, input: Value) -> anyhow::Result<ToolResult> {
        let _scope = ToolScope::new(&context, "read");
        ensure_not_cancelled(&context.cancel)?;
        let path = resolve_path(&context.working_dir, required_string(&input, "path")?);
        let offset = optional_u64(&input, "offset")?.unwrap_or(1);
        let limit = parse_limit(&input)?;
        let timeout = parse_timeout(&input)?;
        let line_numbers = optional_bool(&input, "line_numbers")?.unwrap_or(false);
        let deadline = Instant::now()
            .checked_add(timeout)
            .unwrap_or_else(Instant::now);
        debug!(
            session_id = %context.session_id,
            path = %path.display(),
            offset,
            limit,
            timeout_secs = timeout.as_secs(),
            line_numbers,
            "reading file"
        );

        let canonical = context.policy.check_read_path(&path, 0).await?;
        let path_locks = context.path_locks.clone();
        let canonical_path = canonical.path().to_path_buf();
        let path_for_read = canonical_path.clone();
        let (file, readable_bytes) = tokio::task::spawn_blocking(move || {
            let _lock = path_locks.acquire_read(&path_for_read)?;
            let file = canonical.open_read_blocking()?;
            let len = usize::try_from(file.metadata()?.len()).map_err(|_| {
                anyhow::anyhow!("failed to execute read tool: file is too large for this platform")
            })?;
            Ok::<_, anyhow::Error>((file, len))
        })
        .await??;
        context
            .policy
            .check_read_path(&canonical_path, readable_bytes)
            .await?;

        let path_locks = context.path_locks.clone();
        let path_for_read = canonical_path.clone();
        let read_window = tokio::task::spawn_blocking(move || {
            let _lock = path_locks.acquire_read(&path_for_read)?;
            let reader = BufReader::new(file);
            read_window_from_reader(
                reader,
                offset,
                limit,
                line_numbers,
                timeout,
                deadline,
                readable_bytes,
            )
        })
        .await??;

        let mut result = serde_json::Map::new();
        result.insert("path".to_owned(), json!(canonical_path));
        result.insert("content".to_owned(), json!(read_window.content));
        result.insert("sha256".to_owned(), json!(read_window.sha256));
        result.insert("total_lines".to_owned(), json!(read_window.total_lines));
        if line_numbers {
            let lines: Vec<Value> = read_window
                .lines
                .into_iter()
                .map(|(line_number, line)| {
                    json!({
                        "line_number": line_number,
                        "line": line,
                    })
                })
                .collect();
            result.insert("lines".to_owned(), json!(lines));
        }

        Ok(ToolResult::Json {
            value: Value::Object(result),
        })
    }
}

fn parse_limit(input: &Value) -> anyhow::Result<u64> {
    let limit = optional_u64(input, "limit")?.unwrap_or(DEFAULT_READ_LIMIT);
    if !(1..=MAX_READ_LIMIT).contains(&limit) {
        anyhow::bail!("invalid tool input: field 'limit' must be between 1 and {MAX_READ_LIMIT}");
    }
    Ok(limit)
}

fn parse_timeout(input: &Value) -> anyhow::Result<Duration> {
    let timeout_secs = optional_u64(input, "timeout_secs")?.unwrap_or(DEFAULT_READ_TIMEOUT_SECS);
    if timeout_secs == 0 {
        anyhow::bail!("invalid tool input: field 'timeout_secs' must be at least 1");
    }
    Ok(Duration::from_secs(timeout_secs))
}

fn read_window_from_reader<R: BufRead>(
    mut reader: R,
    offset: u64,
    limit: u64,
    line_numbers: bool,
    timeout: Duration,
    deadline: Instant,
    max_bytes: usize,
) -> anyhow::Result<ReadWindow> {
    let start_line = offset.max(1);
    let end_line = start_line.saturating_add(limit).saturating_sub(1);
    let mut content = String::new();
    let mut total_lines = 0;
    let mut current_line = String::new();
    let mut hasher = Sha256::new();
    let mut bytes_read = 0usize;
    let mut lines = Vec::new();

    loop {
        ensure_before_deadline(timeout, deadline)?;
        current_line.clear();
        let read = reader.read_line(&mut current_line)?;
        bytes_read = bytes_read.saturating_add(read);
        if bytes_read > max_bytes {
            anyhow::bail!(
                "failed to execute read tool: read exceeded authorized byte limit of {max_bytes}"
            );
        }
        ensure_before_deadline(timeout, deadline)?;
        if read == 0 {
            break;
        }

        total_lines += 1;
        hasher.update(current_line.as_bytes());
        if (start_line..=end_line).contains(&total_lines) {
            content.push_str(&current_line);
            if line_numbers {
                lines.push((total_lines, current_line.clone()));
            }
        }
    }

    Ok(ReadWindow {
        content,
        sha256: format!("{:x}", hasher.finalize()),
        total_lines,
        lines,
    })
}

fn ensure_before_deadline(timeout: Duration, deadline: Instant) -> anyhow::Result<()> {
    if Instant::now() >= deadline {
        anyhow::bail!(
            "failed to execute read tool: timed out after {} seconds",
            timeout.as_secs()
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::sync::Arc;

    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    use crate::{
        DefaultToolPolicy, NoopToolEventSink, PathLockMap, PolicySettings, ToolPolicy,
        ToolSessionStore,
    };

    use super::*;

    #[test]
    fn reads_requested_line_window() {
        let window = read_window_from_reader(
            Cursor::new("a\nb\nc\n"),
            2,
            1,
            false,
            Duration::from_secs(10),
            Instant::now()
                .checked_add(Duration::from_secs(10))
                .expect("future deadline"),
            usize::MAX,
        )
        .expect("window read succeeds");

        assert_eq!(window.content, "b\n");
        assert_eq!(window.total_lines, 3);
    }

    fn tool_context(root: &std::path::Path, max_read_bytes: usize) -> ToolContext {
        ToolContext {
            session_id: halter_protocol::SessionId::new(),
            working_dir: root.to_path_buf(),
            path_locks: Arc::new(PathLockMap::default()),
            tool_sessions: Arc::new(ToolSessionStore::default()),
            snapshot: Arc::new(halter_protocol::ResourceSnapshot::empty()),
            cancel: CancellationToken::new(),
            emit: Arc::new(NoopToolEventSink),
            policy: Arc::new(DefaultToolPolicy::new(PolicySettings {
                allowed_write_roots: vec![root.to_path_buf()],
                max_read_bytes,
                ..PolicySettings::default()
            })) as Arc<dyn ToolPolicy>,
            shell_timeout_secs: 30,
            subagent_parent: None,
        }
    }

    #[tokio::test]
    async fn read_checks_total_file_size_before_reading() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("note.txt"), "a\nb\n").expect("write");

        let error = ReadTool
            .execute(
                tool_context(temp.path(), 2),
                json!({
                    "path": "note.txt",
                    "offset": 2,
                    "limit": 1
                }),
            )
            .await
            .expect_err("full file size exceeds policy");

        assert!(
            error
                .to_string()
                .contains("read size 4 exceeds max_read_bytes 2")
        );
    }

    #[tokio::test]
    async fn read_returns_requested_slice_when_file_fits_policy() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("note.txt"), "a\nb\n").expect("write");

        let ToolResult::Json { value } = ReadTool
            .execute(
                tool_context(temp.path(), 4),
                json!({
                    "path": "note.txt",
                    "offset": 2,
                    "limit": 1
                }),
            )
            .await
            .expect("read succeeds")
        else {
            panic!("expected json result");
        };

        assert_eq!(value["content"], "b\n");
        assert!(value.get("lines").is_none());
    }

    #[tokio::test]
    async fn read_returns_lines_array_when_line_numbers_enabled() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("note.txt"), "a\nb\nc\nd\n").expect("write");

        let ToolResult::Json { value } = ReadTool
            .execute(
                tool_context(temp.path(), usize::MAX),
                json!({
                    "path": "note.txt",
                    "offset": 2,
                    "limit": 2,
                    "line_numbers": true
                }),
            )
            .await
            .expect("read succeeds")
        else {
            panic!("expected json result");
        };

        assert_eq!(value["content"], "b\nc\n");
        assert_eq!(value["total_lines"], 4);
        let lines = value["lines"].as_array().expect("lines array");
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["line_number"], 2);
        assert_eq!(lines[0]["line"], "b\n");
        assert_eq!(lines[1]["line_number"], 3);
        assert_eq!(lines[1]["line"], "c\n");
    }

    #[tokio::test]
    async fn read_omits_lines_field_when_line_numbers_disabled() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("note.txt"), "a\nb\nc\nd\n").expect("write");

        let ToolResult::Json { value } = ReadTool
            .execute(
                tool_context(temp.path(), usize::MAX),
                json!({
                    "path": "note.txt",
                    "offset": 2,
                    "limit": 2,
                    "line_numbers": false
                }),
            )
            .await
            .expect("read succeeds")
        else {
            panic!("expected json result");
        };

        assert_eq!(value["content"], "b\nc\n");
        assert_eq!(value["total_lines"], 4);
        assert!(value.get("lines").is_none());
    }

    #[tokio::test]
    async fn read_line_numbers_default_is_false() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("note.txt"), "a\nb\nc\nd\n").expect("write");

        let ToolResult::Json { value } = ReadTool
            .execute(
                tool_context(temp.path(), usize::MAX),
                json!({
                    "path": "note.txt",
                    "offset": 2,
                    "limit": 2
                }),
            )
            .await
            .expect("read succeeds")
        else {
            panic!("expected json result");
        };

        assert!(value.get("lines").is_none());
    }

    #[tokio::test]
    async fn read_line_numbers_respects_offset_and_limit() {
        let temp = tempfile::tempdir().expect("tempdir");
        let text = (1..=10)
            .map(|line| format!("line {line}\n"))
            .collect::<String>();
        std::fs::write(temp.path().join("note.txt"), text).expect("write");

        let ToolResult::Json { value } = ReadTool
            .execute(
                tool_context(temp.path(), usize::MAX),
                json!({
                    "path": "note.txt",
                    "offset": 4,
                    "limit": 3,
                    "line_numbers": true
                }),
            )
            .await
            .expect("read succeeds")
        else {
            panic!("expected json result");
        };

        let lines = value["lines"].as_array().expect("lines array");
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0]["line_number"], 4);
        assert_eq!(lines[0]["line"], "line 4\n");
        assert_eq!(lines[1]["line_number"], 5);
        assert_eq!(lines[1]["line"], "line 5\n");
        assert_eq!(lines[2]["line_number"], 6);
        assert_eq!(lines[2]["line"], "line 6\n");
        assert_eq!(value["total_lines"], 10);
    }

    #[tokio::test]
    async fn read_line_numbers_truncates_to_max_limit() {
        let temp = tempfile::tempdir().expect("tempdir");
        let text = (1..=600)
            .map(|line| format!("line {line}\n"))
            .collect::<String>();
        std::fs::write(temp.path().join("note.txt"), text).expect("write");

        let ToolResult::Json { value } = ReadTool
            .execute(
                tool_context(temp.path(), usize::MAX),
                json!({
                    "path": "note.txt",
                    "line_numbers": true
                }),
            )
            .await
            .expect("read succeeds")
        else {
            panic!("expected json result");
        };

        let lines = value["lines"].as_array().expect("lines array");
        assert_eq!(lines.len(), 500);
        assert_eq!(lines[0]["line_number"], 1);
        assert_eq!(lines[499]["line_number"], 500);
    }

    #[tokio::test]
    async fn read_line_numbers_sha256_matches_legacy_path() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("note.txt"), "a\nb\nc\n").expect("write");

        let ToolResult::Json { value: enabled } = ReadTool
            .execute(
                tool_context(temp.path(), usize::MAX),
                json!({
                    "path": "note.txt",
                    "line_numbers": true
                }),
            )
            .await
            .expect("read succeeds")
        else {
            panic!("expected json result");
        };

        let ToolResult::Json { value: disabled } = ReadTool
            .execute(
                tool_context(temp.path(), usize::MAX),
                json!({ "path": "note.txt" }),
            )
            .await
            .expect("read succeeds")
        else {
            panic!("expected json result");
        };

        assert_eq!(enabled["sha256"], disabled["sha256"]);
    }

    #[tokio::test]
    async fn read_line_numbers_empty_file() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("note.txt"), "").expect("write");

        let ToolResult::Json { value } = ReadTool
            .execute(
                tool_context(temp.path(), usize::MAX),
                json!({
                    "path": "note.txt",
                    "line_numbers": true
                }),
            )
            .await
            .expect("read succeeds")
        else {
            panic!("expected json result");
        };

        let lines = value["lines"].as_array().expect("lines array");
        assert!(lines.is_empty());
        assert_eq!(value["content"], "");
        assert_eq!(value["total_lines"], 0);
    }

    #[tokio::test]
    async fn read_line_numbers_zero_window_is_empty_array() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("note.txt"), "1\n2\n3\n4\n5\n").expect("write");

        let ToolResult::Json { value } = ReadTool
            .execute(
                tool_context(temp.path(), usize::MAX),
                json!({
                    "path": "note.txt",
                    "offset": 10,
                    "limit": 1,
                    "line_numbers": true
                }),
            )
            .await
            .expect("read succeeds")
        else {
            panic!("expected json result");
        };

        let lines = value["lines"].as_array().expect("lines array");
        assert!(lines.is_empty());
        assert_eq!(value["content"], "");
        assert_eq!(value["total_lines"], 5);
    }

    #[tokio::test]
    async fn read_rejects_non_boolean_line_numbers() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("note.txt"), "a\n").expect("write");

        let error = ReadTool
            .execute(
                tool_context(temp.path(), usize::MAX),
                json!({
                    "path": "note.txt",
                    "line_numbers": "yes"
                }),
            )
            .await
            .expect_err("non-boolean line_numbers is rejected");

        assert!(
            error
                .to_string()
                .contains("field 'line_numbers' must be a boolean")
        );
    }

    #[tokio::test]
    async fn read_line_numbers_does_not_break_byte_cap() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("note.txt"), "a\nb\n").expect("write");

        let error = ReadTool
            .execute(
                tool_context(temp.path(), 2),
                json!({
                    "path": "note.txt",
                    "line_numbers": true
                }),
            )
            .await
            .expect_err("full file size exceeds policy");

        assert!(
            error
                .to_string()
                .contains("read size 4 exceeds max_read_bytes 2")
        );
    }

    #[test]
    fn read_window_helper_returns_lines_when_enabled() {
        let window = read_window_from_reader(
            Cursor::new("a\nb\nc\n"),
            1,
            2,
            true,
            Duration::from_secs(10),
            Instant::now()
                .checked_add(Duration::from_secs(10))
                .expect("future deadline"),
            usize::MAX,
        )
        .expect("window read succeeds");

        assert_eq!(window.lines, vec![(1, "a\n".to_owned()), (2, "b\n".to_owned())]);
    }

    #[test]
    fn read_window_helper_lines_empty_when_disabled() {
        let window = read_window_from_reader(
            Cursor::new("a\nb\nc\n"),
            1,
            2,
            false,
            Duration::from_secs(10),
            Instant::now()
                .checked_add(Duration::from_secs(10))
                .expect("future deadline"),
            usize::MAX,
        )
        .expect("window read succeeds");

        assert!(window.lines.is_empty());
    }

    #[test]
    fn read_window_times_out() {
        let error = read_window_from_reader(
            Cursor::new("a\nb\n"),
            1,
            1,
            false,
            Duration::from_secs(10),
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("past deadline"),
            usize::MAX,
        )
        .expect_err("read should time out");

        assert!(
            error
                .to_string()
                .contains("failed to execute read tool: timed out after 10 seconds")
        );
    }

    #[test]
    fn read_line_numbers_does_not_bypass_timeout() {
        let error = read_window_from_reader(
            Cursor::new("a\nb\n"),
            1,
            1,
            true,
            Duration::from_secs(10),
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("past deadline"),
            usize::MAX,
        )
        .expect_err("read should time out");

        assert!(
            error
                .to_string()
                .contains("failed to execute read tool: timed out after 10 seconds")
        );
    }

    #[test]
    fn read_window_fails_when_reader_exceeds_authorized_bytes() {
        let error = read_window_from_reader(
            Cursor::new("a\nb\n"),
            1,
            2,
            false,
            Duration::from_secs(10),
            Instant::now()
                .checked_add(Duration::from_secs(10))
                .expect("future deadline"),
            2,
        )
        .expect_err("read should exceed authorized bytes");

        assert!(
            error
                .to_string()
                .contains("read exceeded authorized byte limit")
        );
    }

    #[test]
    fn read_line_numbers_does_not_bypass_byte_limit_in_helper() {
            )
            .await
            .expect("read succeeds")
        else {
            panic!("expected json result");
        };

        let content = value["content"].as_str().expect("content string");
        assert_eq!(content.lines().count(), 500);
        assert!(content.contains("line 500"));
        assert!(!content.contains("line 501"));
        assert_eq!(value["total_lines"], 600);
    }

    #[tokio::test]
    async fn read_rejects_limit_above_max() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("note.txt"), "a\n").expect("write");

        let error = ReadTool
            .execute(
                tool_context(temp.path(), usize::MAX),
                json!({
                    "path": "note.txt",
                    "limit": MAX_READ_LIMIT + 1
                }),
            )
            .await
            .expect_err("limit above max is rejected");

        assert!(
            error
                .to_string()
                .contains("invalid tool input: field 'limit' must be between 1 and 500")
        );
    }

    #[tokio::test]
    async fn read_line_numbers_does_not_bypass_byte_limit_in_helper() {
        let error = read_window_from_reader(
            Cursor::new("a\nb\n"),
            1,
            2,
            true,
            Duration::from_secs(10),
            Instant::now()
                .checked_add(Duration::from_secs(10))
                .expect("future deadline"),
            2,
        )
        .expect_err("read should exceed authorized bytes");

        assert!(
            error
                .to_string()
                .contains("read exceeded authorized byte limit")
        );
    }

    #[test]
    fn read_window_times_out() {
        let error = read_window_from_reader(
            Cursor::new("a\nb\n"),
            1,
            1,
            false,
            Duration::from_secs(10),
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("past deadline"),
            usize::MAX,
        )
        .expect_err("read should time out");

        assert!(
            error
                .to_string()
                .contains("failed to execute read tool: timed out after 10 seconds")
        );
    }

    #[test]
    fn read_line_numbers_does_not_bypass_timeout() {
        let error = read_window_from_reader(
            Cursor::new("a\nb\n"),
            1,
            1,
            true,
            Duration::from_secs(10),
            Instant::now()
                .checked_sub(Duration::from_secs(1))
                .expect("past deadline"),
            usize::MAX,
        )
        .expect_err("read should time out");

        assert!(
            error
                .to_string()
                .contains("failed to execute read tool: timed out after 10 seconds")
        );
    }

    #[test]
    fn read_window_fails_when_reader_exceeds_authorized_bytes() {
        let error = read_window_from_reader(
            Cursor::new("a\nb\n"),
            1,
            2,
            false,
            Duration::from_secs(10),
            Instant::now()
                .checked_add(Duration::from_secs(10))
                .expect("future deadline"),
            2,
        )
        .expect_err("read should exceed authorized bytes");

        assert!(
            error
                .to_string()
                .contains("read exceeded authorized byte limit")
        );
    }
}
