// pattern: Imperative Shell

use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use halter_protocol::{ToolCapabilities, ToolConcurrency, ToolName, ToolResult, ToolSpec};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tracing::debug;

use crate::{Tool, ToolContext};

use super::common::{ToolScope, ensure_not_cancelled, optional_u64, required_string, resolve_path};

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
}

#[async_trait]
impl Tool for ReadTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("read"),
            description: format!("Read up to {MAX_READ_LIMIT} UTF-8 lines from disk"),
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
        let deadline = Instant::now()
            .checked_add(timeout)
            .unwrap_or_else(Instant::now);
        debug!(
            session_id = %context.session_id,
            path = %path.display(),
            offset,
            limit,
            timeout_secs = timeout.as_secs(),
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
            read_window_from_reader(reader, offset, limit, timeout, deadline, readable_bytes)
        })
        .await??;

        Ok(ToolResult::Json {
            value: json!({
                "path": canonical_path,
                "content": read_window.content,
                "sha256": read_window.sha256,
                "total_lines": read_window.total_lines,
            }),
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
        }
    }

    Ok(ReadWindow {
        content,
        sha256: format!("{:x}", hasher.finalize()),
        total_lines,
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
    }

    #[tokio::test]
    async fn read_defaults_limit_to_500_lines() {
        let temp = tempfile::tempdir().expect("tempdir");
        let text = (1..=600)
            .map(|line| format!("line {line}\n"))
            .collect::<String>();
        std::fs::write(temp.path().join("note.txt"), text).expect("write");

        let ToolResult::Json { value } = ReadTool
            .execute(
                tool_context(temp.path(), usize::MAX),
                json!({ "path": "note.txt" }),
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

    #[test]
    fn read_window_times_out() {
        let error = read_window_from_reader(
            Cursor::new("a\nb\n"),
            1,
            1,
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
