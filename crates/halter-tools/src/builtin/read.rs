// pattern: Imperative Shell

use async_trait::async_trait;
use halter_protocol::{ToolCapabilities, ToolConcurrency, ToolName, ToolResult, ToolSpec};
use serde_json::{Value, json};
use tracing::debug;

use crate::{Tool, ToolContext};

use super::common::{
    ToolScope, ensure_not_cancelled, hash_text, optional_u64, required_string, resolve_path,
};

#[derive(Debug)]
pub struct ReadTool;

#[async_trait]
impl Tool for ReadTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("read"),
            description: "Read a UTF-8 file from disk".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "offset": { "type": "integer", "minimum": 1 },
                    "limit": { "type": "integer", "minimum": 1 }
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
        let limit = optional_u64(&input, "limit")?;
        debug!(session_id = %context.session_id, path = %path.display(), offset, limit, "reading file");

        let path_locks = context.path_locks.clone();
        let path_for_read = path.clone();
        let text = tokio::task::spawn_blocking(move || {
            let _lock = path_locks.acquire_read(&path_for_read)?;
            let text = std::fs::read_to_string(&path_for_read)?;
            Ok::<_, anyhow::Error>(text)
        })
        .await??;
        let (content, total_lines) = slice_by_lines(&text, offset, limit);
        context.policy.check_read(&path, content.len()).await?;

        Ok(ToolResult::Json {
            value: json!({
                "path": path,
                "content": content,
                "sha256": hash_text(&text),
                "total_lines": total_lines,
            }),
        })
    }
}

fn slice_by_lines(text: &str, offset: u64, limit: Option<u64>) -> (String, u64) {
    let starts = line_start_offsets(text);
    let total_lines = starts.len() as u64;
    if total_lines == 0 {
        return (String::new(), 0);
    }

    let start_line = offset.max(1);
    if start_line > total_lines {
        return (String::new(), total_lines);
    }

    let end_line = limit
        .map(|limit| start_line.saturating_add(limit).saturating_sub(1))
        .unwrap_or(total_lines)
        .min(total_lines);

    let start_index = starts[(start_line - 1) as usize];
    let end_index = if end_line < total_lines {
        starts[end_line as usize]
    } else {
        text.len()
    };

    (text[start_index..end_index].to_owned(), total_lines)
}

fn line_start_offsets(text: &str) -> Vec<usize> {
    if text.is_empty() {
        return Vec::new();
    }
    let mut starts = vec![0];
    for (index, byte) in text.bytes().enumerate() {
        if byte == b'\n' && index + 1 < text.len() {
            starts.push(index + 1);
        }
    }
    starts
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    use crate::{
        DefaultToolPolicy, NoopToolEventSink, PathLockMap, PolicySettings, ToolPolicy,
        ToolSessionStore,
    };

    use super::*;

    #[test]
    fn slices_requested_line_window() {
        let (content, total_lines) = slice_by_lines("a\nb\nc\n", 2, Some(1));
        assert_eq!(content, "b\n");
        assert_eq!(total_lines, 3);
    }

    fn tool_context(root: &std::path::Path, max_read_bytes: usize) -> ToolContext {
        ToolContext {
            session_id: halter_protocol::SessionId::new(),
            working_dir: root.to_path_buf(),
            path_locks: Arc::new(PathLockMap::default()),
            tool_sessions: Arc::new(ToolSessionStore::default()),
            file_view: Arc::new(Default::default()),
            snapshot: Arc::new(halter_protocol::ResourceSnapshot::empty()),
            cancel: CancellationToken::new(),
            emit: Arc::new(NoopToolEventSink),
            policy: Arc::new(DefaultToolPolicy::new(PolicySettings {
                allowed_write_roots: vec![root.to_path_buf()],
                max_read_bytes,
                ..PolicySettings::default()
            })) as Arc<dyn ToolPolicy>,
            max_tool_output_bytes: 16_384,
            shell_timeout_secs: 30,
            subagent_parent: None,
        }
    }

    #[tokio::test]
    async fn read_checks_returned_slice_size() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("note.txt"), "a\nb\n").expect("write");

        let ToolResult::Json { value } = ReadTool
            .execute(
                tool_context(temp.path(), 2),
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
}
