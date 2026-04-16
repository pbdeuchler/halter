// pattern: Imperative Shell

use async_trait::async_trait;
use halter_protocol::{ToolCapabilities, ToolConcurrency, ToolName, ToolResult, ToolSpec};
use serde_json::{Value, json};
use tracing::debug;

use crate::{Tool, ToolContext};

use super::common::{
    ToolScope, atomic_write_blocking, ensure_not_cancelled, hash_text, optional_bool,
    required_string, resolve_path,
};

#[derive(Debug)]
pub struct EditTool;

#[async_trait]
impl Tool for EditTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("edit"),
            description: "Replace an exact string in a UTF-8 file using an atomic write".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "old_string": { "type": "string" },
                    "new_string": { "type": "string" },
                    "expected_sha256": { "type": "string" },
                    "replace_all": { "type": "boolean" }
                },
                "required": ["path", "old_string", "new_string"],
            }),
            concurrency: ToolConcurrency::Exclusive,
            capabilities: ToolCapabilities {
                mutating: true,
                requires_approval: false,
                cancellable: false,
                long_running: false,
            },
            provider_aliases: Default::default(),
        }
    }

    async fn execute(&self, context: ToolContext, input: Value) -> anyhow::Result<ToolResult> {
        let _scope = ToolScope::new(&context, "edit");
        ensure_not_cancelled(&context.cancel)?;
        let path = resolve_path(&context.working_dir, required_string(&input, "path")?);
        let old_string = required_string(&input, "old_string")?;
        let new_string = required_string(&input, "new_string")?;
        let expected_sha256 = input
            .get("expected_sha256")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let replace_all = optional_bool(&input, "replace_all")?.unwrap_or(false);
        if old_string.is_empty() {
            anyhow::bail!("failed to execute edit tool: old_string must not be empty");
        }
        debug!(
            session_id = %context.session_id,
            path = %path.display(),
            replace_all,
            "editing file"
        );

        context.policy.check_write(&path).await?;
        let path_locks = context.path_locks.clone();
        let path_for_edit = path.clone();
        let old = old_string.to_owned();
        let new = new_string.to_owned();
        let expected_sha256 = expected_sha256.clone();
        let result = tokio::task::spawn_blocking(move || {
            let _lock = path_locks.acquire_write(&path_for_edit)?;
            let original = std::fs::read_to_string(&path_for_edit)?;
            let file_hash_before = hash_text(&original);
            if let Some(expected_sha256) = expected_sha256.as_deref()
                && expected_sha256 != file_hash_before
            {
                anyhow::bail!(
                    "failed to execute edit tool: expected_sha256 does not match '{}'",
                    path_for_edit.display()
                );
            }
            let matches = original.match_indices(&old).count();
            if matches == 0 {
                anyhow::bail!(
                    "failed to execute edit tool: old_string was not found in '{}'",
                    path_for_edit.display()
                );
            }
            if matches > 1 && !replace_all {
                anyhow::bail!(
                    "failed to execute edit tool: old_string matched {} times in '{}'; set replace_all=true to allow this",
                    matches,
                    path_for_edit.display()
                );
            }

            let updated = if replace_all {
                original.replace(&old, &new)
            } else {
                original.replacen(&old, &new, 1)
            };
            let file_hash_after = hash_text(&updated);
            atomic_write_blocking(&path_for_edit, updated.as_bytes())?;

            Ok::<_, anyhow::Error>((matches, file_hash_before, file_hash_after))
        })
        .await??;

        Ok(ToolResult::Json {
            value: json!({
                "path": path,
                "matches_replaced": if replace_all { result.0 } else { 1 },
                "file_hash_before": result.1,
                "file_hash_after": result.2,
            }),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    use crate::{DefaultToolPolicy, NoopToolEventSink, PathLockMap, PolicySettings, ToolPolicy};

    use super::*;

    fn tool_context(root: &std::path::Path) -> ToolContext {
        ToolContext {
            session_id: halter_protocol::SessionId::new(),
            working_dir: root.to_path_buf(),
            path_locks: Arc::new(PathLockMap::default()),
            tool_sessions: Arc::new(crate::ToolSessionStore::default()),
            file_view: Arc::new(Default::default()),
            snapshot: Arc::new(halter_protocol::ResourceSnapshot::empty()),
            cancel: CancellationToken::new(),
            emit: Arc::new(NoopToolEventSink),
            policy: Arc::new(DefaultToolPolicy::new(PolicySettings {
                allowed_write_roots: vec![root.to_path_buf()],
                ..PolicySettings::permissive()
            })) as Arc<dyn ToolPolicy>,
            max_tool_output_bytes: 16_384,
            shell_timeout_secs: 30,
            subagent_parent: None,
        }
    }

    #[tokio::test]
    async fn edits_single_match() {
        let temp = tempfile::tempdir().expect("tempdir");
        let path = temp.path().join("note.txt");
        std::fs::write(&path, "hello world").expect("write");
        let expected_sha256 = hash_text("hello world");

        let ToolResult::Json { value } = EditTool
            .execute(
                tool_context(temp.path()),
                json!({
                    "path": "note.txt",
                    "old_string": "world",
                    "new_string": "there",
                    "expected_sha256": expected_sha256
                }),
            )
            .await
            .expect("edit succeeds")
        else {
            panic!("expected json result");
        };

        assert_eq!(value["matches_replaced"], 1);
        assert_eq!(std::fs::read_to_string(path).expect("read"), "hello there");
    }

    #[tokio::test]
    async fn edit_requires_replace_all_for_ambiguous_match() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("note.txt"), "x x").expect("write");
        let error = EditTool
            .execute(
                tool_context(temp.path()),
                json!({
                    "path": "note.txt",
                    "old_string": "x",
                    "new_string": "y"
                }),
            )
            .await
            .expect_err("edit should fail");

        assert!(error.to_string().contains("replace_all=true"));
    }

    #[tokio::test]
    async fn edit_rejects_stale_expected_hash() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("note.txt"), "hello").expect("write");

        let error = EditTool
            .execute(
                tool_context(temp.path()),
                json!({
                    "path": "note.txt",
                    "old_string": "hello",
                    "new_string": "goodbye",
                    "expected_sha256": "stale"
                }),
            )
            .await
            .expect_err("stale hash should fail");

        assert!(error.to_string().contains("expected_sha256"));
    }
}
