// pattern: Imperative Shell

use std::path::PathBuf;

use async_trait::async_trait;
use halter_protocol::{ToolCapabilities, ToolConcurrency, ToolName, ToolSpec};
use serde::Serialize;
use serde_json::{Value, json};
use tracing::debug;

use crate::{ToolContext, ToolOutput, TypedTool};

use super::common::{
    ToolScope, ensure_not_cancelled, hash_text, optional_bool, optional_string, required_string,
    resolve_path,
};

#[derive(Debug)]
/// Built-in tool for exact-string file edits.
pub struct EditTool;

#[async_trait]
impl TypedTool for EditTool {
    type Output = EditOutput;

    fn describe(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("edit"),
            description: "Replace an exact string in a UTF-8 file with an atomic write. \
                `old_string` must match the file's raw text exactly, including whitespace, \
                and be unique — keep it minimal, just enough context to be unique. If it is \
                not unique the edit fails: add surrounding context, or set `replace_all` to \
                change every occurrence (handy for renames). Pass `expected_sha256` (the \
                sha256 from `read` or a previous `edit`) to fail fast if the file changed \
                underneath you; omit it otherwise."
                .to_owned(),
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

    async fn run(&self, context: ToolContext, input: Value) -> anyhow::Result<Self::Output> {
        let _scope = ToolScope::new(&context, "edit");
        ensure_not_cancelled(&context.cancel)?;
        let path = resolve_path(&context.working_dir, required_string(&input, "path")?);
        let old_string = required_string(&input, "old_string")?;
        let new_string = required_string(&input, "new_string")?;
        // Anything but a SHA-256 hex digest is a placeholder ("", "unused"),
        // never a hash that could match, so it is treated as absent.
        let expected_sha256 = optional_string(&input, "expected_sha256")
            .filter(|hash| hash.len() == 64 && hash.bytes().all(|byte| byte.is_ascii_hexdigit()))
            .map(str::to_ascii_lowercase);
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

        let canonical = context.policy.check_write_path(&path).await?;
        let canonical_path = canonical.path().to_path_buf();
        let path_locks = context.path_locks.clone();
        let path_for_edit = canonical_path.clone();
        let old = old_string.to_owned();
        let new = new_string.to_owned();
        let result = tokio::task::spawn_blocking(move || {
            use std::io::Read;

            let _lock = path_locks.acquire_write(&path_for_edit)?;
            let mut original = String::new();
            canonical
                .open_read_blocking()?
                .read_to_string(&mut original)?;
            let file_hash_before = hash_text(&original);
            if let Some(expected_sha256) = expected_sha256.as_deref()
                && expected_sha256 != file_hash_before
            {
                anyhow::bail!(
                    "failed to execute edit tool: '{}' changed since expected_sha256 was taken \
                     (its sha256 is now {file_hash_before}); read it again, or omit expected_sha256",
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
            canonical.atomic_write_blocking(updated.as_bytes())?;

            Ok::<_, anyhow::Error>((matches, file_hash_before, file_hash_after))
        })
        .await??;

        let occurrences_in_file = result.0;
        Ok(EditOutput {
            path: canonical_path,
            occurrences_in_file,
            replacements_applied: if replace_all { occurrences_in_file } else { 1 },
            file_hash_before: result.1,
            file_hash_after: result.2,
        })
    }
}

#[derive(Debug, Serialize)]
/// Result of an `edit` call.
pub struct EditOutput {
    path: PathBuf,
    occurrences_in_file: usize,
    replacements_applied: usize,
    file_hash_before: String,
    file_hash_after: String,
}

impl ToolOutput for EditOutput {
    /// The new hash is the `expected_sha256` for a follow-up edit.
    fn to_model_string(&self) -> String {
        format!(
            "Edited {} ({} replaced); sha256={}",
            self.path.display(),
            self.replacements_applied,
            self.file_hash_after
        )
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    use crate::{DefaultToolPolicy, NoopToolEventSink, PathLockMap, PolicySettings, ToolPolicy};

    use super::*;
    use crate::Tool;
    use halter_protocol::ToolResult;

    fn tool_context(root: &std::path::Path) -> ToolContext {
        ToolContext {
            session_id: halter_protocol::SessionId::new(),
            working_dir: root.to_path_buf(),
            path_locks: Arc::new(PathLockMap::default()),
            tool_sessions: Arc::new(crate::ToolSessionStore::default()),
            snapshot: Arc::new(halter_protocol::ResourceSnapshot::empty()),
            cancel: CancellationToken::new(),
            emit: Arc::new(NoopToolEventSink),
            policy: Arc::new(DefaultToolPolicy::new(PolicySettings {
                allowed_write_roots: vec![root.to_path_buf()],
                ..PolicySettings::default()
            })) as Arc<dyn ToolPolicy>,
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

        let ToolResult::Rendered { value, .. } = EditTool
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

        assert_eq!(value["occurrences_in_file"], 1);
        assert_eq!(value["replacements_applied"], 1);
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

    /// Only a real sha256 that no longer matches refuses the edit; anything
    /// that is not a sha256 is a placeholder and is ignored.
    #[tokio::test]
    async fn edit_checks_expected_hash_only_when_it_is_a_sha256() {
        let stale = hash_text("before");
        let current = hash_text("hello").to_ascii_uppercase();
        for (expected_sha256, refused) in [
            (json!(stale), true),
            (json!(current), false),
            (json!("stale"), false),
            (json!("unused"), false),
            (json!(""), false),
            (json!(null), false),
        ] {
            let temp = tempfile::tempdir().expect("tempdir");
            let path = temp.path().join("note.txt");
            std::fs::write(&path, "hello").expect("write");

            let result = EditTool
                .execute(
                    tool_context(temp.path()),
                    json!({
                        "path": "note.txt",
                        "old_string": "hello",
                        "new_string": "goodbye",
                        "expected_sha256": expected_sha256
                    }),
                )
                .await;

            let written = std::fs::read_to_string(&path).expect("read");
            match result {
                Err(error) => {
                    assert!(refused, "{expected_sha256}: {error}");
                    let error = error.to_string();
                    assert!(error.contains("read it again"), "{error}");
                    assert!(error.contains(&hash_text("hello")), "{error}");
                    assert_eq!(written, "hello");
                }
                Ok(_) => {
                    assert!(!refused, "{expected_sha256} should be refused");
                    assert_eq!(written, "goodbye");
                }
            }
        }
    }

    #[test]
    fn edit_output_renders_a_confirmation_with_the_new_hash() {
        let output = EditOutput {
            path: PathBuf::from("/w/a \"é\".rs"),
            occurrences_in_file: 3,
            replacements_applied: 2,
            file_hash_before: "old".to_owned(),
            file_hash_after: "new".to_owned(),
        };

        assert_eq!(
            output.to_model_string(),
            "Edited /w/a \"é\".rs (2 replaced); sha256=new"
        );
        assert_eq!(
            serde_json::to_value(&output).expect("serialize"),
            json!({
                "path": "/w/a \"é\".rs",
                "occurrences_in_file": 3,
                "replacements_applied": 2,
                "file_hash_before": "old",
                "file_hash_after": "new",
            })
        );
    }
}
