// pattern: Imperative Shell

use std::path::PathBuf;

use async_trait::async_trait;
use halter_protocol::{ToolCapabilities, ToolConcurrency, ToolName, ToolSpec};
use serde::Serialize;
use serde_json::{Value, json};
use tracing::debug;

use crate::{ToolContext, ToolOutput, TypedTool};

use super::common::{
    ToolScope, ensure_not_cancelled, optional_bool, required_string, resolve_path,
};

#[derive(Debug)]
/// Built-in tool for atomically writing UTF-8 files.
pub struct WriteTool;

#[async_trait]
impl TypedTool for WriteTool {
    type Output = WriteOutput;

    fn describe(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("write"),
            description: "Write a UTF-8 file to disk, overwriting any existing file at `path` \
                with an atomic write. Prefer `edit` for changing an existing file — it only \
                touches the matched region. Use `write` for new files or full rewrites. Don't \
                create files that aren't needed, and don't add README or `*.md` docs unless \
                asked. The parent directory must exist unless `create_dirs` is true."
                .to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "content": { "type": "string" },
                    "create_dirs": {
                        "type": "boolean",
                        "description": "Create missing parent directories first (default false)."
                    }
                },
                "required": ["path", "content"],
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
        let _scope = ToolScope::new(&context, "write");
        ensure_not_cancelled(&context.cancel)?;
        let path = resolve_path(&context.working_dir, required_string(&input, "path")?);
        let content = required_string(&input, "content")?;
        let create_dirs = optional_bool(&input, "create_dirs")?.unwrap_or(false);
        debug!(session_id = %context.session_id, path = %path.display(), bytes = content.len(), create_dirs, "writing file");

        let policy = context.policy.clone();
        let path_locks = context.path_locks.clone();
        let bytes = content.as_bytes().to_vec();
        let canonical_path = tokio::task::spawn_blocking(move || {
            let missing: Vec<_> = path
                .ancestors()
                .skip(1)
                .take_while(|dir| !dir.exists())
                .collect();
            if let (false, Some(dir)) = (create_dirs, missing.last()) {
                anyhow::bail!(
                    "failed to execute write tool: directory '{}' does not exist; pass \
                     create_dirs=true to create the missing parent directories",
                    dir.display()
                );
            }
            // Shallowest first: the write policy authorizes each directory,
            // the deepest included, before it is created through its
            // parent's pinned fd.
            for dir in missing.into_iter().rev() {
                policy
                    .check_write_path_blocking(dir)?
                    .create_dir_blocking()?;
            }
            let canonical = policy.check_write_path_blocking(&path)?;
            let _lock = path_locks.acquire_write(canonical.path())?;
            canonical.atomic_write_blocking(&bytes)?;
            Ok::<_, anyhow::Error>(canonical.into_path())
        })
        .await??;
        Ok(WriteOutput {
            path: canonical_path,
            bytes: content.len(),
        })
    }
}

#[derive(Debug, Serialize)]
/// Result of a `write` call.
pub struct WriteOutput {
    path: PathBuf,
    #[serde(skip)]
    bytes: usize,
}

impl ToolOutput for WriteOutput {
    fn to_model_string(&self) -> String {
        format!("Wrote {} ({} bytes)", self.path.display(), self.bytes)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::{DefaultToolPolicy, NoopToolEventSink, PathLockMap, PolicySettings, ToolPolicy};

    /// `create_dirs` creates missing parents only when true, and each
    /// directory it creates passes the write policy first.
    #[tokio::test]
    async fn write_creates_missing_parents_only_when_asked_and_allowed() {
        for (path, create_dirs, error, absent) in [
            ("f.txt", json!(null), None, None),
            ("a/b/f.txt", json!(true), None, None),
            (
                "a/b/f.txt",
                json!(false),
                Some("pass create_dirs=true"),
                Some("a"),
            ),
            ("a/b/f.txt", json!(""), Some("directory '"), Some("a")),
            (
                "a/secret/b/f.txt",
                json!(true),
                Some("sensitive-path"),
                Some("a/secret"),
            ),
            (
                "../outside/f.txt",
                json!(true),
                Some("outside"),
                Some("../outside"),
            ),
        ] {
            let temp = tempfile::tempdir().expect("tempdir");
            let root = temp.path().join("root");
            std::fs::create_dir(&root).expect("root");
            let context = ToolContext {
                session_id: halter_protocol::SessionId::new(),
                working_dir: root.clone(),
                path_locks: Arc::new(PathLockMap::default()),
                tool_sessions: Arc::new(crate::ToolSessionStore::default()),
                snapshot: Arc::new(halter_protocol::ResourceSnapshot::empty()),
                cancel: CancellationToken::new(),
                emit: Arc::new(NoopToolEventSink),
                policy: Arc::new(DefaultToolPolicy::new(PolicySettings {
                    allowed_write_roots: vec![root.clone()],
                    sensitive_path_patterns: vec!["**/secret".to_owned()],
                    ..PolicySettings::default()
                })) as Arc<dyn ToolPolicy>,
                shell_timeout_secs: 30,
                subagent_parent: None,
            };

            let result = WriteTool
                .run(
                    context,
                    json!({"path": path, "content": "hi", "create_dirs": create_dirs}),
                )
                .await;

            match (result, error) {
                (Ok(_), None) => {
                    let written = std::fs::read_to_string(root.join(path)).expect("written");
                    assert_eq!(written, "hi", "{path}");
                }
                (Err(got), Some(want)) => {
                    let got = format!("{got:#}");
                    assert!(got.contains(want), "{path} {create_dirs}: {got}");
                }
                (result, _) => panic!("{path} {create_dirs}: unexpected {result:?}"),
            }
            if let Some(absent) = absent {
                assert!(!root.join(absent).exists(), "{path}: created {absent}");
            }
        }
    }

    #[test]
    fn write_output_renders_a_confirmation_and_keeps_the_json_shape() {
        let output = WriteOutput {
            path: PathBuf::from("/work/a \"b\".txt"),
            bytes: 12,
        };

        assert_eq!(
            output.to_model_string(),
            "Wrote /work/a \"b\".txt (12 bytes)"
        );
        assert_eq!(
            serde_json::to_value(&output).expect("serialize"),
            json!({ "path": "/work/a \"b\".txt" })
        );
    }
}
