// pattern: Imperative Shell

use std::path::PathBuf;

use async_trait::async_trait;
use halter_protocol::{ToolCapabilities, ToolConcurrency, ToolName, ToolSpec};
use serde::Serialize;
use serde_json::{Value, json};
use tracing::debug;

use crate::{ToolContext, ToolOutput, TypedTool};

use super::common::{ToolScope, ensure_not_cancelled, required_string, resolve_path};

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
                asked."
                .to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string" },
                    "content": { "type": "string" }
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
        debug!(session_id = %context.session_id, path = %path.display(), bytes = content.len(), "writing file");

        let canonical = context.policy.check_write_path(&path).await?;
        let canonical_path = canonical.path().to_path_buf();
        let path_locks = context.path_locks.clone();
        let path_for_write = canonical_path.clone();
        let bytes = content.as_bytes().to_vec();
        tokio::task::spawn_blocking(move || {
            let _lock = path_locks.acquire_write(&path_for_write)?;
            canonical.atomic_write_blocking(&bytes)?;
            Ok::<_, anyhow::Error>(())
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
    use super::*;

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
