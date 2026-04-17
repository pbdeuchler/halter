// pattern: Imperative Shell

use async_trait::async_trait;
use halter_protocol::{ToolCapabilities, ToolConcurrency, ToolName, ToolResult, ToolSpec};
use serde_json::{Value, json};
use tracing::debug;

use crate::{Tool, ToolContext};

use super::common::{
    ToolScope, atomic_write_blocking, ensure_not_cancelled, required_string, resolve_path,
};

#[derive(Debug)]
pub struct WriteTool;

#[async_trait]
impl Tool for WriteTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("write"),
            description: "Write a UTF-8 file to disk".to_owned(),
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

    async fn execute(&self, context: ToolContext, input: Value) -> anyhow::Result<ToolResult> {
        let _scope = ToolScope::new(&context, "write");
        ensure_not_cancelled(&context.cancel)?;
        let path = resolve_path(&context.working_dir, required_string(&input, "path")?);
        let content = required_string(&input, "content")?;
        debug!(session_id = %context.session_id, path = %path.display(), bytes = content.len(), "writing file");

        let canonical = context.policy.check_write_path(&path).await?;
        let canonical_path = canonical.into_path();
        let path_locks = context.path_locks.clone();
        let path_for_write = canonical_path.clone();
        let bytes = content.as_bytes().to_vec();
        tokio::task::spawn_blocking(move || {
            let _lock = path_locks.acquire_write(&path_for_write)?;
            atomic_write_blocking(&path_for_write, &bytes)
        })
        .await??;
        Ok(ToolResult::Json {
            value: json!({ "path": canonical_path }),
        })
    }
}
