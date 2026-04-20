// pattern: Imperative Shell

mod types;

use async_trait::async_trait;
use halter_protocol::{ToolCapabilities, ToolConcurrency, ToolName, ToolResult, ToolSpec};
use serde_json::{Value, json};

use crate::{Tool, ToolContext};

use super::common::{
    ToolScope, ensure_not_cancelled, optional_bool, optional_string, optional_u64, required_string,
};

#[cfg(feature = "advanced-tools")]
use self::types::run_advanced_search;
#[cfg(not(feature = "advanced-tools"))]
use self::types::run_basic_search;
use self::types::{DEFAULT_MAX_MATCHES, OutputMode, SearchConfig};

#[derive(Debug)]
pub struct GrepTool;

#[async_trait]
impl Tool for GrepTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("grep"),
            description: "Search file contents with regex filters and optional context".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "pattern": { "type": "string" },
                    "path": { "type": "string" },
                    "glob": { "type": "string" },
                    "type": { "type": "string" },
                    "ignore_case": { "type": "boolean" },
                    "multiline": { "type": "boolean" },
                    "context_before": { "type": "integer", "minimum": 0 },
                    "context_after": { "type": "integer", "minimum": 0 },
                    "max_matches": {
                        "type": "integer",
                        "minimum": 1,
                        "description": format!("Maximum number of matches to return (default: {DEFAULT_MAX_MATCHES})"),
                    },
                    "offset": { "type": "integer", "minimum": 0 },
                    "max_columns": { "type": "integer", "minimum": 1 },
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "count", "files_with_matches"]
                    }
                },
                "required": ["pattern"],
            }),
            concurrency: ToolConcurrency::ReadOnly,
            capabilities: ToolCapabilities {
                mutating: false,
                requires_approval: false,
                cancellable: true,
                long_running: true,
            },
            provider_aliases: Default::default(),
        }
    }

    async fn execute(&self, context: ToolContext, input: Value) -> anyhow::Result<ToolResult> {
        let _scope = ToolScope::new(&context, "grep");
        ensure_not_cancelled(&context.cancel)?;
        let pattern = required_string(&input, "pattern")?;
        if pattern.len() > 1000 {
            anyhow::bail!("failed to execute grep tool: pattern length exceeds 1000 characters");
        }

        let config = SearchConfig {
            pattern: pattern.to_owned(),
            path: optional_string(&input, "path")
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| context.working_dir.to_string_lossy().into_owned()),
            glob: optional_string(&input, "glob").map(ToOwned::to_owned),
            type_filter: optional_string(&input, "type").map(ToOwned::to_owned),
            ignore_case: optional_bool(&input, "ignore_case")?.unwrap_or(false),
            multiline: optional_bool(&input, "multiline")?.unwrap_or(false),
            context_before: optional_usize(&input, "context_before")?.unwrap_or(0),
            context_after: optional_usize(&input, "context_after")?.unwrap_or(0),
            max_matches: optional_u64(&input, "max_matches")?.unwrap_or(DEFAULT_MAX_MATCHES),
            offset: optional_u64(&input, "offset")?.unwrap_or(0),
            max_columns: optional_usize(&input, "max_columns")?,
            output_mode: OutputMode::from_str(optional_string(&input, "output_mode")),
        };

        let working_dir = context.working_dir.clone();
        let path_locks = context.path_locks.clone();
        let cancel = context.cancel.clone();
        let response = tokio::task::spawn_blocking(move || {
            #[cfg(feature = "advanced-tools")]
            {
                run_advanced_search(working_dir, path_locks, cancel, config)
            }

            #[cfg(not(feature = "advanced-tools"))]
            {
                run_basic_search(working_dir, path_locks, cancel, config)
            }
        })
        .await??;

        Ok(ToolResult::Json { value: response })
    }
}

fn optional_usize(input: &Value, key: &str) -> anyhow::Result<Option<usize>> {
    optional_u64(input, key)?
        .map(|value| {
            usize::try_from(value)
                .map_err(|_| anyhow::anyhow!("invalid tool input: field '{key}' is out of range"))
        })
        .transpose()
}
