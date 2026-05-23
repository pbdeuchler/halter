// pattern: Imperative Shell

pub mod session;
pub mod streaming;

use std::time::Duration;

use async_trait::async_trait;
use halter_protocol::{ToolCapabilities, ToolConcurrency, ToolName, ToolResult, ToolSpec};
use serde_json::{Value, json};

use crate::{Tool, ToolContext};

use self::session::{ShellRunOptions, run_persistent_shell};
use super::common::{
    ToolScope, ensure_not_cancelled, optional_string, optional_u64, parse_env_map, required_string,
};

#[derive(Debug)]
pub struct ShellTool;

#[async_trait]
impl Tool for ShellTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("shell"),
            description: "Run a command in a persistent shell session".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string" },
                    "cwd": { "type": "string" },
                    "env": {
                        "type": "object",
                        "additionalProperties": { "type": "string" }
                    },
                    "timeout_ms": { "type": "integer", "minimum": 1 }
                },
                "required": ["command"],
            }),
            concurrency: ToolConcurrency::Exclusive,
            capabilities: ToolCapabilities {
                mutating: true,
                requires_approval: true,
                cancellable: true,
                long_running: true,
            },
            provider_aliases: Default::default(),
        }
    }

    async fn execute(&self, context: ToolContext, input: Value) -> anyhow::Result<ToolResult> {
        let _scope = ToolScope::new(&context, "shell");
        ensure_not_cancelled(&context.cancel)?;
        let timeout = optional_u64(&input, "timeout_ms")?
            .map(Duration::from_millis)
            .unwrap_or_else(|| Duration::from_secs(context.shell_timeout_secs.max(1)));
        let options = ShellRunOptions {
            command: required_string(&input, "command")?.to_owned(),
            cwd: optional_string(&input, "cwd").map(ToOwned::to_owned),
            env: parse_env_map(input.get("env"))?,
            timeout: Some(timeout),
        };
        let mode = context.policy.shell_mode();
        context
            .policy
            .check_shell_command_strict(&options.command, mode)
            .await?;
        let session = context.tool_sessions.shell_session(&context.session_id);
        let result = run_persistent_shell(
            session,
            options,
            context.emit.clone(),
            context.cancel.clone(),
        )
        .await?;

        Ok(ToolResult::Json {
            value: json!({
                "exit_code": result.exit_code,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "timed_out": result.timed_out,
                "cancelled": result.cancelled,
            }),
        })
    }
}
