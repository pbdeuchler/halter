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
/// Built-in tool for running commands in a persistent shell session.
pub struct ShellTool;

#[async_trait]
impl Tool for ShellTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("shell"),
            description: "Run a command in a persistent shell session: the working directory, \
                environment, and shell state carry over between calls. Commands are subject to \
                the configured allowlist and timeout."
                .to_owned(),
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
            default_cwd: Some(context.working_dir.to_string_lossy().into_owned()),
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use halter_protocol::ToolResult;
    use serde_json::{Value, json};
    use tokio_util::sync::CancellationToken;

    use crate::{
        DefaultToolPolicy, NoopToolEventSink, PathLockMap, PolicySettings, ToolContext, ToolPolicy,
        ToolSessionStore,
    };

    use super::*;

    fn tool_context(root: &std::path::Path, allowed_shell_commands: Vec<String>) -> ToolContext {
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
                allowed_shell_commands,
                ..PolicySettings::default()
            })) as Arc<dyn ToolPolicy>,
            shell_timeout_secs: 30,
            subagent_parent: None,
        }
    }

    fn json_value(result: ToolResult) -> Value {
        match result {
            ToolResult::Json { value } => value,
            other => panic!("expected json result, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn shell_defaults_cwd_to_tool_context_working_dir() {
        let tool_cwd = tempfile::tempdir().expect("tool cwd tempdir");

        let value = json_value(
            ShellTool
                .execute(
                    tool_context(tool_cwd.path(), vec!["pwd".to_owned()]),
                    json!({
                        "command": "pwd",
                        "timeout_ms": 120_000
                    }),
                )
                .await
                .expect("shell command succeeds"),
        );

        assert_eq!(value["exit_code"], 0);
        assert_eq!(
            value["stdout"].as_str().expect("stdout string").trim(),
            tool_cwd.path().to_string_lossy()
        );
    }

    #[tokio::test]
    async fn shell_preserves_cwd_changes_after_default_initialization() {
        let tool_cwd = tempfile::tempdir().expect("tool cwd tempdir");
        let subdir = tool_cwd.path().join("nested");
        tokio::fs::create_dir_all(&subdir)
            .await
            .expect("create nested dir");
        let context = tool_context(tool_cwd.path(), vec!["cd".to_owned(), "pwd".to_owned()]);

        let cd_result = json_value(
            ShellTool
                .execute(
                    context.clone(),
                    json!({
                        "command": "cd nested",
                        "timeout_ms": 120_000
                    }),
                )
                .await
                .expect("cd succeeds"),
        );
        assert_eq!(cd_result["exit_code"], 0);

        let pwd_result = json_value(
            ShellTool
                .execute(
                    context,
                    json!({
                        "command": "pwd",
                        "timeout_ms": 120_000
                    }),
                )
                .await
                .expect("pwd succeeds"),
        );

        assert_eq!(pwd_result["exit_code"], 0);
        assert_eq!(
            pwd_result["stdout"].as_str().expect("stdout string").trim(),
            subdir.to_string_lossy()
        );
    }
}
