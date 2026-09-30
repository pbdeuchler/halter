// pattern: Imperative Shell

mod job;

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use halter_protocol::{
    SessionId, ToolCapabilities, ToolConcurrency, ToolName, ToolResult, ToolSpec,
};
use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::{CanonicalPath, Tool, ToolContext};

use super::common::{
    ToolScope, ensure_not_cancelled, optional_u64, parse_env_map, required_string, resolve_path,
};
use job::BackgroundJob;

/// Retained stdout and stderr bytes per job. Output cursors count original
/// bytes, including bytes already discarded from this bounded buffer.
const OUTPUT_CAPACITY: usize = 64 * 1024;
// Completed records remain queryable until the session closes. Bound those
// too, so repeated spawning cannot grow the session without limit.
const MAX_JOBS: usize = 64;

#[derive(Default)]
pub(crate) struct BackgroundRegistry {
    state: Mutex<RegistryState>,
}

#[derive(Default)]
struct RegistryState {
    closed: bool,
    jobs: HashMap<String, Arc<BackgroundJob>>,
}

impl BackgroundRegistry {
    async fn spawn(
        &self,
        context: &ToolContext,
        command: &str,
        cwd: CanonicalPath,
        env: Option<HashMap<String, String>>,
    ) -> anyhow::Result<Value> {
        let mut state = self.state.lock().await;
        ensure_not_cancelled(&context.cancel)?;
        anyhow::ensure!(
            !state.closed,
            "failed to spawn background job: session is closed"
        );
        anyhow::ensure!(
            state.jobs.len() < MAX_JOBS,
            "failed to spawn background job: session job limit ({MAX_JOBS}) reached"
        );
        let id = format!("bg-{}", SessionId::new().0);
        // Spawn and registration have no await between them. Once registered,
        // the session owns the process; the originating turn token no longer
        // controls it. Holding the registry lock also serializes shutdown.
        let job = BackgroundJob::spawn(id.clone(), command.to_owned(), cwd, env, OUTPUT_CAPACITY)?;
        let value = job.summary();
        state.jobs.insert(id, job);
        Ok(value)
    }

    async fn get(&self, id: &str) -> anyhow::Result<Arc<BackgroundJob>> {
        self.state
            .lock()
            .await
            .jobs
            .get(id)
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("failed to execute background tool: unknown job '{id}'"))
    }

    async fn list(&self) -> Value {
        let state = self.state.lock().await;
        let mut jobs: Vec<_> = state.jobs.values().collect();
        jobs.sort_by(|left, right| left.id.cmp(&right.id));
        json!({"jobs": jobs.into_iter().map(|job| job.summary()).collect::<Vec<_>>()})
    }

    pub(crate) async fn shutdown(&self) -> anyhow::Result<()> {
        let jobs = {
            let mut state = self.state.lock().await;
            state.closed = true;
            state.jobs.values().cloned().collect::<Vec<_>>()
        };
        // Signal all jobs first; shutdown latency does not multiply the TERM
        // grace period by the number of running jobs.
        for job in &jobs {
            job.request_stop();
        }
        let mut errors = Vec::new();
        for job in jobs {
            if let Err(error) = job.wait().await {
                errors.push(format!("{}: {error}", job.id));
            }
        }
        anyhow::ensure!(
            errors.is_empty(),
            "failed to shut down background jobs: {}",
            errors.join("; ")
        );
        Ok(())
    }
}

#[derive(Debug)]
/// Manage explicitly registered, session-owned background processes.
pub struct BackgroundTool;

#[async_trait]
impl Tool for BackgroundTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("background"),
            description: "Spawn, list, read output from, or stop managed background commands. Jobs survive interruption of agent execution and are terminated when the session closes. Commands use an independent shell with explicit cwd/env; persistent shell state is not inherited. Output is bounded and cursors count bytes of combined stdout/stderr.".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["spawn", "list", "output", "kill"]},
                    "command": {"type": "string", "minLength": 1},
                    "cwd": {"type": "string"},
                    "env": {"type": "object", "additionalProperties": {"type": "string"}},
                    "id": {"type": "string"},
                    "cursor": {"type": "integer", "minimum": 0}
                },
                "required": ["action"]
            }),
            concurrency: ToolConcurrency::Exclusive,
            capabilities: ToolCapabilities {mutating: true, requires_approval: true, cancellable: true, long_running: false},
            provider_aliases: Default::default(),
        }
    }

    async fn execute(&self, context: ToolContext, input: Value) -> anyhow::Result<ToolResult> {
        let _scope = ToolScope::new(&context, "background");
        ensure_not_cancelled(&context.cancel)?;
        let action = required_string(&input, "action")?;
        let registry = context
            .tool_sessions
            .background_session(&context.session_id);
        let value = match action {
            "spawn" => {
                let command = required_string(&input, "command")?;
                anyhow::ensure!(
                    !command.trim().is_empty(),
                    "invalid tool input: command must not be empty"
                );
                context
                    .policy
                    .check_shell_command_strict(command, context.policy.shell_mode())
                    .await?;
                let cwd = match input.get("cwd") {
                    None | Some(Value::Null) => context.working_dir.clone(),
                    Some(Value::String(cwd)) => resolve_path(&context.working_dir, cwd),
                    Some(_) => anyhow::bail!("invalid tool input: cwd must be a string"),
                };
                // The shell policy governs command execution, and the read
                // policy prevents starting in a denied/sensitive directory.
                let cwd = context.policy.check_read_path(&cwd, 0).await?;
                let env = parse_env_map(input.get("env"))?;
                if let Some(env) = &env {
                    for (key, value) in env {
                        anyhow::ensure!(
                            !key.is_empty() && !key.contains(['=', '\0']) && !value.contains('\0'),
                            "invalid tool input: invalid environment entry '{key}'"
                        );
                    }
                }
                registry.spawn(&context, command, cwd, env).await?
            }
            "list" => registry.list().await,
            "output" => {
                let job = registry.get(required_string(&input, "id")?).await?;
                job.output(optional_u64(&input, "cursor")?.unwrap_or(0))?
            }
            "kill" => {
                let job = registry.get(required_string(&input, "id")?).await?;
                // Opaque IDs only resolve within this session's registry.
                // There is no arbitrary-PID signalling API here.
                job.request_stop();
                job.wait().await?;
                job.summary()
            }
            _ => anyhow::bail!("failed to execute background tool: unknown action '{action}'"),
        };
        Ok(ToolResult::Json { value })
    }
}

#[cfg(all(test, unix))]
mod tests;
