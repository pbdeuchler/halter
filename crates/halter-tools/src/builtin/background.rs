// pattern: Imperative Shell

mod job;

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use halter_protocol::{SessionId, ToolCapabilities, ToolConcurrency, ToolName, ToolSpec};
use serde::Serialize;
use serde_json::{Value, json};
use tokio::sync::Mutex;

use crate::{CanonicalPath, ToolContext, ToolOutput, TypedTool};

use super::common::{
    ToolScope, ensure_not_cancelled, optional_u64, parse_env_map, required_string, resolve_path,
};
use job::{BackgroundJob, JobOutput, JobSummary};

/// Retained stdout and stderr bytes per job. Output cursors count original
/// bytes, including bytes already discarded from this bounded buffer.
const OUTPUT_CAPACITY: usize = 64 * 1024;

#[derive(Default)]
pub(crate) struct BackgroundRegistry {
    state: Mutex<RegistryState>,
    lifetime: tokio_util::sync::CancellationToken,
    running: Arc<AtomicUsize>,
    activity: tokio::sync::watch::Sender<()>,
}

#[derive(Default)]
struct RegistryState {
    closed: bool,
    jobs: HashMap<String, Arc<BackgroundJob>>,
}

impl BackgroundRegistry {
    pub(crate) fn new(
        lifetime: tokio_util::sync::CancellationToken,
        activity: tokio::sync::watch::Sender<()>,
    ) -> Self {
        Self {
            lifetime,
            activity,
            ..Self::default()
        }
    }
    pub(crate) fn has_running(&self) -> bool {
        self.running.load(Ordering::Acquire) != 0
    }
    async fn spawn(
        &self,
        context: &ToolContext,
        command: &str,
        cwd: CanonicalPath,
        env: Option<HashMap<String, String>>,
    ) -> anyhow::Result<JobSummary> {
        let mut state = self.state.lock().await;
        ensure_not_cancelled(&context.cancel)?;
        anyhow::ensure!(
            !state.closed && !self.lifetime.is_cancelled(),
            "failed to spawn background job: session is closed"
        );
        let id = format!("bg-{}", SessionId::new().0);
        // Spawn and registration have no await between them. Once registered,
        // the session owns the process; the originating turn token no longer
        // controls it. Holding the registry lock also serializes shutdown.
        let job = BackgroundJob::spawn(
            id.clone(),
            command.to_owned(),
            cwd,
            env,
            OUTPUT_CAPACITY,
            self.running.clone(),
            self.activity.clone(),
        )?;
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

    async fn list(&self) -> Vec<JobSummary> {
        let state = self.state.lock().await;
        let mut jobs: Vec<_> = state.jobs.values().collect();
        jobs.sort_by(|left, right| left.id.cmp(&right.id));
        jobs.into_iter().map(|job| job.summary()).collect()
    }

    async fn prune(&self) -> BackgroundOutput {
        let mut state = self.state.lock().await;
        let mut pruned = Vec::new();
        state.jobs.retain(|id, job| {
            if job.is_finished() {
                pruned.push(id.clone());
                false
            } else {
                true
            }
        });
        pruned.sort();
        BackgroundOutput::Pruned {
            pruned,
            remaining: state.jobs.len(),
        }
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

    pub(crate) async fn request_stop(&self, force: bool) {
        let mut state = self.state.lock().await;
        state.closed = true;
        for job in state.jobs.values() {
            if force {
                job.force_stop();
            } else {
                job.request_stop();
            }
        }
    }
}

#[derive(Debug)]
/// Manage explicitly registered, session-owned background processes.
pub struct BackgroundTool;

#[async_trait]
impl TypedTool for BackgroundTool {
    type Output = BackgroundOutput;

    fn describe(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("background"),
            description: "Spawn, list, read output from, stop, or prune finished managed background commands. Prune removes finished job records and retained output; running jobs are preserved. Jobs survive interruption of agent execution and are terminated when the session closes. Commands use an independent shell with explicit cwd/env; persistent shell state is not inherited. Output is bounded and cursors count bytes of combined stdout/stderr.".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["spawn", "list", "output", "kill", "prune"]},
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

    async fn run(&self, context: ToolContext, input: Value) -> anyhow::Result<BackgroundOutput> {
        let _scope = ToolScope::new(&context, "background");
        ensure_not_cancelled(&context.cancel)?;
        let action = required_string(&input, "action")?;
        let registry = context
            .tool_sessions
            .background_session(&context.session_id);
        Ok(match action {
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
                BackgroundOutput::Job(registry.spawn(&context, command, cwd, env).await?)
            }
            "list" => BackgroundOutput::Jobs {
                jobs: registry.list().await,
            },
            "prune" => registry.prune().await,
            "output" => {
                let job = registry.get(required_string(&input, "id")?).await?;
                BackgroundOutput::Output(job.output(optional_u64(&input, "cursor")?.unwrap_or(0))?)
            }
            "kill" => {
                let job = registry.get(required_string(&input, "id")?).await?;
                // Opaque IDs only resolve within this session's registry.
                // There is no arbitrary-PID signalling API here.
                job.request_stop();
                job.wait().await?;
                BackgroundOutput::Job(job.summary())
            }
            _ => anyhow::bail!("failed to execute background tool: unknown action '{action}'"),
        })
    }
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
/// Result of a `background` call.
pub enum BackgroundOutput {
    Job(JobSummary),
    Jobs {
        jobs: Vec<JobSummary>,
    },
    Pruned {
        pruned: Vec<String>,
        remaining: usize,
    },
    Output(JobOutput),
}

impl ToolOutput for BackgroundOutput {
    fn to_model_string(&self) -> String {
        match self {
            Self::Job(job) => job.render(),
            Self::Jobs { jobs } if jobs.is_empty() => "No background jobs".to_owned(),
            Self::Jobs { jobs } => {
                let jobs = jobs.iter().map(JobSummary::render).collect::<Vec<_>>();
                jobs.join("\n\n")
            }
            Self::Pruned { pruned, remaining } => format!(
                "Pruned {} finished jobs; {remaining} remain\n{}",
                pruned.len(),
                pruned.join("\n")
            )
            .trim_end()
            .to_owned(),
            Self::Output(output) => output.render(),
        }
    }
}

#[cfg(all(test, unix))]
mod tests;
