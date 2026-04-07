// pattern: Imperative Shell

use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::Arc;

use async_trait::async_trait;
use globset::{GlobBuilder, GlobSet};
use halter_protocol::{ToolCapabilities, ToolConcurrency, ToolName, ToolResult, ToolSpec};
use ignore::WalkBuilder;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use tokio::fs;
use tokio::process::Command;
use tokio::time::{Duration, timeout};
use tokio_util::sync::CancellationToken;
use tracing::debug;

use crate::{Tool, ToolContext, ToolRuntime, ToolRuntimeEvent};

/// Registers the built-in tool set against a runtime that is already being assembled.
pub fn register_builtin_tools(runtime: &ToolRuntime, enabled: &[String]) {
    let register_all = enabled.is_empty();
    for tool in [
        Arc::new(ReadTool) as Arc<dyn Tool>,
        Arc::new(WriteTool),
        Arc::new(GlobTool),
        Arc::new(ShellTool),
    ] {
        let tool_name = tool.spec().name.0;
        if register_all || enabled.iter().any(|name| name == &tool_name) {
            runtime.register(tool);
        }
    }
}

#[derive(Debug)]
pub struct ReadTool;

#[async_trait]
impl Tool for ReadTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("read"),
            description: "Read a UTF-8 file from disk".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": { "path": { "type": "string" } },
                "required": ["path"],
            }),
            concurrency: ToolConcurrency::ReadOnly,
            capabilities: ToolCapabilities::default(),
            provider_aliases: Default::default(),
        }
    }

    async fn execute(&self, context: ToolContext, input: Value) -> anyhow::Result<ToolResult> {
        emit_started(&context, "read");
        ensure_not_cancelled(&context.cancel)?;
        let path = resolve_path(&context.working_dir, required_string(&input, "path")?);
        debug!(session_id = %context.session_id, path = %path.display(), "reading file");
        let metadata = fs::metadata(&path).await?;
        context
            .policy
            .check_read(&path, metadata.len() as usize)
            .await?;
        let text = fs::read_to_string(&path).await?;
        let response = json!({
            "path": path,
            "content": text,
            "sha256": hash_text(&text),
        });
        emit_completed(&context, "read");
        Ok(ToolResult::Json { value: response })
    }
}

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
        emit_started(&context, "write");
        ensure_not_cancelled(&context.cancel)?;
        let path = resolve_path(&context.working_dir, required_string(&input, "path")?);
        debug!(session_id = %context.session_id, path = %path.display(), "writing file");
        context.policy.check_write(&path).await?;
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).await?;
        }
        let content = required_string(&input, "content")?;
        fs::write(&path, content).await?;
        emit_completed(&context, "write");
        Ok(ToolResult::Empty)
    }
}

#[derive(Debug)]
pub struct GlobTool;

#[async_trait]
impl Tool for GlobTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("glob"),
            description: "Expand a glob pattern relative to the working directory".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": { "pattern": { "type": "string" } },
                "required": ["pattern"],
            }),
            concurrency: ToolConcurrency::ReadOnly,
            capabilities: ToolCapabilities::default(),
            provider_aliases: Default::default(),
        }
    }

    async fn execute(&self, context: ToolContext, input: Value) -> anyhow::Result<ToolResult> {
        emit_started(&context, "glob");
        ensure_not_cancelled(&context.cancel)?;
        let pattern = required_string(&input, "pattern")?;
        debug!(session_id = %context.session_id, pattern, "expanding glob pattern");
        let matcher = build_glob_matcher(pattern)?;
        let mut builder = WalkBuilder::new(&context.working_dir);
        builder
            .standard_filters(true)
            .require_git(false)
            .follow_links(false)
            .sort_by_file_path(|left, right| left.cmp(right));
        let mut matches = Vec::new();
        for entry in builder.build() {
            ensure_not_cancelled(&context.cancel)?;
            let entry = entry?;
            let path = entry.path();
            if path == context.working_dir {
                continue;
            }
            let relative = path.strip_prefix(&context.working_dir).unwrap_or(path);
            if !matcher.is_match(relative) {
                continue;
            }
            matches.push(path.to_string_lossy().to_string());
        }
        emit_completed(&context, "glob");
        Ok(ToolResult::Json {
            value: json!({ "matches": matches }),
        })
    }
}

#[derive(Debug)]
pub struct ShellTool;

#[async_trait]
impl Tool for ShellTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("shell"),
            description: "Run an allowlisted shell command".to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "program": { "type": "string" },
                    "args": { "type": "array", "items": { "type": "string" } }
                },
                "required": ["program"],
            }),
            concurrency: ToolConcurrency::Exclusive,
            capabilities: ToolCapabilities {
                mutating: false,
                requires_approval: true,
                cancellable: true,
                long_running: true,
            },
            provider_aliases: Default::default(),
        }
    }

    async fn execute(&self, context: ToolContext, input: Value) -> anyhow::Result<ToolResult> {
        emit_started(&context, "shell");
        ensure_not_cancelled(&context.cancel)?;
        let program = required_string(&input, "program")?;
        context.policy.check_shell(program).await?;
        let args = input
            .get("args")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(|value| {
                value
                    .as_str()
                    .map(ToOwned::to_owned)
                    .ok_or_else(|| anyhow::anyhow!("invalid shell args: expected strings"))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        debug!(
            session_id = %context.session_id,
            program,
            arg_count = args.len(),
            timeout_secs = context.shell_timeout_secs,
            "executing shell command"
        );

        let mut command = Command::new(program);
        command
            .args(args)
            .current_dir(&context.working_dir)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let output = timeout(
            Duration::from_secs(context.shell_timeout_secs),
            command.output(),
        )
        .await??;
        ensure_not_cancelled(&context.cancel)?;
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let rendered = format!("stdout:\n{stdout}\nstderr:\n{stderr}");
        if rendered.len() > context.max_tool_output_bytes {
            anyhow::bail!(
                "failed to execute shell tool: output {} bytes exceeds max_tool_output_bytes {}",
                rendered.len(),
                context.max_tool_output_bytes
            );
        }

        emit_completed(&context, "shell");
        Ok(ToolResult::Json {
            value: json!({
                "status": output.status.code(),
                "stdout": stdout,
                "stderr": stderr,
            }),
        })
    }
}

fn required_string<'a>(input: &'a Value, key: &str) -> anyhow::Result<&'a str> {
    input
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("invalid tool input: missing string field '{key}'"))
}

fn resolve_path(working_dir: &Path, path: &str) -> PathBuf {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() {
        candidate
    } else {
        working_dir.join(candidate)
    }
}

fn hash_text(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn build_glob_matcher(pattern: &str) -> anyhow::Result<GlobSet> {
    let glob = GlobBuilder::new(pattern)
        .literal_separator(false)
        .build()
        .map_err(|error| anyhow::anyhow!("invalid glob pattern '{pattern}': {error}"))?;
    let mut builder = globset::GlobSetBuilder::new();
    builder.add(glob);
    builder.build().map_err(Into::into)
}

fn ensure_not_cancelled(cancel: &CancellationToken) -> anyhow::Result<()> {
    if cancel.is_cancelled() {
        anyhow::bail!("failed to execute tool: cancelled");
    }
    Ok(())
}

fn emit_started(context: &ToolContext, tool_name: &str) {
    context.emit.emit(ToolRuntimeEvent::Started {
        tool_name: tool_name.to_owned(),
    });
}

fn emit_completed(context: &ToolContext, tool_name: &str) {
    context.emit.emit(ToolRuntimeEvent::Completed {
        tool_name: tool_name.to_owned(),
    });
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    use crate::{DefaultToolPolicy, NoopToolEventSink, PolicySettings, ToolPolicy};

    use super::*;

    #[tokio::test]
    async fn write_then_read_roundtrips_file() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime = ToolRuntime::new();
        register_builtin_tools(&runtime, &[]);
        let policy: Arc<dyn ToolPolicy> = Arc::new(DefaultToolPolicy::new(PolicySettings {
            allowed_write_roots: vec![temp.path().to_path_buf()],
            ..PolicySettings::default()
        }));
        let context = ToolContext {
            session_id: halter_protocol::SessionId::new(),
            working_dir: temp.path().to_path_buf(),
            file_view: Arc::new(Default::default()),
            snapshot: Arc::new(halter_protocol::ResourceSnapshot::empty()),
            cancel: CancellationToken::new(),
            emit: Arc::new(NoopToolEventSink),
            policy,
            max_tool_output_bytes: 16_384,
            shell_timeout_secs: 30,
            subagent_parent: None,
        };

        runtime
            .execute(
                "write",
                context.clone(),
                json!({ "path": "note.txt", "content": "hello" }),
            )
            .await
            .expect("write file");

        let result = runtime
            .execute("read", context, json!({ "path": "note.txt" }))
            .await
            .expect("read file");

        match result {
            ToolResult::Json { value } => {
                assert_eq!(value["content"], "hello");
            }
            ToolResult::Empty | ToolResult::Text { .. } => panic!("unexpected tool result"),
        }
    }

    #[test]
    fn builtin_registration_respects_enabled_list() {
        let runtime = ToolRuntime::new();
        register_builtin_tools(&runtime, &["read".to_owned(), "glob".to_owned()]);
        let specs = runtime.specs();

        assert!(specs.iter().any(|spec| spec.name.0 == "read"));
        assert!(specs.iter().any(|spec| spec.name.0 == "glob"));
        assert!(!specs.iter().any(|spec| spec.name.0 == "write"));
        assert!(!specs.iter().any(|spec| spec.name.0 == "shell"));
    }

    #[tokio::test]
    async fn write_tool_respects_policy_denials() {
        let temp = tempfile::tempdir().expect("tempdir");
        let policy: Arc<dyn ToolPolicy> = Arc::new(DefaultToolPolicy::new(PolicySettings {
            allowed_write_roots: vec![temp.path().join("allowed")],
            ..PolicySettings::default()
        }));
        let context = ToolContext {
            session_id: halter_protocol::SessionId::new(),
            working_dir: temp.path().to_path_buf(),
            file_view: Arc::new(Default::default()),
            snapshot: Arc::new(halter_protocol::ResourceSnapshot::empty()),
            cancel: CancellationToken::new(),
            emit: Arc::new(NoopToolEventSink),
            policy,
            max_tool_output_bytes: 16_384,
            shell_timeout_secs: 30,
            subagent_parent: None,
        };

        let error = WriteTool
            .execute(context, json!({ "path": "denied.txt", "content": "nope" }))
            .await
            .expect_err("write should be denied");

        assert!(error.to_string().contains("outside allowed_write_roots"));
    }

    #[tokio::test]
    async fn glob_tool_respects_gitignore() {
        let temp = tempfile::tempdir().expect("tempdir");
        fs::write(temp.path().join(".gitignore"), "ignored.txt\n")
            .await
            .expect("write gitignore");
        fs::write(temp.path().join("ignored.txt"), "ignored")
            .await
            .expect("write ignored file");
        fs::write(temp.path().join("visible.txt"), "visible")
            .await
            .expect("write visible file");
        let context = ToolContext {
            session_id: halter_protocol::SessionId::new(),
            working_dir: temp.path().to_path_buf(),
            file_view: Arc::new(Default::default()),
            snapshot: Arc::new(halter_protocol::ResourceSnapshot::empty()),
            cancel: CancellationToken::new(),
            emit: Arc::new(NoopToolEventSink),
            policy: Arc::new(DefaultToolPolicy::new(PolicySettings::default())),
            max_tool_output_bytes: 16_384,
            shell_timeout_secs: 30,
            subagent_parent: None,
        };

        let result = GlobTool
            .execute(context, json!({ "pattern": "*.txt" }))
            .await
            .expect("glob should succeed");

        let ToolResult::Json { value } = result else {
            panic!("expected json result");
        };
        let matches = value["matches"].as_array().expect("matches array");
        assert_eq!(matches.len(), 1);
        assert!(
            matches[0]
                .as_str()
                .expect("match path")
                .ends_with("visible.txt")
        );
    }
}
