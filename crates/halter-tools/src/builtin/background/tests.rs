// pattern: Imperative Shell

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;

use super::*;
use crate::{
    DefaultToolPolicy, NoopToolEventSink, PathLockMap, PolicySettings, ToolPolicy, ToolSessionStore,
};

fn context(root: &Path) -> ToolContext {
    ToolContext {
        session_id: SessionId::new(),
        working_dir: root.to_owned(),
        path_locks: Arc::new(PathLockMap::default()),
        tool_sessions: Arc::new(ToolSessionStore::default()),
        snapshot: Arc::new(halter_protocol::ResourceSnapshot::empty()),
        cancel: CancellationToken::new(),
        emit: Arc::new(NoopToolEventSink),
        policy: Arc::new(DefaultToolPolicy::new(PolicySettings {
            allowed_read_roots: vec![root.to_owned()],
            allowed_shell_commands: [
                "printf", "sleep", "trap", "wait", "head", "exit", "pwd", "sh",
            ]
            .into_iter()
            .map(str::to_owned)
            .collect(),
            ..PolicySettings::default()
        })),
        shell_timeout_secs: 30,
        subagent_parent: None,
    }
}

async fn execute(context: &ToolContext, input: Value) -> Value {
    let ToolResult::Json { value } = BackgroundTool
        .execute(context.clone(), input)
        .await
        .expect("background operation")
    else {
        panic!("expected json")
    };
    value
}

async fn wait_for(context: &ToolContext, id: &str, condition: impl Fn(&Value) -> bool) -> Value {
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let output = execute(context, json!({"action": "output", "id": id})).await;
            if condition(&output) {
                return output;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("job reaches expected state")
}

fn assert_reaped(pid: i32) {
    // SAFETY: signal zero only probes the existence of a PID created by this test.
    let exists = unsafe { libc::kill(pid, 0) };
    assert_eq!(exists, -1, "process {pid} is still alive");
    assert_eq!(
        std::io::Error::last_os_error().raw_os_error(),
        Some(libc::ESRCH)
    );
}

#[tokio::test]
async fn session_shutdown_reaps_shell_jobs_and_preserves_task_list() {
    let root = tempfile::tempdir().unwrap();
    let context = context(root.path());
    context
        .tool_sessions
        .task_session(&context.session_id)
        .lock()
        .create("keep me".to_owned(), None);
    crate::ShellTool
        .execute(
            context.clone(),
            json!({"command": "sh -c 'printf %s \"$$\" > pid; sleep 30' &"}),
        )
        .await
        .unwrap();
    let pid: i32 = tokio::fs::read_to_string(root.path().join("pid"))
        .await
        .unwrap()
        .parse()
        .unwrap();
    // SAFETY: signal zero only probes the owned test child without mutation.
    let exists = unsafe { libc::kill(pid, 0) };
    assert_eq!(exists, 0);
    context
        .tool_sessions
        .shutdown_session(&context.session_id)
        .await
        .unwrap();
    assert_reaped(pid);
    assert_eq!(
        context
            .tool_sessions
            .task_session(&context.session_id)
            .lock()
            .list()[0]
            .subject,
        "keep me"
    );
    assert!(!context.tool_sessions.has_process_state(&context.session_id));
}

#[tokio::test]
async fn registered_job_survives_originating_cancellation_and_shutdown_reaps_it() {
    let root = tempfile::tempdir().unwrap();
    let mut context = context(root.path());
    let job = execute(
        &context,
        json!({"action": "spawn", "command": "trap '' TERM; sleep 30 & printf ready; wait"}),
    )
    .await;
    let id = job["id"].as_str().unwrap();
    wait_for(&context, id, |output| {
        output["output"].as_str().unwrap().contains("ready")
    })
    .await;
    context.cancel.cancel();
    context.cancel = CancellationToken::new();
    let output = execute(&context, json!({"action": "output", "id": id})).await;
    assert_eq!(output["status"]["state"], "running");
    assert_eq!(
        execute(&context, json!({"action": "list"})).await["jobs"]
            .as_array()
            .unwrap()
            .len(),
        1
    );
    // Retain only a cleanup/status handle: spawning, observing the running
    // job, and shutting down still exercise the public tool/store behavior.
    let tracked = context
        .tool_sessions
        .background_session(&context.session_id)
        .get(id)
        .await
        .unwrap();
    match tokio::time::timeout(
        Duration::from_secs(5),
        context.tool_sessions.shutdown_session(&context.session_id),
    )
    .await
    {
        Ok(result) => result.unwrap(),
        Err(_) => {
            tracked.request_stop();
            tracked.wait().await.expect("cleanup of the test-owned job");
            panic!("session shutdown must terminate its job before natural completion");
        }
    }
    assert_eq!(tracked.summary()["status"]["state"], "exited");
    assert_eq!(tracked.summary()["status"]["signal"], libc::SIGKILL);
    assert_reaped(job["pid"].as_i64().unwrap() as i32);
    assert!(!context.tool_sessions.has_process_state(&context.session_id));
    // Reopening receives an empty registry rather than a closed registry.
    assert!(
        execute(&context, json!({"action": "list"})).await["jobs"]
            .as_array()
            .unwrap()
            .is_empty()
    );
}

#[tokio::test]
async fn kill_waits_for_stubborn_process_group_and_is_idempotent() {
    let root = tempfile::tempdir().unwrap();
    let context = context(root.path());
    let job = execute(
        &context,
        json!({"action": "spawn", "command": "trap '' TERM; sleep 30 & printf '%s' \"$!\"; wait"}),
    )
    .await;
    let id = job["id"].as_str().unwrap();
    let output = wait_for(&context, id, |output| {
        !output["output"].as_str().unwrap().is_empty()
    })
    .await;
    let descendant: i32 = output["output"].as_str().unwrap().parse().unwrap();
    let stopped = execute(&context, json!({"action": "kill", "id": id})).await;
    assert_eq!(stopped["status"]["state"], "exited");
    assert_eq!(stopped["status"]["signal"], 9);
    assert_reaped(job["pid"].as_i64().unwrap() as i32);
    // Descendants are reparented and may briefly be zombies owned by init;
    // wait for actual process disappearance rather than guessing a delay.
    tokio::time::timeout(Duration::from_secs(5), async {
        // SAFETY: signal zero only probes the owned test descendant.
        while unsafe { libc::kill(descendant, 0) } == 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("descendant is reaped by its parent/init");
    assert_reaped(descendant);
    assert_eq!(
        execute(&context, json!({"action": "kill", "id": id})).await["status"],
        stopped["status"]
    );
    context
        .tool_sessions
        .shutdown_session(&context.session_id)
        .await
        .unwrap();
}

#[tokio::test]
async fn output_is_retained_bounded_and_cursor_addressed_after_exit() {
    let root = tempfile::tempdir().unwrap();
    let context = context(root.path());
    let job = execute(&context, json!({"action": "spawn", "command": "head -c 70000 /dev/zero; printf tail; printf err >&2; exit 7"})).await;
    let id = job["id"].as_str().unwrap();
    let output = wait_for(&context, id, |output| output["status"]["state"] == "exited").await;
    assert_eq!(output["status"]["exit_code"], 7);
    assert_eq!(output["next_cursor"], 70_007);
    assert_eq!(output["start_cursor"], 70_007 - OUTPUT_CAPACITY);
    assert_eq!(output["truncated"], true);
    assert_eq!(output["output"].as_str().unwrap().len(), OUTPUT_CAPACITY);
    assert!(output["output"].as_str().unwrap().contains("tail"));
    let cursor = output["next_cursor"].as_u64().unwrap();
    let empty = execute(
        &context,
        json!({"action": "output", "id": id, "cursor": cursor}),
    )
    .await;
    assert_eq!(empty["output"], "");
    assert_eq!(empty["truncated"], false);
    assert!(
        BackgroundTool
            .execute(
                context.clone(),
                json!({"action": "output", "id": id, "cursor": cursor + 1})
            )
            .await
            .is_err()
    );
    context
        .tool_sessions
        .shutdown_session(&context.session_id)
        .await
        .unwrap();
}

#[tokio::test]
async fn spawn_preserves_explicit_cwd_and_env_and_checks_policy() {
    let root = tempfile::tempdir().unwrap();
    let context = context(root.path());
    std::fs::create_dir(root.path().join("nested")).unwrap();
    let job = execute(&context, json!({"action": "spawn", "command": "pwd; printf '%s' \"$CUSTOM\"", "cwd": "nested", "env": {"CUSTOM": "value"}})).await;
    let id = job["id"].as_str().unwrap();
    let output = wait_for(&context, id, |output| output["status"]["state"] == "exited").await;
    assert_eq!(
        output["output"],
        format!(
            "{}\nvalue",
            std::fs::canonicalize(root.path().join("nested"))
                .unwrap()
                .display()
        )
    );
    for input in [
        json!({"action": "spawn", "command": "printf forbidden", "cwd": "/etc"}),
        json!({"action": "spawn", "command": "printf forbidden", "env": {"BAD=KEY": "value"}}),
        json!({"action": "spawn", "command": "rm missing"}),
        json!({"action": "spawn", "command": ""}),
    ] {
        assert!(
            BackgroundTool
                .execute(context.clone(), input)
                .await
                .is_err()
        );
    }
    let mut disabled = context.clone();
    disabled.policy = Arc::new(DefaultToolPolicy::new(PolicySettings {
        shell_enabled: false,
        ..PolicySettings::default()
    })) as Arc<dyn ToolPolicy>;
    assert!(
        BackgroundTool
            .execute(
                disabled,
                json!({"action": "spawn", "command": "printf denied"})
            )
            .await
            .unwrap_err()
            .to_string()
            .contains("disabled")
    );
    let mut cancelled = context.clone();
    cancelled.cancel = CancellationToken::new();
    cancelled.cancel.cancel();
    assert!(
        BackgroundTool
            .execute(
                cancelled,
                json!({"action": "spawn", "command": "printf denied"})
            )
            .await
            .is_err()
    );
    let mut other = context.clone();
    other.session_id = SessionId::new();
    assert!(
        BackgroundTool
            .execute(other, json!({"action": "kill", "id": id}))
            .await
            .is_err()
    );
    context
        .tool_sessions
        .shutdown_session(&context.session_id)
        .await
        .unwrap();
}
