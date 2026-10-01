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
    let session_id = SessionId::new();
    let tool_sessions = Arc::new(ToolSessionStore::default());
    tool_sessions.open_session(&session_id);
    ToolContext {
        session_id,
        working_dir: root.to_owned(),
        path_locks: Arc::new(PathLockMap::default()),
        tool_sessions,
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
async fn stale_background_spawn_is_denied_without_slots_and_reopen_is_fresh() {
    let root = tempfile::tempdir().unwrap();
    let context = context(root.path());
    let store = &context.tool_sessions;
    store.shutdown_session(&context.session_id).await.unwrap();
    let error = BackgroundTool
        .execute(
            context.clone(),
            json!({"action": "spawn", "command": "printf late"}),
        )
        .await
        .unwrap_err();
    assert!(error.to_string().contains("session is closed"));
    assert!(!store.has_process_state(&context.session_id));
    assert!(store.process_lifetime(&context.session_id).is_cancelled());
    store.open_session(&context.session_id);
    let job = execute(
        &context,
        json!({"action": "spawn", "command": "printf fresh"}),
    )
    .await;
    let output = wait_for(&context, job["id"].as_str().unwrap(), |output| {
        output["status"]["state"] == "exited"
    })
    .await;
    assert_eq!(output["output"], "fresh");
    store.shutdown_session(&context.session_id).await.unwrap();
    assert!(!store.has_process_state(&context.session_id));
}

#[tokio::test]
async fn activity_notifies_after_last_managed_job_and_output_finish() {
    let root = tempfile::tempdir().unwrap();
    let context = context(root.path());
    let store = &context.tool_sessions;
    let mut activity = store.subscribe_activity();
    let first = execute(&context, json!({"action": "spawn", "command": "sleep 30"})).await;
    let last = execute(&context, json!({"action": "spawn", "command": "sleep 30"})).await;
    assert!(store.has_running_jobs(&context.session_id));
    assert!(!store.has_running_jobs(&SessionId::new()));
    execute(&context, json!({"action": "kill", "id": first["id"]})).await;
    assert!(
        store.has_running_jobs(&context.session_id),
        "the other job remains owned"
    );
    activity.borrow_and_update();
    // Exit outside the tool API, so completion must wake an otherwise idle
    // observer directly from the monitor rather than from another tool call.
    super::super::process::kill_process_group(last["pid"].as_i64().unwrap() as i32, 15);
    tokio::time::timeout(Duration::from_secs(5), async {
        while store.has_running_jobs(&context.session_id) {
            activity.changed().await.unwrap();
        }
    })
    .await
    .expect("final process completion publishes activity");
    assert!(
        store.has_process_state(&context.session_id),
        "completed records remain retained"
    );
    let jobs = execute(&context, json!({"action": "list"})).await;
    assert!(
        jobs["jobs"]
            .as_array()
            .unwrap()
            .iter()
            .all(|job| job["status"]["state"] == "exited")
    );
    activity.borrow_and_update();
    assert!(
        tokio::time::timeout(Duration::from_millis(50), activity.changed())
            .await
            .is_err(),
        "no polling watcher keeps publishing after completion"
    );
    store.shutdown_session(&context.session_id).await.unwrap();
}

#[tokio::test]
async fn running_jobs_include_descendant_cleanup_after_leader_exit() {
    let root = tempfile::tempdir().unwrap();
    let context = context(root.path());
    let mut activity = context.tool_sessions.subscribe_activity();
    // Publish output before creating the inherited writer. This fixture tests
    // retaining buffered output while cleaning up a descendant, separately
    // from the shell's background-command startup ordering.
    let job = execute(
        &context,
        json!({"action": "spawn", "command": "printf done; sleep 30 & exit"}),
    )
    .await;
    assert!(context.tool_sessions.has_running_jobs(&context.session_id));
    tokio::time::timeout(Duration::from_secs(5), async {
        while context.tool_sessions.has_running_jobs(&context.session_id) {
            activity.changed().await.unwrap();
        }
    })
    .await
    .expect("process group and output drain finish without another tool call");
    let output = execute(&context, json!({"action": "output", "id": job["id"]})).await;
    assert_eq!(output["status"]["state"], "exited");
    assert_eq!(
        output["status"]["exit_code"], 0,
        "leader did not exit successfully: {output}"
    );
    assert!(
        output["status"]["signal"].is_null(),
        "leader received an unexpected signal: {output}"
    );
    assert_eq!(output["output"], "done", "complete output record: {output}");
    context
        .tool_sessions
        .shutdown_session(&context.session_id)
        .await
        .unwrap();
}

#[tokio::test]
async fn completed_job_records_do_not_limit_later_spawns() {
    let root = tempfile::tempdir().unwrap();
    let context = context(root.path());
    for _ in 0..65 {
        let job = execute(
            &context,
            json!({"action": "spawn", "command": "printf done"}),
        )
        .await;
        context
            .tool_sessions
            .background_session(&context.session_id)
            .get(job["id"].as_str().unwrap())
            .await
            .unwrap()
            .wait()
            .await
            .unwrap();
    }
    assert_eq!(
        execute(&context, json!({"action": "list"})).await["jobs"]
            .as_array()
            .unwrap()
            .len(),
        65
    );
    context
        .tool_sessions
        .shutdown_session(&context.session_id)
        .await
        .unwrap();
}

#[tokio::test]
async fn prune_releases_finished_records_and_output_while_preserving_running_jobs() {
    let root = tempfile::tempdir().unwrap();
    let context = context(root.path());
    let finished = execute(
        &context,
        json!({"action": "spawn", "command": "printf retained-output"}),
    )
    .await;
    let finished_id = finished["id"].as_str().unwrap();
    wait_for(&context, finished_id, |output| {
        output["status"]["state"] == "exited"
    })
    .await;
    let running = execute(&context, json!({"action": "spawn", "command": "sleep 30"})).await;
    let running_id = running["id"].as_str().unwrap();
    let pruned = execute(&context, json!({"action": "prune"})).await;
    assert_eq!(pruned["pruned"], json!([finished_id]));
    assert_eq!(pruned["remaining"], 1);
    let jobs = execute(&context, json!({"action": "list"})).await;
    assert_eq!(jobs["jobs"][0]["id"], running_id);
    assert!(
        BackgroundTool
            .execute(
                context.clone(),
                json!({"action": "output", "id": finished_id})
            )
            .await
            .unwrap_err()
            .to_string()
            .contains("unknown job")
    );
    execute(&context, json!({"action": "kill", "id": running_id})).await;
    context
        .tool_sessions
        .shutdown_session(&context.session_id)
        .await
        .unwrap();
}

#[tokio::test]
async fn aborting_foreground_shell_future_stops_its_worker_and_owned_descendants() {
    let root = tempfile::tempdir().unwrap();
    let context = context(root.path());
    let worker = tokio::spawn({
        let context = context.clone();
        async move {
            crate::ShellTool.execute(context, json!({"command": "sh -c 'trap \"\" TERM; sleep 30 & printf \"%s %s\" \"$$\" \"$!\" > owned-pids; wait'"})).await
        }
    });
    let pids = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if let Ok(pids) = tokio::fs::read_to_string(root.path().join("owned-pids")).await {
                let parsed = pids
                    .split_whitespace()
                    .map(str::parse::<i32>)
                    .collect::<Result<Vec<_>, _>>();
                if let Ok(pids) = parsed
                    && pids.len() == 2
                {
                    break pids;
                }
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("foreground children publish their pids");
    worker.abort();
    assert!(worker.await.unwrap_err().is_cancelled());
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            // SAFETY: probes only children created by this test.
            if pids.iter().all(|pid| unsafe { libc::kill(*pid, 0) } == -1) {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("aborted foreground worker kills and reaps its process group");
    context
        .tool_sessions
        .shutdown_session(&context.session_id)
        .await
        .unwrap();
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
