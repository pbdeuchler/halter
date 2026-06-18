// pattern: Imperative Shell
//
// `TaskTool` exposes a simple in-memory todo list to the model — create, list,
// and complete actions over a per-session set of tasks. The storage layer is
// abstracted behind `TaskStore` so a persistent backend (sqlite, file, etc.)
// can be slotted in without touching the tool surface.

use std::collections::BTreeMap;
use std::sync::Arc;

use async_trait::async_trait;
use halter_protocol::{
    SessionId, ToolCapabilities, ToolConcurrency, ToolName, ToolResult, ToolSpec,
};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{Tool, ToolContext};

use super::common::{
    ToolScope, ensure_not_cancelled, optional_string, optional_u64, required_string,
};

/// Status of a single task. Kept minimal on purpose: the tool only exposes
/// `create`, `list`, and `complete`, which maps to two states. Adding richer
/// states (in_progress, blocked, …) is a forward-compatible change because the
/// enum is `serde(rename_all = "snake_case")`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatus {
    Pending,
    Completed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
/// User-visible task tracked by the task tool.
pub struct Task {
    pub id: u64,
    pub subject: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub status: TaskStatus,
}

/// Per-session, ordered task list. Iteration order matches creation order
/// because `next_id` is monotonic and `BTreeMap` iterates by key.
#[derive(Debug, Default)]
pub struct TaskList {
    items: BTreeMap<u64, Task>,
    next_id: u64,
}

impl TaskList {
    /// Create a pending task.
    pub fn create(&mut self, subject: String, description: Option<String>) -> Task {
        self.next_id += 1;
        let task = Task {
            id: self.next_id,
            subject,
            description,
            status: TaskStatus::Pending,
        };
        self.items.insert(task.id, task.clone());
        task
    }

    /// Return all tasks in creation order.
    pub fn list(&self) -> Vec<Task> {
        self.items.values().cloned().collect()
    }

    /// Returns the updated task. Idempotent: completing an already-completed
    /// task is a no-op rather than an error so a retry from the model is safe.
    pub fn complete(&mut self, id: u64) -> anyhow::Result<Task> {
        let task = self
            .items
            .get_mut(&id)
            .ok_or_else(|| anyhow::anyhow!("failed to execute task tool: no task with id {id}"))?;
        task.status = TaskStatus::Completed;
        Ok(task.clone())
    }

    /// Count tasks by status.
    pub fn summary(&self) -> TaskSummary {
        let mut pending = 0u64;
        let mut completed = 0u64;
        for task in self.items.values() {
            match task.status {
                TaskStatus::Pending => pending += 1,
                TaskStatus::Completed => completed += 1,
            }
        }
        TaskSummary {
            total: u64::try_from(self.items.len()).unwrap_or(u64::MAX),
            pending,
            completed,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize)]
/// Aggregate task counts.
pub struct TaskSummary {
    pub total: u64,
    pub pending: u64,
    pub completed: u64,
}

/// Storage abstraction for the task tool. The default in-memory implementation
/// is wired in via `ToolSessionStore`, but a persistent backend (sqlite, file,
/// remote) can implement this trait without touching the tool itself.
pub trait TaskStore: Send + Sync {
    /// Return the task list for a session.
    fn task_list(&self, session_id: &SessionId) -> Arc<Mutex<TaskList>>;
}

/// Default backend used by `ToolSessionStore`: a `DashMap` keyed by session id.
/// Lifetime is the lifetime of the harness process; nothing is persisted.
#[derive(Default)]
pub struct InMemoryTaskStore {
    sessions: dashmap::DashMap<String, Arc<Mutex<TaskList>>>,
}

impl TaskStore for InMemoryTaskStore {
    fn task_list(&self, session_id: &SessionId) -> Arc<Mutex<TaskList>> {
        self.sessions
            .entry(session_id.0.clone())
            .or_insert_with(|| Arc::new(Mutex::new(TaskList::default())))
            .clone()
    }
}

#[derive(Debug)]
/// Built-in tool for a per-session task list.
pub struct TaskTool;

#[async_trait]
impl Tool for TaskTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("task"),
            description: "Manage an in-memory todo list scoped to the current session. \
                Pass action='create' with a subject (and optional description) to add a task, \
                action='list' to view all tasks, or action='complete' with the task id to mark \
                a task done."
                .to_owned(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["create", "list", "complete"]
                    },
                    "subject": {
                        "type": "string",
                        "description": "Required when action='create'."
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional details for action='create'."
                    },
                    "id": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Required when action='complete'."
                    }
                },
                "required": ["action"]
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
        let _scope = ToolScope::new(&context, "task");
        ensure_not_cancelled(&context.cancel)?;
        let action = required_string(&input, "action")?;
        let store = context.tool_sessions.task_session(&context.session_id);
        let response = match action {
            "create" => create_action(&store, &input)?,
            "list" => list_action(&store),
            "complete" => complete_action(&store, &input)?,
            other => anyhow::bail!(
                "invalid tool input: field 'action' must be one of 'create', 'list', 'complete' (got '{other}')"
            ),
        };
        Ok(ToolResult::Json { value: response })
    }
}

fn create_action(store: &Arc<Mutex<TaskList>>, input: &Value) -> anyhow::Result<Value> {
    let subject = required_string(input, "subject")?.trim();
    if subject.is_empty() {
        anyhow::bail!("invalid tool input: field 'subject' must not be empty");
    }
    let description = optional_string(input, "description")
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let mut list = store.lock();
    let task = list.create(subject.to_owned(), description);
    let summary = list.summary();
    Ok(json!({
        "task": task,
        "summary": summary,
    }))
}

fn list_action(store: &Arc<Mutex<TaskList>>) -> Value {
    let list = store.lock();
    json!({
        "tasks": list.list(),
        "summary": list.summary(),
    })
}

fn complete_action(store: &Arc<Mutex<TaskList>>, input: &Value) -> anyhow::Result<Value> {
    let id = optional_u64(input, "id")?
        .ok_or_else(|| anyhow::anyhow!("invalid tool input: missing integer field 'id'"))?;
    let mut list = store.lock();
    let task = list.complete(id)?;
    let summary = list.summary();
    Ok(json!({
        "task": task,
        "summary": summary,
    }))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    use crate::{
        DefaultToolPolicy, NoopToolEventSink, PathLockMap, PolicySettings, ToolPolicy,
        ToolSessionStore,
    };

    use super::*;

    fn tool_context(sessions: Arc<ToolSessionStore>) -> ToolContext {
        ToolContext {
            session_id: halter_protocol::SessionId::new(),
            working_dir: std::env::current_dir().expect("cwd"),
            path_locks: Arc::new(PathLockMap::default()),
            tool_sessions: sessions,
            snapshot: Arc::new(halter_protocol::ResourceSnapshot::empty()),
            cancel: CancellationToken::new(),
            emit: Arc::new(NoopToolEventSink),
            policy: Arc::new(DefaultToolPolicy::new(PolicySettings::default()))
                as Arc<dyn ToolPolicy>,
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
    async fn create_then_list_returns_pending_task() {
        let sessions = Arc::new(ToolSessionStore::default());
        let context = tool_context(sessions.clone());

        let created = json_value(
            TaskTool
                .execute(
                    context.clone(),
                    json!({ "action": "create", "subject": "Write the docs" }),
                )
                .await
                .expect("create"),
        );
        assert_eq!(created["task"]["id"], 1);
        assert_eq!(created["task"]["subject"], "Write the docs");
        assert_eq!(created["task"]["status"], "pending");
        assert!(created["task"]["description"].is_null());
        assert_eq!(created["summary"]["total"], 1);
        assert_eq!(created["summary"]["pending"], 1);
        assert_eq!(created["summary"]["completed"], 0);

        let listed = json_value(
            TaskTool
                .execute(context, json!({ "action": "list" }))
                .await
                .expect("list"),
        );
        let tasks = listed["tasks"].as_array().expect("tasks array");
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0]["id"], 1);
        assert_eq!(tasks[0]["status"], "pending");
    }

    #[tokio::test]
    async fn create_carries_optional_description() {
        let sessions = Arc::new(ToolSessionStore::default());
        let context = tool_context(sessions);

        let created = json_value(
            TaskTool
                .execute(
                    context,
                    json!({
                        "action": "create",
                        "subject": "Investigate",
                        "description": "Look at runtime.rs"
                    }),
                )
                .await
                .expect("create"),
        );
        assert_eq!(created["task"]["description"], "Look at runtime.rs");
    }

    #[tokio::test]
    async fn complete_marks_task_completed() {
        let sessions = Arc::new(ToolSessionStore::default());
        let context = tool_context(sessions);

        let _ = TaskTool
            .execute(
                context.clone(),
                json!({ "action": "create", "subject": "First" }),
            )
            .await
            .expect("create");
        let _ = TaskTool
            .execute(
                context.clone(),
                json!({ "action": "create", "subject": "Second" }),
            )
            .await
            .expect("create");

        let completed = json_value(
            TaskTool
                .execute(context.clone(), json!({ "action": "complete", "id": 1 }))
                .await
                .expect("complete"),
        );
        assert_eq!(completed["task"]["id"], 1);
        assert_eq!(completed["task"]["status"], "completed");
        assert_eq!(completed["summary"]["pending"], 1);
        assert_eq!(completed["summary"]["completed"], 1);

        let listed = json_value(
            TaskTool
                .execute(context, json!({ "action": "list" }))
                .await
                .expect("list"),
        );
        let tasks = listed["tasks"].as_array().expect("tasks array");
        assert_eq!(tasks.len(), 2);
        assert_eq!(tasks[0]["status"], "completed");
        assert_eq!(tasks[1]["status"], "pending");
    }

    #[tokio::test]
    async fn complete_is_idempotent() {
        let sessions = Arc::new(ToolSessionStore::default());
        let context = tool_context(sessions);

        TaskTool
            .execute(
                context.clone(),
                json!({ "action": "create", "subject": "Once" }),
            )
            .await
            .expect("create");
        TaskTool
            .execute(context.clone(), json!({ "action": "complete", "id": 1 }))
            .await
            .expect("complete first time");
        let second = json_value(
            TaskTool
                .execute(context, json!({ "action": "complete", "id": 1 }))
                .await
                .expect("complete second time"),
        );
        assert_eq!(second["task"]["status"], "completed");
        assert_eq!(second["summary"]["completed"], 1);
    }

    #[tokio::test]
    async fn create_rejects_missing_subject() {
        let sessions = Arc::new(ToolSessionStore::default());
        let context = tool_context(sessions);
        let error = TaskTool
            .execute(context, json!({ "action": "create" }))
            .await
            .expect_err("missing subject is rejected");
        assert!(error.to_string().contains("missing string field 'subject'"));
    }

    #[tokio::test]
    async fn create_rejects_empty_subject() {
        let sessions = Arc::new(ToolSessionStore::default());
        let context = tool_context(sessions);
        let error = TaskTool
            .execute(context, json!({ "action": "create", "subject": "   " }))
            .await
            .expect_err("empty subject is rejected");
        assert!(
            error
                .to_string()
                .contains("field 'subject' must not be empty")
        );
    }

    #[tokio::test]
    async fn complete_rejects_missing_id() {
        let sessions = Arc::new(ToolSessionStore::default());
        let context = tool_context(sessions);
        let error = TaskTool
            .execute(context, json!({ "action": "complete" }))
            .await
            .expect_err("missing id is rejected");
        assert!(error.to_string().contains("missing integer field 'id'"));
    }

    #[tokio::test]
    async fn complete_rejects_unknown_id() {
        let sessions = Arc::new(ToolSessionStore::default());
        let context = tool_context(sessions);
        let error = TaskTool
            .execute(context, json!({ "action": "complete", "id": 42 }))
            .await
            .expect_err("unknown id is rejected");
        assert!(error.to_string().contains("no task with id 42"));
    }

    #[tokio::test]
    async fn unknown_action_is_rejected() {
        let sessions = Arc::new(ToolSessionStore::default());
        let context = tool_context(sessions);
        let error = TaskTool
            .execute(context, json!({ "action": "delete", "id": 1 }))
            .await
            .expect_err("unknown action is rejected");
        assert!(
            error
                .to_string()
                .contains("must be one of 'create', 'list', 'complete'")
        );
    }

    #[tokio::test]
    async fn tasks_are_isolated_per_session() {
        let sessions = Arc::new(ToolSessionStore::default());
        let context_a = tool_context(sessions.clone());
        let context_b = tool_context(sessions);

        TaskTool
            .execute(
                context_a.clone(),
                json!({ "action": "create", "subject": "session A task" }),
            )
            .await
            .expect("create A");

        let listed_b = json_value(
            TaskTool
                .execute(context_b, json!({ "action": "list" }))
                .await
                .expect("list B"),
        );
        assert_eq!(listed_b["tasks"].as_array().expect("array").len(), 0);

        let listed_a = json_value(
            TaskTool
                .execute(context_a, json!({ "action": "list" }))
                .await
                .expect("list A"),
        );
        assert_eq!(listed_a["tasks"].as_array().expect("array").len(), 1);
    }
}
