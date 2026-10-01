// pattern: Imperative Shell

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use anyhow::Context;
use async_trait::async_trait;
use futures::TryStreamExt;
use halter_protocol::{
    AgentId, AgentName, CloseSubagentRequest, CloseSubagentResponse, SendSubagentInputRequest,
    SessionId, SpawnSubagentRequest, SubagentRecord, SubagentState, SubagentStatus, Turn, TurnId,
    Usage, WaitSubagentRequest, WaitSubagentResponse,
};
use halter_tools::{SubagentControl, SubagentParentContext};
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio::time::timeout;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};

use crate::session::create_session_seeded;
use crate::session_lease::OutOfTurn;
use crate::subagent_session::{
    build_subagent_session_init, build_subagent_state, extract_subagent_output,
    extract_subagent_usage,
};
use crate::{
    HookInvocationContext, RuntimeServices, SessionExecutor, run_subagent_start, run_subagent_stop,
};

#[derive(Clone)]
pub struct RuntimeSubagentControl {
    inner: Arc<RuntimeSubagentState>,
}

struct RuntimeSubagentState {
    services: Arc<RuntimeServices>,
    registry: Mutex<SubagentRegistry>,
    activity: watch::Sender<()>,
}

impl RuntimeSubagentState {
    fn registry(&self) -> MutexGuard<'_, SubagentRegistry> {
        self.registry
            .lock()
            .unwrap_or_else(|error| error.into_inner())
    }
}

#[derive(Default)]
struct SubagentRegistry {
    entries: HashMap<String, RegisteredSubagent>,
    closing_sessions: HashMap<SessionId, usize>,
    closing_turns: HashMap<SessionId, ClosingTurn>,
}

#[derive(Clone)]
struct ClosingTurn {
    wrapper: Option<tokio::task::AbortHandle>,
    current_turn: Arc<std::sync::Mutex<Option<TurnId>>>,
    settled: CancellationToken,
    failure: Arc<std::sync::Mutex<Option<Arc<anyhow::Error>>>>,
}

impl ClosingTurn {
    fn new(running: Option<&RunningTurn>) -> Self {
        Self {
            wrapper: running.map(|task| task.join_handle.abort_handle()),
            current_turn: running.map_or_else(
                || Arc::new(std::sync::Mutex::new(None)),
                |task| task.current_turn.clone(),
            ),
            settled: CancellationToken::new(),
            failure: Default::default(),
        }
    }

    fn finish(&self, failure: Option<Arc<anyhow::Error>>) {
        *self
            .failure
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = failure;
        self.settled.cancel();
    }

    async fn result(&self) -> anyhow::Result<()> {
        self.settled.cancelled().await;
        match self
            .failure
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
        {
            Some(error) => Err(anyhow::Error::new(SharedCloseError(error))),
            None => Ok(()),
        }
    }
}

#[derive(Debug)]
struct SharedCloseError(Arc<anyhow::Error>);

impl std::fmt::Display for SharedCloseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self.0.as_ref(), f)
    }
}

impl std::error::Error for SharedCloseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(self.0.as_ref().as_ref())
    }
}

struct RegisteredSubagent {
    /// The session whose log records this agent.
    parent: SessionId,
    status: SubagentStatus,
    /// Zero is an unpublished reservation; the first turn publishes generation one.
    generation: u64,
    running: Option<RunningTurn>,
    /// Completion of this incarnation's closure, retained for repeated close.
    closure: Option<ClosingTurn>,
}

struct RunningTurn {
    cancel: CancellationToken,
    join_handle: JoinHandle<()>,
    current_turn: Arc<std::sync::Mutex<Option<TurnId>>>,
}

/// A dropped spawn still owns its reserved child until cleanup finishes.
struct SpawnReservation {
    controller: RuntimeSubagentControl,
    agent_id: Option<AgentId>,
}

impl SpawnReservation {
    async fn cleanup(&mut self) {
        if let Some(agent_id) = self.agent_id.take() {
            self.controller.remove_reserved_subagent(&agent_id).await;
        }
    }
}

impl Drop for SpawnReservation {
    fn drop(&mut self) {
        if let Some(agent_id) = self.agent_id.take() {
            let controller = self.controller.clone();
            tokio::spawn(async move {
                controller.remove_reserved_subagent(&agent_id).await;
            });
        }
    }
}

/// Unexpected wrapper abortion must settle the actual executor before the
/// parent becomes idle. Normal closure takes over that ownership separately.
struct TurnTaskGuard {
    controller: RuntimeSubagentControl,
    agent_id: AgentId,
    session_id: SessionId,
    generation: u64,
    current_turn: Arc<std::sync::Mutex<Option<TurnId>>>,
    armed: bool,
}

impl TurnTaskGuard {
    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for TurnTaskGuard {
    fn drop(&mut self) {
        if self.armed {
            let controller = self.controller.clone();
            let agent_id = self.agent_id.clone();
            let session_id = self.session_id.clone();
            let generation = self.generation;
            let turn_id = self
                .current_turn
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .clone();
            tokio::spawn(async move {
                controller
                    .settle_aborted_turn(agent_id, session_id, generation, turn_id)
                    .await;
            });
        }
    }
}

struct TurnOutcome {
    state: SubagentState,
    last_message: Option<String>,
    usage: Option<Usage>,
    error: Option<String>,
}

/// Upper bound on `SubagentStop`-hook-driven turn resubmissions per subagent
/// task. A hook that returns a `block_reason` resubmits the blocked turn with
/// that reason as input; a hook that *always* blocks would otherwise loop
/// forever (each resubmission is a full provider turn). Tripping the cap
/// fails the subagent with a descriptive error instead.
const SUBAGENT_STOP_RESUBMISSION_CAP: u32 = 8;

fn active_subagent_count(registry: &SubagentRegistry) -> usize {
    registry
        .entries
        .values()
        .filter(|entry| {
            entry.running.is_some() || matches!(entry.status.state, SubagentState::Running)
        })
        .count()
}

/// Historical closed entries remain queryable but no longer own an edge in
/// the live session tree. An unfinished closure retains that edge's fence.
fn owned_session_tree(registry: &SubagentRegistry, root: &SessionId) -> HashSet<SessionId> {
    let mut sessions = HashSet::from([root.clone()]);
    loop {
        let descendants = registry
            .entries
            .values()
            .filter(|entry| {
                sessions.contains(&entry.parent)
                    && (entry.status.state != SubagentState::Closed
                        || registry
                            .closing_turns
                            .contains_key(&entry.status.session_id)
                        || registry
                            .closing_sessions
                            .contains_key(&entry.status.session_id))
            })
            .map(|entry| entry.status.session_id.clone())
            .collect::<Vec<_>>();
        let before = sessions.len();
        sessions.extend(descendants);
        if sessions.len() == before {
            return sessions;
        }
    }
}

impl RuntimeSubagentControl {
    #[must_use]
    pub fn new(services: Arc<RuntimeServices>) -> Self {
        Self {
            inner: Arc::new(RuntimeSubagentState {
                services,
                registry: Mutex::new(SubagentRegistry::default()),
                activity: watch::channel(()).0,
            }),
        }
    }

    /// Coalesced notifications after subagent execution or cleanup changes.
    /// Subscribe before querying activity so a concurrent finish is retained.
    pub(crate) fn subscribe_activity(&self) -> watch::Receiver<()> {
        self.inner.activity.subscribe()
    }

    /// Execution and cleanup in any live descendant keep its parent alive.
    pub(crate) fn has_running_subagents(&self, session_id: &SessionId) -> bool {
        let sessions = {
            let registry = self.inner.registry();
            let sessions = owned_session_tree(&registry, session_id);
            if sessions.iter().any(|session| {
                registry.closing_sessions.contains_key(session)
                    || registry.closing_turns.contains_key(session)
            }) || registry.entries.values().any(|entry| {
                sessions.contains(&entry.parent)
                    && (entry.running.is_some() || entry.status.state == SubagentState::Running)
            }) {
                return true;
            }
            sessions
        };
        // Never hold the subagent lock while acquiring a tool-store lock.
        sessions
            .iter()
            .filter(|child| *child != session_id)
            .any(|child| self.inner.services.tool_sessions.has_running_jobs(child))
    }

    /// Ids of every agent this process knows, whatever its parent.
    pub(crate) async fn agent_ids(&self) -> HashSet<AgentId> {
        let registry = self.inner.registry();
        registry
            .entries
            .values()
            .map(|entry| entry.status.agent_id.clone())
            .collect()
    }

    pub(crate) async fn owns_session(&self, session_id: &SessionId) -> bool {
        let registry = self.inner.registry();
        registry.closing_sessions.contains_key(session_id)
            || registry.closing_turns.contains_key(session_id)
            || registry.entries.values().any(|entry| {
                &entry.status.session_id == session_id
                    && entry.status.state != SubagentState::Closed
            })
    }

    /// Stop and settle every descendant owned by this session before cleaning
    /// up the descendants' persistent tool resources.
    #[cfg(test)]
    pub(crate) async fn close_session(&self, session_id: &SessionId) -> anyhow::Result<()> {
        self.close_session_with_deadline(session_id, None).await
    }

    pub(crate) async fn close_session_with_deadline(
        &self,
        session_id: &SessionId,
        deadline: Option<tokio::time::Instant>,
    ) -> anyhow::Result<()> {
        let controller = self.clone();
        let session_id = session_id.clone();
        // Dropping a caller's await cannot abandon descendant joins or release
        // their incarnation fences. This task owns cleanup through settlement.
        tokio::spawn(async move {
            let cleanup = controller.settle_session_descendants(&session_id, deadline);
            tokio::pin!(cleanup);
            match deadline {
                Some(deadline) => match tokio::time::timeout_at(deadline, &mut cleanup).await {
                    Ok(result) => result,
                    Err(_) => {
                        // Escalate during repair, hooks, or resource reaping,
                        // and retain the same cleanup until it settles.
                        controller.force_close_session(&session_id).await;
                        cleanup.await
                    }
                },
                None => cleanup.await,
            }
        })
        .await
        .context("failed to settle descendant cleanup")?
    }

    async fn settle_session_descendants(
        &self,
        session_id: &SessionId,
        deadline: Option<tokio::time::Instant>,
    ) -> anyhow::Result<()> {
        let (sessions, mut running, records, closures, previous_closing) = {
            let mut registry = self.inner.registry();
            let sessions = owned_session_tree(&registry, session_id);
            for session in &sessions {
                *registry
                    .closing_sessions
                    .entry(session.clone())
                    .or_default() += 1;
            }
            let previous_closing = registry
                .closing_turns
                .iter()
                .filter(|(child, _)| *child != session_id && sessions.contains(*child))
                .map(|(_, task)| task.clone())
                .collect::<Vec<_>>();
            let mut running = Vec::new();
            let mut records = Vec::new();
            let mut closures = Vec::new();
            for entry in registry.entries.values_mut().filter(|entry| {
                sessions.contains(&entry.parent) && entry.status.state != SubagentState::Closed
            }) {
                let published = entry.generation != 0;
                entry.generation = entry.generation.saturating_add(1);
                let task = entry.running.take();
                let closure = ClosingTurn::new(task.as_ref());
                entry.closure = Some(closure.clone());
                closures.push((entry.status.session_id.clone(), closure));
                if let Some(task) = task {
                    task.cancel.cancel();
                    running.push((entry.status.session_id.clone(), task));
                }
                entry.status.state = SubagentState::Closed;
                entry.status.error = Some("closed by session shutdown".to_owned());
                if published {
                    records.push((entry.parent.clone(), entry.status.clone(), entry.generation));
                }
            }
            for (child, closure) in &closures {
                registry
                    .closing_turns
                    .insert(child.clone(), closure.clone());
            }
            (sessions, running, records, closures, previous_closing)
        };
        self.signal_change();
        for child in sessions.iter().filter(|child| *child != session_id) {
            self.inner
                .services
                .tool_sessions
                .request_stop_session(child)
                .await;
        }
        let mut failures = HashMap::new();
        let settled =
            futures::future::join_all(running.iter_mut().map(|(_, task)| &mut task.join_handle));
        let results = match deadline {
            Some(deadline) => tokio::time::timeout_at(deadline, settled).await.ok(),
            None => Some(settled.await),
        };
        if results.is_none() {
            // Kill all descendant resources first, then abort stream wrappers
            // to prevent late registration of another executor.
            for child in sessions.iter().filter(|child| *child != session_id) {
                self.inner
                    .services
                    .tool_sessions
                    .force_stop_session(child)
                    .await;
            }
            for (_, task) in &running {
                task.join_handle.abort();
            }
            for (_, task) in &mut running {
                let _ = (&mut task.join_handle).await;
            }
        }
        for ((child, _), result) in running.iter().zip(results.into_iter().flatten()) {
            if let Err(error) = result
                && !error.is_cancelled()
            {
                failures
                    .entry(child.clone())
                    .or_insert_with(|| anyhow::Error::new(error));
            }
        }
        for (child, task) in &running {
            let turn_id = task
                .current_turn
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .clone();
            if let Some(turn_id) = turn_id {
                let repaired =
                    match SessionExecutor::new(self.inner.services.clone(), child.clone()) {
                        Ok(executor) => executor.force_interrupt_turn(&turn_id).await,
                        Err(error) => Err(error),
                    };
                if let Err(error) = repaired {
                    failures.entry(child.clone()).or_insert(error);
                }
            }
        }
        for (parent, status, generation) in records {
            let child = status.session_id.clone();
            if let Err(error) = self.record(parent, status, generation).await {
                failures.entry(child).or_insert(error);
            }
        }
        for (child, _) in &closures {
            if let Err(error) = self
                .inner
                .services
                .tool_sessions
                .shutdown_session(child)
                .await
            {
                failures.entry(child.clone()).or_insert(error);
            }
            if let Err(error) = self.end_session(child).await {
                failures.entry(child.clone()).or_insert(error);
            }
        }
        let mut failure = None;
        // A previously closing child may be awaiting these descendants.
        // Finish this operation's children before joining existing cleanup.
        {
            let mut registry = self.inner.registry();
            for (closed, control) in closures {
                registry.closing_turns.remove(&closed);
                let child_failure = failures.remove(&closed).map(Arc::new);
                if failure.is_none() {
                    failure = child_failure.clone();
                }
                control.finish(child_failure);
            }
        }
        self.signal_change();
        let previous_settled =
            futures::future::join_all(previous_closing.iter().map(ClosingTurn::result));
        let results = match deadline {
            Some(deadline) => match tokio::time::timeout_at(deadline, previous_settled).await {
                Ok(results) => results,
                Err(_) => {
                    self.force_close_session(session_id).await;
                    futures::future::join_all(previous_closing.iter().map(ClosingTurn::result))
                        .await
                }
            },
            None => previous_settled.await,
        };
        for result in results {
            if let Err(error) = result {
                failure.get_or_insert_with(|| Arc::new(error));
            }
        }
        let mut registry = self.inner.registry();
        for closed in &sessions {
            let remaining = registry
                .closing_sessions
                .get_mut(closed)
                .expect("closure owns its fence");
            *remaining -= 1;
            if *remaining == 0 {
                registry.closing_sessions.remove(closed);
            }
        }
        drop(registry);
        self.signal_change();
        failure.map_or(Ok(()), |error| {
            Err(anyhow::Error::new(SharedCloseError(error)))
        })
    }

    /// Upgrade graceful descendant cleanup to immediate cancellation. The
    /// existing cleanup task keeps ownership of joins and transcript repair.
    pub(crate) async fn force_close_session(&self, session_id: &SessionId) {
        let (sessions, turns) = {
            let registry = self.inner.registry();
            let sessions = owned_session_tree(&registry, session_id);
            let mut turns = registry
                .closing_turns
                .iter()
                .filter(|(child, _)| sessions.contains(*child))
                .map(|(_, control)| control.clone())
                .collect::<Vec<_>>();
            turns.extend(
                registry
                    .entries
                    .values()
                    .filter(|entry| sessions.contains(&entry.parent))
                    .filter_map(|entry| entry.running.as_ref())
                    .map(|task| ClosingTurn::new(Some(task))),
            );
            (sessions, turns)
        };
        for child in sessions {
            self.inner
                .services
                .tool_sessions
                .force_stop_session(&child)
                .await;
        }
        for turn in turns {
            if let Some(wrapper) = turn.wrapper {
                wrapper.abort();
            }
            if let Some(turn_id) = turn
                .current_turn
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .as_ref()
            {
                self.inner.services.turn_registry.abort(turn_id);
            }
        }
    }

    /// Register the agents `parent`'s log recorded, keeping any entry this
    /// process already holds.
    pub(crate) async fn restore(
        &self,
        parent: &SessionId,
        records: impl IntoIterator<Item = SubagentRecord>,
    ) {
        let mut registry = self.inner.registry();
        for SubagentRecord { status, generation } in records {
            let already_registered = registry.entries.contains_key(&status.agent_id.0);
            if !already_registered && status.state != SubagentState::Closed {
                self.inner
                    .services
                    .tool_sessions
                    .open_session(&status.session_id);
            }
            // Restored children run turns without a resume, so they rejoin
            // the parent's trace here.
            if let Some(recorder) = &self.inner.services.trace_recorder
                && let Err(error) = recorder.attach(&status.session_id, parent)
            {
                warn!(session_id = %status.session_id, error = %error, "failed to reattach subagent trace");
            }
            registry
                .entries
                .entry(status.agent_id.0.clone())
                .or_insert_with(|| RegisteredSubagent {
                    parent: parent.clone(),
                    status,
                    generation,
                    running: None,
                    closure: None,
                });
        }
        drop(registry);
        self.signal_change();
    }

    async fn settle_closed_turn(
        &self,
        session_id: SessionId,
        mut running: Option<RunningTurn>,
        control: ClosingTurn,
        deadline: Option<tokio::time::Instant>,
        record: Option<(SessionId, SubagentStatus, u64)>,
    ) -> anyhow::Result<()> {
        let mut failure = None;
        {
            self.inner
                .services
                .tool_sessions
                .request_stop_session(&session_id)
                .await;
            if let Some(running) = &mut running {
                let joined = match deadline {
                    Some(deadline) => tokio::time::timeout_at(deadline, &mut running.join_handle)
                        .await
                        .ok(),
                    None => Some((&mut running.join_handle).await),
                };
                if joined.is_none() {
                    self.force_close_session(&session_id).await;
                    let _ = (&mut running.join_handle).await;
                }
                let turn_id = running
                    .current_turn
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .clone();
                if let Some(turn_id) = turn_id {
                    let repaired =
                        match SessionExecutor::new(self.inner.services.clone(), session_id.clone())
                        {
                            Ok(executor) => executor.force_interrupt_turn(&turn_id).await,
                            Err(error) => Err(error),
                        };
                    if let Err(error) = repaired {
                        failure.get_or_insert(error);
                    }
                }
                if let Some(Err(error)) = joined
                    && !error.is_cancelled()
                {
                    failure.get_or_insert_with(|| anyhow::Error::new(error));
                }
            }
            if let Err(error) = self
                .close_session_with_deadline(&session_id, deadline)
                .await
            {
                failure.get_or_insert(error);
            }
            if let Err(error) = self
                .inner
                .services
                .tool_sessions
                .shutdown_session(&session_id)
                .await
            {
                failure.get_or_insert(error);
            }
            if let Err(error) = self.end_session(&session_id).await {
                failure.get_or_insert(error);
            }
            // Record after execution/resource cancellation has started, so a
            // blocked writer cannot postpone the cancellation deadline.
            if let Some((parent, status, generation)) = record
                && let Err(error) = self.record(parent, status, generation).await
            {
                failure.get_or_insert(error);
            }
        }
        let failure = failure.map(Arc::new);
        if let Some(error) = &failure {
            warn!(%session_id, %error, "failed to settle closed subagent");
        }
        self.inner.registry().closing_turns.remove(&session_id);
        control.finish(failure);
        self.signal_change();
        control.result().await
    }

    async fn end_session(&self, session_id: &SessionId) -> anyhow::Result<()> {
        // Failed seeded creation may never have persisted a child session.
        if self
            .inner
            .services
            .sessions
            .load_session(session_id)
            .await?
            .is_none()
        {
            return Ok(());
        }
        let events = self.inner.services.sessions.replay(session_id).await?;
        let already_ended = events.iter().rev().find_map(|event| match event.payload {
            halter_protocol::SessionEventPayload::SessionShutdownComplete => Some(true),
            halter_protocol::SessionEventPayload::SessionStarted
            | halter_protocol::SessionEventPayload::SessionResumed => Some(false),
            _ => None,
        }) == Some(true);
        if !already_ended {
            SessionExecutor::new(self.inner.services.clone(), session_id.clone())?
                .shutdown("parent_session_closed")
                .await?;
        }
        Ok(())
    }

    /// Append an agent's new status to its parent's log.
    async fn record(
        &self,
        parent: SessionId,
        status: SubagentStatus,
        generation: u64,
    ) -> anyhow::Result<()> {
        let record = OutOfTurn::Subagent(SubagentRecord { status, generation });
        let dispatched = match SessionExecutor::new(self.inner.services.clone(), parent.clone()) {
            Ok(session) => session.dispatch_out_of_turn(record).await,
            Err(error) => Err(error),
        };
        if let Err(ref error) = dispatched {
            warn!(session_id = %parent, error = %error, "failed to record subagent status");
        }
        dispatched
    }

    fn signal_change(&self) {
        self.inner.activity.send_replace(());
    }

    async fn reserve_subagent_slot(
        &self,
        parent: &SubagentParentContext,
        agent_id: &AgentId,
        status: SubagentStatus,
    ) -> anyhow::Result<()> {
        loop {
            let active = {
                let registry = self.inner.registry();
                active_subagent_count(&registry)
            };
            self.inner
                .services
                .policy
                .check_subagent_spawn_typed(parent.blueprint.subagent_depth, active)
                .await?;

            let mut registry = self.inner.registry();
            if registry
                .closing_sessions
                .contains_key(&parent.blueprint.session_id)
            {
                anyhow::bail!("failed to execute spawn_agent tool: parent session is closing");
            }
            let active_now = active_subagent_count(&registry);
            if active_now != active {
                continue;
            }
            registry.entries.insert(
                agent_id.0.clone(),
                RegisteredSubagent {
                    parent: parent.blueprint.session_id.clone(),
                    status,
                    generation: 0,
                    running: None,
                    closure: None,
                },
            );
            drop(registry);
            self.signal_change();
            return Ok(());
        }
    }

    async fn remove_reserved_subagent(&self, agent_id: &AgentId) {
        if let Err(error) = self
            .close(CloseSubagentRequest {
                target: agent_id.clone(),
                timeout_ms: None,
            })
            .await
        {
            warn!(%agent_id, %error, "failed to clean up reserved subagent");
        }
        self.inner.registry().entries.remove(&agent_id.0);
        self.signal_change();
    }

    async fn settle_aborted_turn(
        &self,
        agent_id: AgentId,
        session_id: SessionId,
        generation: u64,
        turn_id: Option<TurnId>,
    ) {
        let still_owned = self
            .inner
            .registry()
            .entries
            .get(&agent_id.0)
            .is_some_and(|entry| {
                entry.generation == generation && entry.status.state != SubagentState::Closed
            });
        if !still_owned {
            return;
        }
        let error = if let Some(turn_id) = turn_id {
            match SessionExecutor::new(self.inner.services.clone(), session_id) {
                Ok(session) => session.force_interrupt_turn(&turn_id).await.err(),
                Err(error) => Some(error),
            }
        } else {
            None
        };
        self.finish_turn(
            agent_id,
            generation,
            TurnOutcome {
                state: SubagentState::Failed,
                last_message: None,
                usage: None,
                error: Some(error.map_or_else(
                    || "subagent task aborted before completion".to_owned(),
                    |error| error.to_string(),
                )),
            },
        )
        .await;
    }

    async fn start_turn(
        &self,
        agent_id: &AgentId,
        session_id: &SessionId,
        agent_type: Option<AgentName>,
        message: String,
    ) -> anyhow::Result<SubagentStatus> {
        let parent_session_id = self
            .inner
            .services
            .sessions
            .load_session(session_id)
            .await?
            .and_then(|stored| stored.blueprint.parent_session_id);
        // Subagents outlive the parent turn; close_agent and runtime shutdown
        // cancel them.
        let cancel = self.inner.services.turn_registry.child_token();
        let (parent, generation, status) = {
            let mut registry = self.inner.registry();
            let entry = registry.entries.get_mut(&agent_id.0).with_context(|| {
                format!(
                    "failed to execute subagent request: unknown agent '{}'",
                    agent_id.0
                )
            })?;
            if matches!(entry.status.state, SubagentState::Closed) {
                anyhow::bail!(
                    "failed to execute subagent request: agent '{}' is closed",
                    agent_id.0
                );
            }
            if entry.running.is_some() {
                anyhow::bail!(
                    "failed to execute subagent request: agent '{}' is still running; use wait_agent to wait for completion, or close_agent to stop it",
                    agent_id.0
                );
            }
            entry.generation = entry.generation.saturating_add(1);
            entry.status.task = message.clone();
            if let Some(ref agent_type) = agent_type {
                entry.status.agent_type = Some(agent_type.clone());
            }
            entry.status.state = SubagentState::Running;
            entry.status.last_message = None;
            entry.status.usage = None;
            entry.status.error = None;
            (entry.parent.clone(), entry.generation, entry.status.clone())
        };
        self.signal_change();
        let current_turn = Arc::new(std::sync::Mutex::new(None));
        let mut guard = TurnTaskGuard {
            controller: self.clone(),
            agent_id: agent_id.clone(),
            session_id: session_id.clone(),
            generation,
            current_turn: current_turn.clone(),
            armed: true,
        };
        // Before the turn is spawned, so its outcome is recorded after it.
        let _ = self.record(parent, status.clone(), generation).await;

        let services = self.inner.services.clone();
        let task_agent_id = agent_id.clone();
        let task_session_id = session_id.clone();
        let task_message = message.clone();
        let task_cancel = cancel.clone();
        let controller = self.clone();
        let current_turn_for_task = current_turn.clone();
        let session = match SessionExecutor::new(services.clone(), task_session_id.clone()) {
            Ok(session) => session,
            Err(error) => {
                self.finish_turn(
                    agent_id.clone(),
                    generation,
                    TurnOutcome {
                        state: SubagentState::Failed,
                        last_message: None,
                        usage: None,
                        error: Some(error.to_string()),
                    },
                )
                .await;
                guard.disarm();
                return Err(error);
            }
        };
        let mut registry = self.inner.registry();
        let can_start = registry.entries.get(&agent_id.0).is_some_and(|entry| {
            entry.generation == generation && matches!(entry.status.state, SubagentState::Running)
        });
        if !can_start {
            anyhow::bail!(
                "failed to execute subagent request: agent '{}' closed before execution started",
                agent_id.0
            );
        }
        // Spawn and registration share the lock so closure always owns the task.
        let join_handle = tokio::spawn(async move {
            controller
                .run_turn_task(
                    task_agent_id,
                    task_session_id,
                    parent_session_id,
                    agent_type.clone(),
                    generation,
                    task_message,
                    task_cancel,
                    session,
                    current_turn_for_task,
                )
                .await;
            guard.disarm();
        });

        registry
            .entries
            .get_mut(&agent_id.0)
            .expect("validated under the registry lock")
            .running = Some(RunningTurn {
            cancel,
            join_handle,
            current_turn,
        });
        drop(registry);
        self.signal_change();
        info!(
            agent_id = %status.agent_id,
            session_id = %status.session_id,
            task = %status.task,
            "started subagent turn"
        );
        Ok(status)
    }

    #[expect(clippy::too_many_arguments)]
    async fn run_turn_task(
        &self,
        agent_id: AgentId,
        session_id: SessionId,
        parent_session_id: Option<SessionId>,
        agent_type: Option<AgentName>,
        generation: u64,
        message: String,
        cancel: CancellationToken,
        session: SessionExecutor,
        current_turn: Arc<std::sync::Mutex<Option<TurnId>>>,
    ) {
        let mut next_input = message;
        let mut resubmissions = 0u32;
        let outcome = loop {
            let turn = Turn::user(next_input.clone());
            *current_turn
                .lock()
                .unwrap_or_else(|error| error.into_inner()) = Some(turn.id.clone());
            let turn_events = match session.submit_turn_with_cancel(turn, cancel.clone()).await {
                Ok(events) => match events.try_collect::<Vec<_>>().await {
                    Ok(events) => events,
                    Err(error) => {
                        break TurnOutcome {
                            state: if cancel.is_cancelled() {
                                SubagentState::Cancelled
                            } else {
                                SubagentState::Failed
                            },
                            last_message: None,
                            usage: None,
                            error: Some(error.to_string()),
                        };
                    }
                },
                Err(error) => {
                    break TurnOutcome {
                        state: if cancel.is_cancelled() {
                            SubagentState::Cancelled
                        } else {
                            SubagentState::Failed
                        },
                        last_message: None,
                        usage: None,
                        error: Some(error.to_string()),
                    };
                }
            };

            if cancel.is_cancelled() {
                break TurnOutcome {
                    state: SubagentState::Cancelled,
                    last_message: None,
                    usage: None,
                    error: None,
                };
            }
            let own_turn_events = turn_events
                .iter()
                .filter(|event| event.session_id == session_id)
                .cloned()
                .collect::<Vec<_>>();

            let Some(parent_session_id) = parent_session_id.as_ref() else {
                break TurnOutcome {
                    state: SubagentState::Completed,
                    last_message: extract_subagent_output(&own_turn_events),
                    usage: extract_subagent_usage(&own_turn_events),
                    error: None,
                };
            };

            let continuation = match self
                .run_subagent_stop_hooks(
                    parent_session_id,
                    &agent_id,
                    agent_type.as_ref(),
                    &session_id,
                    &cancel,
                )
                .await
            {
                Ok(value) => value,
                Err(error) => {
                    break TurnOutcome {
                        state: SubagentState::Failed,
                        last_message: None,
                        usage: None,
                        error: Some(error.to_string()),
                    };
                }
            };

            if let Some(next_message) = continuation {
                // Bounded: a SubagentStop hook that always blocks must not
                // resubmit turns forever (each resubmission is a full
                // provider turn).
                if resubmissions >= SUBAGENT_STOP_RESUBMISSION_CAP {
                    break TurnOutcome {
                        state: SubagentState::Failed,
                        last_message: None,
                        usage: extract_subagent_usage(&own_turn_events),
                        error: Some(format!(
                            "subagent stop hooks kept blocking; resubmission cap of {SUBAGENT_STOP_RESUBMISSION_CAP} reached"
                        )),
                    };
                }
                resubmissions += 1;
                next_input = next_message;
                continue;
            }

            break TurnOutcome {
                state: SubagentState::Completed,
                last_message: extract_subagent_output(&own_turn_events),
                usage: extract_subagent_usage(&own_turn_events),
                error: None,
            };
        };

        self.finish_turn(agent_id, generation, outcome).await;
    }

    async fn finish_turn(&self, agent_id: AgentId, generation: u64, outcome: TurnOutcome) {
        let (parent, status) = {
            let registry = self.inner.registry();
            let Some(entry) = registry.entries.get(&agent_id.0) else {
                return;
            };
            if entry.generation != generation || entry.status.state == SubagentState::Closed {
                return;
            }
            let mut status = entry.status.clone();
            status.state = outcome.state;
            status.last_message = outcome.last_message;
            status.usage = outcome.usage;
            status.error = outcome.error;
            (entry.parent.clone(), status)
        };
        // Retain the Running state and task until final status recording has
        // settled, including the pre-spawn failure path with no task handle.
        let _ = self.record(parent, status.clone(), generation).await;
        {
            let mut registry = self.inner.registry();
            if let Some(entry) = registry.entries.get_mut(&agent_id.0)
                && entry.generation == generation
                && entry.status.state != SubagentState::Closed
            {
                entry.status = status;
                entry.running = None;
                debug!(
                    agent_id = %entry.status.agent_id,
                    state = ?entry.status.state,
                    "completed subagent turn"
                );
            }
        }
        self.signal_change();
    }

    async fn run_subagent_start_hooks(
        &self,
        parent: &SubagentParentContext,
        status: &SubagentStatus,
        cancel: &CancellationToken,
    ) -> anyhow::Result<()> {
        let session = SessionExecutor::new(
            self.inner.services.clone(),
            parent.blueprint.session_id.clone(),
        )?;
        let Some((stored, fired_hook_ids)) = session.load_for_out_of_turn_hooks().await? else {
            return Ok(());
        };
        let turn_id = TurnId::new();
        // Runs inside the parent's spawn tool call, under its token.
        let dispatch = run_subagent_start(
            &session,
            &fired_hook_ids,
            HookInvocationContext {
                turn_id: &turn_id,
                model: &stored.blueprint.default_model,
                working_dir: &stored.blueprint.working_dir,
                cancel,
            },
            &status.agent_id,
            status
                .agent_type
                .as_ref()
                .map_or("default", |agent_type| agent_type.0.as_str()),
            &parent.blueprint.session_id,
        )
        .await?;
        if dispatch.merged.block_reason.is_some() || dispatch.merged.stop_reason.is_some() {
            warn!(session_id = %parent.blueprint.session_id, "hooks.ignored_block");
        }
        // The parent is mid-turn (this runs inside its spawn tool call), so
        // the dispatch queues behind the parent turn instead of racing it.
        session
            .dispatch_out_of_turn(crate::session_lease::OutOfTurn::Hooks(dispatch))
            .await
    }

    async fn run_subagent_stop_hooks(
        &self,
        parent_session_id: &SessionId,
        agent_id: &AgentId,
        agent_type: Option<&AgentName>,
        child_session_id: &SessionId,
        cancel: &CancellationToken,
    ) -> anyhow::Result<Option<String>> {
        let session = SessionExecutor::new(self.inner.services.clone(), parent_session_id.clone())?;
        let Some((stored, fired_hook_ids)) = session.load_for_out_of_turn_hooks().await? else {
            return Ok(None);
        };
        let turn_id = TurnId::new();
        let transcript_path = self
            .inner
            .services
            .sessions
            .transcript_path(child_session_id);
        // Runs under the subagent's token, so close_agent stops it.
        let dispatch = run_subagent_stop(
            &session,
            &fired_hook_ids,
            HookInvocationContext {
                turn_id: &turn_id,
                model: &stored.blueprint.default_model,
                working_dir: &stored.blueprint.working_dir,
                cancel,
            },
            agent_id,
            agent_type.map_or("default", |agent_type| agent_type.0.as_str()),
            transcript_path.as_deref(),
        )
        .await?;
        let block_reason = dispatch.merged.block_reason.clone();
        session
            .dispatch_out_of_turn(crate::session_lease::OutOfTurn::Hooks(dispatch))
            .await?;
        Ok(block_reason)
    }

    async fn terminal_status_for_targets(
        &self,
        targets: &[AgentId],
    ) -> anyhow::Result<Option<SubagentStatus>> {
        let registry = self.inner.registry();
        let statuses = load_target_statuses(&registry, targets)?;
        Ok(statuses.into_iter().find(|status| status.is_terminal()))
    }
}

#[async_trait]
impl SubagentControl for RuntimeSubagentControl {
    async fn spawn(
        &self,
        parent: &SubagentParentContext,
        request: SpawnSubagentRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<SubagentStatus> {
        if request.message.trim().is_empty() {
            anyhow::bail!("failed to execute spawn_agent tool: message cannot be empty");
        }

        let session_id = SessionId::new();
        let agent_id = AgentId::new();
        let init = build_subagent_session_init(parent, &session_id, &request)?;
        let state =
            build_subagent_state(parent, &session_id, &request.message, request.fork_context);
        let status = SubagentStatus {
            agent_id: agent_id.clone(),
            session_id: session_id.clone(),
            agent_type: request.agent_type.clone(),
            task: request.message.clone(),
            state: SubagentState::Running,
            last_message: None,
            usage: None,
            error: None,
        };

        self.reserve_subagent_slot(parent, &agent_id, status.clone())
            .await?;
        let mut reservation = SpawnReservation {
            controller: self.clone(),
            agent_id: Some(agent_id.clone()),
        };

        if let Err(error) = create_session_seeded(
            self.inner.services.clone(),
            init,
            state,
            parent.snapshot.clone(),
        )
        .await
        {
            reservation.cleanup().await;
            return Err(error);
        }
        self.inner.services.tool_sessions.open_session(&session_id);

        if let Err(error) = self
            .run_subagent_start_hooks(parent, &status, &cancel)
            .await
        {
            reservation.cleanup().await;
            return Err(error);
        }
        // A parent cancelled mid-spawn could never wait on or close the
        // child, so it is not started.
        if cancel.is_cancelled() {
            reservation.cleanup().await;
            anyhow::bail!("failed to execute spawn_agent tool: cancelled");
        }

        let started = self
            .start_turn(
                &agent_id,
                &session_id,
                request.agent_type.clone(),
                request.message,
            )
            .await;
        if started.is_ok() {
            reservation.agent_id = None;
        } else {
            reservation.cleanup().await;
        }
        started
    }

    async fn send_input(
        &self,
        request: SendSubagentInputRequest,
    ) -> anyhow::Result<SubagentStatus> {
        if request.message.trim().is_empty() {
            anyhow::bail!("failed to execute send_input tool: message cannot be empty");
        }

        let (session_id, agent_type) = {
            let registry = self.inner.registry();
            let entry = registry.entries.get(&request.target.0).with_context(|| {
                format!(
                    "failed to execute send_input tool: unknown agent '{}'",
                    request.target.0
                )
            })?;
            if entry.running.is_some() {
                anyhow::bail!(
                    "failed to execute send_input tool: agent '{}' is still running; use wait_agent to wait for completion, or close_agent to stop it",
                    request.target.0
                );
            }
            if matches!(entry.status.state, SubagentState::Closed) {
                anyhow::bail!(
                    "failed to execute send_input tool: agent '{}' is closed",
                    request.target.0
                );
            }
            (
                entry.status.session_id.clone(),
                entry.status.agent_type.clone(),
            )
        };

        self.start_turn(&request.target, &session_id, agent_type, request.message)
            .await
    }

    async fn wait(
        &self,
        request: WaitSubagentRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<WaitSubagentResponse> {
        if request.targets.is_empty() {
            anyhow::bail!("failed to execute wait_agent tool: targets cannot be empty");
        }

        let mut activity = self.subscribe_activity();

        if let Some(status) = self.terminal_status_for_targets(&request.targets).await? {
            return Ok(WaitSubagentResponse {
                status: Some(status),
                timed_out: false,
                target_statuses: Vec::new(),
            });
        }

        let wait_for_status = async {
            loop {
                if cancel.is_cancelled() {
                    anyhow::bail!("failed to execute wait_agent tool: cancelled");
                }
                if let Some(status) = self.terminal_status_for_targets(&request.targets).await? {
                    return anyhow::Result::<SubagentStatus>::Ok(status);
                }
                tokio::select! {
                    biased;
                    () = cancel.cancelled() => {}
                    changed = activity.changed() => {
                        changed.context("failed to observe subagent activity")?;
                    }
                }
            }
        };

        match request.timeout_ms {
            Some(timeout_ms) => {
                match timeout(Duration::from_millis(timeout_ms), wait_for_status).await {
                    Ok(status) => Ok(WaitSubagentResponse {
                        status: Some(status?),
                        timed_out: false,
                        target_statuses: Vec::new(),
                    }),
                    Err(_) => {
                        let registry = self.inner.registry();
                        let target_statuses = match load_target_statuses(
                            &registry,
                            &request.targets,
                        ) {
                            Ok(statuses) => statuses,
                            Err(error) => {
                                warn!(
                                    error = %error,
                                    "failed to snapshot subagent statuses after wait_agent timeout"
                                );
                                Vec::new()
                            }
                        };
                        Ok(WaitSubagentResponse {
                            status: None,
                            timed_out: true,
                            target_statuses,
                        })
                    }
                }
            }
            None => Ok(WaitSubagentResponse {
                status: Some(wait_for_status.await?),
                timed_out: false,
                target_statuses: Vec::new(),
            }),
        }
    }

    async fn close(&self, request: CloseSubagentRequest) -> anyhow::Result<CloseSubagentResponse> {
        let deadline =
            request
                .timeout_ms
                .map(|milliseconds| {
                    tokio::time::Instant::now().checked_add(Duration::from_millis(milliseconds))
                .context("failed to close subagent: timeout is outside the supported clock range")
                })
                .transpose()?;
        let (previous_status, completion) = {
            let mut registry = self.inner.registry();
            let entry = registry
                .entries
                .get_mut(&request.target.0)
                .with_context(|| {
                    format!(
                        "failed to execute close_agent tool: unknown agent '{}'",
                        request.target.0
                    )
                })?;
            let previous = entry.status.clone();
            let already_closed = matches!(entry.status.state, SubagentState::Closed);
            if already_closed {
                (previous, entry.closure.clone())
            } else {
                let published = entry.generation != 0;
                entry.generation = entry.generation.saturating_add(1);
                let closed_running_turn = entry.running.is_some();
                let running = entry.running.take();
                if let Some(running) = &running {
                    running.cancel.cancel();
                }
                entry.status.state = SubagentState::Closed;
                entry.status.error = closed_running_turn
                    .then(|| "closed by close_agent (work was cancelled)".to_owned());
                let record = published
                    .then(|| (entry.parent.clone(), entry.status.clone(), entry.generation));
                let session_id = previous.session_id.clone();
                let control = ClosingTurn::new(running.as_ref());
                entry.closure = Some(control.clone());
                registry
                    .closing_turns
                    .insert(session_id.clone(), control.clone());
                let controller = self.clone();
                let completion = control.clone();
                tokio::spawn(async move {
                    let _ = controller
                        .settle_closed_turn(session_id, running, control, deadline, record)
                        .await;
                });
                (previous, Some(completion))
            }
        };
        self.signal_change();
        if let Some(completion) = completion {
            match deadline {
                Some(deadline) => {
                    match tokio::time::timeout_at(deadline, completion.result()).await {
                        Ok(result) => result?,
                        Err(_) => {
                            self.force_close_session(&previous_status.session_id).await;
                            return Err(anyhow::Error::new(crate::SessionError::TimedOut));
                        }
                    }
                }
                None => completion.result().await?,
            }
        }

        warn!(
            agent_id = %previous_status.agent_id,
            session_id = %previous_status.session_id,
            "closed subagent"
        );
        self.signal_change();
        Ok(CloseSubagentResponse { previous_status })
    }
}

fn load_target_statuses(
    registry: &SubagentRegistry,
    targets: &[AgentId],
) -> anyhow::Result<Vec<SubagentStatus>> {
    targets
        .iter()
        .map(|target| {
            registry
                .entries
                .get(&target.0)
                .map(|entry| entry.status.clone())
                .with_context(|| {
                    format!(
                        "failed to execute wait_agent tool: unknown agent '{}'",
                        target.0
                    )
                })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use futures::StreamExt;
    use futures::stream::{self, BoxStream};
    use halter_protocol::{
        ApiKind, BlockId, Message, ModelId, ModelRole, ProviderCapabilities, ProviderError,
        ProviderKind, ProviderName, ProviderRequest, ResolvedModel, StopReason, StreamEvent,
    };
    use halter_providers::{ModelRegistry, Provider};
    use halter_session::{InMemorySessionStore, SessionStore, StoredSession};
    use halter_tools::{
        DefaultToolPolicy, NoopToolEventSink, PathLockMap, PolicySettings, ShellTool, Tool,
        ToolContext, ToolRuntime, ToolSessionStore,
    };
    use tokio::sync::Notify;

    use super::*;
    use crate::{
        DefaultContextManager, DefaultPromptAssembler, EventBus, ResourceHandle, SessionRuntime,
    };

    #[tokio::test]
    async fn spawn_and_wait_complete_child_session() {
        let provider_requests = Arc::new(Mutex::new(Vec::<ProviderRequest>::new()));
        let services = test_services(Arc::new(RecordingProvider::new(provider_requests.clone())));
        let control = RuntimeSubagentControl::new(services);
        let parent = parent_context();

        let status = control
            .spawn(
                &parent,
                SpawnSubagentRequest {
                    message: "delegate this".to_owned(),
                    agent_type: None,
                    fork_context: true,
                    model: None,
                },
                CancellationToken::new(),
            )
            .await
            .expect("spawn");

        assert_eq!(status.state, SubagentState::Running);
        let waited = control
            .wait(
                WaitSubagentRequest {
                    targets: vec![status.agent_id.clone()],
                    timeout_ms: Some(5_000),
                },
                CancellationToken::new(),
            )
            .await
            .expect("wait");

        assert!(!waited.timed_out);
        assert!(waited.target_statuses.is_empty());
        let waited_status = waited.status.expect("completed status");
        assert_eq!(waited_status.state, SubagentState::Completed);
        assert_eq!(
            waited_status.last_message.as_deref(),
            Some("child reply [subagent/model]")
        );

        let requests = provider_requests.lock().expect("requests");
        assert_eq!(requests.len(), 1);
        assert_eq!(requests[0].model.id, ModelId::from("subagent"));
        assert_eq!(requests[0].messages.len(), parent.state.messages.len() + 1);
        assert_eq!(requests[0].messages[0], parent.state.messages[0]);
        assert!(matches!(
            &requests[0].messages[1],
            Message::User(user) if user.plain_text() == "delegate this"
        ));
    }

    #[tokio::test]
    async fn subagent_activity_retains_last_completion_and_tracks_descendants() {
        let release = CancellationToken::new();
        let services = test_services(Arc::new(GatedProvider {
            release: release.clone(),
        }));
        let control = RuntimeSubagentControl::new(services.clone());
        let parent = parent_context();
        store_parent_session(&services, &parent).await;
        let mut activity = control.subscribe_activity();
        assert!(!control.has_running_subagents(&parent.blueprint.session_id));
        let child = control
            .spawn(&parent, spawn_request("child"), CancellationToken::new())
            .await
            .unwrap();
        assert!(activity.has_changed().unwrap());
        assert!(control.has_running_subagents(&parent.blueprint.session_id));
        assert!(!control.has_running_subagents(&SessionId::from("unrelated")));
        let mut descendant_parent = parent.clone();
        descendant_parent.blueprint.session_id = child.session_id.clone();
        descendant_parent.blueprint.subagent_depth = 1;
        let grandchild = control
            .spawn(
                &descendant_parent,
                spawn_request("grandchild"),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert!(control.has_running_subagents(&child.session_id));
        activity.borrow_and_update();
        release.cancel();
        wait_for_no_subagent_activity(&control, &parent.blueprint.session_id, &mut activity).await;
        for agent in [child.agent_id, grandchild.agent_id] {
            let status = control.inner.registry().entries[&agent.0].status.clone();
            assert_eq!(status.state, SubagentState::Completed);
        }
        assert!(!control.has_running_subagents(&child.session_id));
    }

    async fn wait_for_no_subagent_activity(
        control: &RuntimeSubagentControl,
        parent: &SessionId,
        activity: &mut watch::Receiver<()>,
    ) {
        timeout(Duration::from_secs(5), async {
            while control.has_running_subagents(parent) {
                activity.changed().await.unwrap();
            }
        })
        .await
        .expect("final cleanup wakes the activity subscriber");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn completed_child_background_jobs_keep_parent_active_but_closed_history_does_not() {
        let root = tempfile::tempdir().unwrap();
        let mut services = test_services(Arc::new(RecordingProvider::new(Arc::default())));
        Arc::get_mut(&mut services).unwrap().policy =
            Arc::new(DefaultToolPolicy::new(PolicySettings {
                allowed_read_roots: vec![root.path().to_owned()],
                allowed_shell_commands: ["sleep".to_owned()].into_iter().collect(),
                ..PolicySettings::default()
            }));
        let runtime = SessionRuntime::new(services.clone());
        let control = runtime.subagents.clone();
        let parent = parent_context();
        store_parent_session(&services, &parent).await;
        let child = control
            .spawn(&parent, spawn_request("child"), CancellationToken::new())
            .await
            .unwrap();
        let waited = control
            .wait(
                WaitSubagentRequest {
                    targets: vec![child.agent_id.clone()],
                    timeout_ms: Some(5_000),
                },
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(waited.status.unwrap().state, SubagentState::Completed);
        assert!(!control.has_running_subagents(&parent.blueprint.session_id));
        let context = ToolContext {
            session_id: child.session_id.clone(),
            working_dir: root.path().to_owned(),
            path_locks: services.path_locks.clone(),
            tool_sessions: services.tool_sessions.clone(),
            snapshot: parent.snapshot.clone(),
            cancel: CancellationToken::new(),
            emit: Arc::new(NoopToolEventSink),
            policy: services.policy.clone(),
            shell_timeout_secs: 30,
            subagent_parent: None,
        };
        let halter_protocol::ToolResult::Json { value: job } = halter_tools::BackgroundTool
            .execute(
                context.clone(),
                serde_json::json!({"action": "spawn", "command": "sleep 30"}),
            )
            .await
            .unwrap()
        else {
            panic!("expected job metadata");
        };
        assert!(control.has_running_subagents(&parent.blueprint.session_id));
        halter_tools::BackgroundTool
            .execute(
                context.clone(),
                serde_json::json!({"action": "kill", "id": job["id"]}),
            )
            .await
            .unwrap();
        assert!(!control.has_running_subagents(&parent.blueprint.session_id));
        assert!(services.tool_sessions.has_process_state(&child.session_id));
        control
            .close(CloseSubagentRequest {
                target: child.agent_id,
                timeout_ms: None,
            })
            .await
            .unwrap();
        let (reopened, _) = runtime.resume_session(&child.session_id).await.unwrap();
        halter_tools::BackgroundTool
            .execute(
                context,
                serde_json::json!({"action": "spawn", "command": "sleep 30"}),
            )
            .await
            .unwrap();
        assert!(services.tool_sessions.has_running_jobs(&child.session_id));
        assert!(
            !control.has_running_subagents(&parent.blueprint.session_id),
            "the old parent has no live edge to the independently resumed child"
        );
        reopened.shutdown(None).await.unwrap();
        assert!(!services.tool_sessions.has_running_jobs(&child.session_id));
    }

    #[tokio::test]
    async fn send_input_reuses_existing_child_session() {
        let provider_requests = Arc::new(Mutex::new(Vec::<ProviderRequest>::new()));
        let services = test_services(Arc::new(RecordingProvider::new(provider_requests.clone())));
        let control = RuntimeSubagentControl::new(services);
        let parent = parent_context();

        let spawned = control
            .spawn(
                &parent,
                SpawnSubagentRequest {
                    message: "first task".to_owned(),
                    agent_type: None,
                    fork_context: false,
                    model: None,
                },
                CancellationToken::new(),
            )
            .await
            .expect("spawn");
        control
            .wait(
                WaitSubagentRequest {
                    targets: vec![spawned.agent_id.clone()],
                    timeout_ms: Some(5_000),
                },
                CancellationToken::new(),
            )
            .await
            .expect("wait");

        let restarted = control
            .send_input(SendSubagentInputRequest {
                target: spawned.agent_id.clone(),
                message: "follow up".to_owned(),
            })
            .await
            .expect("follow up");
        assert_eq!(restarted.session_id, spawned.session_id);

        let waited = control
            .wait(
                WaitSubagentRequest {
                    targets: vec![spawned.agent_id.clone()],
                    timeout_ms: Some(5_000),
                },
                CancellationToken::new(),
            )
            .await
            .expect("wait");
        assert_eq!(
            waited.status.expect("status").last_message.as_deref(),
            Some("child reply [subagent/model]")
        );
        assert_eq!(provider_requests.lock().expect("requests").len(), 2);
    }

    #[tokio::test]
    async fn wait_timeout_returns_target_statuses() {
        let services = test_services(Arc::new(PendingProvider));
        let control = RuntimeSubagentControl::new(services);
        let parent = parent_context();

        let first = control
            .spawn(
                &parent,
                SpawnSubagentRequest {
                    message: "first task".to_owned(),
                    agent_type: None,
                    fork_context: false,
                    model: None,
                },
                CancellationToken::new(),
            )
            .await
            .expect("spawn first");
        let second = control
            .spawn(
                &parent,
                SpawnSubagentRequest {
                    message: "second task".to_owned(),
                    agent_type: None,
                    fork_context: false,
                    model: None,
                },
                CancellationToken::new(),
            )
            .await
            .expect("spawn second");

        let waited = control
            .wait(
                WaitSubagentRequest {
                    targets: vec![first.agent_id.clone(), second.agent_id.clone()],
                    timeout_ms: Some(5),
                },
                CancellationToken::new(),
            )
            .await
            .expect("wait timeout");

        assert!(waited.timed_out);
        assert!(waited.status.is_none());
        assert_eq!(waited.target_statuses.len(), 2);
        assert!(
            waited
                .target_statuses
                .iter()
                .all(|status| status.state == SubagentState::Running),
            "target statuses should snapshot running agents: {:?}",
            waited.target_statuses
        );

        control
            .close(CloseSubagentRequest {
                target: first.agent_id,
                timeout_ms: None,
            })
            .await
            .expect("close first");
        control
            .close(CloseSubagentRequest {
                target: second.agent_id,
                timeout_ms: None,
            })
            .await
            .expect("close second");
    }

    #[tokio::test]
    async fn wait_unknown_target_errors_before_timeout() {
        let services = test_services(Arc::new(PendingProvider));
        let control = RuntimeSubagentControl::new(services);

        let error = control
            .wait(
                WaitSubagentRequest {
                    targets: vec![AgentId::from("missing-agent")],
                    timeout_ms: Some(5),
                },
                CancellationToken::new(),
            )
            .await
            .expect_err("unknown target should error");

        assert!(
            error.to_string().contains("unknown agent 'missing-agent'"),
            "unexpected error: {error:#}"
        );
    }

    #[tokio::test]
    async fn send_input_running_agent_error_suggests_control_flow() {
        let services = test_services(Arc::new(PendingProvider));
        let control = RuntimeSubagentControl::new(services);
        let parent = parent_context();
        let spawned = control
            .spawn(
                &parent,
                SpawnSubagentRequest {
                    message: "running task".to_owned(),
                    agent_type: None,
                    fork_context: false,
                    model: None,
                },
                CancellationToken::new(),
            )
            .await
            .expect("spawn");

        let error = control
            .send_input(SendSubagentInputRequest {
                target: spawned.agent_id.clone(),
                message: "follow up too soon".to_owned(),
            })
            .await
            .expect_err("running send_input should fail");
        let message = error.to_string();
        assert!(
            message.contains("wait_agent"),
            "unexpected error: {message}"
        );
        assert!(
            message.contains("close_agent"),
            "unexpected error: {message}"
        );

        control
            .close(CloseSubagentRequest {
                target: spawned.agent_id,
                timeout_ms: None,
            })
            .await
            .expect("close");
    }

    #[tokio::test]
    async fn close_running_agent_records_cancellation_status() {
        let services = test_services(Arc::new(PendingProvider));
        let control = RuntimeSubagentControl::new(services);
        let parent = parent_context();
        let spawned = control
            .spawn(
                &parent,
                SpawnSubagentRequest {
                    message: "running task".to_owned(),
                    agent_type: None,
                    fork_context: false,
                    model: None,
                },
                CancellationToken::new(),
            )
            .await
            .expect("spawn");

        let closed = control
            .close(CloseSubagentRequest {
                target: spawned.agent_id.clone(),
                timeout_ms: None,
            })
            .await
            .expect("close");
        assert_eq!(closed.previous_status.state, SubagentState::Running);

        let waited = control
            .wait(
                WaitSubagentRequest {
                    targets: vec![spawned.agent_id],
                    timeout_ms: None,
                },
                CancellationToken::new(),
            )
            .await
            .expect("wait");
        let status = waited.status.expect("closed status");
        assert_eq!(status.state, SubagentState::Closed);
        assert_eq!(
            status.error.as_deref(),
            Some("closed by close_agent (work was cancelled)")
        );
    }

    #[tokio::test]
    async fn descendant_deadlines_force_actual_execution_and_can_upgrade_graceful_closure() {
        for (agent_api, upgrade, sqlite) in [
            (false, false, false),
            (false, true, false),
            (true, false, false),
            (true, true, false),
            (false, false, true),
            (false, true, true),
            (true, false, true),
            (true, true, true),
        ] {
            let root = tempfile::tempdir().unwrap();
            let mut services = test_services(Arc::new(PendingProvider));
            if sqlite {
                Arc::get_mut(&mut services).unwrap().sessions = Arc::new(
                    halter_session::SqliteSessionStore::open(root.path().join("sessions.db"))
                        .unwrap(),
                );
            }
            let runtime = SessionRuntime::new(services.clone());
            let parent = runtime
                .new_session(crate::SessionInit::default())
                .await
                .unwrap();
            let child = runtime
                .new_session(crate::SessionInit::default())
                .await
                .unwrap();
            let control = RuntimeSubagentControl::new(services.clone());
            let turn_id = TurnId::new();
            let agent_id = AgentId::new();
            let cancel = CancellationToken::new();
            let dropped = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let (ready, started) = tokio::sync::oneshot::channel();
            let execution = tokio::spawn({
                let dropped = dropped.clone();
                async move {
                    struct Dropped(Arc<std::sync::atomic::AtomicBool>);
                    impl Drop for Dropped {
                        fn drop(&mut self) {
                            self.0.store(true, std::sync::atomic::Ordering::SeqCst);
                        }
                    }
                    let _dropped = Dropped(dropped);
                    ready.send(()).unwrap();
                    std::future::pending::<()>().await;
                }
            });
            services
                .turn_registry
                .register(turn_id.clone(), cancel.clone(), execution)
                .unwrap();
            started.await.unwrap();
            let mut stored = services
                .sessions
                .load_session(child.session_id())
                .await
                .unwrap()
                .unwrap();
            stored.state.open_turn = Some(turn_id.clone());
            services
                .sessions
                .commit(
                    child.session_id(),
                    None,
                    Some(stored.head_sequence),
                    Some(stored.state),
                    vec![halter_protocol::PendingEvent::new(
                        child.session_id().clone(),
                        halter_protocol::Delivery::Lossless,
                        halter_protocol::SessionEventPayload::TurnStarted {
                            turn_id: turn_id.clone(),
                            default_model: None,
                            subagent_model: None,
                        },
                    )],
                )
                .await
                .unwrap();
            control.inner.registry().entries.insert(
                agent_id.0.clone(),
                RegisteredSubagent {
                    parent: parent.session_id().clone(),
                    generation: 1,
                    closure: None,
                    status: SubagentStatus {
                        agent_id: agent_id.clone(),
                        session_id: child.session_id().clone(),
                        agent_type: None,
                        task: "ignore cancellation".to_owned(),
                        state: SubagentState::Running,
                        last_message: None,
                        usage: None,
                        error: None,
                    },
                    running: Some(RunningTurn {
                        cancel: cancel.clone(),
                        join_handle: tokio::spawn(std::future::pending()),
                        current_turn: Arc::new(std::sync::Mutex::new(Some(turn_id.clone()))),
                    }),
                },
            );
            if agent_api {
                let initial = tokio::spawn({
                    let control = control.clone();
                    let agent_id = agent_id.clone();
                    async move {
                        control
                            .close(CloseSubagentRequest {
                                target: agent_id,
                                timeout_ms: if upgrade { None } else { Some(0) },
                            })
                            .await
                    }
                });
                tokio::time::timeout(Duration::from_secs(5), cancel.cancelled())
                    .await
                    .unwrap();
                if upgrade {
                    assert!(
                        !initial.is_finished(),
                        "unlimited close retains ownership while waiting"
                    );
                    let error = control
                        .close(CloseSubagentRequest {
                            target: agent_id.clone(),
                            timeout_ms: Some(0),
                        })
                        .await
                        .unwrap_err();
                    assert!(matches!(
                        error.downcast_ref::<crate::SessionError>(),
                        Some(crate::SessionError::TimedOut)
                    ));
                    tokio::time::timeout(Duration::from_secs(5), initial)
                        .await
                        .unwrap()
                        .unwrap()
                        .unwrap();
                } else {
                    let error = initial.await.unwrap().unwrap_err();
                    assert!(matches!(
                        error.downcast_ref::<crate::SessionError>(),
                        Some(crate::SessionError::TimedOut)
                    ));
                }
                tokio::time::timeout(
                    Duration::from_secs(5),
                    control.close(CloseSubagentRequest {
                        target: agent_id,
                        timeout_ms: None,
                    }),
                )
                .await
                .unwrap()
                .unwrap();
            } else {
                let closure = tokio::spawn({
                    let control = control.clone();
                    let parent_id = parent.session_id().clone();
                    async move {
                        control
                            .close_session_with_deadline(
                                &parent_id,
                                if upgrade {
                                    None
                                } else {
                                    Some(tokio::time::Instant::now())
                                },
                            )
                            .await
                    }
                });
                tokio::time::timeout(Duration::from_secs(5), cancel.cancelled())
                    .await
                    .unwrap();
                if upgrade {
                    assert!(
                        !closure.is_finished(),
                        "unlimited descendant closure retains execution"
                    );
                    control.force_close_session(parent.session_id()).await;
                }
                tokio::time::timeout(Duration::from_secs(5), closure)
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap();
            }
            assert!(
                dropped.load(std::sync::atomic::Ordering::SeqCst),
                "registered executor was aborted, not only its stream wrapper"
            );
            let events = child.replay().await.unwrap();
            assert_eq!(events.iter().filter(|event| matches!(&event.payload, halter_protocol::SessionEventPayload::TurnFailed { turn_id: failed, cancelled: true, .. } if failed == &turn_id)).count(), 1);
            let stored = services
                .sessions
                .load_session(child.session_id())
                .await
                .unwrap()
                .unwrap();
            assert!(stored.state.open_turn.is_none());
        }
    }

    #[tokio::test]
    async fn completed_owned_child_cannot_be_resumed_but_parent_can_send_input() {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let services = test_services(Arc::new(RecordingProvider::new(requests.clone())));
        let runtime = SessionRuntime::new(services.clone());
        let parent = runtime
            .new_session(crate::SessionInit::default())
            .await
            .unwrap();
        let mut context = parent_context();
        context.blueprint.session_id = parent.session_id().clone();
        let control = runtime.subagents.clone();
        let child = control
            .spawn(
                &context,
                spawn_request("first task"),
                CancellationToken::new(),
            )
            .await
            .unwrap();
        let waited = control
            .wait(
                WaitSubagentRequest {
                    targets: vec![child.agent_id.clone()],
                    timeout_ms: Some(5_000),
                },
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(waited.status.unwrap().state, SubagentState::Completed);
        let result = runtime.resume_session(&child.session_id).await;
        assert!(
            matches!(result, Err(crate::SessionError::AlreadyOpen(ref id)) if id == &child.session_id),
            "parent retains exclusive ownership of completed child sessions"
        );
        control
            .send_input(SendSubagentInputRequest {
                target: child.agent_id.clone(),
                message: "second task".to_owned(),
            })
            .await
            .unwrap();
        let waited = control
            .wait(
                WaitSubagentRequest {
                    targets: vec![child.agent_id.clone()],
                    timeout_ms: Some(5_000),
                },
                CancellationToken::new(),
            )
            .await
            .unwrap();
        assert_eq!(waited.status.unwrap().state, SubagentState::Completed);
        assert_eq!(requests.lock().unwrap().len(), 2);
        let _ = services.tool_sessions.shell_session(&child.session_id);
        assert!(services.tool_sessions.has_process_state(&child.session_id));
        control
            .close(CloseSubagentRequest {
                target: child.agent_id.clone(),
                timeout_ms: None,
            })
            .await
            .unwrap();
        assert!(!services.tool_sessions.has_process_state(&child.session_id));
        assert!(!control.owns_session(&child.session_id).await);
        let (reopened, _) = runtime.resume_session(&child.session_id).await.unwrap();
        let shell = services.tool_sessions.shell_session(&child.session_id);
        control.force_close_session(parent.session_id()).await;
        control.close_session(parent.session_id()).await.unwrap();
        control
            .close(CloseSubagentRequest {
                target: child.agent_id,
                timeout_ms: None,
            })
            .await
            .unwrap();
        assert!(
            Arc::ptr_eq(
                &shell,
                &services.tool_sessions.shell_session(&child.session_id)
            ),
            "historical closure must not stop a separately reopened child"
        );
        assert_shell_accepts_work(&services, &child.session_id).await;
        reopened.shutdown(None).await.unwrap();
    }

    #[tokio::test]
    async fn restored_closed_records_do_not_own_or_close_reopened_sessions() {
        let services = test_services(Arc::new(RecordingProvider::new(Arc::default())));
        let runtime = SessionRuntime::new(services.clone());
        let parent = runtime
            .new_session(crate::SessionInit::default())
            .await
            .unwrap();
        let child = runtime
            .new_session(crate::SessionInit::default())
            .await
            .unwrap();
        let control = runtime.subagents.clone();
        let agent = AgentId::new();
        control
            .restore(
                parent.session_id(),
                [SubagentRecord {
                    status: SubagentStatus {
                        agent_id: agent.clone(),
                        session_id: child.session_id().clone(),
                        agent_type: None,
                        task: "historical task".to_owned(),
                        state: SubagentState::Closed,
                        last_message: None,
                        usage: None,
                        error: None,
                    },
                    generation: 1,
                }],
            )
            .await;
        assert!(control.agent_ids().await.contains(&agent));
        assert!(!control.owns_session(child.session_id()).await);
        let (reopened, _) = runtime.resume_session(child.session_id()).await.unwrap();
        let shell = services.tool_sessions.shell_session(child.session_id());
        control
            .close(CloseSubagentRequest {
                target: agent,
                timeout_ms: None,
            })
            .await
            .unwrap();
        control.force_close_session(parent.session_id()).await;
        control.close_session(parent.session_id()).await.unwrap();
        assert!(Arc::ptr_eq(
            &shell,
            &services.tool_sessions.shell_session(child.session_id())
        ));
        assert_shell_accepts_work(&services, child.session_id()).await;
        reopened.shutdown(None).await.unwrap();
    }

    async fn assert_shell_accepts_work(services: &RuntimeServices, session_id: &SessionId) {
        let result = ShellTool
            .execute(
                ToolContext {
                    session_id: session_id.clone(),
                    working_dir: std::env::temp_dir(),
                    path_locks: services.path_locks.clone(),
                    tool_sessions: services.tool_sessions.clone(),
                    snapshot: Arc::new(halter_protocol::ResourceSnapshot::empty()),
                    cancel: CancellationToken::new(),
                    emit: Arc::new(NoopToolEventSink),
                    policy: services.policy.clone(),
                    shell_timeout_secs: 5,
                    subagent_parent: None,
                },
                serde_json::json!({"command": "true"}),
            )
            .await
            .unwrap();
        let halter_protocol::ToolResult::Json { value } = result else {
            panic!("shell result");
        };
        assert_eq!(
            value["cancelled"], false,
            "historical parent closure must not seal the new child"
        );
        assert_eq!(value["exit_code"], 0);
    }

    #[derive(Debug, thiserror::Error)]
    #[error("child repair unavailable")]
    struct ChildRepairUnavailable;

    struct GatedRepairStore {
        inner: InMemorySessionStore,
        child: SessionId,
        fail_once: std::sync::atomic::AtomicBool,
        started: Notify,
        release: tokio::sync::Semaphore,
    }

    #[async_trait]
    impl SessionStore for GatedRepairStore {
        async fn create_session(&self, session: StoredSession) -> anyhow::Result<()> {
            self.inner.create_session(session).await
        }
        async fn load_session(&self, id: &SessionId) -> anyhow::Result<Option<StoredSession>> {
            if id == &self.child && self.fail_once.swap(false, Ordering::SeqCst) {
                self.started.notify_one();
                self.release.acquire().await.unwrap().forget();
                return Err(ChildRepairUnavailable.into());
            }
            self.inner.load_session(id).await
        }
        async fn commit(
            &self,
            id: &SessionId,
            snapshot: Option<Arc<halter_protocol::ResourceSnapshot>>,
            expected: Option<u64>,
            state: Option<halter_protocol::SessionState>,
            events: Vec<halter_protocol::PendingEvent>,
        ) -> anyhow::Result<Vec<halter_protocol::SessionEvent>> {
            self.inner
                .commit(id, snapshot, expected, state, events)
                .await
        }
        async fn replay(
            &self,
            id: &SessionId,
        ) -> anyhow::Result<Vec<halter_protocol::SessionEvent>> {
            self.inner.replay(id).await
        }
        async fn list_sessions(&self) -> anyhow::Result<Vec<halter_protocol::SessionBlueprint>> {
            self.inner.list_sessions().await
        }
    }

    #[tokio::test]
    async fn subagent_activity_cleans_failed_and_dropped_start_hooks() {
        for (drop_spawn, close_parent) in [(false, false), (true, false), (false, true)] {
            let parent = parent_context();
            let store = Arc::new(GatedRepairStore {
                inner: Default::default(),
                child: parent.blueprint.session_id.clone(),
                fail_once: std::sync::atomic::AtomicBool::new(true),
                started: Notify::new(),
                release: tokio::sync::Semaphore::new(0),
            });
            let mut services = (*test_services(Arc::new(PendingProvider))).clone();
            services.sessions = store.clone();
            let services = Arc::new(services);
            store_parent_session(&services, &parent).await;
            let control = RuntimeSubagentControl::new(services.clone());
            let mut activity = control.subscribe_activity();
            let spawning = tokio::spawn({
                let control = control.clone();
                let parent = parent.clone();
                async move {
                    control
                        .spawn(&parent, spawn_request("starts"), CancellationToken::new())
                        .await
                }
            });
            timeout(Duration::from_secs(5), store.started.notified())
                .await
                .unwrap();
            assert!(control.has_running_subagents(&parent.blueprint.session_id));
            let child = control
                .inner
                .registry()
                .entries
                .values()
                .next()
                .unwrap()
                .status
                .session_id
                .clone();
            let _ = services.tool_sessions.shell_session(&child);
            activity.borrow_and_update();
            if close_parent {
                control
                    .close_session(&parent.blueprint.session_id)
                    .await
                    .unwrap();
                assert!(!services.tool_sessions.has_process_state(&child));
            }
            if drop_spawn {
                spawning.abort();
                assert!(spawning.await.unwrap_err().is_cancelled());
            } else {
                store.release.add_permits(1);
                let error = spawning.await.unwrap().unwrap_err();
                assert!(
                    error
                        .chain()
                        .any(|cause| cause.is::<ChildRepairUnavailable>())
                );
            }
            wait_for_no_subagent_activity(&control, &parent.blueprint.session_id, &mut activity)
                .await;
            assert!(!services.tool_sessions.has_process_state(&child));
            assert!(!control.owns_session(&child).await);
            assert!(control.inner.registry().entries.is_empty());
            assert!(
                services
                    .sessions
                    .load_session(&parent.blueprint.session_id)
                    .await
                    .unwrap()
                    .unwrap()
                    .state
                    .subagents
                    .is_empty(),
                "an unpublished failed spawn never enters the parent's checkpoint"
            );
            assert!(
                services
                    .sessions
                    .replay(&parent.blueprint.session_id)
                    .await
                    .unwrap()
                    .iter()
                    .all(|event| !matches!(
                        event.payload,
                        halter_protocol::SessionEventPayload::SubagentUpdated { .. }
                    )),
                "failed reservations never publish parent status records"
            );
            assert!(
                services
                    .sessions
                    .replay(&child)
                    .await
                    .unwrap()
                    .iter()
                    .any(|event| matches!(
                        event.payload,
                        halter_protocol::SessionEventPayload::SessionShutdownComplete
                    )),
                "the already-created failed child still completes session cleanup"
            );
        }
    }

    #[tokio::test]
    async fn failed_seeded_creation_does_not_publish_an_uncreated_subagent() {
        for sqlite in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let mut services = test_services(Arc::new(PendingProvider));
            if sqlite {
                Arc::get_mut(&mut services).unwrap().sessions = Arc::new(
                    halter_session::SqliteSessionStore::open(root.path().join("sessions.db"))
                        .unwrap(),
                );
            }
            let mut parent = parent_context();
            parent.blueprint.snapshot_revision = parent.snapshot.revision.clone();
            store_parent_session(&services, &parent).await;
            let control = RuntimeSubagentControl::new(services.clone());
            let mut request = spawn_request("never created");
            request.model = Some(ModelId::from("missing-model"));
            control
                .spawn(&parent, request, CancellationToken::new())
                .await
                .unwrap_err();
            assert_eq!(services.sessions.list_sessions().await.unwrap().len(), 1);
            assert!(control.inner.registry().entries.is_empty());
            assert!(!control.has_running_subagents(&parent.blueprint.session_id));
            let stored = services
                .sessions
                .load_session(&parent.blueprint.session_id)
                .await
                .unwrap()
                .unwrap();
            assert!(stored.state.subagents.is_empty());
            assert!(
                services
                    .sessions
                    .replay(&parent.blueprint.session_id)
                    .await
                    .unwrap()
                    .is_empty()
            );
        }
    }

    #[tokio::test]
    async fn session_end_skips_uncreated_children_but_closes_created_children_without_start_log() {
        for sqlite in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let mut services = test_services(Arc::new(PendingProvider));
            if sqlite {
                Arc::get_mut(&mut services).unwrap().sessions = Arc::new(
                    halter_session::SqliteSessionStore::open(root.path().join("sessions.db"))
                        .unwrap(),
                );
            }
            let control = RuntimeSubagentControl::new(services.clone());
            let child = SessionId::new();
            control.end_session(&child).await.unwrap();
            assert!(services.sessions.list_sessions().await.unwrap().is_empty());
            let mut context = parent_context();
            context.blueprint.session_id = child.clone();
            context.blueprint.snapshot_revision = context.snapshot.revision.clone();
            services
                .sessions
                .create_session(StoredSession::new(
                    context.blueprint,
                    halter_protocol::SessionState::default(),
                    context.snapshot,
                ))
                .await
                .unwrap();
            assert!(services.sessions.replay(&child).await.unwrap().is_empty());
            control.end_session(&child).await.unwrap();
            control.end_session(&child).await.unwrap();
            assert_eq!(
                services
                    .sessions
                    .replay(&child)
                    .await
                    .unwrap()
                    .iter()
                    .filter(|event| matches!(
                        event.payload,
                        halter_protocol::SessionEventPayload::SessionShutdownComplete
                    ))
                    .count(),
                1,
                "an existing child ends once even when its SessionStarted append never happened"
            );
        }
    }

    #[tokio::test]
    async fn subagent_activity_settles_unexpected_execution_abort() {
        let services = test_services(Arc::new(PendingProvider));
        let control = RuntimeSubagentControl::new(services.clone());
        let parent = parent_context();
        store_parent_session(&services, &parent).await;
        let mut activity = control.subscribe_activity();
        let child = control
            .spawn(&parent, spawn_request("running"), CancellationToken::new())
            .await
            .unwrap();
        assert!(control.has_running_subagents(&parent.blueprint.session_id));
        activity.borrow_and_update();
        control.inner.registry().entries[&child.agent_id.0]
            .running
            .as_ref()
            .unwrap()
            .join_handle
            .abort();
        wait_for_no_subagent_activity(&control, &parent.blueprint.session_id, &mut activity).await;
        let status = control.inner.registry().entries[&child.agent_id.0]
            .status
            .clone();
        assert_eq!(status.state, SubagentState::Failed);
        assert!(
            services
                .sessions
                .load_session(&child.session_id)
                .await
                .unwrap()
                .unwrap()
                .state
                .open_turn
                .is_none()
        );
        control
            .close(CloseSubagentRequest {
                target: child.agent_id,
                timeout_ms: None,
            })
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn closing_subagent_wins_over_paused_final_status_recording() {
        let parent = parent_context();
        let store = Arc::new(GatedRepairStore {
            inner: Default::default(),
            child: parent.blueprint.session_id.clone(),
            fail_once: std::sync::atomic::AtomicBool::new(false),
            started: Notify::new(),
            release: tokio::sync::Semaphore::new(0),
        });
        let mut services = (*test_services(Arc::new(PendingProvider))).clone();
        services.sessions = store.clone();
        let services = Arc::new(services);
        store_parent_session(&services, &parent).await;
        let runtime = SessionRuntime::new(services.clone());
        let child = runtime
            .new_session(crate::SessionInit::default())
            .await
            .unwrap();
        let control = RuntimeSubagentControl::new(services);
        let agent = AgentId::new();
        let cancel = CancellationToken::new();
        let mut activity = control.subscribe_activity();
        store.fail_once.store(true, Ordering::SeqCst);
        {
            let mut registry = control.inner.registry();
            let task = tokio::spawn({
                let control = control.clone();
                let agent = agent.clone();
                async move {
                    control
                        .finish_turn(
                            agent,
                            1,
                            TurnOutcome {
                                state: SubagentState::Completed,
                                last_message: Some("finished".to_owned()),
                                usage: None,
                                error: None,
                            },
                        )
                        .await;
                }
            });
            registry.entries.insert(
                agent.0.clone(),
                RegisteredSubagent {
                    parent: parent.blueprint.session_id.clone(),
                    status: SubagentStatus {
                        agent_id: agent.clone(),
                        session_id: child.session_id().clone(),
                        agent_type: None,
                        task: "work".to_owned(),
                        state: SubagentState::Running,
                        last_message: None,
                        usage: None,
                        error: None,
                    },
                    generation: 1,
                    running: Some(RunningTurn {
                        cancel: cancel.clone(),
                        join_handle: task,
                        current_turn: Arc::new(std::sync::Mutex::new(None)),
                    }),
                    closure: None,
                },
            );
        }
        timeout(Duration::from_secs(5), store.started.notified())
            .await
            .unwrap();
        assert!(control.has_running_subagents(&parent.blueprint.session_id));
        let closing = tokio::spawn({
            let control = control.clone();
            let agent = agent.clone();
            async move {
                control
                    .close(CloseSubagentRequest {
                        target: agent,
                        timeout_ms: None,
                    })
                    .await
            }
        });
        timeout(Duration::from_secs(5), cancel.cancelled())
            .await
            .unwrap();
        assert_eq!(
            control.inner.registry().entries[&agent.0].status.state,
            SubagentState::Closed
        );
        assert!(control.has_running_subagents(&parent.blueprint.session_id));
        activity.borrow_and_update();
        store.release.add_permits(1);
        timeout(Duration::from_secs(5), closing)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        wait_for_no_subagent_activity(&control, &parent.blueprint.session_id, &mut activity).await;
        assert_eq!(
            control.inner.registry().entries[&agent.0].status.state,
            SubagentState::Closed
        );
    }

    #[tokio::test]
    async fn failed_child_repair_still_cleans_resources_and_repeated_close_retains_error() {
        for agent_api in [false, true] {
            let child_id = SessionId::new();
            let store = Arc::new(GatedRepairStore {
                inner: Default::default(),
                child: child_id.clone(),
                fail_once: std::sync::atomic::AtomicBool::new(false),
                started: Notify::new(),
                release: tokio::sync::Semaphore::new(0),
            });
            let mut services = (*test_services(Arc::new(PendingProvider))).clone();
            services.sessions = store.clone();
            let services = Arc::new(services);
            let runtime = SessionRuntime::new(services.clone());
            let parent = runtime
                .new_session(crate::SessionInit::default())
                .await
                .unwrap();
            runtime
                .new_session(crate::SessionInit {
                    session_id: Some(child_id.clone()),
                    ..Default::default()
                })
                .await
                .unwrap();
            let control = runtime.subagents.clone();
            let agent_id = AgentId::new();
            let cancel = CancellationToken::new();
            let task_cancel = cancel.clone();
            control.inner.registry().entries.insert(
                agent_id.0.clone(),
                RegisteredSubagent {
                    parent: parent.session_id().clone(),
                    generation: 1,
                    closure: None,
                    status: SubagentStatus {
                        agent_id: agent_id.clone(),
                        session_id: child_id.clone(),
                        agent_type: None,
                        task: "repair before closing".to_owned(),
                        state: SubagentState::Running,
                        last_message: None,
                        usage: None,
                        error: None,
                    },
                    running: Some(RunningTurn {
                        cancel,
                        join_handle: tokio::spawn(async move { task_cancel.cancelled().await }),
                        current_turn: Arc::new(std::sync::Mutex::new(Some(TurnId::new()))),
                    }),
                },
            );
            let _ = services.tool_sessions.shell_session(&child_id);
            store.fail_once.store(true, Ordering::SeqCst);
            let closing = tokio::spawn({
                let control = control.clone();
                let agent_id = agent_id.clone();
                let parent_id = parent.session_id().clone();
                async move {
                    if agent_api {
                        control
                            .close(CloseSubagentRequest {
                                target: agent_id,
                                timeout_ms: None,
                            })
                            .await
                            .map(|_| ())
                    } else {
                        control.close_session(&parent_id).await
                    }
                }
            });
            timeout(Duration::from_secs(5), store.started.notified())
                .await
                .unwrap();
            assert!(
                control.owns_session(&child_id).await,
                "repair retains the incarnation fence"
            );
            assert!(matches!(
                runtime.resume_session(&child_id).await,
                Err(crate::SessionError::AlreadyOpen(_))
            ));
            let error = if agent_api {
                store.release.add_permits(1);
                timeout(Duration::from_secs(5), closing)
                    .await
                    .unwrap()
                    .unwrap()
                    .unwrap_err()
            } else {
                closing.abort();
                assert!(closing.await.unwrap_err().is_cancelled());
                assert!(
                    control.owns_session(&child_id).await,
                    "a dropped caller does not drop the cleanup owner"
                );
                store.release.add_permits(1);
                timeout(
                    Duration::from_secs(5),
                    control.close(CloseSubagentRequest {
                        target: agent_id.clone(),
                        timeout_ms: None,
                    }),
                )
                .await
                .unwrap()
                .unwrap_err()
            };
            assert!(
                error
                    .chain()
                    .any(|cause| cause.is::<ChildRepairUnavailable>())
            );
            assert!(
                !services.tool_sessions.has_process_state(&child_id),
                "repair errors cannot skip resource cleanup"
            );
            let mut activity = control.subscribe_activity();
            timeout(Duration::from_secs(5), async {
                while control.owns_session(&child_id).await {
                    activity.changed().await.unwrap();
                }
            })
            .await
            .expect("the remaining ancestor cleanup fence settles");
            assert!(!control.owns_session(&child_id).await);
            for _ in 0..2 {
                let repeated = control
                    .close(CloseSubagentRequest {
                        target: agent_id.clone(),
                        timeout_ms: None,
                    })
                    .await
                    .unwrap_err();
                assert!(
                    repeated
                        .chain()
                        .any(|cause| cause.is::<ChildRepairUnavailable>())
                );
            }
            let (reopened, _) = runtime.resume_session(&child_id).await.unwrap();
            assert_shell_accepts_work(&services, &child_id).await;
            reopened.shutdown(None).await.unwrap();
        }
    }

    #[tokio::test]
    async fn close_session_waits_for_descendants_and_preserves_unrelated_agents() {
        let services = test_services(Arc::new(PendingProvider));
        let runtime = SessionRuntime::new(services.clone());
        let parent = runtime
            .new_session(crate::SessionInit::default())
            .await
            .unwrap();
        let child = runtime
            .new_session(crate::SessionInit::default())
            .await
            .unwrap();
        let grandchild = runtime
            .new_session(crate::SessionInit::default())
            .await
            .unwrap();
        let unrelated = runtime
            .new_session(crate::SessionInit::default())
            .await
            .unwrap();
        let control = RuntimeSubagentControl::new(services);
        let mut activity = control.subscribe_activity();
        let cleanup_release = Arc::new(tokio::sync::Semaphore::new(0));
        let (cleanup_started, mut cleanup_events) = tokio::sync::mpsc::channel(2);
        let child_id = AgentId::from("child");
        let grandchild_id = AgentId::from("grandchild");
        let unrelated_id = AgentId::from("unrelated");
        let unrelated_cancel = CancellationToken::new();
        {
            let mut registry = control.inner.registry();
            for (agent_id, session_id, parent_id) in [
                (
                    child_id.clone(),
                    child.session_id().clone(),
                    parent.session_id().clone(),
                ),
                (
                    grandchild_id.clone(),
                    grandchild.session_id().clone(),
                    child.session_id().clone(),
                ),
                (
                    unrelated_id.clone(),
                    unrelated.session_id().clone(),
                    SessionId::from("other-parent"),
                ),
            ] {
                let cancel = if agent_id == unrelated_id {
                    unrelated_cancel.clone()
                } else {
                    CancellationToken::new()
                };
                let task_cancel = cancel.clone();
                let release = cleanup_release.clone();
                let started = cleanup_started.clone();
                let join_handle = tokio::spawn(async move {
                    task_cancel.cancelled().await;
                    started.send(()).await.unwrap();
                    let _permit = release.acquire().await.unwrap();
                });
                registry.entries.insert(
                    agent_id.0.clone(),
                    RegisteredSubagent {
                        parent: parent_id,
                        closure: None,
                        status: SubagentStatus {
                            agent_id,
                            session_id,
                            agent_type: None,
                            task: "work".to_owned(),
                            state: SubagentState::Running,
                            last_message: None,
                            usage: None,
                            error: None,
                        },
                        generation: 1,
                        running: Some(RunningTurn {
                            cancel,
                            join_handle,
                            current_turn: Arc::new(std::sync::Mutex::new(None)),
                        }),
                    },
                );
            }
        }
        let closing = control.clone();
        let parent_id = parent.session_id().clone();
        let shutdown = tokio::spawn(async move { closing.close_session(&parent_id).await });
        for _ in 0..2 {
            timeout(Duration::from_secs(5), cleanup_events.recv())
                .await
                .unwrap()
                .unwrap();
        }
        assert!(
            !shutdown.is_finished(),
            "closure must await foreground cleanup"
        );
        assert!(!unrelated_cancel.is_cancelled());
        assert!(control.has_running_subagents(parent.session_id()));
        assert!(control.has_running_subagents(child.session_id()));
        {
            let registry = control.inner.registry();
            for id in [&child_id, &grandchild_id] {
                assert_eq!(registry.entries[&id.0].status.state, SubagentState::Closed);
            }
            assert_eq!(
                registry.entries[&unrelated_id.0].status.state,
                SubagentState::Running
            );
            assert!(registry.closing_sessions.contains_key(parent.session_id()));
        }
        let mut context = parent_context();
        context.blueprint.session_id = parent.session_id().clone();
        let error = control
            .spawn(
                &context,
                spawn_request("late child"),
                CancellationToken::new(),
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("parent session is closing"));
        activity.borrow_and_update();
        cleanup_release.add_permits(2);
        timeout(Duration::from_secs(5), shutdown)
            .await
            .unwrap()
            .unwrap()
            .unwrap();
        wait_for_no_subagent_activity(&control, parent.session_id(), &mut activity).await;
        control.close_session(parent.session_id()).await.unwrap();
        for descendant in [&child, &grandchild] {
            let events = descendant.replay().await.unwrap();
            assert_eq!(
                events
                    .iter()
                    .filter(|event| matches!(
                        event.payload,
                        halter_protocol::SessionEventPayload::SessionShutdownComplete
                    ))
                    .count(),
                1,
                "each owned child session ends once"
            );
        }
        assert!(
            !control
                .inner
                .registry()
                .closing_sessions
                .contains_key(parent.session_id())
        );
        let running = control
            .inner
            .registry()
            .entries
            .get_mut(&unrelated_id.0)
            .unwrap()
            .running
            .take()
            .unwrap();
        running.join_handle.abort();
        let _ = running.join_handle.await;
    }

    #[tokio::test]
    async fn parent_close_settles_descendants_before_joining_a_previous_child_close() {
        let services = test_services(Arc::new(PendingProvider));
        let runtime = SessionRuntime::new(services.clone());
        let parent = runtime
            .new_session(crate::SessionInit::default())
            .await
            .unwrap();
        let child = runtime
            .new_session(crate::SessionInit::default())
            .await
            .unwrap();
        let grandchild = runtime
            .new_session(crate::SessionInit::default())
            .await
            .unwrap();
        let control = RuntimeSubagentControl::new(services);
        let child_agent = AgentId::new();
        let grandchild_agent = AgentId::new();
        let child_cancel = CancellationToken::new();
        let grandchild_cancel = CancellationToken::new();
        let release_child = Arc::new(tokio::sync::Semaphore::new(0));
        let release_grandchild = Arc::new(tokio::sync::Semaphore::new(0));
        for (agent, session, owner, cancel, release) in [
            (
                child_agent.clone(),
                child.session_id().clone(),
                parent.session_id().clone(),
                child_cancel.clone(),
                release_child.clone(),
            ),
            (
                grandchild_agent,
                grandchild.session_id().clone(),
                child.session_id().clone(),
                grandchild_cancel.clone(),
                release_grandchild.clone(),
            ),
        ] {
            let task_cancel = cancel.clone();
            control.inner.registry().entries.insert(
                agent.0.clone(),
                RegisteredSubagent {
                    parent: owner,
                    generation: 1,
                    closure: None,
                    status: SubagentStatus {
                        agent_id: agent,
                        session_id: session,
                        agent_type: None,
                        task: "nested cleanup".to_owned(),
                        state: SubagentState::Running,
                        last_message: None,
                        usage: None,
                        error: None,
                    },
                    running: Some(RunningTurn {
                        cancel,
                        join_handle: tokio::spawn(async move {
                            task_cancel.cancelled().await;
                            release.acquire().await.unwrap().forget();
                        }),
                        current_turn: Arc::new(std::sync::Mutex::new(None)),
                    }),
                },
            );
        }
        let child_closure = tokio::spawn({
            let control = control.clone();
            async move {
                control
                    .close(CloseSubagentRequest {
                        target: child_agent,
                        timeout_ms: None,
                    })
                    .await
            }
        });
        timeout(Duration::from_secs(5), child_cancel.cancelled())
            .await
            .unwrap();
        let parent_closure = tokio::spawn({
            let control = control.clone();
            let parent_id = parent.session_id().clone();
            async move { control.close_session(&parent_id).await }
        });
        timeout(Duration::from_secs(5), grandchild_cancel.cancelled())
            .await
            .unwrap();
        release_child.add_permits(1);
        release_grandchild.add_permits(1);
        timeout(Duration::from_secs(5), async {
            child_closure.await.unwrap().unwrap();
            parent_closure.await.unwrap().unwrap();
        })
        .await
        .expect("overlapping cleanup cannot wait on itself");
        assert!(!control.owns_session(child.session_id()).await);
        assert!(!control.owns_session(grandchild.session_id()).await);
    }

    #[tokio::test]
    async fn wait_without_timeout_returns_when_cancelled() {
        let services = test_services(Arc::new(PendingProvider));
        let control = RuntimeSubagentControl::new(services);
        let spawned = control
            .spawn(
                &parent_context(),
                spawn_request("running task"),
                CancellationToken::new(),
            )
            .await
            .expect("spawn");
        let cancel = CancellationToken::new();
        let waiting = control.wait(
            WaitSubagentRequest {
                targets: vec![spawned.agent_id.clone()],
                timeout_ms: None,
            },
            cancel.clone(),
        );
        cancel.cancel();

        let error = tokio::time::timeout(Duration::from_secs(5), waiting)
            .await
            .expect("cancelled wait must not block")
            .expect_err("cancelled wait must fail");

        assert!(error.to_string().contains("cancelled"), "{error}");
        let still_running = control
            .wait(
                WaitSubagentRequest {
                    targets: vec![spawned.agent_id],
                    timeout_ms: Some(10),
                },
                CancellationToken::new(),
            )
            .await
            .expect("wait");
        assert!(
            still_running.timed_out,
            "cancelling the wait must not cancel the subagent"
        );
    }

    #[tokio::test]
    async fn spawn_with_cancelled_token_starts_no_subagent() {
        let services = test_services(Arc::new(PendingProvider));
        let control = RuntimeSubagentControl::new(services);
        let cancel = CancellationToken::new();
        cancel.cancel();

        let error = control
            .spawn(&parent_context(), spawn_request("never runs"), cancel)
            .await
            .expect_err("cancelled spawn must fail");

        assert!(error.to_string().contains("cancelled"), "{error}");
        assert!(control.inner.registry().entries.is_empty());
    }

    fn spawn_request(message: &str) -> SpawnSubagentRequest {
        SpawnSubagentRequest {
            message: message.to_owned(),
            agent_type: None,
            fork_context: false,
            model: None,
        }
    }

    /// Another process over `services`' store, with nothing in memory.
    fn restarted(services: &Arc<RuntimeServices>, provider: Arc<dyn Provider>) -> SessionRuntime {
        let mut restarted = test_services(provider);
        Arc::get_mut(&mut restarted).expect("unique").sessions = services.sessions.clone();
        SessionRuntime::new(restarted)
    }

    async fn status_of(
        control: &Arc<dyn SubagentControl>,
        agent: &AgentId,
        timeout_ms: u64,
    ) -> SubagentStatus {
        let waited = control
            .wait(
                WaitSubagentRequest {
                    targets: vec![agent.clone()],
                    timeout_ms: Some(timeout_ms),
                },
                CancellationToken::new(),
            )
            .await
            .expect("the agent is registered");
        waited
            .status
            .into_iter()
            .chain(waited.target_statuses)
            .next()
            .expect("status")
    }

    async fn last_recorded(services: &Arc<RuntimeServices>) -> Option<SubagentStatus> {
        let log = services
            .sessions
            .replay(&SessionId::from("parent"))
            .await
            .expect("replay");
        log.into_iter().rev().find_map(|event| match event.payload {
            halter_protocol::SessionEventPayload::SubagentUpdated { record } => Some(record.status),
            _ => None,
        })
    }

    /// #210: a resumed parent rebuilds its subagents from its own log.
    #[tokio::test]
    async fn resume_rebuilds_subagents_from_the_parent_log() {
        fn completes() -> Arc<dyn Provider> {
            Arc::new(RecordingProvider::new(Arc::default()))
        }
        fn pends() -> Arc<dyn Provider> {
            Arc::new(PendingProvider)
        }
        let interrupted = Some(crate::session::INTERRUPTED_SUBAGENT);
        type Case = (
            &'static str,
            fn() -> Arc<dyn Provider>,
            bool,
            SubagentState,
            Option<&'static str>,
        );
        let cases: [Case; 3] = [
            (
                "finished before the stop",
                completes,
                true,
                SubagentState::Completed,
                None,
            ),
            (
                "running at the stop",
                pends,
                true,
                SubagentState::Cancelled,
                interrupted,
            ),
            (
                "running in this process",
                pends,
                false,
                SubagentState::Running,
                None,
            ),
        ];
        for (name, provider, restart, state, error) in cases {
            let services = test_services(provider());
            let parent = parent_context();
            store_parent_session(&services, &parent).await;
            let first = SessionRuntime::new(services.clone());
            let agent = first
                .subagent_control()
                .spawn(&parent, spawn_request("task"), CancellationToken::new())
                .await
                .expect("spawn")
                .agent_id;
            let settle = if state == SubagentState::Completed {
                5_000
            } else {
                50
            };
            status_of(&first.subagent_control(), &agent, settle).await;

            let resumed = if restart {
                restarted(&services, provider())
            } else {
                first
            };
            resumed
                .resume(&parent.blueprint.session_id)
                .await
                .expect("resume")
                .expect("found");

            let status = status_of(&resumed.subagent_control(), &agent, 50).await;
            assert_eq!(
                (status.state, status.error.as_deref()),
                (state, error),
                "{name}"
            );
            assert_eq!(
                status.last_message.is_some(),
                state == SubagentState::Completed,
                "{name}"
            );
            assert_eq!(last_recorded(&services).await, Some(status), "{name}");
        }
    }

    #[tokio::test]
    async fn restored_subagents_take_input_and_close() {
        let services = test_services(Arc::new(PendingProvider));
        let parent = parent_context();
        store_parent_session(&services, &parent).await;
        let spawned = SessionRuntime::new(services.clone())
            .subagent_control()
            .spawn(&parent, spawn_request("task"), CancellationToken::new())
            .await
            .expect("spawn");
        let agent = spawned.agent_id;
        // The first process stops once the child's turn is under way.
        tokio::time::timeout(Duration::from_secs(5), async {
            while services
                .sessions
                .load_session(&spawned.session_id)
                .await
                .expect("load")
                .is_none_or(|stored| stored.state.open_turn.is_none())
            {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("child turn started");

        // The restarted process traces, so the restored child must rejoin
        // the parent's trace.
        let traces = tempfile::tempdir().expect("tempdir");
        let mut restarted = test_services(Arc::new(RecordingProvider::new(Arc::default())));
        let unique = Arc::get_mut(&mut restarted).expect("unique");
        unique.sessions = services.sessions.clone();
        unique.trace_recorder = Some(Arc::new(
            crate::TraceRecorder::open(traces.path().to_path_buf()).expect("recorder"),
        ));
        let runtime = SessionRuntime::new(restarted);
        runtime
            .resume(&parent.blueprint.session_id)
            .await
            .expect("resume")
            .expect("found");
        let control = runtime.subagent_control();
        assert_eq!(
            status_of(&control, &agent, 50).await.state,
            SubagentState::Cancelled
        );

        let input = |message: &str| SendSubagentInputRequest {
            target: agent.clone(),
            message: message.to_owned(),
        };
        control
            .send_input(input("continue"))
            .await
            .expect("send_input");
        let waited = control
            .wait(
                WaitSubagentRequest {
                    targets: vec![agent.clone()],
                    timeout_ms: Some(5_000),
                },
                CancellationToken::new(),
            )
            .await
            .expect("wait")
            .status
            .expect("finished");
        assert_eq!(waited.state, SubagentState::Completed, "{:?}", waited.error);
        assert_eq!(
            waited.last_message.as_deref(),
            Some("child reply [subagent/model]")
        );
        let trace = std::fs::read_to_string(traces.path().join("parent.txt")).expect("trace");
        assert!(
            trace.lines().any(|line| {
                let line: serde_json::Value = serde_json::from_str(line).expect("json");
                line.get("sequence").is_some() && line["session_id"] == spawned.session_id.0
            }),
            "{trace}"
        );

        control
            .close(CloseSubagentRequest {
                target: agent.clone(),
                timeout_ms: None,
            })
            .await
            .expect("close");
        assert_eq!(
            last_recorded(&services).await.map(|status| status.state),
            Some(SubagentState::Closed)
        );
        let error = control
            .send_input(input("again"))
            .await
            .expect_err("closed agents take no input");
        assert!(error.to_string().contains("is closed"), "{error}");
    }

    #[tokio::test]
    async fn spawn_respects_depth_policy() {
        let services = test_services(Arc::new(RecordingProvider::new(Arc::new(Mutex::new(
            Vec::new(),
        )))));
        let control = RuntimeSubagentControl::new(services);
        let mut parent = parent_context();
        parent.blueprint.subagent_depth = 3;

        let error = control
            .spawn(
                &parent,
                SpawnSubagentRequest {
                    message: "delegate this".to_owned(),
                    agent_type: None,
                    fork_context: true,
                    model: None,
                },
                CancellationToken::new(),
            )
            .await
            .expect_err("depth should fail");

        let message = error.to_string();
        assert!(
            message.contains("subagent limit reached: depth"),
            "expected typed SubagentLimit error, got: {message}"
        );
    }

    fn test_services(provider: Arc<dyn Provider>) -> Arc<RuntimeServices> {
        test_services_with_hooks(provider, halter_hooks::RegisteredHooks::default())
    }

    fn test_services_with_hooks(
        provider: Arc<dyn Provider>,
        registered_hooks: halter_hooks::RegisteredHooks,
    ) -> Arc<RuntimeServices> {
        let mut models = ModelRegistry::new();
        models.set_default_model(ResolvedModel {
            role: ModelRole::default(),
            id: ModelId::from("default"),
            provider: ProviderName::from("fake"),
            provider_kind: ProviderKind::Fake,
            api_kind: ApiKind::Fake,
            model: "default/model".to_owned(),
            max_input_tokens: Some(32_000),
            max_output_tokens: Some(4_096),
            reasoning: None,
            tokens_per_minute: None,
        });
        models.set_subagent_model(ResolvedModel {
            role: ModelRole::subagent(),
            id: ModelId::from("subagent"),
            provider: ProviderName::from("fake"),
            provider_kind: ProviderKind::Fake,
            api_kind: ApiKind::Fake,
            model: "subagent/model".to_owned(),
            max_input_tokens: Some(32_000),
            max_output_tokens: Some(4_096),
            reasoning: None,
            tokens_per_minute: None,
        });
        models.register_provider(ProviderName::from("fake"), provider);

        Arc::new(RuntimeServices {
            resources: Arc::new(ResourceHandle::new(
                halter_protocol::ResourceSnapshot::empty(),
                Arc::new(halter_hooks::Hooks::default()),
                Vec::new(),
            )),
            registered_hooks: Arc::new(registered_hooks),
            session_hook_store: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            models: Arc::new(models),
            tools: Arc::new(ToolRuntime::new()),
            path_locks: Arc::new(PathLockMap::default()),
            tool_sessions: Arc::new(ToolSessionStore::default()),
            sessions: Arc::new(InMemorySessionStore::default()),
            policy: Arc::new(DefaultToolPolicy::new(PolicySettings::default())),
            prompt_assembler: Arc::new(DefaultPromptAssembler),
            context_manager: Arc::new(DefaultContextManager),
            context: crate::ContextSettings::default(),
            compaction: Arc::new(crate::ModelSummary),

            event_bus: Arc::new(EventBus::default()),

            parent_streams: Arc::new(crate::ParentStreamRegistry::default()),
            turn_registry: Arc::new(crate::TurnRegistry::new()),
            session_leases: Arc::new(crate::SessionLeases::default()),
            subagent_event_forwarding: halter_protocol::SubagentEventForwarding::Off,
            subagent_event_forwarding_cap: 100_000,
            shell_timeout_secs: 30,
            trace_recorder: None,
        })
    }

    /// Persist the parent session so `run_subagent_stop_hooks` can load it
    /// and dispatch `SubagentStop` hooks against real parent state.
    async fn store_parent_session(services: &Arc<RuntimeServices>, parent: &SubagentParentContext) {
        services
            .sessions
            .create_session(halter_session::StoredSession::new(
                parent.blueprint.clone(),
                parent.state.clone(),
                parent.snapshot.clone(),
            ))
            .await
            .expect("store parent session");
    }

    /// Regression (M4): a `SubagentStop` hook that always blocks must not
    /// resubmit turns forever — the resubmission cap fails the subagent with
    /// a descriptive error after a bounded number of full provider turns.
    #[tokio::test]
    async fn always_blocking_stop_hook_trips_resubmission_cap() {
        let provider_requests = Arc::new(Mutex::new(Vec::<ProviderRequest>::new()));
        let mut registered = halter_hooks::RegisteredHooks::default();
        registered.register(
            halter_protocol::PluginId::from("internal"),
            halter_hooks::RegisteredHookPriority::AfterPlugins,
            halter_hooks::Hook::callback(
                halter_hooks::HookEventName::SubagentStop,
                |_input| async move { halter_hooks::HookResponse::block("do it again") },
            ),
        );
        let services = test_services_with_hooks(
            Arc::new(RecordingProvider::new(provider_requests.clone())),
            registered,
        );
        let control = RuntimeSubagentControl::new(services.clone());
        let parent = parent_context();
        store_parent_session(&services, &parent).await;
        // Keep a parent handle alive for the whole run, as a real parent
        // session would while its subagents are managed.
        let _parent_handle =
            SessionExecutor::new(services.clone(), parent.blueprint.session_id.clone())
                .expect("parent handle");

        let spawned = control
            .spawn(
                &parent,
                SpawnSubagentRequest {
                    message: "delegate this".to_owned(),
                    agent_type: None,
                    fork_context: false,
                    model: None,
                },
                CancellationToken::new(),
            )
            .await
            .expect("spawn");
        let waited = control
            .wait(
                WaitSubagentRequest {
                    targets: vec![spawned.agent_id],
                    timeout_ms: Some(30_000),
                },
                CancellationToken::new(),
            )
            .await
            .expect("wait");

        let status = waited.status.expect("terminal status");
        assert_eq!(status.state, SubagentState::Failed);
        let error = status.error.expect("cap error");
        assert!(
            error.contains("resubmission cap of 8 reached"),
            "unexpected error: {error}"
        );
        // Initial turn + capped resubmissions, then the loop stops.
        assert_eq!(
            provider_requests.lock().expect("requests").len(),
            9,
            "provider turns must stop at the cap"
        );
    }

    /// Regression (H1): `SubagentStop` dispatch builds temporary parent
    /// handles; dropping one between dispatches used to evict the parent's
    /// hook-store entry, so the next dispatch saw freshly instantiated
    /// (stateless) hooks. A stateful hook that blocks only on its first
    /// invocation proves state now survives across dispatches: the subagent
    /// completes after exactly one resubmission instead of looping to the cap.
    #[tokio::test]
    async fn stop_hook_state_persists_across_subagent_dispatches() {
        let provider_requests = Arc::new(Mutex::new(Vec::<ProviderRequest>::new()));
        let mut registered = halter_hooks::RegisteredHooks::default();
        registered.register(
            halter_protocol::PluginId::from("internal"),
            halter_hooks::RegisteredHookPriority::AfterPlugins,
            halter_hooks::Hook::function(halter_hooks::HookEventName::SubagentStop, || {
                let calls = Arc::new(Mutex::new(0usize));
                move |_input| {
                    let calls = calls.clone();
                    async move {
                        let mut calls = calls.lock().expect("calls");
                        *calls += 1;
                        if *calls == 1 {
                            halter_hooks::HookResponse::block("one more pass")
                        } else {
                            halter_hooks::HookResponse::passthrough()
                        }
                    }
                }
            }),
        );
        let services = test_services_with_hooks(
            Arc::new(RecordingProvider::new(provider_requests.clone())),
            registered,
        );
        let control = RuntimeSubagentControl::new(services.clone());
        let parent = parent_context();
        store_parent_session(&services, &parent).await;
        let _parent_handle =
            SessionExecutor::new(services.clone(), parent.blueprint.session_id.clone())
                .expect("parent handle");

        let spawned = control
            .spawn(
                &parent,
                SpawnSubagentRequest {
                    message: "delegate this".to_owned(),
                    agent_type: None,
                    fork_context: false,
                    model: None,
                },
                CancellationToken::new(),
            )
            .await
            .expect("spawn");
        let waited = control
            .wait(
                WaitSubagentRequest {
                    targets: vec![spawned.agent_id],
                    timeout_ms: Some(30_000),
                },
                CancellationToken::new(),
            )
            .await
            .expect("wait");

        let status = waited.status.expect("terminal status");
        assert_eq!(
            status.state,
            SubagentState::Completed,
            "hook state was lost between dispatches: {:?}",
            status.error
        );
        // First turn blocked once, second turn passed through.
        assert_eq!(provider_requests.lock().expect("requests").len(), 2);
    }

    fn parent_context() -> SubagentParentContext {
        SubagentParentContext {
            blueprint: halter_protocol::SessionBlueprint {
                session_id: SessionId::from("parent"),
                parent_session_id: None,
                default_model: ModelId::from("default"),
                subagent_model: ModelId::from("subagent"),
                subagent_event_forwarding: halter_protocol::SubagentEventForwarding::Off,
                snapshot_revision: halter_protocol::Revision::from("revision"),
                working_dir: ".".into(),
                system_prompt_seed: Vec::new(),
                max_turns: None,
                subagent_depth: 0,
            },
            state: halter_protocol::SessionState {
                messages: vec![Message::User(halter_protocol::UserMessage::text(
                    "root context",
                ))],
                ..halter_protocol::SessionState::default()
            },
            snapshot: Arc::new(halter_protocol::ResourceSnapshot::empty()),
            model: ModelId::from("default"),
            subagent_model: ModelId::from("subagent"),
        }
    }

    struct PendingProvider;

    struct GatedProvider {
        release: CancellationToken,
    }

    #[async_trait]
    impl Provider for GatedProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }

        async fn stream(
            &self,
            request: ProviderRequest,
            cancel: CancellationToken,
        ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
            let events = RecordingProvider::new(Arc::default())
                .stream(request, cancel)
                .await?;
            let release = self.release.clone();
            Ok(stream::once(async move {
                release.cancelled().await;
                events
            })
            .flatten()
            .boxed())
        }
    }

    #[async_trait]
    impl Provider for PendingProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }

        async fn stream(
            &self,
            _request: ProviderRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
            Ok(stream::pending().boxed())
        }
    }

    struct RecordingProvider {
        requests: Arc<Mutex<Vec<ProviderRequest>>>,
    }

    impl RecordingProvider {
        fn new(requests: Arc<Mutex<Vec<ProviderRequest>>>) -> Self {
            Self { requests }
        }
    }

    #[async_trait]
    impl Provider for RecordingProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }

        async fn stream(
            &self,
            request: ProviderRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
            self.requests
                .lock()
                .expect("requests")
                .push(request.clone());
            let message_id = halter_protocol::MessageId::new();
            let block_id = BlockId::new();
            Ok(stream::iter(vec![
                Ok(StreamEvent::MessageStart {
                    id: message_id.clone(),
                }),
                Ok(StreamEvent::TextStart {
                    id: block_id.clone(),
                }),
                Ok(StreamEvent::TextDelta {
                    id: block_id.clone(),
                    delta: format!("child reply [{}]", request.model.model),
                }),
                Ok(StreamEvent::TextEnd {
                    id: block_id.clone(),
                }),
                Ok(StreamEvent::UsageUpdate {
                    usage: Usage {
                        input_tokens: 10,
                        output_tokens: 4,
                        cache_creation_input_tokens: 0,
                        cache_read_input_tokens: 0,
                    },
                }),
                Ok(StreamEvent::MessageEnd {
                    id: message_id,
                    stop_reason: StopReason::EndTurn,
                    response_id: None,
                }),
            ])
            .boxed())
        }
    }
}
