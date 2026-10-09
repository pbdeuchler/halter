// pattern: Imperative Shell
//
// One writer per session. A turn (or a session-level operation such as
// `compact`) holds the session's write lease for its whole duration and
// commits against its own in-memory state; the next writer waits for it. Hook dispatches that originate
// outside that writer — subagent lifecycle hooks, `notify` — queue behind
// the lease instead of racing it on `expected_head_sequence`, and are
// committed when the lease is released. With no lease held they commit
// immediately, under the same lock, so a writer cannot start mid-commit.
// Subagent records touch no transcript, so the holder also takes them at each
// of its own commits (`take_subagent_records`) rather than at release.
//
// A subagent record whose commit fails is not dropped: it waits as
// undelivered and goes out ahead of every newer write for its session, seeded
// into the next lease or committed with the next immediate dispatch. Records
// carry generations, so a late delivery cannot override a newer generation,
// and delivering in order keeps it from overwriting a newer record of the
// same generation (`Running` then `Completed` share one). Later
// dispatches read the `once` hook ids of queued ones (`queued_hook_ids`),
// since the checkpoint has not caught up with them.

use std::collections::hash_map::Entry;
use std::collections::{BTreeSet, HashMap};
use std::future::Future;

use halter_protocol::{SessionId, SubagentRecord};
use tokio::sync::{Mutex, Notify};
use tracing::warn;

use crate::ExecutedHookDispatch;
use crate::session::SessionExecutor;

#[derive(Default)]
/// Per-session write leases plus the writes queued behind them.
pub struct SessionLeases {
    writes: Mutex<Writes>,
    released: Notify,
}

#[derive(Default)]
struct Writes {
    /// Sessions with a lease holder, and the writes queued behind it.
    held: HashMap<SessionId, Vec<OutOfTurn>>,
    /// Subagent records whose commit failed, oldest first.
    undelivered: HashMap<SessionId, Vec<SubagentRecord>>,
}

impl Writes {
    /// Hold `records` for redelivery ahead of anything already waiting.
    fn hold_undelivered(&mut self, session_id: &SessionId, mut records: Vec<SubagentRecord>) {
        if records.is_empty() {
            return;
        }
        let waiting = self.undelivered.entry(session_id.clone()).or_default();
        records.append(waiting);
        *waiting = records;
    }
}

fn subagent_records(writes: &[OutOfTurn]) -> Vec<SubagentRecord> {
    writes
        .iter()
        .filter_map(|write| match write {
            OutOfTurn::Subagent(record) => Some(record.clone()),
            OutOfTurn::Hooks(_) => None,
        })
        .collect()
}

/// A write that originated outside the session's writer.
pub(crate) enum OutOfTurn {
    Hooks(ExecutedHookDispatch),
    Subagent(SubagentRecord),
}

impl SessionLeases {
    /// Take the write lease for `session_id`, waiting for the current holder
    /// to release it. A holder that acquires its own session again
    /// deadlocks.
    pub(crate) async fn acquire(&self, session_id: &SessionId) {
        loop {
            // Created before the check so a release in between still wakes us.
            let released = self.released.notified();
            let mut writes = self.writes.lock().await;
            let undelivered = writes.undelivered.remove(session_id).unwrap_or_default();
            match writes.held.entry(session_id.clone()) {
                Entry::Vacant(slot) => {
                    slot.insert(undelivered.into_iter().map(OutOfTurn::Subagent).collect());
                    return;
                }
                Entry::Occupied(_) => writes.hold_undelivered(session_id, undelivered),
            }
            drop(writes);
            released.await;
        }
    }

    /// Release the lease and commit whatever queued behind it.
    pub(crate) async fn release<F, Fut>(
        &self,
        session_id: &SessionId,
        commit: F,
    ) -> anyhow::Result<()>
    where
        F: FnOnce(Vec<OutOfTurn>) -> Fut,
        Fut: Future<Output = anyhow::Result<()>>,
    {
        let mut writes = self.writes.lock().await;
        let outcome = match writes.held.remove(session_id) {
            Some(queued) if !queued.is_empty() => {
                let records = subagent_records(&queued);
                let outcome = commit(queued).await;
                if outcome.is_err() {
                    writes.hold_undelivered(session_id, records);
                }
                outcome
            }
            _ => Ok(()),
        };
        drop(writes);
        self.released.notify_waiters();
        outcome
    }

    /// Take the subagent records queued behind the lease, for the holder
    /// to fold into its next commit.
    pub(crate) async fn take_subagent_records(
        &self,
        session_id: &SessionId,
    ) -> Vec<SubagentRecord> {
        let mut writes = self.writes.lock().await;
        let Some(queued) = writes.held.get_mut(session_id) else {
            return Vec::new();
        };
        queued
            .extract_if(.., |write| matches!(write, OutOfTurn::Subagent(_)))
            .filter_map(|write| match write {
                OutOfTurn::Subagent(record) => Some(record),
                OutOfTurn::Hooks(_) => None,
            })
            .collect()
    }

    /// Return records taken by [`Self::take_subagent_records`] whose commit
    /// failed, ahead of anything queued since, so they are not lost.
    pub(crate) async fn requeue_subagent_records(
        &self,
        session_id: &SessionId,
        records: Vec<SubagentRecord>,
    ) {
        let mut writes = self.writes.lock().await;
        match writes.held.get_mut(session_id) {
            Some(queued) => {
                queued.splice(0..0, records.into_iter().map(OutOfTurn::Subagent));
            }
            None => writes.hold_undelivered(session_id, records),
        }
    }

    /// The `once` hook ids fired by hook dispatches queued behind the lease,
    /// which the checkpoint does not hold yet.
    pub(crate) async fn queued_hook_ids(&self, session_id: &SessionId) -> BTreeSet<String> {
        let writes = self.writes.lock().await;
        writes
            .held
            .get(session_id)
            .into_iter()
            .flatten()
            .filter_map(|write| match write {
                OutOfTurn::Hooks(dispatch) => Some(&dispatch.fired_hook_ids),
                OutOfTurn::Subagent(_) => None,
            })
            .flatten()
            .cloned()
            .collect()
    }

    /// Queue `dispatch` behind the lease holder, or commit it now (after any
    /// undelivered subagent records) when the session has no writer.
    pub(crate) async fn dispatch<F, Fut>(
        &self,
        session_id: &SessionId,
        dispatch: OutOfTurn,
        commit: F,
    ) -> anyhow::Result<()>
    where
        F: FnOnce(Vec<OutOfTurn>) -> Fut,
        Fut: Future<Output = anyhow::Result<()>>,
    {
        let mut writes = self.writes.lock().await;
        if let Some(queued) = writes.held.get_mut(session_id) {
            queued.push(dispatch);
            return Ok(());
        }
        let mut batch = writes
            .undelivered
            .remove(session_id)
            .unwrap_or_default()
            .into_iter()
            .map(OutOfTurn::Subagent)
            .collect::<Vec<_>>();
        batch.push(dispatch);
        let records = subagent_records(&batch);
        let outcome = commit(batch).await;
        if outcome.is_err() {
            writes.hold_undelivered(session_id, records);
        }
        outcome
    }
}

/// A held write lease. `release` commits the queue behind it. Dropping the
/// lease without releasing (panic, abort, early return) releases it in the
/// background so the session never stays locked.
pub(crate) struct SessionLease {
    session: Option<SessionExecutor>,
}

impl SessionLease {
    pub(crate) fn new(session: SessionExecutor) -> Self {
        Self {
            session: Some(session),
        }
    }

    pub(crate) async fn release(mut self) -> anyhow::Result<()> {
        match self.session.take() {
            Some(session) => session.release_lease().await,
            None => Ok(()),
        }
    }
}

impl Drop for SessionLease {
    fn drop(&mut self) {
        if let Some(session) = self.session.take()
            && let Ok(runtime) = tokio::runtime::Handle::try_current()
        {
            runtime.spawn(async move {
                if let Err(error) = session.release_lease().await {
                    warn!(session_id = %session.session_id(), error = %error, "failed to release session lease");
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use super::*;

    type CommitResult = fn() -> anyhow::Result<()>;

    fn hooks() -> OutOfTurn {
        OutOfTurn::Hooks(ExecutedHookDispatch::default())
    }

    fn committer(
        count: Arc<AtomicUsize>,
        result: CommitResult,
    ) -> impl FnOnce(Vec<OutOfTurn>) -> std::future::Ready<anyhow::Result<()>> {
        move |dispatches| {
            count.fetch_add(dispatches.len(), Ordering::SeqCst);
            std::future::ready(result())
        }
    }

    #[tokio::test]
    async fn acquire_is_exclusive_per_session() {
        let leases = Arc::new(SessionLeases::default());
        let (a, b) = (SessionId::new(), SessionId::new());
        let acquired = |session: &SessionId| {
            let (leases, session) = (leases.clone(), session.clone());
            async move {
                tokio::time::timeout(Duration::from_millis(50), leases.acquire(&session))
                    .await
                    .is_ok()
            }
        };
        // (case, session, acquires without waiting)
        let cases = [
            ("first", &a, true),
            ("same again", &a, false),
            ("other", &b, true),
        ];
        for (case, session, ok) in cases {
            assert_eq!(acquired(session).await, ok, "{case}");
        }

        let waiter = tokio::spawn({
            let (leases, a) = (leases.clone(), a.clone());
            async move { leases.acquire(&a).await }
        });
        tokio::task::yield_now().await;
        assert!(!waiter.is_finished(), "held lease blocks the next writer");
        leases
            .release(&a, |_| async { Ok(()) })
            .await
            .expect("release");
        tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .expect("release wakes the waiter")
            .expect("waiter");
    }

    #[tokio::test]
    async fn dispatch_queues_while_held_and_commits_on_release() {
        let leases = SessionLeases::default();
        let session = SessionId::new();
        let committed = Arc::new(AtomicUsize::new(0));

        leases.acquire(&session).await;
        for _ in 0..2 {
            leases
                .dispatch(&session, hooks(), committer(committed.clone(), || Ok(())))
                .await
                .expect("queue");
        }
        assert_eq!(
            committed.load(Ordering::SeqCst),
            0,
            "held lease defers commits"
        );

        leases
            .release(&session, committer(committed.clone(), || Ok(())))
            .await
            .expect("release");
        assert_eq!(
            committed.load(Ordering::SeqCst),
            2,
            "release drains the queue"
        );
    }

    #[tokio::test]
    async fn dispatch_commits_immediately_without_lease() {
        let leases = SessionLeases::default();
        let session = SessionId::new();
        let committed = Arc::new(AtomicUsize::new(0));
        // (case, commit result, dispatch should succeed)
        let cases: [(&str, CommitResult, bool); 2] = [
            ("commit ok", || Ok(()), true),
            ("commit fails", || Err(anyhow::anyhow!("boom")), false),
        ];
        for (case, result, ok) in cases {
            let outcome = leases
                .dispatch(&session, hooks(), committer(committed.clone(), result))
                .await;
            assert_eq!(outcome.is_ok(), ok, "{case}");
        }
        assert_eq!(committed.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn release_surfaces_commit_errors_and_skips_empty_queues() {
        let leases = SessionLeases::default();
        let session = SessionId::new();
        let committed = Arc::new(AtomicUsize::new(0));

        leases.acquire(&session).await;
        leases
            .release(
                &session,
                committer(committed.clone(), || Err(anyhow::anyhow!("x"))),
            )
            .await
            .expect("empty queue never commits");
        assert_eq!(committed.load(Ordering::SeqCst), 0);

        leases.acquire(&session).await;
        leases
            .dispatch(&session, hooks(), |_| async { Ok(()) })
            .await
            .expect("queue");
        assert!(
            leases
                .release(
                    &session,
                    committer(committed.clone(), || Err(anyhow::anyhow!("x")))
                )
                .await
                .is_err()
        );
        tokio::time::timeout(Duration::from_millis(50), leases.acquire(&session))
            .await
            .expect("failed release still frees the lease");
    }
    #[tokio::test]
    async fn queued_hook_ids_unions_the_queued_dispatches() {
        let fired = |ids: &[&str]| {
            OutOfTurn::Hooks(ExecutedHookDispatch {
                fired_hook_ids: ids.iter().map(|id| (*id).to_owned()).collect(),
                ..ExecutedHookDispatch::default()
            })
        };
        // (case, lease held, queued writes, expected ids)
        type Case = (&'static str, bool, Vec<OutOfTurn>, Vec<&'static str>);
        let cases: [Case; 3] = [
            ("no writer", false, vec![fired(&["a"])], vec![]),
            ("nothing fired", true, vec![hooks()], vec![]),
            (
                "several dispatches",
                true,
                vec![fired(&["b", "a"]), hooks(), fired(&["a", "c"])],
                vec!["a", "b", "c"],
            ),
        ];
        for (case, held, queued, expected) in cases {
            let leases = SessionLeases::default();
            let session = SessionId::new();
            if held {
                leases.acquire(&session).await;
            }
            for write in queued {
                leases
                    .dispatch(&session, write, |_| async { Ok(()) })
                    .await
                    .expect("queue");
            }
            let ids = leases.queued_hook_ids(&session).await;
            assert_eq!(
                ids.iter().map(String::as_str).collect::<Vec<_>>(),
                expected,
                "{case}"
            );
        }
    }

    #[tokio::test]
    async fn the_holder_takes_subagent_records_and_leaves_hooks() {
        let record = || {
            OutOfTurn::Subagent(SubagentRecord {
                status: halter_protocol::SubagentStatus {
                    agent_id: halter_protocol::AgentId::from("agent"),
                    session_id: SessionId::from("child"),
                    agent_type: None,
                    task: "task".to_owned(),
                    state: halter_protocol::SubagentState::Running,
                    last_message: None,
                    usage: None,
                    error: None,
                },
                generation: 1,
            })
        };
        // (case, lease held, queued writes, records taken, hooks left for release)
        type Case = (&'static str, bool, Vec<fn() -> OutOfTurn>, usize, usize);
        let cases: [Case; 3] = [
            ("no writer", false, Vec::new(), 0, 0),
            ("hooks only", true, vec![hooks], 0, 1),
            ("mixed", true, vec![record, hooks, record], 2, 1),
        ];
        for (case, held, queued, taken, left) in cases {
            let leases = SessionLeases::default();
            let session = SessionId::new();
            if held {
                leases.acquire(&session).await;
            }
            for write in queued {
                leases
                    .dispatch(&session, write(), |_| async { Ok(()) })
                    .await
                    .expect("queue");
            }
            let records = leases.take_subagent_records(&session).await;
            assert_eq!(records.len(), taken, "{case}");
            let committed = Arc::new(AtomicUsize::new(0));
            leases
                .release(&session, committer(committed.clone(), || Ok(())))
                .await
                .expect("release");
            assert_eq!(committed.load(Ordering::SeqCst), left, "{case}");
        }
    }

    fn record(generation: u64, state: halter_protocol::SubagentState) -> SubagentRecord {
        SubagentRecord {
            status: halter_protocol::SubagentStatus {
                agent_id: halter_protocol::AgentId::from("child"),
                session_id: SessionId::from("child-session"),
                agent_type: None,
                task: "task".to_owned(),
                state,
                last_message: None,
                usage: None,
                error: None,
            },
            generation,
        }
    }

    /// Commits that fail with `fail`, else record the subagent records they carry.
    fn capture(
        delivered: Arc<std::sync::Mutex<Vec<SubagentRecord>>>,
        fail: bool,
    ) -> impl FnOnce(Vec<OutOfTurn>) -> std::future::Ready<anyhow::Result<()>> {
        move |writes| {
            if fail {
                return std::future::ready(Err(anyhow::anyhow!("commit unavailable")));
            }
            delivered.lock().unwrap().extend(subagent_records(&writes));
            std::future::ready(Ok(()))
        }
    }

    /// A subagent record whose commit fails is redelivered ahead of newer
    /// writes, on every path that can lose it: release, an immediate
    /// dispatch, and a holder's own failed commit.
    #[tokio::test]
    async fn failed_subagent_records_are_redelivered_in_order() {
        use halter_protocol::SubagentState::{Completed, Running};

        // Release fails: the next lease is seeded with the record.
        let leases = SessionLeases::default();
        let session = SessionId::new();
        let delivered = Arc::new(std::sync::Mutex::new(Vec::new()));
        leases.acquire(&session).await;
        let running = OutOfTurn::Subagent(record(1, Running));
        leases
            .dispatch(&session, running, capture(delivered.clone(), false))
            .await
            .unwrap();
        assert!(
            leases
                .release(&session, capture(delivered.clone(), true))
                .await
                .is_err()
        );
        leases.acquire(&session).await;
        assert_eq!(
            leases.take_subagent_records(&session).await,
            [record(1, Running)],
            "failed release seeds the next lease"
        );

        // An immediate dispatch fails: the next one delivers both, in order.
        let leases = SessionLeases::default();
        let failed = leases
            .dispatch(
                &session,
                OutOfTurn::Subagent(record(1, Running)),
                capture(delivered.clone(), true),
            )
            .await;
        assert!(failed.is_err());
        leases
            .dispatch(
                &session,
                OutOfTurn::Subagent(record(1, Completed)),
                capture(delivered.clone(), false),
            )
            .await
            .unwrap();
        assert_eq!(
            *delivered.lock().unwrap(),
            [record(1, Running), record(1, Completed)],
            "older record first, so the same-generation update wins"
        );

        // A holder's commit fails: requeued records precede newer queued ones.
        let leases = SessionLeases::default();
        leases.acquire(&session).await;
        let taken = vec![record(1, Running)];
        leases
            .dispatch(
                &session,
                OutOfTurn::Subagent(record(1, Completed)),
                capture(delivered.clone(), false),
            )
            .await
            .unwrap();
        leases.requeue_subagent_records(&session, taken).await;
        assert_eq!(
            leases.take_subagent_records(&session).await,
            [record(1, Running), record(1, Completed)]
        );
    }
}
