// pattern: Imperative Shell
//
// One writer per session. A turn (or a session-level operation such as
// `compact`) holds the session's write lease for its whole duration and
// commits against its own in-memory state; the next writer waits for it. Hook dispatches that originate
// outside that writer — subagent lifecycle hooks, `notify` — queue behind
// the lease instead of racing it on `expected_head_sequence`, and are
// committed when the lease is released. With no lease held they commit
// immediately, under the same lock, so a writer cannot start mid-commit.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::future::Future;

use halter_protocol::SessionId;
use tokio::sync::{Mutex, Notify};
use tracing::warn;

use crate::ExecutedHookDispatch;
use crate::session::SessionHandle;

#[derive(Default)]
/// Per-session write leases plus the hook dispatches queued behind them.
pub struct SessionLeases {
    held: Mutex<HashMap<SessionId, Vec<ExecutedHookDispatch>>>,
    released: Notify,
}

impl SessionLeases {
    /// Take the write lease for `session_id`, waiting for the current holder
    /// to release it. A holder that acquires its own session again
    /// deadlocks.
    pub(crate) async fn acquire(&self, session_id: &SessionId) {
        loop {
            // Created before the check so a release in between still wakes us.
            let released = self.released.notified();
            if let Entry::Vacant(slot) = self.held.lock().await.entry(session_id.clone()) {
                slot.insert(Vec::new());
                return;
            }
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
        F: FnOnce(Vec<ExecutedHookDispatch>) -> Fut,
        Fut: Future<Output = anyhow::Result<()>>,
    {
        let mut held = self.held.lock().await;
        let outcome = match held.remove(session_id) {
            Some(queued) if !queued.is_empty() => commit(queued).await,
            _ => Ok(()),
        };
        drop(held);
        self.released.notify_waiters();
        outcome
    }

    /// Queue `dispatch` behind the lease holder, or commit it now when the
    /// session has no writer.
    pub(crate) async fn dispatch<F, Fut>(
        &self,
        session_id: &SessionId,
        dispatch: ExecutedHookDispatch,
        commit: F,
    ) -> anyhow::Result<()>
    where
        F: FnOnce(Vec<ExecutedHookDispatch>) -> Fut,
        Fut: Future<Output = anyhow::Result<()>>,
    {
        let mut held = self.held.lock().await;
        match held.get_mut(session_id) {
            Some(queued) => {
                queued.push(dispatch);
                Ok(())
            }
            None => commit(vec![dispatch]).await,
        }
    }
}

/// A held write lease. `release` commits the queue behind it. Dropping the
/// lease without releasing (panic, abort, early return) releases it in the
/// background so the session never stays locked.
pub(crate) struct SessionLease {
    session: Option<SessionHandle>,
}

impl SessionLease {
    pub(crate) fn new(session: SessionHandle) -> Self {
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

    fn committer(
        count: Arc<AtomicUsize>,
        result: CommitResult,
    ) -> impl FnOnce(Vec<ExecutedHookDispatch>) -> std::future::Ready<anyhow::Result<()>> {
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
                .dispatch(
                    &session,
                    ExecutedHookDispatch::default(),
                    committer(committed.clone(), || Ok(())),
                )
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
                .dispatch(
                    &session,
                    ExecutedHookDispatch::default(),
                    committer(committed.clone(), result),
                )
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
            .dispatch(&session, ExecutedHookDispatch::default(), |_| async {
                Ok(())
            })
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
}
