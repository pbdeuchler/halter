// pattern: Imperative Shell
//
// Owns actual execution tasks independently of their event-stream consumers.
// Supervisors retain join handles and share completion with session-level
// interruption and runtime shutdown, so overlapping waiters cannot detach work.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::Duration;

use halter_protocol::TurnId;
use thiserror::Error;
use tokio::sync::watch;
use tokio::task::{AbortHandle, JoinHandle};
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

#[derive(Debug, Error)]
/// Error returned when a turn cannot be registered.
pub enum TurnRegistryError {
    #[error("runtime is shutting down: refusing to register turn '{0}'")]
    ShuttingDown(TurnId),
    #[error("turn '{0}' is already registered")]
    DuplicateTurn(TurnId),
}

#[derive(Debug)]
/// Summary returned by runtime shutdown.
pub struct ShutdownReport {
    pub turns_drained: usize,
    pub turns_aborted: usize,
    pub timed_out: bool,
}

/// Runtime-wide registry of in-flight execution tasks. Entries hold their
/// cancellation and abort controls plus a shared completion notification;
/// supervisors own the actual join handles until execution has settled.
#[derive(Default)]
pub struct TurnRegistry {
    inner: Arc<Mutex<TurnRegistryInner>>,
    /// Parent of every runtime-issued token, cancelled by `shutdown`, so work
    /// that is not (yet) a registered turn still observes runtime shutdown.
    root: CancellationToken,
}

#[derive(Default)]
struct TurnRegistryInner {
    in_flight: HashMap<TurnId, RegisteredTurn>,
    shutting_down: bool,
}

#[derive(Clone)]
struct RegisteredTurn {
    cancel: CancellationToken,
    abort: AbortHandle,
    complete: watch::Receiver<bool>,
}

impl RegisteredTurn {
    async fn wait(&self) {
        let mut complete = self.complete.clone();
        while !*complete.borrow_and_update() {
            if complete.changed().await.is_err() {
                break;
            }
        }
    }
}

impl TurnRegistry {
    /// Create an empty turn registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Bind a freshly spawned turn task to the registry. If the runtime
    /// is already shutting down the caller's task is cancelled and
    /// aborted before this call returns and `ShuttingDown` is surfaced.
    pub fn register(
        &self,
        turn_id: TurnId,
        cancel: CancellationToken,
        handle: JoinHandle<()>,
    ) -> Result<(), TurnRegistryError> {
        let mut inner = self.lock();
        if inner.shutting_down {
            cancel.cancel();
            handle.abort();
            return Err(TurnRegistryError::ShuttingDown(turn_id));
        }
        if inner.in_flight.contains_key(&turn_id) {
            // Caller still owns the handle on the failure path; abort it
            // so we don't leak a zombie task.
            handle.abort();
            return Err(TurnRegistryError::DuplicateTurn(turn_id));
        }
        let abort = handle.abort_handle();
        let (complete_tx, complete) = watch::channel(false);
        inner.in_flight.insert(
            turn_id.clone(),
            RegisteredTurn {
                cancel,
                abort,
                complete,
            },
        );
        // A supervisor retains the actual task's join handle even when a
        // shutdown waiter times out. Every cancellation path can observe the
        // same completion, and aborting a stream consumer cannot detach work.
        let registry = self.inner.clone();
        tokio::spawn(async move {
            let _ = handle.await;
            complete_tx.send_replace(true);
            let mut inner = registry.lock().unwrap_or_else(|error| error.into_inner());
            inner.in_flight.remove(&turn_id);
        });
        Ok(())
    }

    /// A token cancelled when runtime shutdown starts.
    #[must_use]
    pub fn child_token(&self) -> CancellationToken {
        self.root.child_token()
    }

    pub(crate) fn begin_shutdown(&self) {
        let mut inner = self.lock();
        inner.shutting_down = true;
        self.root.cancel();
        for turn in inner.in_flight.values() {
            turn.cancel.cancel();
        }
    }

    /// Remove a turn from the registry. Idempotent: deregistering an
    /// unknown id is a no-op (covers the race where shutdown drains
    /// the entry just before the task body's deregister runs).
    pub fn deregister(&self, turn_id: &TurnId) {
        let mut inner = self.lock();
        inner.in_flight.remove(turn_id);
    }

    /// Abort the executor itself and wait until its future has been dropped.
    /// Registration remains discoverable while global shutdown is draining.
    pub(crate) async fn abort_and_wait(&self, turn_id: &TurnId) {
        let registered = self.lock().in_flight.get(turn_id).cloned();
        if let Some(registered) = registered {
            registered.cancel.cancel();
            registered.abort.abort();
            registered.wait().await;
        }
    }

    pub(crate) fn abort(&self, turn_id: &TurnId) {
        if let Some(turn) = self.lock().in_flight.get(turn_id) {
            turn.cancel.cancel();
            turn.abort.abort();
        }
    }

    /// Whether shutdown has started and new turns are rejected.
    #[must_use]
    pub fn is_shutting_down(&self) -> bool {
        self.lock().shutting_down
    }

    /// Number of currently registered turns.
    #[must_use]
    pub fn in_flight_count(&self) -> usize {
        self.lock().in_flight.len()
    }

    /// Close admission, cancel in-flight execution, and await settlement.
    /// `None` waits without a deadline. A finite timeout aborts remaining
    /// executors; supervisors keep ownership until their futures are dropped.
    pub async fn shutdown(&self, timeout: impl Into<Option<Duration>>) -> ShutdownReport {
        // A duration beyond the representable clock range is effectively
        // unlimited; no reachable runtime instant could exhaust it.
        let deadline = timeout
            .into()
            .and_then(|timeout| Instant::now().checked_add(timeout));
        self.shutdown_at(deadline).await
    }

    pub(crate) async fn shutdown_at(&self, deadline: Option<Instant>) -> ShutdownReport {
        self.begin_shutdown();
        let turns = {
            let inner = self.lock();
            let mut taken = Vec::with_capacity(inner.in_flight.len());
            for registered in inner.in_flight.values() {
                registered.cancel.cancel();
                taken.push(registered.clone());
            }
            taken
        };

        if turns.is_empty() {
            debug!("turn registry shutdown: no in-flight turns");
            return ShutdownReport {
                turns_drained: 0,
                turns_aborted: 0,
                timed_out: false,
            };
        }

        debug!(in_flight = turns.len(), "turn registry shutdown: draining");
        let settled = futures::future::join_all(turns.iter().map(RegisteredTurn::wait));
        tokio::pin!(settled);
        let elapsed = async {
            match deadline {
                Some(deadline) => tokio::time::sleep_until(deadline).await,
                None => std::future::pending::<()>().await,
            }
        };
        tokio::select! {
            _ = &mut settled => ShutdownReport {
                turns_drained: turns.len(),
                turns_aborted: 0,
                timed_out: false,
            },
            _ = elapsed => {
                let drained_count = turns.iter().filter(|turn| *turn.complete.borrow()).count();
                let aborted = turns.len() - drained_count;
                for turn in &turns {
                    turn.abort.abort();
                }
                warn!(
                    drained = drained_count,
                    pending = aborted,
                    "turn registry shutdown: drain timeout, aborting remaining tasks"
                );
                ShutdownReport {
                    turns_drained: drained_count,
                    turns_aborted: aborted,
                    timed_out: true,
                }
            }
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, TurnRegistryInner> {
        // Mutex contention on this lock is brief (insert/remove on a
        // small HashMap). Poisoning means a held panic which is bad
        // enough that recovering the inner state is the right choice;
        // we surface it via `into_inner` so subsequent operations can
        // continue rather than panic the runtime.
        self.inner.lock().unwrap_or_else(|poisoned| {
            warn!("turn registry mutex poisoned; recovering inner state");
            poisoned.into_inner()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
    use tokio::sync::oneshot;

    #[tokio::test]
    async fn force_abort_remains_available_after_global_shutdown_starts() {
        let registry = Arc::new(TurnRegistry::new());
        let turn_id = TurnId::new();
        let dropped = Arc::new(AtomicBool::new(false));
        let (started, ready) = oneshot::channel();
        let handle = tokio::spawn({
            let dropped = dropped.clone();
            async move {
                struct Dropped(Arc<AtomicBool>);
                impl Drop for Dropped {
                    fn drop(&mut self) {
                        self.0.store(true, AtomicOrdering::SeqCst);
                    }
                }
                let _dropped = Dropped(dropped);
                started.send(()).unwrap();
                std::future::pending::<()>().await;
            }
        });
        let cancel = CancellationToken::new();
        registry
            .register(turn_id.clone(), cancel.clone(), handle)
            .unwrap();
        ready.await.unwrap();
        let shutdown = tokio::spawn({
            let registry = registry.clone();
            async move { registry.shutdown(None).await }
        });
        cancel.cancelled().await;
        assert!(
            !shutdown.is_finished(),
            "unlimited drain waits for actual execution"
        );
        registry.abort_and_wait(&turn_id).await;
        assert!(
            dropped.load(AtomicOrdering::SeqCst),
            "force abort joins the actual execution future"
        );
        let report = shutdown.await.unwrap();
        assert_eq!(report.turns_drained, 1);
        assert!(!report.timed_out);
    }

    #[tokio::test]
    async fn child_tokens_fire_only_on_shutdown() {
        let registry = TurnRegistry::new();
        let before = registry.child_token();
        assert!(!before.is_cancelled(), "live runtime must not cancel");

        let _ = registry.shutdown(Duration::from_millis(0)).await;

        assert!(before.is_cancelled(), "shutdown must cancel issued tokens");
        assert!(
            registry.child_token().is_cancelled(),
            "tokens issued after shutdown start cancelled"
        );
    }

    #[tokio::test]
    async fn register_and_deregister_round_trip() {
        let registry = TurnRegistry::new();
        let cancel = CancellationToken::new();
        let handle = tokio::spawn(async {});
        let turn_id = TurnId::from("turn-1");
        registry
            .register(turn_id.clone(), cancel, handle)
            .expect("register");
        assert_eq!(registry.in_flight_count(), 1);
        registry.deregister(&turn_id);
        assert_eq!(registry.in_flight_count(), 0);
    }

    #[tokio::test]
    async fn duplicate_turn_id_rejected() {
        let registry = TurnRegistry::new();
        let turn_id = TurnId::from("turn-dup");
        let first_handle = tokio::spawn(async { std::future::pending::<()>().await });
        registry
            .register(turn_id.clone(), CancellationToken::new(), first_handle)
            .expect("first register");

        let second_handle = tokio::spawn(async {});
        let err = registry
            .register(turn_id.clone(), CancellationToken::new(), second_handle)
            .expect_err("second register must fail");
        match err {
            TurnRegistryError::DuplicateTurn(id) => assert_eq!(id, turn_id),
            other => panic!("expected DuplicateTurn, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn shutdown_cancels_in_flight_and_returns_report() {
        let registry = TurnRegistry::new();
        let cancel = CancellationToken::new();
        let token_for_task = cancel.clone();
        let (started_tx, started_rx) = oneshot::channel();
        let handle = tokio::spawn(async move {
            let _ = started_tx.send(());
            token_for_task.cancelled().await;
        });
        registry
            .register(TurnId::from("turn-cancel"), cancel, handle)
            .expect("register");
        started_rx.await.expect("task started");

        let report = registry.shutdown(Duration::from_secs(1)).await;
        assert_eq!(report.turns_drained, 1);
        assert_eq!(report.turns_aborted, 0);
        assert!(!report.timed_out);
        assert!(registry.is_shutting_down());
    }

    #[tokio::test]
    async fn shutdown_aborts_uncancellable_tasks_after_deadline() {
        let registry = TurnRegistry::new();
        let cancel = CancellationToken::new();
        let dropped = Arc::new(AtomicBool::new(false));
        let dropped_in_task = dropped.clone();
        // Task ignores cancellation and waits forever.
        let handle = tokio::spawn(async move {
            struct DropFlag(Arc<AtomicBool>);
            impl Drop for DropFlag {
                fn drop(&mut self) {
                    self.0.store(true, AtomicOrdering::SeqCst);
                }
            }
            let _flag = DropFlag(dropped_in_task);
            std::future::pending::<()>().await
        });
        registry
            .register(TurnId::from("turn-stuck"), cancel, handle)
            .expect("register");

        let report = registry.shutdown(Duration::from_millis(50)).await;
        assert_eq!(report.turns_drained, 0);
        assert_eq!(report.turns_aborted, 1);
        assert!(report.timed_out);
        assert!(registry.is_shutting_down());
        tokio::time::timeout(Duration::from_secs(1), async {
            while !dropped.load(AtomicOrdering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("aborted task should be dropped");
    }

    #[tokio::test]
    async fn register_after_shutdown_rejected_and_aborts_caller_handle() {
        let registry = TurnRegistry::new();
        let _ = registry.shutdown(Duration::from_millis(0)).await;

        let cancel = CancellationToken::new();
        let token_for_task = cancel.clone();
        let handle = tokio::spawn(async move {
            token_for_task.cancelled().await;
        });
        let turn_id = TurnId::from("turn-late");
        let err = registry
            .register(turn_id.clone(), cancel.clone(), handle)
            .expect_err("must reject post-shutdown registration");
        match err {
            TurnRegistryError::ShuttingDown(id) => assert_eq!(id, turn_id),
            other => panic!("expected ShuttingDown, got {other:?}"),
        }
        assert!(cancel.is_cancelled());
    }
}
