// pattern: Imperative Shell
//
// Tracks in-flight turn `JoinHandle`s and per-turn cancellation tokens
// so the runtime can drain (or abort) outstanding work on shutdown.
//
// Pre-Phase-4 the spawned turn loop in `SessionHandle::submit_turn_with_cancel`
// returned only the live event stream — the `JoinHandle` was dropped on the
// floor and there was no way to wait for in-flight turns to settle when
// the host process wanted to exit cleanly. AC2.3 / AC2.4 require that
// runtime shutdown drain those tasks, with a per-shutdown deadline.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use halter_protocol::TurnId;
use thiserror::Error;
use tokio::task::JoinHandle;
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

/// Runtime-wide registry of in-flight turn tasks. Each entry pairs a
/// `JoinHandle` with the `CancellationToken` that controls the turn so
/// shutdown can both signal cooperative cancellation and reclaim the
/// handle for `JoinHandle::abort` if the drain deadline expires.
#[derive(Default)]
pub struct TurnRegistry {
    inner: Mutex<TurnRegistryInner>,
}

#[derive(Default)]
struct TurnRegistryInner {
    in_flight: HashMap<TurnId, RegisteredTurn>,
    shutting_down: bool,
}

struct RegisteredTurn {
    cancel: CancellationToken,
    handle: JoinHandle<()>,
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
        inner
            .in_flight
            .insert(turn_id, RegisteredTurn { cancel, handle });
        Ok(())
    }

    /// Remove a turn from the registry. Idempotent: deregistering an
    /// unknown id is a no-op (covers the race where shutdown drains
    /// the entry just before the task body's deregister runs).
    pub fn deregister(&self, turn_id: &TurnId) {
        let mut inner = self.lock();
        inner.in_flight.remove(turn_id);
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

    /// Mark the registry as shutting down, cancel every in-flight turn
    /// token, and wait for the spawned tasks to settle. After `drain`
    /// elapses any still-running tasks are aborted.
    pub async fn shutdown(&self, drain: Duration) -> ShutdownReport {
        let handles = {
            let mut inner = self.lock();
            inner.shutting_down = true;
            let mut taken = Vec::with_capacity(inner.in_flight.len());
            for (_, registered) in inner.in_flight.drain() {
                registered.cancel.cancel();
                taken.push(registered.handle);
            }
            taken
        };

        if handles.is_empty() {
            debug!("turn registry shutdown: no in-flight turns");
            return ShutdownReport {
                turns_drained: 0,
                turns_aborted: 0,
                timed_out: false,
            };
        }

        debug!(in_flight = handles.len(), drain_ms = %drain.as_millis(), "turn registry shutdown: draining");

        let abort_handles: Vec<_> = handles
            .iter()
            .map(tokio::task::JoinHandle::abort_handle)
            .collect();
        let total = handles.len();
        let drained = Arc::new(AtomicUsize::new(0));
        let drained_in_loop = drained.clone();
        let join_all = async move {
            for handle in handles {
                let _ = handle.await;
                drained_in_loop.fetch_add(1, Ordering::Relaxed);
            }
        };

        match tokio::time::timeout(drain, join_all).await {
            Ok(()) => ShutdownReport {
                turns_drained: drained.load(Ordering::Relaxed),
                turns_aborted: 0,
                timed_out: false,
            },
            Err(_) => {
                let drained_count = drained.load(Ordering::Relaxed);
                let aborted = total - drained_count;
                for abort_handle in abort_handles {
                    abort_handle.abort();
                }
                warn!(
                    drained = drained_count,
                    pending = aborted,
                    drain_ms = %drain.as_millis(),
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
