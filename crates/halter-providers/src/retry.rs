// pattern: Functional Core

use std::time::{Duration, Instant, SystemTime};

/// Bounded retry policy shared by the streaming provider pipeline. Replaces
/// the previous unbounded `loop { ... continue; }` retry in
/// `responses_provider`, which had no per-stream cap and no cumulative
/// deadline (AC3.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RetryPolicy {
    /// Total attempts allowed (initial attempt + retries).
    pub(crate) max_attempts: u32,
    /// Base delay used as the seed for exponential growth.
    pub(crate) base_backoff: Duration,
    /// Upper bound for any single backoff, even if exponential growth or a
    /// server-supplied hint would exceed it.
    pub(crate) max_backoff: Duration,
    /// Cumulative wall time budget across all attempts. Once exceeded,
    /// further retries are denied even if `max_attempts` has not been hit.
    pub(crate) deadline: Duration,
    /// Random offset, expressed as a percentage of the computed backoff,
    /// added on top to break thundering-herd synchronization across
    /// concurrent retrying clients. `0` disables jitter (used in tests for
    /// deterministic timing).
    pub(crate) jitter_pct: u32,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 5,
            base_backoff: Duration::from_millis(500),
            max_backoff: Duration::from_secs(30),
            deadline: Duration::from_secs(60),
            jitter_pct: 25,
        }
    }
}

/// Stateful retry tracker. Combines the per-attempt counter with a cumulative
/// deadline so callers don't have to thread `Instant` and `attempt: u32`
/// separately. The contract is intentionally narrow: callers ask for the next
/// backoff after a failed attempt; `None` means the budget is exhausted and
/// the failure should be propagated.
#[derive(Debug)]
pub(crate) struct RetryGate {
    policy: RetryPolicy,
    started: Instant,
    attempt: u32,
}

impl RetryGate {
    pub(crate) fn new(policy: RetryPolicy) -> Self {
        Self {
            policy,
            started: Instant::now(),
            attempt: 0,
        }
    }

    /// 1-indexed attempt counter for the *next* attempt that will be tried.
    /// Useful for tagging a stream attempt with its sequence number when
    /// deduplicating cross-attempt events (AC3.6).
    pub(crate) fn next_attempt_id(&self) -> u32 {
        self.attempt + 1
    }

    /// Records a failed attempt and returns the backoff to wait before the
    /// next attempt, or `None` if the budget (max_attempts or deadline) is
    /// exhausted. The optional `hint` is a server-supplied retry-after
    /// duration (e.g. parsed from `Please try again in 1.25s`) which takes
    /// precedence over the computed exponential when present.
    pub(crate) fn record_failure_and_next_backoff(
        &mut self,
        hint: Option<Duration>,
    ) -> Option<Duration> {
        self.attempt = self.attempt.saturating_add(1);
        if self.attempt >= self.policy.max_attempts
            || self.started.elapsed() >= self.policy.deadline
        {
            return None;
        }
        Some(compute_backoff(&self.policy, self.attempt, hint))
    }
}

/// Pure backoff computation. Honors a server-supplied hint when present
/// (capped at `policy.max_backoff` so a hostile server can't pin us into a
/// 24-hour wait), otherwise produces jittered exponential.
#[must_use]
pub(crate) fn compute_backoff(
    policy: &RetryPolicy,
    attempt: u32,
    hint: Option<Duration>,
) -> Duration {
    if let Some(hint) = hint {
        return hint.min(policy.max_backoff);
    }
    let exponent = attempt.saturating_sub(1).min(20);
    let exp_ms = (policy.base_backoff.as_millis() as u64).saturating_mul(1u64 << exponent);
    let capped_ms = exp_ms.min(policy.max_backoff.as_millis() as u64);
    let jitter = jitter_offset(capped_ms, policy.jitter_pct);
    Duration::from_millis(capped_ms.saturating_add(jitter))
}

/// Pseudo-random jitter offset in `[0, capped * jitter_pct / 100]`. Seeded
/// from the current wall clock so concurrent retries don't synchronize. We
/// intentionally avoid pulling in a `rand` dependency for this single
/// callsite — the entropy needed is "enough to avoid thundering herd," not
/// cryptographic.
fn jitter_offset(capped_ms: u64, jitter_pct: u32) -> u64 {
    if jitter_pct == 0 || capped_ms == 0 {
        return 0;
    }
    let max_offset = capped_ms.saturating_mul(jitter_pct as u64) / 100;
    if max_offset == 0 {
        return 0;
    }
    let entropy = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    entropy % max_offset
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic_policy(max_attempts: u32) -> RetryPolicy {
        RetryPolicy {
            max_attempts,
            base_backoff: Duration::from_millis(10),
            max_backoff: Duration::from_millis(80),
            deadline: Duration::from_secs(60),
            jitter_pct: 0,
        }
    }

    #[test]
    fn compute_backoff_grows_exponentially_until_capped() {
        let policy = deterministic_policy(10);
        assert_eq!(compute_backoff(&policy, 1, None), Duration::from_millis(10));
        assert_eq!(compute_backoff(&policy, 2, None), Duration::from_millis(20));
        assert_eq!(compute_backoff(&policy, 3, None), Duration::from_millis(40));
        assert_eq!(compute_backoff(&policy, 4, None), Duration::from_millis(80));
        // Capped: further growth is clamped to max_backoff.
        assert_eq!(compute_backoff(&policy, 5, None), Duration::from_millis(80));
        assert_eq!(
            compute_backoff(&policy, 12, None),
            Duration::from_millis(80)
        );
    }

    #[test]
    fn compute_backoff_honors_hint_capped_to_max() {
        let policy = deterministic_policy(5);
        // Within cap: hint passes through unchanged.
        assert_eq!(
            compute_backoff(&policy, 1, Some(Duration::from_millis(45))),
            Duration::from_millis(45)
        );
        // Above cap: hostile server hint is clamped to max_backoff.
        assert_eq!(
            compute_backoff(&policy, 1, Some(Duration::from_secs(86_400))),
            Duration::from_millis(80)
        );
    }

    #[test]
    fn compute_backoff_with_jitter_stays_within_bounds() {
        let policy = RetryPolicy {
            max_attempts: 5,
            base_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_millis(100),
            deadline: Duration::from_secs(60),
            jitter_pct: 25,
        };
        for _ in 0..32 {
            let backoff = compute_backoff(&policy, 1, None);
            assert!(
                backoff >= Duration::from_millis(100),
                "below floor: {backoff:?}"
            );
            assert!(
                backoff <= Duration::from_millis(125),
                "above 25% jitter ceiling: {backoff:?}"
            );
        }
    }

    #[test]
    fn retry_gate_returns_none_after_max_attempts() {
        let mut gate = RetryGate::new(deterministic_policy(3));
        assert!(gate.record_failure_and_next_backoff(None).is_some()); // attempt 1
        assert!(gate.record_failure_and_next_backoff(None).is_some()); // attempt 2
        assert!(gate.record_failure_and_next_backoff(None).is_none()); // 3 → exhausted
    }

    #[tokio::test]
    async fn retry_gate_respects_deadline_independent_of_attempt_count() {
        let policy = RetryPolicy {
            max_attempts: 100,
            base_backoff: Duration::from_millis(1),
            max_backoff: Duration::from_millis(10),
            deadline: Duration::from_millis(50),
            jitter_pct: 0,
        };
        let mut gate = RetryGate::new(policy);
        assert!(gate.record_failure_and_next_backoff(None).is_some());
        tokio::time::sleep(Duration::from_millis(60)).await;
        // Even though we still have 98 attempts left, the deadline budget is
        // exhausted and we must stop.
        assert!(gate.record_failure_and_next_backoff(None).is_none());
    }

    #[test]
    fn retry_gate_next_attempt_id_tracks_observable_attempt() {
        let mut gate = RetryGate::new(deterministic_policy(5));
        assert_eq!(gate.next_attempt_id(), 1, "before any failure");
        gate.record_failure_and_next_backoff(None);
        assert_eq!(gate.next_attempt_id(), 2, "after first failure");
        gate.record_failure_and_next_backoff(None);
        assert_eq!(gate.next_attempt_id(), 3);
    }
}
