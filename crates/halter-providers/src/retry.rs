// pattern: Functional Core

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::time::{Duration, Instant, SystemTime};

/// Bounded retry policy shared by the streaming provider pipeline. Replaces
/// the previous unbounded `loop { ... continue; }` retry in
/// `responses_provider`, which had no per-stream cap and no cumulative
/// deadline (AC3.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetryPolicy {
    /// Total attempts allowed (initial attempt + retries).
    pub max_attempts: u32,
    /// Base delay used as the seed for exponential growth.
    pub base_backoff: Duration,
    /// Upper bound for any single backoff, even if exponential growth or a
    /// server-supplied hint would exceed it.
    pub max_backoff: Duration,
    /// Cumulative wall time budget across all attempts. Once exceeded,
    /// further retries are denied even if `max_attempts` has not been hit.
    pub deadline: Duration,
    /// Random offset, expressed as a percentage of the computed backoff,
    /// added on top to break thundering-herd synchronization across
    /// concurrent retrying clients. `0` disables jitter (used in tests for
    /// deterministic timing).
    pub jitter_pct: u32,
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
    completed_attempts: u32,
}

impl RetryGate {
    pub(crate) fn new(policy: RetryPolicy) -> Self {
        Self {
            policy,
            started: Instant::now(),
            completed_attempts: 0,
        }
    }

    /// 1-indexed attempt counter for the *next* attempt that will be tried.
    /// Useful for tagging a stream attempt with its sequence number when
    /// deduplicating cross-attempt events (AC3.6).
    pub(crate) fn next_attempt_id(&self) -> u32 {
        self.completed_attempts.saturating_add(1)
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
        self.completed_attempts = self.completed_attempts.saturating_add(1);
        if self.completed_attempts >= self.policy.max_attempts
            || self.started.elapsed() >= self.policy.deadline
        {
            return None;
        }
        Some(compute_backoff(&self.policy, self.completed_attempts, hint))
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
        let capped_ms = duration_millis_u64(hint.min(policy.max_backoff));
        let jitter = jitter_offset(capped_ms, policy.jitter_pct);
        return Duration::from_millis(capped_ms.saturating_add(jitter));
    }
    let exponent = attempt.saturating_sub(1).min(20);
    let exp_ms = (policy.base_backoff.as_millis() as u64).saturating_mul(1u64 << exponent);
    let capped_ms = exp_ms.min(duration_millis_u64(policy.max_backoff));
    let jitter = jitter_offset(capped_ms, policy.jitter_pct);
    Duration::from_millis(capped_ms.saturating_add(jitter))
}

fn duration_millis_u64(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

/// Jitter offset in `[0, capped * jitter_pct / 100]`. `RandomState` seeds from
/// the platform RNG, then the wall-clock sample gives each call a distinct
/// input without adding a direct RNG dependency to this crate.
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
    let mut hasher = RandomState::new().build_hasher();
    hasher.write_u64(entropy);
    hasher.finish() % max_offset
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
    fn compute_backoff_applies_jitter_to_hints() {
        let policy = RetryPolicy {
            max_attempts: 5,
            base_backoff: Duration::from_millis(10),
            max_backoff: Duration::from_millis(100),
            deadline: Duration::from_secs(60),
            jitter_pct: 25,
        };
        for _ in 0..32 {
            let backoff = compute_backoff(&policy, 1, Some(Duration::from_millis(40)));
            assert!(backoff >= Duration::from_millis(40));
            assert!(backoff <= Duration::from_millis(50));
        }
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

    #[test]
    fn retry_gate_max_attempts_means_total_calls() {
        let mut gate = RetryGate::new(deterministic_policy(5));

        for failed_attempt in 1..=4 {
            assert_eq!(gate.next_attempt_id(), failed_attempt);
            assert!(gate.record_failure_and_next_backoff(None).is_some());
        }
        assert_eq!(gate.next_attempt_id(), 5);
        assert!(gate.record_failure_and_next_backoff(None).is_none());
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
