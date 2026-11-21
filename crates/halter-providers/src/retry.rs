// pattern: Functional Core

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hasher};
use std::time::Duration;

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
    /// Cumulative budget for time spent *waiting between* attempts (the sum
    /// of every backoff). Time spent inside an attempt never counts, so no
    /// attempt timer (setup `request`, `stream_idle`, a slow prefix) can
    /// exhaust the budget on its own: each attempt is bounded by its own
    /// timers, and the sequence by `max_attempts` and this budget. Once
    /// spent, further retries are denied even if `max_attempts` has not been
    /// hit; the last backoff is clamped to what remains.
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

/// Stateful retry tracker. Combines the per-attempt counter with the
/// cumulative backoff budget so callers don't have to thread `waited` and
/// `attempt: u32` separately. It reads no clock: the outcome depends only on
/// the policy and the sequence of hints. The contract is intentionally narrow: callers ask for the next
/// backoff after a failed attempt; `None` means the budget is exhausted and
/// the failure should be propagated.
#[derive(Debug)]
pub(crate) struct RetryGate {
    policy: RetryPolicy,
    waited: Duration,
    completed_attempts: u32,
}

impl RetryGate {
    pub(crate) fn new(policy: RetryPolicy) -> Self {
        Self {
            policy,
            waited: Duration::ZERO,
            completed_attempts: 0,
        }
    }

    /// 1-indexed attempt counter for the *next* attempt that will be tried.
    /// Used to tag retry logs with the attempt sequence number.
    pub(crate) fn next_attempt_id(&self) -> u32 {
        self.completed_attempts.saturating_add(1)
    }

    /// Records a failed attempt and returns the backoff to wait before the
    /// next attempt, or `None` if the budget (`max_attempts` or the backoff
    /// budget `deadline`) is exhausted. The returned delay is charged to the
    /// budget before it is slept. The optional `hint` is a server-supplied retry-after
    /// duration (e.g. parsed from `Please try again in 1.25s`) which takes
    /// precedence over the computed exponential when present.
    pub(crate) fn record_failure_and_next_backoff(
        &mut self,
        hint: Option<Duration>,
    ) -> Option<Duration> {
        self.completed_attempts = self.completed_attempts.saturating_add(1);
        let remaining = self.policy.deadline.saturating_sub(self.waited);
        if self.completed_attempts >= self.policy.max_attempts || remaining.is_zero() {
            return None;
        }
        let delay = compute_backoff(&self.policy, self.completed_attempts, hint).min(remaining);
        self.waited += delay;
        Some(delay)
    }
}

/// Pure backoff computation. Honors a server-supplied hint when present
/// (capped at `policy.max_backoff` so a hostile server can't pin us into a
/// 24-hour wait), otherwise produces jittered exponential. Jitter is applied
/// inside the `max_backoff` cap: the returned delay never exceeds
/// `policy.max_backoff`.
#[must_use]
pub(crate) fn compute_backoff(
    policy: &RetryPolicy,
    attempt: u32,
    hint: Option<Duration>,
) -> Duration {
    let max_ms = duration_millis_u64(policy.max_backoff);
    let base_ms = match hint {
        Some(hint) => duration_millis_u64(hint).min(max_ms),
        None => {
            let exponent = attempt.saturating_sub(1).min(20);
            (policy.base_backoff.as_millis() as u64)
                .saturating_mul(1u64 << exponent)
                .min(max_ms)
        }
    };
    let jitter = jitter_offset(base_ms, policy.jitter_pct);
    Duration::from_millis(base_ms.saturating_add(jitter).min(max_ms))
}

fn duration_millis_u64(duration: Duration) -> u64 {
    u64::try_from(duration.as_millis()).unwrap_or(u64::MAX)
}

/// Jitter offset in `[0, capped * jitter_pct / 100)`. Every `RandomState`
/// instance carries fresh per-instance keys from the platform RNG, so an
/// empty hasher's `finish()` already differs call to call — no direct RNG
/// dependency needed. The modulo bias is irrelevant at backoff granularity.
fn jitter_offset(capped_ms: u64, jitter_pct: u32) -> u64 {
    if jitter_pct == 0 || capped_ms == 0 {
        return 0;
    }
    let max_offset = capped_ms.saturating_mul(jitter_pct as u64) / 100;
    if max_offset == 0 {
        return 0;
    }
    RandomState::new().build_hasher().finish() % max_offset
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
    fn compute_backoff_jitter_never_exceeds_max_backoff() {
        // At the cap, jitter must be absorbed by the clamp: previously the
        // offset was added after the clamp and could reach 125ms here.
        let policy = RetryPolicy {
            max_attempts: 5,
            base_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_millis(100),
            deadline: Duration::from_secs(60),
            jitter_pct: 25,
        };
        for _ in 0..32 {
            assert_eq!(
                compute_backoff(&policy, 1, None),
                Duration::from_millis(100),
                "jitter must stay inside max_backoff"
            );
            assert_eq!(
                compute_backoff(&policy, 1, Some(Duration::from_millis(100))),
                Duration::from_millis(100),
                "hint jitter must stay inside max_backoff"
            );
        }
    }

    #[test]
    fn compute_backoff_applies_jitter_below_the_cap() {
        let policy = RetryPolicy {
            max_attempts: 5,
            base_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_secs(10),
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
                backoff < Duration::from_millis(125),
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

    /// Budget accounting over (policy, time inside attempts, hints).
    /// Mutations killed: charging wall time spent inside attempts (the first
    /// row sleeps past the budget before failing and must still retry);
    /// dropping `self.waited += delay` (rows two and three keep returning full
    /// backoffs); dropping `.min(remaining)` (the clamped 5ms and 20ms
    /// entries); dropping `|| remaining.is_zero()` (the trailing `None`
    /// entries of rows two and three).
    #[test]
    fn retry_gate_budget_counts_backoff_only() {
        let ms = Duration::from_millis;
        // (name, max_attempts, base, max, deadline, ms inside attempt 1, hints, expected)
        type Case = (
            &'static str,
            u32,
            u64,
            u64,
            u64,
            u64,
            &'static [Option<u64>],
            &'static [Option<u64>],
        );
        let cases: [Case; 4] = [
            (
                "attempt longer than the budget is still retried",
                3,
                10,
                10,
                20,
                40,
                &[None, None, None],
                &[Some(10), Some(10), None],
            ),
            (
                "backoffs spend the budget before max_attempts",
                100,
                10,
                10,
                25,
                0,
                &[None, None, None, None],
                &[Some(10), Some(10), Some(5), None],
            ),
            (
                "server hint is charged and clamped to the remaining budget",
                100,
                1,
                100,
                50,
                0,
                &[Some(30), Some(30), Some(30)],
                &[Some(30), Some(20), None],
            ),
            (
                "single attempt never retries",
                1,
                10,
                10,
                1_000,
                0,
                &[None],
                &[None],
            ),
        ];
        for (name, max_attempts, base, max, deadline, inside, hints, expected) in cases {
            let mut gate = RetryGate::new(RetryPolicy {
                max_attempts,
                base_backoff: ms(base),
                max_backoff: ms(max),
                deadline: ms(deadline),
                jitter_pct: 0,
            });
            std::thread::sleep(ms(inside));
            let observed: Vec<_> = hints
                .iter()
                .map(|hint| gate.record_failure_and_next_backoff(hint.map(ms)))
                .collect();
            let expected: Vec<_> = expected.iter().map(|value| value.map(ms)).collect();
            assert_eq!(observed, expected, "{name}");
        }
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
