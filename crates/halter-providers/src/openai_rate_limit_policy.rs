// pattern: Functional Core

use std::time::{Duration, Instant, SystemTime};

use chrono::{DateTime, Utc};

pub(crate) const SAME_WINDOW_TOLERANCE: Duration = Duration::from_millis(1500);
const UNKNOWN_RESET_FALLBACK: Duration = Duration::from_secs(1);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) struct OpenAiReservation {
    pub requests: u64,
    pub tokens: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct OpenAiWindowSnapshot {
    pub limit: u64,
    pub remaining: u64,
    pub reset_after: Option<Duration>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct OpenAiRateLimitSnapshot {
    pub requests: Option<OpenAiWindowSnapshot>,
    pub tokens: Option<OpenAiWindowSnapshot>,
    pub retry_after: Option<Duration>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OpenAiWindowState {
    pub limit: u64,
    pub remaining: u64,
    pub reset_at: Option<Instant>,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct OpenAiRateLimitState {
    pub requests: Option<OpenAiWindowState>,
    pub tokens: Option<OpenAiWindowState>,
    pub reserved: OpenAiReservation,
    pub cooldown_until: Option<Instant>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AcquireDecision {
    Ready,
    WaitUntil(Instant),
}

pub(crate) fn estimate_openai_request_cost(
    request_json_bytes: usize,
    max_output_tokens: Option<u32>,
) -> OpenAiReservation {
    let input_tokens = (u64::try_from(request_json_bytes)
        .unwrap_or(u64::MAX)
        .saturating_add(3))
        / 4;
    OpenAiReservation {
        requests: 1,
        tokens: input_tokens.saturating_add(u64::from(max_output_tokens.unwrap_or(0))),
    }
}

pub(crate) fn try_acquire_reservation(
    state: &mut OpenAiRateLimitState,
    reservation: OpenAiReservation,
    now: Instant,
) -> AcquireDecision {
    refresh_state(state, now);

    let required_requests = state.reserved.requests.saturating_add(reservation.requests);
    let required_tokens = state.reserved.tokens.saturating_add(reservation.tokens);

    let next_deadline = [
        state.cooldown_until.filter(|deadline| *deadline > now),
        window_ready_deadline(state.requests.as_ref(), required_requests, now),
        window_ready_deadline(state.tokens.as_ref(), required_tokens, now),
    ]
    .into_iter()
    .flatten()
    .min();

    match next_deadline {
        Some(deadline) => AcquireDecision::WaitUntil(deadline),
        None => {
            state.reserved.requests = required_requests;
            state.reserved.tokens = required_tokens;
            AcquireDecision::Ready
        }
    }
}

pub(crate) fn reconcile_rate_limit_snapshot(
    state: &mut OpenAiRateLimitState,
    reservation: OpenAiReservation,
    snapshot: Option<OpenAiRateLimitSnapshot>,
    now: Instant,
) {
    refresh_state(state, now);
    state.reserved.requests = state.reserved.requests.saturating_sub(reservation.requests);
    state.reserved.tokens = state.reserved.tokens.saturating_sub(reservation.tokens);

    let Some(snapshot) = snapshot else {
        return;
    };

    if let Some(retry_after) = snapshot.retry_after {
        let retry_at = now + retry_after;
        state.cooldown_until = Some(
            state
                .cooldown_until
                .map_or(retry_at, |current| current.max(retry_at)),
        );
    }

    if let Some(window) = snapshot.requests {
        apply_window_snapshot(&mut state.requests, window, now);
    }
    if let Some(window) = snapshot.tokens {
        apply_window_snapshot(&mut state.tokens, window, now);
    }
}

pub(crate) fn parse_openai_reset_duration(value: &str) -> Option<Duration> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }

    let bytes = value.as_bytes();
    let mut index = 0usize;
    let mut total_millis = 0u64;

    while index < bytes.len() {
        let number_start = index;
        while index < bytes.len() && bytes[index].is_ascii_digit() {
            index += 1;
        }
        if number_start == index {
            return None;
        }

        let amount = value[number_start..index].parse::<u64>().ok()?;
        let (unit_millis, consumed) = if value[index..].starts_with("ms") {
            (1u64, 2usize)
        } else if value[index..].starts_with('s') {
            (1_000u64, 1usize)
        } else if value[index..].starts_with('m') {
            (60_000u64, 1usize)
        } else if value[index..].starts_with('h') {
            (3_600_000u64, 1usize)
        } else {
            return None;
        };
        index += consumed;
        total_millis = total_millis.checked_add(amount.checked_mul(unit_millis)?)?;
    }

    Some(Duration::from_millis(total_millis))
}

pub(crate) fn parse_retry_after_duration(value: &str, now_wall: SystemTime) -> Option<Duration> {
    let value = value.trim();
    if value.is_empty() {
        return None;
    }

    if let Ok(seconds) = value.parse::<u64>() {
        return Some(Duration::from_secs(seconds));
    }

    let retry_at = DateTime::parse_from_rfc2822(value)
        .ok()?
        .with_timezone(&Utc);
    let now = DateTime::<Utc>::from(now_wall);
    let millis = retry_at
        .signed_duration_since(now)
        .num_milliseconds()
        .max(0);
    Some(Duration::from_millis(u64::try_from(millis).ok()?))
}

fn refresh_state(state: &mut OpenAiRateLimitState, now: Instant) {
    refresh_window(&mut state.requests, now);
    refresh_window(&mut state.tokens, now);

    if state.cooldown_until.is_some_and(|deadline| deadline <= now) {
        state.cooldown_until = None;
    }
}

fn refresh_window(window: &mut Option<OpenAiWindowState>, now: Instant) {
    let Some(window) = window.as_mut() else {
        return;
    };

    if window.reset_at.is_some_and(|reset_at| reset_at <= now) {
        window.remaining = window.limit;
        window.reset_at = None;
    }
}

fn window_ready_deadline(
    window: Option<&OpenAiWindowState>,
    required_reserved: u64,
    now: Instant,
) -> Option<Instant> {
    let window = window?;
    if window.remaining >= required_reserved {
        return None;
    }

    Some(window.reset_at.unwrap_or(now + UNKNOWN_RESET_FALLBACK))
}

fn apply_window_snapshot(
    slot: &mut Option<OpenAiWindowState>,
    snapshot: OpenAiWindowSnapshot,
    now: Instant,
) {
    let incoming = OpenAiWindowState {
        limit: snapshot.limit,
        remaining: snapshot.remaining.min(snapshot.limit),
        reset_at: snapshot.reset_after.map(|duration| now + duration),
    };

    if slot.is_none() {
        *slot = Some(incoming);
        return;
    }
    refresh_window(slot, now);
    let current = slot.as_mut().expect("window must exist after refresh");

    match (current.reset_at, incoming.reset_at) {
        (Some(current_reset), Some(incoming_reset))
            if same_window(current_reset, incoming_reset) =>
        {
            current.limit = incoming.limit;
            current.remaining = current.remaining.min(incoming.remaining);
            current.reset_at = Some(current_reset.max(incoming_reset));
        }
        (Some(current_reset), Some(incoming_reset)) if incoming_reset > current_reset => {
            *current = incoming;
        }
        (None, _) => {
            *current = incoming;
        }
        (_, None) => {
            current.limit = incoming.limit;
            current.remaining = current.remaining.min(incoming.remaining);
        }
        _ => {}
    }
}

fn same_window(left: Instant, right: Instant) -> bool {
    if left >= right {
        left.duration_since(right) <= SAME_WINDOW_TOLERANCE
    } else {
        right.duration_since(left) <= SAME_WINDOW_TOLERANCE
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_openai_reset_duration_supports_compound_values() {
        struct TestCase {
            name: &'static str,
            input: &'static str,
            want: Option<Duration>,
        }

        let tests = [
            TestCase {
                name: "seconds",
                input: "1s",
                want: Some(Duration::from_secs(1)),
            },
            TestCase {
                name: "minutes_and_seconds",
                input: "6m0s",
                want: Some(Duration::from_secs(360)),
            },
            TestCase {
                name: "minutes_seconds_and_millis",
                input: "2m3s150ms",
                want: Some(Duration::from_millis(123_150)),
            },
            TestCase {
                name: "invalid",
                input: "soon",
                want: None,
            },
        ];

        for test in tests {
            assert_eq!(
                parse_openai_reset_duration(test.input),
                test.want,
                "{}",
                test.name
            );
        }
    }

    #[test]
    fn parse_retry_after_duration_supports_seconds_and_http_dates() {
        let now_wall = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
        let http_date = "Tue, 14 Nov 2023 22:13:25 GMT";

        assert_eq!(
            parse_retry_after_duration("2", now_wall),
            Some(Duration::from_secs(2))
        );
        assert_eq!(
            parse_retry_after_duration(http_date, now_wall),
            Some(Duration::from_secs(5))
        );
    }

    #[test]
    fn try_acquire_reservation_waits_for_reset_when_requests_are_exhausted() {
        let now = Instant::now();
        let mut state = OpenAiRateLimitState {
            requests: Some(OpenAiWindowState {
                limit: 2,
                remaining: 1,
                reset_at: Some(now + Duration::from_secs(3)),
            }),
            reserved: OpenAiReservation {
                requests: 1,
                tokens: 0,
            },
            ..OpenAiRateLimitState::default()
        };

        let decision = try_acquire_reservation(
            &mut state,
            OpenAiReservation {
                requests: 1,
                tokens: 0,
            },
            now,
        );

        assert_eq!(
            decision,
            AcquireDecision::WaitUntil(now + Duration::from_secs(3))
        );
        assert_eq!(state.reserved.requests, 1);
    }

    #[test]
    fn reconcile_rate_limit_snapshot_uses_lowest_remaining_within_same_window() {
        let now = Instant::now();
        let mut state = OpenAiRateLimitState {
            requests: Some(OpenAiWindowState {
                limit: 60,
                remaining: 10,
                reset_at: Some(now + Duration::from_secs(30)),
            }),
            ..OpenAiRateLimitState::default()
        };

        reconcile_rate_limit_snapshot(
            &mut state,
            OpenAiReservation::default(),
            Some(OpenAiRateLimitSnapshot {
                requests: Some(OpenAiWindowSnapshot {
                    limit: 60,
                    remaining: 12,
                    reset_after: Some(Duration::from_secs(29)),
                }),
                ..OpenAiRateLimitSnapshot::default()
            }),
            now,
        );

        let requests = state.requests.expect("request window");
        assert_eq!(requests.remaining, 10);
        assert!(requests.reset_at.expect("reset") >= now + Duration::from_secs(29));
    }

    #[test]
    fn reconcile_rate_limit_snapshot_replaces_stale_window_after_reset() {
        let now = Instant::now();
        let mut state = OpenAiRateLimitState {
            requests: Some(OpenAiWindowState {
                limit: 60,
                remaining: 0,
                reset_at: Some(now + Duration::from_secs(5)),
            }),
            ..OpenAiRateLimitState::default()
        };

        reconcile_rate_limit_snapshot(
            &mut state,
            OpenAiReservation::default(),
            Some(OpenAiRateLimitSnapshot {
                requests: Some(OpenAiWindowSnapshot {
                    limit: 60,
                    remaining: 59,
                    reset_after: Some(Duration::from_secs(65)),
                }),
                ..OpenAiRateLimitSnapshot::default()
            }),
            now,
        );

        let requests = state.requests.expect("request window");
        assert_eq!(requests.remaining, 59);
        assert_eq!(requests.reset_at, Some(now + Duration::from_secs(65)));
    }
}
