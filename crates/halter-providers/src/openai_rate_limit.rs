// pattern: Imperative Shell

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime};

use anyhow::Context;
use reqwest::StatusCode;
use reqwest::header::HeaderMap;
use sha2::{Digest, Sha256};
use tokio::select;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;
use tracing::debug;

use crate::openai_rate_limit_policy::{
    AcquireDecision, OpenAiRateLimitSnapshot, OpenAiRateLimitState, OpenAiReservation,
    OpenAiWindowSnapshot, initial_token_window, parse_openai_reset_duration,
    parse_retry_after_duration, reconcile_rate_limit_snapshot, try_acquire_reservation,
};

static OPENAI_RATE_LIMITS: OnceLock<Mutex<HashMap<OpenAiRateLimitKey, Arc<OpenAiRateLimitEntry>>>> =
    OnceLock::new();

const HEADER_LIMIT_REQUESTS: &str = "x-ratelimit-limit-requests";
const HEADER_LIMIT_TOKENS: &str = "x-ratelimit-limit-tokens";
const HEADER_REMAINING_REQUESTS: &str = "x-ratelimit-remaining-requests";
const HEADER_REMAINING_TOKENS: &str = "x-ratelimit-remaining-tokens";
const HEADER_RESET_REQUESTS: &str = "x-ratelimit-reset-requests";
const HEADER_RESET_TOKENS: &str = "x-ratelimit-reset-tokens";
const HEADER_RETRY_AFTER: &str = "retry-after";
const HEADER_RETRY_AFTER_MS: &str = "retry-after-ms";

#[derive(Debug, Clone)]
pub(crate) struct OpenAiRateLimiter {
    scope: OpenAiRateLimitScope,
}

#[derive(Debug, Clone)]
struct OpenAiRateLimitScope {
    base_url: String,
    credential_fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct OpenAiRateLimitKey {
    base_url: String,
    credential_fingerprint: String,
    model: String,
}

#[derive(Debug, Default)]
struct OpenAiRateLimitEntry {
    state: Mutex<OpenAiRateLimitState>,
    notify: Notify,
}

#[derive(Debug)]
pub(crate) struct OpenAiRateLimitPermit {
    entry: Arc<OpenAiRateLimitEntry>,
    reservation: OpenAiReservation,
    resolved: bool,
}

impl OpenAiRateLimiter {
    #[must_use]
    pub(crate) fn new(api_key: &str, base_url: &str) -> Self {
        Self {
            scope: OpenAiRateLimitScope {
                base_url: base_url.trim_end_matches('/').to_owned(),
                credential_fingerprint: fingerprint_api_key(api_key),
            },
        }
    }

    pub(crate) async fn acquire(
        &self,
        model: &str,
        reservation: OpenAiReservation,
        tokens_per_minute: Option<u64>,
        cancel: CancellationToken,
    ) -> anyhow::Result<OpenAiRateLimitPermit> {
        let entry = self.entry(model, tokens_per_minute);

        loop {
            let decision = {
                let mut state = entry.state.lock().expect("rate limit lock poisoned");
                try_acquire_reservation(&mut state, reservation, Instant::now())
            };

            match decision {
                AcquireDecision::Ready => {
                    return Ok(OpenAiRateLimitPermit {
                        entry: entry.clone(),
                        reservation,
                        resolved: false,
                    });
                }
                AcquireDecision::WaitUntil(deadline) => {
                    let delay = deadline.saturating_duration_since(Instant::now());
                    debug!(
                        model,
                        delay_ms = delay.as_millis(),
                        "waiting for openai rate limit"
                    );
                    let notified = entry.notify.notified();
                    tokio::pin!(notified);
                    let sleep = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline));
                    tokio::pin!(sleep);
                    select! {
                        _ = cancel.cancelled() => {
                            anyhow::bail!("failed to execute provider request: request cancelled");
                        }
                        _ = &mut notified => {}
                        _ = &mut sleep => {}
                    }
                }
            }
        }
    }

    fn entry(&self, model: &str, tokens_per_minute: Option<u64>) -> Arc<OpenAiRateLimitEntry> {
        let key = OpenAiRateLimitKey {
            base_url: self.scope.base_url.clone(),
            credential_fingerprint: self.scope.credential_fingerprint.clone(),
            model: model.to_owned(),
        };
        let registry = OPENAI_RATE_LIMITS.get_or_init(|| Mutex::new(HashMap::new()));
        let mut registry = registry.lock().expect("rate limit registry lock poisoned");
        registry
            .entry(key)
            .or_insert_with(|| {
                let state = OpenAiRateLimitState {
                    tokens: tokens_per_minute.map(initial_token_window),
                    ..OpenAiRateLimitState::default()
                };
                Arc::new(OpenAiRateLimitEntry {
                    state: Mutex::new(state),
                    notify: Notify::new(),
                })
            })
            .clone()
    }

    pub(crate) fn apply_retry_after(&self, model: &str, retry_after: Duration) {
        let entry = self.entry(model, None);
        {
            let mut state = entry.state.lock().expect("rate limit lock poisoned");
            reconcile_rate_limit_snapshot(
                &mut state,
                OpenAiReservation::default(),
                Some(OpenAiRateLimitSnapshot {
                    requests: None,
                    tokens: None,
                    retry_after: Some(retry_after),
                }),
                Instant::now(),
            );
        }
        entry.notify.notify_waiters();
    }
}

impl OpenAiRateLimitPermit {
    pub(crate) fn update_from_headers(&mut self, headers: &HeaderMap, _status: StatusCode) {
        self.resolve(snapshot_from_headers(headers));
    }

    pub(crate) fn release_without_headers(&mut self) {
        self.resolve(None);
    }

    fn resolve(&mut self, snapshot: Option<OpenAiRateLimitSnapshot>) {
        if self.resolved {
            return;
        }

        {
            let mut state = self.entry.state.lock().expect("rate limit lock poisoned");
            reconcile_rate_limit_snapshot(&mut state, self.reservation, snapshot, Instant::now());
        }
        self.entry.notify.notify_waiters();
        self.resolved = true;
    }
}

impl Drop for OpenAiRateLimitPermit {
    fn drop(&mut self) {
        self.release_without_headers();
    }
}

fn snapshot_from_headers(headers: &HeaderMap) -> Option<OpenAiRateLimitSnapshot> {
    let now_wall = SystemTime::now();
    let snapshot = OpenAiRateLimitSnapshot {
        requests: parse_window_snapshot(
            headers,
            HEADER_LIMIT_REQUESTS,
            HEADER_REMAINING_REQUESTS,
            HEADER_RESET_REQUESTS,
        ),
        tokens: parse_window_snapshot(
            headers,
            HEADER_LIMIT_TOKENS,
            HEADER_REMAINING_TOKENS,
            HEADER_RESET_TOKENS,
        ),
        retry_after: parse_retry_after(headers, now_wall),
    };

    (snapshot.requests.is_some() || snapshot.tokens.is_some() || snapshot.retry_after.is_some())
        .then_some(snapshot)
}

fn parse_window_snapshot(
    headers: &HeaderMap,
    limit_header: &str,
    remaining_header: &str,
    reset_header: &str,
) -> Option<OpenAiWindowSnapshot> {
    let limit = header_u64(headers, limit_header)?;
    let remaining = header_u64(headers, remaining_header)?;
    let reset_after = header_string(headers, reset_header)
        .as_deref()
        .and_then(parse_openai_reset_duration);
    Some(OpenAiWindowSnapshot {
        limit,
        remaining,
        reset_after,
    })
}

fn parse_retry_after(headers: &HeaderMap, now_wall: SystemTime) -> Option<Duration> {
    header_string(headers, HEADER_RETRY_AFTER_MS)
        .and_then(|value| value.trim().parse::<u64>().ok())
        .map(Duration::from_millis)
        .or_else(|| {
            header_string(headers, HEADER_RETRY_AFTER)
                .and_then(|value| parse_retry_after_duration(&value, now_wall))
        })
}

fn header_string(headers: &HeaderMap, name: &str) -> Option<String> {
    headers
        .get(name)
        .with_context(|| format!("missing header '{name}'"))
        .ok()?
        .to_str()
        .ok()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn header_u64(headers: &HeaderMap, name: &str) -> Option<u64> {
    header_string(headers, name)?.parse::<u64>().ok()
}

fn fingerprint_api_key(api_key: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(api_key.as_bytes());
    let digest = hasher.finalize();
    let mut fingerprint = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        let _ = write!(&mut fingerprint, "{byte:02x}");
    }
    fingerprint
}
