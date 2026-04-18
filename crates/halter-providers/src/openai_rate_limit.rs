// pattern: Imperative Shell

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
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
    registry: Arc<Mutex<HashMap<OpenAiRateLimitKey, Arc<OpenAiRateLimitEntry>>>>,
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
            registry: Arc::new(Mutex::new(HashMap::new())),
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
        let mut registry = self.registry.lock().expect("rate limit registry lock poisoned");
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

    /// Push a server-supplied retry-after hint into the limiter window for
    /// `model`. `tokens_per_minute` (H20) is plumbed in so a first-touch
    /// `apply_retry_after` for an entry that doesn't yet exist initializes
    /// the token window with the model's TPM ceiling. Without this the
    /// fresh entry is created with `tokens: None`, silently dropping the
    /// caller's TPM context until the next successful header reconcile.
    pub(crate) fn apply_retry_after(
        &self,
        model: &str,
        tokens_per_minute: Option<u64>,
        retry_after: Duration,
    ) {
        let entry = self.entry(model, tokens_per_minute);
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio_util::sync::CancellationToken;
    // `OpenAiReservation` is re-exported into scope by `use super::*;` via the
    // file-level `use crate::openai_rate_limit_policy::{OpenAiReservation, ...}`
    // at openai_rate_limit.rs:16-20. Do not add a duplicate `use` here — clippy
    // would flag it as unused.

    // helpers ------------------------------------------------------------

    fn limiter(api_key: &str, base_url: &str) -> OpenAiRateLimiter {
        OpenAiRateLimiter::new(api_key, base_url)
    }

    fn small_reservation() -> OpenAiReservation {
        // The exact numbers don't matter for isolation tests; pick a
        // reservation small enough that no policy will reject it.
        OpenAiReservation { requests: 1, tokens: 1 }
    }

    // AC1.1 Isolation ----------------------------------------------------

    #[test]
    fn separate_limiters_with_different_credentials_have_separate_entries() {
        let a = limiter("key-a", "https://api.openai.com");
        let b = limiter("key-b", "https://api.openai.com");

        let entry_a = a.entry("gpt-5", Some(500_000));
        let entry_b = b.entry("gpt-5", Some(500_000));

        assert!(
            !Arc::ptr_eq(&entry_a, &entry_b),
            "limiters with different credentials must not share per-model entries",
        );
    }

    // AC1.2 Lifecycle ----------------------------------------------------

    #[tokio::test]
    async fn dropping_limiter_drops_its_registry() {
        // Construct a limiter, apply a cooldown via apply_retry_after, then drop it.
        // Create a new limiter with the same scope. If the registry were leaked into
        // a static, the new limiter's entry would still have the cooldown. With
        // per-instance registries, the new entry's state is fresh (cooldown_until is None).
        {
            let lim = limiter("key-shared", "https://api.openai.com");
            lim.apply_retry_after("gpt-5", Some(500_000), Duration::from_secs(60));
        }

        // Create a fresh limiter with the same scope.
        let lim = limiter("key-shared", "https://api.openai.com");
        let entry = lim.entry("gpt-5", Some(500_000));

        // Lock the entry's state and verify cooldown_until is None.
        // If a static leaked state across limiter drops, cooldown_until would be set.
        let state = entry.state.lock().expect("state lock poisoned");
        assert!(
            state.cooldown_until.is_none(),
            "a fresh limiter's entry must have no cooldown — registry was leaked into a static",
        );
    }

    // AC1.3 Test isolation -----------------------------------------------

    #[tokio::test]
    async fn parallel_limiters_for_same_model_do_not_interfere() {
        // Set a cooldown on lim_a, then verify lim_b acquires immediately.
        // With a shared registry, lim_b would see lim_a's cooldown and block.
        // With isolated registries, lim_b acquires without waiting.
        let lim_a = limiter("key-iso", "https://api.openai.com");
        let lim_b = limiter("key-iso", "https://api.openai.com");

        // Apply a long cooldown to lim_a.
        lim_a.apply_retry_after("gpt-5", Some(500_000), Duration::from_secs(100));

        // Try to acquire from lim_b. With isolated registries, this should
        // succeed immediately (no cooldown visible). With shared state, it
        // would wait. We measure wall time to confirm "immediately".
        let cancel = CancellationToken::new();
        let start = Instant::now();
        let permit_b = lim_b
            .acquire("gpt-5", small_reservation(), Some(500_000), cancel)
            .await
            .expect("limiter b acquire");
        let elapsed = start.elapsed();

        // If lim_b saw lim_a's cooldown, elapsed would be ~100s.
        // With isolation, it should be << 1s.
        assert!(
            elapsed.as_secs() < 1,
            "lim_b must not see lim_a's cooldown — acquired in {}ms, expected <1000ms",
            elapsed.as_millis()
        );

        drop(permit_b);
    }

    // AC1.4 Per-credential keying preserved ------------------------------

    #[test]
    fn one_limiter_with_two_models_keeps_separate_entries() {
        let lim = limiter("key-multi", "https://api.openai.com");

        let gpt5 = lim.entry("gpt-5", Some(500_000));
        let gpt5_again = lim.entry("gpt-5", Some(500_000));
        let other = lim.entry("gpt-4o", Some(300_000));

        assert!(
            Arc::ptr_eq(&gpt5, &gpt5_again),
            "same model on the same limiter must reuse the same entry",
        );
        assert!(
            !Arc::ptr_eq(&gpt5, &other),
            "different models on the same limiter must allocate different entries",
        );
    }

    // Observer-clone contract -----------------------------------------------

    #[test]
    fn cloned_limiter_shares_registry_with_original() {
        // Cloning the limiter must hand back the same Arc-backed registry,
        // so the observer and the transport see the same per-model entry.
        let lim = limiter("key-clone", "https://api.openai.com");
        let cloned = lim.clone();

        let entry_orig = lim.entry("gpt-5", Some(500_000));
        let entry_clone = cloned.entry("gpt-5", Some(500_000));

        assert!(
            Arc::ptr_eq(&entry_orig, &entry_clone),
            "Clone must share the registry — observer construction relies on this",
        );
    }
}
