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
use crate::secret::SecretString;

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
    pub(crate) fn new(api_key: &SecretString, base_url: &str) -> Self {
        Self {
            scope: OpenAiRateLimitScope {
                base_url: base_url.trim_end_matches('/').to_owned(),
                credential_fingerprint: fingerprint_api_key(api_key.expose_secret()),
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
        let mut registry = self
            .registry
            .lock()
            .expect("rate limit registry lock poisoned");
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

#[cfg(test)]
impl OpenAiRateLimiter {
    /// Test-only inspection of the per-model cooldown. Returns `Some(_)`
    /// when a `Retry-After` window has been applied (via header or
    /// `apply_retry_after`) and is still in effect.
    pub(crate) fn cooldown_for_test(
        &self,
        model: &str,
        tokens_per_minute: Option<u64>,
    ) -> Option<Instant> {
        let entry = self.entry(model, tokens_per_minute);
        let state = entry.state.lock().expect("rate limit lock poisoned");
        state.cooldown_until
    }

    /// Test-only inspection of the per-model token window limit. Returns
    /// `Some(limit)` when a token window has been seeded (via `apply_retry_after`
    /// with `Some(tpm)` or via header reconciliation), or `None` if the window
    /// does not exist.
    pub(crate) fn token_window_limit_for_test(
        &self,
        model: &str,
        tokens_per_minute: Option<u64>,
    ) -> Option<u64> {
        let entry = self.entry(model, tokens_per_minute);
        let state = entry.state.lock().expect("rate limit lock poisoned");
        state.tokens.as_ref().map(|w| w.limit)
    }
}

impl OpenAiRateLimitPermit {
    pub(crate) fn update_from_headers(&mut self, headers: &HeaderMap, status: StatusCode) {
        let snapshot = if status.is_success() || status == StatusCode::TOO_MANY_REQUESTS {
            // 2xx success and 429 rate-limit responses both carry
            // authoritative budget headers — reconcile per-model windows
            // from them.
            snapshot_from_headers(headers)
        } else {
            // Other 4xx and all 5xx: release the reservation but do not
            // reconcile windows. A non-rate-limit 4xx may have rejected
            // the request before the server applied budget; a 5xx may
            // carry stale or fabricated headers from an upstream proxy.
            None
        };
        self.resolve(snapshot);
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
    hex::encode(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    use reqwest::header::HeaderMap;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio_util::sync::CancellationToken;
    // `OpenAiReservation` is re-exported into scope by `use super::*;` via the
    // file-level `use crate::openai_rate_limit_policy::{OpenAiReservation, ...}`
    // at openai_rate_limit.rs:16-20. Do not add a duplicate `use` here — clippy
    // would flag it as unused.

    // helpers ------------------------------------------------------------

    fn limiter(api_key: &str, base_url: &str) -> OpenAiRateLimiter {
        OpenAiRateLimiter::new(&SecretString::from(api_key), base_url)
    }

    fn small_reservation() -> OpenAiReservation {
        // The exact numbers don't matter for isolation tests; pick a
        // reservation small enough that no policy will reject it.
        OpenAiReservation {
            requests: 1,
            tokens: 1,
        }
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

    #[test]
    fn dropping_limiter_drops_its_registry() {
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

    // Task 2 helpers and tests for status-aware header reconcile -----------

    // Helper: build a HeaderMap that snapshot_from_headers() will parse
    // into a non-None snapshot. We use the same headers shape as
    // spawn_sse_server in responses_transport.rs.
    fn rate_limit_headers(remaining_requests: &str, reset_requests: &str) -> HeaderMap {
        let mut h = HeaderMap::new();
        h.insert("x-ratelimit-limit-requests", "100".parse().unwrap());
        h.insert(
            "x-ratelimit-remaining-requests",
            remaining_requests.parse().unwrap(),
        );
        h.insert(
            "x-ratelimit-reset-requests",
            reset_requests.parse().unwrap(),
        );
        h
    }

    // Read the per-model `remaining` count for the requests window via
    // the entry registry. `OpenAiRateLimitState` exposes its windows as
    // bare `pub` fields (`pub requests: Option<OpenAiWindowState>`), so
    // we use `.as_ref().map(...)` directly — there is no
    // `requests_window()` accessor method.
    fn remaining_requests_for(lim: &OpenAiRateLimiter, model: &str) -> Option<u64> {
        let entry = lim.entry(model, Some(500_000));
        let state = entry.state.lock().expect("lock");
        state.requests.as_ref().map(|w| w.remaining)
    }

    // AC3.1 2xx reconciles ---------------------------------------------

    #[tokio::test]
    async fn update_from_headers_with_2xx_status_reconciles_windows() {
        let lim = limiter("key-2xx", "https://api.openai.com");
        let mut permit = lim
            .acquire(
                "gpt-5",
                OpenAiReservation {
                    requests: 1,
                    tokens: 1,
                },
                Some(500_000),
                CancellationToken::new(),
            )
            .await
            .expect("acquire");

        let headers = rate_limit_headers("42", "60s");
        permit.update_from_headers(&headers, StatusCode::OK);

        let remaining = remaining_requests_for(&lim, "gpt-5");
        assert_eq!(
            remaining,
            Some(42),
            "2xx must reconcile per-model windows from headers",
        );
    }

    // AC3.2 429 reconciles ---------------------------------------------

    #[tokio::test]
    async fn update_from_headers_with_429_status_reconciles_windows() {
        let lim = limiter("key-429", "https://api.openai.com");
        let mut permit = lim
            .acquire(
                "gpt-5",
                OpenAiReservation {
                    requests: 1,
                    tokens: 1,
                },
                Some(500_000),
                CancellationToken::new(),
            )
            .await
            .expect("acquire");

        let headers = rate_limit_headers("0", "30s");
        permit.update_from_headers(&headers, StatusCode::TOO_MANY_REQUESTS);

        let remaining = remaining_requests_for(&lim, "gpt-5");
        assert_eq!(
            remaining,
            Some(0),
            "429 must reconcile per-model windows from headers (limit hint is authoritative)",
        );
    }

    // AC3.3 5xx skips reconcile ----------------------------------------

    #[tokio::test]
    async fn update_from_headers_with_5xx_status_skips_reconcile() {
        let lim = limiter("key-5xx", "https://api.openai.com");

        // First, prime the window via a successful 2xx so we have a
        // known "remaining" baseline.
        let mut prime = lim
            .acquire(
                "gpt-5",
                OpenAiReservation {
                    requests: 1,
                    tokens: 1,
                },
                Some(500_000),
                CancellationToken::new(),
            )
            .await
            .expect("acquire prime");
        prime.update_from_headers(&rate_limit_headers("99", "60s"), StatusCode::OK);
        let baseline = remaining_requests_for(&lim, "gpt-5");
        assert_eq!(baseline, Some(99));

        // Now: a 5xx with bogus "remaining=1" must NOT reconcile.
        let mut permit = lim
            .acquire(
                "gpt-5",
                OpenAiReservation {
                    requests: 1,
                    tokens: 1,
                },
                Some(500_000),
                CancellationToken::new(),
            )
            .await
            .expect("acquire 5xx");
        permit.update_from_headers(
            &rate_limit_headers("1", "60s"),
            StatusCode::INTERNAL_SERVER_ERROR,
        );

        // Window state must remain at the baseline; the bogus header is
        // ignored.
        assert_eq!(
            remaining_requests_for(&lim, "gpt-5"),
            baseline,
            "5xx must not reconcile per-model windows",
        );
    }

    // AC3.4 Other 4xx skips reconcile ----------------------------------

    #[tokio::test]
    async fn update_from_headers_with_non_429_4xx_skips_reconcile() {
        let lim = limiter("key-4xx", "https://api.openai.com");

        let mut prime = lim
            .acquire(
                "gpt-5",
                OpenAiReservation {
                    requests: 1,
                    tokens: 1,
                },
                Some(500_000),
                CancellationToken::new(),
            )
            .await
            .expect("acquire prime");
        prime.update_from_headers(&rate_limit_headers("99", "60s"), StatusCode::OK);
        let baseline = remaining_requests_for(&lim, "gpt-5");

        let mut permit = lim
            .acquire(
                "gpt-5",
                OpenAiReservation {
                    requests: 1,
                    tokens: 1,
                },
                Some(500_000),
                CancellationToken::new(),
            )
            .await
            .expect("acquire 4xx");
        permit.update_from_headers(&rate_limit_headers("1", "60s"), StatusCode::BAD_REQUEST);

        assert_eq!(
            remaining_requests_for(&lim, "gpt-5"),
            baseline,
            "non-429 4xx must not reconcile per-model windows",
        );
    }

    // AC3.5 Reservation always released --------------------------------

    #[tokio::test]
    async fn update_from_headers_releases_reservation_for_every_status() {
        // Every supported status path must drop the reserved budget.
        // We count `state.reserved.requests` after each call and assert
        // it returns to 0.
        for (label, status) in [
            ("200", StatusCode::OK),
            ("429", StatusCode::TOO_MANY_REQUESTS),
            ("500", StatusCode::INTERNAL_SERVER_ERROR),
            ("400", StatusCode::BAD_REQUEST),
        ] {
            let lim = limiter(&format!("key-{label}"), "https://api.openai.com");
            let mut permit = lim
                .acquire(
                    "gpt-5",
                    OpenAiReservation {
                        requests: 1,
                        tokens: 1,
                    },
                    Some(500_000),
                    CancellationToken::new(),
                )
                .await
                .expect("acquire");

            // Sanity: reservation is recorded.
            {
                let entry = lim.entry("gpt-5", Some(500_000));
                let state = entry.state.lock().expect("lock");
                assert_eq!(
                    state.reserved.requests, 1,
                    "{label}: reservation should be recorded after acquire",
                );
            }

            permit.update_from_headers(&rate_limit_headers("50", "60s"), status);

            let entry = lim.entry("gpt-5", Some(500_000));
            let state = entry.state.lock().expect("lock");
            assert_eq!(
                state.reserved.requests, 0,
                "{label}: reservation must be released by update_from_headers",
            );
        }
    }

    // AC3.5 (bis) Drop also releases when permit was never resolved -----

    #[tokio::test]
    async fn drop_releases_reservation_when_update_from_headers_was_not_called() {
        let lim = limiter("key-drop", "https://api.openai.com");
        {
            let _permit = lim
                .acquire(
                    "gpt-5",
                    OpenAiReservation {
                        requests: 1,
                        tokens: 1,
                    },
                    Some(500_000),
                    CancellationToken::new(),
                )
                .await
                .expect("acquire");

            let entry = lim.entry("gpt-5", Some(500_000));
            let state = entry.state.lock().expect("lock");
            assert_eq!(state.reserved.requests, 1);
        } // _permit drops here

        let entry = lim.entry("gpt-5", Some(500_000));
        let state = entry.state.lock().expect("lock");
        assert_eq!(
            state.reserved.requests, 0,
            "Drop must release reservation even without an explicit update_from_headers",
        );
    }

    // AC2.1 Stream startup: acquire(model, _, Some(tpm)) creates a token
    // window with that TPM ceiling.
    #[tokio::test]
    async fn acquire_with_tpm_some_creates_token_window_at_that_ceiling() {
        let lim = limiter("key-tpm-startup", "https://api.openai.com");
        let permit = lim
            .acquire(
                "gpt-5",
                OpenAiReservation {
                    requests: 1,
                    tokens: 1,
                },
                Some(123_456),
                CancellationToken::new(),
            )
            .await
            .expect("acquire");

        let entry = lim.entry("gpt-5", Some(123_456));
        let state = entry.state.lock().expect("lock");
        let window = state
            .tokens
            .as_ref()
            .expect("acquire with TPM Some must seed a token window");
        assert_eq!(window.limit, 123_456);
        drop(state);
        drop(permit);
    }

    // AC2.2 Mid-stream observer: apply_retry_after with Some(tpm) sets a
    // cooldown that subsequent acquires honor.
    #[tokio::test]
    async fn apply_retry_after_with_tpm_some_sets_cooldown() {
        let lim = limiter("key-tpm-midstream", "https://api.openai.com");
        // Seed an entry first.
        let permit = lim
            .acquire(
                "gpt-5",
                OpenAiReservation {
                    requests: 1,
                    tokens: 1,
                },
                Some(500_000),
                CancellationToken::new(),
            )
            .await
            .expect("acquire");
        drop(permit);

        // Apply a retry-after as if the observer had seen a mid-stream
        // rate-limit error.
        lim.apply_retry_after("gpt-5", Some(500_000), Duration::from_millis(50));

        let entry = lim.entry("gpt-5", Some(500_000));
        let state = entry.state.lock().expect("lock");
        assert!(
            state.cooldown_until.is_some(),
            "apply_retry_after with TPM Some must set cooldown",
        );
    }

    // AC2.4 No silent default: apply_retry_after with TPM None does not
    // clobber an existing token window seeded by a prior Some(tpm) call.
    #[tokio::test]
    async fn apply_retry_after_with_tpm_none_does_not_clobber_existing_window() {
        let lim = limiter("key-tpm-nonclobber", "https://api.openai.com");

        // First call: seed token window with Some(500_000).
        {
            let permit = lim
                .acquire(
                    "gpt-5",
                    OpenAiReservation {
                        requests: 1,
                        tokens: 1,
                    },
                    Some(500_000),
                    CancellationToken::new(),
                )
                .await
                .expect("acquire");
            drop(permit);
        }

        let window_before = {
            let entry = lim.entry("gpt-5", Some(500_000));
            let state = entry.state.lock().expect("lock");
            state.tokens.as_ref().map(|w| w.limit)
        };
        assert_eq!(window_before, Some(500_000));

        // Now apply_retry_after with None.
        lim.apply_retry_after("gpt-5", None, Duration::from_millis(10));

        // Token window must NOT have been replaced with None / dropped.
        let window_after = {
            let entry = lim.entry("gpt-5", None);
            let state = entry.state.lock().expect("lock");
            state.tokens.as_ref().map(|w| w.limit)
        };
        assert_eq!(
            window_after,
            Some(500_000),
            "apply_retry_after(None) must not clobber an existing token window",
        );
    }
}
