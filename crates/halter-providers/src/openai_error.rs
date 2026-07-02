// pattern: Functional Core

use std::time::Duration;

use async_openai::error::{ApiError, OpenAIError};
use halter_protocol::ProviderErrorKind;
use serde_json::Value;

const RATE_LIMIT_CODE: &str = "rate_limit_exceeded";
const RETRY_AFTER_PREFIX: &str = "Please try again in ";
const INFERRED_CAPACITY_BACKOFF_MS: u64 = 750;
/// Sentinel substring OpenRouter uses to wrap upstream-provider failures
/// (`{"error": {"message": "Provider returned error", "metadata": {...}}}`).
/// These are typically transient — the upstream model rate-limited, timed
/// out, or hiccuped — and should be retried instead of cascading into a
/// fatal turn failure.
const OPENROUTER_UPSTREAM_WRAPPER: &str = "Provider returned error";
/// Synthetic code stamped by the transport layer when an HTTP 5xx response
/// has no parseable JSON error body. `classify` treats this as `Retryable`
/// without depending on substring matches against the error message.
pub(crate) const SYNTHETIC_SERVER_ERROR_CODE: &str = "transport.server_error";
/// Synthetic code stamped on OpenRouter's "Provider returned error" wrapper
/// (HTTP body or SSE event), so `classify` retries the request instead of
/// dying on the first transient upstream blip. Bounded retries (5 attempts /
/// 60s) cap the cost when the underlying error is permanent.
pub(crate) const UPSTREAM_PROVIDER_ERROR_CODE: &str = "transport.upstream_provider_error";

/// Whether a transport/stream failure should be retried. Retryable variants
/// optionally carry a backoff hint extracted from the upstream response
/// (e.g. `Please try again in 1.25s` or `Retry-After`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Retryability {
    pub(crate) kind: ProviderErrorKind,
    pub(crate) backoff_hint: Option<Duration>,
}

impl Retryability {
    pub(crate) const fn transient(backoff_hint: Option<Duration>) -> Self {
        Self {
            kind: ProviderErrorKind::Transient,
            backoff_hint,
        }
    }

    pub(crate) const fn rate_limited(backoff_hint: Option<Duration>) -> Self {
        Self {
            kind: ProviderErrorKind::RateLimited,
            backoff_hint,
        }
    }

    pub(crate) const fn fatal() -> Self {
        Self {
            kind: ProviderErrorKind::Fatal,
            backoff_hint: None,
        }
    }

    #[cfg(test)]
    pub(crate) fn is_retryable(&self) -> bool {
        self.kind.retryable()
    }

    #[cfg(test)]
    pub(crate) fn backoff_hint(&self) -> Option<Duration> {
        self.backoff_hint
    }
}

/// Single retryability classifier used by both the transport-startup retry
/// gate and the typed provider error kind. Replaces ad-hoc per-call-site
/// substring matches against `"rate limit"` (M19) and the divergent
/// `stream_error_is_retryable` / `provider_error_from_*` predicates that
/// previously could disagree on the same error (AC3.7).
#[must_use]
pub(crate) fn classify(error: &OpenAIError) -> Retryability {
    match error {
        OpenAIError::ApiError(api) => classify_api_error(api),
        OpenAIError::JSONDeserialize(_, content) => parse_openai_stream_error(content)
            .as_ref()
            .map(classify_api_error)
            .unwrap_or_else(Retryability::fatal),
        // Network-layer faults and SSE framing errors are inherently
        // retryable — they are the connection failing, not the API rejecting
        // the request.
        OpenAIError::Reqwest(error) => {
            Retryability::transient(capacity_backoff_hint(&error.to_string()))
        }
        OpenAIError::StreamError(error) => {
            Retryability::transient(capacity_backoff_hint(&error.to_string()))
        }
        OpenAIError::FileSaveError(_)
        | OpenAIError::FileReadError(_)
        | OpenAIError::InvalidArgument(_) => Retryability::fatal(),
    }
}

fn classify_api_error(api: &ApiError) -> Retryability {
    if openai_api_error_is_rate_limit(api) {
        Retryability::rate_limited(openai_api_error_retry_after(api))
    } else if matches!(
        api.code.as_deref(),
        Some(SYNTHETIC_SERVER_ERROR_CODE) | Some(UPSTREAM_PROVIDER_ERROR_CODE)
    ) || api_error_has_transient_tag(api)
    {
        Retryability::transient(api_capacity_backoff_hint(api))
    } else if openai_api_error_is_explicit_fatal(api) {
        Retryability::fatal()
    } else if let Some(backoff_hint) = api_capacity_backoff_hint(api) {
        Retryability::transient(Some(backoff_hint))
    } else {
        Retryability::fatal()
    }
}

/// Classify a provider error payload that is not shaped as an
/// `async_openai` error — e.g. an Anthropic Messages HTTP error body or an
/// in-band SSE `error` event. Shares the tag tables with [`classify`] so
/// retryability decisions cannot drift between provider families. An HTTP
/// `status`, when known, takes precedence over payload tags for 429 and 5xx.
pub(crate) fn classify_error_payload(
    status: Option<u16>,
    error_type: Option<&str>,
    message: &str,
) -> Retryability {
    let api = ApiError {
        message: message.to_owned(),
        r#type: error_type.map(ToOwned::to_owned),
        param: None,
        code: None,
    };
    let by_tags = classify_api_error(&api);
    match status {
        Some(429) => Retryability::rate_limited(by_tags.backoff_hint),
        Some(status) if status >= 500 => Retryability::transient(by_tags.backoff_hint),
        _ => by_tags,
    }
}

/// 5xx-class tags emitted in-band by some providers (Anthropic `api_error` /
/// `overloaded_error`, OpenAI `server_error`). Retryable by nature: the API
/// failed, the request did not get rejected.
fn api_error_has_transient_tag(error: &ApiError) -> bool {
    api_error_tags(error)
        .any(|tag| matches!(tag, "overloaded_error" | "api_error" | "server_error"))
}

fn openai_api_error_is_explicit_fatal(error: &ApiError) -> bool {
    api_error_tags(error).any(|tag| {
        matches!(
            tag,
            "invalid_request_error"
                | "invalid_request"
                | "authentication_error"
                | "permission_error"
                | "insufficient_quota"
                | "model_not_found"
                | "not_found_error"
                | "request_too_large"
                | "context_length_exceeded"
                | "content_policy_violation"
                | "invalid_api_key"
        )
    })
}

fn api_error_tags(error: &ApiError) -> impl Iterator<Item = &str> {
    [error.code.as_deref(), error.r#type.as_deref()]
        .into_iter()
        .flatten()
}

pub(crate) fn parse_openai_http_error(body: &[u8]) -> Option<ApiError> {
    let value = serde_json::from_slice::<Value>(body).ok()?;
    parse_openai_error_value(&value)
}

pub(crate) fn parse_openai_stream_error(data: &str) -> Option<ApiError> {
    let value = serde_json::from_str::<Value>(data).ok()?;
    parse_openai_error_value(&value)
}

pub(crate) fn openai_api_error_is_rate_limit(error: &ApiError) -> bool {
    // Typed-only path (M19, AC3.5). The previous implementation also
    // fell back to a `to_ascii_lowercase().contains("rate limit")` test on
    // the raw message; that substring heuristic was vendor-string-fragile
    // and lived in production hot paths. Detection now requires that the
    // upstream tag the failure with one of the documented codes / types.
    error.code.as_deref() == Some(RATE_LIMIT_CODE)
        || error.r#type.as_deref() == Some("requests")
        || error.r#type.as_deref() == Some("tokens")
        || error.r#type.as_deref() == Some("rate_limit_error")
}

pub(crate) fn openai_api_error_retry_after(error: &ApiError) -> Option<Duration> {
    openai_api_error_is_rate_limit(error)
        .then(|| parse_openai_retry_after_message(&error.message))
        .flatten()
}

fn capacity_backoff_hint(message: &str) -> Option<Duration> {
    let lower = message.to_ascii_lowercase();
    (lower.contains("overloaded") || lower.contains("capacity"))
        .then_some(Duration::from_millis(INFERRED_CAPACITY_BACKOFF_MS))
}

fn api_capacity_backoff_hint(error: &ApiError) -> Option<Duration> {
    capacity_backoff_hint(&error.message)
        .or_else(|| error.r#type.as_deref().and_then(capacity_backoff_hint))
        .or_else(|| error.code.as_deref().and_then(capacity_backoff_hint))
}

pub(crate) fn parse_openai_retry_after_message(message: &str) -> Option<Duration> {
    let start = message.find(RETRY_AFTER_PREFIX)? + RETRY_AFTER_PREFIX.len();
    let token = message[start..].split_whitespace().next()?;
    parse_duration_token(token)
}

fn parse_openai_error_value(value: &Value) -> Option<ApiError> {
    value
        .get("error")
        .and_then(parse_api_error_object)
        .or_else(|| {
            (value.get("type").and_then(Value::as_str) == Some("error"))
                .then(|| parse_api_error_object(value))
                .flatten()
        })
}

fn parse_api_error_object(value: &Value) -> Option<ApiError> {
    let message = value.get("message").and_then(Value::as_str)?.trim();
    if message.is_empty() {
        return None;
    }

    // OpenRouter wraps upstream-provider failures as
    // `{"error": {"message": "Provider returned error", "metadata": {"provider_name": "...", "raw": "..."}}}`.
    // Without lifting `metadata.raw` into the user-facing message, every
    // upstream failure surfaces as the generic top-level string and the
    // real cause is lost.
    let provider_name = value
        .pointer("/metadata/provider_name")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let raw_detail = value
        .pointer("/metadata/raw")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty());
    let is_upstream_wrapper = message.starts_with(OPENROUTER_UPSTREAM_WRAPPER);
    let composed_message = match (provider_name, raw_detail) {
        (Some(provider), Some(raw)) => format!("{message} ({provider}: {raw})"),
        (None, Some(raw)) => format!("{message}: {raw}"),
        (Some(provider), None) => format!("{message} ({provider})"),
        (None, None) => message.to_owned(),
    };
    // OpenRouter's `code` is a numeric HTTP status (e.g. `400`); the existing
    // `as_str()` extraction drops it. We only need the code path for
    // classification, so stamp the synthetic upstream-error code when we
    // recognize the wrapper, regardless of the raw `code` shape.
    let code = value
        .get("code")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .or_else(|| is_upstream_wrapper.then(|| UPSTREAM_PROVIDER_ERROR_CODE.to_owned()));

    Some(ApiError {
        message: composed_message,
        r#type: value
            .get("type")
            .and_then(Value::as_str)
            .filter(|kind| *kind != "error")
            .map(ToOwned::to_owned),
        param: value
            .get("param")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        code,
    })
}

/// Parse a single retry-after token: `250ms`, `1.25s`, `1m`, `2min`, and the
/// combined `6m59.6s` shape OpenAI emits for waits over a minute.
fn parse_duration_token(token: &str) -> Option<Duration> {
    let token = token.trim_end_matches(['.', ',', ';']);
    if let Some(milliseconds) = token.strip_suffix("ms") {
        return duration_from_millis(milliseconds.parse::<f64>().ok()?);
    }
    if let Some(rest) = token.strip_suffix('s') {
        return match rest.split_once('m') {
            Some((minutes, seconds)) => {
                let minutes = minutes.parse::<f64>().ok()?;
                let seconds = if seconds.is_empty() {
                    0.0
                } else {
                    seconds.parse::<f64>().ok()?
                };
                duration_from_secs(minutes * 60.0 + seconds)
            }
            None => duration_from_secs(rest.parse::<f64>().ok()?),
        };
    }
    if let Some(minutes) = token
        .strip_suffix("min")
        .or_else(|| token.strip_suffix('m'))
    {
        return duration_from_secs(minutes.parse::<f64>().ok()? * 60.0);
    }
    None
}

fn duration_from_secs(value: f64) -> Option<Duration> {
    if !value.is_finite() || value < 0.0 {
        return None;
    }
    Some(Duration::from_secs_f64(value))
}

fn duration_from_millis(value: f64) -> Option<Duration> {
    if !value.is_finite() || value < 0.0 {
        return None;
    }
    Some(Duration::from_secs_f64(value / 1_000.0))
}

#[cfg(test)]
mod tests {
    use async_openai::error::StreamError;
    use serde_json::json;

    use super::*;

    fn api_error(code: Option<&str>, kind: Option<&str>, message: &str) -> ApiError {
        ApiError {
            message: message.to_owned(),
            r#type: kind.map(ToOwned::to_owned),
            param: None,
            code: code.map(ToOwned::to_owned),
        }
    }

    #[test]
    fn classify_marks_explicit_rate_limit_retryable_with_hint() {
        let err = OpenAIError::ApiError(api_error(
            Some(RATE_LIMIT_CODE),
            Some("tokens"),
            "Rate limit reached. Please try again in 250ms.",
        ));
        assert_eq!(
            classify(&err),
            Retryability::rate_limited(Some(Duration::from_millis(250)))
        );
    }

    #[test]
    fn classify_marks_synthetic_server_error_retryable_without_hint() {
        let err = OpenAIError::ApiError(api_error(
            Some(SYNTHETIC_SERVER_ERROR_CODE),
            None,
            "internal server error",
        ));
        assert_eq!(classify(&err), Retryability::transient(None));
    }

    #[test]
    fn classify_marks_other_api_errors_fatal() {
        let err = OpenAIError::ApiError(api_error(
            Some("invalid_request_error"),
            Some("invalid_request"),
            "missing required parameter",
        ));
        assert_eq!(classify(&err), Retryability::fatal());
    }

    #[test]
    fn classify_marks_capacity_signals_retryable_with_small_hint() {
        let cases = [
            (None, None, "upstream provider is overloaded"),
            (None, None, "The selected model is currently at CAPACITY."),
            (Some("server_overloaded"), None, "temporarily unavailable"),
        ];

        for (code, kind, message) in cases {
            let err = OpenAIError::ApiError(api_error(code, kind, message));
            assert_eq!(
                classify(&err),
                Retryability::transient(Some(Duration::from_millis(INFERRED_CAPACITY_BACKOFF_MS))),
                "{message}"
            );
        }
    }

    #[test]
    fn classify_keeps_typed_fatal_errors_fatal_even_with_capacity_words() {
        let err = OpenAIError::ApiError(api_error(
            Some("invalid_request_error"),
            Some("invalid_request"),
            "context length is at capacity",
        ));

        assert_eq!(classify(&err), Retryability::fatal());
    }

    #[test]
    fn classify_marks_stream_and_reqwest_failures_retryable() {
        let stream_err = OpenAIError::StreamError(Box::new(StreamError::EventStream(
            "connection reset".to_owned(),
        )));
        assert_eq!(classify(&stream_err), Retryability::transient(None));
    }

    #[test]
    fn classify_marks_stream_capacity_failure_retryable_with_small_hint() {
        let stream_err = OpenAIError::StreamError(Box::new(StreamError::EventStream(
            "provider overloaded".to_owned(),
        )));
        assert_eq!(
            classify(&stream_err),
            Retryability::transient(Some(Duration::from_millis(INFERRED_CAPACITY_BACKOFF_MS)))
        );
    }

    #[test]
    fn classify_decodes_stream_rate_limit_inside_jsondeserialize() {
        let payload = json!({
            "type": "error",
            "error": {
                "type": "tokens",
                "code": RATE_LIMIT_CODE,
                "message": "Rate limit reached. Please try again in 50ms.",
                "param": null
            }
        })
        .to_string();
        let json_err = serde_json::from_str::<u32>("not a number").unwrap_err();
        let err = OpenAIError::JSONDeserialize(json_err, payload);
        assert_eq!(
            classify(&err),
            Retryability::rate_limited(Some(Duration::from_millis(50)))
        );
    }

    #[test]
    fn classify_marks_unknown_jsondeserialize_payload_fatal() {
        let json_err = serde_json::from_str::<u32>("not a number").unwrap_err();
        let err = OpenAIError::JSONDeserialize(json_err, "not json".to_owned());
        assert_eq!(classify(&err), Retryability::fatal());
    }

    #[test]
    fn classify_marks_unknown_api_errors_fatal() {
        let err = OpenAIError::ApiError(api_error(
            Some("unknown_bad_request"),
            Some("unknown_type"),
            "provider rejected this request",
        ));
        assert_eq!(classify(&err), Retryability::fatal());
    }

    #[test]
    fn retryability_helpers_match_pattern() {
        let retryable = Retryability::rate_limited(Some(Duration::from_secs(2)));
        assert!(retryable.is_retryable());
        assert_eq!(retryable.backoff_hint(), Some(Duration::from_secs(2)));

        let fatal = Retryability::fatal();
        assert!(!fatal.is_retryable());
        assert_eq!(fatal.backoff_hint(), None);
    }

    type PayloadCase<'a> = (&'a str, Option<u16>, Option<&'a str>, &'a str, Retryability);

    #[test]
    fn classify_error_payload_covers_provider_statuses_and_tags() {
        let cases: &[PayloadCase] = &[
            (
                "http 429 without tags",
                Some(429),
                None,
                "rate limited",
                Retryability::rate_limited(None),
            ),
            (
                "http 529 overloaded",
                Some(529),
                Some("overloaded_error"),
                "Overloaded",
                Retryability::transient(Some(Duration::from_millis(INFERRED_CAPACITY_BACKOFF_MS))),
            ),
            (
                "http 500 plain",
                Some(500),
                None,
                "internal server error",
                Retryability::transient(None),
            ),
            (
                "in-band rate_limit_error",
                None,
                Some("rate_limit_error"),
                "Number of requests exceeded your rate limit",
                Retryability::rate_limited(None),
            ),
            (
                "in-band overloaded_error",
                None,
                Some("overloaded_error"),
                "Overloaded",
                Retryability::transient(Some(Duration::from_millis(INFERRED_CAPACITY_BACKOFF_MS))),
            ),
            (
                "in-band api_error",
                None,
                Some("api_error"),
                "internal server error",
                Retryability::transient(None),
            ),
            (
                "in-band invalid_request_error",
                None,
                Some("invalid_request_error"),
                "max_tokens too large",
                Retryability::fatal(),
            ),
            (
                "http 404 not_found_error",
                Some(404),
                Some("not_found_error"),
                "model not found",
                Retryability::fatal(),
            ),
            (
                "http 413 request_too_large",
                Some(413),
                Some("request_too_large"),
                "request exceeds the maximum size",
                Retryability::fatal(),
            ),
            (
                "untagged unknown payload",
                None,
                None,
                "something odd",
                Retryability::fatal(),
            ),
        ];

        for (label, status, error_type, message, want) in cases {
            assert_eq!(
                classify_error_payload(*status, *error_type, message),
                *want,
                "{label}"
            );
        }
    }

    #[test]
    fn parse_duration_token_handles_minute_units() {
        let cases: &[(&str, Option<Duration>)] = &[
            ("Please try again in 1m.", Some(Duration::from_secs(60))),
            ("Please try again in 2min.", Some(Duration::from_secs(120))),
            (
                "Please try again in 6m59.6s.",
                Some(Duration::from_secs_f64(419.6)),
            ),
            ("Please try again in 2m30s.", Some(Duration::from_secs(150))),
            (
                "Please try again in 1.25s.",
                Some(Duration::from_secs_f64(1.25)),
            ),
            (
                "Please try again in 250ms.",
                Some(Duration::from_millis(250)),
            ),
            ("Please try again in 3h.", None),
            ("Please try again in soon.", None),
            ("Please try again in -1m.", None),
        ];

        for (message, want) in cases {
            assert_eq!(
                parse_openai_retry_after_message(message),
                *want,
                "{message}"
            );
        }
    }

    #[test]
    fn parses_wrapped_http_error_payload() {
        let body = json!({
            "error": {
                "type": "requests",
                "code": "rate_limit_exceeded",
                "message": "Rate limit reached for gpt-5.4 on requests per min (RPM). Please try again in 1.25s.",
                "param": null
            }
        });

        let error = parse_openai_http_error(body.to_string().as_bytes()).expect("api error");

        assert_eq!(error.r#type.as_deref(), Some("requests"));
        assert_eq!(error.code.as_deref(), Some("rate_limit_exceeded"));
        assert_eq!(
            openai_api_error_retry_after(&error),
            Some(Duration::from_millis(1_250))
        );
    }

    #[test]
    fn parses_wrapped_stream_error_payload() {
        let data = json!({
            "type": "error",
            "error": {
                "type": "tokens",
                "code": "rate_limit_exceeded",
                "message": "Rate limit reached for gpt-5.4 on tokens per min (TPM). Please try again in 75ms.",
                "param": null
            },
            "sequence_number": 2
        });

        let error = parse_openai_stream_error(&data.to_string()).expect("api error");

        assert_eq!(error.r#type.as_deref(), Some("tokens"));
        assert!(openai_api_error_is_rate_limit(&error));
        assert_eq!(
            openai_api_error_retry_after(&error),
            Some(Duration::from_millis(75))
        );
    }

    #[test]
    fn classifies_openrouter_upstream_wrapper_as_retryable() {
        // OpenRouter wraps transient upstream failures (rate limit, timeout,
        // model hiccup) as `{"error": {"message": "Provider returned error",
        // ...}}`. Previously these surfaced with no `code`, fell through to
        // `Fatal`, and killed the main turn loop on a single blip — bypassing
        // the bounded retry budget entirely.
        let body = json!({
            "error": {
                "code": 400,
                "message": "Provider returned error",
                "metadata": {
                    "provider_name": "Z.AI",
                    "raw": "internal server error"
                }
            }
        });
        let api_error = parse_openai_http_error(body.to_string().as_bytes()).expect("api error");
        assert_eq!(
            api_error.code.as_deref(),
            Some(UPSTREAM_PROVIDER_ERROR_CODE)
        );
        let err = OpenAIError::ApiError(api_error);
        assert_eq!(classify(&err), Retryability::transient(None));
    }

    #[test]
    fn classifies_openrouter_upstream_capacity_wrapper_with_small_hint() {
        let body = json!({
            "error": {
                "code": 400,
                "message": "Provider returned error",
                "metadata": {
                    "provider_name": "Upstream",
                    "raw": "model is at capacity"
                }
            }
        });
        let api_error = parse_openai_http_error(body.to_string().as_bytes()).expect("api error");
        let err = OpenAIError::ApiError(api_error);
        assert_eq!(
            classify(&err),
            Retryability::transient(Some(Duration::from_millis(INFERRED_CAPACITY_BACKOFF_MS)))
        );
    }

    #[test]
    fn classifies_openrouter_upstream_wrapper_without_metadata_as_retryable() {
        // The wrapper is sometimes returned without a populated `metadata`
        // object — same upstream reality, just less detail. The classifier
        // must still treat it as retryable, otherwise a single transient
        // OpenRouter blip kills the turn.
        let body = json!({
            "error": { "message": "Provider returned error" }
        });
        let api_error = parse_openai_http_error(body.to_string().as_bytes()).expect("api error");
        assert!(classify(&OpenAIError::ApiError(api_error)).is_retryable());
    }

    #[test]
    fn lifts_openrouter_metadata_raw_into_message() {
        let body = json!({
            "error": {
                "code": 400,
                "message": "Provider returned error",
                "metadata": {
                    "provider_name": "Z.AI",
                    "raw": "The requested model 'z-ai/glm-5.1' does not exist."
                }
            }
        });

        let error = parse_openai_http_error(body.to_string().as_bytes()).expect("api error");

        assert_eq!(
            error.message,
            "Provider returned error (Z.AI: The requested model 'z-ai/glm-5.1' does not exist.)"
        );
    }

    #[test]
    fn lifts_openrouter_metadata_raw_without_provider_name() {
        let body = json!({
            "error": {
                "message": "Provider returned error",
                "metadata": { "raw": "upstream detail" }
            }
        });

        let error = parse_openai_http_error(body.to_string().as_bytes()).expect("api error");

        assert_eq!(error.message, "Provider returned error: upstream detail");
    }

    #[test]
    fn leaves_message_untouched_when_metadata_raw_absent() {
        let body = json!({
            "error": {
                "message": "invalid api key"
            }
        });

        let error = parse_openai_http_error(body.to_string().as_bytes()).expect("api error");

        assert_eq!(error.message, "invalid api key");
    }
}
