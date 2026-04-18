// pattern: Functional Core

use std::time::Duration;

use async_openai::error::{ApiError, OpenAIError};
use serde_json::Value;

const RATE_LIMIT_CODE: &str = "rate_limit_exceeded";
const RETRY_AFTER_PREFIX: &str = "Please try again in ";
/// Synthetic code stamped by the transport layer when an HTTP 5xx response
/// has no parseable JSON error body. `classify` treats this as `Retryable`
/// without depending on substring matches against the error message.
pub(crate) const SYNTHETIC_SERVER_ERROR_CODE: &str = "transport.server_error";

/// Whether a transport/stream failure should be retried. Retryable variants
/// optionally carry a backoff hint extracted from the upstream response
/// (e.g. `Please try again in 1.25s` or `Retry-After`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Retryability {
    Retryable { backoff_hint: Option<Duration> },
    Fatal,
}

impl Retryability {
    pub(crate) fn is_retryable(&self) -> bool {
        matches!(self, Self::Retryable { .. })
    }

    pub(crate) fn backoff_hint(&self) -> Option<Duration> {
        match self {
            Self::Retryable { backoff_hint } => *backoff_hint,
            Self::Fatal => None,
        }
    }
}

/// Single retryability classifier used by both the transport-startup retry
/// gate and the `ProviderError.retryable` flag. Replaces ad-hoc per-call-site
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
            .unwrap_or(Retryability::Fatal),
        // Network-layer faults and SSE framing errors are inherently
        // retryable — they are the connection failing, not the API rejecting
        // the request.
        OpenAIError::Reqwest(_) | OpenAIError::StreamError(_) => {
            Retryability::Retryable { backoff_hint: None }
        }
        OpenAIError::FileSaveError(_)
        | OpenAIError::FileReadError(_)
        | OpenAIError::InvalidArgument(_) => Retryability::Fatal,
    }
}

fn classify_api_error(api: &ApiError) -> Retryability {
    if openai_api_error_is_rate_limit(api) {
        Retryability::Retryable {
            backoff_hint: openai_api_error_retry_after(api),
        }
    } else if api.code.as_deref() == Some(SYNTHETIC_SERVER_ERROR_CODE) {
        Retryability::Retryable { backoff_hint: None }
    } else {
        Retryability::Fatal
    }
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
}

pub(crate) fn openai_api_error_retry_after(error: &ApiError) -> Option<Duration> {
    openai_api_error_is_rate_limit(error)
        .then(|| parse_openai_retry_after_message(&error.message))
        .flatten()
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

    Some(ApiError {
        message: message.to_owned(),
        r#type: value
            .get("type")
            .and_then(Value::as_str)
            .filter(|kind| *kind != "error")
            .map(ToOwned::to_owned),
        param: value
            .get("param")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        code: value
            .get("code")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
    })
}

fn parse_duration_token(token: &str) -> Option<Duration> {
    let token = token.trim_end_matches(['.', ',', ';']);
    if let Some(milliseconds) = token.strip_suffix("ms") {
        let value = milliseconds.parse::<f64>().ok()?;
        return duration_from_millis(value);
    }
    if let Some(seconds) = token.strip_suffix('s') {
        let value = seconds.parse::<f64>().ok()?;
        return duration_from_secs(value);
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
            Retryability::Retryable {
                backoff_hint: Some(Duration::from_millis(250)),
            }
        );
    }

    #[test]
    fn classify_marks_synthetic_server_error_retryable_without_hint() {
        let err = OpenAIError::ApiError(api_error(
            Some(SYNTHETIC_SERVER_ERROR_CODE),
            None,
            "internal server error",
        ));
        assert_eq!(
            classify(&err),
            Retryability::Retryable { backoff_hint: None }
        );
    }

    #[test]
    fn classify_marks_other_api_errors_fatal() {
        let err = OpenAIError::ApiError(api_error(
            Some("invalid_request_error"),
            Some("invalid_request"),
            "missing required parameter",
        ));
        assert_eq!(classify(&err), Retryability::Fatal);
    }

    #[test]
    fn classify_marks_stream_and_reqwest_failures_retryable() {
        let stream_err = OpenAIError::StreamError(Box::new(StreamError::EventStream(
            "connection reset".to_owned(),
        )));
        assert_eq!(
            classify(&stream_err),
            Retryability::Retryable { backoff_hint: None }
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
            Retryability::Retryable {
                backoff_hint: Some(Duration::from_millis(50)),
            }
        );
    }

    #[test]
    fn classify_marks_unknown_jsondeserialize_payload_fatal() {
        let json_err = serde_json::from_str::<u32>("not a number").unwrap_err();
        let err = OpenAIError::JSONDeserialize(json_err, "not json".to_owned());
        assert_eq!(classify(&err), Retryability::Fatal);
    }

    #[test]
    fn retryability_helpers_match_pattern() {
        let retryable = Retryability::Retryable {
            backoff_hint: Some(Duration::from_secs(2)),
        };
        assert!(retryable.is_retryable());
        assert_eq!(retryable.backoff_hint(), Some(Duration::from_secs(2)));

        let fatal = Retryability::Fatal;
        assert!(!fatal.is_retryable());
        assert_eq!(fatal.backoff_hint(), None);
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
}
