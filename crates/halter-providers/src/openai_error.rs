// pattern: Functional Core

use std::time::Duration;

use async_openai::error::ApiError;
use serde_json::Value;

const RATE_LIMIT_CODE: &str = "rate_limit_exceeded";
const RETRY_AFTER_PREFIX: &str = "Please try again in ";

pub(crate) fn parse_openai_http_error(body: &[u8]) -> Option<ApiError> {
    let value = serde_json::from_slice::<Value>(body).ok()?;
    parse_openai_error_value(&value)
}

pub(crate) fn parse_openai_stream_error(data: &str) -> Option<ApiError> {
    let value = serde_json::from_str::<Value>(data).ok()?;
    parse_openai_error_value(&value)
}

pub(crate) fn openai_api_error_is_rate_limit(error: &ApiError) -> bool {
    error.code.as_deref() == Some(RATE_LIMIT_CODE)
        || error.r#type.as_deref() == Some("requests")
        || error.r#type.as_deref() == Some("tokens")
        || openai_message_is_rate_limit(&error.message)
}

pub(crate) fn openai_message_is_rate_limit(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    message.contains("rate_limit_exceeded") || message.contains("rate limit")
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
    use serde_json::json;

    use super::*;

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
