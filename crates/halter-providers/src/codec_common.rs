// pattern: Functional Core

use base64::Engine;
use halter_protocol::{
    AssistantMessage, AssistantPart, Message, ProviderKind, ProviderRequest, ToolCallId, ToolError,
    ToolName, ToolResult, ToolSpec, UserMessage, UserPart,
};
use sha2::{Digest, Sha256};

/// Max byte length shared by all provider-facing identifier aliases (tool
/// call ids, Responses item ids, etc.). Consolidated to `64`.
pub(crate) const PROVIDER_ID_MAX_LEN: usize = 64;

pub(crate) fn collect_system_text(request: &ProviderRequest) -> Option<String> {
    let mut sections = Vec::new();
    let rendered_prefix = request.prompt.rendered_prefix.trim();
    if !rendered_prefix.is_empty() {
        sections.push(rendered_prefix.to_owned());
    }

    for message in &request.messages {
        if let Message::System(system) = message {
            let text = system.text.trim();
            if !text.is_empty() {
                sections.push(text.to_owned());
            }
        }
    }

    if sections.is_empty() {
        None
    } else {
        Some(sections.join("\n\n"))
    }
}

pub(crate) fn assistant_text(message: &AssistantMessage) -> String {
    message
        .parts
        .iter()
        .filter_map(|part| match part {
            AssistantPart::Text { text } if !text.is_empty() => Some(text.as_str()),
            AssistantPart::Text { .. }
            | AssistantPart::Thinking(_)
            | AssistantPart::ToolCall(_) => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub(crate) fn has_user_media(message: &UserMessage) -> bool {
    message
        .parts
        .iter()
        .any(|part| !matches!(part, UserPart::Text { .. }))
}

pub(crate) fn user_text(message: &UserMessage) -> String {
    message
        .parts
        .iter()
        .filter_map(|part| match part {
            UserPart::Text { text } if !text.is_empty() => Some(text.as_str()),
            UserPart::Text { .. } | UserPart::Image { .. } | UserPart::Document { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("\n")
}

pub(crate) fn tool_name_for_provider(
    name: &ToolName,
    tool_specs: &[ToolSpec],
    provider_kind: ProviderKind,
) -> String {
    tool_specs
        .iter()
        .find(|spec| spec.name == *name)
        .and_then(|spec| spec.provider_aliases.get(&provider_kind))
        .map(|alias| alias.0.clone())
        .unwrap_or_else(|| name.0.clone())
}

pub(crate) fn canonical_tool_name(
    provided_name: &str,
    tool_specs: &[ToolSpec],
    provider_kind: ProviderKind,
) -> ToolName {
    if let Some(spec) = tool_specs.iter().find(|spec| {
        spec.provider_aliases
            .get(&provider_kind)
            .is_some_and(|alias| alias.0 == provided_name)
    }) {
        return spec.name.clone();
    }

    tool_specs
        .iter()
        .find(|spec| spec.name.0 == provided_name)
        .map(|spec| spec.name.clone())
        .unwrap_or_else(|| ToolName::from(provided_name.to_owned()))
}

pub(crate) fn tool_result_text(result: &ToolResult, error: &Option<ToolError>) -> String {
    match (result, error) {
        (_, Some(error)) if matches!(result, ToolResult::Empty) => error.message.clone(),
        (ToolResult::Text { text }, Some(error)) => {
            format!("{text}\n\nerror: {}", error.message)
        }
        (ToolResult::Json { value }, Some(error)) => serde_json::json!({
            "result": value,
            "error": error.message,
        })
        .to_string(),
        (ToolResult::Empty, None) => String::new(),
        (ToolResult::Text { text }, None) => text.clone(),
        (ToolResult::Json { value }, None) => value.to_string(),
        (ToolResult::Empty, Some(error)) => error.message.clone(),
    }
}

pub(crate) fn normalized_tool_call_id(call_id: &ToolCallId) -> String {
    bounded_provider_id_with_prefix("", &call_id.0, PROVIDER_ID_MAX_LEN, "tool")
}

pub(crate) fn bounded_provider_id(value: &str, max_len: usize, empty_prefix: &str) -> String {
    bounded_provider_id_with_prefix("", value, max_len, empty_prefix)
}

pub(crate) fn bounded_provider_id_with_prefix(
    prefix: &str,
    value: &str,
    max_len: usize,
    empty_prefix: &str,
) -> String {
    let sanitized = sanitize_provider_id(value);
    let base = if sanitized.is_empty() {
        format!("{empty_prefix}_{}", short_hash(value))
    } else {
        sanitized
    };
    // `base` is ASCII for any non-empty `value`, because
    // `sanitize_provider_id` maps every non alphanumeric/-/_ rune to `_`.
    // It may contain multi-byte runes only when `value` is empty and
    // `empty_prefix` is not ASCII. Truncation below uses a
    // `floor_char_boundary`-style clamp so the byte slices are
    // char-boundary-safe for any UTF-8 input. The `debug_assert!` on the
    // suffix (always hex / ASCII) remains as a defense-in-depth sentinel,
    // not as the source of release-mode panic safety.
    if prefix.len() + base.len() <= max_len {
        return format!("{prefix}{base}");
    }

    let available = max_len.saturating_sub(prefix.len());
    if available == 0 {
        return short_hash(value).chars().take(max_len).collect();
    }

    let suffix = short_hash(value);
    debug_assert!(suffix.is_ascii(), "provider id suffix must be ASCII");
    if suffix.len() >= available {
        return format!(
            "{prefix}{}",
            &suffix[..floor_char_boundary(&suffix, available)]
        );
    }

    let head_len = available - suffix.len() - 1;
    let head_end = floor_char_boundary(&base, head_len);
    format!("{prefix}{}_{}", &base[..head_end], suffix)
}

pub(crate) fn data_url(media_type: &str, data: &[u8]) -> String {
    let encoded = base64::engine::general_purpose::STANDARD.encode(data);
    format!("data:{media_type};base64,{encoded}")
}

pub(crate) fn document_filename(media_type: &str) -> String {
    match media_type {
        "application/pdf" => "upload.pdf".to_owned(),
        "text/plain" => "upload.txt".to_owned(),
        _ => "upload.bin".to_owned(),
    }
}

fn short_hash(value: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(value.as_bytes());
    let digest = hasher.finalize();
    let mut encoded = hex::encode(digest);
    encoded.truncate(8);
    encoded
}

fn sanitize_provider_id(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

/// Walks back from `idx` to the nearest UTF-8 char boundary at or before it.
fn floor_char_boundary(s: &str, idx: usize) -> usize {
    let mut boundary = idx.min(s.len());
    while boundary > 0 && !s.is_char_boundary(boundary) {
        boundary -= 1;
    }
    boundary
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_ascii_value_with_prefix_fits_unchanged() {
        let value = "hello_world";
        let result = bounded_provider_id_with_prefix("pre_", value, 64, "tool");
        assert!(result.starts_with("pre_hello_world"));
        assert_eq!(result.len(), "pre_hello_world".len());
    }

    #[test]
    fn long_ascii_value_truncated_to_exact_max_len() {
        let max_len = 16;
        let value = "a".repeat(64);
        let result = bounded_provider_id_with_prefix("", &value, max_len, "tool");
        assert_eq!(
            result.len(),
            max_len,
            "result should be exactly {max_len} bytes"
        );
        let expected_head_len = max_len - short_hash(&value).len() - "_".len();
        let expected = format!("{}_{}", "a".repeat(expected_head_len), short_hash(&value));
        assert_eq!(result, expected);
    }

    #[test]
    fn empty_ascii_value_is_truncated_to_max_len() {
        let value = "";
        let max_len = 10;
        let result = bounded_provider_id_with_prefix("", value, max_len, "tool");
        // base "tool_<hash>" is longer than max_len, so it is truncated to
        // {head "t"}_{suffix}, giving exactly max_len bytes.
        assert!(result.starts_with("t_"));
        assert_eq!(result.len(), max_len);
    }

    #[test]
    fn prefix_longer_than_max_len_returns_truncated_hash() {
        let value = "ignored";
        let result = bounded_provider_id_with_prefix("verylongprefix", value, 4, "tool");
        assert_eq!(result.len(), 4);
        // Should be the first 4 chars of the hex hash, all ASCII hex digits.
        assert!(result.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn sanitize_provider_id_strips_non_ascii_to_underscores() {
        let input = "héllo-世界_";
        let out = sanitize_provider_id(input);
        assert!(out.is_ascii());
        assert_eq!(out, "h_llo-___");
    }

    #[test]
    fn non_ascii_empty_prefix_suffix_path_is_safe() {
        // `empty_prefix` is not sanitized, so empty `value` causes `base` to
        // contain a multi-byte rune. With a small `max_len` we hit the suffix
        // branch and must never slice on a char boundary.
        let empty_prefix = "é";
        let value = "";
        let max_len = 2;
        let result = bounded_provider_id_with_prefix("", value, max_len, empty_prefix);
        assert!(
            std::str::from_utf8(result.as_bytes()).is_ok(),
            "result must be valid UTF-8"
        );
        assert!(result.len() <= max_len, "{} <= {}", result.len(), max_len);
    }

    #[test]
    fn non_ascii_empty_prefix_head_path_is_safe() {
        // Same non-ASCII `empty_prefix`, but with a max_len large enough to
        // reach the {head}_{suffix} branch. The head slice must clamp to a
        // char boundary (the 2-byte "é" would be split otherwise).
        let empty_prefix = "é";
        let value = "";
        let max_len = 10;
        let result = bounded_provider_id_with_prefix("", value, max_len, empty_prefix);
        assert!(result.is_char_boundary(0));
        assert!(result.is_char_boundary(result.len()));
        assert!(result.len() <= max_len);
    }

    #[test]
    fn suffix_truncation_path_is_char_boundary_safe() {
        // Force the suffix branch by making the combined prefix+base exceed
        // max_len while `available` is at most 8 (suffix length).
        let value = "123456789";
        let max_len = 12;
        let prefix = "pre_";
        let result = bounded_provider_id_with_prefix(prefix, value, max_len, "tool");
        assert_eq!(result.len(), max_len);
        assert!(result.is_char_boundary(0));
        assert!(result.is_char_boundary(result.len()));
    }
}
