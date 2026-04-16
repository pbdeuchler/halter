// pattern: Functional Core

use base64::Engine;
use halter_protocol::{
    AssistantMessage, AssistantPart, Message, ProviderKind, ProviderRequest, ToolCallId, ToolError,
    ToolName, ToolResult, ToolSpec, UserMessage, UserPart,
};
use sha2::{Digest, Sha256};

const DEFAULT_PROVIDER_ID_MAX_LEN: usize = 64;

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
    bounded_provider_id_with_prefix("", &call_id.0, DEFAULT_PROVIDER_ID_MAX_LEN, "tool")
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
    if prefix.len() + base.len() <= max_len {
        return format!("{prefix}{base}");
    }

    let available = max_len.saturating_sub(prefix.len());
    if available == 0 {
        return short_hash(value).chars().take(max_len).collect();
    }

    let suffix = short_hash(value);
    if suffix.len() >= available {
        return format!("{prefix}{}", &suffix[..available]);
    }

    let head_len = available - suffix.len() - 1;
    format!("{prefix}{}_{}", &base[..head_len], suffix)
}

pub(crate) fn provider_url(base_url: &str, path: &str) -> String {
    format!("{}{}", base_url.trim_end_matches('/'), path)
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
    hex_prefix(&digest, 8)
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

fn hex_prefix(bytes: &[u8], hex_chars: usize) -> String {
    let byte_count = hex_chars.div_ceil(2).min(bytes.len());
    let mut out = String::with_capacity(byte_count * 2);
    for byte in &bytes[..byte_count] {
        use std::fmt::Write as _;
        let _ = write!(out, "{byte:02x}");
    }
    out.truncate(hex_chars);
    out
}
