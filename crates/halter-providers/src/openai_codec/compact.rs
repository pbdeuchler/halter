// pattern: Functional Core

use halter_protocol::{ApiKind, ProviderCompactionRequest, ProviderCompactionResponse};
use serde_json::{Map, Value, json};

use super::{
    append_responses_messages, decode_openai_usage, encode_responses_developer_message,
    validate_responses_input_item_ids,
};

const COMPACTED_CONTEXT_PREFIX: &str = "[Compacted context]\n\n";

pub(crate) fn encode_responses_compact_request(
    request: &ProviderCompactionRequest,
) -> anyhow::Result<Value> {
    encode_compact_request_body(request, "openai", false)
}

pub(crate) fn encode_openrouter_compact_request(
    request: &ProviderCompactionRequest,
) -> anyhow::Result<Value> {
    encode_compact_request_body(request, "openrouter", true)
}

fn encode_compact_request_body(
    request: &ProviderCompactionRequest,
    provider_label: &str,
    include_stream_store: bool,
) -> anyhow::Result<Value> {
    if request.model.api_kind() != ApiKind::OpenAiResponses {
        anyhow::bail!(
            "failed to encode {provider_label} compaction request: unsupported api kind"
        );
    }

    let input = encode_responses_compact_input(request)?;
    validate_responses_input_item_ids(&input)?;
    let mut body = Map::new();
    body.insert("model".to_owned(), json!(request.model.model));
    body.insert("input".to_owned(), Value::Array(input));
    body.insert("instructions".to_owned(), json!(request.instructions));
    if include_stream_store {
        body.insert("stream".to_owned(), json!(false));
        body.insert("store".to_owned(), json!(false));
    }
    Ok(Value::Object(body))
}

pub(crate) fn decode_responses_compact_response(
    response: &Value,
) -> anyhow::Result<ProviderCompactionResponse> {
    let output = compact_output_array(response, "openai")?.clone();
    Ok(ProviderCompactionResponse {
        output,
        usage: decode_openai_usage(response),
    })
}

pub(crate) fn decode_openrouter_compact_response(
    response: &Value,
) -> anyhow::Result<ProviderCompactionResponse> {
    let output_ref = compact_output_array(response, "openrouter")?;
    let compacted_text = openrouter_compaction_output_text(output_ref);
    let output = if compacted_text.is_empty() {
        Vec::new()
    } else {
        vec![encode_responses_developer_message(&format!(
            "{COMPACTED_CONTEXT_PREFIX}{compacted_text}"
        ))]
    };

    Ok(ProviderCompactionResponse {
        output,
        usage: decode_openai_usage(response),
    })
}

fn compact_output_array<'a>(
    response: &'a Value,
    provider_label: &str,
) -> anyhow::Result<&'a Vec<Value>> {
    response
        .get("output")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "failed to decode {provider_label} compaction response: missing output array"
            )
        })
}

fn encode_responses_compact_input(
    request: &ProviderCompactionRequest,
) -> anyhow::Result<Vec<Value>> {
    let mut input = request.compacted_prefix.clone();
    append_responses_messages(
        &mut input,
        &request.messages,
        &request.tools,
        request.model.provider_kind,
    )?;
    Ok(input)
}

fn openrouter_compaction_output_text(output: &[Value]) -> String {
    output
        .iter()
        .filter_map(openrouter_compaction_message_text)
        .filter(|text| !text.is_empty())
        .collect::<Vec<_>>()
        .join("\n\n")
}

fn openrouter_compaction_message_text(item: &Value) -> Option<String> {
    if item.get("type").and_then(Value::as_str) != Some("message") {
        return None;
    }

    let text = item
        .get("content")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(openrouter_compaction_content_text)
        .collect::<Vec<_>>()
        .join("");
    if text.is_empty() { None } else { Some(text) }
}

fn openrouter_compaction_content_text(content: &Value) -> Option<&str> {
    content
        .get("text")
        .and_then(Value::as_str)
        .or_else(|| content.get("output_text").and_then(Value::as_str))
}
