// pattern: Functional Core
//
// This module holds pure helpers (input parsing, path resolution, ToolScope
// RAII).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::ToolOutput;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio_util::sync::CancellationToken;

use crate::{ToolContext, ToolEventSink, ToolRuntimeEvent};

use super::profiling::{FlatProfileGuard, profile_flat_region};

pub struct ToolScope {
    emit: Arc<dyn ToolEventSink>,
    tool_name: &'static str,
    _profile: FlatProfileGuard,
}

impl ToolScope {
    #[must_use]
    pub fn new(context: &ToolContext, tool_name: &'static str) -> Self {
        context.emit.emit(ToolRuntimeEvent::Started {
            tool_name: tool_name.to_owned(),
        });
        Self {
            emit: context.emit.clone(),
            tool_name,
            _profile: profile_flat_region(tool_name, &context.session_id.0),
        }
    }
}

impl Drop for ToolScope {
    fn drop(&mut self) {
        self.emit.emit(ToolRuntimeEvent::Completed {
            tool_name: self.tool_name.to_owned(),
        });
    }
}

pub fn ensure_not_cancelled(cancel: &CancellationToken) -> anyhow::Result<()> {
    if cancel.is_cancelled() {
        anyhow::bail!("failed to execute tool: cancelled");
    }
    Ok(())
}

pub fn required_string<'a>(input: &'a Value, key: &str) -> anyhow::Result<&'a str> {
    input
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("invalid tool input: missing string field '{key}'"))
}

pub fn optional_string<'a>(input: &'a Value, key: &str) -> Option<&'a str> {
    input.get(key).and_then(Value::as_str)
}

pub fn optional_bool(input: &Value, key: &str) -> anyhow::Result<Option<bool>> {
    match input.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => anyhow::bail!("invalid tool input: field '{key}' must be a boolean"),
    }
}

pub fn optional_u64(input: &Value, key: &str) -> anyhow::Result<Option<u64>> {
    match input.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Number(value)) => value
            .as_u64()
            .map(Some)
            .ok_or_else(|| anyhow::anyhow!("invalid tool input: field '{key}' must be a u64")),
        Some(_) => anyhow::bail!("invalid tool input: field '{key}' must be a u64"),
    }
}

pub fn parse_env_map(value: Option<&Value>) -> anyhow::Result<Option<HashMap<String, String>>> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let object = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("invalid tool input: env must be an object"))?;
    let mut env = HashMap::with_capacity(object.len());
    for (key, value) in object {
        let value = value
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("invalid tool input: env values must be strings"))?;
        env.insert(key.clone(), value.to_owned());
    }
    Ok(Some(env))
}

pub fn resolve_path(working_dir: &Path, path: &str) -> PathBuf {
    let candidate = PathBuf::from(path);
    if candidate.is_absolute() {
        candidate
    } else {
        working_dir.join(candidate)
    }
}

#[must_use]
pub fn hash_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

#[must_use]
pub fn hash_text(text: &str) -> String {
    hash_bytes(text.as_bytes())
}

/// Typed output for the tools whose result is open-ended (a browser `eval`
/// returns whatever the page did). The value is the report; the model reads
/// its fields as `key: value` lines.
#[derive(Debug, Serialize)]
#[serde(transparent)]
pub struct Report(pub Value);

impl ToolOutput for Report {
    /// Fields in key order, whatever order the map holds them in. Strings
    /// are verbatim, never JSON-escaped; multi-line ones come last, each
    /// under its own `key:` line. Null fields are dropped. Nested values are
    /// compact JSON with sorted keys.
    fn to_model_string(&self) -> String {
        let inline = |value: &Value| match value {
            Value::String(text) => text.clone(),
            other => sorted(other).to_string(),
        };
        let Value::Object(fields) = &self.0 else {
            return inline(&self.0);
        };
        let mut fields = fields.iter().collect::<Vec<_>>();
        fields.retain(|(_, value)| !value.is_null());
        let is_block = |value: &Value| value.as_str().is_some_and(|text| text.contains('\n'));
        fields.sort_by_key(|(key, value)| (is_block(value), *key));
        let line = |(key, value): &(&String, &Value)| {
            let separator = if is_block(value) { ":\n" } else { ": " };
            format!("{key}{separator}{}", inline(value))
        };
        fields.iter().map(line).collect::<Vec<_>>().join("\n")
    }
}

/// `value` with every object rebuilt in key order, so its JSON text does not
/// depend on whether `serde_json` preserves insertion order.
fn sorted(value: &Value) -> Value {
    match value {
        Value::Array(items) => items.iter().map(sorted).collect(),
        Value::Object(fields) => {
            let mut fields = fields.iter().collect::<Vec<_>>();
            fields.sort_by_key(|(key, _)| *key);
            let fields = fields
                .into_iter()
                .map(|(key, value)| (key.clone(), sorted(value)));
            Value::Object(fields.collect())
        }
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::{Map, json};

    use super::*;

    #[test]
    fn report_renders_sorted_fields_as_lines_without_escaping() {
        // Inserted out of key order at both levels: with serde_json's
        // `preserve_order` an unsorted rendering would follow insertion.
        let nested = Map::from_iter([("z".to_owned(), json!(1)), ("a".to_owned(), json!(null))]);
        let report = Report(Value::Object(Map::from_iter([
            ("output".to_owned(), json!("é line1\nline2")),
            ("status".to_owned(), json!("running")),
            ("note".to_owned(), json!(null)),
            ("jobs".to_owned(), json!([nested, "b \"q\""])),
            ("command".to_owned(), json!("echo \"hi\"")),
        ])));

        assert_eq!(
            report.to_model_string(),
            "command: echo \"hi\"\njobs: [{\"a\":null,\"z\":1},\"b \\\"q\\\"\"]\n\
             status: running\noutput:\né line1\nline2"
        );
        assert_eq!(
            Report(json!("plain \"text\"")).to_model_string(),
            "plain \"text\""
        );
        assert_eq!(
            Report(json!([{"b": 1, "a": 2}])).to_model_string(),
            "[{\"a\":2,\"b\":1}]"
        );
        assert_eq!(serde_json::to_value(&report).expect("serialize"), report.0);
    }

    #[test]
    fn output_that_cannot_serialize_is_an_error() {
        #[derive(Serialize)]
        struct Unkeyed(std::collections::HashMap<(u8, u8), u8>);
        impl ToolOutput for Unkeyed {
            fn to_model_string(&self) -> String {
                "text".to_owned()
            }
        }

        let error = Unkeyed([((1, 2), 3)].into()).to_tool_result();
        let error = error.expect_err("a tuple is not a JSON object key");
        assert!(
            error.to_string().contains("key must be a string"),
            "{error}"
        );
    }
}
