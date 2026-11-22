// pattern: Functional Core
//
// This module holds pure helpers (input parsing, path resolution, ToolScope
// RAII).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::ToolOutput;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize};
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

/// Whether a supplied value says nothing: `null`, a blank string, `0`, `[]`
/// or `{}`. Models fill the optional parameters they don't use with these
/// (strict schemas make them send every property), so built-in tools read
/// them as absent. No built-in optional parameter gives one of them a
/// meaning of its own: optional integers default to 0 or start at 1, and
/// `false` is never a placeholder.
fn is_placeholder(value: &Value) -> bool {
    match value {
        Value::Null => true,
        Value::Bool(_) => false,
        Value::Number(number) => number.as_u64() == Some(0),
        Value::String(text) => text.trim().is_empty(),
        Value::Array(items) => items.is_empty(),
        Value::Object(fields) => fields.is_empty(),
    }
}

/// `input[key]`, unless it is missing or a placeholder (`null`, a blank
/// string, `0`, `[]` or `{}`). Every `optional_*` helper reads through this.
pub fn provided<'a>(input: &'a Value, key: &str) -> Option<&'a Value> {
    input.get(key).filter(|value| !is_placeholder(value))
}

/// Serde `deserialize_with` for an optional tool-input field, paired with
/// `#[serde(default)]`: a placeholder (`null`, a blank string, `0`, `[]` or
/// `{}`) deserializes as `T::default()`, as [`provided`] reads it.
///
/// # Errors
///
/// Fails when a value that is not a placeholder does not deserialize as `T`.
pub fn placeholder_as_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: DeserializeOwned + Default,
{
    let value = Value::deserialize(deserializer)?;
    if is_placeholder(&value) {
        return Ok(T::default());
    }
    T::deserialize(value).map_err(serde::de::Error::custom)
}

/// Deserialize a whole tool input. The error keeps serde's account of which
/// field is wrong and why, for the model to correct its call.
pub fn parse_input<T: DeserializeOwned>(input: Value) -> anyhow::Result<T> {
    serde_json::from_value(input).map_err(|error| anyhow::anyhow!("invalid tool input: {error}"))
}

pub fn optional_string<'a>(input: &'a Value, key: &str) -> Option<&'a str> {
    provided(input, key).and_then(Value::as_str)
}

pub fn optional_bool(input: &Value, key: &str) -> anyhow::Result<Option<bool>> {
    match provided(input, key) {
        None => Ok(None),
        Some(Value::Bool(value)) => Ok(Some(*value)),
        Some(_) => anyhow::bail!("invalid tool input: field '{key}' must be a boolean"),
    }
}

pub fn optional_u64(input: &Value, key: &str) -> anyhow::Result<Option<u64>> {
    provided(input, key)
        .map(|value| {
            value
                .as_u64()
                .ok_or_else(|| anyhow::anyhow!("invalid tool input: field '{key}' must be a u64"))
        })
        .transpose()
}

pub fn parse_env_map(value: Option<&Value>) -> anyhow::Result<Option<HashMap<String, String>>> {
    let Some(value) = value.filter(|value| !is_placeholder(value)) else {
        return Ok(None);
    };
    let object = value
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("invalid tool input: env must be an object"))?;
    let mut env = HashMap::with_capacity(object.len());
    for (key, value) in object {
        let value = value
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("invalid tool input: env values must be strings"))?;
        anyhow::ensure!(
            !key.is_empty() && !key.contains(['=', '\0']) && !value.contains('\0'),
            "invalid tool input: invalid environment entry '{key}'"
        );
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
        // Built out of key order at both levels. This pins the layout; the
        // explicit sorts only show under serde_json's `preserve_order`,
        // which no build of this workspace enables.
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
    fn placeholders_read_as_absent_and_other_values_as_given() {
        for (value, placeholder) in [
            (json!(null), true),
            (json!(""), true),
            (json!(" \n"), true),
            (json!(0), true),
            (json!([]), true),
            (json!({}), true),
            (json!(false), false),
            (json!(true), false),
            (json!(1), false),
            (json!(-1), false),
            (json!(0.5), false),
            (json!("x"), false),
            (json!([0]), false),
            (json!({"a": 1}), false),
        ] {
            let input = json!({ "k": value });
            assert_eq!(provided(&input, "k").is_none(), placeholder, "{value}");
            let parsed: Option<Value> = placeholder_as_default(value.clone()).expect("any value");
            assert_eq!(parsed.is_none(), placeholder, "{value}");
        }
        assert_eq!(provided(&json!({}), "k"), None);
    }

    #[test]
    fn optional_helpers_skip_placeholders_and_reject_wrong_types() {
        let input = json!({
            "blank": "", "zero": 0, "null": null, "no": false, "n": 7, "s": "x", "list": []
        });
        for key in ["blank", "zero", "null", "list", "missing"] {
            assert_eq!(optional_string(&input, key), None, "{key}");
            assert_eq!(optional_bool(&input, key).unwrap(), None, "{key}");
            assert_eq!(optional_u64(&input, key).unwrap(), None, "{key}");
        }
        assert_eq!(optional_bool(&input, "no").unwrap(), Some(false));
        assert_eq!(optional_u64(&input, "n").unwrap(), Some(7));
        assert_eq!(optional_string(&input, "s"), Some("x"));
        assert_eq!(optional_string(&input, "n"), None);
        assert!(optional_bool(&input, "s").is_err());
        assert!(optional_u64(&input, "s").is_err());
        let typed: Result<Option<u64>, _> = placeholder_as_default(json!("x"));
        assert!(typed.is_err());
    }

    #[test]
    fn parse_input_names_the_field_serde_rejected() {
        #[derive(Debug, Deserialize)]
        struct Input {
            id: u64,
        }
        assert_eq!(parse_input::<Input>(json!({"id": 3})).unwrap().id, 3);
        let error = parse_input::<Input>(json!({"id": "3"}))
            .unwrap_err()
            .to_string();
        assert!(error.starts_with("invalid tool input: "), "{error}");
        assert!(error.contains("invalid type"), "{error}");
    }

    #[test]
    fn parse_env_map_accepts_string_maps_and_rejects_malformed_entries() {
        assert_eq!(parse_env_map(None).unwrap(), None);
        assert_eq!(parse_env_map(Some(&json!(null))).unwrap(), None);
        assert_eq!(parse_env_map(Some(&json!({}))).unwrap(), None);
        assert_eq!(
            parse_env_map(Some(&json!({"A": "1"}))).unwrap(),
            Some(HashMap::from([("A".to_owned(), "1".to_owned())]))
        );
        for bad in [
            json!("A=1"),
            json!({"A": 1}),
            json!({"": "1"}),
            json!({"A=B": "1"}),
            json!({"A\u{0}": "1"}),
            json!({"A": "1\u{0}"}),
        ] {
            assert!(parse_env_map(Some(&bad)).is_err(), "{bad}");
        }
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
