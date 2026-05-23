// pattern: Functional Core
//
// This module holds pure helpers (input parsing, path resolution, ToolScope
// RAII).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
