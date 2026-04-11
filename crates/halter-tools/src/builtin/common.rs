// pattern: Imperative Shell

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tempfile::NamedTempFile;
use tokio_util::sync::CancellationToken;

use crate::{ToolContext, ToolEventSink, ToolRuntimeEvent};

use super::profiling::{ProfileGuard, profile_region};

pub struct ToolScope {
    emit: Arc<dyn ToolEventSink>,
    tool_name: &'static str,
    _profile: ProfileGuard,
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
            _profile: profile_region(tool_name),
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

#[allow(dead_code)]
pub async fn atomic_write(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let path = path.to_path_buf();
    let bytes = bytes.to_vec();
    tokio::task::spawn_blocking(move || atomic_write_blocking(&path, &bytes))
        .await
        .context("failed to join atomic write task")??;
    Ok(())
}

pub(crate) fn atomic_write_blocking(path: &Path, bytes: &[u8]) -> anyhow::Result<()> {
    let parent = path.parent().with_context(|| {
        format!(
            "failed to write '{}': target path has no parent directory",
            path.display()
        )
    })?;
    std::fs::create_dir_all(parent)?;
    let mut temp =
        NamedTempFile::new_in(parent).with_context(|| format!("failed to create temp file"))?;
    std::io::Write::write_all(&mut temp, bytes)?;
    std::io::Write::flush(&mut temp)?;
    temp.persist(path)
        .map(|_| ())
        .map_err(|error| error.error.into())
}
