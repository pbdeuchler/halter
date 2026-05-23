// pattern: Imperative Shell
//
// Single `browser` tool with action-based dispatch. Mirrors the surface of
// the upstream Python tool (navigate, snapshot, click, type, scroll, back,
// press, screenshot, eval, console, close) but as one entry point keyed by
// `action`, matching halter's `process` / `image` style.
//
// State (the playwright Page + provider session) lives in
// `ToolSessionStore::browser_session(session_id)` so the tool itself stays a
// ZST and concurrent agents working in different halter sessions can each
// hold their own browser without collision.

use std::sync::Arc;

use async_trait::async_trait;
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64_STANDARD;
use halter_protocol::{ToolCapabilities, ToolConcurrency, ToolName, ToolResult, ToolSpec};
use serde_json::{Value, json};
use tracing::debug;

use crate::{Tool, ToolContext};

use super::common::{
    ToolScope, ensure_not_cancelled, optional_bool, optional_string, required_string, resolve_path,
};

pub mod browserbase;
pub mod provider;
pub mod session;

use browserbase::{BrowserbaseConfig, BrowserbaseProvider};
use provider::BrowserProvider;
use session::{BrowserSession, default_goto_options, default_screenshot_options};

const SNAPSHOT_TRUNCATE_THRESHOLD: usize = 8_000;

#[derive(Debug)]
pub struct BrowserTool;

#[async_trait]
impl Tool for BrowserTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("browser"),
            description: include_str!("description.txt").trim().to_owned(),
            input_schema: input_schema(),
            // Browser actions mutate page state — even snapshot/screenshot
            // can race a concurrent click. Serialize them per session.
            concurrency: ToolConcurrency::Exclusive,
            capabilities: ToolCapabilities {
                mutating: true,
                requires_approval: false,
                cancellable: true,
                long_running: true,
            },
            provider_aliases: Default::default(),
        }
    }

    async fn execute(&self, context: ToolContext, input: Value) -> anyhow::Result<ToolResult> {
        let _scope = ToolScope::new(&context, "browser");
        ensure_not_cancelled(&context.cancel)?;
        let action = required_string(&input, "action")?.to_owned();
        debug!(session_id = %context.session_id, action = %action, "browser action dispatch");

        let value = match action.as_str() {
            "navigate" => action_navigate(&context, &input).await?,
            "snapshot" => action_snapshot(&context).await?,
            "click" => action_click(&context, &input).await?,
            "type" => action_type(&context, &input).await?,
            "scroll" => action_scroll(&context, &input).await?,
            "back" => action_back(&context).await?,
            "press" => action_press(&context, &input).await?,
            "screenshot" => action_screenshot(&context, &input).await?,
            "eval" => action_eval(&context, &input).await?,
            "console" => action_console(&context).await?,
            "close" => action_close(&context).await?,
            other => anyhow::bail!("failed to execute browser tool: unknown action '{other}'"),
        };
        Ok(ToolResult::Json { value })
    }
}

fn input_schema() -> Value {
    json!({
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "navigate", "snapshot", "click", "type", "scroll",
                    "back", "press", "screenshot", "eval", "console", "close"
                ],
                "description": "Which browser operation to perform."
            },
            "url": { "type": "string", "description": "URL to load (action=navigate)." },
            "ref": {
                "type": "string",
                "description": "Aria ref id from a prior snapshot, e.g. 's1e3'. Used by click/type/press."
            },
            "selector": {
                "type": "string",
                "description": "Optional Playwright selector (CSS, role=, text=) — alternative to ref."
            },
            "text": { "type": "string", "description": "Text to type (action=type)." },
            "key": { "type": "string", "description": "Key to press, e.g. 'Enter' (action=press)." },
            "direction": {
                "type": "string",
                "enum": ["up", "down"],
                "description": "Scroll direction (action=scroll)."
            },
            "expression": {
                "type": "string",
                "description": "JavaScript expression to evaluate (action=eval)."
            },
            "output_path": {
                "type": "string",
                "description": "Optional path to save screenshot. If omitted, the PNG is returned base64."
            },
            "full_page": {
                "type": "boolean",
                "description": "Capture full scrollable page (action=screenshot, default true)."
            },
            "submit": {
                "type": "boolean",
                "description": "Press Enter after typing (action=type, default false)."
            }
        },
        "required": ["action"]
    })
}

// ----------------------------------------------------------------------------
// Action handlers
// ----------------------------------------------------------------------------

async fn action_navigate(context: &ToolContext, input: &Value) -> anyhow::Result<Value> {
    let url = required_string(input, "url")?.to_owned();
    context.policy.check_network(&url).await?;

    let session_handle = context.tool_sessions.browser_session(&context.session_id);
    let mut guard = session_handle.lock().await;
    let session = ensure_session(&mut guard).await?;
    session
        .page()
        .goto(&url, Some(default_goto_options()))
        .await
        .map_err(|err| anyhow::anyhow!("failed to navigate to {url}: {err}"))?;

    let final_url = session.page().url();
    // Re-check the post-redirect URL — a 302 can land us somewhere unsafe.
    if final_url != url {
        context.policy.check_network(&final_url).await?;
    }
    let title = session
        .page()
        .title()
        .await
        .map_err(|err| anyhow::anyhow!("failed to read page title: {err}"))?;
    session.record_url(final_url.clone());

    let snapshot = take_snapshot(session).await?;

    Ok(json!({
        "url": final_url,
        "title": title,
        "provider": session.provider_name(),
        "features": session.features(),
        "snapshot": truncate_snapshot(&snapshot),
        "snapshot_truncated": snapshot.len() > SNAPSHOT_TRUNCATE_THRESHOLD,
    }))
}

async fn action_snapshot(context: &ToolContext) -> anyhow::Result<Value> {
    let session_handle = context.tool_sessions.browser_session(&context.session_id);
    let mut guard = session_handle.lock().await;
    let session = require_open_session(&mut guard)?;
    let snapshot = take_snapshot(session).await?;
    Ok(json!({
        "snapshot": truncate_snapshot(&snapshot),
        "snapshot_truncated": snapshot.len() > SNAPSHOT_TRUNCATE_THRESHOLD,
    }))
}

async fn action_click(context: &ToolContext, input: &Value) -> anyhow::Result<Value> {
    let target = parse_target(input)?;
    let session_handle = context.tool_sessions.browser_session(&context.session_id);
    let mut guard = session_handle.lock().await;
    let session = require_open_session(&mut guard)?;
    let locator = session.page().locator(&target.selector_string()).await;
    locator
        .click(None)
        .await
        .map_err(|err| anyhow::anyhow!("failed to click {}: {err}", target.display()))?;
    Ok(json!({ "clicked": target.display() }))
}

async fn action_type(context: &ToolContext, input: &Value) -> anyhow::Result<Value> {
    let target = parse_target(input)?;
    let text = required_string(input, "text")?.to_owned();
    let submit = optional_bool(input, "submit")?.unwrap_or(false);
    let session_handle = context.tool_sessions.browser_session(&context.session_id);
    let mut guard = session_handle.lock().await;
    let session = require_open_session(&mut guard)?;
    let locator = session.page().locator(&target.selector_string()).await;
    locator
        .fill(&text, None)
        .await
        .map_err(|err| anyhow::anyhow!("failed to type into {}: {err}", target.display()))?;
    if submit {
        locator
            .press("Enter", None)
            .await
            .map_err(|err| anyhow::anyhow!("failed to submit {}: {err}", target.display()))?;
    }
    Ok(json!({
        "element": target.display(),
        "submitted": submit,
    }))
}

async fn action_scroll(context: &ToolContext, input: &Value) -> anyhow::Result<Value> {
    let direction = required_string(input, "direction")?.to_owned();
    let delta = match direction.as_str() {
        "up" => -500,
        "down" => 500,
        other => anyhow::bail!(
            "failed to execute browser tool: scroll direction must be 'up' or 'down', got '{other}'"
        ),
    };
    let session_handle = context.tool_sessions.browser_session(&context.session_id);
    let mut guard = session_handle.lock().await;
    let session = require_open_session(&mut guard)?;
    session
        .page()
        .evaluate_expression(&format!("window.scrollBy(0, {delta})"))
        .await
        .map_err(|err| anyhow::anyhow!("failed to scroll {direction}: {err}"))?;
    Ok(json!({ "scrolled": direction }))
}

async fn action_back(context: &ToolContext) -> anyhow::Result<Value> {
    let session_handle = context.tool_sessions.browser_session(&context.session_id);
    let mut guard = session_handle.lock().await;
    let session = require_open_session(&mut guard)?;
    session
        .page()
        .go_back(Some(default_goto_options()))
        .await
        .map_err(|err| anyhow::anyhow!("failed to navigate back: {err}"))?;
    let url = session.page().url();
    session.record_url(url.clone());
    Ok(json!({ "url": url }))
}

async fn action_press(context: &ToolContext, input: &Value) -> anyhow::Result<Value> {
    let key = required_string(input, "key")?.to_owned();
    let session_handle = context.tool_sessions.browser_session(&context.session_id);
    let mut guard = session_handle.lock().await;
    let session = require_open_session(&mut guard)?;
    // Page-level keypress: target the body so something always has focus.
    let locator = session.page().locator("body").await;
    locator
        .press(&key, None)
        .await
        .map_err(|err| anyhow::anyhow!("failed to press {key}: {err}"))?;
    Ok(json!({ "pressed": key }))
}

async fn action_screenshot(context: &ToolContext, input: &Value) -> anyhow::Result<Value> {
    let output_path =
        optional_string(input, "output_path").map(|path| resolve_path(&context.working_dir, path));
    let mut options = default_screenshot_options();
    if let Some(full_page) = optional_bool(input, "full_page")? {
        options.full_page = Some(full_page);
    }

    if let Some(path) = output_path.as_ref() {
        // Authorize the destination before writing. We don't yet know the
        // size of the screenshot, so pre-check the parent path first.
        let canonical = context.policy.check_write_path(path).await?;
        let canonical_path = canonical.path().to_path_buf();
        let session_handle = context.tool_sessions.browser_session(&context.session_id);
        let mut guard = session_handle.lock().await;
        let session = require_open_session(&mut guard)?;
        let bytes = session
            .page()
            .screenshot(Some(options))
            .await
            .map_err(|err| anyhow::anyhow!("failed to capture screenshot: {err}"))?;
        let path_locks = context.path_locks.clone();
        let path_for_write = canonical_path.clone();
        let byte_len = bytes.len();
        tokio::task::spawn_blocking(move || {
            let _lock = path_locks.acquire_write(&path_for_write)?;
            canonical.atomic_write_blocking(&bytes)?;
            Ok::<_, anyhow::Error>(())
        })
        .await??;
        Ok(json!({
            "path": canonical_path,
            "bytes": byte_len,
        }))
    } else {
        let session_handle = context.tool_sessions.browser_session(&context.session_id);
        let mut guard = session_handle.lock().await;
        let session = require_open_session(&mut guard)?;
        let bytes = session
            .page()
            .screenshot(Some(options))
            .await
            .map_err(|err| anyhow::anyhow!("failed to capture screenshot: {err}"))?;
        Ok(json!({
            "bytes": bytes.len(),
            "data_base64": BASE64_STANDARD.encode(&bytes),
            "mime_type": "image/png",
        }))
    }
}

async fn action_eval(context: &ToolContext, input: &Value) -> anyhow::Result<Value> {
    let expression = required_string(input, "expression")?.to_owned();
    let session_handle = context.tool_sessions.browser_session(&context.session_id);
    let mut guard = session_handle.lock().await;
    let session = require_open_session(&mut guard)?;
    let raw = session
        .page()
        .evaluate_value(&expression)
        .await
        .map_err(|err| anyhow::anyhow!("failed to evaluate expression: {err}"))?;
    let parsed: Value = serde_json::from_str(&raw).unwrap_or(Value::String(raw));
    Ok(json!({ "result": parsed }))
}

async fn action_console(context: &ToolContext) -> anyhow::Result<Value> {
    let session_handle = context.tool_sessions.browser_session(&context.session_id);
    let mut guard = session_handle.lock().await;
    let session = require_open_session(&mut guard)?;
    let messages: Vec<Value> = session
        .page()
        .console_messages()
        .into_iter()
        .map(|msg| {
            json!({
                "type": msg.type_(),
                "text": msg.text(),
            })
        })
        .collect();
    let errors: Vec<Value> = session
        .page()
        .page_errors()
        .into_iter()
        .map(Value::String)
        .collect();
    let total_messages = messages.len();
    let total_errors = errors.len();
    Ok(json!({
        "console_messages": messages,
        "js_errors": errors,
        "total_messages": total_messages,
        "total_errors": total_errors,
    }))
}

async fn action_close(context: &ToolContext) -> anyhow::Result<Value> {
    let session_handle = context.tool_sessions.browser_session(&context.session_id);
    let mut guard = session_handle.lock().await;
    if let Some(session) = guard.take() {
        session.close().await;
        Ok(json!({ "closed": true }))
    } else {
        Ok(json!({ "closed": false }))
    }
}

// ----------------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------------

async fn ensure_session(slot: &mut Option<BrowserSession>) -> anyhow::Result<&mut BrowserSession> {
    if slot.is_none() {
        let provider = resolve_default_provider()?;
        let session = BrowserSession::open(provider).await?;
        *slot = Some(session);
    }
    Ok(slot.as_mut().expect("session was just inserted"))
}

fn require_open_session(slot: &mut Option<BrowserSession>) -> anyhow::Result<&mut BrowserSession> {
    slot.as_mut().ok_or_else(|| {
        anyhow::anyhow!(
            "failed to execute browser tool: no open browser session — call action='navigate' first"
        )
    })
}

fn resolve_default_provider() -> anyhow::Result<Arc<dyn BrowserProvider>> {
    if let Some(config) = BrowserbaseConfig::from_env() {
        let provider = BrowserbaseProvider::new(config)?;
        return Ok(Arc::new(provider) as Arc<dyn BrowserProvider>);
    }
    anyhow::bail!(
        "failed to execute browser tool: no browser provider configured (set BROWSERBASE_API_KEY \
         + BROWSERBASE_PROJECT_ID)"
    )
}

async fn take_snapshot(session: &BrowserSession) -> anyhow::Result<String> {
    let value = session
        .page()
        .accessibility()
        .snapshot(None)
        .await
        .map_err(|err| anyhow::anyhow!("failed to capture page snapshot: {err}"))?;
    // The accessibility snapshot is returned as a YAML string wrapped in a
    // JSON Value::String. Other Value variants would be a playwright bug;
    // surface them as text rather than crash.
    let text = value
        .as_str()
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| value.to_string());
    Ok(text)
}

fn truncate_snapshot(snapshot: &str) -> String {
    if snapshot.len() <= SNAPSHOT_TRUNCATE_THRESHOLD {
        return snapshot.to_owned();
    }
    let mut buf = String::with_capacity(SNAPSHOT_TRUNCATE_THRESHOLD + 80);
    let mut consumed = 0usize;
    let mut kept_lines = 0usize;
    let mut total_lines = 0usize;
    for line in snapshot.lines() {
        total_lines += 1;
        if consumed + line.len() + 1 < SNAPSHOT_TRUNCATE_THRESHOLD {
            buf.push_str(line);
            buf.push('\n');
            consumed += line.len() + 1;
            kept_lines += 1;
        }
    }
    buf.push_str(&format!(
        "\n[... {} more lines truncated; use ref-targeted actions or 'screenshot' for context]",
        total_lines.saturating_sub(kept_lines)
    ));
    buf
}

/// Element addressing: prefer aria refs (which Playwright assigns natively in
/// snapshot output) and fall back to a raw selector. Refs are *much* more
/// stable across DOM changes than CSS selectors in dynamic apps.
#[derive(Debug)]
struct Target {
    kind: TargetKind,
}

#[derive(Debug)]
enum TargetKind {
    AriaRef(String),
    Selector(String),
}

impl Target {
    fn selector_string(&self) -> String {
        match &self.kind {
            TargetKind::AriaRef(r) => format!("aria-ref={r}"),
            TargetKind::Selector(s) => s.clone(),
        }
    }

    fn display(&self) -> String {
        match &self.kind {
            TargetKind::AriaRef(r) => format!("@{r}"),
            TargetKind::Selector(s) => s.clone(),
        }
    }
}

fn parse_target(input: &Value) -> anyhow::Result<Target> {
    if let Some(reference) = optional_string(input, "ref") {
        let trimmed = reference.trim();
        if trimmed.is_empty() {
            anyhow::bail!("invalid tool input: 'ref' must not be empty");
        }
        // Tolerate the @ prefix the python tool uses (`@e5`) — strip it so
        // the underlying selector is `aria-ref=e5`.
        let cleaned = trimmed.strip_prefix('@').unwrap_or(trimmed).to_owned();
        return Ok(Target {
            kind: TargetKind::AriaRef(cleaned),
        });
    }
    if let Some(selector) = optional_string(input, "selector") {
        let trimmed = selector.trim();
        if trimmed.is_empty() {
            anyhow::bail!("invalid tool input: 'selector' must not be empty");
        }
        return Ok(Target {
            kind: TargetKind::Selector(trimmed.to_owned()),
        });
    }
    anyhow::bail!(
        "invalid tool input: provide either 'ref' (preferred, from snapshot) or 'selector'"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_target_strips_at_prefix_and_prefers_ref() {
        let target = parse_target(&json!({ "ref": "@s1e3" })).expect("ref");
        assert_eq!(target.selector_string(), "aria-ref=s1e3");
        assert_eq!(target.display(), "@s1e3");
    }

    #[test]
    fn parse_target_falls_back_to_selector() {
        let target = parse_target(&json!({ "selector": "button.cta" })).expect("selector");
        assert_eq!(target.selector_string(), "button.cta");
        assert_eq!(target.display(), "button.cta");
    }

    #[test]
    fn parse_target_requires_one_of_ref_or_selector() {
        let err = parse_target(&json!({})).expect_err("must error");
        assert!(err.to_string().contains("ref"));
    }

    #[test]
    fn truncate_snapshot_keeps_full_text_when_short() {
        assert_eq!(truncate_snapshot("hello"), "hello");
    }

    #[test]
    fn truncate_snapshot_appends_marker_when_over_limit() {
        let big = "line\n".repeat(SNAPSHOT_TRUNCATE_THRESHOLD);
        let truncated = truncate_snapshot(&big);
        assert!(truncated.contains("more lines truncated"));
        assert!(truncated.len() < big.len());
    }
}
