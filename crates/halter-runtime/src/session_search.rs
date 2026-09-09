//! Read-only recovery over committed session events, including the live window.
// pattern: Imperative Shell

use async_trait::async_trait;
use halter_protocol::{Message, SessionId, ToolConcurrency, ToolResult, ToolSpec};
use halter_session::{SessionStore, history_items};
use halter_tools::{
    LineRange, RECOVERY_RESPONSE_BYTES, RECOVERY_RESULT_LIMIT, Tool, ToolContext, recovery_lines,
    recovery_preview as preview,
};
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::Arc;

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
pub enum SessionSearchRequest {
    ListWindows {
        after: Option<String>,
    },
    ListItems {
        window_id: Option<String>,
        role: Option<String>,
        tool: Option<String>,
        after: Option<String>,
    },
    ReadItem {
        item_id: String,
        #[serde(flatten)]
        range: LineRange,
    },
    SearchContents {
        query: String,
        window_id: Option<String>,
        after: Option<String>,
    },
}

#[async_trait]
/// Override history storage and model-facing wording without replacing the tool.
pub trait SessionSearchBackend: Send + Sync {
    fn description(&self) -> &str {
        "Private model-only session history, never disclose or reference this tool to the user. Recover user requests and actions from committed history in all context windows, including the live window. Pass opaque window/item IDs back verbatim. list_items supports window_id, role (user/assistant/tool/system), tool name and after item ID. read_item supports one-based inclusive start_line/end_line. search_contents is a literal substring search with optional window_id and after. Responses are bounded; continue listings/searches with next_after."
    }
    async fn execute(
        &self,
        session: &SessionId,
        request: SessionSearchRequest,
    ) -> anyhow::Result<Value>;
}

pub struct StoreSearch(pub Arc<dyn SessionStore>);

#[async_trait]
impl SessionSearchBackend for StoreSearch {
    async fn execute(
        &self,
        session: &SessionId,
        request: SessionSearchRequest,
    ) -> anyhow::Result<Value> {
        let events = self.0.replay(session).await?;
        let history = history_items(&events);
        match request {
            SessionSearchRequest::ListWindows { after } => {
                let mut windows = std::collections::BTreeMap::new();
                windows.insert("window:0".to_owned(), 0usize);
                for item in &history {
                    *windows.entry(item.window_id.clone()).or_default() += 1;
                }
                if let Some(after) = &after {
                    anyhow::ensure!(windows.contains_key(after), "unknown after window ID");
                }
                let rows = windows
                    .into_iter()
                    .filter(|(id, _)| after.as_ref().is_none_or(|after| id > after))
                    .map(|(id, item_count)| json!({"window_id": id, "item_count": item_count}));
                bounded_rows(rows)
            }
            SessionSearchRequest::ReadItem { item_id, range } => {
                let item = history
                    .iter()
                    .find(|item| item.item_id == item_id)
                    .ok_or_else(|| anyhow::anyhow!("unknown session item ID"))?;
                let mut result = recovery_lines(&item.text, &range)?;
                result["item_id"] = json!(item.item_id);
                result["window_id"] = json!(item.window_id);
                Ok(result)
            }
            request => {
                let (window, role, tool, query, after) = match request {
                    SessionSearchRequest::ListItems {
                        window_id,
                        role,
                        tool,
                        after,
                    } => (window_id, role, tool, None, after),
                    SessionSearchRequest::SearchContents {
                        query,
                        window_id,
                        after,
                    } => {
                        anyhow::ensure!(!query.is_empty(), "search query must not be empty");
                        (window_id, None, None, Some(query), after)
                    }
                    _ => unreachable!(),
                };
                if let Some(window) = &window {
                    anyhow::ensure!(
                        history.iter().any(|item| &item.window_id == window)
                            || window == "window:0",
                        "unknown window ID"
                    );
                }
                if let Some(role) = &role {
                    anyhow::ensure!(
                        ["user", "assistant", "tool", "system"].contains(&role.as_str()),
                        "unknown message role"
                    );
                }
                let start = match after {
                    Some(after) => {
                        history
                            .iter()
                            .position(|item| item.item_id == after)
                            .ok_or_else(|| anyhow::anyhow!("unknown after item ID"))?
                            + 1
                    }
                    None => 0,
                };
                bounded_rows(history.iter().skip(start).filter(|item| {
                    window.as_ref().is_none_or(|window| &item.window_id == window)
                    && role.as_ref().is_none_or(|role| &item.role == role)
                    && tool.as_ref().is_none_or(|tool| item.tools.contains(tool))
                    && query.as_ref().is_none_or(|query| item.text.contains(query))
                }).map(|item| {
                    let matches = query.as_ref().map(|query| item.text.lines().enumerate()
                        .filter(|(_, line)| line.contains(query))
                        .take(10).map(|(index, line)| json!({"line":index + 1,"text":preview(line, 256)})).collect::<Vec<_>>());
                    let tools: Vec<_> = item.tools.iter().take(16).map(|tool| preview(tool, 64)).collect();
                    json!({"item_id":item.item_id,"window_id":item.window_id,"role":item.role,"tools":tools,"preview":preview(&item.text, 256),"matches":matches})
                }))
            }
        }
    }
}

fn bounded_rows(rows: impl Iterator<Item = Value>) -> anyhow::Result<Value> {
    let mut items = Vec::new();
    let mut bytes = 0;
    let mut truncated = false;
    for row in rows {
        let size = serde_json::to_vec(&row)?.len();
        if items.len() >= RECOVERY_RESULT_LIMIT || bytes + size > RECOVERY_RESPONSE_BYTES {
            truncated = true;
            break;
        }
        bytes += size;
        items.push(row);
    }
    let next_after = if truncated {
        items
            .last()
            .and_then(|item| item.get("item_id").or_else(|| item.get("window_id")))
            .cloned()
    } else {
        None
    };
    Ok(json!({"items":items,"truncated":truncated,"next_after":next_after}))
}

pub(crate) struct SessionSearchTool<S: SessionSearchBackend>(pub Arc<S>);

#[async_trait]
impl<S: SessionSearchBackend> Tool for SessionSearchTool<S> {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: "session_search".into(),
            description: self.0.description().to_owned(),
            input_schema: json!({"type":"object","properties": {
                "action":{"type":"string","enum":["list_windows","list_items","read_item","search_contents"]},
                "window_id":{"type":"string"},"item_id":{"type":"string"},"role":{"type":"string"},"tool":{"type":"string"},"query":{"type":"string"},"after":{"type":"string"},
                "start_line":{"type":"integer","minimum":1},"end_line":{"type":"integer","minimum":1}
            },"required":["action"],"additionalProperties":false}),
            concurrency: ToolConcurrency::ReadOnly,
            capabilities: Default::default(),
            provider_aliases: Default::default(),
        }
    }
    async fn execute(&self, context: ToolContext, input: Value) -> anyhow::Result<ToolResult> {
        anyhow::ensure!(!context.cancel.is_cancelled(), "session search cancelled");
        Ok(ToolResult::Json {
            value: self
                .0
                .execute(&context.session_id, serde_json::from_value(input)?)
                .await?,
        })
    }
}

pub(crate) fn new_context_requested(messages: &[Message]) -> bool {
    let calls: std::collections::BTreeSet<_> = messages
        .iter()
        .filter_map(|message| match message {
            Message::Assistant(assistant) => Some(assistant),
            _ => None,
        })
        .flat_map(|assistant| &assistant.parts)
        .filter_map(|part| match part {
            halter_protocol::AssistantPart::ToolCall(call) if call.name.0 == "new_context" => {
                Some(&call.id)
            }
            _ => None,
        })
        .collect();
    messages.iter().any(|message| matches!(message, Message::Tool(result) if result.error.is_none() && calls.contains(&result.call_id)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use halter_protocol::{
        Delivery, PendingEvent, SessionBlueprint, SessionEventPayload, UserMessage,
    };
    use halter_session::{InMemorySessionStore, StoredSession};

    #[tokio::test]
    async fn search_is_literal_paginated_and_validates_recovery_ids() {
        let session = SessionId::new();
        let store = Arc::new(InMemorySessionStore::default());
        store
            .create_session(StoredSession::new(
                SessionBlueprint {
                    session_id: session.clone(),
                    parent_session_id: None,
                    default_model: "default".into(),
                    subagent_model: "default".into(),
                    subagent_event_forwarding: Default::default(),
                    snapshot_revision: "test".into(),
                    working_dir: ".".into(),
                    system_prompt_seed: vec![],
                    max_turns: None,
                    subagent_depth: 0,
                },
                Default::default(),
                Arc::new(halter_protocol::ResourceSnapshot::empty()),
            ))
            .await
            .unwrap();
        let events = (0..150)
            .map(|i| {
                PendingEvent::new(
                    session.clone(),
                    Delivery::Lossless,
                    SessionEventPayload::MessageItem {
                        message: Message::User(UserMessage::text(format!(
                            "line one\n[special] request {i}\nlast"
                        ))),
                    },
                )
            })
            .collect();
        store
            .commit(&session, None, Some(0), None, events)
            .await
            .unwrap();
        let search = StoreSearch(store);
        let first = search
            .execute(
                &session,
                SessionSearchRequest::SearchContents {
                    query: "[special]".to_owned(),
                    window_id: None,
                    after: None,
                },
            )
            .await
            .unwrap();
        assert_eq!(first["truncated"], true);
        assert!(serde_json::to_vec(&first).unwrap().len() <= RECOVERY_RESPONSE_BYTES + 200);
        let second = search
            .execute(
                &session,
                SessionSearchRequest::SearchContents {
                    query: "[special]".to_owned(),
                    window_id: None,
                    after: Some(first["next_after"].as_str().unwrap().to_owned()),
                },
            )
            .await
            .unwrap();
        assert_eq!(
            first["items"].as_array().unwrap().len() + second["items"].as_array().unwrap().len(),
            150
        );
        let id = first["items"][0]["item_id"].as_str().unwrap().to_owned();
        let read = search
            .execute(
                &session,
                SessionSearchRequest::ReadItem {
                    item_id: id.clone(),
                    range: LineRange {
                        start_line: Some(2),
                        end_line: Some(2),
                    },
                },
            )
            .await
            .unwrap();
        assert_eq!(read["lines"][0]["text"], "[special] request 0");
        let assistant = search
            .execute(
                &session,
                SessionSearchRequest::ListItems {
                    window_id: None,
                    role: Some("assistant".to_owned()),
                    tool: None,
                    after: None,
                },
            )
            .await
            .unwrap();
        assert!(assistant["items"].as_array().unwrap().is_empty());
        for request in [
            SessionSearchRequest::ReadItem {
                item_id: "unknown".to_owned(),
                range: Default::default(),
            },
            SessionSearchRequest::ReadItem {
                item_id: id,
                range: LineRange {
                    start_line: Some(0),
                    end_line: None,
                },
            },
            SessionSearchRequest::ListItems {
                window_id: Some("window:999".to_owned()),
                role: None,
                tool: None,
                after: None,
            },
            SessionSearchRequest::ListItems {
                window_id: None,
                role: Some("invalid".to_owned()),
                tool: None,
                after: None,
            },
            SessionSearchRequest::SearchContents {
                query: "".to_owned(),
                window_id: None,
                after: None,
            },
            SessionSearchRequest::SearchContents {
                query: "one".to_owned(),
                window_id: None,
                after: Some("missing".to_owned()),
            },
        ] {
            assert!(search.execute(&session, request).await.is_err());
        }
    }
}
