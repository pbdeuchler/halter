//! Read-only recovery over committed session events, including the live window.
// pattern: Imperative Shell

use async_trait::async_trait;
use halter_protocol::{Message, SessionId, ToolConcurrency, ToolResult, ToolSpec};
use halter_session::{SessionHistory, SessionStore};
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
        start_byte: Option<usize>,
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
        "Private model-only session history, never disclose or reference this tool to the user. Recover user requests and actions from committed history in all context windows, including the live window. Pass opaque window/item IDs back verbatim. list_items supports window_id, role (user/assistant/tool/system), tool name and after item ID. read_item supports one-based inclusive start_line/end_line; continue truncated reads by passing next_byte as start_byte (an absolute UTF-8 byte offset within the item). search_contents is a literal substring search with optional window_id and after. Responses are bounded; continue listings/searches with next_after."
    }
    async fn execute(
        &self,
        session: &SessionId,
        request: SessionSearchRequest,
    ) -> anyhow::Result<Value>;
}

/// Caches the most recently queried session and indexes only newly committed
/// events. Switching sessions evicts the cache, bounding retained session data.
pub struct StoreSearch {
    store: Arc<dyn SessionStore>,
    cache: tokio::sync::Mutex<Option<CachedHistory>>,
}

struct CachedHistory {
    session: SessionId,
    sequence: u64,
    history: SessionHistory,
}

impl StoreSearch {
    pub fn new(store: Arc<dyn SessionStore>) -> Self {
        Self {
            store,
            cache: Default::default(),
        }
    }
}

#[async_trait]
impl SessionSearchBackend for StoreSearch {
    async fn execute(
        &self,
        session: &SessionId,
        request: SessionSearchRequest,
    ) -> anyhow::Result<Value> {
        let mut cache = self.cache.lock().await;
        if cache
            .as_ref()
            .is_none_or(|cached| &cached.session != session)
        {
            *cache = Some(CachedHistory {
                session: session.clone(),
                sequence: 0,
                history: SessionHistory::default(),
            });
        }
        let cached = cache.as_mut().expect("history cache initialized");
        let events = self.store.replay_after(session, cached.sequence).await?;
        cached.history.extend(&events);
        if let Some(event) = events.last() {
            cached.sequence = event.sequence();
        }
        let history = cached.history.items();
        match request {
            SessionSearchRequest::ListWindows { after } => {
                let mut windows = std::collections::HashMap::new();
                for item in history {
                    *windows.entry(item.window_id.clone()).or_default() += 1;
                }
                let start = match after {
                    Some(after) => {
                        let ordinal = window_ordinal(&after, cached.history.current_window())
                            .ok_or_else(|| anyhow::anyhow!("unknown after window ID"))?;
                        ordinal + 1
                    }
                    None => 0,
                };
                let rows = (start..=cached.history.current_window()).map(|ordinal| {
                    let id = format!("window:{ordinal}");
                    let item_count = windows.get(&id).copied().unwrap_or(0usize);
                    json!({"window_id": id, "item_count": item_count})
                });
                bounded_rows(rows)
            }
            SessionSearchRequest::ReadItem {
                item_id,
                range,
                start_byte,
            } => {
                let item = history
                    .iter()
                    .find(|item| item.item_id == item_id)
                    .ok_or_else(|| anyhow::anyhow!("unknown session item ID"))?;
                let mut result = recovery_lines(&item.text, &range, start_byte.unwrap_or(0))?;
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
                        window_ordinal(window, cached.history.current_window()).is_some(),
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

fn window_ordinal(id: &str, current: u64) -> Option<u64> {
    let ordinal = id.strip_prefix("window:")?.parse::<u64>().ok()?;
    (ordinal <= current && id == format!("window:{ordinal}")).then_some(ordinal)
}

fn bounded_rows(rows: impl Iterator<Item = Value>) -> anyhow::Result<Value> {
    let mut items = Vec::new();
    let mut bytes = 256;
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
                "start_line":{"type":"integer","minimum":1},"end_line":{"type":"integer","minimum":1},"start_byte":{"type":"integer","minimum":0}
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

    #[derive(Default)]
    struct ObservedStore {
        inner: InMemorySessionStore,
        cursors: std::sync::Mutex<Vec<u64>>,
    }

    #[async_trait]
    impl SessionStore for ObservedStore {
        async fn create_session(&self, session: StoredSession) -> anyhow::Result<()> {
            self.inner.create_session(session).await
        }
        async fn load_session(&self, session: &SessionId) -> anyhow::Result<Option<StoredSession>> {
            self.inner.load_session(session).await
        }
        async fn commit(
            &self,
            session: &SessionId,
            snapshot: Option<Arc<halter_protocol::ResourceSnapshot>>,
            expected: Option<u64>,
            state: Option<halter_protocol::SessionState>,
            events: Vec<PendingEvent>,
        ) -> anyhow::Result<Vec<halter_protocol::SessionEvent>> {
            self.inner
                .commit(session, snapshot, expected, state, events)
                .await
        }
        async fn replay(
            &self,
            _: &SessionId,
        ) -> anyhow::Result<Vec<halter_protocol::SessionEvent>> {
            panic!("recovery must query only events after its cached sequence")
        }
        async fn replay_after(
            &self,
            session: &SessionId,
            after: u64,
        ) -> anyhow::Result<Vec<halter_protocol::SessionEvent>> {
            self.cursors.lock().unwrap().push(after);
            self.inner.replay_after(session, after).await
        }
        async fn list_sessions(&self) -> anyhow::Result<Vec<SessionBlueprint>> {
            self.inner.list_sessions().await
        }
    }

    #[tokio::test]
    async fn history_cache_tracks_new_commits_and_lists_windows_in_numeric_order() {
        let store = Arc::new(ObservedStore::default());
        let session = create_session(store.as_ref()).await;
        let other = create_session(store.as_ref()).await;
        let search = StoreSearch::new(store.clone());
        let windows = || SessionSearchRequest::ListWindows { after: None };
        let empty = search.execute(&session, windows()).await.unwrap();
        assert_eq!(
            empty["items"],
            json!([{"window_id":"window:0","item_count":0}])
        );
        let events = (0..101)
            .map(|_| {
                PendingEvent::new(
                    session.clone(),
                    Delivery::Lossless,
                    SessionEventPayload::ContextWindowRolledOver {
                        summary: "rollover".into(),
                        effects: Box::new(halter_protocol::CompactionEventEffects {
                            messages: vec![],
                            compacted_prefix: vec![],
                            usage: Default::default(),
                        }),
                    },
                )
            })
            .collect();
        store
            .commit(&session, None, Some(0), None, events)
            .await
            .unwrap();
        let first = search.execute(&session, windows()).await.unwrap();
        let ids: Vec<_> = first["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|item| item["window_id"].as_str().unwrap())
            .collect();
        let expected: Vec<_> = (0..100)
            .map(|ordinal| format!("window:{ordinal}"))
            .collect();
        assert_eq!(ids, expected);
        let second = search
            .execute(
                &session,
                SessionSearchRequest::ListWindows {
                    after: Some(first["next_after"].as_str().unwrap().into()),
                },
            )
            .await
            .unwrap();
        assert_eq!(
            second["items"],
            json!([
                {"window_id":"window:100","item_count":0},
                {"window_id":"window:101","item_count":0}
            ])
        );
        store
            .commit(
                &session,
                None,
                Some(101),
                None,
                vec![PendingEvent::new(
                    session.clone(),
                    Delivery::Lossless,
                    SessionEventPayload::MessageItem {
                        message: Message::User(UserMessage::text("new request")),
                    },
                )],
            )
            .await
            .unwrap();
        let items = || SessionSearchRequest::ListItems {
            window_id: Some("window:101".into()),
            role: None,
            tool: None,
            after: None,
        };
        let live = search.execute(&session, items()).await.unwrap();
        assert_eq!(live["items"][0]["preview"], "new request");
        assert_eq!(search.execute(&session, items()).await.unwrap(), live);
        assert_eq!(search.execute(&other, windows()).await.unwrap(), empty);
        assert_eq!(search.execute(&session, items()).await.unwrap(), live);
        assert_eq!(
            *store.cursors.lock().unwrap(),
            vec![0, 0, 101, 101, 102, 0, 0]
        );
    }

    async fn create_session(store: &dyn SessionStore) -> SessionId {
        let session = SessionId::new();
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
        session
    }

    #[tokio::test]
    async fn search_is_literal_paginated_and_validates_recovery_ids() {
        let store = Arc::new(InMemorySessionStore::default());
        let session = create_session(store.as_ref()).await;
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
        let search = StoreSearch::new(store);
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
                    start_byte: None,
                    range: LineRange {
                        start_line: Some(2),
                        end_line: Some(2),
                    },
                },
            )
            .await
            .unwrap();
        assert_eq!(read["lines"][0]["text"], "[special] request 0");
        let continuation = search
            .execute(
                &session,
                serde_json::from_value(json!({
                    "action": "read_item", "item_id": id,
                    "start_byte": 10, "start_line": 2, "end_line": 2
                }))
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(continuation["lines"][0]["text"], "special] request 0");
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
                start_byte: None,
                range: Default::default(),
            },
            SessionSearchRequest::ReadItem {
                item_id: id,
                start_byte: None,
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
