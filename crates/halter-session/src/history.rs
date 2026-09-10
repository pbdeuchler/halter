//! Stable window/item identities derived solely from committed event order.
// pattern: Functional Core
use halter_protocol::{AssistantPart, Message, SessionEvent, SessionEventPayload, ToolCallId};

pub struct HistoryItem {
    pub window_id: String,
    pub item_id: String,
    pub role: String,
    pub tools: Vec<String>,
    pub text: String,
}

/// Includes message events and bootstrap messages carried by window rewrites.
/// Failed compaction exchanges remain searchable exactly as recorded.
pub fn history_items(events: &[SessionEvent]) -> Vec<HistoryItem> {
    let mut history = SessionHistory::default();
    history.extend(events);
    history.items
}

/// Incremental index over append-only, sequence-ordered committed events.
#[derive(Default)]
pub struct SessionHistory {
    items: Vec<HistoryItem>,
    window: u64,
    calls: std::collections::BTreeMap<ToolCallId, String>,
}

impl SessionHistory {
    pub fn items(&self) -> &[HistoryItem] {
        &self.items
    }

    pub fn current_window(&self) -> u64 {
        self.window
    }

    /// Append only events after the last batch already indexed.
    pub fn extend(&mut self, events: &[SessionEvent]) {
        for event in events {
            let messages: &[Message] = match &event.payload {
                SessionEventPayload::MessageItem { message } => std::slice::from_ref(message),
                SessionEventPayload::ContextCompacted {
                    effects: Some(effects),
                    ..
                }
                | SessionEventPayload::ContextWindowRolledOver { effects, .. } => {
                    self.window = self.window.saturating_add(1);
                    &effects.messages
                }
                _ => continue,
            };
            for (index, message) in messages.iter().enumerate() {
                let (role, tools, text) = match message {
                    Message::User(user) => ("user", vec![], user.plain_text()),
                    Message::System(system) => ("system", vec![], system.text.clone()),
                    Message::Assistant(assistant) => {
                        let mut tools = Vec::new();
                        let text = assistant
                            .parts
                            .iter()
                            .map(|part| match part {
                                AssistantPart::Text { text } => text.clone(),
                                AssistantPart::Thinking(thinking) => thinking.text.clone(),
                                AssistantPart::ToolCall(call) => {
                                    self.calls.insert(call.id.clone(), call.name.0.clone());
                                    tools.push(call.name.0.clone());
                                    format!("{} {}", call.name, call.arguments)
                                }
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        ("assistant", tools, text)
                    }
                    Message::Tool(result) => (
                        "tool",
                        self.calls
                            .get(&result.call_id)
                            .cloned()
                            .into_iter()
                            .collect(),
                        match &result.content {
                            halter_protocol::ToolResult::Text { text } => text.clone(),
                            halter_protocol::ToolResult::Json { value } => value.to_string(),
                            halter_protocol::ToolResult::Empty => result
                                .error
                                .as_ref()
                                .map_or_else(String::new, |error| error.message.clone()),
                        },
                    ),
                };
                self.items.push(HistoryItem {
                    window_id: format!("window:{}", self.window),
                    item_id: format!("item:{}:{}:{index}", self.window, event.sequence()),
                    role: role.to_owned(),
                    tools,
                    text,
                });
            }
        }
    }
}
