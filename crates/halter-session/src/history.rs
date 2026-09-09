//! Stable window/item identities derived solely from committed event order.
// pattern: Functional Core
use halter_protocol::{AssistantPart, Message, SessionEvent, SessionEventPayload};

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
    let mut window = 0u64;
    let mut items = Vec::new();
    let mut calls = std::collections::BTreeMap::new();
    for event in events {
        let messages: &[Message] = match &event.payload {
            SessionEventPayload::MessageItem { message } => std::slice::from_ref(message),
            SessionEventPayload::ContextCompacted {
                effects: Some(effects),
                ..
            }
            | SessionEventPayload::ContextWindowRolledOver { effects, .. } => {
                window = window.saturating_add(1);
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
                                calls.insert(call.id.clone(), call.name.0.clone());
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
                    calls.get(&result.call_id).cloned().into_iter().collect(),
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
            items.push(HistoryItem {
                window_id: format!("window:{window}"),
                item_id: format!("item:{window}:{}:{index}", event.sequence()),
                role: role.to_owned(),
                tools,
                text,
            });
        }
    }
    items
}
