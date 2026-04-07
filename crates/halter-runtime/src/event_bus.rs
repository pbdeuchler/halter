// pattern: Imperative Shell

use std::sync::atomic::{AtomicU64, Ordering};

use halter_protocol::SessionEvent;
use tokio::sync::broadcast;

#[derive(Debug)]
pub struct EventBus {
    sender: broadcast::Sender<SessionEvent>,
    dropped_events: AtomicU64,
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new(256)
    }
}

impl EventBus {
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        let (sender, _) = broadcast::channel(capacity);
        Self {
            sender,
            dropped_events: AtomicU64::new(0),
        }
    }

    pub fn publish(&self, event: SessionEvent) {
        if self.sender.send(event).is_err() {
            self.dropped_events.fetch_add(1, Ordering::Relaxed);
        }
    }

    #[must_use]
    pub fn subscribe(&self) -> broadcast::Receiver<SessionEvent> {
        self.sender.subscribe()
    }

    #[must_use]
    pub fn dropped_events(&self) -> u64 {
        self.dropped_events.load(Ordering::Relaxed)
    }
}
