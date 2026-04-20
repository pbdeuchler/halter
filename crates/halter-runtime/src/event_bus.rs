// pattern: Imperative Shell

use std::sync::atomic::{AtomicU64, Ordering};

use futures::Stream;
use halter_protocol::{Delivery, PendingEvent, SessionEvent, SessionEventPayload, SessionId};
use tokio::sync::broadcast;
use tokio_stream::{
    StreamExt,
    wrappers::{BroadcastStream, errors::BroadcastStreamRecvError},
};

/// Session identifier used for synthesized bus-level notifications (such as
/// `SessionEventPayload::Lagged`) that do not originate from any single
/// session. Consumers filtering by session id should ignore events carrying
/// this sentinel.
pub const BUS_SESSION_ID: &str = "__bus__";

#[derive(Debug)]
pub struct EventBus {
    sender: broadcast::Sender<SessionEvent>,
    /// Counts publish attempts that found no active subscriber. Exposed via
    /// `dropped_events()` for observability; lag observed at the receiver is
    /// signaled via `SessionEventPayload::Lagged` in the subscribed stream
    /// instead.
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

    /// Subscribe to the bus as a stream. When the underlying broadcast
    /// channel reports lag (subscriber fell behind buffer capacity), the
    /// stream yields a synthesized `SessionEvent` whose payload is
    /// `SessionEventPayload::Lagged { dropped_events: n }` so the gap is
    /// visible to downstream consumers instead of silently swallowed.
    pub fn subscribe(&self) -> impl Stream<Item = SessionEvent> + Send + 'static {
        BroadcastStream::new(self.sender.subscribe()).filter_map(|item| match item {
            Ok(event) => Some(event),
            Err(BroadcastStreamRecvError::Lagged(dropped)) => Some(lagged_event(dropped)),
        })
    }

    /// Raw subscribe for consumers that want to handle `RecvError` directly
    /// (for example, to distinguish `Closed` from `Lagged`). Prefer
    /// `subscribe()` for most call sites.
    #[must_use]
    pub fn subscribe_raw(&self) -> broadcast::Receiver<SessionEvent> {
        self.sender.subscribe()
    }

    #[must_use]
    pub fn dropped_events(&self) -> u64 {
        self.dropped_events.load(Ordering::Relaxed)
    }
}

fn lagged_event(dropped: u64) -> SessionEvent {
    PendingEvent::new(
        SessionId::from(BUS_SESSION_ID),
        Delivery::BestEffort,
        SessionEventPayload::Lagged {
            dropped_events: dropped,
        },
    )
    .into_committed(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use halter_protocol::{Delivery, SessionEventPayload, SessionId};
    use tokio_stream::StreamExt;

    fn test_event(session: &str, sequence: u64) -> SessionEvent {
        PendingEvent::new(
            SessionId::from(session),
            Delivery::Lossless,
            SessionEventPayload::ContextCompacted {
                summary: format!("summary-{sequence}"),
            },
        )
        .into_committed(sequence)
    }

    #[tokio::test]
    async fn publish_without_subscribers_increments_drop_counter() {
        let bus = EventBus::new(4);
        bus.publish(test_event("session-a", 1));
        bus.publish(test_event("session-a", 2));
        assert_eq!(bus.dropped_events(), 2);
    }

    #[tokio::test]
    async fn subscribe_delivers_published_events() {
        let bus = EventBus::new(4);
        let mut stream = bus.subscribe();
        bus.publish(test_event("session-a", 1));
        let received = stream
            .next()
            .await
            .expect("subscriber receives first event");
        assert_eq!(received.sequence(), 1);
        assert!(matches!(
            received.payload,
            SessionEventPayload::ContextCompacted { .. }
        ));
        assert_eq!(bus.dropped_events(), 0);
    }

    #[tokio::test]
    async fn lagged_subscriber_yields_synthetic_lagged_event() {
        let bus = EventBus::new(2);
        let mut stream = bus.subscribe();
        for sequence in 1..=6 {
            bus.publish(test_event("session-a", sequence));
        }
        let mut saw_lagged = false;
        while let Some(event) = stream.next().await {
            if let SessionEventPayload::Lagged { dropped_events } = &event.payload {
                assert!(
                    *dropped_events >= 1,
                    "expected non-zero dropped count, saw {dropped_events}"
                );
                assert_eq!(event.session_id.0, BUS_SESSION_ID);
                saw_lagged = true;
                break;
            }
        }
        assert!(saw_lagged, "expected synthetic Lagged event on overflow");
    }
}
