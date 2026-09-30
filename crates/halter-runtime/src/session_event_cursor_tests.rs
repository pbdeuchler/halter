// pattern: Integration Tests

use std::sync::atomic::{AtomicUsize, Ordering};

use halter_protocol::{Delivery, ModelId, SubagentEventForwarding};
use halter_session::InMemorySessionStore;

use super::*;

#[derive(Default)]
struct CountingStore {
    inner: InMemorySessionStore,
    reads: AtomicUsize,
}

#[async_trait]
impl SessionStore for CountingStore {
    async fn create_session(&self, session: StoredSession) -> anyhow::Result<()> {
        self.inner.create_session(session).await
    }
    async fn load_session(&self, id: &SessionId) -> anyhow::Result<Option<StoredSession>> {
        self.inner.load_session(id).await
    }
    async fn commit(
        &self,
        id: &SessionId,
        snapshot: Option<Arc<ResourceSnapshot>>,
        expected: Option<u64>,
        state: Option<SessionState>,
        events: Vec<PendingEvent>,
    ) -> anyhow::Result<Vec<SessionEvent>> {
        self.inner
            .commit(id, snapshot, expected, state, events)
            .await
    }
    async fn replay(&self, id: &SessionId) -> anyhow::Result<Vec<SessionEvent>> {
        self.inner.replay(id).await
    }
    async fn replay_after(&self, id: &SessionId, after: u64) -> anyhow::Result<Vec<SessionEvent>> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.inner.replay_after(id, after).await
    }
    async fn list_sessions(&self) -> anyhow::Result<Vec<SessionBlueprint>> {
        self.inner.list_sessions().await
    }
}

fn event(id: &SessionId, sequence: u64) -> SessionEvent {
    PendingEvent::new(
        id.clone(),
        Delivery::Lossless,
        SessionEventPayload::SessionStarted,
    )
    .into_committed(sequence)
}

#[tokio::test]
async fn cursor_ignores_foreign_and_stale_wakes_and_forwarding_but_recovers_gaps_and_close() {
    let store = Arc::new(CountingStore::default());
    let id = SessionId::new();
    let snapshot = Arc::new(ResourceSnapshot::empty());
    store
        .create_session(StoredSession::new(
            SessionBlueprint {
                session_id: id.clone(),
                parent_session_id: None,
                default_model: ModelId::from("default"),
                subagent_model: ModelId::from("default"),
                subagent_event_forwarding: SubagentEventForwarding::Off,
                snapshot_revision: snapshot.revision.clone(),
                working_dir: PathBuf::from("."),
                system_prompt_seed: vec![],
                max_turns: None,
                subagent_depth: 0,
            },
            SessionState::default(),
            snapshot,
        ))
        .await
        .unwrap();
    let services = Arc::new(RuntimeServices {
        sessions: store.clone(),
        event_bus: Arc::new(crate::EventBus::new(2)),
        ..Default::default()
    });
    let (tx, _rx) = mpsc::channel(1);
    let (forwarded, _) = broadcast::channel(8);
    let control = Arc::new(DriverControl {
        id: id.clone(),
        tx,
        stop_tx: mpsc::unbounded_channel().0,
        closed: CancellationToken::new(),
        services: services.clone(),
        forwarded,
        failure: Default::default(),
    });
    let mut events = session_events(control.clone(), 0);
    assert!(futures::poll!(events.next()).is_pending());
    assert_eq!(store.reads.load(Ordering::SeqCst), 1);

    for stale in [event(&SessionId::new(), 1), event(&id, 0)] {
        services.event_bus.publish(stale);
        assert!(futures::poll!(events.next()).is_pending());
        assert_eq!(store.reads.load(Ordering::SeqCst), 1);
    }
    for sequence in 1..=3 {
        control
            .forwarded
            .send(event(&SessionId::from("child"), sequence))
            .unwrap();
        assert_eq!(events.next().await.unwrap().unwrap().session_id.0, "child");
        assert_eq!(store.reads.load(Ordering::SeqCst), 1);
    }
    assert!(futures::poll!(events.next()).is_pending());
    assert_eq!(store.reads.load(Ordering::SeqCst), 1);

    let input = || {
        PendingEvent::new(
            id.clone(),
            Delivery::Lossless,
            SessionEventPayload::SessionStarted,
        )
    };
    store
        .commit(&id, None, Some(0), None, vec![input()])
        .await
        .unwrap();
    services.event_bus.publish(event(&id, 1));
    assert_eq!(events.next().await.unwrap().unwrap().sequence(), 1);
    assert_eq!(store.reads.load(Ordering::SeqCst), 2);

    // A lost wake could have belonged to this session, even when all retained
    // bus events are foreign. Replay must still recover its committed event.
    store
        .commit(&id, None, Some(1), None, vec![input()])
        .await
        .unwrap();
    for sequence in 1..=3 {
        services
            .event_bus
            .publish(event(&SessionId::from("foreign"), sequence));
    }
    assert_eq!(events.next().await.unwrap().unwrap().sequence(), 2);
    assert_eq!(store.reads.load(Ordering::SeqCst), 3);

    // Closing flushes durable tail events even without a bus wake.
    store
        .commit(&id, None, Some(2), None, vec![input()])
        .await
        .unwrap();
    control.closed.cancel();
    assert_eq!(events.next().await.unwrap().unwrap().sequence(), 3);
    assert!(events.next().await.is_none());
}
