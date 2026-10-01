use std::sync::Arc;
use std::task::Poll;
use std::time::Duration;

use halter_protocol::{SessionEventPayload, SessionState, TurnId};
use halter_providers::FakeProvider;

use super::TemporarySession;
use crate::SessionInit;
use crate::session::create_session_seeded;
use crate::session_driver_tests::services;

#[tokio::test]
async fn abandoned_temporary_owner_cleans_up_before_or_during_finish() {
    for abandon_finish in [false, true] {
        let root = tempfile::tempdir().unwrap();
        let services = services(Arc::new(FakeProvider::default()));
        let session = create_session_seeded(
            services.clone(),
            SessionInit {
                working_dir: root.path().to_owned(),
                ..Default::default()
            },
            SessionState::default(),
            services.resources.snapshot(),
        )
        .await
        .unwrap();
        let owner = TemporarySession::new(&session, &TurnId::new());
        let _ = services.tool_sessions.shell_session(session.session_id());
        assert!(
            services
                .tool_sessions
                .has_process_state(session.session_id())
        );

        // Hold cleanup at the session lease to abandon finish while its
        // independently owned task is still pending, without timing a sleep.
        services.session_leases.acquire(session.session_id()).await;
        let mut events = services.event_bus.subscribe_raw();
        if abandon_finish {
            let mut finish = Box::pin(owner.finish());
            assert!(matches!(futures::poll!(&mut finish), Poll::Pending));
            drop(finish);
        } else {
            drop(owner);
        }
        services
            .session_leases
            .release(session.session_id(), |_| async { Ok(()) })
            .await
            .unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let event = events.recv().await.unwrap();
                if event.session_id == *session.session_id()
                    && matches!(event.payload, SessionEventPayload::SessionShutdownComplete)
                {
                    break;
                }
            }
        })
        .await
        .expect("cleanup survives abandonment and completes session shutdown");
        assert!(
            !services
                .tool_sessions
                .has_process_state(session.session_id())
        );
        assert_eq!(services.turn_registry.in_flight_count(), 0);
        // A stale getter must neither readmit the session nor leave a slot.
        let _ = services.tool_sessions.shell_session(session.session_id());
        assert!(
            !services
                .tool_sessions
                .has_process_state(session.session_id())
        );
    }
}
