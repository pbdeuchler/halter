//! Own admission and cleanup for one-shot hook and model-judge sessions.

use anyhow::Context;
use futures::future::BoxFuture;
use halter_protocol::TurnId;
use tokio::task::JoinHandle;

use crate::SessionExecutor;

/// The caller must stop polling submission before finishing or dropping this
/// owner. Cleanup then settles the actual executor before removing tool slots.
pub(crate) struct TemporarySession {
    execution: Option<(SessionExecutor, TurnId)>,
}

impl TemporarySession {
    pub(crate) fn new(session: &SessionExecutor, turn_id: &TurnId) -> Self {
        session
            .services()
            .tool_sessions
            .open_session(session.session_id());
        Self {
            execution: Some((session.clone(), turn_id.clone())),
        }
    }

    pub(crate) async fn finish(mut self) -> anyhow::Result<()> {
        let (session, turn_id) = self.execution.take().expect("temporary session is owned");
        start_cleanup(session, turn_id)
            .await
            .context("failed to join temporary session cleanup")?
    }
}

impl Drop for TemporarySession {
    fn drop(&mut self) {
        if let Some((session, turn_id)) = self.execution.take() {
            start_cleanup(session, turn_id);
        }
    }
}

fn start_cleanup(session: SessionExecutor, turn_id: TurnId) -> JoinHandle<anyhow::Result<()>> {
    // Stop execution promptly even when the caller drops its future. The
    // separately owned cleanup task survives cancellation of finish().
    session.services().turn_registry.abort(&turn_id);
    tokio::spawn(cleanup(session, turn_id))
}

// The boxed Send boundary prevents inference recursing through session-end
// hook dispatch, which can itself run a temporary hook-agent session.
fn cleanup(session: SessionExecutor, turn_id: TurnId) -> BoxFuture<'static, anyhow::Result<()>> {
    Box::pin(async move {
        let tools = &session.services().tool_sessions;
        let session_id = session.session_id();
        tools.request_stop_session(session_id).await;
        let mut first_error = None;
        for outcome in [
            session.force_interrupt_turn(&turn_id).await,
            tools.shutdown_session(session_id).await,
            session.shutdown("temporary_session_closed").await,
        ] {
            if let Err(error) = outcome {
                tracing::warn!(%session_id, %error, "failed to clean up temporary session");
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    })
}

#[cfg(test)]
#[path = "temporary_session_tests.rs"]
mod tests;
