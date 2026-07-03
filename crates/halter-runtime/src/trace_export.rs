//! Derive a trace serialization from the session store's event log.
//!
//! The live [`crate::TraceRecorder`] tails events into per-session files as
//! they commit; this module produces the equivalent committed-line view on
//! demand by serializing `SessionStore::replay`. The store's append-only log
//! is the source of truth, so an export is available for every persisted
//! session — whether or not a `traces_dir` was configured while it ran — and
//! contains exactly the committed `SessionEvent` lines (no `pending_event`
//! preview lines, which exist only for live tailing).

use anyhow::Context;
use chrono::Utc;
use halter_protocol::{SessionBlueprint, SessionId};
use halter_session::SessionStore;
use serde_json::json;

use crate::trace_recorder::TRACE_FORMAT_VERSION;

/// Serialize the full trace of `session_id` and every subagent session
/// descended from it, as JSONL matching the live trace-file format: a
/// `trace_header` line, the root session's committed events in sequence
/// order, then per descendant (depth-first, children ordered by session id)
/// a `subagent_header` line followed by that session's events.
///
/// Unlike live trace files — which interleave subagent events with the
/// parent's in commit-arrival order — the export groups each session's
/// events contiguously; per-session sequence order is identical. Fails when
/// the root session is unknown.
pub async fn export_session_trace(
    store: &dyn SessionStore,
    session_id: &SessionId,
) -> anyhow::Result<String> {
    let root = store
        .load_session(session_id)
        .await?
        .with_context(|| format!("failed to export trace: unknown session '{}'", session_id.0))?;

    let blueprints = store.list_sessions().await?;
    let mut children_by_parent: std::collections::BTreeMap<&str, Vec<&SessionBlueprint>> =
        std::collections::BTreeMap::new();
    for blueprint in &blueprints {
        if let Some(parent) = &blueprint.parent_session_id {
            children_by_parent
                .entry(parent.0.as_str())
                .or_default()
                .push(blueprint);
        }
    }
    for children in children_by_parent.values_mut() {
        children.sort_by(|a, b| a.session_id.cmp(&b.session_id));
    }

    let mut lines = Vec::new();
    lines.push(
        serde_json::to_string(&json!({
            "kind": "trace_header",
            "trace_version": TRACE_FORMAT_VERSION,
            "session_id": session_id.0,
            "exported_at": Utc::now().to_rfc3339(),
            "blueprint": root.blueprint,
        }))
        .context("failed to serialize trace header")?,
    );
    append_session_events(store, session_id, &mut lines).await?;

    let mut stack: Vec<&SessionBlueprint> = children_of(&children_by_parent, session_id);
    while let Some(blueprint) = stack.pop() {
        lines.push(
            serde_json::to_string(&json!({
                "kind": "subagent_header",
                "trace_version": TRACE_FORMAT_VERSION,
                "session_id": blueprint.session_id.0,
                "parent_session_id": blueprint.parent_session_id.as_ref().map(|id| id.0.as_str()),
                "blueprint": blueprint,
            }))
            .context("failed to serialize subagent trace header")?,
        );
        append_session_events(store, &blueprint.session_id, &mut lines).await?;
        stack.extend(children_of(&children_by_parent, &blueprint.session_id));
    }

    let mut output = lines.join("\n");
    output.push('\n');
    Ok(output)
}

fn children_of<'a>(
    children_by_parent: &std::collections::BTreeMap<&str, Vec<&'a SessionBlueprint>>,
    session_id: &SessionId,
) -> Vec<&'a SessionBlueprint> {
    // Reversed so the Vec-as-stack pops children in ascending session-id
    // order, giving a deterministic depth-first layout.
    children_by_parent
        .get(session_id.0.as_str())
        .map(|children| children.iter().rev().copied().collect())
        .unwrap_or_default()
}

async fn append_session_events(
    store: &dyn SessionStore,
    session_id: &SessionId,
    lines: &mut Vec<String>,
) -> anyhow::Result<()> {
    for event in store.replay(session_id).await? {
        lines.push(serde_json::to_string(&event).context("failed to serialize session event")?);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use halter_protocol::{
        Delivery, ModelId, PendingEvent, ResourceSnapshot, SessionEventPayload, SessionState,
        SubagentEventForwarding,
    };
    use halter_session::{InMemorySessionStore, StoredSession};

    use super::*;

    fn blueprint(session_id: &str, parent: Option<&str>) -> SessionBlueprint {
        SessionBlueprint {
            session_id: SessionId::from(session_id),
            parent_session_id: parent.map(SessionId::from),
            default_model: ModelId::from("default"),
            subagent_model: ModelId::from("subagent"),
            subagent_event_forwarding: SubagentEventForwarding::Off,
            snapshot_revision: ResourceSnapshot::empty().revision,
            working_dir: PathBuf::from("."),
            system_prompt_seed: Vec::new(),
            max_turns: None,
            subagent_depth: u32::from(parent.is_some()),
        }
    }

    async fn seed_session(store: &InMemorySessionStore, id: &str, parent: Option<&str>) {
        store
            .create_session(StoredSession::new(
                blueprint(id, parent),
                SessionState::default(),
                Arc::new(ResourceSnapshot::empty()),
            ))
            .await
            .expect("create session");
        store
            .commit(
                &SessionId::from(id),
                None,
                None,
                None,
                vec![
                    PendingEvent::new(
                        SessionId::from(id),
                        Delivery::Lossless,
                        SessionEventPayload::SessionStarted,
                    ),
                    PendingEvent::new(
                        SessionId::from(id),
                        Delivery::Lossless,
                        SessionEventPayload::Warning {
                            message: format!("from-{id}"),
                        },
                    ),
                ],
            )
            .await
            .expect("commit events");
    }

    #[tokio::test]
    async fn export_serializes_root_and_descendants_in_trace_format() {
        let store = InMemorySessionStore::default();
        seed_session(&store, "root", None).await;
        seed_session(&store, "child-b", Some("root")).await;
        seed_session(&store, "child-a", Some("root")).await;
        seed_session(&store, "grandchild", Some("child-a")).await;
        seed_session(&store, "unrelated", None).await;

        let output = export_session_trace(&store, &SessionId::from("root"))
            .await
            .expect("export trace");
        let lines: Vec<serde_json::Value> = output
            .lines()
            .map(|line| serde_json::from_str(line).expect("valid json line"))
            .collect();

        assert_eq!(lines[0]["kind"], "trace_header");
        assert_eq!(lines[0]["trace_version"], 1);
        assert_eq!(lines[0]["session_id"], "root");
        assert_eq!(lines[0]["blueprint"]["session_id"], "root");

        // Root events precede any subagent header; each session contributes
        // a header plus its two events, depth-first with children in
        // session-id order: root, child-a, grandchild, child-b.
        let headers: Vec<(&str, &str)> = lines
            .iter()
            .filter(|line| line["kind"] == "subagent_header")
            .map(|line| {
                (
                    line["session_id"].as_str().expect("session id"),
                    line["parent_session_id"].as_str().expect("parent id"),
                )
            })
            .collect();
        assert_eq!(
            headers,
            vec![
                ("child-a", "root"),
                ("grandchild", "child-a"),
                ("child-b", "root"),
            ]
        );
        assert!(
            !output.contains("unrelated"),
            "sessions outside the tree must not be exported"
        );

        // Committed event lines round-trip as SessionEvents with sequences.
        let root_events: Vec<halter_protocol::SessionEvent> = lines[1..3]
            .iter()
            .map(|line| serde_json::from_value(line.clone()).expect("session event"))
            .collect();
        assert_eq!(root_events[0].session_id, SessionId::from("root"));
        assert_eq!(root_events[0].sequence(), 1);
        assert_eq!(root_events[1].sequence(), 2);
        assert_eq!(lines.len(), 1 + 2 + 3 * (1 + 2));
    }

    #[tokio::test]
    async fn export_fails_for_unknown_session() {
        let store = InMemorySessionStore::default();
        let error = export_session_trace(&store, &SessionId::from("missing"))
            .await
            .expect_err("unknown session must fail");
        assert!(
            error.to_string().contains("unknown session"),
            "unexpected error: {error}"
        );
    }
}
