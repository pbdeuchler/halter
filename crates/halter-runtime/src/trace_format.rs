//! Shared JSONL trace header records.
//!
//! Live traces and replay-derived exports deliberately use these same types so
//! a format version always describes one header schema.

use chrono::Utc;
use halter_protocol::{SessionBlueprint, SessionId};
use serde::Serialize;

/// Trace stream format identifier; bumped when readers must be taught a new
/// on-disk shape. Version 2 replaces the ambiguous live-only `started_at`
/// field with `generated_at`, which is accurate for both live and exported
/// headers.
pub(crate) const TRACE_FORMAT_VERSION: u32 = 2;

#[derive(Debug, Serialize)]
pub(crate) struct RootTraceHeader<'a> {
    kind: &'static str,
    trace_version: u32,
    session_id: &'a SessionId,
    generated_at: String,
    blueprint: &'a SessionBlueprint,
}

impl<'a> RootTraceHeader<'a> {
    pub(crate) fn new(session_id: &'a SessionId, blueprint: &'a SessionBlueprint) -> Self {
        Self {
            kind: "trace_header",
            trace_version: TRACE_FORMAT_VERSION,
            session_id,
            generated_at: Utc::now().to_rfc3339(),
            blueprint,
        }
    }
}

#[derive(Debug, Serialize)]
pub(crate) struct SubagentTraceHeader<'a> {
    kind: &'static str,
    trace_version: u32,
    session_id: &'a SessionId,
    parent_session_id: &'a SessionId,
    generated_at: String,
    blueprint: &'a SessionBlueprint,
}

impl<'a> SubagentTraceHeader<'a> {
    pub(crate) fn new(
        session_id: &'a SessionId,
        parent_session_id: &'a SessionId,
        blueprint: &'a SessionBlueprint,
    ) -> Self {
        Self {
            kind: "subagent_header",
            trace_version: TRACE_FORMAT_VERSION,
            session_id,
            parent_session_id,
            generated_at: Utc::now().to_rfc3339(),
            blueprint,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use halter_protocol::{ModelId, ResourceSnapshot, SubagentEventForwarding};

    use super::*;

    fn blueprint(session_id: &SessionId, parent_session_id: Option<SessionId>) -> SessionBlueprint {
        SessionBlueprint {
            session_id: session_id.clone(),
            parent_session_id,
            default_model: ModelId::from("default"),
            subagent_model: ModelId::from("subagent"),
            subagent_event_forwarding: SubagentEventForwarding::Off,
            snapshot_revision: ResourceSnapshot::empty().revision,
            working_dir: PathBuf::from("."),
            system_prompt_seed: Vec::new(),
            max_turns: None,
            subagent_depth: 0,
        }
    }

    #[test]
    fn root_and_subagent_headers_share_version_and_timestamp_schema() {
        let root_id = SessionId::from("root");
        let child_id = SessionId::from("child");
        let root_blueprint = blueprint(&root_id, None);
        let child_blueprint = blueprint(&child_id, Some(root_id.clone()));

        let root = serde_json::to_value(RootTraceHeader::new(&root_id, &root_blueprint))
            .expect("serialize root header");
        let child = serde_json::to_value(SubagentTraceHeader::new(
            &child_id,
            &root_id,
            &child_blueprint,
        ))
        .expect("serialize subagent header");

        assert_eq!(root["trace_version"], TRACE_FORMAT_VERSION);
        assert_eq!(child["trace_version"], TRACE_FORMAT_VERSION);
        assert!(root["generated_at"].is_string());
        assert!(child["generated_at"].is_string());
        assert!(root.get("started_at").is_none());
        assert!(child.get("started_at").is_none());
        assert!(root.get("exported_at").is_none());
        assert!(child.get("exported_at").is_none());
    }
}
