// pattern: Imperative Shell

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{LineWriter, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::Context;
use chrono::Utc;
use halter_protocol::{PendingEvent, SessionBlueprint, SessionEvent, SessionId};
use serde_json::json;
use tracing::{debug, warn};

use crate::trace_format::{RootTraceHeader, SubagentTraceHeader};

/// Writes a per-root-session JSONL trace into a configured directory. Each
/// root session gets one `<session_id>.txt` file containing a header line
/// followed by interleaved `pending_event` lines (each event as it is
/// generated, before commit) and committed `SessionEvent` lines (each event
/// after the store has assigned a monotonic sequence). Subagent sessions do
/// not get their own file: their blueprint and events are appended to the
/// root's trace, distinguished by the `session_id` carried on each line. The
/// format is dual-purpose: `pending_event` lines give live, in-turn
/// visibility for human debugging, while the committed lines carry the
/// sequence numbers a future `halter` replay feature needs to reconstruct
/// the full session tree.
#[derive(Debug)]
pub struct TraceRecorder {
    dir: PathBuf,
    writers: Mutex<HashMap<SessionId, Arc<Mutex<LineWriter<File>>>>>,
}

impl TraceRecorder {
    /// Creates the trace directory if it does not already exist and returns a
    /// recorder rooted there. Errors when the directory cannot be created or
    /// when `dir` already names a non-directory entry.
    pub fn open(dir: PathBuf) -> anyhow::Result<Self> {
        match fs::metadata(&dir) {
            Ok(metadata) if metadata.is_dir() => {}
            Ok(_) => anyhow::bail!(
                "invalid traces_dir {}: path exists but is not a directory",
                dir.display()
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir_all(&dir)
                    .with_context(|| format!("failed to create traces_dir {}", dir.display()))?;
            }
            Err(error) => {
                return Err(anyhow::Error::new(error)
                    .context(format!("failed to stat traces_dir {}", dir.display())));
            }
        }
        Ok(Self {
            dir,
            writers: Mutex::new(HashMap::new()),
        })
    }

    /// Opens or attaches a trace stream for `session_id`.
    ///
    /// When `parent_session_id` is `None` (root session), creates
    /// `<session_id>.txt` and writes a `trace_header` line. When it is `Some`
    /// and the parent already has an active writer, the subagent is aliased
    /// to that writer — no new file is created — and a `subagent_header`
    /// marker line is appended so that readers can reconstruct the session
    /// tree from a single trace file. If the parent's writer is missing
    /// (already closed, never opened), the subagent's events are silently
    /// dropped: no orphan trace file is produced.
    pub fn open_session(
        &self,
        session_id: &SessionId,
        parent_session_id: Option<&SessionId>,
        blueprint: &SessionBlueprint,
    ) -> anyhow::Result<()> {
        if let Some(parent_id) = parent_session_id {
            let parent_writer = {
                let writers = self.writers.lock().map_err(|_| {
                    anyhow::anyhow!("trace recorder writer map mutex poisoned during open_session")
                })?;
                writers.get(parent_id).cloned()
            };
            let Some(parent_writer) = parent_writer else {
                warn!(
                    session_id = %session_id,
                    parent_session_id = %parent_id,
                    "no parent trace writer; subagent trace events will be dropped"
                );
                return Ok(());
            };
            let header = SubagentTraceHeader::new(session_id, parent_id, blueprint);
            let mut line =
                serde_json::to_vec(&header).context("failed to serialize subagent trace header")?;
            line.push(b'\n');
            {
                let mut writer = parent_writer.lock().map_err(|_| {
                    anyhow::anyhow!(
                        "trace recorder file mutex poisoned during open_session for subagent"
                    )
                })?;
                writer
                    .write_all(&line)
                    .context("failed to append subagent header to trace file")?;
            }
            let mut writers = self.writers.lock().map_err(|_| {
                anyhow::anyhow!(
                    "trace recorder writer map mutex poisoned during open_session for subagent"
                )
            })?;
            writers.insert(session_id.clone(), parent_writer);
            debug!(
                session_id = %session_id,
                parent_session_id = %parent_id,
                "aliased subagent trace to parent trace file"
            );
            return Ok(());
        }

        let path = self.session_path(session_id);
        let file = File::create(&path)
            .with_context(|| format!("failed to create trace file {}", path.display()))?;
        let mut writer = LineWriter::new(file);
        let header = RootTraceHeader::new(session_id, blueprint);
        let mut line = serde_json::to_vec(&header).context("failed to serialize trace header")?;
        line.push(b'\n');
        writer
            .write_all(&line)
            .with_context(|| format!("failed to write trace header to {}", path.display()))?;
        let mut writers = self.writers.lock().map_err(|_| {
            anyhow::anyhow!("trace recorder writer map mutex poisoned during open_session")
        })?;
        writers.insert(session_id.clone(), Arc::new(Mutex::new(writer)));
        debug!(session_id = %session_id, path = %path.display(), "opened session trace file");
        Ok(())
    }

    /// Appends a `pending_event` preview line to the session's trace file as
    /// soon as the event is generated, before the session store has assigned
    /// a sequence. Lets a long-running turn show progress in the trace
    /// instead of buffering everything until the turn commits. The committed
    /// counterpart still arrives via [`TraceRecorder::record`] after the commit, so replay
    /// tools can ignore `pending_event` lines and rely on `SessionEvent`
    /// lines as ground truth. Best-effort: failures are logged at WARN and
    /// do not interrupt the caller.
    pub fn record_pending(&self, pending: &PendingEvent) {
        let writer = match self.writers.lock() {
            Ok(map) => map.get(&pending.session_id).cloned(),
            Err(_) => {
                warn!(session_id = %pending.session_id, "trace recorder writer map mutex poisoned; dropping pending event");
                return;
            }
        };
        let Some(writer) = writer else { return };
        let envelope = json!({
            "kind": "pending_event",
            "session_id": pending.session_id.0,
            "delivery": pending.delivery,
            "recorded_at": Utc::now().to_rfc3339(),
            "payload": pending.payload,
        });
        let mut line = match serde_json::to_vec(&envelope) {
            Ok(line) => line,
            Err(error) => {
                warn!(session_id = %pending.session_id, %error, "failed to serialize pending event for trace");
                return;
            }
        };
        line.push(b'\n');
        match writer.lock() {
            Ok(mut writer) => {
                if let Err(error) = writer.write_all(&line) {
                    warn!(session_id = %pending.session_id, %error, "failed to append pending event to trace file");
                }
            }
            Err(_) => {
                warn!(session_id = %pending.session_id, "trace recorder file mutex poisoned; dropping pending event");
            }
        }
    }

    /// Appends a single committed `SessionEvent` to the session's trace file.
    /// Best-effort: failures (poisoned mutex, broken file handle) are logged
    /// at WARN and do not interrupt the publish path.
    pub fn record(&self, event: &SessionEvent) {
        let writer = match self.writers.lock() {
            Ok(map) => map.get(&event.session_id).cloned(),
            Err(_) => {
                warn!(session_id = %event.session_id, "trace recorder writer map mutex poisoned; dropping event");
                return;
            }
        };
        let Some(writer) = writer else { return };
        let mut line = match serde_json::to_vec(event) {
            Ok(line) => line,
            Err(error) => {
                warn!(session_id = %event.session_id, %error, "failed to serialize session event for trace");
                return;
            }
        };
        line.push(b'\n');
        match writer.lock() {
            Ok(mut writer) => {
                if let Err(error) = writer.write_all(&line) {
                    warn!(session_id = %event.session_id, %error, "failed to append event to trace file");
                }
            }
            Err(_) => {
                warn!(session_id = %event.session_id, "trace recorder file mutex poisoned; dropping event");
            }
        }
    }

    /// Drops the writer for `session_id`, flushing any buffered output.
    pub fn close_session(&self, session_id: &SessionId) {
        let removed = match self.writers.lock() {
            Ok(mut map) => map.remove(session_id),
            Err(_) => {
                warn!(session_id = %session_id, "trace recorder writer map mutex poisoned during close_session");
                return;
            }
        };
        if let Some(writer) = removed {
            if let Ok(mut writer) = writer.lock() {
                let _ = writer.flush();
            }
            debug!(session_id = %session_id, "closed session trace file");
        }
    }

    fn session_path(&self, session_id: &SessionId) -> PathBuf {
        self.dir
            .join(format!("{}.txt", sanitize_session_id(&session_id.0)))
    }

    /// Test-only accessor returning the directory backing this recorder.
    #[cfg(test)]
    pub(crate) fn dir(&self) -> &std::path::Path {
        &self.dir
    }
}

/// Replaces every byte that is not safe for an unquoted filename with `_`.
/// Path separators, control bytes, and drive letters all collapse to `_`,
/// which means the recorder cannot be tricked into writing outside `dir`
/// regardless of how a session id was constructed by upstream callers.
fn sanitize_session_id(id: &str) -> String {
    if id.is_empty() {
        return "_".to_owned();
    }
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use halter_protocol::{
        Delivery, ModelId, PendingEvent, Revision, SessionBlueprint, SessionEventPayload,
        SessionId, SubagentEventForwarding,
    };
    use tempfile::tempdir;

    use super::*;

    fn blueprint(session_id: &SessionId) -> SessionBlueprint {
        SessionBlueprint {
            session_id: session_id.clone(),
            parent_session_id: None,
            default_model: ModelId::from("default"),
            subagent_model: ModelId::from("subagent"),
            subagent_event_forwarding: SubagentEventForwarding::Off,
            snapshot_revision: Revision::from("rev-1".to_owned()),
            working_dir: PathBuf::from("/tmp/halter-test"),
            system_prompt_seed: Vec::new(),
            max_turns: None,
            subagent_depth: 0,
        }
    }

    fn warning_pending(session_id: &SessionId, text: &str) -> PendingEvent {
        PendingEvent::new(
            session_id.clone(),
            Delivery::Lossless,
            SessionEventPayload::Warning {
                message: text.to_owned(),
            },
        )
    }

    fn warning_event(
        session_id: &SessionId,
        sequence: u64,
        text: &str,
    ) -> halter_protocol::SessionEvent {
        warning_pending(session_id, text).into_committed(sequence)
    }

    #[test]
    fn open_creates_directory_when_missing() {
        let temp = tempdir().expect("tempdir");
        let target = temp.path().join("nested/traces");
        let recorder = TraceRecorder::open(target.clone()).expect("recorder");
        assert!(target.is_dir());
        assert_eq!(recorder.dir(), target.as_path());
    }

    #[test]
    fn open_rejects_path_pointing_at_a_file() {
        let temp = tempdir().expect("tempdir");
        let path = temp.path().join("not-a-dir.txt");
        std::fs::write(&path, b"hi").expect("seed file");
        let error = TraceRecorder::open(path.clone()).expect_err("open should fail");
        assert!(
            error.to_string().contains("not a directory"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn records_header_and_events_to_per_session_file() {
        let temp = tempdir().expect("tempdir");
        let recorder = TraceRecorder::open(temp.path().to_path_buf()).expect("recorder");
        let session_id = SessionId::from("session-abc");

        recorder
            .open_session(&session_id, None, &blueprint(&session_id))
            .expect("open session");
        recorder.record(&warning_event(&session_id, 1, "hello"));
        recorder.record(&warning_event(&session_id, 2, "again"));
        recorder.close_session(&session_id);

        let path = temp.path().join("session-abc.txt");
        let contents = std::fs::read_to_string(&path).expect("read trace");
        let mut lines = contents.lines();

        let header: serde_json::Value =
            serde_json::from_str(lines.next().expect("header line")).expect("header json");
        assert_eq!(header["kind"], "trace_header");
        assert_eq!(header["trace_version"], 2);
        assert_eq!(header["session_id"], "session-abc");
        assert!(header["generated_at"].is_string());
        assert!(header.get("started_at").is_none());
        assert!(header.get("exported_at").is_none());
        assert_eq!(header["blueprint"]["session_id"], "session-abc");

        let event_one: halter_protocol::SessionEvent =
            serde_json::from_str(lines.next().expect("event one")).expect("event one json");
        assert_eq!(event_one.sequence(), 1);
        let event_two: halter_protocol::SessionEvent =
            serde_json::from_str(lines.next().expect("event two")).expect("event two json");
        assert_eq!(event_two.sequence(), 2);
        assert!(lines.next().is_none());
    }

    #[test]
    fn record_without_open_session_is_a_silent_noop() {
        let temp = tempdir().expect("tempdir");
        let recorder = TraceRecorder::open(temp.path().to_path_buf()).expect("recorder");
        // No file should be created and no panic should occur.
        recorder.record(&warning_event(&SessionId::from("orphan"), 1, "ignored"));
        let entries: Vec<_> = std::fs::read_dir(temp.path())
            .expect("read dir")
            .map(|entry| entry.expect("entry").file_name())
            .collect();
        assert!(entries.is_empty(), "unexpected files: {entries:?}");
    }

    #[test]
    fn close_session_drops_writer_so_subsequent_records_are_noops() {
        let temp = tempdir().expect("tempdir");
        let recorder = TraceRecorder::open(temp.path().to_path_buf()).expect("recorder");
        let session_id = SessionId::from("session-close");
        recorder
            .open_session(&session_id, None, &blueprint(&session_id))
            .expect("open session");
        recorder.close_session(&session_id);
        recorder.record(&warning_event(&session_id, 1, "after-close"));

        let path = temp.path().join("session-close.txt");
        let contents = std::fs::read_to_string(&path).expect("read trace");
        // Only the header line should be present; the post-close record must
        // not be appended once the writer has been dropped.
        assert_eq!(contents.lines().count(), 1, "contents:\n{contents}");
    }

    #[test]
    fn record_pending_writes_preview_line_before_commit() {
        let temp = tempdir().expect("tempdir");
        let recorder = TraceRecorder::open(temp.path().to_path_buf()).expect("recorder");
        let session_id = SessionId::from("session-live");

        recorder
            .open_session(&session_id, None, &blueprint(&session_id))
            .expect("open session");
        recorder.record_pending(&warning_pending(&session_id, "live-1"));
        recorder.record_pending(&warning_pending(&session_id, "live-2"));
        recorder.record(&warning_event(&session_id, 1, "live-1"));
        recorder.close_session(&session_id);

        let contents =
            std::fs::read_to_string(temp.path().join("session-live.txt")).expect("read trace");
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 4, "contents:\n{contents}");

        let preview_one: serde_json::Value =
            serde_json::from_str(lines[1]).expect("preview-1 json");
        assert_eq!(preview_one["kind"], "pending_event");
        assert_eq!(preview_one["session_id"], "session-live");
        assert_eq!(preview_one["payload"]["kind"], "warning");
        assert_eq!(preview_one["payload"]["message"], "live-1");
        assert!(
            preview_one.get("sequence").is_none(),
            "pending events must not carry a sequence"
        );

        let preview_two: serde_json::Value =
            serde_json::from_str(lines[2]).expect("preview-2 json");
        assert_eq!(preview_two["kind"], "pending_event");
        assert_eq!(preview_two["payload"]["message"], "live-2");

        let committed: halter_protocol::SessionEvent =
            serde_json::from_str(lines[3]).expect("committed event json");
        assert_eq!(committed.sequence(), 1);
    }

    #[test]
    fn record_pending_without_open_session_is_a_silent_noop() {
        let temp = tempdir().expect("tempdir");
        let recorder = TraceRecorder::open(temp.path().to_path_buf()).expect("recorder");
        recorder.record_pending(&warning_pending(&SessionId::from("orphan"), "ignored"));
        let entries: Vec<_> = std::fs::read_dir(temp.path())
            .expect("read dir")
            .map(|entry| entry.expect("entry").file_name())
            .collect();
        assert!(entries.is_empty(), "unexpected files: {entries:?}");
    }

    #[test]
    fn subagent_events_land_in_parent_trace_file_with_subagent_header() {
        let temp = tempdir().expect("tempdir");
        let recorder = TraceRecorder::open(temp.path().to_path_buf()).expect("recorder");
        let parent_id = SessionId::from("session-parent");
        let child_id = SessionId::from("session-child");

        recorder
            .open_session(&parent_id, None, &blueprint(&parent_id))
            .expect("open parent session");
        let mut child_blueprint = blueprint(&child_id);
        child_blueprint.parent_session_id = Some(parent_id.clone());
        child_blueprint.subagent_depth = 1;
        recorder
            .open_session(&child_id, Some(&parent_id), &child_blueprint)
            .expect("open subagent session");
        recorder.record(&warning_event(&parent_id, 1, "from-parent"));
        recorder.record(&warning_event(&child_id, 1, "from-child"));
        recorder.close_session(&child_id);
        recorder.close_session(&parent_id);

        let entries: Vec<_> = std::fs::read_dir(temp.path())
            .expect("read dir")
            .map(|entry| entry.expect("entry").file_name())
            .collect();
        assert_eq!(
            entries.len(),
            1,
            "subagent must not produce its own file: {entries:?}"
        );
        assert_eq!(entries[0].to_string_lossy(), "session-parent.txt");

        let contents =
            std::fs::read_to_string(temp.path().join("session-parent.txt")).expect("read trace");
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 4, "contents:\n{contents}");

        let header: serde_json::Value = serde_json::from_str(lines[0]).expect("trace header json");
        assert_eq!(header["kind"], "trace_header");
        assert_eq!(header["session_id"], "session-parent");

        let subagent_header: serde_json::Value =
            serde_json::from_str(lines[1]).expect("subagent header json");
        assert_eq!(subagent_header["kind"], "subagent_header");
        assert_eq!(subagent_header["session_id"], "session-child");
        assert_eq!(subagent_header["parent_session_id"], "session-parent");
        assert_eq!(subagent_header["blueprint"]["subagent_depth"], 1);

        let parent_event: halter_protocol::SessionEvent =
            serde_json::from_str(lines[2]).expect("parent event json");
        assert_eq!(parent_event.session_id, parent_id);
        let child_event: halter_protocol::SessionEvent =
            serde_json::from_str(lines[3]).expect("child event json");
        assert_eq!(child_event.session_id, child_id);
    }

    #[test]
    fn subagent_open_without_known_parent_drops_events_silently() {
        let temp = tempdir().expect("tempdir");
        let recorder = TraceRecorder::open(temp.path().to_path_buf()).expect("recorder");
        let orphan_parent = SessionId::from("session-missing");
        let child_id = SessionId::from("session-child");
        let mut child_blueprint = blueprint(&child_id);
        child_blueprint.parent_session_id = Some(orphan_parent.clone());

        recorder
            .open_session(&child_id, Some(&orphan_parent), &child_blueprint)
            .expect("open subagent session");
        recorder.record(&warning_event(&child_id, 1, "should-be-dropped"));

        let entries: Vec<_> = std::fs::read_dir(temp.path())
            .expect("read dir")
            .map(|entry| entry.expect("entry").file_name())
            .collect();
        assert!(
            entries.is_empty(),
            "no orphan trace file should be written: {entries:?}"
        );
    }

    #[test]
    fn unsafe_session_ids_are_sanitized_and_cannot_escape_dir() {
        let temp = tempdir().expect("tempdir");
        let recorder = TraceRecorder::open(temp.path().to_path_buf()).expect("recorder");
        let session_id = SessionId::from("../../etc/passwd");
        recorder
            .open_session(&session_id, None, &blueprint(&session_id))
            .expect("open session");
        recorder.close_session(&session_id);

        let entries: Vec<_> = std::fs::read_dir(temp.path())
            .expect("read dir")
            .map(|entry| entry.expect("entry").file_name())
            .collect();
        assert_eq!(entries.len(), 1);
        let name = entries[0].to_string_lossy();
        assert!(
            !name.contains('/') && !name.contains(".."),
            "filename leaked path traversal: {name}"
        );
        assert!(name.ends_with(".txt"));
    }
}
