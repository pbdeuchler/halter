// pattern: Imperative Shell

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use async_trait::async_trait;
use halter_protocol::{
    Delivery, PendingEvent, ResourceSnapshot, SessionBlueprint, SessionEvent, SessionEventPayload,
    SessionId, SessionState,
};
use rusqlite::{Connection, ErrorCode, OptionalExtension, TransactionBehavior, params};
use tracing::{debug, info};

use crate::{SessionCommitConflict, SessionStore, StoredSession};

const MIGRATIONS: &[(u32, &str)] = &[(
    1,
    r#"
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    parent_session_id TEXT,
    blueprint TEXT NOT NULL,
    state TEXT NOT NULL,
    snapshot_revision TEXT NOT NULL REFERENCES snapshots(revision),
    created_at INTEGER NOT NULL
);

CREATE TABLE snapshots (
    revision TEXT PRIMARY KEY,
    data TEXT NOT NULL
);

CREATE TABLE events (
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    sequence INTEGER NOT NULL,
    delivery TEXT NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (session_id, sequence)
);
"#,
)];

// Compile-time guarantee that MIGRATIONS is sorted by strictly increasing
// version. run_migrations skips any entry whose version is <= the current
// schema version, so an unsorted table would silently drop migrations.
// (finding L15)
const _: () = {
    let mut i = 1;
    while i < MIGRATIONS.len() {
        assert!(
            MIGRATIONS[i - 1].0 < MIGRATIONS[i].0,
            "MIGRATIONS must be strictly monotonic in version"
        );
        i += 1;
    }
};

pub struct SqliteSessionStore {
    connection: Arc<Mutex<Connection>>,
}

impl SqliteSessionStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if !is_in_memory_path(path)
            && let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent).with_context(|| {
                format!(
                    "failed to create sqlite session store directory at {}",
                    parent.display()
                )
            })?;
        }

        let mut connection = Connection::open(path).with_context(|| {
            format!("failed to open sqlite session store at {}", path.display())
        })?;
        verify_integrity(&connection)?;
        configure_connection(&connection)?;
        run_migrations(&mut connection, MIGRATIONS)?;
        let schema_version = current_version(&connection)?;

        info!(
            path = %path.display(),
            schema_version,
            "opened sqlite session store"
        );

        Ok(Self {
            connection: Arc::new(Mutex::new(connection)),
        })
    }

    pub fn open_default() -> Result<Self> {
        let path = default_db_path()?;
        Self::open(path)
    }

    async fn with_conn<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&mut Connection) -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let connection = Arc::clone(&self.connection);
        tokio::task::spawn_blocking(move || {
            let mut guard = connection
                .lock()
                .map_err(|_| anyhow::anyhow!("failed to lock sqlite session store connection"))?;
            f(&mut guard)
        })
        .await
        .context("failed to join sqlite session store task")?
    }
}

#[async_trait]
impl SessionStore for SqliteSessionStore {
    async fn create_session(&self, session: StoredSession) -> Result<()> {
        let session_id = session.blueprint.session_id.clone();
        debug!(session_id = %session_id, "creating sqlite session");
        self.with_conn(move |conn| create_session_with_conn(conn, session))
            .await
    }

    async fn load_session(&self, session_id: &SessionId) -> Result<Option<StoredSession>> {
        let session_id = session_id.clone();
        let log_session_id = session_id.clone();
        let loaded = self
            .with_conn(move |conn| load_session_with_conn(conn, &session_id))
            .await?;
        debug!(session_id = %log_session_id, found = loaded.is_some(), "loading sqlite session");
        Ok(loaded)
    }

    async fn commit(
        &self,
        session_id: &SessionId,
        snapshot: Option<Arc<ResourceSnapshot>>,
        expected_state: Option<SessionState>,
        state: Option<SessionState>,
        events: Vec<PendingEvent>,
    ) -> Result<Vec<SessionEvent>> {
        let session_id = session_id.clone();
        self.with_conn(move |conn| {
            commit_with_conn(conn, &session_id, snapshot, expected_state, state, events)
        })
        .await
    }

    async fn replay(&self, session_id: &SessionId) -> Result<Vec<SessionEvent>> {
        let session_id = session_id.clone();
        let log_session_id = session_id.clone();
        let events = self
            .with_conn(move |conn| replay_with_conn(conn, &session_id))
            .await?;
        debug!(session_id = %log_session_id, event_count = events.len(), "replaying sqlite session");
        Ok(events)
    }

    async fn list_sessions(&self) -> Result<Vec<SessionBlueprint>> {
        let sessions = self.with_conn(list_sessions_with_conn).await?;
        debug!(session_count = sessions.len(), "listing sqlite sessions");
        Ok(sessions)
    }
}

fn create_session_with_conn(conn: &mut Connection, session: StoredSession) -> Result<()> {
    ensure_snapshot_revision_matches(&session.blueprint, session.snapshot.as_ref())?;

    let session_id = session.blueprint.session_id.0.clone();
    let parent_session_id = session
        .blueprint
        .parent_session_id
        .as_ref()
        .map(|id| id.0.clone());
    let snapshot_revision = session.snapshot.revision.0.clone();
    let blueprint_json = serde_json::to_string(&session.blueprint)
        .context("failed to serialize session blueprint")?;
    let state_json =
        serde_json::to_string(&session.state).context("failed to serialize session state")?;
    let created_at = unix_timestamp_seconds()?;

    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .context("failed to start sqlite create_session transaction")?;
    store_snapshot(&tx, session.snapshot.as_ref())?;

    match tx.execute(
        "INSERT INTO sessions (
            session_id,
            parent_session_id,
            blueprint,
            state,
            snapshot_revision,
            created_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            session_id.as_str(),
            parent_session_id,
            blueprint_json,
            state_json,
            snapshot_revision.as_str(),
            created_at,
        ],
    ) {
        Ok(_) => {}
        Err(error) if is_constraint_violation(&error) => {
            anyhow::bail!(
                "failed to create session: session '{}' already exists",
                session_id
            );
        }
        Err(error) => {
            return Err(error)
                .with_context(|| format!("failed to insert sqlite session '{}'", session_id));
        }
    }

    tx.commit()
        .context("failed to commit sqlite create_session transaction")?;
    Ok(())
}

fn load_session_with_conn(
    conn: &mut Connection,
    session_id: &SessionId,
) -> Result<Option<StoredSession>> {
    let row = conn
        .query_row(
            "SELECT s.blueprint, s.state, snap.data
             FROM sessions s
             LEFT JOIN snapshots snap ON snap.revision = s.snapshot_revision
             WHERE s.session_id = ?1",
            [session_id.0.as_str()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                ))
            },
        )
        .optional()
        .with_context(|| format!("failed to load session '{}'", session_id.0))?;

    let Some((blueprint_json, state_json, snapshot_json)) = row else {
        return Ok(None);
    };
    let snapshot_json = snapshot_json.with_context(|| {
        format!(
            "failed to load session '{}': snapshot row is missing",
            session_id.0
        )
    })?;

    let blueprint: SessionBlueprint =
        serde_json::from_str(&blueprint_json).context("failed to deserialize session blueprint")?;
    let state: SessionState =
        serde_json::from_str(&state_json).context("failed to deserialize session state")?;
    let snapshot: ResourceSnapshot =
        serde_json::from_str(&snapshot_json).context("failed to deserialize resource snapshot")?;
    ensure_snapshot_revision_matches(&blueprint, &snapshot)?;

    Ok(Some(StoredSession {
        blueprint,
        state,
        snapshot: Arc::new(snapshot),
    }))
}

fn commit_with_conn(
    conn: &mut Connection,
    session_id: &SessionId,
    snapshot: Option<Arc<ResourceSnapshot>>,
    expected_state: Option<SessionState>,
    state: Option<SessionState>,
    events: Vec<PendingEvent>,
) -> Result<Vec<SessionEvent>> {
    let started_at = Instant::now();
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .context("failed to start sqlite commit transaction")?;

    let (blueprint_json, current_state_json) = tx
        .query_row(
            "SELECT blueprint, state FROM sessions WHERE session_id = ?1",
            [session_id.0.as_str()],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .with_context(|| format!("failed to load session '{}'", session_id.0))?
        .with_context(|| {
            format!(
                "failed to commit session: unknown session '{}'",
                session_id.0
            )
        })?;

    if let Some(expected_state) = expected_state.as_ref() {
        let expected_state_json = serde_json::to_string(expected_state)
            .context("failed to serialize expected session state")?;
        if current_state_json != expected_state_json {
            return Err(SessionCommitConflict {
                session_id: session_id.clone(),
            }
            .into());
        }
    }

    if let Some(snapshot) = snapshot.as_ref() {
        store_snapshot(&tx, snapshot.as_ref())?;
        let mut blueprint: SessionBlueprint = serde_json::from_str(&blueprint_json)
            .context("failed to deserialize session blueprint")?;
        blueprint.snapshot_revision = snapshot.revision.clone();
        let updated_blueprint_json =
            serde_json::to_string(&blueprint).context("failed to serialize session blueprint")?;
        tx.execute(
            "UPDATE sessions
             SET snapshot_revision = ?2, blueprint = ?3
             WHERE session_id = ?1",
            params![
                session_id.0.as_str(),
                snapshot.revision.0.as_str(),
                updated_blueprint_json,
            ],
        )
        .with_context(|| format!("failed to update snapshot for session '{}'", session_id.0))?;
    }

    if let Some(state) = state.as_ref() {
        let state_json =
            serde_json::to_string(state).context("failed to serialize session state")?;
        tx.execute(
            "UPDATE sessions SET state = ?2 WHERE session_id = ?1",
            params![session_id.0.as_str(), state_json],
        )
        .with_context(|| format!("failed to update state for session '{}'", session_id.0))?;
    }

    let starting_sequence = next_event_sequence(&tx, session_id)?;
    let mut committed = Vec::with_capacity(events.len());
    for (offset, event) in events.into_iter().enumerate() {
        let sequence = starting_sequence + offset as u64;
        let payload_json = serde_json::to_string(&event.payload)
            .context("failed to serialize session event payload")?;
        tx.execute(
            "INSERT INTO events (session_id, sequence, delivery, payload)
             VALUES (?1, ?2, ?3, ?4)",
            params![
                session_id.0.as_str(),
                sequence_to_sql(sequence)?,
                delivery_to_sql(event.delivery),
                payload_json,
            ],
        )
        .with_context(|| {
            format!(
                "failed to persist event {} for session '{}'",
                sequence, session_id.0
            )
        })?;

        committed.push(
            PendingEvent {
                session_id: session_id.clone(),
                delivery: event.delivery,
                payload: event.payload,
            }
            .into_committed(sequence),
        );
    }

    tx.commit()
        .context("failed to commit sqlite session transaction")?;
    debug!(
        session_id = %session_id,
        event_count = committed.len(),
        replace_snapshot = snapshot.is_some(),
        replace_state = state.is_some(),
        duration_ms = started_at.elapsed().as_millis(),
        "committed sqlite session state"
    );
    Ok(committed)
}

fn replay_with_conn(conn: &mut Connection, session_id: &SessionId) -> Result<Vec<SessionEvent>> {
    let mut statement = conn
        .prepare(
            "SELECT session_id, sequence, delivery, payload
             FROM events
             WHERE session_id = ?1
             ORDER BY sequence ASC",
        )
        .context("failed to prepare sqlite replay query")?;
    let rows = statement
        .query_map([session_id.0.as_str()], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
            ))
        })
        .with_context(|| format!("failed to replay session '{}'", session_id.0))?;

    let mut events = Vec::new();
    for row in rows {
        let (raw_session_id, raw_sequence, raw_delivery, raw_payload) = row?;
        let payload: SessionEventPayload = serde_json::from_str(&raw_payload)
            .context("failed to deserialize session event payload")?;
        events.push(SessionEvent::new_committed(
            SessionId::from(raw_session_id),
            sequence_from_sql(raw_sequence)?,
            delivery_from_sql(&raw_delivery)?,
            payload,
        ));
    }
    Ok(events)
}

fn list_sessions_with_conn(conn: &mut Connection) -> Result<Vec<SessionBlueprint>> {
    let mut statement = conn
        .prepare(
            "SELECT blueprint
             FROM sessions
             ORDER BY created_at ASC, session_id ASC",
        )
        .context("failed to prepare sqlite list_sessions query")?;
    let rows = statement
        .query_map([], |row| row.get::<_, String>(0))
        .context("failed to list sqlite sessions")?;

    let mut sessions = Vec::new();
    for row in rows {
        let blueprint_json = row?;
        let blueprint: SessionBlueprint = serde_json::from_str(&blueprint_json)
            .context("failed to deserialize session blueprint")?;
        sessions.push(blueprint);
    }
    Ok(sessions)
}

fn configure_connection(conn: &Connection) -> Result<()> {
    conn.pragma_update(None, "journal_mode", "WAL")
        .context("failed to set sqlite journal mode")?;
    conn.pragma_update(None, "synchronous", "NORMAL")
        .context("failed to set sqlite synchronous mode")?;
    conn.pragma_update(None, "foreign_keys", "ON")
        .context("failed to enable sqlite foreign keys")?;
    conn.busy_timeout(Duration::from_millis(5_000))
        .context("failed to set sqlite busy timeout")?;
    conn.pragma_update(None, "cache_size", -8_000i64)
        .context("failed to set sqlite cache size")?;
    Ok(())
}

fn verify_integrity(conn: &Connection) -> Result<()> {
    let status: String = conn
        .query_row("PRAGMA integrity_check", [], |row| row.get(0))
        .context("failed integrity check")?;
    if status != "ok" {
        anyhow::bail!("failed integrity check: {status}");
    }
    Ok(())
}

fn current_version(conn: &Connection) -> Result<u32> {
    let version = conn
        .query_row("PRAGMA user_version", [], |row| row.get::<_, i64>(0))
        .context("failed to read sqlite schema version")?;
    u32::try_from(version).context("failed to convert sqlite schema version to u32")
}

fn run_migrations(conn: &mut Connection, migrations: &[(u32, &str)]) -> Result<()> {
    let mut version = current_version(conn)?;
    for (target_version, sql) in migrations {
        if *target_version <= version {
            continue;
        }

        info!(
            from_version = version,
            to_version = *target_version,
            "applying sqlite session store migration"
        );
        let tx = conn
            .transaction()
            .context("failed to start sqlite migration transaction")?;
        tx.execute_batch(sql)
            .with_context(|| format!("failed to apply sqlite migration {}", target_version))?;
        tx.pragma_update(None, "user_version", i64::from(*target_version))
            .with_context(|| {
                format!("failed to bump sqlite schema version to {}", target_version)
            })?;
        tx.commit()
            .with_context(|| format!("failed to commit sqlite migration {}", target_version))?;
        version = *target_version;
    }
    Ok(())
}

fn next_event_sequence(conn: &Connection, session_id: &SessionId) -> Result<u64> {
    let next_sequence = conn
        .query_row(
            "SELECT COALESCE(MAX(sequence), 0) + 1
             FROM events
             WHERE session_id = ?1",
            [session_id.0.as_str()],
            |row| row.get::<_, i64>(0),
        )
        .with_context(|| {
            format!(
                "failed to compute next event sequence for '{}'",
                session_id.0
            )
        })?;
    sequence_from_sql(next_sequence)
}

fn store_snapshot(conn: &Connection, snapshot: &ResourceSnapshot) -> Result<()> {
    let snapshot_json =
        serde_json::to_string(snapshot).context("failed to serialize resource snapshot")?;
    let inserted = conn
        .execute(
            "INSERT INTO snapshots (revision, data)
             VALUES (?1, ?2)
             ON CONFLICT(revision) DO NOTHING",
            params![snapshot.revision.0.as_str(), snapshot_json.as_str()],
        )
        .with_context(|| format!("failed to persist snapshot '{}'", snapshot.revision.0))?;
    if inserted == 0 {
        let existing_json: String = conn
            .query_row(
                "SELECT data FROM snapshots WHERE revision = ?1",
                [snapshot.revision.0.as_str()],
                |row| row.get(0),
            )
            .with_context(|| {
                format!("failed to load existing snapshot '{}'", snapshot.revision.0)
            })?;
        if existing_json != snapshot_json {
            anyhow::bail!(
                "failed to persist snapshot: revision '{}' already exists with different data",
                snapshot.revision.0
            );
        }
    }
    Ok(())
}

fn ensure_snapshot_revision_matches(
    blueprint: &SessionBlueprint,
    snapshot: &ResourceSnapshot,
) -> Result<()> {
    if blueprint.snapshot_revision != snapshot.revision {
        anyhow::bail!(
            "failed to persist session: blueprint snapshot revision '{}' does not match snapshot '{}'",
            blueprint.snapshot_revision,
            snapshot.revision
        );
    }
    Ok(())
}

fn delivery_to_sql(delivery: Delivery) -> &'static str {
    match delivery {
        Delivery::Lossless => "lossless",
        Delivery::BestEffort => "best_effort",
    }
}

fn delivery_from_sql(raw: &str) -> Result<Delivery> {
    match raw {
        "lossless" => Ok(Delivery::Lossless),
        "best_effort" => Ok(Delivery::BestEffort),
        _ => anyhow::bail!("failed to decode session event delivery '{}'", raw),
    }
}

fn sequence_to_sql(sequence: u64) -> Result<i64> {
    i64::try_from(sequence).context("failed to convert event sequence to sqlite integer")
}

fn sequence_from_sql(sequence: i64) -> Result<u64> {
    u64::try_from(sequence).context("failed to convert sqlite event sequence to u64")
}

fn unix_timestamp_seconds() -> Result<i64> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("failed to compute unix timestamp")?;
    i64::try_from(duration.as_secs()).context("failed to convert unix timestamp to i64")
}

/// Resolves the default sqlite session-store path.
///
/// Precedence (all platforms):
/// 1. `$XDG_DATA_HOME/halter/sessions.db` if `XDG_DATA_HOME` is set and
///    non-empty. This is checked **before** platform-native fallbacks, so a
///    developer who exports `XDG_DATA_HOME` globally on Windows will see the
///    XDG path used instead of `%LOCALAPPDATA%`. This is intentional —
///    cross-platform dotfile/portable-install workflows benefit from a
///    single consistent override. (finding M27)
/// 2. Platform fallback:
///    - Windows: `%LOCALAPPDATA%/halter/sessions.db`
///    - Unix:    `$HOME/.local/share/halter/sessions.db`
fn default_db_path() -> Result<PathBuf> {
    if let Some(path) = env::var_os("XDG_DATA_HOME").filter(|value| !value.is_empty()) {
        return Ok(PathBuf::from(path).join("halter").join("sessions.db"));
    }

    #[cfg(windows)]
    {
        let local_app_data = env::var_os("LOCALAPPDATA")
            .filter(|value| !value.is_empty())
            .context("failed to resolve LOCALAPPDATA for sqlite session store")?;
        Ok(PathBuf::from(local_app_data)
            .join("halter")
            .join("sessions.db"))
    }

    #[cfg(not(windows))]
    {
        let home = env::var_os("HOME")
            .filter(|value| !value.is_empty())
            .context("failed to resolve HOME for sqlite session store")?;
        Ok(PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("halter")
            .join("sessions.db"))
    }
}

fn is_constraint_violation(error: &rusqlite::Error) -> bool {
    matches!(
        error,
        rusqlite::Error::SqliteFailure(details, _)
            if details.code == ErrorCode::ConstraintViolation
    )
}

fn is_in_memory_path(path: &Path) -> bool {
    path == Path::new(":memory:")
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::sync::Arc;

    use halter_protocol::{
        InstructionFile, ModelId, PromptSegment, ResourceSnapshot, Revision, SubagentRef,
        SummarySlice,
    };
    use tempfile::tempdir;

    use super::*;

    #[tokio::test]
    async fn open_in_memory_creates_schema() {
        let store = SqliteSessionStore::open(":memory:").expect("open sqlite store");

        assert_eq!(
            store
                .with_conn(|conn| current_version(conn))
                .await
                .expect("schema version"),
            latest_schema_version()
        );
        for table in ["sessions", "snapshots", "events"] {
            let exists = store
                .with_conn(move |conn| table_exists(conn, table))
                .await
                .expect("table exists query");
            assert!(exists, "expected table '{table}' to exist");
        }
    }

    #[tokio::test]
    async fn open_rejects_corrupt_database() {
        let dir = tempdir().expect("tempdir");
        let db_path = dir.path().join("corrupt.db");
        fs::write(&db_path, b"not a sqlite database").expect("write corrupt db");

        let error = match SqliteSessionStore::open(&db_path) {
            Ok(_) => panic!("open should fail"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("integrity"));
    }

    #[tokio::test]
    async fn create_and_load_roundtrip_preserves_session() {
        let store = SqliteSessionStore::open(":memory:").expect("open sqlite store");
        let session = test_session("alpha", "revision-a");

        store
            .create_session(session.clone())
            .await
            .expect("create session");

        let loaded = store
            .load_session(&session.blueprint.session_id)
            .await
            .expect("load session")
            .expect("session exists");

        assert_eq!(loaded.blueprint, session.blueprint);
        assert_eq!(loaded.state, session.state);
        assert_eq!(loaded.snapshot.as_ref(), session.snapshot.as_ref());
    }

    #[tokio::test]
    async fn commit_updates_state_snapshot_and_events() {
        let store = SqliteSessionStore::open(":memory:").expect("open sqlite store");
        let session = test_session("beta", "revision-a");
        let updated_snapshot = Arc::new(test_snapshot("revision-b"));
        let updated_state = test_state("beta-updated");
        let events = vec![
            test_event("event-one", Delivery::Lossless),
            test_event("event-two", Delivery::BestEffort),
        ];

        store
            .create_session(session.clone())
            .await
            .expect("create session");
        let committed = store
            .commit(
                &session.blueprint.session_id,
                Some(Arc::clone(&updated_snapshot)),
                None,
                Some(updated_state.clone()),
                events,
            )
            .await
            .expect("commit session");

        assert_eq!(committed.len(), 2);
        assert_eq!(committed[0].sequence(), 1);
        assert_eq!(committed[1].sequence(), 2);
        assert_eq!(committed[0].session_id, session.blueprint.session_id);
        assert_eq!(committed[1].session_id, session.blueprint.session_id);

        let loaded = store
            .load_session(&session.blueprint.session_id)
            .await
            .expect("load session")
            .expect("session exists");
        assert_eq!(loaded.state, updated_state);
        assert_eq!(loaded.snapshot.as_ref(), updated_snapshot.as_ref());
        assert_eq!(
            loaded.blueprint.snapshot_revision,
            updated_snapshot.revision
        );

        let replayed = store
            .replay(&session.blueprint.session_id)
            .await
            .expect("replay events");
        assert_eq!(replayed, committed);
    }

    #[tokio::test]
    async fn list_sessions_returns_blueprints_without_cross_pollinating_events() {
        let store = SqliteSessionStore::open(":memory:").expect("open sqlite store");
        let first = test_session("first", "revision-a");
        let second = test_session("second", "revision-b");

        store
            .create_session(first.clone())
            .await
            .expect("create first session");
        store
            .create_session(second.clone())
            .await
            .expect("create second session");
        store
            .commit(
                &first.blueprint.session_id,
                None,
                None,
                None,
                vec![test_event("first-only", Delivery::Lossless)],
            )
            .await
            .expect("commit first session events");

        let mut sessions = store.list_sessions().await.expect("list sessions");
        sessions.sort_by(|left, right| left.session_id.0.cmp(&right.session_id.0));
        assert_eq!(
            sessions,
            vec![first.blueprint.clone(), second.blueprint.clone()]
        );

        let first_events = store
            .replay(&first.blueprint.session_id)
            .await
            .expect("replay first");
        let second_events = store
            .replay(&second.blueprint.session_id)
            .await
            .expect("replay second");
        assert_eq!(first_events.len(), 1);
        assert!(second_events.is_empty());
    }

    #[tokio::test]
    async fn snapshots_are_deduplicated_by_revision() {
        let store = SqliteSessionStore::open(":memory:").expect("open sqlite store");
        let shared_snapshot = Arc::new(test_snapshot("shared-revision"));
        let first = test_session_with_snapshot("first", Arc::clone(&shared_snapshot));
        let second = test_session_with_snapshot("second", Arc::clone(&shared_snapshot));

        store
            .create_session(first)
            .await
            .expect("create first session");
        store
            .create_session(second)
            .await
            .expect("create second session");

        let count = store
            .with_conn(|conn| row_count(conn, "snapshots"))
            .await
            .expect("snapshot count");
        assert_eq!(count, 1);
    }

    #[tokio::test]
    async fn create_session_rejects_duplicate_session_id() {
        let store = SqliteSessionStore::open(":memory:").expect("open sqlite store");
        let session = test_session("duplicate", "revision-a");

        store
            .create_session(session.clone())
            .await
            .expect("create session");
        let error = store
            .create_session(session)
            .await
            .expect_err("duplicate create should fail");

        assert!(error.to_string().contains("already exists"));
    }

    #[tokio::test]
    async fn commit_rejects_unknown_session() {
        let store = SqliteSessionStore::open(":memory:").expect("open sqlite store");
        let error = store
            .commit(
                &SessionId::from("missing-session"),
                None,
                None,
                None,
                vec![test_event("missing", Delivery::Lossless)],
            )
            .await
            .expect_err("commit should fail");

        assert!(error.to_string().contains("unknown session"));
    }

    #[tokio::test]
    async fn commit_rejects_stale_expected_state() {
        let store = SqliteSessionStore::open(":memory:").expect("open sqlite store");
        let session = test_session("stale-state", "revision-a");
        let original_state = session.state.clone();

        store
            .create_session(session.clone())
            .await
            .expect("create session");
        store
            .commit(
                &session.blueprint.session_id,
                None,
                Some(original_state.clone()),
                Some(SessionState {
                    pending_warning_messages: vec![halter_protocol::HookWarning {
                        category: "test".to_owned(),
                        message: "updated".to_owned(),
                        ..halter_protocol::HookWarning::default()
                    }],
                    ..SessionState::default()
                }),
                Vec::new(),
            )
            .await
            .expect("commit updated state");

        let error = store
            .commit(
                &session.blueprint.session_id,
                None,
                Some(original_state),
                Some(SessionState::default()),
                Vec::new(),
            )
            .await
            .expect_err("stale commit should fail");

        assert!(error.downcast_ref::<SessionCommitConflict>().is_some());
    }

    #[tokio::test]
    async fn load_and_replay_handle_missing_data() {
        let store = SqliteSessionStore::open(":memory:").expect("open sqlite store");

        assert!(
            store
                .load_session(&SessionId::from("missing"))
                .await
                .expect("load missing")
                .is_none()
        );
        assert!(
            store
                .replay(&SessionId::from("missing"))
                .await
                .expect("replay missing")
                .is_empty()
        );
    }

    #[tokio::test]
    async fn commit_assigns_monotonic_gap_free_sequences_across_calls() {
        let store = SqliteSessionStore::open(":memory:").expect("open sqlite store");
        let session = test_session("sequence", "revision-a");

        store
            .create_session(session.clone())
            .await
            .expect("create session");
        let first = store
            .commit(
                &session.blueprint.session_id,
                None,
                None,
                None,
                vec![
                    test_event("one", Delivery::Lossless),
                    test_event("two", Delivery::Lossless),
                ],
            )
            .await
            .expect("first commit");
        let second = store
            .commit(
                &session.blueprint.session_id,
                None,
                None,
                None,
                vec![test_event("three", Delivery::BestEffort)],
            )
            .await
            .expect("second commit");

        assert_eq!(
            first.iter().map(SessionEvent::sequence).collect::<Vec<_>>(),
            vec![1, 2]
        );
        assert_eq!(
            second
                .iter()
                .map(|event| event.sequence())
                .collect::<Vec<_>>(),
            vec![3]
        );
        let replayed = store
            .replay(&session.blueprint.session_id)
            .await
            .expect("replay session");
        assert_eq!(
            replayed
                .iter()
                .map(|event| event.sequence())
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn run_migrations_applies_fresh_database() {
        let mut conn = Connection::open_in_memory().expect("open in-memory sqlite");

        run_migrations(&mut conn, MIGRATIONS).expect("run migrations");

        assert_eq!(
            current_version(&conn).expect("current version"),
            latest_schema_version()
        );
        assert!(table_exists(&conn, "sessions").expect("sessions table exists"));
        assert!(table_exists(&conn, "snapshots").expect("snapshots table exists"));
        assert!(table_exists(&conn, "events").expect("events table exists"));
    }

    #[test]
    fn run_migrations_is_noop_when_database_is_current() {
        let mut conn = Connection::open_in_memory().expect("open in-memory sqlite");

        run_migrations(&mut conn, MIGRATIONS).expect("run migrations");
        conn.execute(
            "INSERT INTO snapshots (revision, data) VALUES (?1, ?2)",
            params!["revision-a", "{}"],
        )
        .expect("insert snapshot");
        run_migrations(&mut conn, MIGRATIONS).expect("run migrations again");

        assert_eq!(
            current_version(&conn).expect("current version"),
            latest_schema_version()
        );
        assert_eq!(row_count(&conn, "snapshots").expect("row count"), 1);
    }

    #[test]
    fn run_migrations_supports_forward_upgrade_sequences() {
        const V1: &[(u32, &str)] = &[(1, "CREATE TABLE demo (id INTEGER PRIMARY KEY);")];
        const V2: &[(u32, &str)] = &[
            (1, "CREATE TABLE demo (id INTEGER PRIMARY KEY);"),
            (
                2,
                "ALTER TABLE demo ADD COLUMN note TEXT NOT NULL DEFAULT '';",
            ),
        ];

        let mut conn = Connection::open_in_memory().expect("open in-memory sqlite");
        run_migrations(&mut conn, V1).expect("run v1 migrations");
        assert_eq!(current_version(&conn).expect("current version"), 1);

        run_migrations(&mut conn, V2).expect("run v2 migrations");

        assert_eq!(current_version(&conn).expect("current version"), 2);
        assert!(column_exists(&conn, "demo", "note").expect("note column exists"));
    }

    fn latest_schema_version() -> u32 {
        MIGRATIONS.last().map(|(version, _)| *version).unwrap_or(0)
    }

    fn table_exists(conn: &Connection, table: &str) -> Result<bool> {
        let exists = conn
            .query_row(
                "SELECT EXISTS(
                    SELECT 1
                    FROM sqlite_master
                    WHERE type = 'table' AND name = ?1
                )",
                [table],
                |row| row.get::<_, i64>(0),
            )
            .context("failed to query sqlite_master")?;
        Ok(exists == 1)
    }

    fn column_exists(conn: &Connection, table: &str, column: &str) -> Result<bool> {
        let mut statement = conn
            .prepare(&format!("PRAGMA table_info({table})"))
            .context("failed to prepare table_info query")?;
        let rows = statement
            .query_map([], |row| row.get::<_, String>(1))
            .context("failed to query table_info rows")?;

        for row in rows {
            if row? == column {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn row_count(conn: &Connection, table: &str) -> Result<i64> {
        conn.query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
            row.get::<_, i64>(0)
        })
        .with_context(|| format!("failed to count rows in '{table}'"))
    }

    fn test_session(name: &str, revision: &str) -> StoredSession {
        test_session_with_snapshot(name, Arc::new(test_snapshot(revision)))
    }

    fn test_session_with_snapshot(name: &str, snapshot: Arc<ResourceSnapshot>) -> StoredSession {
        let session_id = SessionId::from(format!("session-{name}"));
        let blueprint = SessionBlueprint {
            session_id,
            parent_session_id: None,
            default_model: ModelId::from("default"),
            subagent_model: ModelId::from("subagent"),
            snapshot_revision: snapshot.revision.clone(),
            working_dir: PathBuf::from(format!("/tmp/{name}")),
            system_prompt_seed: vec![PromptSegment {
                id: "seed".into(),
                text: format!("seed-{name}"),
                volatility: halter_protocol::Volatility::SessionStable,
                cache_scope: halter_protocol::CacheScope::PrefixCacheable,
                content_hash: format!("hash-{name}"),
            }],
            max_turns: Some(16),
            subagent_depth: 1,
        };

        StoredSession {
            blueprint,
            state: test_state(name),
            snapshot,
        }
    }

    fn test_state(name: &str) -> SessionState {
        SessionState {
            summaries: vec![SummarySlice {
                id: format!("summary-{name}"),
                text: format!("summary text {name}"),
            }],
            lineage: vec![SubagentRef {
                session_id: SessionId::from(format!("lineage-{name}")),
                task: format!("task-{name}"),
            }],
            ..SessionState::default()
        }
    }

    fn test_snapshot(revision: &str) -> ResourceSnapshot {
        let mut snapshot = ResourceSnapshot::empty();
        snapshot.revision = Revision::from(revision);
        snapshot.instruction_files.push(InstructionFile {
            path: PathBuf::from(format!("{revision}.md")),
            body: format!("instruction body for {revision}"),
        });
        snapshot
    }

    fn test_event(summary: &str, delivery: Delivery) -> PendingEvent {
        PendingEvent::new(
            SessionId::from("ignored"),
            delivery,
            SessionEventPayload::ContextCompacted {
                summary: summary.to_owned(),
            },
        )
    }
}
