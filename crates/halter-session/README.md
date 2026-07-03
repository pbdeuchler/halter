# halter-session

`halter-session` provides halter's session persistence layer.

It defines the storage contract for session state and transcripts, plus concrete backends for in-memory and SQLite-backed persistence.

If `halter-runtime` is the execution engine, `halter-session` is where session durability and replay come from.

---

## Who this crate is for

### Primary: programmers embedding halter with persistence requirements

Use this crate when you need to:

- create a custom session store backend
- persist sessions across process restarts
- replay transcripts from canonical event history
- list stored sessions for administration or UIs
- choose between volatile memory storage and SQLite persistence

### Secondary: CLI users

CLI users usually interact with this crate indirectly through config:

```toml
[sessions]
backend = "memory"
# or
backend = "sqlite"
sqlite_path = "./.halter/sessions.db"
```

If you want resumable sessions or durable transcripts, this crate is what makes that possible.

---

## Public API at a glance

Core exports:

- `SessionStore`
- `StoredSession`
- `SessionCommitConflict`
- `InMemorySessionStore`
- `SqliteSessionStore` (behind the `sqlite` feature)

This crate intentionally focuses on storage contracts and concrete persistence implementations. It does not run model inference or tools.

---

## Mental model

The append-only event log is the source of truth for a session; the stored
`SessionState` is a **checkpoint** of that log, stamped with the log position
it reflects. A backend must support six responsibilities:

1. create a new stored session (empty log, checkpoint at sequence 0)
2. load an existing stored session (checkpoint plus its log positions)
3. commit new events — and optionally a checkpoint — atomically
4. replay the event stream, in full or after a given sequence
5. list known sessions

That contract is expressed by `SessionStore`. The runtime reproduces current
state as `fold(checkpoint, events after checkpoint)` using
`halter_protocol::fold`, so a checkpoint that lags the log head is closed on
load rather than trusted blindly.

---

## The `SessionStore` trait

Important methods:

- `create_session`
- `load_session`
- `commit`
- `replay`
- `replay_after` (default filters `replay`; backends should push the bound into the query)
- `list_sessions`
- `transcript_path` (default returns `None`)

### Why this trait matters

The runtime can remain backend-agnostic.

Whether events are stored:

- only in memory
- in SQLite on disk
- or in your own custom implementation

`halter-runtime` talks to a stable storage interface.

---

## `StoredSession`

`StoredSession` is the persisted state envelope loaded from a backend: the
blueprint, the resource snapshot, the `SessionState` checkpoint, and two log
positions:

- `state_sequence` — the highest event sequence the checkpoint reflects
- `head_sequence` — the highest committed sequence at load time

Construct new records with `StoredSession::new` (both sequences start at 0;
`create_session` rejects anything else). You will most often encounter it
when:

- resuming a session
- writing your own backend
- diagnosing persistence issues

---

## Commit conflicts

`SessionCommitConflict` exists to prevent silent concurrent state corruption.

Error message:

> `failed to commit session '{session_id}': event log advanced concurrently (expected head {expected}, found {actual})`

This is exactly the right failure mode for multi-writer hazards.

### Conflict semantics

Staleness is decided by the **event-log head**: a commit that supplies
`expected_head_sequence` fails unless it equals the highest committed
sequence at commit time. Every state-changing runtime commit also appends at
least one event, so any concurrent writer moves the head and the loser
conflicts instead of silently clobbering the checkpoint. Events receive
gap-free monotonic sequences starting at the head + 1, and a checkpoint
supplied with the commit is stamped with the post-append head. Both built-in
backends implement this contract, locked in by the shared conformance suite
in `tests/store_conformance.rs` — run it against any custom backend too. The
suite also locks in the fold invariant: replaying the full log through
`halter_protocol::fold` must agree with the checkpoint on every fold-covered
field.

### Practical interpretation

If you see this error, two writers attempted to advance the same session from different assumptions about its latest state.

Your options are usually:

- retry after reloading current state
- serialize access to a given session
- partition responsibilities so only one worker owns a session at a time

---

## In-memory backend

## `InMemorySessionStore`

This is the simplest implementation.

Use it when you want:

- fast local development
- tests
- ephemeral CLI sessions
- no filesystem dependency

Characteristics:

- state exists only for the lifetime of the process
- great for unit/integration tests
- simple semantics
- no cross-process durability

### Event numbering

In-memory commits assign event sequence numbers by taking `max(existing sequence) + 1`,
matching the sqlite backend's `COALESCE(MAX(sequence), 0) + 1` semantics. This keeps
replay deterministic and preserves gap-free monotonicity across commit batches.

### Example

```rust
use halter_session::{InMemorySessionStore, SessionStore};
use std::sync::Arc;

let store: Arc<dyn SessionStore> = Arc::new(InMemorySessionStore::default());
```

This is the default used by many examples and the workspace's default session backend.

---

## SQLite backend

## `SqliteSessionStore`

Available behind the Cargo `sqlite` feature.

Use it when you want:

- sessions to survive process restarts
- replayable transcripts on disk
- a simple embedded durable backend
- easy local admin/inspection with standard SQLite tooling

Important constructors:

- `SqliteSessionStore::open(path)`
- `SqliteSessionStore::open_default()`

### Concurrency model

The database runs in WAL mode. Writes serialize through one writer
connection; reads (`load_session`, `replay`, `list_sessions`) are served by a
small pool of read-only connections, so they proceed concurrently with each
other and with an in-flight write. `:memory:` databases skip the pool and
route reads through the writer connection.

### Migration schema

The backend manages tables including:

- `sessions`
- `snapshots`
- `events`

This gives it the ingredients for:

- session metadata
- durable snapshotting
- append-oriented event replay

### Example

```rust
#[cfg(feature = "sqlite")]
{
    use halter_session::SqliteSessionStore;

    let store = SqliteSessionStore::open("./.halter/sessions.db")?;
}
```

### Config integration

To use SQLite via config, you need both:

```toml
[sessions]
backend = "sqlite"
sqlite_path = "./.halter/sessions.db"
```

and the `sqlite` Cargo feature enabled in the embedding build.

---

## Config validation constraints

The config layer enforces several SQLite-related constraints.

You may see these validation errors:

- `invalid configuration: sessions.sqlite_path requires sessions.backend = 'sqlite'`
- `invalid configuration: sessions.sqlite_path requires the 'sqlite' cargo feature`

This is good behavior. It prevents configurations that appear valid but cannot work at runtime.

---

## Replay model

The `replay(...)` method is important.

It means the system stores canonical event history rather than just the latest assistant text blob.

That enables:

- transcript reconstruction
- runtime debugging
- analytics on tool usage / model output
- session hydration after restart
- auditing and compliance review

A mature agent system should treat replay as a first-class feature, and this crate does.

---

## Listing sessions

`list_sessions()` provides a lightweight inventory of stored sessions.

Use cases:

- admin commands
- dashboards
- cleanup scripts
- resume pickers in a UI
- debugging orphaned or stuck sessions

---

## `transcript_path()`

The trait provides `transcript_path()` with a default of `None`.

Backends that have a meaningful stable transcript location can override it.

This is useful for:

- exposing transcript files to operators
- linking logs or artifacts to a session
- integrating with external archival systems

---

## Realistic usage patterns

## Pattern: ephemeral local dev

Use `InMemorySessionStore`.

Pros:

- zero setup
- fast
- clean reset every run

Cons:

- nothing persists after process exit

---

## Pattern: durable local CLI agent

Use `SqliteSessionStore` with a repo-local or user-local path.

Pros:

- restart-safe
- inspectable with SQLite tooling
- easy to back up or archive

Cons:

- requires the feature and database file management

---

## Pattern: custom backend for services

If you're building a multi-user or multi-process service, you may want to implement `SessionStore` yourself.

Typical reasons:

- central database
- object-store backed transcripts
- custom compliance or retention requirements
- integration with an existing platform data model

### Guidelines for custom backends

- preserve optimistic concurrency behavior
- store canonical events faithfully
- make replay deterministic
- avoid lossy serialization shortcuts
- document ordering guarantees clearly

---

## Example: custom store skeleton

```rust
use async_trait::async_trait;
use halter_session::{SessionStore, StoredSession};

struct MyStore;

#[async_trait]
impl SessionStore for MyStore {
    // implement create_session, load_session, commit, replay, list_sessions
}
```

When writing a real implementation, pay special attention to commit conflict semantics.

---

## Operational advice

- Use memory in tests and local throwaway runs.
- Use SQLite when you care about continuity and replay.
- Do not ignore commit conflicts; they are protecting session integrity.
- If your service can write the same session from multiple workers, design explicit ownership.
- Keep session storage canonical; render human-readable transcripts as a derivative artifact, not the primary record.

---

## Failure modes

### Concurrent writes

This surfaces as `SessionCommitConflict`.

### Misconfigured SQLite

Happens when config enables `sqlite_path` without the right backend/feature combination.

### Partial transcript expectations

If external consumers expect line-oriented text but the store persists structured events, you need a rendering layer. Replay gives you raw truth, not prettified output.

---

## Relationship to the rest of the workspace

### `halter-runtime`

Uses `SessionStore` to create, commit, replay, and resume sessions.

### `halter`

Lets you inject a custom session store with `HalterBuilder::with_session_store(...)`.

### `halter-cli`

Selects session backend through config rather than importing this crate directly.

### `halter-protocol`

Provides the canonical event structures that the session store persists and replays.

---

## Related docs

- `../halter-runtime/README.md` — how sessions are executed and resumed
- `../halter-config/README.md` — session backend configuration rules
- `../halter/README.md` — high-level builder API for injecting custom stores
- `../halter-protocol/README.md` — canonical events and transcript structures
