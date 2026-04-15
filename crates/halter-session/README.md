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

A session store backend must support five responsibilities:

1. create a new stored session
2. load an existing stored session
3. commit new state/events atomically
4. replay the event stream
5. list known sessions

That contract is expressed by `SessionStore`.

---

## The `SessionStore` trait

Important methods:

- `create_session`
- `load_session`
- `commit`
- `replay`
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

`StoredSession` is the persisted state envelope loaded from a backend.

At a high level, it includes the metadata and session snapshot needed to resume and continue execution safely.

You will most often encounter it when:

- resuming a session
- writing your own backend
- diagnosing persistence issues

---

## Commit conflicts

`SessionCommitConflict` exists to prevent silent concurrent state corruption.

Error message:

> `failed to commit session '{session_id}': session state changed concurrently`

This is exactly the right failure mode for multi-writer hazards.

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

In-memory commits assign event sequence numbers based on the current event count.

That makes replay deterministic within the lifetime of the process.

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
