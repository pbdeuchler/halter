# halter-runtime

`halter-runtime` is the execution engine for halter sessions.

It is responsible for turning configuration, resources, provider models, tools, hooks, and persistence into an actual running agent session.

If `halter` is the assembly layer, `halter-runtime` is the component that does the work.

---

## Who this crate is for

### Primary: programmers embedding or extending halter

Use this crate when you need to:

- create or resume sessions programmatically
- submit turns and consume runtime events
- manage context compaction
- integrate custom prompt assembly or context strategies
- coordinate subagents
- hot-swap compiled resources
- build a service on top of halter without going through the CLI

### Secondary: advanced CLI users

The CLI is a thin wrapper over this runtime. If you want to understand what actually happens when you run:

- `halter run`
- `halter chat`
- `halter resources`

this crate explains the core machinery.

---

## Public API at a glance

Key exports include:

- `SessionRuntime`
- `HalterSession`
- `SessionInit`
- `RuntimeServices`
- `ResourceHandle`
- `EventBus`
- `ContextSettings`
- `ContextManager`, `DefaultContextManager`
- `PromptAssembler`, `DefaultPromptAssembler`
- hook runtime integration helpers
- `score_message(...)`

These are the building blocks for long-lived, tool-using, event-emitting agent sessions.

---

## Mental model

At a high level, `halter-runtime` owns the session lifecycle.

A typical flow looks like this:

1. create `SessionRuntime`
2. open a new session with `new_session(...)`
3. receive a `HalterSession` handle
4. submit turns with `submit_turn(...)`
5. observe emitted `SessionEvent`s via the event bus and/or session store
6. optionally compact, replay, resume, notify, or shut down the session

`SessionRuntime` is the factory and coordinator.
`HalterSession` is the live handle for one session.

---

## `SessionRuntime`

This is the main runtime object.

Important methods:

- `new`
- `subagent_control`
- `new_session`
- `resume`
- `list_sessions`
- `replace_resources`

### Responsibilities

A `SessionRuntime` bundles the services required to execute sessions:

- model registry / providers
- tool runtime
- hook engine integration
- session persistence
- prompt assembly
- context management
- resource handle
- event publication

In other words, it is the runtime environment in which sessions live.

---

## `SessionInit`

`SessionInit` controls how a session starts.

Fields include:

- `session_id`
- `parent_session_id`
- `working_dir`
- `system_prompt_seed`
- `max_turns`
- `default_model`
- `subagent_model`
- `subagent_depth`

### Typical use

```rust
use halter_runtime::SessionInit;

let init = SessionInit {
    session_id: None,
    parent_session_id: None,
    working_dir: Some(std::env::current_dir()?),
    system_prompt_seed: None,
    max_turns: None,
    default_model: None,
    subagent_model: None,
    subagent_depth: 0,
};
```

A parent session spawning a subagent would set:

- `parent_session_id`
- `subagent_depth`
- often `subagent_model`

---

## Creating a session

Typical sketch:

```rust
use halter_runtime::{SessionInit, SessionRuntime};

async fn start(runtime: &SessionRuntime) -> anyhow::Result<()> {
    let session = runtime.new_session(SessionInit {
        session_id: None,
        parent_session_id: None,
        working_dir: Some(std::env::current_dir()?),
        system_prompt_seed: None,
        max_turns: None,
        default_model: None,
        subagent_model: None,
        subagent_depth: 0,
    }).await?;

    // use session here
    Ok(())
}
```

The top-level `halter` crate wraps this with simpler convenience APIs.

---

## `HalterSession` / `SessionHandle`

`SessionHandle` is the live control surface for a single session.
`HalterSession` is a backwards-compatible alias for `SessionHandle`.

The handle is cheaply cloneable: each clone is an `Arc` bump on the
shared session state. Hooks registered for the session are evicted from
the runtime store only when the *last* clone of the handle is dropped,
not when an arbitrary clone goes out of scope (this is the AC2.1
guarantee — pre-Phase-3 a clone moved into the spawned turn loop would
evict hooks under the still-live caller handle).

Important methods:

- `session_id`
- `submit_turn`
- `replay`
- `shutdown`
- `notify`
- `compact(trigger, custom_instructions)`

### `submit_turn(...)`

This is the main way to drive work.

Conceptually, submitting a turn causes the runtime to:

1. persist the user input/event
2. assemble prompt context
3. invoke provider inference
4. execute tool calls if requested
5. emit session events
6. persist the resulting timeline updates

### `replay()`

Reconstructs the event timeline from persisted state.

This is useful for:

- transcript display
- testing
- UI hydration after reconnect
- postmortem debugging

### `notify(...)`

Injects runtime notifications into the session stream.

### `compact(...)`

Triggers session compaction according to the configured context manager and provider support.

If the underlying provider does not support compaction, you can see an error like:

> `failed to compact session: provider '{}' does not support compaction`

### `shutdown()`

Gracefully stops the session.

---

## Event flow

## `EventBus`

The runtime exposes a publish/subscribe event bus.

Important methods:

- `EventBus::new(capacity)`
- `publish(...)`
- `subscribe()`
- `dropped_events()`

### Why it matters

This lets you separate execution from observation.

You can:

- stream session events to a terminal UI
- feed them into logs or metrics
- power a web frontend
- react to tool starts/stops
- inspect event drops under backpressure

Example sketch:

```rust
use halter_runtime::EventBus;

let bus = EventBus::new(1024);
let mut rx = bus.subscribe();

// elsewhere: bus.publish(event)
// consumer: read events from rx
```

If consumers fall behind, `dropped_events()` provides visibility into loss.

---

## Context management

Large sessions need pruning and compaction.

This crate exports:

- `ContextSettings`
- `ContextManager`
- `DefaultContextManager`
- `score_message(...)`

### What the context manager does

It decides how much of the existing transcript to keep inline when constructing the next model request.

That includes:

- estimating message weight/importance
- honoring compaction thresholds
- selecting candidates for pruning or summarization
- coordinating with provider-backed compaction when available

### `score_message(...)`

This helper supports ranking/prioritization decisions about which messages matter most for context retention.

### Practical implication

If you are building long-lived coding sessions, context management is what keeps them alive without unbounded prompt growth.

---

## Prompt assembly

Prompt assembly is exposed through:

- `PromptAssembler`
- `DefaultPromptAssembler`

The runtime includes an embedded default system prompt sourced from:

- `crates/halter-runtime/prompts/default-system.md`

### What prompt assembly combines

A realistic prompt assembly pass can include:

- built-in system prompt
- `system_prompt_seed` from `SessionInit`
- configured prompt overrides
- compiled resource instructions
- active skills/plugins
- hook-injected system messages
- selected conversation history

This is important because the final model request is not just "the last user message".

---

## Resource hot-swapping

`SessionRuntime::replace_resources(...)` lets you atomically swap the resource handle used by future prompt assembly and runtime lookups.

This is useful when:

- you recompiled repo-local skills/plugins
- a project changed instructions on disk
- you want to reload policy or prompt resources without rebuilding the whole runtime

Existing sessions keep running against the runtime, but new prompt assembly operations use the updated resource set.

---

## Resuming and listing sessions

### `resume(session_id)`

Resumes a previously persisted session.

Use this when you have a durable session store and want continuity across process restarts.

### `list_sessions()`

Returns the session inventory known to the backing store.

This is useful for:

- building admin UIs
- selecting resumable sessions
- cleanup or maintenance workflows

---

## Subagent support

Subagents are not a bolt-on feature. The runtime has first-class support for them.

Key components include:

- `subagent_control()` on `SessionRuntime`
- `parent_session_id` and `subagent_depth` in `SessionInit`
- dedicated subagent modules in the crate

### Why this matters

A parent session can delegate work while preserving:

- lineage
- policy constraints
- event attribution
- model routing
- persistence

Subagent control is also coordinated with the tool layer, where tools like `spawn_agent`, `wait_agent`, `send_input`, and `close_agent` live.

---

## Hooks integration

This runtime is where hooks become operational.

It is responsible for invoking hook dispatch around meaningful lifecycle boundaries like:

- session start/end
- user prompt submission
- tool execution
- permission requests
- compaction
- subagent lifecycle

If you're debugging why a tool call was blocked or why extra system guidance appeared in a prompt, the hook invocation path through `halter-runtime` is usually where to look.

---

## Runtime services

`RuntimeServices` and `ResourceHandle` help package runtime dependencies and the currently active compiled resources.

You will care about these when building a custom embedding, custom runtime composition layer, or resource reload system.

---

## Realistic embedding pattern

A service embedding halter-runtime directly often looks like this architecturally:

1. load config with `halter-config`
2. build providers and model registry with `halter-providers`
3. build tool runtime with `halter-tools`
4. load/compile resources with `halter`
5. create a durable session store with `halter-session`
6. construct `SessionRuntime`
7. create/resume sessions on demand
8. subscribe to event streams for UI/logging

If you don't want to assemble all of that manually, use `halter::Halter`.

---

## Failure modes and operational concerns

### Provider capability mismatch

If your session needs compaction but the provider does not support it, compaction fails.

### Event backpressure

If subscribers are slow, `EventBus` can drop events depending on capacity and downstream behavior.

### Persistence conflicts

If the backing store rejects concurrent commits, the session layer may surface commit conflict errors from `halter-session`.

### Misconfigured subagent depth or model routing

If tool policy and runtime session init disagree about subagent constraints, delegation may fail.

---

## Guidance for advanced users

- Use the default assembler and context manager first; replace them only if you have a concrete reason.
- Subscribe to the event bus early in development. It makes runtime behavior dramatically easier to understand.
- Treat session replay as a first-class debugging tool.
- Use durable session backends if you want resumability across restarts.
- Be explicit about default vs subagent model roles.

---

## Example: building a simple runner abstraction

```rust
use halter_runtime::{SessionInit, SessionRuntime};

pub async fn run_once(runtime: &SessionRuntime, prompt: &str) -> anyhow::Result<()> {
    let session = runtime.new_session(SessionInit {
        session_id: None,
        parent_session_id: None,
        working_dir: Some(std::env::current_dir()?),
        system_prompt_seed: None,
        max_turns: Some(1),
        default_model: None,
        subagent_model: None,
        subagent_depth: 0,
    }).await?;

    session.submit_turn(prompt).await?;
    session.shutdown().await?;
    Ok(())
}
```

This is not the full richness of the runtime, but it shows the core shape.

---

## Related docs

- `../halter/README.md` — high-level builder and convenience layer
- `../halter-session/README.md` — session persistence backing the runtime
- `../halter-tools/README.md` — tool execution surface used inside turns
- `../halter-hooks/README.md` — event interception and policy overlays
- `../halter-providers/README.md` — model backends invoked by the runtime
- `../halter-cli/README.md` — command-line wrapper over this engine
