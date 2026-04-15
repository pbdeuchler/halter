# halter-protocol

`halter-protocol` defines the canonical data model shared across the halter workspace.

If you want to understand what a "session event", "tool call", "assistant chunk", "resource", or "message item" means in halter, this is the crate that defines it.

It contains the structs and enums that let the other crates interoperate.

---

## Who this crate is for

### Primary: programmers integrating multiple halter crates

You will touch `halter-protocol` if you are:

- implementing providers
- implementing tools
- building custom runtimes or event consumers
- persisting transcripts or replaying sessions
- consuming streaming output programmatically
- authoring tests around canonical runtime events

### Secondary: advanced CLI users

CLI users normally interact with these types indirectly through:

- `--streaming-json`
- `--json-result`
- saved transcripts
- tool and assistant event streams

If you're parsing CLI output in another system, this crate defines the structures being emitted.

---

## What this crate contains

This crate is intentionally foundational. It exports the shared vocabulary for:

- models and provider metadata
- prompts and messages
- assistant output items and tool calls
- session events and transcripts
- resources discovered from the filesystem
- hooks, notifications, and runtime bookkeeping
- compaction metadata

It does **not** execute anything by itself.

Think of it as the type-level treaty between the rest of the workspace.

---

## Mental model

At a high level, halter operates on a few core concepts:

### Messages

User, system, assistant, and tool-related content exchanged during a session.

### Items

Structured assistant output components such as text, reasoning, tool calls, or other model-emitted units.

### Events

Immutable session timeline records emitted as the runtime executes.

### Resources

Compiled repo-local assets like skills, plugins, prompts, and instructions.

### Model/provider descriptors

The metadata needed to route requests to providers and interpret capabilities.

`halter-protocol` gives these concepts stable shapes.

---

## Why this matters

Without a shared protocol crate, every layer would invent its own half-compatible representation.

Using a single protocol crate gives you:

- consistent event streaming across runtime and CLI
- stable test fixtures
- provider-independent session storage
- shared tool call structures
- simpler replay and debugging
- less accidental schema drift between crates

---

## Common usage patterns

## Pattern: consume streamed session events

If you are building a wrapper around `halter-runtime` or parsing `halter-cli --streaming-json`, you will typically deserialize newline-delimited protocol events.

Conceptually:

```rust
use halter_protocol::SessionEvent;

fn handle(event: SessionEvent) {
    match event {
        SessionEvent::AssistantMessage { .. } => {
            // render assistant output
        }
        SessionEvent::ToolCallStarted { .. } => {
            // update UI / audit trail
        }
        SessionEvent::ToolCallFinished { .. } => {
            // attach result
        }
        _ => {}
    }
}
```

The exact event variants are defined in this crate and re-exported elsewhere.

---

## Pattern: provider adapters emit canonical assistant items

A provider implementation may receive provider-native streaming chunks from OpenAI, Anthropic, or OpenRouter, but it should normalize them into halter protocol items/events.

That lets downstream consumers stay provider-agnostic.

---

## Pattern: session stores persist protocol events

`halter-session` stores canonical session events rather than provider-specific payloads. That makes replay stable and runtime-independent.

---

## Resource model

The resource-related types in this crate are used by the resource compiler and runtime prompt assembly.

These typically represent things like:

- skills
- plugins
- instructions
- prompts
- metadata attached to compiled resources

This is important because resource discovery happens before runtime execution, but the results still need a portable format the runtime can consume.

---

## Streaming and transcript consumers

If you're writing a UI or automation wrapper, this crate is especially important.

You will likely want to:

- deserialize session events as they are emitted
- distinguish assistant text from tool activity
- group events by session ID / parent session ID
- display notifications and interruptions cleanly
- persist raw protocol events for later replay

Because the rest of the workspace shares these types, you avoid writing one parser for CLI output and another for runtime internals.

---

## Relationship to other crates

### `halter-runtime`

Uses these types to drive the session state machine and event bus.

### `halter-providers`

Normalizes provider-native I/O into protocol requests, responses, chunks, and capability descriptors.

### `halter-tools`

Uses protocol-adjacent data when recording tool call events and results.

### `halter-session`

Persists protocol events and snapshots for replay and durable transcripts.

### `halter-cli`

Serializes protocol events to JSON for streaming output modes.

---

## Practical guidance for integrators

### Prefer protocol types at system boundaries

If you're exposing runtime events over IPC, HTTP, or to another process, use the protocol types directly rather than inventing local wrappers too early.

### Keep provider-specific details at the edge

Translate to halter protocol as soon as possible. That keeps the rest of your system portable.

### Persist canonical events, not rendered text

Rendered text is useful for logs. Canonical events are useful for replay, debugging, and downstream analysis.

### Treat protocol changes as important changes

If you upgrade this crate, re-check:

- streaming consumers
- transcript parsers
- persisted test fixtures
- any custom serialization or filtering code

---

## Example: parsing CLI streaming JSON

A realistic pattern for infrastructure code:

```rust
use halter_protocol::SessionEvent;
use std::io::{self, BufRead};

fn main() -> anyhow::Result<()> {
    for line in io::stdin().lock().lines() {
        let line = line?;
        let event: SessionEvent = serde_json::from_str(&line)?;
        println!("got event: {:?}", event);
    }
    Ok(())
}
```

This works because the CLI's streaming mode emits canonical session events.

---

## Example: storing raw events for later replay

```rust
use halter_protocol::SessionEvent;

fn append_event(buf: &mut Vec<SessionEvent>, event: SessionEvent) {
    buf.push(event);
}
```

That sounds trivial, but it is exactly the point of this crate: the other crates can share one event representation.

---

## Stability expectations

This crate is foundational, so even small changes can have wide blast radius.

When working with it:

- favor additive evolution where possible
- keep serialization semantics explicit
- document variant/field changes carefully
- update tests in downstream crates that rely on replay or streamed JSON

---

## When to read deeper elsewhere

This README explains the role of the protocol crate, but the behavior of specific data flows lives in the higher-level crates:

- for runtime sequencing: `../halter-runtime/README.md`
- for tool behavior: `../halter-tools/README.md`
- for provider transport mapping: `../halter-providers/README.md`
- for durable transcript storage: `../halter-session/README.md`
- for CLI JSON modes: `../halter-cli/README.md`

---

## Summary

`halter-protocol` is the shared schema layer for the entire project.

It is the right abstraction when you need:

- consistent events
- portable transcripts
- provider-independent data structures
- reliable inter-crate contracts

If you are integrating multiple halter crates, you should think in terms of `halter-protocol` types first and crate-specific execution logic second.
