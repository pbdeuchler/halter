# halter

`halter` is a **local-first agent runtime, protocol, and tool harness** for building and operating serious tool-using agents.

This repository is organized as a Rust workspace with two primary audiences:

- **Programmers embedding halter** as an SDK and extending its runtime, tools, hooks, providers, and persistence
- **Users/operators running the CLI** to execute tasks, inspect resources, validate config, and interact with sessions

The SDK story is the primary one. The CLI is a thin, practical wrapper around that SDK.

---

## What halter gives you

At a high level, halter combines:

- a typed **protocol** for sessions, turns, events, resources, and tool calls
- a configurable **runtime** for prompt assembly, context management, provider execution, hooks, and subagents
- a built-in **tool harness** for reading, editing, writing, shell execution, process control, and delegated work
- **resource loading** for repo-local skills and plugins
- **policy enforcement** around filesystem writes, shell usage, tool output size, and subagent fanout
- **session persistence** with memory and SQLite backends
- a usable **CLI** for day-to-day workflows

If you're familiar with agentic coding systems, halter is the substrate that lets you build one cleanly rather than re-deriving the same runtime and tool patterns from scratch.

---

## Audience guide

### If you are embedding halter in Rust

Start here, then read these in order:

1. `crates/halter/README.md`
2. `crates/halter-config/README.md`
3. `crates/halter-runtime/README.md`
4. `crates/halter-tools/README.md`
5. whichever subsystem crate you need next

### If you are using the CLI

Start here, then read:

1. `crates/halter-cli/README.md`
2. `crates/halter-config/README.md`
3. `crates/halter-tools/README.md`
4. optionally `crates/halter-hooks/README.md` if you care about policy interception

---

## Workspace layout

This repository contains these crates:

- `crates/halter` — high-level SDK and builder
- `crates/halter-cli` — command-line entrypoint
- `crates/halter-config` — config schema, loading, overrides, validation
- `crates/halter-protocol` — shared types and wire-format vocabulary
- `crates/halter-runtime` — session engine, prompt assembly, event bus, compaction, subagents
- `crates/halter-providers` — provider adapters and model registry
- `crates/halter-tools` — tool runtime, built-in tools, policy, subagent control tools
- `crates/halter-hooks` — event-driven hook and policy interception layer
- `crates/halter-session` — session persistence and replay

A useful mental model is:

```text
halter-config      -> load/validate config
halter-protocol    -> shared data model
halter-providers   -> model backends
halter-tools       -> tools + policy
halter-hooks       -> interception + approvals + annotations
halter-session     -> persistence + replay
halter-runtime     -> session execution engine
halter             -> high-level assembly layer
halter-cli         -> user-facing binary
```

---

## Two usage modes

## 1) SDK / embedding mode

You embed halter into your own Rust program and use it as an agent runtime.

Typical responsibilities:

- loading config
- compiling resources
- injecting custom tools or hooks
- selecting persistence strategy
- consuming session events programmatically

## 2) CLI / operator mode

You create a `halter.toml`, point the CLI at a repo or environment, and run tasks.

Typical responsibilities:

- managing credentials
- choosing enabled tools and shell allowlists
- inspecting loaded skills/plugins
- capturing JSON output for automation
- tuning policy and compaction thresholds

---

## Quick start for CLI users

## 1. Create a config

```bash
cargo run -p halter-cli -- init
```

This writes a starter `halter.toml`.

You can also inspect the example config in:

- `examples/halter.example.toml`

---

## 2. Set credentials

At minimum, configure the API key for the provider used by `[models.default]`.

Examples:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export OPENROUTER_API_KEY=...
```

Which one you need depends on your config.

---

## 3. Validate config and runtime prerequisites

```bash
cargo run -p halter-cli -- validate
```

This checks more than TOML syntax. It also checks things like:

- `version = 1`
- `[models.default]` exists
- selected providers have credentials available
- context thresholds are coherent
- session backend settings are valid

---

## 4. Inspect discovered resources

```bash
cargo run -p halter-cli -- resources
```

Use this to verify that your skill and plugin roots are being discovered and compiled the way you expect.

---

## 5. Run a task

```bash
cargo run -p halter-cli -- run "Summarize this repository's architecture"
```

By default, `run` emits the final assistant result as JSON.

To stream raw session events instead:

```bash
cargo run -p halter-cli -- run --streaming-json "Summarize this repository's architecture"
```

To write output and tracing to one file:

```bash
cargo run -p halter-cli -- \
  --output-file out.jsonl \
  run --streaming-json "Summarize this repository's architecture"
```

---

## 6. Use interactive mode

```bash
cargo run -p halter-cli -- chat
```

This opens a REPL-style interactive session backed by the same runtime and config.

---

## Quick start for SDK users

The simplest path is to use the high-level `halter` crate.

```rust
use futures::StreamExt;
use halter::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let harness = Halter::from_config_file("halter.toml").await?;
    let session = harness.new_session(SessionInit::default()).await?;

    let mut events = session
        .submit_turn(Turn::user("Summarize the session persistence design"))
        .await?;

    while let Some(event) = events.next().await {
        let event = event?;
        println!("{:?}", event.payload);
    }

    Ok(())
}
```

That flow does all of the following:

- loads and validates config
- compiles resources
- builds providers, tools, hooks, policy, and session storage
- creates a runtime
- creates a session
- executes one turn and streams the resulting events

For a deeper walkthrough, read `crates/halter/README.md`.

---

## A realistic config

This is the central contract for both SDK and CLI usage.

```toml
version = 1

[models.default]
provider = "openai"
model = "gpt-5.4"
reasoning = "high"

[models.subagent]
provider = "openai"
model = "gpt-5.4-mini"
reasoning = "medium"

[resources.skills]
roots = ["./.agent/skills"]

[resources.plugins]
roots = ["./.agent/plugins"]

[tools]
enabled = [
  "read",
  "glob",
  "grep",
  "write",
  "edit",
  "shell",
  "process",
  "wait_agent",
  "spawn_agent",
  "send_input",
  "close_agent",
]

[context]
compaction_threshold = 200_000
pre_compaction_target = 150_000
prune_signal_threshold = "low"

[policy]
allowed_write_roots = ["./", "/tmp/halter"]
max_read_bytes = 1048576
max_tool_output_bytes = 262144
max_subagent_depth = 3
max_concurrent_subagents = 8

[policy.shell]
enabled = true
allow = ["git", "cargo", "rg", "ls", "find", "python", "pwd", "cwd", "echo"]
timeout_secs = 30

[sessions]
backend = "memory"
```

You can derive this from `examples/halter.example.toml` and tailor it to your environment.

---

## How the system is structured

## Configuration layer

`halter-config` defines the schema for:

- providers
- model roles (`default`, `small`, `subagent`)
- resource roots
- prompts
- context compaction settings
- tool enablement
- policy
- session persistence
- runtime settings

It also handles:

- file loading
- environment overrides
- layered merges
- JSON Schema export
- starter config generation

If you care about what is valid in `halter.toml`, read `crates/halter-config/README.md`.

---

## Protocol layer

`halter-protocol` defines the shared vocabulary used by the rest of the workspace.

That includes types for:

- turns
- messages
- session events
- tool calls and tool results
- resources and compiled artifacts
- provider-facing request/response chunks

If you are building integrations or parsing structured output, this crate matters a lot.

---

## Provider layer

`halter-providers` adapts concrete model backends into halter's normalized provider interface.

Built-in providers include:

- OpenAI
- Anthropic
- OpenRouter
- Fake/test provider
- Unsupported placeholder for builds where a transport is not wired in

Important operational differences:

- OpenAI supports compaction
- OpenRouter does not support compaction
- Anthropic currently advertises no streaming and no compaction
- capability differences are explicit and should be handled intentionally

---

## Tool layer

`halter-tools` is what makes the agent do real work in the local environment.

Built-in tools include:

- `read`
- `glob`
- `grep`
- `write`
- `edit`
- `shell`
- `process`

Optional feature-gated tools include:

- `pty`
- `ast_grep`
- `image`
- `profile`

Subagent tools include:

- `spawn_agent`
- `send_input`
- `wait_agent`
- `close_agent`

This crate also enforces policy boundaries such as:

- shell allowlisting
- write-root restrictions
- read/output size limits
- subagent depth and concurrency limits

If you are a CLI user, this is one of the most important per-crate READMEs to read.

---

## Hook layer

`halter-hooks` lets you observe and influence runtime behavior by reacting to lifecycle events.

Hooks can:

- approve or block actions
- request or deny permissions
- add system messages
- attach additional context
- rewrite inputs and outputs
- suppress output visibility
- stop execution

This is where you implement runtime policy that is more semantic than the hard mechanical policy enforced by the tool layer.

---

## Session layer

`halter-session` provides persistence and replay.

Built-in backends:

- `InMemorySessionStore`
- `SqliteSessionStore` (behind the `sqlite` feature)

Use memory for:

- tests
- ephemeral local runs
- simplest setup

Use SQLite for:

- resumable local agents
- durable transcripts
- replay after process restart

---

## Runtime layer

`halter-runtime` executes sessions.

It owns:

- session lifecycle
- prompt assembly
- context management and compaction
- event publication
- hook dispatch
- tool execution orchestration
- subagent lineage and coordination
- session replay/resume

If you want to understand what really happens after `submit_turn(...)`, this is the crate to read.

---

## High-level assembly layer

`halter` is the convenience layer that builds the whole runtime from config and resources.

Key types:

- `Halter`
- `HalterBuilder`
- `ResourceCompiler`
- `PluginLoader`
- `SkillLoader`

Use `Halter` unless you have a good reason to assemble the lower-level crates manually.

---

## CLI layer

`halter-cli` exposes a practical command surface:

- `halter init`
- `halter validate`
- `halter resources`
- `halter run`
- `halter chat`
- `halter config schema`

It is intentionally thin. Reading its `README.md` is useful both for users and for programmers who want a reference implementation of how to wire the SDK together.

---

## Typical workflows

## Workflow: local coding-agent usage from the CLI

1. create `halter.toml`
2. set provider credentials
3. enable the tools you want
4. constrain shell allowlists and write roots
5. run a task with `halter run`
6. switch to `halter chat` for interactive refinement
7. use `--streaming-json` or `--output-file` for automation

Example:

```bash
halter validate
halter resources
halter run "Review the staged diff and summarize risk"
halter run --streaming-json "Refactor the config loader and explain changes"
```

---

## Workflow: repository-aware agent with local skills/plugins

1. put skills under `.agent/skills`
2. put plugins under `.agent/plugins`
3. point `resources.skills.roots` and `resources.plugins.roots` at them
4. use `halter resources` to confirm they load
5. run tasks that rely on those instructions and hooks

This is where halter starts to feel like a real harness rather than just a thin model wrapper.

---

## Workflow: embedded application using halter as a library

1. load config with `Halter::from_config_file(...)` or manually with `halter-config`
2. compile resources
3. optionally inject custom tools or session stores with `HalterBuilder`
4. create sessions
5. submit turns and consume streamed events
6. persist or replay sessions as needed

---

## Workflow: delegated parallel work

1. enable subagent tools in config
2. set sensible `max_subagent_depth` and `max_concurrent_subagents`
3. let a parent session use `spawn_agent`
4. synchronize with `wait_agent`
5. use `send_input` for corrections and `close_agent` for cleanup

This is especially useful for:

- codebase documentation
- per-crate analysis
- parallel test or failure triage
- structured audits across multiple subsystems

---

## CLI reference

All CLI commands accept:

- `--config <CONFIG>` (default: `halter.toml`)
- `--output-file <OUTPUT_FILE>`

### `halter init`

Generate a starter config.

```bash
halter init
```

### `halter validate`

Validate config and runtime prerequisites.

```bash
halter validate
```

### `halter resources`

Compile and summarize resources.

```bash
halter resources
```

### `halter run`

Run one task in a fresh session.

```bash
halter run "Summarize this repository"
```

Useful flags:

- `--json-result` — final answer as JSON
- `--streaming-json` — newline-delimited `SessionEvent` JSON

### `halter chat`

Open an interactive chat loop.

```bash
halter chat
```

### `halter config schema`

Print the JSON Schema for `halter.toml`.

```bash
halter config schema
```

---

## SDK reference

The most common entrypoints for library users are:

- `Halter::from_config_file(...)`
- `Halter::from_config(...)`
- `Halter::from_compiled_resources(...)`
- `Halter::new_session(...)`
- `Halter::replace_resources(...)`
- `HalterBuilder`

A realistic advanced composition path:

```rust
use std::sync::Arc;

use halter::{HalterBuilder, ResourceCompiler};
use halter_config::load_path;
use halter_session::InMemorySessionStore;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = load_path("halter.toml").await?;
    let resources = ResourceCompiler::from_config(&config).compile().await?;

    let harness = HalterBuilder::new()
        .with_config(config)
        .with_compiled_resources(resources)
        .with_session_store(Arc::new(InMemorySessionStore::default()))
        .build()
        .await?;

    let session = harness.new_session(halter_runtime::SessionInit::default()).await?;
    let _events = session
        .submit_turn(halter_protocol::Turn::user("Describe the active runtime"))
        .await?;

    Ok(())
}
```

For deeper examples, use the crate-specific READMEs.

---

## Policy and safety model

Halter's model is deliberately layered.

### Hard boundaries

Enforced mechanically by tool policy:

- where writes may occur
- which shell programs may run
- how much can be read or emitted
- how many subagents may be active
- how deep delegation may go

### Semantic/runtime boundaries

Enforced or influenced by hooks:

- approvals
- denials
- stop conditions
- input/output rewriting
- extra context or warnings
- audit annotations

This is a good design because it keeps non-negotiable constraints in the tool layer while leaving richer workflow policy to hooks.

---

## Feature flags

Across the workspace, common optional features include:

- `advanced-tools`
- `ast-tools`
- `image-tools`
- `pty`
- `profiling`
- `full`
- `sqlite`

Practical rules:

- a feature-gated tool must be compiled in before it can be enabled in config
- `sqlite` must be enabled if you want SQLite-backed sessions
- `full` is the easiest way to turn on the broadest tool set

Example install:

```bash
cargo install --path crates/halter-cli --features full,sqlite
```

---

## Environment overrides

The config crate supports a focused set of environment overrides, including:

- `HALTER_SESSION_BACKEND`
- `HALTER_POLICY_SHELL_ENABLED`
- `HALTER_POLICY_NETWORK_ENABLED`
- `HALTER_SKILL_ROOTS`
- `HALTER_PLUGIN_ROOTS`
- `HALTER_POLICY_SHELL_ALLOW`
- `HALTER_POLICY_ALLOWED_HOSTS`
- `HALTER_TOOLS_ENABLED`

These are useful for CI, local overrides, or environment-specific deployment adjustments without duplicating full config files.

---

## Common mistakes

### 1. Supplying an incomplete config

`[models.default]` is required. A config without it is not valid.

### 2. Forgetting provider credentials

If your config references OpenAI, Anthropic, or OpenRouter, the corresponding credentials must be available either in config or environment.

### 3. Assuming every provider supports streaming or compaction

Those are provider capabilities, not universal guarantees.

### 4. Enabling tools in config without compiling their features

For example, `pty` or `ast_grep` must exist in the build before config can make them useful.

### 5. Shell allowlists that are too narrow or too broad

Too narrow makes the agent ineffective. Too broad expands risk. Tune deliberately.

### 6. Using `write` where `edit` is safer

For small source changes, targeted edits are generally easier to reason about and verify.

### 7. Ignoring subagent limits

If you rely on delegated work, configure depth and concurrency intentionally.

---

## Recommendations

### For SDK users

- Start with `halter::Halter` before dropping to lower-level crates.
- Use `InMemorySessionStore` in tests and SQLite only when you need durability.
- Subscribe to or persist session events early; they are invaluable for debugging.
- Treat provider capabilities as explicit contracts.
- Add custom tools only when built-ins do not cover the need.

### For CLI users

- Keep `halter.toml` explicit and small.
- Use `halter validate` and `halter resources` before blaming runtime behavior.
- Treat `tools.enabled` and shell allowlists as important operational policy.
- Use `--streaming-json` when integrating with other systems.
- Use `--output-file` when you want a single artifact containing output and tracing.

---

## Per-crate documentation

Detailed READMEs have been written for every crate in this workspace:

- [`crates/halter/README.md`](crates/halter/README.md)
- [`crates/halter-cli/README.md`](crates/halter-cli/README.md)
- [`crates/halter-config/README.md`](crates/halter-config/README.md)
- [`crates/halter-hooks/README.md`](crates/halter-hooks/README.md)
- [`crates/halter-protocol/README.md`](crates/halter-protocol/README.md)
- [`crates/halter-providers/README.md`](crates/halter-providers/README.md)
- [`crates/halter-runtime/README.md`](crates/halter-runtime/README.md)
- [`crates/halter-session/README.md`](crates/halter-session/README.md)
- [`crates/halter-tools/README.md`](crates/halter-tools/README.md)

If you want one takeaway from this repo structure, it is this:

- use **`halter`** if you want the easiest embedding path
- use **`halter-cli`** if you want the easiest operator path
- use the lower-level crates when you need deeper control

---

## Build notes

Workspace metadata currently targets:

- Rust edition: `2024`
- Rust version: `1.93`

Standard development entrypoints:

```bash
cargo run -p halter-cli -- --help
cargo run -p halter-cli -- validate
cargo test
```

---

## Final summary

Halter is not just a model wrapper. It is a composable agent harness with:

- a real runtime
- a real tool system
- explicit policy boundaries
- pluggable providers
- durable session semantics
- hookable lifecycle events
- both SDK and CLI entrypoints

If you're building or operating serious local-first agents, that's the right level of abstraction.
