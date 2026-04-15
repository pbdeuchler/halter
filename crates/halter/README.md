# halter

`halter` is the high-level SDK crate for assembling and running a halter harness.

If you are embedding halter into another Rust application, this is the crate you generally want to start with. It wires together:

- configuration from `halter-config`
- resource loading and compilation
- model/provider registration
- session runtime setup
- tool registration and policy enforcement
- session persistence
- plugin and SDK hook registration

The CLI in `crates/halter-cli` is a thin wrapper over this crate.

---

## Who this crate is for

### Primary: programmers embedding halter

Use `halter` when you want a single entry point for:

- loading a `halter.toml`
- compiling skills/plugins/hooks into a `ResourceSnapshot`
- constructing a `SessionRuntime`
- opening sessions and submitting turns
- swapping resources at runtime
- injecting custom tools, hooks, or session stores

### Secondary: CLI-oriented users

You typically do **not** use this crate directly from the shell. The `halter` binary builds a `Halter` value internally, then uses it to run `chat`, `run`, `resources`, and `validate` workflows.

If you are trying to operate halter rather than embed it, read:

- `../halter-cli/README.md`
- `../halter-config/README.md`
- `../halter-tools/README.md`

---

## What this crate exports

The public surface is intentionally small:

- `HalterBuilder`
- `Halter`
- resource-loading types such as `ResourceCompiler`, `PluginLoader`, `SkillLoader`, `CompiledResources`
- a `session` module re-exporting session store types
- a `prelude` module for the most common SDK types

```rust
use halter::prelude::*;
```

The prelude currently re-exports:

- `HarnessConfig`
- `Message`, `ResourceSnapshot`, `SessionEvent`, `SessionEventPayload`, `SessionId`, `Turn`
- `SessionInit`, `SessionRuntime`
- `Halter`, `HalterBuilder`, `PluginLoader`, `ResourceCompiler`, `SkillLoader`

---

## Mental model

At a high level, `halter` does four things:

1. **Load config**: parse and validate a `HarnessConfig`
2. **Compile resources**: turn skills/plugins/hooks into a `ResourceSnapshot`
3. **Build services**: tools, providers, policy, context manager, event bus, session store
4. **Run sessions**: create `HalterSession` values and stream `SessionEvent`s

The dependency graph looks roughly like this:

```text
HarnessConfig
    + ResourceSnapshot / CompiledResources
    + registered hooks
    + tools
    + session store
        ↓
     HalterBuilder
        ↓
       Halter
        ↓
   SessionRuntime
        ↓
    HalterSession
```

---

## Quick start

### From a config file

This is the simplest path and mirrors what the CLI does.

```rust
use futures::StreamExt;
use halter::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let harness = Halter::from_config_file("halter.toml").await?;
    let session = harness.new_session(SessionInit::default()).await?;

    let mut events = session.submit_turn(Turn::user("Summarize this repository")).await?;
    while let Some(event) = events.next().await {
        let event = event?;
        println!("{:?}", event.payload);
    }

    Ok(())
}
```

### From an in-memory config and snapshot

Use this path when you already control configuration and resource compilation.

```rust
use halter::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = HarnessConfig::default();
    let snapshot = ResourceSnapshot::empty();

    let harness = Halter::from_config(config, snapshot).await?;
    let _session = harness.new_session(SessionInit::default()).await?;
    Ok(())
}
```

That example is structurally correct, but in practice `HarnessConfig::default()` is incomplete because `[models.default]` is required. For real usage, either load a config file or build a fully populated `HarnessConfig`.

---

## The easiest constructor paths

### `Halter::from_config_file(path)`

This is the highest-level API.

It performs all of the following:

- loads TOML via `halter_config::load_path`
- validates configuration
- compiles skills/plugins/hooks with `ResourceCompiler::from_config(&config).compile()`
- builds the runtime with providers, built-in tools, subagent tools, policy, and session storage

Use it when you want conventional halter behavior.

### `Halter::from_compiled_resources(config, resources)`

Use this when:

- you already precompiled resources
- you want to cache snapshots between process launches
- you want to hot-reload resources yourself

### `Halter::from_config(config, snapshot)`

Use this when you have a raw `ResourceSnapshot` but no compiled hook bundle.

This works, but if you want plugin hooks and hook warnings preserved, prefer `CompiledResources` over a naked snapshot.

---

## `HalterBuilder`

`HalterBuilder` is the composition API for advanced embedding.

Available builder methods include:

- `with_config(config)`
- `with_resource_snapshot(snapshot)`
- `with_compiled_resources(resources)`
- `with_loaded_skills(skills)`
- `with_loaded_plugins(plugins)`
- `with_plugin_hook(plugin_id, hook)`
- `with_plugin_hook_priority(plugin_id, priority, hook)`
- `with_tool(tool)`
- `with_session_store(store)`
- `build().await`

### Example: build from config + compiled resources

```rust
use halter::{HalterBuilder, ResourceCompiler};
use halter_config::load_path;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = load_path("halter.toml").await?;
    let resources = ResourceCompiler::from_config(&config).compile().await?;

    let harness = HalterBuilder::new()
        .with_config(config)
        .with_compiled_resources(resources)
        .build()
        .await?;

    println!("default model = {:?}", harness.config().default_model()?.model);
    Ok(())
}
```

### Example: add a custom tool

```rust
use std::sync::Arc;

use async_trait::async_trait;
use halter::HalterBuilder;
use halter_config::load_path;
use halter_protocol::{ToolCapabilities, ToolConcurrency, ToolName, ToolResult, ToolSpec};
use halter_tools::{Tool, ToolContext};
use serde_json::{json, Value};

#[derive(Debug)]
struct EchoTool;

#[async_trait]
impl Tool for EchoTool {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("echo_json"),
            description: "Return the input JSON unchanged".to_owned(),
            input_schema: json!({
                "type": "object",
                "additionalProperties": true
            }),
            concurrency: ToolConcurrency::ParallelSafe,
            capabilities: ToolCapabilities::default(),
            provider_aliases: Default::default(),
        }
    }

    async fn execute(&self, _context: ToolContext, input: Value) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::Json { value: input })
    }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = load_path("halter.toml").await?;
    let resources = halter::ResourceCompiler::from_config(&config).compile().await?;

    let _halter = HalterBuilder::new()
        .with_config(config)
        .with_compiled_resources(resources)
        .with_tool(Arc::new(EchoTool))
        .build()
        .await?;

    Ok(())
}
```

### Example: provide your own session store

```rust
use std::sync::Arc;

use halter::{HalterBuilder, ResourceCompiler};
use halter::session::InMemorySessionStore;
use halter_config::load_path;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = load_path("halter.toml").await?;
    let resources = ResourceCompiler::from_config(&config).compile().await?;
    let store = Arc::new(InMemorySessionStore::default());

    let _halter = HalterBuilder::new()
        .with_config(config)
        .with_compiled_resources(resources)
        .with_session_store(store)
        .build()
        .await?;

    Ok(())
}
```

---

## Resource compilation

The `ResourceCompiler` is how this crate turns on-disk skills/plugins/hooks into a `CompiledResources` bundle.

```rust
use halter::ResourceCompiler;
use halter_config::load_path;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = load_path("halter.toml").await?;
    let compiled = ResourceCompiler::from_config(&config).compile().await?;

    println!("revision = {}", compiled.snapshot.revision);
    println!("skills   = {}", compiled.snapshot.skills.len());
    println!("agents   = {}", compiled.snapshot.agents.len());
    println!("plugins  = {}", compiled.snapshot.plugins.len());
    Ok(())
}
```

### Skill loading

A skill is loaded from a directory containing `SKILL.md`.

The loader:

- recursively searches configured roots
- treats any directory with `SKILL.md` as a skill root
- reads optional frontmatter fields like `name` and `description`
- also records executables under `scripts/`

Minimal skill example:

```text
.agent/skills/review/
└── SKILL.md
```

Example `SKILL.md`:

```markdown
---
name: review
description: Review a patch for correctness and maintainability
---

# Review

When asked to review code, focus on correctness first, then risk, then style.
```

### Plugin loading

Plugins can be discovered from several manifest locations:

- `.claude-plugin/plugin.json`
- `.agent-plugin/plugin.json`
- `.halter-plugin/plugin.json`
- `plugin.json`

The manifest can declare:

- `skills`
- `agents`
- `hooks`
- `mcpServers`
- `lspServers`
- `allowedHttpHosts`
- `allowedEnvVars`

Paths are constrained to stay within the plugin root. Relative paths must start with `./`, and path traversal outside the plugin root is rejected.

Example plugin manifest:

```json
{
  "name": "example",
  "version": "0.1.0",
  "skills": ["./skills"],
  "agents": ["./agents"],
  "hooks": "./hooks/hooks.json",
  "allowedHttpHosts": ["api.example.com"],
  "allowedEnvVars": ["EXAMPLE_TOKEN"]
}
```

### Agents

Agent prompts are loaded from files referenced by the plugin manifest. The runtime exposes them as named subagent roles in the resource snapshot, and `spawn_agent` can select them via `agent_type`.

### Hooks

Compiled plugin hooks become a `Hooks` registry in `CompiledResources`, plus a list of `HookWarning`s that the runtime can surface to sessions.

---

## Hot reloading resources

A built `Halter` can swap in a new `CompiledResources` bundle without rebuilding the whole runtime.

```rust
use halter::{Halter, ResourceCompiler};
use halter_config::load_path;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = load_path("halter.toml").await?;
    let initial = ResourceCompiler::from_config(&config).compile().await?;
    let harness = Halter::from_compiled_resources(config.clone(), initial).await?;

    // Later, after files change:
    let updated = ResourceCompiler::from_config(&config).compile().await?;
    harness.replace_resources(updated);

    Ok(())
}
```

What this changes:

- the active `ResourceSnapshot`
- the compiled plugin hook registry
- accumulated hook warnings

What this does **not** do:

- rebuild the whole `Halter` object
- replace the session store
- change the already-constructed config

---

## Session lifecycle through this crate

Once you have a `Halter`, session creation is straightforward:

```rust
use futures::StreamExt;
use halter::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let harness = Halter::from_config_file("halter.toml").await?;

    let session = harness
        .new_session(SessionInit {
            working_dir: std::env::current_dir()?,
            ..SessionInit::default()
        })
        .await?;

    let mut stream = session.submit_turn(Turn::user("List the major crates in this repo")).await?;
    while let Some(event) = stream.next().await {
        let event = event?;
        match event.payload {
            SessionEventPayload::DeltaItem { delta } => print!("{}", delta.text),
            SessionEventPayload::TurnCompleted { usage, .. } => {
                println!("\nusage: in={} out={}", usage.input_tokens, usage.output_tokens);
            }
            SessionEventPayload::TurnFailed { error, .. } => {
                eprintln!("turn failed: {error}");
            }
            _ => {}
        }
    }

    Ok(())
}
```

For lower-level control, use `halter.runtime()` and work directly with `SessionRuntime`.

---

## Hook registration from the SDK

You can layer SDK-defined hooks on top of plugin hooks using `with_plugin_hook` or `with_plugin_hook_priority`.

```rust
use halter::HalterBuilder;
use halter_hooks::{Hook, HookEventName, HookResponse, RegisteredHookPriority};
use halter_protocol::PluginId;

let hook = Hook::callback(HookEventName::PreToolUse, |input| async move {
    if input.tool_name() == Some("shell") {
        Ok(HookResponse::passthrough().with_system_message(
            "Shell usage was requested; verify the command is minimal and necessary.",
        ))
    } else {
        Ok(HookResponse::passthrough())
    }
});

let _builder = HalterBuilder::new()
    .with_plugin_hook_priority(
        PluginId::from("sdk-observer"),
        RegisteredHookPriority::BeforePlugins,
        hook,
    );
```

See `../halter-hooks/README.md` for the hook model itself.

---

## Feature flags

This crate forwards a set of workspace-wide capabilities:

- `advanced-tools`
- `ast-tools`
- `image-tools`
- `pty`
- `profiling`
- `full` = enable all tool-related features above
- `sqlite` = enable SQLite session persistence support

Examples:

```bash
cargo add halter --features full,sqlite
```

or in `Cargo.toml`:

```toml
halter = { path = "../crates/halter", features = ["full", "sqlite"] }
```

---

## Errors and builder constraints

Two builder constraints are especially important:

1. You must supply a valid `HarnessConfig`
2. You must supply resources in one of these forms:
   - `with_resource_snapshot(...)`
   - `with_compiled_resources(...)`
   - `with_loaded_skills(...)` and/or `with_loaded_plugins(...)`

You **cannot** combine:

- a prebuilt `resource_snapshot`
- with separately supplied `loaded_skills` or `loaded_plugins`

That combination is rejected during `build()`.

---

## Relationship to the rest of the workspace

Use this crate when you want the batteries-included assembly layer.

Use lower-level crates when you need finer control:

- `halter-runtime`: direct session/runtime engine APIs
- `halter-tools`: tools, tool runtime, policy
- `halter-config`: config schema, loading, env overrides, starter config generation
- `halter-providers`: provider transports and model registry
- `halter-session`: persistence backends
- `halter-hooks`: hook registry, merging, SDK hook registration
- `halter-protocol`: shared wire/domain types
- `halter-cli`: end-user binary

---

## Practical recommendations

- Start with `Halter::from_config_file(...)` unless you have a reason not to.
- Prefer `CompiledResources` over a raw `ResourceSnapshot` when hooks matter.
- Use `HalterBuilder` when you need to inject custom tools, hooks, or persistence.
- If you want runtime reloads, keep the original config and call `replace_resources(...)` with a fresh compilation.
- If you need SQLite persistence, enable the `sqlite` feature and configure sessions explicitly.

---

## See also

- `../halter-cli/README.md`
- `../halter-config/README.md`
- `../halter-runtime/README.md`
- `../halter-tools/README.md`
- `../halter-hooks/README.md`
- `../halter-session/README.md`
- `../halter-protocol/README.md`
