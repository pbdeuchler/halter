# halter

`halter` is a **simple and configurable agent harness and SDK** for building and operating thoroughbred agents.

> [!CAUTION]
> `halter` is still a heavy work in progress. Proceed at your own risk.

> [!TIP]
> `halter` is explictly designed for long running, multi model, and dynamic workflows. If you have a preferred model family you like to use, don't need to spin up agents dynamically, or keep your workflow or agent setup local then `halter` is probably not for you

## Design Goals

- Cache Friendliness
- Obsessive token optimization
- Best in class multi model support
- Best in class tool calling and hook support

## Tradeoffs

- halter implements it's own compaction strategy. This _can be_ (but is not always) less token effecient than the managed compaction functionality offered by inference providers. The goal of the custom compaction is to result in a _higher quality_ context window, hopefully reducing overall token use throughout the turn. This also allows halter to provide a consistent, baseline experience regardless of which inference provider or model is used.
- There are no plans for halter to implement MCP. It's a bad, poorly designed protocol that serves little to no purpose. If you absolutely need MCP like functionality you can provide it with either skills or custom tools.

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

## Quick start for CLI users

### 1. Create a config

```bash
cargo run -p halter-cli -- init
```

This writes a starter `halter.toml`.

You can also inspect the example config in:

- `examples/halter.example.toml`

---

### 2. Set credentials

At minimum, configure the API key for the provider used by `[models.default]`.

Examples:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export OPENROUTER_API_KEY=...
```

Which one you need depends on your config.

---

### 3. Validate config and runtime prerequisites

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

### 4. Inspect discovered resources

```bash
cargo run -p halter-cli -- resources
```

Use this to verify that your skill and plugin roots are being discovered and compiled the way you expect.

---

### 5. Run a task

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

### 6. Use interactive mode

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

> [!NOTE]
> .toml config file usage is a thin serialization veneer over the programmatic config. For full customization programmatic configuration should be used, and probably preferred in headless, automated, or dynamic environments.

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

> [!NOTE]
> The vast majority of original ideas (and code) in this crate is taken from other FOSS projects, namely [pi-mono](https://github.com/badlogic/pi-mono) and [oh-my-pi](https://github.com/can1357/oh-my-pi/tree/main/crates/pi-natives)'s native Rust tool.

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

A realistic advanced composition path, built entirely in Rust:

```rust
use std::path::PathBuf;
use std::sync::Arc;

use halter::session::InMemorySessionStore;
use halter::{HalterBuilder, LoadedSkill};
use halter_config::{
    ConfiguredProvider, ContextConfig, HarnessConfig, ModelConfig, ModelsConfig,
    NetworkPolicyConfig, PolicyConfig, PromptsConfig, ProviderConfig, ProvidersConfig,
    ResourcesConfig, RuntimeConfig, SearchRoots, SessionBackend, SessionsConfig,
    ShellPolicyConfig, ToolsConfig,
};
use halter_protocol::{PruneSignalThreshold, ReasoningEffort, SkillId, Turn};
use halter_runtime::SessionInit;

const SYSTEM_PROMPT: &str =
    "You are a careful local coding agent. Prefer concrete, verifiable answers.";
const REPO_REVIEW_SKILL: &str = r#"When asked to review a codebase:
1. Start with correctness risks.
2. Then call out maintainability issues.
3. End with the smallest high-leverage next steps.
"#;

fn build_config() -> anyhow::Result<HarnessConfig> {
    let working_dir = std::env::current_dir()?;
    let temp_write_root = std::env::temp_dir().join("halter");

    Ok(HarnessConfig {
        version: 1,
        providers: ProvidersConfig {
            openai: Some(ProviderConfig {
                base_url: Some("https://api.openai.com".to_owned()),
                api_key: Some(std::env::var("OPENAI_API_KEY")?),
            }),
            anthropic: None,
            openrouter: None,
        },
        models: ModelsConfig {
            default: Some(ModelConfig {
                provider: ConfiguredProvider::OpenAi,
                model: "gpt-5.4".to_owned(),
                max_input_tokens: Some(200_000),
                max_output_tokens: Some(8_192),
                reasoning: Some(ReasoningEffort::High),
                tokens_per_minute: Some(500_000),
            }),
            fast: Some(ModelConfig {
                provider: ConfiguredProvider::OpenAi,
                model: "gpt-5.4-mini".to_owned(),
                max_input_tokens: Some(200_000),
                max_output_tokens: Some(4_096),
                reasoning: Some(ReasoningEffort::Low),
                tokens_per_minute: Some(1_000_000),
            }),
            subagent: Some(ModelConfig {
                provider: ConfiguredProvider::OpenAi,
                model: "gpt-5.4-mini".to_owned(),
                max_input_tokens: Some(200_000),
                max_output_tokens: Some(4_096),
                reasoning: Some(ReasoningEffort::Medium),
                tokens_per_minute: Some(750_000),
            }),
        },
        resources: ResourcesConfig {
            skills: SearchRoots { roots: Vec::new() },
            plugins: SearchRoots { roots: Vec::new() },
        },
        prompts: PromptsConfig {
            system_prompt: Some(SYSTEM_PROMPT.to_owned()),
        },
        context: ContextConfig {
            compaction_threshold: 200_000,
            pre_compaction_target: 150_000,
            prune_signal_threshold: PruneSignalThreshold::Low,
        },
        tools: ToolsConfig {
            enabled: vec![
                "read".to_owned(),
                "glob".to_owned(),
                "grep".to_owned(),
                "write".to_owned(),
                "edit".to_owned(),
                "shell".to_owned(),
                "process".to_owned(),
                "spawn_agent".to_owned(),
                "send_input".to_owned(),
                "wait_agent".to_owned(),
                "close_agent".to_owned(),
            ],
        },
        policy: PolicyConfig {
            allowed_write_roots: vec![working_dir.clone(), temp_write_root],
            max_read_bytes: 1_048_576,
            max_tool_output_bytes: 262_144,
            max_subagent_depth: 3,
            max_concurrent_subagents: 8,
            shell: ShellPolicyConfig {
                enabled: true,
                allow: vec![
                    "git".to_owned(),
                    "cargo".to_owned(),
                    "rg".to_owned(),
                    "ls".to_owned(),
                    "find".to_owned(),
                    "python".to_owned(),
                    "pwd".to_owned(),
                    "echo".to_owned(),
                ],
                timeout_secs: 30,
            },
            network: NetworkPolicyConfig {
                enabled: false,
                allowed_hosts: Vec::new(),
                allowed_loopback: Vec::new(),
            },
        },
        sessions: SessionsConfig {
            backend: SessionBackend::Memory,
            sqlite_path: None,
        },
        runtime: RuntimeConfig {
            working_dir: Some(working_dir),
        },
    })
}

fn inline_skills() -> Vec<LoadedSkill> {
    vec![LoadedSkill {
        id: SkillId::from("repo-review"),
        name: "repo-review".to_owned(),
        description: "Review a repository for correctness, maintainability, and next steps."
            .to_owned(),
        root: PathBuf::from("inline-skills/repo-review"),
        body: REPO_REVIEW_SKILL.to_owned(),
        supporting_files: Vec::new(),
        scripts: Vec::new(),
        revision: "repo-review-v1".to_owned(),
    }]
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let harness = HalterBuilder::new()
        .with_config(build_config()?)
        .with_loaded_skills(inline_skills())
        .with_session_store(Arc::new(InMemorySessionStore::default()))
        .build()
        .await?;

    let session = harness.new_session(SessionInit::default()).await?;
    let _events = session
        .submit_turn(Turn::user("Describe the active runtime and available skills"))
        .await?;

    Ok(())
}
```

This skips `halter.toml` entirely: model roles, policy, runtime settings, and even skills are assembled in memory before the harness is built.

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
