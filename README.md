# halter

`halter` is a **simple and configurable agent harness and SDK** for building and operating thoroughbred agents.

> [!CAUTION]
> `halter` is still a heavy work in progress. Proceed at your own risk.

## Design Goals

- Cache friendliness
- Obsessive token optimization
- Best in class multi model support
- Best in class tool calling and hook support
- Simple and expressive API

## What halter gives you

At a high level, halter combines:

- a typed **protocol** for sessions, turns, events, resources, and tool calls
- a configurable **runtime** for prompt assembly, context management, provider execution, hooks, and subagents
- a built-in **tool harness** for reading, editing, writing, shell execution, process control, and delegated work
- **resource loading** for repo-local skills and plugins
- **policy enforcement** around filesystem writes, shell usage, tool output size, and subagent fanout
- **session persistence** with memory and SQLite backends

---

## Quickstart

Halter is designed to be plug and play in existing Rust code and services. The goal of the `halter` SDK is to abstract away the details of a harness, however there is still some small boilerplate:

- loading config
- compiling resources
- injecting custom tools or hooks
- selecting persistence strategy
- consuming session events programmatically

### Basic example with config file

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

This code does all of the following:

- loads and validates config
- compiles resources
- builds providers, tools, hooks, policy, and session storage
- creates a runtime
- creates a session
- executes one turn and streams the resulting events

### Detailed events

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

### From an in-memory config and snapshot

```rust
use halter::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = HarnessConfig::default(); // Include model config
    let snapshot = ResourceSnapshot::empty();

    let harness = Halter::from_config(config, snapshot).await?;
    let _session = harness.new_session(SessionInit::default()).await?;
    Ok(())
}
```

### Build from config + compiled resources

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

### Add a custom tool

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

### Provide your own session store

```rust
use std::sync::Arc;

use halter::{HalterBuilder, ResourceCompiler};
use halter_config::load_path;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = load_path("halter.toml").await?;
    let resources = ResourceCompiler::from_config(&config).compile().await?;
    let store = Arc::new(MyFancySessionStore::default());

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

## Crates

- `halter` — high-level SDK and builder
- `halter-config` — config schema, loading, overrides, validation
- `halter-protocol` — shared types and wire-format vocabulary
- `halter-runtime` — session engine, prompt assembly, event bus, compaction, subagents
- `halter-providers` — provider adapters and model registry
- `halter-tools` — tool runtime, built-in tools, policy, subagent control tools
- `halter-hooks` — event-driven hook and policy interception layer
- `halter-session` — session persistence and replay

### halter-config

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

`halter` provides convienence functionality for consuming `.toml` config files for where that makes sense operationally.

> [!NOTE]
> `.toml` config file usage is a thin serialization veneer over the programmatic config. For full customization programmatic configuration should be used, and probably preferred in headless, automated, or dynamic environments.

A non exhuastive example:

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
max_subagent_depth = 3
max_concurrent_subagents = 8

[policy.shell]
enabled = true
allow = ["git", "cargo", "rg", "ls", "find", "python", "pwd", "cwd", "echo"]
timeout_secs = 30

[sessions]
backend = "memory"

[runtime]
# Optional. When set, halter writes a `<session_id>.txt` JSONL trace per session
# into this directory: one header line followed by every committed SessionEvent.
# Useful for offline debugging and for restoring session state from disk.
# traces_dir = "/tmp/halter/traces"

# Optional. Keep off unless the caller wants the parent turn stream to include
# raw events from subagents spawned under that parent.
# subagent_event_forwarding = "off"
# subagent_event_forwarding = "all"
# subagent_event_forwarding_cap = 100_000 # 0 = unbounded
```

And a detailed programmatic config:

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
            ..RuntimeConfig::default()
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

### halter-protocol

`halter-protocol` defines the shared vocabulary used by the rest of the workspace.

That includes types for:

- turns
- messages
- session events
- tool calls and tool results
- resources and compiled artifacts
- provider-facing request/response chunks

If you are building integrations or parsing structured output, this crate matters a lot.

### halter-providers

`halter-providers` adapts concrete model backends into halter's normalized provider interface.

Built-in providers include:

- OpenAI
- Anthropic
- OpenRouter
- Fake/test provider
- Unsupported placeholder for builds where a transport is not wired in

Important operational differences:

- OpenAI supports streaming, prompt caching, and dedicated Responses compaction
- OpenRouter supports streaming, prompt caching, and inline Responses compaction
- Anthropic supports streaming, prompt caching, interleaved thinking, and inline Messages compaction
- capability differences are explicit and should be handled intentionally

### halter-tools

> [!NOTE]
> Many of the original ideas (and code) in this crate are taken from other FOSS projects, namely [pi-mono](https://github.com/badlogic/pi-mono) and [oh-my-pi](https://github.com/can1357/oh-my-pi/tree/main/crates/pi-natives)'s native Rust tool.

Built-in tools include:

- `read`
- `glob`
- `grep`
- `write`
- `edit`
- `shell`
- `process`
- `task` (in-memory todo list scoped to the session)

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

### halter-hooks

`halter-hooks` lets you observe and influence runtime behavior by reacting to lifecycle events.

Hooks can:

- approve or block actions
- request or deny permissions
- add system messages
- attach additional context
- rewrite inputs and outputs
- suppress output visibility
- stop execution

#### Example

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

See `../halter-hooks/README.md` for more information.

### halter-session

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

### halter-runtime

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

> [!NOTE]
> halter implements it's own compaction strategy. This _can be_ (but is not always) less token effecient than the managed compaction functionality offered by inference providers or frontier harnesses. The goal of the custom compaction is to result in a _higher quality_ context window, hopefully reducing overall token use throughout the turn. This also allows halter to provide a consistent, baseline experience regardless of which inference provider or model is used.

#### Example

---

## Security model

Halter does it's best to operate in a sane and safe way, but does not provide any hard security boundaries. As always, run sensitive workloads in fully sandboxed environments with the requisite defense in depth beyond process level safeguards.

### Tool boundaries

Enforced mechanically, best effort, by tool policy:

- where writes may occur
- which shell programs may run (kind of)
- how much can be read or emitted
- how many subagents may be active
- how deep delegation may go

### Semantic/runtime boundaries

The following can all be implemented in custom hooks:

- approvals
- denials
- stop conditions
- input/output rewriting
- extra context or warnings
- audit annotations

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
