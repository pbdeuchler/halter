# Halter: Comprehensive Implementation Plan

This is the canonical implementation plan for `halter`.

It is intended to stand on its own as the implementation handoff artifact. It restores the detailed protocol, crate, runtime, and milestone material from the earlier plan, then incorporates the additional design work around service boundaries, config ergonomics, model roles, lightweight policy, and the heavier host runtime surface.

## 1. Purpose

Halter is a lightweight Rust agent harness SDK designed for two related deployment shapes:

1. A portable binary that can be copied into different environments, pointed at one config file, and run agentic loops with minimal setup.
2. An embeddable runtime used by richer systems such as terminal chat UIs, coding assistants, messaging-backed personal assistants, and background agent workers.

The core challenge is not just "call a model and run tools." The hard part is keeping one clean contract while adding:

- multiple providers
- prompt caching
- append-only history
- efficient context planning
- native tools
- child sessions and subagents
- replay and future persistence
- host-level session management

This plan takes the strongest ideas from the learnings in `docs/learnings/` and combines them into one design that is small enough to build, but strong enough not to paint the SDK into a corner.

## 2. Product Shape

Halter has two primary user-facing shapes:

1. A portable binary that is easy to ship and configure with one file.
2. A Rust SDK that can be embedded into richer hosts such as terminal chat apps, coding assistants, and messaging-backed personal assistants.

The target is not "copy Codex" or "copy pi." The target is a smaller runtime with cleaner seams and a stronger library boundary.

### 2.1 Hallmarks

- one config file, plus strong config helpers
- efficient context management with stable prompt prefixes and explicit context planning
- first-class Anthropic, OpenAI, and OpenRouter support
- native Rust tooling for hot paths
- strong subagent support built on real child sessions
- flexibility for future host products without dragging app-shell complexity into the SDK

## 3. Comparison Summary

The earlier detailed plan got several major things right:

- a protocol-first design
- segmented prompts and prompt-prefix stability
- a replay-friendly session/event model
- hashline-based file-view correctness
- native scan, grep, glob, and process control
- provider isolation
- OpenRouter as protocol compatibility rather than a separate runtime path
- declarative plugins
- cheap subagent forks

The independent design pass added missing structure:

- a dedicated `halter-config` crate
- `RuntimeServices` as an explicit service graph
- a second public surface for session-runtime hosts
- model roles and a model registry instead of a single implicit default model
- a real policy layer for bounded writes, shell/network allowlists, and subagent quotas
- explicit `ContextPlan` generation before model calls

The merged design keeps the detailed technical depth from the earlier plan and extends it where the previous version was too thin.

### 3.1 Merged decisions

The final design keeps these choices:

- protocol-first runtime design
- a single tooling crate that owns both built-in tools and native tool services
- a storage-agnostic session backend seam with an in-memory first implementation
- provider dispatch by API kind rather than vendor name
- a service graph instead of a god object session
- model roles and registry-driven resolution
- explicit context planning before model calls
- declarative plugins only in v1

## 4. Goals And Non-goals

### 4.1 Goals

- A host can construct a working harness from one config file or one builder chain.
- Anthropic, OpenAI, and OpenRouter all run through one runtime loop and one canonical transcript model.
- Context management is efficient and debuggable: stable prompt prefix, transcript windowing, elision, summaries, and file-view awareness.
- Common filesystem and search operations do not shell out.
- Subagents are first-class child sessions with lineage, quotas, and structured results.
- The runtime surface is small and composable for SDK users.

### 4.2 Non-goals

- A full TUI or desktop app
- A rich approval workflow product
- A pretend sandbox abstraction with no real enforcement
- Arbitrary code-loading plugins
- Multi-host orchestration
- Remote model discovery or catalog sync
- A broad app-plugin platform with UI hooks and lifecycle interceptors

## 5. Architectural Principles

These principles are load-bearing:

1. One canonical transcript and event contract across providers, replay, session backends, and host surfaces.
2. Stable prompt bytes matter. Segment churn is a performance bug.
3. `Session` should orchestrate a turn, not become a service locator or god object.
4. Model-visible file state is tracked separately from transcript history.
5. Tools are typed operations with metadata, not random helper functions.
6. Subagents are child sessions, not recursive prompt hacks.
7. Session history is modeled as append-oriented events even when the first backend is in memory, so future flat-file and database backends can reuse the same contract.
8. Provider quirks live in provider adapters, not in the runtime loop.
9. OpenRouter is modeled as compatibility on an OpenAI-style wire protocol.
10. Minor security guarantees should come from structure: bounded roots, allowlists, output ceilings, and quotas.

## 6. Workspace Layout

The workspace uses seven library crates plus one binary:

```text
crates/
  halter-protocol/
  halter-config/
  halter-runtime/
  halter-providers/
  halter-tools/
  halter-session/
  halter/
  halter-cli/
```

### 6.1 `halter-protocol`

Purpose:

- canonical message IR
- stream events
- session commands and session events
- tool spec and result types
- prompt segment types
- provider capability types

Rules:

- no tokio
- no HTTP stack
- no filesystem I/O
- only shared, stable contracts

### 6.2 `halter-config`

Purpose:

- config schema
- config loading
- layered config merging
- environment overrides
- validation
- migration hooks
- JSON Schema export

Rules:

- should be usable by CLI and SDK callers without pulling in the full runtime
- should not depend on providers or native tools

### 6.3 `halter-runtime`

Purpose:

- runtime service graph
- prompt assembly
- context planning
- session engine
- event routing
- subagent runtime
- host APIs

Rules:

- owns orchestration only
- does not own provider-specific request shapes
- does not own filesystem scan or grep engines
- does not own session backend implementations

### 6.4 `halter-providers`

Purpose:

- Anthropic adapter
- OpenAI Responses adapter
- OpenAI Chat adapter
- OpenRouter compat profile
- transport helpers
- error normalization

Rules:

- provider quirks stay here
- the runtime sees only canonical requests, canonical stream events, and normalized errors

### 6.5 `halter-tools`

Purpose:

- built-in tools
- native scan, grep, glob, and process services
- hashline helpers
- command-backed tool backend
- HTTP-backed tool backend

Rules:

- long-lived tooling services are shared
- hot-path operations are cancellation-aware
- tools use `ToolContext`
- tools route through policy
- tools never mutate session state directly

### 6.6 `halter-session`

Purpose:

- session backend trait
- in-memory session implementation
- replay and projection helpers
- later flat-file and database implementations

Rules:

- the first implementation is purely in memory
- future backends must plug into the same trait

### 6.7 `halter`

Purpose:

- facade crate
- curated re-exports
- `Halter`, `HalterBuilder`, `Session`, `SessionRuntime`

### 6.8 `halter-cli`

Purpose:

- portable binary
- config loader entrypoint
- session runner
- runtime and validation commands

### 6.9 What not to do

- Do not collapse this workspace into a single `core` crate.
- Do not split it into dozens of tiny crates.
- Do not let `Session` construct resources, backends, providers, and process-global state itself.
- Do not let future durable backends force storage concerns into the session loop.

## 7. Service Graph And Runtime Boundaries

The runtime must be wired from explicit services. This keeps APIs honest and prevents `Session` from turning into a giant mutable bag of helpers.

```rust
pub struct RuntimeServices {
    pub resources: Arc<ResourceHandle>,
    pub models: Arc<ModelRegistry>,
    pub tools: Arc<ToolRuntime>,
    pub sessions: Arc<dyn SessionStore>,
    pub policy: Arc<dyn ToolPolicy>,
    pub prompt_assembler: Arc<dyn PromptAssembler>,
    pub context_manager: Arc<dyn ContextManager>,
    pub event_bus: Arc<EventBus>,
}
```

### 6.1 Core runtime-owned types

```rust
pub struct SessionBlueprint {
    pub session_id: SessionId,
    pub parent_session_id: Option<SessionId>,
    pub model_role: ModelRole,
    pub resolved_model: ModelId,
    pub snapshot_revision: Revision,
    pub working_dir: PathBuf,
    pub system_prompt_seed: Vec<PromptSegment>,
    pub max_turns: Option<u32>,
    pub max_tool_calls_per_turn: u32,
    pub subagent_depth: u32,
}

pub struct SessionState {
    pub messages: Vec<Message>,
    pub file_view_cache: FileViewCache,
    pub appended_prompt_segments: Vec<PromptSegment>,
    pub pending_tool_calls: IndexMap<ToolCallId, PendingToolCall>,
    pub usage_so_far: Usage,
    pub summaries: Vec<SummarySlice>,
    pub lineage: Vec<SubagentRef>,
}

pub struct ObservedState {
    pub cwd: PathBuf,
    pub git_branch: Option<String>,
    pub git_dirty: Option<bool>,
    pub now_utc: Timestamp,
    pub env_facts: IndexMap<String, String>,
}
```

### 6.2 Public runtime surfaces

There are two public surfaces:

- `Halter`: simple path for one-off sessions
- `SessionRuntime`: heavier host surface for applications that need switching, resuming, background subagents, and runtime-managed session replacement

This mirrors the practical split seen in pi-style SDKs without importing their full app shell.

## 8. Configuration System

The config system is part of the product, not just an implementation detail.

### 7.1 Layering

Config precedence:

1. built-in defaults
2. user config
3. project config
4. explicit config path
5. environment overrides
6. programmatic overrides on `HalterBuilder`

### 7.2 Top-level schema

```toml
version = 1

[providers]
[models]
[roles]
[resources]
[prompts]
[context]
[tools]
[policy]
[sessions]
[runtime]
```

### 7.3 Providers, models, and roles

The registry should resolve from role names rather than hardcoded model IDs in runtime code.

```toml
[providers.anthropic]
kind = "anthropic"
api_key_env = "ANTHROPIC_API_KEY"

[providers.openai]
kind = "openai"
api_key_env = "OPENAI_API_KEY"

[providers.openrouter]
kind = "openrouter"
api_key_env = "OPENROUTER_API_KEY"
http_referer = "https://example.com"
x_title = "halter"

[models.claude_default]
provider = "anthropic"
api_kind = "anthropic_messages"
model = "claude-sonnet-4-6"
max_input_tokens = 200000
max_output_tokens = 8192
reasoning = "medium"

[models.gpt_plan]
provider = "openai"
api_kind = "openai_responses"
model = "gpt-5"
max_input_tokens = 200000
max_output_tokens = 8192
reasoning = "high"

[models.or_background]
provider = "openrouter"
api_kind = "openai_chat"
model = "openai/gpt-5-mini"

[roles]
default = "claude_default"
plan = "gpt_plan"
subagent = "or_background"
background = "or_background"
fast = "or_background"
slow = "gpt_plan"

[sessions]
backend = "memory"
```

`memory` is the only v1 backend. The config shape should already reserve future values such as `flat_file`, `sqlite`, and `postgres`.

### 7.4 Policy config

```toml
[policy]
allowed_write_roots = ["./", "/tmp/halter"]
max_read_bytes = 1048576
max_tool_output_bytes = 262144
max_subagent_depth = 3
max_concurrent_subagents = 8

[policy.shell]
enabled = true
allow = ["git", "cargo", "rg", "ls", "find"]
timeout_secs = 30

[policy.network]
enabled = false
allowed_hosts = []
```

### 7.5 Config helpers

`halter-config` should ship:

- `load_layered()`
- `apply_env_overrides()`
- `validate()`
- `export_json_schema()`
- `generate_starter_config()`

CLI affordances:

- `halter init`
- `halter validate`
- `halter config schema`

### 7.6 Skill and plugin source config

The config must allow users to define any number of skill and plugin roots. These roots are scanned by helper code before the agent is created.

```toml
[resources.skills]
roots = [
  "~/.config/agent/skills",
  "~/.claude/skills",
  "./.agent/skills",
]

[resources.plugins]
roots = [
  "~/.config/agent/plugins",
  "~/.claude/plugins",
  "./.agent/plugins",
]
```

Rules:

- roots are ordered, deterministic, and may be empty
- duplicate paths are deduplicated after canonicalization
- the runtime itself never walks these directories
- config helpers do all discovery, parsing, validation, and expansion before runtime instantiation

### 7.7 Programmatic resource configuration

SDK users must be able to bypass config-file scanning entirely and provide loaded skills/plugins in code.

Required builder paths:

- `HalterBuilder::with_resource_snapshot(snapshot)`
- `HalterBuilder::with_loaded_skills(skills)`
- `HalterBuilder::with_loaded_plugins(plugins)`
- helper-side builders may also expose `with_skill_roots(paths)` and `with_plugin_roots(paths)`

The boundary is strict:

- helper APIs may read the filesystem
- runtime APIs may not

## 9. Canonical Protocol And Data Model

The protocol crate is the center of gravity for the whole system.

### 8.1 Messages

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Message {
    System(SystemMessage),
    User(UserMessage),
    Assistant(AssistantMessage),
    Tool(ToolResultMessage),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMessage {
    pub id: MessageId,
    pub created_at: Timestamp,
    pub parts: Vec<UserPart>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UserPart {
    Text { text: SharedStr },
    Image { media_type: MediaType, data: Bytes },
    Document { media_type: MediaType, data: Bytes },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssistantMessage {
    pub id: MessageId,
    pub created_at: Timestamp,
    pub parts: Vec<AssistantPart>,
    pub stop_reason: Option<StopReason>,
    pub usage: Option<Usage>,
    pub replay_meta: ReplayMeta,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum AssistantPart {
    Text { text: SharedStr },
    Thinking(ThinkingBlock),
    ToolCall(ToolCall),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingBlock {
    pub text: SharedStr,
    pub signature: Option<ReplaySignature>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: ToolCallId,
    pub name: ToolName,
    pub arguments: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultMessage {
    pub id: MessageId,
    pub call_id: ToolCallId,
    pub content: ToolResult,
    pub error: Option<ToolError>,
    pub created_at: Timestamp,
}
```

### 8.2 Stream events

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StreamEvent {
    MessageStart { id: MessageId },
    TextStart { id: BlockId },
    TextDelta { id: BlockId, delta: SharedStr },
    TextEnd { id: BlockId },
    ThinkingStart { id: BlockId },
    ThinkingDelta { id: BlockId, delta: SharedStr },
    ThinkingEnd { id: BlockId, signature: Option<ReplaySignature> },
    ToolCallStart { id: BlockId, tool_call_id: ToolCallId, name: ToolName },
    ToolArgsDelta { id: BlockId, delta: SharedStr },
    ToolCallEnd { id: BlockId },
    UsageUpdate { usage: Usage },
    MessageEnd { id: MessageId, stop_reason: StopReason },
    ProviderWarning { message: SharedStr },
    Error { error: ProviderError },
}
```

### 8.3 Session commands and events

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SessionCommand {
    SubmitTurn { turn: Turn },
    InterruptTurn,
    AppendSystemPrompt { id: PromptId, text: SharedStr },
    SetModelRole { role: ModelRole },
    SetModel { model: ModelId },
    SpawnSubagent { spec: SubagentSpecWire },
    ReloadResources,
    Shutdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEvent {
    pub session_id: SessionId,
    pub sequence: u64,
    pub delivery: Delivery,
    pub payload: SessionEventPayload,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Delivery {
    Lossless,
    BestEffort,
}
```

Key event payloads:

- `SessionStarted`
- `TurnStarted`
- `MessageItem`
- `DeltaItem`
- `ToolExecutionStarted`
- `ToolExecutionCompleted`
- `ApprovalRequested`
- `ContextCompacted`
- `TurnCompleted`
- `TurnFailed`
- `Lagged`
- `SessionShutdownComplete`

### 8.4 Tool spec

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSpec {
    pub name: ToolName,
    pub description: SharedStr,
    pub input_schema: serde_json::Value,
    pub concurrency: ToolConcurrency,
    pub capabilities: ToolCapabilities,
    pub provider_aliases: IndexMap<ProviderKind, ToolAlias>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum ToolConcurrency {
    Exclusive,
    ReadOnly,
    ParallelSafe,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ToolCapabilities {
    pub mutating: bool,
    pub requires_approval: bool,
    pub cancellable: bool,
    pub long_running: bool,
}
```

### 8.5 Prompt segments

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptSegment {
    pub id: PromptSegmentId,
    pub text: SharedStr,
    pub volatility: Volatility,
    pub cache_scope: CacheScope,
    pub content_hash: ContentHash,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Volatility {
    Static,
    SessionStable,
    TurnDynamic,
    AlwaysDynamic,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum CacheScope {
    PrefixCacheable,
    Dynamic,
}
```

### 8.6 File-view cache

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileViewEntry {
    pub path: PathBuf,
    pub full_hash: ContentHash,
    pub mtime: Timestamp,
    pub size: u64,
    pub viewed_ranges: Vec<ViewedRange>,
    pub last_shown_turn: TurnId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewedRange {
    pub start_line: u32,
    pub end_line: u32,
    pub line_anchors: Vec<LineAnchor>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LineAnchor {
    pub line: u32,
    pub anchor: [u8; 3],
}
```

Rule:

- `read`, `grep`, and any content-viewing tool annotate viewed content with anchors
- `edit` operations must prove they reference anchors the model actually saw

## 10. Resource Loading And Snapshots

### 9.1 Resource kinds

The runtime loads:

- instruction files like `AGENTS.md`
- skills
- named agent definitions
- prompt fragments
- plugin manifests
- declarative tool definitions

### 9.2 Discovery sources

Precedence:

1. bundled defaults
2. user resources
3. project resources
4. runtime overrides

### 9.3 Immutable snapshots

```rust
pub struct ResourceSnapshot {
    pub revision: Revision,
    pub tools: IndexMap<ToolName, ToolSpec>,
    pub skills: IndexMap<SkillName, SkillDef>,
    pub agents: IndexMap<AgentName, AgentDef>,
    pub prompts: PromptRegistry,
    pub plugins: IndexMap<PluginId, PluginManifest>,
    pub instruction_files: Vec<InstructionFile>,
}
```

Sessions capture an `Arc<ResourceSnapshot>` at turn start. Reload swaps the live handle. A running turn continues against the snapshot it started with.

### 9.4 Resource loader boundary

```rust
#[async_trait]
pub trait ResourceLoader: Send + Sync {
    async fn load(&self, config: &HarnessConfig) -> anyhow::Result<ResourceSnapshot>;
    async fn reload(&self) -> anyhow::Result<ResourceSnapshot>;
}
```

Production implementations:

- `FsResourceLoader`
- `InMemoryResourceLoader` for tests

### 9.5 Resource compiler boundary

Filesystem discovery, path expansion, parsing, and validation for skills/plugins must live outside the agent contract.

Introduce a helper-side resource compilation pipeline:

```rust
pub struct ResourceCompiler { /* helper-side only */ }

impl ResourceCompiler {
    pub fn from_config(config: &HarnessConfig) -> Self;
    pub fn with_skill_roots(self, roots: Vec<PathBuf>) -> Self;
    pub fn with_plugin_roots(self, roots: Vec<PathBuf>) -> Self;
    pub fn with_loaded_skills(self, skills: Vec<LoadedSkill>) -> Self;
    pub fn with_loaded_plugins(self, plugins: Vec<LoadedPlugin>) -> Self;
    pub async fn compile(self) -> Result<ResourceSnapshot>;
}
```

Supporting helper-side loaders may be split explicitly:

```rust
pub struct SkillLoader { /* scans skill roots */ }
pub struct PluginLoader { /* scans plugin roots */ }
```

Rules:

- `ResourceCompiler` may read the filesystem
- `Halter`, `SessionRuntime`, and `Session` may not
- after instantiation, the runtime consumes only `ResourceSnapshot`
- refreshing skills/plugins means re-running helper-side compilation and swapping in a new snapshot

### 9.6 No filesystem reads after instantiation

This is a hard API rule:

- `Session` never scans skill directories
- `Session` never scans plugin directories
- `SessionRuntime` never parses `SKILL.md` or `plugin.json`
- `Halter::replace_resources(snapshot)` swaps a prebuilt snapshot; it does not discover one

This keeps the agent contract deterministic and host-friendly.

### 9.7 Instruction file handling

Rules:

- hash every instruction file
- store applied hashes in session state
- if unchanged, avoid re-injecting full contents
- if changed, inject the new content after the dynamic boundary

This preserves caching while keeping instruction changes visible.

## 11. Prompt Assembly, Context Planning, And Compaction

### 10.1 Prompt order

Canonical order:

1. harness identity and operating rules
2. rendered tool block
3. stable instruction resources
4. session-stable prompt additions
5. dynamic boundary marker
6. turn-dynamic environment hints
7. transcript window
8. file-view slices
9. carried summaries and elision markers
10. per-call additions

### 10.2 Prompt assembler

```rust
pub trait PromptAssembler: Send + Sync {
    fn assemble(
        &self,
        blueprint: &SessionBlueprint,
        state: &SessionState,
        observed: &ObservedState,
        snapshot: &ResourceSnapshot,
        plan: &ContextPlan,
    ) -> Result<AssembledPrompt>;
}

pub struct AssembledPrompt {
    pub segments: Vec<PromptSegment>,
    pub prefix_cache_key: CacheKey,
    pub transcript: Vec<Message>,
}
```

### 10.3 Context planning

```rust
pub struct ContextPlan {
    pub transcript_window: TranscriptWindow,
    pub file_views: Vec<FileViewSlice>,
    pub carried_summaries: Vec<SummarySlice>,
    pub elided_tool_results: Vec<ElisionMarker>,
    pub memory_items: Vec<MemoryItem>,
    pub projected_input_tokens: u64,
}
```

Context lanes:

- `stable`
- `working`
- `compressed`
- `memory`

### 10.4 Compaction strategy

V1 compaction layers:

1. tool-result elision
2. turn summarization

Post-v1:

3. durable session memory

### 10.5 Prefix caching

All `PrefixCacheable` segments must appear before the boundary.

Prefix cache key:

```text
sha256(segment_1.hash || segment_2.hash || ... || boundary.hash)
```

This key is translated into provider-specific cache hints by the provider adapter.

### 10.6 Skill catalog and activation in context

Best-in-class skills support requires progressive disclosure:

- the model sees a lightweight skill catalog up front
- full skill instructions are loaded only when a skill is activated
- supporting files are loaded individually on demand

Rules:

- if no skills are available, omit the skill catalog entirely
- if no skills are available, do not register a skill-activation tool
- skill activations must be constrained to valid skill names
- activated skill bodies should be protected from immediate compaction
- repeated activations of the same skill should deduplicate unless the underlying skill revision changed

The default activation path should be a dedicated tool such as `activate_skill`, with provider-facing aliases allowed. A host may also activate skills explicitly on behalf of the user.

## 12. Model Registry And Provider Layer

### 11.1 API kinds

V1 API kinds:

- `AnthropicMessages`
- `OpenAIResponses`
- `OpenAIChatCompletions`

OpenRouter is a compat profile on `OpenAIChatCompletions`.

### 11.2 Model registry

The runtime resolves model behavior from a registry rather than hardcoding the current model.

```rust
pub struct ModelRegistry {
    pub roles: IndexMap<ModelRole, ModelProfileRef>,
    pub profiles: IndexMap<ModelProfileId, ModelProfile>,
    pub providers: IndexMap<ProviderId, Arc<dyn ModelProvider>>,
}
```

### 11.3 Provider capability metadata

```rust
pub struct ProviderCapabilities {
    pub supports_tools: bool,
    pub supports_streaming: bool,
    pub supports_reasoning: bool,
    pub supports_interleaved_reasoning: bool,
    pub supports_images: bool,
    pub supports_documents: bool,
    pub supports_prompt_cache: bool,
    pub supports_tool_result_media: bool,
    pub requires_non_empty_assistant_content: bool,
    pub tool_call_id_policy: ToolCallIdPolicy,
    pub max_input_tokens: u64,
    pub max_output_tokens: u64,
}
```

### 11.4 Provider trait

```rust
#[async_trait]
pub trait ModelProvider: Send + Sync + 'static {
    fn kind(&self) -> ProviderKind;
    fn api_kind(&self) -> ApiKind;
    fn capabilities(&self) -> &ProviderCapabilities;

    async fn complete(
        &self,
        req: ModelRequest,
        cancel: CancellationToken,
    ) -> Result<ModelResponse, ProviderError>;

    async fn stream(
        &self,
        req: ModelRequest,
        cancel: CancellationToken,
    ) -> Result<BoxStream<'static, StreamEvent>, ProviderError>;
}
```

### 11.5 Cross-provider handoff

This is a first-class subsystem, not a utility:

- normalize tool-call IDs
- synthesize missing tool results
- drop empty content when required
- degrade or omit reasoning blocks
- filter replay metadata
- apply compat-profile-specific repairs

### 11.6 Retry and error normalization

Error taxonomy:

- `Auth`
- `RateLimited`
- `Overloaded`
- `ContextOverflow`
- `InvalidRequest`
- `Transport`
- `Protocol`
- `StreamCorruption`
- `Provider`
- `Cancelled`

Retry belongs around provider request attempts, not around arbitrary session state changes.

## 13. Tooling Runtime

### 12.1 Shared tooling runtime

```rust
pub struct ToolRuntime {
    pub registry: Arc<ToolRegistry>,
    pub scan_cache: Arc<FsScanCache>,
    pub search_db: Arc<SearchDb>,
    pub process_manager: Arc<ProcessManager>,
    pub profiler: Arc<Profiler>,
    cancel_root: CancellationToken,
}
```

### 12.2 Scan cache

Cache key dimensions:

- root
- follow symlinks
- respect gitignore
- include hidden

Guarantees:

- deterministic ordering
- `.git` always skipped
- TTL reuse
- empty-result recheck
- explicit invalidation after writes

### 12.3 Grep engine

Requirements:

- ripgrep-like semantics
- smart case
- bounded file size
- optional mmap
- regex sanitization for malformed brace cases
- anchors in content results

### 12.4 Glob engine

Requirements:

- `globset` + `ignore`
- scan-cache-backed
- deterministic ordering

### 12.5 Process manager

Requirements:

- spawn child processes
- enumerate descendants
- kill tree bottom-up
- work on Linux, macOS, and Windows
- integrate with cancellation tokens

### 12.6 Future native features

Post-v1 candidates:

- fuzzy search DB
- chunk tree editing
- AST search and rewrite
- ANSI helpers and syntax highlighting

## 14. Tool System

### 13.1 Tool trait

```rust
#[async_trait]
pub trait Tool: Send + Sync + 'static {
    fn spec(&self) -> Cow<'_, ToolSpec>;

    async fn invoke(
        &self,
        input: serde_json::Value,
        ctx: ToolContext<'_>,
    ) -> Result<ToolResult, ToolError>;
}
```

### 13.2 Tool context

```rust
pub struct ToolContext<'a> {
    pub runtime: &'a ToolRuntime,
    pub session: SessionStateView<'a>,
    pub file_view: &'a FileViewCache,
    pub cancel: CancellationToken,
    pub emit: &'a dyn ToolEventSink,
    pub working_dir: &'a Path,
    pub snapshot: &'a ResourceSnapshot,
}
```

Rules:

- tools never mutate session state directly
- tools never spawn detached tasks
- tools never change process-global cwd
- tools must use cancellation-aware runtime helpers

### 13.3 V1 built-in tools

- `read`
- `write`
- `edit`
- `glob`
- `grep`
- `shell`
- `skill`
- `agent`

### 13.4 Provider aliases

Models are often biased toward specific tool names. Keep aliasing in the tool spec:

- Anthropic may see `edit`
- OpenAI may see `apply_patch`

One implementation, multiple provider-facing names.

### 13.5 Anchored edit contract

Supported edit operations:

- replace line
- replace range
- insert after
- delete range

Validation steps:

1. file must exist in the file-view cache
2. on-disk content must still match viewed anchors
3. edits apply atomically
4. file-view cache updates after success

### 13.6 Declarative backends

V1 declarative tool backends:

- command-backed tools
- HTTP-backed tools

Both must route through policy and structured input/output contracts.

## 15. Lightweight Policy Layer

The runtime is not a full security product, but it must offer real structural guardrails.

### 14.1 Policy trait

```rust
pub trait ToolPolicy: Send + Sync {
    fn check(&self, call: &ToolCall, ctx: &PolicyContext) -> Result<PolicyDecision, PolicyError>;
}
```

### 14.2 Default policy controls

- allowed write roots
- max read bytes
- max tool output bytes
- allowed shell commands
- network host allowlist for HTTP tools
- subprocess timeout ceilings
- max concurrent subagents
- max subagent depth

### 14.3 Approval surface

The runtime may emit `ApprovalRequested`, but v1 does not own a rich approval UX. The host can deny, allow, or ignore depending on its own product needs.

## 16. Session Engine

### 15.1 Session loop

```rust
async fn run_turn(
    &mut self,
    turn: Turn,
    cancel: CancellationToken,
) -> Result<CompletedTurn, SessionError> {
    self.state.push_user_turn(turn);
    self.emit(SessionEventPayload::TurnStarted { turn_id });

    loop {
        let observed = self.observe_environment().await?;
        let plan = self.context_manager.before_turn(
            &mut self.state,
            &self.blueprint,
            &observed,
        ).await?;

        let assembled = self.prompt_assembler.assemble(
            &self.blueprint,
            &self.state,
            &observed,
            &self.snapshot,
            &plan,
        )?;

        let request = self.request_builder.build(
            &self.blueprint,
            &assembled,
            self.model_registry.resolve(self.blueprint.model_role)?,
        )?;

        let stream = self.provider.stream(request, cancel.child_token()).await?;
        let message = self.materialize_assistant_message(stream).await?;

        self.state.push_assistant(message.clone());
        self.emit_materialized_message(&message);

        if let Some(tool_calls) = message.tool_calls() {
            let results = self.execute_tool_calls(tool_calls, cancel.child_token()).await?;
            for result in results {
                self.state.push_tool_result(result.clone());
                self.emit_tool_result(&result);
            }
            continue;
        }

        self.context_manager.after_turn(&mut self.state, &message).await?;
        let completed = CompletedTurn::from_state(&self.state, turn_id);
        self.emit(SessionEventPayload::TurnCompleted { turn_id, usage: completed.usage.clone(), duration: completed.duration });
        return Ok(completed);
    }
}
```

### 15.2 Tool execution concurrency

Rules:

- `Exclusive`: sequential
- `ReadOnly`: parallel with other `ReadOnly`
- `ParallelSafe`: parallel
- any `Exclusive` in a mixed batch forces a sequential batch

### 15.3 Cancellation

Hierarchy:

- runtime cancellation root
- session token
- turn token
- tool-call token

Interruption must leave the transcript in a valid state. Any dangling tool call becomes a structured error result before the turn fails or exits.

## 17. Session Backend And Replay

### 16.1 In-memory session management first

The initial implementation in `halter-session` is purely in memory.

`InMemorySessionStore` owns live session records for the current process and is the only v1 backend. That keeps the runtime simple while preserving a backend seam from day one.

### 16.2 Session backend trait

The runtime should depend on a storage-agnostic trait:

```rust
#[async_trait]
pub trait SessionStore: Send + Sync {
    async fn create(&self, blueprint: &SessionBlueprint) -> Result<()>;
    async fn append(&self, session_id: &SessionId, event: SessionEvent) -> Result<()>;
    async fn load(&self, session_id: &SessionId) -> Result<ReplayedSession>;
    async fn list(&self) -> Result<Vec<SessionMetadata>>;
    async fn fork(&self, parent: &SessionId, child: &SessionBlueprint) -> Result<()>;
}
```

Future implementations should be able to drop in behind this trait:

- `FlatFileSessionStore`
- `SqliteSessionStore`
- `PostgresSessionStore`

### 16.3 Replay

Replay reconstructs:

- blueprint
- session state
- lineage
- summaries
- tool results

Replay is also the test substrate for deterministic session-loop fixtures.

### 16.4 Branching and forking

- forks create child session records with parent linkage
- branch metadata records parent session ID and copied boundary
- subagents get their own session IDs even in the in-memory backend

### 16.5 Future durable backends

Flat-file, SQLite, and Postgres backends are explicitly post-v1. They should satisfy the same append-oriented event contract rather than inventing new runtime semantics.

## 18. Subagent Architecture

### 17.1 Requirements

Subagents must be:

- cheap
- deterministic
- quota-bound
- observable
- isolated

### 17.2 Subagent spec

```rust
pub struct SubagentSpec {
    pub agent_name: AgentName,
    pub task: String,
    pub role_override: Option<ModelRole>,
    pub model_override: Option<ModelId>,
    pub tools_override: Option<Vec<ToolName>>,
    pub max_turns: Option<u32>,
    pub context_handoff: ContextHandoffPolicy,
    pub run_mode: SubagentRunMode,
}
```

Context handoff modes:

- `Clean`
- `InheritFileViews`
- `InheritRecent { n_turns }`

Run modes:

- `Wait`
- `Background`

### 17.3 Fork semantics

- child reuses the parent's stable prompt seed
- child clones mutable state according to handoff policy
- child gets its own session ID and backend-managed history
- child returns a structured result, not raw mutable state

### 17.4 Structured result

```rust
pub struct SubagentResult {
    pub session_id: SessionId,
    pub summary: String,
    pub final_output: Option<String>,
    pub viewed_files: Vec<PathBuf>,
    pub artifacts: Vec<ArtifactRef>,
    pub usage: Usage,
    pub exit: SubagentExit,
}
```

### 17.5 Quotas

Mandatory quotas:

- max depth
- max concurrent children
- max turns per child
- optional token budget ceiling

## 19. Resource Extensions, Skills, Agents, And Plugins

### 18.1 Best-in-class skills support

Skills are not just static prompt snippets. The loader/runtime split should support a proper skill lifecycle:

1. Discover skill roots.
2. Parse a skill catalog.
3. Present the catalog to the model without paying for full skill bodies up front.
4. Activate a skill on demand.
5. Keep active skill context stable and deduplicated over time.

The base shape should match the ecosystem-standard structure:

```text
skills/
  pdf-processor/
    SKILL.md
    reference.md        # optional
    scripts/            # optional
  code-reviewer/
    SKILL.md
```

Each loaded skill should capture:

- stable skill ID
- display name
- description and invocation guidance
- parsed frontmatter
- `SKILL.md` body
- auxiliary files and scripts
- source root and revision hash

Compatibility note:

- `skills/<name>/SKILL.md` is the preferred format
- plugin `commands/` markdown files should be supported as a legacy/compatibility input and normalized into the same internal skill or command resource model when appropriate

### 18.2 Skill catalog and activation

The runtime should expose skills through progressive disclosure:

- tier 1: skill catalog in the base context
- tier 2: full `SKILL.md` content only when activated
- tier 3: referenced support files loaded individually if needed

Best-practice rules:

- if no skills are available, omit the skill catalog entirely
- if no skills are available, do not register a skill activation tool
- activation parameters must be constrained to valid skill names
- active skill content should be protected from immediate compaction
- repeated activations of the same skill should deduplicate unless the skill revision changed
- subagents may inherit active skills selectively through context handoff

The default activation tool should be generic, for example `activate_skill`, with host or provider aliases allowed.

### 18.3 Named agents

Agent definitions should support:

- description
- prompt body
- role override
- optional tool subset
- optional default max turns
- optional skill preload list

### 18.4 Best-in-class plugin loading

Plugin support is load-only in v1. Installation and management are explicitly out of scope.

The loader must support Claude-compatible plugin layouts while also exposing generic naming:

- compatibility manifest path: `.claude-plugin/plugin.json`
- generic manifest path: `.agent-plugin/plugin.json`
- optional halter alias: `.halter-plugin/plugin.json`

If no manifest exists, the loader should still support default-location discovery for plugin components.

Critical Claude-compatible directory rule:

- only the manifest lives under `.claude-plugin/`
- all components live at the plugin root

Reference layout:

```text
my-plugin/
  .claude-plugin/
    plugin.json
  skills/
    triage/
      SKILL.md
  agents/
    reviewer.md
  hooks/
    hooks.json
  .mcp.json
  .lsp.json
  bin/
    helper
  settings.json
```

Representative manifest fields to support:

```json
{
  "name": "my-plugin",
  "version": "0.1.0",
  "skills": ["./skills/"],
  "agents": ["./agents/"],
  "hooks": "./hooks/hooks.json",
  "mcpServers": "./.mcp.json",
  "lspServers": "./.lsp.json"
}
```

### 18.5 Plugin component support

The loader should understand and normalize these plugin components into a generic internal contract:

- skills
- agents
- hooks
- MCP server definitions
- LSP server definitions
- output styles
- `bin/` executables exposed to the shell tool
- plugin default settings
- optional user-config schema and channels metadata if present

The core runtime may not use every component in v1, but the loader should preserve them in the parsed plugin contract so host applications can.

### 18.6 Plugin path and env compatibility

Claude-specific functionality should be implemented with generic aliases.

Required expansions:

- `${CLAUDE_PLUGIN_ROOT}` and `${PLUGIN_ROOT}`
- `${CLAUDE_PLUGIN_DATA}` and `${PLUGIN_DATA}`

Recommended halter aliases:

- `${HALTER_PLUGIN_ROOT}`
- `${HALTER_PLUGIN_DATA}`

Configuration-field compatibility:

- Claude-style `userConfig` should be parsed
- generic alias `pluginConfig` should be supported in the same internal model

Path rules:

- relative component paths must start with `./`
- paths are resolved relative to the plugin root
- paths may not traverse outside the plugin root
- custom component-path arrays replace defaults for commands, agents, skills, and output styles unless the default path is explicitly included
- if a custom skill path points directly at a directory containing `SKILL.md`, the skill frontmatter `name` field should determine the stable skill name; otherwise use the directory basename as fallback

### 18.7 Plugin and skill loader contract

Once the agent runtime is instantiated, it must never read the filesystem for skills or plugins.

That means:

- skill scanning happens in helpers
- plugin scanning happens in helpers
- `SKILL.md` parsing happens in helpers
- `plugin.json`, `hooks.json`, `.mcp.json`, `.lsp.json`, and related parsing happen in helpers
- the runtime consumes only `LoadedSkill`, `LoadedPlugin`, and `ResourceSnapshot`

Suggested helper-side types:

```rust
pub struct LoadedSkill {
    pub id: SkillId,
    pub name: String,
    pub description: String,
    pub root: PathBuf,
    pub body: Arc<str>,
    pub supporting_files: Vec<LoadedResourceFile>,
    pub scripts: Vec<LoadedExecutable>,
    pub revision: ContentHash,
}

pub struct LoadedPlugin {
    pub id: PluginId,
    pub root: PathBuf,
    pub manifest: PluginManifest,
    pub skills: Vec<LoadedSkill>,
    pub agents: Vec<LoadedAgent>,
    pub hooks: Vec<LoadedHook>,
    pub mcp_servers: Vec<LoadedMcpServer>,
    pub lsp_servers: Vec<LoadedLspServer>,
    pub output_styles: Vec<LoadedOutputStyle>,
    pub bin_paths: Vec<PathBuf>,
    pub defaults: PluginDefaults,
}
```

### 18.8 Refresh semantics

Programmatic refresh is required, but it must respect the boundary:

1. host runs helper-side discovery/loading again
2. host gets a new `ResourceSnapshot`
3. host calls `Halter::replace_resources(snapshot)` or the equivalent runtime API
4. active turns continue on the old snapshot; later turns use the new one

No runtime API should accept "a directory to scan" once the harness exists.

### 18.9 Plugin limits

Plugins may not:

- load arbitrary code into the runtime
- intercept the runtime loop through hidden hooks
- rewrite provider traffic directly
- patch other plugins

## 20. Public SDK And Host APIs

### 19.1 Facade crate

The `halter` crate should re-export the common surface:

- protocol types
- `Halter`
- `HalterBuilder`
- `Session`
- `SessionRuntime`
- `Tool`
- `ToolContext`
- `ToolError`

Required runtime-facing resource APIs:

- `Halter::from_resource_snapshot(snapshot)`
- `HalterBuilder::with_resource_snapshot(snapshot)`
- `HalterBuilder::with_loaded_skills(skills)`
- `HalterBuilder::with_loaded_plugins(plugins)`
- `Halter::replace_resources(snapshot)`

Required helper-side APIs:

- `ResourceCompiler::from_config(&config).compile()`
- `SkillLoader`
- `PluginLoader`

`Halter::from_config_file(...)` is a convenience constructor that performs config loading and resource compilation before runtime creation. After `Halter` exists, skill/plugin filesystem access is no longer allowed.

### 19.2 Minimal SDK example

```rust
use futures::StreamExt;
use halter::prelude::*;

#[tokio::main]
async fn main() -> Result<()> {
    let harness = Halter::from_config_file("./halter.toml").await?;
    let mut session = harness.new_session(SessionInit::default()).await?;

    let mut events = session.submit_turn(Turn::user("List files in src/")).await?;
    while let Some(event) = events.next().await {
        match event?.payload {
            SessionEventPayload::MessageItem(item) => {
                if let Some(text) = item.as_assistant_text() {
                    print!("{text}");
                }
            }
            SessionEventPayload::TurnCompleted { .. } => break,
            SessionEventPayload::TurnFailed { error, .. } => return Err(error.into()),
            _ => {}
        }
    }
    Ok(())
}
```

### 19.3 Builder example

```rust
let resources = ResourceCompiler::from_config(&config).compile().await?;

let harness = Halter::builder()
    .with_config(config)
    .with_resource_snapshot(resources)
    .with_tool(Read::default())
    .with_tool(Edit::default())
    .with_role_override(ModelRole::Plan, "gpt_plan")
    .build()
    .await?;
```

### 19.4 Host runtime example

`SessionRuntime` is for the secondary use case:

- long-running terminal apps
- messaging-backed assistants
- runtimes that switch active sessions
- background child-agent supervisors

### 19.5 Programmatic refresh example

```rust
let refreshed = ResourceCompiler::from_config(&config).compile().await?;
harness.replace_resources(refreshed).await?;
```

This is the only allowed refresh shape. The harness swaps an already-loaded snapshot; it does not read from disk itself.

## 21. CLI Surface

The binary stays small:

```text
halter [--config PATH] [COMMAND]

Commands:
  chat
  run <task>
  resources
  validate
  config schema
```

Future commands once a durable session backend exists:

- `sessions list`
- `sessions show <id>`
- `sessions fork <id>`

### 20.1 `halter chat`

Line-oriented REPL:

- submit stdin line as user turn
- stream assistant deltas
- show tool calls as compact summaries
- first `Ctrl-C` interrupts turn
- second `Ctrl-C` exits

### 20.2 `halter run`

Non-interactive one-shot mode. This is the primary portable-binary path.

## 22. Testing Strategy

### 21.1 Test categories

1. protocol round-trip tests
2. config layering and validation tests
3. prompt assembly and cache-key golden tests
4. provider request encoding tests
5. provider stream fixture tests
6. skill catalog and activation tests
7. plugin loader and compatibility tests
8. cross-provider handoff fixtures
9. native runtime tests
10. tool tests
11. session-loop tests with fake providers
12. session backend and replay tests
13. subagent tests
14. end-to-end facade tests
15. live smoke tests behind env flags

### 21.2 Test support

Feature-gated test helpers:

- `FakeProvider`
- `InMemorySessionStore`
- `InMemoryResourceLoader`
- `RecordingEventSink`

### 21.3 High-risk areas

Prioritize tests for:

- provider transcript normalization
- anchor validation and stale edit rejection
- skill progressive disclosure and activation dedupe
- plugin compatibility loading and path expansion
- interruption and cancellation
- replay correctness
- subagent quotas and lineage

## 23. Detailed Implementation Plan

This section restores the detailed execution shape from the earlier plan while aligning it to the merged architecture.

### Phase 0: Workspace and scaffolding

Deliverables:

- Cargo workspace
- crate skeletons
- `rust-toolchain.toml`
- fmt/clippy configuration
- CI skeleton
- root README with hello-world usage

Acceptance:

- `cargo build --workspace --all-features`
- `cargo test --workspace --all-features`
- `cargo clippy --workspace --all-features -- -D warnings`

### Phase 1: Protocol and config

Deliverables:

- `halter-protocol` core types
- `halter-config` schema, loader, env overrides, validation, schema export
- config support for multiple skill/plugin roots
- typed IDs and capability enums

Acceptance:

- protocol types serialize and deserialize cleanly
- JSON Schema exports for public config and protocol types
- config fixtures validate and merge predictably

### Phase 2: Runtime skeleton and service graph

Deliverables:

- `RuntimeServices`
- `SessionBlueprint`, `SessionState`, `ObservedState`
- `HalterBuilder`
- `Session` skeleton
- `SessionRuntime` skeleton
- `SessionStore` trait
- fake provider and in-memory session store

Acceptance:

- a session can be created with fake services
- a turn can be submitted and fail in a controlled way if provider execution is missing
- resource snapshots can be injected without any runtime filesystem dependency

### Phase 3: Prompt assembly, context planning, and event routing

Deliverables:

- prompt segment ordering
- prefix cache key generation
- `ContextPlan`
- `ContextManager`
- event bus with lossless and best-effort channels
- lag handling

Acceptance:

- deterministic prompt and cache-key golden tests
- unchanged dynamic segments do not perturb prefix cache keys
- lagging subscribers do not stall the session loop

### Phase 4: Session engine and fake-provider loop

Deliverables:

- materialization of stream events into assistant messages
- full turn loop
- retry and cancellation
- tool call detection and executor plumbing

Acceptance:

- fake-provider text-only turn succeeds
- fake-provider tool-call turn reaches the executor boundary
- interruption produces valid session state

### Phase 5: Real providers and model registry

Deliverables:

- Anthropic adapter
- OpenAI Responses adapter
- OpenAI Chat adapter
- OpenRouter compat profile
- model roles and registry
- capability metadata
- cross-provider transcript transforms

Acceptance:

- all three providers work through the same runtime path
- adapter-local quirks stay out of the runtime
- cross-provider handoff fixtures pass

### Phase 6: Tooling runtime and core built-in tools

Deliverables:

- scan cache
- grep engine
- glob engine
- process manager
- hashline helpers
- `read`, `write`, `edit`, `glob`, `grep`, `shell`

Acceptance:

- read/edit lifecycle works with anchors
- out-of-band file mutation invalidates edits
- shell allowlist and timeout enforcement works

### Phase 7: Policy layer, skills, and declarative tool backends

Deliverables:

- default `ToolPolicy`
- `skill` tool
- `SkillLoader`
- `PluginLoader`
- `ResourceCompiler`
- skill catalog + activation flow
- Claude-compatible and generic plugin layout loading
- command-backed tool backend
- HTTP-backed tool backend

Acceptance:

- skill catalogs are omitted when no skills are loaded
- activated skills use progressive disclosure and deduplicate cleanly
- plugin root env expansion supports both Claude-compatible and generic variable names
- plugin directory loading works without any runtime filesystem reads after instantiation
- policy rejects forbidden writes and shell invocations
- command-backed and HTTP-backed tools round-trip structured input and output

### Phase 8: Session backend and replay

Deliverables:

- `halter-session`
- `InMemorySessionStore`
- replay
- in-process session listing
- fork lineage metadata

Acceptance:

- in-memory sessions resume accurately within the host process
- replay is deterministic on fixtures
- replacing the resource snapshot updates skills/plugins for later turns without rescanning from inside the runtime

### Phase 9: Compaction and context pressure handling

Deliverables:

- tool-result elision
- turn summarization
- compacted context events

Acceptance:

- over-budget sessions compact before failure
- irreducible over-budget sessions fail with typed overflow

### Phase 10: Subagents

Deliverables:

- child session runtime
- `agent` tool
- structured `SubagentResult`
- quotas
- background mode

Acceptance:

- subagents reuse stable prompt state without sharing mutable state
- parent sees structured results and lineage
- quota violations fail predictably

### Phase 11: Facade polish, CLI, and plugin/skill loading ergonomics

Deliverables:

- polished `halter` facade
- CLI commands
- resource-loading helper docs and examples
- docs and examples

Acceptance:

- portable binary works from one config file
- SDK examples compile and read cleanly
- skills and plugins can be loaded from config-defined roots or from programmatic loaded objects
- declarative plugins contribute resources without code loading

## 24. Definition Of Done

Halter v1 is done when:

- one config file or one builder chain produces a working harness
- Anthropic, OpenAI, and OpenRouter run through one canonical runtime loop
- prompt prefix stability and file-view correctness are both implemented and tested
- native hot-path tools eliminate most shell dependence
- the default policy layer enforces real structural guardrails
- skills support progressive disclosure and activation without inflating base context
- plugins and skills load from helper-side roots or programmatic objects without any runtime filesystem access after instantiation
- subagents are real child sessions with quotas and lineage
- sessions are managed through a pluggable backend trait with a correct in-memory implementation
- the public API feels small and unsurprising to Rust developers

## 25. Post-v1 Queue

Explicitly deferred:

- flat-file session backend
- SQLite session backend
- Postgres session backend
- MCP client/server integration
- chunk-tree and AST-native editing
- richer remote transports such as JSON-RPC over stdio
- stronger sandbox integrations where the platform supports them
- durable memory stores beyond summary checkpoints

The key constraint is that post-v1 work should not require changing the canonical transcript contract or the prompt-segmentation model.
