# Halter: Refined Implementation Plan

This document is the short merged plan. The full merged spec, including the restored detailed material from the earlier plan, lives in [docs/halter-plan-comprehensive.md](/Users/pbdeuchler/workspace/src/github.com/pbdeuchler/halter/docs/halter-plan-comprehensive.md).

## 1. Why This Revision Exists

This document replaces the earlier `docs/halter-plan.md` after a two-pass design process:

1. Read `docs/learnings/` and form an independent architecture first.
2. Read the existing plan only after that, compare both designs, and merge the best ideas.

That comparison changed the shape of the final plan in useful ways.

The previous plan was already strong on:

- protocol detail
- replay-friendly session semantics
- file-view correctness
- native tooling
- provider isolation

The independent pass added several missing pieces:

- a clearer service graph so the runtime does not become a god object
- model registry and model-role ergonomics for real multi-provider use
- a lightweight policy layer for minor security guarantees without pretending to be a full sandbox product
- a more explicit context-planning pipeline
- a clearer split between a simple embedded SDK surface and a heavier multi-session host runtime

The merged result below is the ground-truth plan.

## 2. Product Shape

Halter is a Rust agent harness SDK for two related use cases:

1. A portable binary that can be dropped into different environments, pointed at one config file, and run agentic loops with minimal ceremony.
2. An embeddable SDK for richer agent systems: messaging assistants, terminal chat UIs, coding agents, background workers, and subagent-heavy orchestration.

The target is not "copy Codex" or "copy pi". The target is a smaller runtime with cleaner seams.

### 2.1 Hallmarks

- One config file, plus helpers that make config easy to generate, validate, layer, and override.
- Efficient context handling built around stable prompt prefixes, append-only history, file-view awareness, and explicit context planning.
- First-class support for Anthropic, OpenAI, and OpenRouter.
- Native Rust tools for hot paths so common agent work does not shell out.
- Best-in-class subagents built on real child sessions, not prompt hacks.
- Strong extension seams without dragging in product-shell complexity.

### 2.2 Non-goals

These are intentionally out of scope for v1:

- a rich TUI
- a web UI
- a full approval workflow product
- a fake sandbox abstraction that cannot enforce anything
- arbitrary plugin code execution
- cloud orchestration across multiple hosts
- provider catalog sync or remote model discovery

## 3. Compare And Contrast

### 3.1 What the earlier plan got right

- One canonical IR across providers.
- Segmented prompts with a hard static/dynamic boundary.
- Hashline-style file-view correctness.
- Native scan, grep, glob, and process control.
- A replay-friendly session/event model that can support durable backends later.
- OpenRouter treated as protocol compatibility, not just a base URL swap.
- Declarative plugins instead of a giant extension runtime.
- Subagents modeled as cheap forks with shared stable context.

### 3.2 What the independent design added

- Rename `halter-core` to `halter-runtime`. `core` attracts unrelated code.
- Add a dedicated `halter-config` crate because config is a first-class user API, not an implementation detail.
- Make `RuntimeServices` explicit so resource loading, models, tools, storage, policy, and context all stay outside `Session`.
- Add model roles and a registry layer: `default`, `plan`, `subagent`, `background`, `fast`, `slow`.
- Add a real policy layer for bounded writes, shell/network allowlists, byte ceilings, and subagent quotas.
- Add a `ContextPlan` object so context management is a deliberate planning step, not just compaction after the fact.
- Add a second public runtime surface for hosts that need session switching, forks, and background child sessions.

### 3.3 Merged decisions

The final design keeps the previous plan's strongest details and adds the missing seams:

- Keep the protocol-first architecture.
- Keep the native tooling plan.
- Keep a storage-agnostic session backend API, but make the first implementation purely in memory.
- Keep the provider API-kind model.
- Add a service graph.
- Add a real config subsystem.
- Add model-role ergonomics.
- Add a lightweight policy subsystem.
- Add an explicit host runtime surface.

## 4. Architectural Principles

These are the real spec. Everything else is subordinate to them.

1. One canonical transcript and event contract. Providers convert to and from it. The session loop never learns provider-specific field names.
2. Keep the runtime narrow. Session execution, context planning, and orchestration belong in the runtime. Providers, tooling internals, session backends, and config do not.
3. Stable prompt bytes matter. Prompt segments are ordered, hashed, and volatility-classified. Anything that churns bytes casually is a bug.
4. Model-visible state is not the same thing as transcript state. The file-view cache tracks what the model actually saw.
5. Tools are typed operations with concurrency and safety metadata, not ad hoc helpers.
6. Subagents are real child sessions with lineage, quotas, and structured results.
7. Session history is modeled as append-oriented events even when the first backend is in memory, so flat-file and database backends can slot in later without changing the runtime contract.
8. OpenRouter is a compatibility profile on an OpenAI-style protocol adapter, not a separate runtime path.
9. Capability metadata beats model-name heuristics.
10. Native Rust services should own traversal, search, edit validation, and process cleanup.

## 5. Workspace Layout

The merged design uses seven library crates plus one binary:

```text
crates/
  halter-protocol/   # Shared message, event, tool, and session types.
  halter-config/     # Config schema, layering, validation, env overrides.
  halter-runtime/    # Session engine, prompt assembly, context planning,
                     # service graph, host APIs, subagents.
  halter-providers/  # Anthropic, OpenAI Responses, OpenAI Chat compat.
  halter-tools/      # Built-in tools, native scan/grep/glob/process services,
                     # hashline helpers, and declarative tool backends.
  halter-session/    # Session backend trait plus in-memory implementation first;
                     # flat-file, SQLite, and Postgres implementations later.
  halter/            # Facade crate for most users.
  halter-cli/        # Portable binary.
```

### 5.1 Why this split

- `halter-protocol` stays dependency-light and transport-neutral.
- `halter-config` keeps config loading and schema export available without pulling in the runtime.
- `halter-runtime` owns orchestration, not "everything useful."
- `halter-providers`, `halter-tools`, and `halter-session` stay swappable and testable.
- `halter` keeps the public API small.

### 5.2 What not to do

- Do not collapse this into `halter-core`.
- Do not split into Codex-scale microcrates.
- Do not make `Session` own resources, storage, models, tools, and process-global state.

## 6. Service Graph And API Boundaries

The missing piece in the earlier plan was an explicit service graph. This is now mandatory.

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

`Session` depends on `RuntimeServices` plus an immutable `SessionBlueprint` and mutable `SessionState`. It does not construct these services itself.

### 6.1 Public surfaces

There are two public runtime surfaces:

- `Halter`: the simple SDK surface for most users.
- `SessionRuntime`: the heavier host surface for multi-session applications that need switching, resuming, forking, and background child work.

This follows the pi lesson that a single embedded session and a session-replacing runtime are different needs.

### 6.2 Public API shape

`halter` should expose a small, boring API:

```rust
impl Halter {
    pub fn builder() -> HalterBuilder;
    pub async fn from_config(config: HarnessConfig) -> Result<Self>;
    pub async fn from_config_file(path: impl AsRef<Path>) -> Result<Self>;

    pub async fn new_session(&self, init: SessionInit) -> Result<Session>;
    pub async fn resume_session(&self, id: &SessionId) -> Result<Session>;
    pub async fn new_runtime(&self, init: RuntimeInit) -> Result<SessionRuntime>;

    pub async fn reload_resources(&self) -> Result<()>;
    pub async fn reload_config(&self, config: HarnessConfig) -> Result<()>;
}
```

`SessionRuntime` is for the secondary use case:

```rust
impl SessionRuntime {
    pub fn session(&self) -> &Session;
    pub async fn new_session(&mut self, init: SessionInit) -> Result<()>;
    pub async fn switch_session(&mut self, id: &SessionId) -> Result<()>;
    pub async fn fork_session(&mut self, parent: &SessionId, init: SessionInit) -> Result<()>;
    pub async fn spawn_background_subagent(&self, spec: SubagentSpec) -> Result<SubagentHandle>;
}
```

The simple path stays simple. The complex path is available without forcing every caller to care.

## 7. Configuration Model

The config system is a first-class feature, not a thin serde wrapper.

### 7.1 One file, many concerns

Halter uses one config file, usually `halter.toml`, with these major sections:

- `providers`
- `models`
- `roles`
- `resources`
- `prompts`
- `context`
- `tools`
- `policy`
- `sessions`
- `runtime`

### 7.2 Providers, models, and roles

The earlier plan only modeled one provider well. That is not enough.

The merged config model must support:

- multiple providers configured at once
- multiple model profiles per provider
- stable role names that higher-level code can target
- subagent and background defaults

Example:

```toml
version = 1

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

`memory` is the only v1 backend. The config shape should already leave room for future values such as `flat_file`, `sqlite`, and `postgres`.

### 7.3 Config helpers

`halter-config` should provide:

- layered load: built-ins, user config, project config, explicit file, env, builder overrides
- typed validation
- JSON Schema export
- config migration hooks by version
- helpers for generating starter configs

CLI commands:

- `halter init`
- `halter validate`
- `halter config schema`

### 7.4 Config ergonomics principles

- Explicit beats implicit.
- Role names should be stable across config reloads.
- Provider and model names should be human-friendly and user-defined.
- Validation must point to exact config paths.
- The host should be able to override one field programmatically without reimplementing config loading.

## 8. Canonical Data Model

The existing plan was right to go protocol-first. Keep that. The key refinement is to emphasize the boundary types that shape code flow.

### 8.1 Core protocol types

`halter-protocol` owns:

- `Message`
- `AssistantPart`
- `ToolCall`
- `ToolResult`
- `StreamEvent`
- `SessionCommand`
- `SessionEvent`
- `ToolSpec`
- `PromptSegment`
- `Usage`
- `ProviderCapabilities`
- `FileViewEntry`
- `LineAnchor`

### 8.2 Runtime-only types

`halter-runtime` owns:

- `SessionBlueprint`
- `SessionState`
- `ObservedState`
- `ContextPlan`
- `CompletedTurn`
- `SubagentSpec`
- `SubagentResult`
- `RuntimeServices`

### 8.3 State split

Keep the earlier plan's three-bucket state model, but tighten the semantics:

- `SessionBlueprint`: immutable per session
- `SessionState`: mutable conversation state and projections
- `ObservedState`: ephemeral environment facts gathered at turn boundaries

That split prevents "helpful" code from mutating session identity or mixing environment facts into durable history.

## 9. Code Flow

This section matters because the SDK is for developers. The shape of the code path must be obvious.

### 9.1 Startup flow

1. Load layered config through `halter-config`.
2. Build `ModelRegistry`, `ToolRuntime`, `ResourceHandle`, `SessionStore`, and `ToolPolicy`.
3. Create `RuntimeServices`.
4. Expose either `Halter` or `SessionRuntime`.

### 9.2 Turn flow

1. Host submits a `Turn`.
2. `Session` appends the user message to `SessionState`.
3. `ContextManager` creates a `ContextPlan`.
4. `PromptAssembler` builds segmented prompt output from the blueprint, context plan, and resource snapshot.
5. Provider adapter encodes the request from the canonical transcript plus prompt segments.
6. Provider stream emits `StreamEvent`s.
7. Runtime materializes one `AssistantMessage`.
8. If tool calls exist, runtime routes them through policy, executes them, appends results, and loops.
9. If no tool calls remain, runtime finalizes the turn, updates context projections, emits `TurnCompleted`, and records lossless events through the session backend.

### 9.3 Tool call flow

1. `AssistantMessage` contains one or more `ToolCall`s.
2. `ToolExecutor` groups them by concurrency class.
3. `ToolPolicy` validates each call: write roots, byte limits, network allowlists, shell allowlists, subagent quotas.
4. Tool executes against `ToolContext`.
5. Runtime records the structured `ToolResult`.
6. Provider sees the normalized tool result on the next loop iteration.

### 9.4 Subagent flow

1. Parent invokes the built-in `agent` tool or host API.
2. Runtime resolves an `AgentDef` plus role/model policy.
3. Child session reuses the parent's stable prompt seed and optionally selected context lanes.
4. Child runs as a real session with its own event stream and backend-managed history.
5. Child returns a structured `SubagentResult`.
6. Parent records the result as a tool result. Optional handback merges approved artifacts such as file views or summaries.

## 10. Prompt Assembly And Context Planning

The earlier plan had strong compaction ideas. The merged design makes context planning explicit.

### 10.1 Prompt assembly stays segmented

Prompt segment order remains:

1. identity and harness rules
2. stable tool descriptions
3. stable instruction files
4. session-stable additions
5. dynamic boundary
6. turn-dynamic environment hints
7. transcript window
8. file-view excerpts and change summaries
9. optional summary checkpoints
10. per-call additions

### 10.2 ContextPlan

Before each model call, `ContextManager` produces:

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

This is better than letting compaction mutate state blindly. The runtime can explain why a turn included or excluded a given context lane.

### 10.3 Context lanes

There are four lanes:

- `stable`: prompt prefix and resource-derived instructions
- `working`: recent transcript and active file views
- `compressed`: summaries and elided tool outputs
- `memory`: optional durable facts or external retrieval

This keeps advanced context management understandable and debuggable.

### 10.4 Prefix caching

Keep the earlier plan's hard dynamic boundary and segment hashing.

Additional merged rule:

- tool descriptions and instruction resources are rendered once per snapshot revision
- the runtime should expose cache-debug data so hosts can inspect why a prefix hit or missed

### 10.5 File-view cache and hashline

Keep the earlier plan's hashline design. It is one of the highest-leverage ideas in the entire repo.

Rules:

- every content-bearing read tags lines with short anchors
- `grep` results carry anchors too
- `edit` refuses to act on lines the model has not seen
- changed files require a re-read or an explicit diff summary

## 11. Model And Provider Architecture

### 11.1 API kinds

Keep the three real API kinds:

- `AnthropicMessages`
- `OpenAIResponses`
- `OpenAIChatCompletions`

OpenRouter is `OpenAIChatCompletions + CompatProfile::OpenRouter`.

### 11.2 Registry structure

`ModelRegistry` resolves:

- provider config
- API kind
- model capabilities
- role aliases
- optional per-role defaults like reasoning level or verbosity

This avoids hardcoding "the current model" in too many places and makes subagents much easier to reason about.

### 11.3 Adapter responsibilities

Each provider adapter owns:

- request encoding
- stream decoding
- usage normalization
- error classification
- provider capability declaration
- transcript normalization hooks local to that protocol

The runtime must not know about fields like `prompt_cache_key`, `store`, `cache_control`, or provider-specific reasoning knobs.

### 11.4 Cross-provider handoff

Keep the earlier plan's transform subsystem, but elevate it to a first-class compatibility layer.

It must handle:

- tool-call ID normalization
- empty-content filtering
- dangling-tool repair
- reasoning downgrade or omission
- replay metadata filtering
- compat-profile-specific quirks

This is a real subsystem. Do not bury it in ad hoc helper functions.

## 12. Tooling Runtime And Minor Security Guarantees

The user asked for native tooling partly for security and ergonomics. That needs to be explicit.

### 12.1 Native services

`halter-tools` owns both the built-in tools and the native services they rely on:

- deterministic filesystem scan cache
- glob engine
- grep engine
- optional fuzzy search DB
- process manager
- hashline helpers
- optional chunk and AST services

### 12.2 Built-in tools

V1 built-ins:

- `read`
- `write`
- `edit`
- `glob`
- `grep`
- `shell`
- `skill`
- `agent`

V1.5 or phase-gated:

- `fetch`
- `mcp`
- `chunk_read`
- `chunk_edit`

### 12.3 Lightweight policy layer

This is one of the biggest changes from the earlier plan.

Halter does not ship a full approval product in v1, but it must ship structural guardrails.

`ToolPolicy` validates tool execution against config:

```rust
pub trait ToolPolicy: Send + Sync {
    fn check(&self, call: &ToolCall, ctx: &PolicyContext) -> Result<PolicyDecision, PolicyError>;
}
```

Default policy controls:

- allowed write roots
- maximum read bytes
- maximum tool output bytes
- allowed shell commands
- allowed network hosts for HTTP-backed tools
- subprocess timeout ceilings
- max concurrent subagents
- max subagent depth

This gives real, minor safety guarantees without pretending to be a hardened sandbox.

### 12.4 Shell is the exception, not the substrate

The portable binary can still offer `shell`, but common operations must not rely on it.

Native implementations should cover:

- traversal
- glob
- grep
- file reads
- edit validation
- process cleanup

If a user disables `shell`, the core agent must still be useful.

## 13. Resource Loading, Skills, Agents, And Plugins

Keep the earlier plan's discovery model, but tighten the boundary:

- resources are declarative
- snapshots are immutable
- sessions capture a snapshot at turn start
- reload swaps the snapshot, not the running turn

### 13.1 Resource sources

In precedence order:

1. bundled defaults
2. user resources
3. project resources
4. runtime overrides

### 13.2 Resource kinds

- instruction files like `AGENTS.md`
- skills
- agent definitions
- prompt fragments
- plugin manifests
- declarative tool specs

### 13.3 Plugin rules

Plugins in v1 are resource bundles only.

They may contribute:

- prompt fragments
- skills
- agent definitions
- command-backed tools
- HTTP-backed tools
- model presets

They may not:

- load arbitrary code
- intercept runtime events
- mutate other plugin resources
- patch provider logic

### 13.4 Extension future

The design must leave room for later MCP support and richer extension seams, but that is post-v1. The first extension boundary is resource contribution plus declarative tool backends.

## 14. Session Backend And Event Model

### 14.1 In-memory first

Session management is purely in memory in v1 through `halter-session`.

The first implementation should be `InMemorySessionStore`. It owns live session records for the current process only. Cross-process persistence is intentionally deferred.

### 14.2 Backend boundary

The important design constraint is swappability. The runtime should depend on a backend trait that later backends can satisfy without changing session, tool, provider, or host code.

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

Future implementations should include:

- `FlatFileSessionStore`
- `SqliteSessionStore`
- `PostgresSessionStore`

### 14.3 Replay and projections

Replay and projection logic should operate over the same append-oriented event model even when the first backend is in memory.

Replay builds:

- `SessionBlueprint`
- `SessionState`
- searchable metadata
- UI-friendly summaries

### 14.4 Delivery classes

Keep the earlier plan's `Lossless` vs `BestEffort` event split. It is correct and should survive.

### 14.5 Future durability

Durable backends are explicitly post-v1. Flat-file, SQLite, and Postgres implementations should plug into the trait above rather than forcing new runtime APIs.

## 15. Subagent Architecture

This is a headline feature, so the plan needs more than "fork state and hope."

### 15.1 Design goals

Subagents must be:

- cheap to create
- easy to limit
- observable by the host
- structurally isolated
- able to return more than plain text

### 15.2 Subagent spec

```rust
pub struct SubagentSpec {
    pub agent: AgentName,
    pub task: String,
    pub role_override: Option<ModelRole>,
    pub model_override: Option<ModelId>,
    pub tool_subset: Option<Vec<ToolName>>,
    pub context_handoff: ContextHandoffPolicy,
    pub run_mode: SubagentRunMode,
    pub max_turns: Option<u32>,
}
```

`run_mode` matters:

- `Wait`: used by the built-in `agent` tool
- `Background`: used by host runtimes such as a messaging assistant or TUI

### 15.3 Structured result

Do not reduce subagents to "string in, string out." Return:

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

This gives the parent and the host something useful to work with.

### 15.4 Quotas and policy

Subagent spawning must go through `ToolPolicy` and runtime quotas:

- depth limit
- concurrency limit
- token budget limit
- allowed roles and providers

This keeps "best in class subagents" from becoming "easy way to melt your machine."

## 16. Testing Strategy

The earlier plan's testing direction was strong. Keep it, but organize it around the merged seams.

### 16.1 Test categories

1. Protocol round-trip tests.
2. Config layering and validation tests.
3. Prompt assembly and cache-key golden tests.
4. Provider adapter encoding and stream-decoding fixtures.
5. Cross-provider handoff fixtures.
6. Native runtime tests for scan cache, grep, glob, process cleanup, and anchors.
7. Tool tests with happy and sad paths.
8. Session-loop tests with fake providers.
9. Session backend and replay tests.
10. Subagent tests, including background mode and quotas.
11. End-to-end facade tests.
12. Live smoke tests behind env flags.

### 16.2 Testing priorities

If schedule pressure forces tradeoffs, prioritize:

1. provider fixtures
2. tool correctness and anchor validation
3. session loop and replay
4. subagent lineage and quotas

Those are the failure modes that will hurt users fastest.

## 17. Six-Milestone Implementation Plan

The earlier plan had many good steps but too much phase granularity for actual execution. The merged plan compresses that into six milestones with clearer vertical slices.

### Milestone 1: Foundation, config, and protocol

Build the workspace, crate boundaries, protocol types, config schema, config loader, and service graph skeleton.

Deliverables:

- crate scaffolding
- `halter-protocol` core types
- `halter-config` schema, layering, and validation
- `RuntimeServices` skeleton
- `HalterBuilder` skeleton
- fake `SessionStore` and fake `ModelProvider`

Acceptance:

- `cargo test --workspace` passes with mostly skeleton coverage
- config files round-trip and validate
- a minimal `Halter::from_config` can construct a session even if provider calls are still fake

### Milestone 2: Runtime loop, prompt assembly, and event pipeline

Implement the session engine, prompt segmentation, context planning skeleton, event streams, and retry/cancellation behavior.

Deliverables:

- `SessionBlueprint`, `SessionState`, `ObservedState`
- `ContextManager` and initial `ContextPlan`
- `PromptAssembler` with stable/dynamic boundary
- event bus with lossless and best-effort delivery
- full fake-provider-driven turn loop

Acceptance:

- text-only turns work end to end with a fake provider
- prompt cache keys are deterministic
- lagging subscribers receive `Lagged` instead of stalling the runtime
- turn cancellation works cleanly

### Milestone 3: Real providers, model registry, and cross-provider handoff

Implement Anthropic, OpenAI Responses, and OpenAI Chat compat with OpenRouter profiles, plus the role-aware model registry and transcript transform layer.

Deliverables:

- provider adapters
- model registry with roles
- capability metadata
- usage normalization
- cross-provider handoff fixtures

Acceptance:

- the same runtime loop works unchanged across all three providers
- OpenRouter-specific headers and compat behavior are localized to the adapter layer
- cross-provider transcript transforms pass fixtures

### Milestone 4: Tooling runtime, tools, and policy

Implement the native filesystem and process services, built-in tools, hashline-based file-view cache, and the default policy layer inside `halter-tools`.

Deliverables:

- scan cache
- grep and glob engines
- process manager
- `read`, `write`, `edit`, `glob`, `grep`, `shell`, `skill`, `agent`
- `ToolPolicy`
- anchor-based edit validation

Acceptance:

- read/edit lifecycle works with anchors
- stale edits fail safely
- shell allowlists and timeout ceilings are enforced
- common workflows do not require shelling out

### Milestone 5: Session backend, compaction, and subagents

Implement the session backend trait, the in-memory session store, replay APIs, context compaction, child sessions, quotas, and structured subagent results.

Deliverables:

- `halter-session`
- `InMemorySessionStore`
- replay and in-process resume
- tool-result elision and summary checkpoints
- child session runtime
- `SubagentResult`
- lineage and quota enforcement

Acceptance:

- an in-memory session can be resumed and forked within the host process through the backend trait
- subagents reuse stable context without sharing mutable state
- background subagents can be observed through the host runtime
- context pressure triggers compaction predictably

### Milestone 6: SDK polish, CLI, plugins, and extensibility seams

Finalize the facade crate, CLI binary, declarative plugins, command/HTTP tools, documentation, and post-v1 hooks such as MCP-ready seams.

Deliverables:

- `halter` facade polish
- `halter-cli`
- plugin manifest loader
- command-backed and HTTP-backed tools
- docs and examples
- explicit extension seam docs for future MCP and chunk tooling

Acceptance:

- the portable binary is usable with one config file
- the SDK examples compile and read naturally
- plugins can contribute declarative tools and resources without code loading
- the design leaves room for MCP and richer host shells without changing the transcript contract

## 18. Definition Of Done

Halter v1 is done when all of these are true:

- A user can start a session from one config file or one builder chain.
- Anthropic, OpenAI, and OpenRouter all run through the same canonical runtime loop.
- Context handling preserves a stable prefix and validates edits against model-visible file state.
- Native tools cover the common hot paths without relying on shell subprocesses.
- The default policy layer provides real structural guardrails.
- Subagents are real child sessions with quotas, lineage, and structured results.
- Sessions are managed through a backend trait with a correct in-memory implementation, and future flat-file or database backends do not require runtime API changes.
- The public API feels small, composable, and unsurprising to Rust developers.

## 19. Post-v1 Queue

These are intentionally deferred:

- flat-file session backend
- SQLite session backend
- Postgres session backend
- MCP client/server integration
- chunk-tree and AST-native editing
- richer host transports such as stdio JSON-RPC
- stronger sandbox integrations for supported platforms
- durable memory stores beyond summary checkpoints

The critical constraint is that none of these should require changing the canonical transcript or prompt-segmentation contract.
