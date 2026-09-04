# Changelog

All notable changes to this project are documented in this file.

The format loosely follows [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/)
once a `1.0.0` line is cut.

## [Unreleased]

### Added

- `[providers.openrouter.routing]` pins which upstream provider OpenRouter
  routes to. `order` lists OpenRouter provider slugs most-preferred first and
  `allow_fallbacks = false` turns that list into an exact allowlist. The
  preference is sent as the `provider` object of every OpenRouter request body,
  streaming turns and compaction alike. SDK callers pass the same
  `halter_protocol::OpenRouterRouting` to
  `OpenRouterProvider::new_with_headers(...)`.
  `OpenRouterRouting::normalized` is the single door for the invariant — it
  trims slugs and rejects an empty order, blank slugs, and repeated slugs —
  and both the config resolver and the provider constructors apply it, so a
  value handed straight to the SDK cannot reach a request body unchecked.
- `policy.shell.allow = ["*"]` allows every program, mirroring the wildcard
  `policy.network.allowed_hosts` already accepts. Deployments that want
  arbitrary shell execution no longer have to allowlist `bash` and route every
  command through `bash -c`. `policy.shell.mode` remains the only shell
  restriction under a wildcard, and an empty list still denies everything.
- `policy.allowed_read_roots`, `policy.sensitive_path_patterns`, and
  `policy.shell.mode` — the last `PolicySettings` fields the builder pinned to
  their defaults — are now configurable. Unset read roots keep the built-in
  working-directory and temporary-directory roots; an explicit list replaces
  them, and configured `allowed_write_roots` stay readable either way. Unset
  sensitive patterns keep the built-in globs; an explicit list replaces them
  and an empty list disables the check. `mode` is `"strict"` (default) or
  `"relaxed"`.
- Compaction v2, part 1: a session **token ledger**, Halter-owned triggers, a
  hard context cap, and a public compaction strategy API (#194).
  - `SessionState::token_ledger` (`halter_protocol::TokenLedger`) carries the
    context size the provider reported with the last completed assistant
    response plus a heuristic estimate of every message appended since. It is
    advanced by `SessionState::append`, the single door for transcript growth
    that the runtime and the event fold share, and rebuilt from the compacted
    state by compaction. Plan-time transcript scans are gone.
  - Compaction now runs at two Halter-side trigger points: immediately after
    an assistant response with no tool calls, and before every provider
    request once the appends since the last one have landed, so a tool call is
    never separated from its result. Server-side auto-compaction is never used;
    the invariant is documented on `CompactionStrategy`.
  - `context.max_tokens` hard-caps the session context. It defaults to
    `models.default.max_input_tokens` and is checked before every provider
    request, after compaction has had its chance; a turn past it fails with
    `halter_runtime::ContextCapExceeded`.
  - `context.compaction_threshold` defaults to `models.default.max_input_tokens`
    minus `COMPACTION_HEADROOM_TOKENS` (20,000) and `context.pre_compaction_target`
    to three quarters of the resolved threshold. A config that sets neither the
    threshold nor the window fails when loaded or built. `ContextConfig::resolve`
    and `HarnessConfig::resolved_context` expose the resolved values.
  - `halter_runtime::CompactionStrategy` (with `CompactionContext`,
    `CompactionTrigger`, and `CompactionEffects`) is the seam above the provider:
    the runtime owns *when*, the strategy owns *what happens*, and may also
    contribute tools, system-prompt segments, and threshold reminders. Install
    one with `HalterBuilder::with_compaction(...)`; the default
    `ProviderCompaction` re-homes the provider-delegated flow. Re-exported from
    `halter::compaction`.
  - `PreCompact`/`PostCompact` hooks now fire around automatic compaction too,
    with trigger `auto`; a blocking `PreCompact` hook skips the pass with a
    `Warning` event.

### Changed

- **Breaking:** `OpenRouterProvider::new_with_headers` and
  `OpenRouterProvider::new_with_headers_and_resilience` take an additional
  `Option<OpenRouterRouting>` argument after `temperature`. Pass `None` for the
  previous behavior. `OpenRouterProvider::new` is unchanged.
- **Breaking:** `ContextConfig::compaction_threshold` and
  `pre_compaction_target` are `Option<u64>` and the struct gained `max_tokens`.
  Wrap literal values in `Some(..)` and add `max_tokens: None`. The fixed
  80,000/60,000 defaults are gone (see Added).
- **Breaking:** `ContextManager::plan` no longer takes a compaction model and
  provider, `ContextManager::compact_now` is removed, and `DefaultContextManager`
  is a unit struct (`new`, `from_settings`, and `settings` are gone). Planning
  is read-only; compaction happens before it through the strategy.
- **Breaking:** `CompactionOutcome` is removed. `CompactionEffects` is
  `{ messages, compacted_context, result }` and `apply` returns the
  `CompactionResult` plus the `ContextCompacted` payload that records it.
  `ContextPlan` lost `compaction` and `compaction_warning`.
- **Breaking:** `SessionState::usage_anchor_floor` is replaced by
  `token_ledger`; `estimate_context_tokens`, `find_usage_anchor`, and
  `UsageAnchor` are gone. The estimator (`estimate_text_tokens`,
  `estimate_message_tokens`, `estimate_messages_tokens`,
  `estimate_segment_tokens`, `estimate_summary_tokens`,
  `estimate_compacted_prefix_tokens`, `estimate_json_tokens`, `stable_json`,
  `TokenEstimator`, `CharHeuristicEstimator`) moved to `halter-protocol`.
- **Breaking:** `halter_runtime::resolve_response_chain` takes
  `(last_response_id, messages_seen_by_provider, total_messages, has_compacted_prefix)`;
  the window-size and compacted-this-turn arguments were always derivable.
- **Breaking:** `RuntimeServices` gained `context: ContextSettings` and
  `compaction: Arc<dyn CompactionStrategy>`, and `ContextSettings` gained
  `max_tokens`.

## [0.5.0] - 2026-07-27


The OpenAI Responses adapter no longer fails a turn when the upstream sends a
stream event it does not model. This requires a minor release on the pre-1.0
line because surfacing the event's payload adds a variant to two public
exhaustive enums, which breaks downstream exhaustive matches.

Published crates: `halter`, `halter-config`, `halter-hooks`,
`halter-protocol`, `halter-providers`, `halter-runtime`, `halter-session`,
and `halter-tools`. `halter-cli` also moves to `0.5.0` but remains
`publish = false`.

### Added

- `StreamEvent::ProviderMetadata` and `SessionEventPayload::ProviderMetadata`,
  carrying out-of-band annotations a provider attaches to a response as
  verbatim JSON text. The OpenAI Responses adapter populates them from
  `response.metadata` events, whose payload carries moderation scores and
  verification recommendations. The runtime forwards them to the consumer
  event stream and does not otherwise interpret them.

### Fixed

- The OpenAI Responses stream no longer fails the turn on an event type the
  `async-openai` schema does not model. `response.metadata` — emitted by the
  ChatGPT/Codex backend — aborted the turn with a deserialization error;
  unrecognized `response.*` frames are now logged and skipped, so a single
  unmodelled frame costs one event rather than the whole turn.

### Upgrading from 0.4

- `StreamEvent` and `SessionEventPayload` each gained a variant. Exhaustive
  matches over either enum need a `ProviderMetadata { metadata }` arm;
  consumers with nothing to do with provider annotations can ignore it
  alongside their existing catch-all.

## [0.4.0] - 2026-07-26

This release expands the public `ReasoningEffort` enum, which requires a
minor release on the pre-1.0 line because downstream exhaustive matches must
handle the new variants.

Published crates: `halter`, `halter-config`, `halter-hooks`,
`halter-protocol`, `halter-providers`, `halter-runtime`, `halter-session`,
and `halter-tools`. `halter-cli` also moves to `0.4.0` but remains
`publish = false`.

### Added

- `ReasoningEffort::{None, Minimal, Max}`, serialized as `"none"`,
  `"minimal"`, and `"max"`. OpenAI-compatible requests preserve these wire
  values. Anthropic requests disable thinking for `None`, normalize
  `Minimal` to its lowest supported effort, preserve `Max` for adaptive
  thinking, and cap `Max` at the existing 8,192-token legacy budget.

### Upgrading from 0.3

- Exhaustive matches on `ReasoningEffort` must add the three new variants.
- Use `reasoning = "none"` (not `"non"`) to explicitly disable reasoning in
  configuration. Omitting `reasoning` still leaves provider behavior
  unspecified.

## [0.3.0] - 2026-07-26

This release cuts the `0.3` line. The minor bump is required: public
protocol and runtime types gained fields and `estimate_context_tokens`
gained a parameter, so `0.2` struct literals and call sites do not
compile against it.

Published crates: `halter`, `halter-config`, `halter-hooks`,
`halter-protocol`, `halter-providers`, `halter-runtime`,
`halter-session`, `halter-tools`. `halter-cli` also moves to `0.3.0` but
remains `publish = false`.

This release also carries the first publish of the rebased vendored
shell crates — `halter-brush-core` `0.4.0` → `0.5.0` and
`halter-brush-builtins` `0.1.0` → `0.2.0` — whose version bumps landed
with the brush rebase below and have not shipped before.

### Upgrading from 0.2

- `ContextPlan`, `SessionState`, and `CompactionOutcome` each gained a
  public field. Struct literals need the new field; `..Default::default()`
  construction is unaffected.
- `estimate_context_tokens` takes a trailing `usage_anchor_floor: usize`.
  Pass `SessionState::usage_anchor_floor`, or `0` to keep the previous
  whole-transcript behavior.
- `Usage::input_tokens` from the Anthropic provider now includes cache
  traffic, so reported input totals for Anthropic turns increase. Code
  that summed `input_tokens + cache_read_input_tokens +
  cache_creation_input_tokens` to get a total should now read
  `input_tokens` alone, or `Usage::context_tokens()` for input+output.

### Event-log-unified sessions

Sessions are now **log-authoritative with checkpoints** (see
`docs/event-log-unification.md`): the append-only per-session event log is
the source of truth, the persisted `SessionState` is a checkpoint stamped
with the log position it reflects, and traces/telemetry derive from the same
log.

#### Added

- `halter_protocol::fold` — the pure fold from committed events onto
  `SessionState` (`apply_event`, `fold_events`, `covered_state_matches`),
  covering `messages`, `compacted_prefix`, and `usage_so_far`. The store
  conformance suite now verifies `fold(replay()) == checkpoint` on the
  covered fields for every backend, while property tests exercise ordered
  replay and re-folding from arbitrary checkpoint boundaries.
- `SessionEventPayload::ContextCompacted` carries optional
  `CompactionEventEffects` (post-compaction message window + provider-native
  prefix), making compaction — the one operation that rewrites history —
  reproducible from the log. Legacy effect-less events still deserialize.
- `SessionEventPayload::SessionResumed`, appended by
  `SessionRuntime::resume` so every state mutation advances the log.
- `SessionStore::replay_after(session_id, after_sequence)` (default filters
  `replay`; SQLite pushes the bound into the query), and
  `StoredSession::{state_sequence, head_sequence}` plus
  `StoredSession::new`. Loaders hydrate a lagging checkpoint by folding the
  log tail and reject inconsistent positions, gaps, reordered events, and
  cross-session tails before mutating the checkpoint.
- `HalterSession::export_trace()` / `halter_runtime::export_session_trace`:
  serialize a session's trace (including subagent sessions) from the store's
  event log in the trace-file format, available with or without a configured
  `traces_dir`. Live and exported traces share the version-2 header schema
  (`generated_at`), and export rejects cyclic or duplicate ancestry. (A
  `halter trace` CLI subcommand over this is follow-up work.)
- `Usage::saturating_accumulate`, used by both the runtime and the fold so
  lifetime token counters cannot overflow and the two accumulations cannot
  diverge.

#### Changed

- **Breaking (custom `SessionStore` impls):** `commit` takes
  `expected_head_sequence: Option<u64>` instead of
  `expected_state: Option<SessionState>`; optimistic concurrency is now an
  event-log head check (`SessionCommitConflict` reports expected/actual
  heads) rather than a structural state comparison. `create_session`
  rejects records with non-zero sequences.
- SQLite schema migration v2 adds `sessions.state_sequence`, backfilled to
  each session's log head (v1 states reflected everything committed).
- Mid-turn flushes no longer clone-and-compare the entire expected
  `SessionState` per commit; the turn loop threads a `u64` head instead,
  and state-only intermediate changes ride the next event-ful flush or the
  final turn commit.

### Compaction

#### Fixed

- OpenAI OAuth (ChatGPT Codex) compaction reached the wrong endpoint. The
  OAuth URL rewrite collapsed every Responses-shaped path onto a single
  constant, so `/v1/responses/compact` resolved to `.../codex/responses` —
  the streaming turn endpoint — which rejects compaction bodies with
  `Store must be set to false`. Each ChatGPT-served endpoint now keeps its
  own path, matching the reference Codex client
  (`.../codex/responses/compact`). Automatic compaction was therefore
  unreachable for OAuth sessions, which combined with the issue below made
  every turn past the compaction threshold fail outright.

#### Changed

- Context estimation now anchors on provider-reported usage instead of
  estimating the whole transcript with a character heuristic. When the
  transcript contains a usable report from a completed assistant turn,
  `estimate_context_tokens` takes that figure as ground truth and estimates
  only the messages after it, bounding heuristic error (documented at ±20%
  for code-heavy text) to one turn's tail rather than letting it compound
  across the entire context. Interrupted turns, errored turns, and zero-usage
  reports are rejected as anchors, and a session with none falls back to the
  previous whole-transcript estimate.
- `SessionState` gained `usage_anchor_floor`. Compaction preserves a tail of
  real messages whose usage describes the *pre*-compaction context; without a
  floor, that stale figure re-triggers compaction on every following turn
  indefinitely. Compaction advances the floor past the preserved tail, in
  both the runtime (`CompactionEffects::apply`) and the event fold, so
  replayed and resumed sessions agree. Forked subagents start with no anchor,
  since the parent's reports describe a different system prompt and tool set.
- **`Usage::input_tokens` now uniformly means total input including cache
  traffic.** OpenAI already reported it that way; Anthropic reports cache
  reads and writes as separate counters, and its decoder now folds them in,
  keeping `cache_read_input_tokens` / `cache_creation_input_tokens` as
  breakdown fields. Without this, a mostly-cached Anthropic prompt reported a
  small `input_tokens` and context budgeting under-counted the live context.
  This changes reported Anthropic input totals — including the accumulated
  `usage_so_far` — for turns recorded after the upgrade; historical values in
  existing sessions are unaffected.
- Automatic compaction is now best-effort. A provider that cannot compact —
  failing endpoint, missing capability, or no compaction window — degrades
  the turn to an uncompacted context and emits
  `SessionEventPayload::Warning` instead of failing the turn. `ContextPlan`
  gained `compaction_warning` to carry the reason. Manual `compact()` is
  unchanged and still propagates its errors, since the caller asked for
  compaction explicitly.

Blank-slate review fixes on top of the provider resilience primitive
(issue #183). Highlights:

### Changed

- Vendored brush shell crates rebased wholesale onto upstream releases
  (`brush-core` 0.5.0 and `brush-builtins` 0.2.0, with `brush-parser`
  bumped to 0.4.0); the only functional divergence carried forward is
  the cancellation plumbing, reimplemented on the new base. The
  pre-0.5.0 fork's bespoke Windows layer and other drift are dropped
  (upstream 0.5.0 builds for Windows on stable natively). See
  `vendor/VENDORING.md` for the exact divergence inventory.
- **Breaking:** `AnthropicProvider` is now built around the same
  `ResilientProvider` wrapping as the OpenAI and OpenRouter providers,
  so all provider families share one retry/backoff/classification
  strategy. `AnthropicProvider::new_with_headers_and_timeouts` is
  replaced by `new_with_headers_and_resilience`.
- **Breaking:** workspace MSRV raised from 1.86 to 1.88 (the declared
  1.86 was already unbuildable due to transitive dependencies).
- Session-store optimistic concurrency now uses structural state
  equality in both the SQLite and in-memory backends (previously the
  backends could disagree on conflicts for logically-equal states); a
  shared conformance suite locks the contract in.
- The SQLite session store serves reads from a read-only WAL
  connection pool; writes keep the single writer connection.
- Setup-time provider errors are classified (deterministic
  encode/validation failures are fatal and no longer burn the retry
  budget), Anthropic errors route through the shared retryability
  classifier, and backoff jitter now respects `max_backoff`.
- Hook execution is wired into the turn cancellation graph: an
  interrupted turn aborts in-flight and pending hooks.
- Session hook eviction is session-scoped rather than handle-scoped,
  so subagent hook dispatch no longer resets the parent session's
  stateful hooks.
- Git working-tree probes run off the async executor, once per turn,
  with hostile-repo hardening (`core.fsmonitor`, `core.hooksPath`, and
  ambient git config neutralized).
- CI now covers the default (no-sqlite) feature set, MSRV, a Windows
  check, and `cargo audit`, with per-ref run cancellation.

## [0.2.0] - 2026-06-24

This release cuts the first `0.2` line for the Halter crates. The minor
version bump is intentional: several public protocol, hooks, runtime,
and facade APIs changed in ways that are not patch-compatible with the
`0.1` line.

Published crates:

- `halter`
- `halter-config`
- `halter-hooks`
- `halter-protocol`
- `halter-providers`
- `halter-runtime`
- `halter-session`
- `halter-tools`

`halter-cli` also moves to `0.2.0`, but remains `publish = false`.
The vendored `halter-brush-core` and `halter-brush-builtins` crates are
unchanged in this release.

### Security hardening

- **Capability-oriented tool policy.** `ToolPolicy` is now a typed
  capability trait (`check_read_path`, `check_write_path`,
  `check_process_signal`, `check_shell_enabled`,
  `check_shell_command_strict`, `check_network`,
  `check_subagent_spawn_typed`). The previous name-based
  `check_shell(program)` surface with magic-string bypasses for
  `"shell"` and `"process"` is removed. Every built-in tool routes
  through the new surface. A new `halter-tools/src/policy/security_tests.rs`
  module covers symlink escape, allowlist bypass via
  builtins/functions/aliases, `sh -lc` rc-file inheritance, and
  reads on sensitive paths.
- **Write-path TOCTOU closed.** Canonicalization happens inside the
  blocking task immediately before open/write under the `CanonicalPath`
  parent-fd contract. Applies to `read`, `write`, `edit`, `image`, and
  `ast/replace`.
- **PTY no longer sources user rc files.** `sh -c` replaces `sh -lc`;
  environment is `env_clear()`ed then overlaid with a strict
  `PTY_ENV_ALLOWLIST`.
- **Hook runtime network policy.** All hook URLs flow through
  `policy.check_network`. `allowed_loopback` is explicit and
  deny-by-default; `127.0.0.0/8` is no longer a blanket allow.
  Response bodies stream via `Response::chunk()` with a 1 MiB cap;
  oversize replies surface `HookError::ResponseTooLarge`.
- **Hook-template UTF-8 correctness.** `expand_env_placeholders` uses
  `str`-indexed scanning; multi-byte codepoints no longer corrupt
  template output or HMAC-signed request bodies.
- **`SecretString`.** API keys in `AnthropicProvider`,
  `ResponsesTransport`, and the rate limiter are now `SecretString` with
  redacting `Debug`/`Display`.
- **Instance-scoped rate-limit registry.** The `'static
  OPENAI_RATE_LIMITS` map is gone; each `ResponsesTransport` owns its
  own `Arc<Mutex<HashMap<_,_>>>`, restoring test isolation and removing
  the monotonic-growth leak.

### API changes

- **`MergeConflict.field`.** `MergeConflict.field` is now a typed
  `ConflictField` enum (`UpdatedInput`, `UpdatedOutput`) instead of a
  `&'static str`; the rendered form in `hooks.merge_conflict` tracing
  output is unchanged. `ConflictField` and `MergeConflict` are now
  re-exported from `halter_hooks`.
- **`PanelIsolation`.** Model-judge full-turn panelists can run in
  read-only, shared-full, or worktree isolation mode.
- **`WaitSubagentResponse.target_statuses`.** Timed-out waits now
  include the current status of every requested target.
- **`SessionHandle` / `SessionInner`.** `HalterSession` no longer
  derives `Clone + Drop` over shared state. `SessionHandle` is the
  public cheap-clonable surface; `SessionInner` holds the owned graph.
  Turn submission returns a `JoinHandle`;
  `SessionRuntime::shutdown` drains in-flight turns.
- **`TransportError { Cancelled, Retryable, Fatal }`.** Replaces
  anyhow round-tripping in `ResponsesTransport::stream_response`.
  A single `classify(&OpenAIError) -> Retryability` drives both the
  retry gate and the reported `ProviderError.retryable` flag.
- **`ProviderError::cancelled()` / `is_cancelled()`.** Cancellation is
  now distinguishable from provider failure at the type level.
- **`RetryGate` + `RetryPolicy`.** Retries are bounded by attempt count
  and cumulative deadline with jittered exponential backoff and a
  server-hint cap; the previous unbounded `loop { ... }` keyed on a
  `contains("rate limit")` substring is gone.
- **Commit-then-publish event pipeline.** `make_event` returns a
  `PendingEvent`; `commit_and_publish` is the sole publication surface.
  `SessionEvent.sequence` is crate-private; renumbering is
  `max(sequence)+1` in both `InMemorySessionStore` and the sqlite
  backend.
- **`ToolConcurrency` honored.** `execute_tool_calls` runs `Exclusive`
  tools alone; `ReadOnly`/`ParallelSafe` runs dispatch via
  `futures::join_all`.
- **`ModelRole`** is a closed enum (snake_case serde);
  `ModelRegistry` grows a `plan_model` resolver.
- **`SkillId`** is now content-addressed off the canonical `SKILL.md`
  root; stable across reloads.

### Features

- **Model judge.** `models.default` and `models.subagent` accept
  `"model_judge"`, referencing a shared `[models.model_judge]` block
  with a `default` model, a `synthesis` model, and a `panel`.
  `ModelJudgeProvider` multiplexes each call to the panel, asks the synthesis
  model to stack-rank responses via `rank_responses`, then gives the
  synthesis output to the default model as internal guidance. Panel
  responses, the synthesis message, and rankings are emitted as
  structured `tracing` telemetry on the `halter::model_judge` target.
- **Resource and plugin loading.** `halter-config` now exposes loaded
  skill, plugin, hook, MCP, LSP, executable, output-style, and agent
  resource types. The facade re-exports these from `halter-config`.
- **Remote plugins.** The `remote-plugins` feature adds in-memory
  GitHub plugin loading without forcing callers to unpack plugin
  archives to disk.
- **Prompt configuration.** Config can select a built-in system prompt
  preset, append extra system prompt text, and access built-in prompt
  segment helpers through the runtime and facade crates.
- **Line-numbered reads.** The `read` tool can return line-numbered
  output while preserving byte-limit handling.
- **Software-factory example.** A full example harness was added with
  panel planning, file-output coordination, worktree handling, and
  stricter trigger-role defaults.

### Protocol additions

- `Message::Meta` - out-of-band synthesis messages for model-judge
  guidance.
- `ProviderError::Cancelled` — first-class cancellation signal at the
  provider boundary.
- `SessionEventPayload::Lagged { dropped_events }` — emitted by
  `EventBus` when a broadcast subscriber falls behind.
- `ExecutedHookDispatch` event — hook lifecycle flows through the
  session event pipeline on the same commit boundary as other events.

### Observability

- CLI default log level is `warn` (was `off`). `HALTER_LOG=…`
  overrides.
- `observe_state` populates `git_branch` / `git_dirty` from a real
  `git` probe.
- `tool_output` events include per-PID kill results on
  `process.kill_tree` (`Vec<KillTreeEntry { pid, killed }>`) instead
  of a bare count.

### Correctness fixes

- Pagination in `grep`'s sequential path no longer double-applies
  `offset`.
- `EditTool` reports `occurrences_in_file` + `replacements_applied`
  (old `matches_replaced` field removed).
- Compaction now plans once per turn (was calling `execute_compaction`
  twice); bulk-eviction replaced with iterative evict-until-target.
- PTY `start` reports spawn failures synchronously via a
  `sync_channel<Result<()>>(1)` bridge.
- OpenAI `reset_*` / `Retry-After` durations accept fractional values
  (`1.2s`, `0.5`).
- SQLite migration table has a compile-time strict-monotonicity
  assertion; silently-skipped entries in an unsorted future table are
  now a build error.
- CLI noisy-target log suppressions now layer *under* the user's
  `RUST_LOG` filter instead of overriding it; an explicit
  `RUST_LOG=hyper=trace` is honored while `RUST_LOG=debug` still
  suppresses noisy targets to `warn`. (#99)
- Bounded provider IDs now truncate on character boundaries in release
  builds instead of relying on debug-only ASCII assertions.
- Snapshot truncation preserves line ordering and avoids extra
  `format!` allocations.
- Shell working-directory handling and software-factory worktree resume
  behavior were fixed.

### Release tooling

- `bin/crate-release-candidates` now emits `halter-hooks` before
  `halter-config`, matching the actual dependency graph for fresh
  coordinated crate publishes.

### Known follow-ups

Tracked in the roadmap's "Deferred findings" section:

- **H19** Anthropic incremental streaming.
- **H25** SQLite optimistic-concurrency hash column.
- **M4** subagent parent-context fork-on-write.
- **M13** `SharedFileWriter` contention.
- **M37** glob bounded-heap mtime sort.
- **L2** `session.rs` split.
- **L3** clippy pedantic + `redundant_clone`.
- **L4** `.expect` / `.context` / `anyhow::bail!` consistency.
- **L5** `run_output.rs` signature-stripping fuzzer.
- **L7** `string_wrapper!` phantom-typed `Id<Tag>`.
- **L10** `wiremock` / `mockito` adoption.
- **L11** provider registry `set_/get_` pairs collapse.
- **L28** `tools/process.rs` PID width.

One roadmap item was rescoped rather than deferred:

- **L8** `hooks_runtime.rs` split. At 1759 lines the module remains a
  single coherent trust-boundary surface; splitting would scatter
  policy, HTTP, matcher, and failure-reporting coupling. Dropped from
  the roadmap.
