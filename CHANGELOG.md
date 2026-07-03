# Changelog

All notable changes to this project are documented in this file.

The format loosely follows [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/)
once a `1.0.0` line is cut.

## [Unreleased]

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
  covered fields for every backend.
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
  log tail.
- `HalterSession::export_trace()` / `halter_runtime::export_session_trace`:
  serialize a session's trace (including subagent sessions) from the store's
  event log in the trace-file format, available with or without a configured
  `traces_dir`. (A `halter trace` CLI subcommand over this is follow-up
  work.)
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
