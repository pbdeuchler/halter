# Changelog

All notable changes to this project are documented in this file.

The format loosely follows [Keep a Changelog](https://keepachangelog.com/)
and this project adheres to [Semantic Versioning](https://semver.org/)
once a `1.0.0` line is cut.

## [Unreleased] — `pbd/opus-4.7-review`

Feature branch that consolidates the 2026-04-16 codebase review
remediation. Thirteen sub-branches plus a final-polish commit close 121
of the 135 enumerated findings. The remaining 14 are tracked as
follow-ups in
[`docs/plans/2026-04-17-review-remediation-roadmap.md`](docs/plans/2026-04-17-review-remediation-roadmap.md).
Status for every finding is in
[`docs/review-2026-04-16.md`](docs/review-2026-04-16.md) under "Status
matrix".

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

### Protocol additions

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

### Known follow-ups (not blocking this branch)

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
