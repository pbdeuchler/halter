# Session Log Review

Started 2026-09-28 at `7878ca6` (v0.6.0). Prompted by
[#210](https://github.com/pbdeuchler/halter/issues/210): the subagent registry
lives only in memory, so a resumed parent cannot reconstruct its children.

**Goal:** the session log fully encodes all session state, and a session can
be fully resumed from its log.

This file doubles as the burn-down tracker. Each finding has a status:
`open`, `fixed (<commit>)`, `wontfix (<reason>)` or `partial`.

## Structural diagnosis

The log is not authoritative. The `SessionState` checkpoint is.
`halter_protocol::fold::apply_event` covers only `messages`,
`compacted_prefix`, `usage_so_far`, `token_ledger` and `context_window`.
`covered_state_matches` compares only those fields. Hydration folds the tail
after `state_sequence` onto the checkpoint. Because every write path that
produces events also writes a full checkpoint, that tail is almost always
empty. The fold is effectively dead code, and about ten `SessionState` fields
(`file_view_cache`, `appended_prompt_segments`, `pending_tool_calls`,
`lineage`, `fired_hook_ids`, `pending_session_start_source`,
`pending_warning_messages`, `last_response_id`, `messages_seen_by_provider`,
`compaction_notifications`) are never derivable from events.

The session also has several writers with no coordinator: the turn loop,
subagent lifecycle hooks, `notify`, manual `compact`, `resume` and `shutdown`.
They all race on `expected_head_sequence`, and the loser's turn fails.

## Findings

### Critical / High

| # | Finding | Confidence | Status |
|---|---------|-----------:|--------|
| 1 | Subagent Start/Stop hook dispatch commits into the parent log mid-turn, so the parent's next `flush_turn_progress` fails with `event log advanced concurrently` (reproduced: `expected head 4, found 6` for Start, `8, found 10` for Stop). | 99% | fixed |
| 2 | The resource snapshot revision hashes only `skill.revision`, plugin name/version, hooks revision and agent revision. v0.6 added `SkillDef.root`, so a skill-bearing snapshot now serialises differently under the same revision. sqlite `store_snapshot` bails with `revision already exists with different data`, so every commit on an upgraded store fails. | 85% | fixed (7573143) |
| 3 | Dangling turns. A failed final commit only calls `live.emit_error` and never writes `TurnFailed`. Shutdown aborts and crashes leave `TurnStarted` open. Resume never reconciles an open turn. | 95% | fixed |
| 4 | A crash during tool execution loses the log record of side effects and usage. The assistant tool-call message and tool results commit only after the whole batch. `pending_tool_calls` is inserted and removed with no commit in between, so it is never persisted non-empty. | 90% | open |
| 5 | The task list (`ToolSessionStore::task_sessions`) lives only in memory, but the compaction strategies promise that todos survive compaction and rollover. Resume loses it. | 95% | open |
| 6 | `fork_context` children start with the parent's messages, which end in an unanswered spawn `tool_use` (strict providers reject this). The inherited state is not in the child's log. The registry itself is in memory only (#210). | 85% | open |

### Medium

| # | Finding | Confidence | Status |
|---|---------|-----------:|--------|
| 7 | `MessageItem` means both "append to transcript" (fold) and "logged only" (`CompactionContext::record`/`infer`). A custom strategy that uses `record` breaks fold parity. | 90% | open |
| 8 | sqlite `load_session` and `replay_after` are separate reads, so a concurrent commit in between produces a spurious `exceeds advertised head` error. | 80% | open |
| 9 | `SessionEventPayload` has no catch-all variant and the log has no schema version, so an older binary cannot read a newer log (and the reverse fails silently). | 90% | open |
| 10 | Resume rebinds to the current resources and ignores the stored snapshot, while manual compact uses the stored one. Per-turn model overrides are not logged. | 80% | open |
| 11 | Out-of-turn writers (`notify`, `compact`, `shutdown`, `resume`) fail an in-flight turn on the conflict check. | 90% | fixed |
| 12 | Shell, pty and browser state and stateful Function hooks reset silently on resume. Once-hook ids are positional, so reordering hooks re-fires or suppresses them. | 75% | open |
| 13 | The notes root falls back to the temp dir, so notes don't survive a reboot. | 70% | open |
| 14 | Model-judge panel sessions are orphaned: their usage never reaches the parent, and the injected advisory is not logged. | 75% | open |

### Low

| # | Finding | Confidence | Status |
|---|---------|-----------:|--------|
| 15 | `run_parent_hook_dispatch` can commit zero events with `Some(state)`. | 90% | fixed |
| 16 | `compaction_notifications` is updated by `context_boundary` without an event. | 90% | open |
| 17 | `file_view_cache` is dead state. | 70% | open |
| 18 | The trace recorder is not reopened on resume. | 70% | open |
| 19 | sqlite runs `synchronous=NORMAL`, so the last commits can be lost on power failure. | 60% | open |
| 20 | The legacy token-ledger migration doesn't match the fold. | 60% | open |
| 21 | During a live parent turn, subagent hooks read `fired_hook_ids` from the parent's last checkpoint, not the turn's in-memory set. A `once` SubagentStart hook can fire twice across two spawns in one turn. State converges (the ids are unioned on commit), but the hook runs twice. | 80% | open |

## Target architecture

1. **One writer per session.** An actor or event loop that serialises every
   write: turns, subagent lifecycle, hook dispatch, notify, compact. This
   closes #1, #11 and #15 at the root.
2. **Make the fold total.** Every `SessionState` field is derived from events:
   `SessionStarted{seed}`, hook latches and prompt segments, notifications, the
   response id on the assistant message, tasks, subagent lifecycle, and a
   separate "logged-only" message event. Add a test-only assertion that
   folding the full log from default equals the checkpoint on every field.
   The checkpoint then becomes a cache.
3. **Hash the whole serialised snapshot into its revision** (#2, urgent).

## Log

Entries are oldest first.

- **#2 fixed.** `ResourceCompiler` now hashes the serialised snapshot plus the
  hook file revisions, so any stored field (skill `root`, `description`,
  manifest fields) changes the revision. Test:
  `snapshot_revision_tracks_every_serialized_field`. Older stores keep their
  old rows. New compiles land under new keys, so nothing needs migrating.

- **#1, #11, #15 fixed.** Added `SessionLeases` (`halter-runtime/src/session_lease.rs`), a per-session write lease with an inbox:
  - Turns, `compact`, `shutdown` and `resume` hold the lease. A second writer waits for it.
  - Out-of-turn hook dispatches (`SubagentStart`, `SubagentStop`, `notify`) queue behind the lease and commit on release, or immediately under the same lock when no writer holds it.
  - `run_parent_hook_dispatch` and its conflict-retry loop are gone.
  - Queued commits skip dispatches that produced no events, which fixes #15's zero-event state commits.
  - The CLI now drains in-flight turns before `session.shutdown`.
  - Tests: `subagent_lifecycle_hooks_queue_behind_the_parent_turn` (the original repro, over both hooks), `session_writers_wait_for_the_in_flight_turn`, and `session_lease::tests`.
  - Residual: the lease is process-local, and #21 is new. Out-of-process writers still rely on `SessionCommitConflict`.

- **#3 fixed.** `SessionState::open_turn` is checkpointed with `TurnStarted` and cleared by the turn's final commit or by `commit_turn_failure`.
  - A failed final commit now goes down the `TurnFailed` path instead of only emitting a live error.
  - An open turn left by a crash, abort or shutdown is closed with `TurnFailed { cancelled: true, retryable: false }`. The next writer on the session does this: `resume` (before `SessionResumed`) or the next turn (before its `TurnStarted`).
  - Tests: `turn_whose_final_commit_fails_is_recorded_as_failed`, `interrupted_turns_are_closed_by_the_next_writer` (resume or next turn × interrupted or not), and `open_turn` assertions in `session_writers_wait_for_the_in_flight_turn`.
  - Residual: `open_turn` is checkpoint-only, like most `SessionState`; the fold does not derive it (target architecture item 2).
