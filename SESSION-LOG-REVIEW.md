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
`compacted_prefix`, `usage_so_far`, `token_ledger` and `context_window` (and, since #6, `subagents`).
`covered_state_matches` compares only those fields. Hydration folds the tail
after `state_sequence` onto the checkpoint. Because every write path that
produces events also writes a full checkpoint, that tail is almost always
empty. The fold is effectively dead code, and about ten `SessionState` fields
(`file_view_cache` [removed, #17], `appended_prompt_segments`, `pending_tool_calls`,
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
| 4 | A crash during tool execution loses the log record of side effects and usage. The assistant tool-call message and tool results commit only after the whole batch. `pending_tool_calls` is inserted and removed with no commit in between, so it is never persisted non-empty. | 90% | fixed |
| 5 | The task list (`ToolSessionStore::task_sessions`) lives only in memory, but the compaction strategies promise that todos survive compaction and rollover. Resume loses it. | 95% | fixed |
| 6 | `fork_context` children start with the parent's messages, which end in an unanswered spawn `tool_use` (strict providers reject this). The inherited state is not in the child's log. The registry itself is in memory only (#210). | 85% | fixed |

### Medium

| # | Finding | Confidence | Status |
|---|---------|-----------:|--------|
| 7 | `MessageItem` means both "append to transcript" (fold) and "logged only" (`CompactionContext::record`/`infer`). A custom strategy that uses `record` breaks fold parity. | 90% | fixed |
| 8 | sqlite `load_session` and `replay_after` are separate reads, so a concurrent commit in between produces a spurious `exceeds advertised head` error. | 80% | fixed |
| 9 | `SessionEventPayload` has no catch-all variant and the log has no schema version, so an older binary cannot read a newer log (and the reverse fails silently). | 90% | fixed |
| 10 | Resume rebinds to the current resources and ignores the stored snapshot, while manual compact uses the stored one. Per-turn model overrides are not logged. | 80% | fixed |
| 11 | Out-of-turn writers (`notify`, `compact`, `shutdown`, `resume`) fail an in-flight turn on the conflict check. | 90% | fixed |
| 12 | Shell, pty and browser state and stateful Function hooks reset silently on resume. Once-hook ids are positional, so reordering hooks re-fires or suppresses them. | 75% | open |
| 13 | The notes root falls back to the temp dir, so notes don't survive a reboot. | 70% | fixed |
| 14 | Model-judge panel sessions are orphaned: their usage never reaches the parent, and the injected advisory is not logged. | 75% | fixed |

### Low

| # | Finding | Confidence | Status |
|---|---------|-----------:|--------|
| 15 | `run_parent_hook_dispatch` can commit zero events with `Some(state)`. | 90% | fixed |
| 16 | `compaction_notifications` is updated by `context_boundary` without an event. | 90% | fixed |
| 17 | `file_view_cache` is dead state. | 70% | fixed |
| 18 | The trace recorder is not reopened on resume. | 70% | fixed |
| 19 | sqlite runs `synchronous=NORMAL`, so the last commits can be lost on power failure. | 60% | open |
| 20 | The legacy token-ledger migration doesn't match the fold. | 60% | open |
| 21 | During a live parent turn, subagent hooks read `fired_hook_ids` from the parent's last checkpoint, not the turn's in-memory set. A `once` SubagentStart hook can fire twice across two spawns in one turn. State converges (the ids are unioned on commit), but the hook runs twice. | 80% | fixed |

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

- **#4 fixed.** `execute_tool_calls` checkpoints each batch twice inside a turn: before the batch runs (the assistant `tool_use`, usage, `ToolExecutionStarted`, and a non-empty `pending_tool_calls`) and after its results. Compaction passes `None` and stays atomic.
  - `answer_unresolved_tool_calls` gives every unanswered call of the last assistant message an error result. The error distinguishes "interrupted while running; it may have partially completed" (the call was in `pending_tool_calls`) from "interrupted before it ran".
  - It runs when an interrupted turn is closed, and in `commit_turn_failure`, so the transcript never ends in an unanswered `tool_use`.
  - Tests: `unresolved_tool_calls_get_error_results` (table), `tool_batches_are_checkpointed_and_recovered_after_a_crash` (a second process resumes mid-batch), and `cancelling_a_checkpointed_tool_batch_answers_its_calls`.
  - Residual: tool side effects are not idempotent. The model is told a call may have run, and nothing more. The runtime events a tool emits mid-execution are still buffered until its batch ends.

- **#5 fixed.** The log already holds every task mutation, because each `task` tool result carries the full record of each task it touched. So nothing new is persisted.
  - `TaskList::from_results` folds those results, in log order, back into the list, ids included.
  - `mark_resumed` replays the log and installs the result with `ToolSessionStore::restore_task_session`. That never replaces a list the process already holds.
  - Tests: `from_results_rebuilds_the_list_from_tool_output` (table: full log, mutations only, empty, foreign output; ids continue after a rebuild), `restore_task_session_never_replaces_a_live_list`, and `resume_restores_the_task_list_from_the_log` (fresh process vs same process).
  - Residual: a `PostToolUse` hook that rewrites the task tool's output also rewrites what gets rebuilt. Only `resume` restores the list; a same-process `HalterSession::new` on an existing id shares the in-memory list anyway. `InMemoryTaskStore` and `TaskStore` are unused (`TaskTool` goes through `ToolSessionStore`); they are dead abstractions left for a separate cleanup.

- **#6 partial: the fork's unanswered `tool_use` is fixed.** `build_subagent_state` with `fork_context` now answers the parent's in-flight tool calls (the spawn among them). It reuses `answer_unresolved_tool_calls`, which now takes the result text. The child sees "result delivered to the parent session; this session is the subagent it forked". Test: `forked_state_answers_the_parents_in_flight_tool_calls` (fork mid-spawn, fork without tool calls, no fork).
  - Still open: the inherited state lives only in the child's creation checkpoint, not its log (target architecture item 2). The registry is in memory only (#210).

- **#6 fixed: the subagent registry is in the parent's log (#210).** `SessionEventPayload::SubagentUpdated { record }` carries a `SubagentRecord` (status plus generation). It is the first `SessionState` field the fold owns since the review began: `SessionState::subagents` is an upsert, and a record older than the held generation is ignored.
  - One variant, not one per transition. The status's `state` already says which transition it was.
  - `RuntimeSubagentControl` records the agent at turn start (spawn and `send_input`), turn finish and close. The start record is written before the turn task is spawned, so the same generation's finish always lands after it. The dispatch goes through the lease inbox as `OutOfTurn::Subagent`.
  - The lease holder drains subagent records at every `flush_turn_progress`, not only at release. Records touch no transcript, so this is safe mid-turn. A spawn's record therefore commits with the tool result that hands the model its agent id, so no orphan sweep is needed. Hook dispatches still wait for release.
  - `resume` runs under the lease, in this order: it collects the agent ids this process holds; `mark_resumed` cancels every `Running` record not among them, with "interrupted: the process stopped before this subagent's turn finished; use send_input to continue", in the same commit as `SessionResumed`; then `RuntimeSubagentControl::restore` registers the missing agents with `running: None`.
  - Tests:
    - `subagent_records_older_than_the_held_generation_are_ignored` (fold, table);
    - `resume_rebuilds_subagents_from_the_parent_log` (table: finished before the stop, running at the stop, still running in this process; the log agrees with the registry);
    - `restored_subagents_take_input_and_close` (`send_input` continues an interrupted child in a new process, `close` records `Closed`, and input to a closed agent fails);
    - `the_holder_takes_subagent_records_and_leaves_hooks` (lease, table);
    - `subagent_lifecycle_hooks_queue_behind_the_parent_turn` now also asserts the record commits before `TurnCompleted` and that the checkpoint holds the finished child.
  - Residual:
    - The fork's inherited state is still only in the child's creation checkpoint, not its log.
    - Recording is best-effort: a failed dispatch logs a warning, and the in-memory registry stays authoritative for the process.
    - Two processes resuming the same parent can both think the other's agents are interrupted, because the lease is process-local.
    - The child session is not reconciled on resume; the child's own open turn is closed when `send_input` starts its next turn (#3).

- **#7 fixed.** Added `SessionEventPayload::MessageRecorded`, a message that is logged but not in the transcript. `CompactionContext::record` and `infer` emit it; `append` keeps `MessageItem`. The fold counts a recorded assistant reply's usage (the runtime does too) and does not append it.
  - The real divergence was wider than custom strategies. `ModelSummary`'s own `infer` reply was a `MessageItem`. So any pass that inferred and then returned `Ok(None)` or `Err` without touching the transcript left the fold one message ahead: no `ContextRestored` was written, because the live window hadn't changed.
  - Side fixes: `extract_subagent_output` and the CLI's final-result tracker take the last assistant `MessageItem`. A compaction summary inferred at the end of a turn no longer counts as either. `history.rs` (session search) indexes both variants, so failed compaction exchanges stay searchable.
  - Tests:
    - `turn_commits_keep_fold_and_checkpoint_in_agreement_across_compaction` is now a table. It adds a strategy that records, infers and gives up, and it fails when the fold appends recorded messages;
    - `recorded_messages_count_usage_without_entering_the_transcript` (fold, table);
    - `indexes_appended_and_recorded_messages_only` (history, table).
  - Residual: a strategy that `append`s an inferred reply (instead of `append_unlogged`) logs it twice and double-counts its usage in the fold. This is documented on `append_unlogged`, not enforced. It adds to #9: a pre-change binary cannot read logs that contain the new variant.

- **#8 fixed.** It was not sqlite-specific. `hydrate_stored_session` (runtime) reads the tail with a second `replay_after` call after `load_session`, and any backend can commit in between. The tail is now cut at the loaded `head_sequence` (`take_while`). The events after it belong to a later state, and the caller's own commit conflicts on them through `expected_head_sequence`.
  - The explicit "exceeds advertised head" check is gone. An overlong tail from a malformed backend still fails the final "tail ended at sequence N, expected head" check, and the validation case now asserts that.
  - Test: `hydrate_folds_event_log_tail_onto_checkpoint` now also commits between load and hydrate, and expects hydration as of the loaded head. It failed before the fix with the reported error.

- **#10 fixed.** Rebinding to the current resources is intended (hot-swap), not a resume bug. Every turn does it and stores the snapshot, and `later_turns_commit_latest_resource_snapshot` locks that in. What didn't match was manual `compact`, which ran against the stored snapshot. It now binds to the current resources and stores them.
  - `TurnStarted` carries the turn's `default_model` and `subagent_model` overrides as given (`None` means the blueprint's). The blueprint plus the log now determine every model used.
  - Tests:
    - `later_writers_commit_latest_resource_snapshot` (turn or manual compact);
    - `turn_default_model_override_selects_overridden_provider` asserts the logged override;
    - `turn_started_records_overrides_and_reads_older_logs` (older JSON reads as no override; round-trips).
  - Residual: the log doesn't say which snapshot revision each turn used; only the latest snapshot is stored with the session. Exhaustive `TurnStarted { turn_id }` patterns break (the software-factory example was fixed).

- **#9 fixed** ("gate plus tolerant reader").
  - `halter_protocol::SESSION_LOG_FORMAT` is set to 1. Sqlite migration 3 adds `sessions.log_format` (default 0). `create_session` stamps the current format, and each commit raises the stamp to the current format and refuses sessions stamped newer, before anything is written. The in-memory store has no gate: it cannot outlive its process, so no other build can write to it.
  - `SessionEventPayload::Unknown` (`#[serde(other)]`) lets an older build read a newer log. Unknown kinds decode as `Unknown`, which the fold ignores, while a known kind with a malformed body still fails to decode. Resuming such a session fails at the first commit (`SessionResumed`) with an "upgrade halter" error.
  - Tests:
    - `commit_gates_on_and_raises_the_log_format_stamp` (legacy, current, newer; a mutation that disables the gate fails it);
    - `unknown_event_kinds_decode_as_unknown_and_known_kinds_stay_strict`;
    - `unknown_events_leave_state_untouched`.
  - Residual: binaries from before this change have no gate and fail on unknown kinds. The stamp only takes effect from this build on. Nothing forces a bump of `SESSION_LOG_FORMAT` when a variant or field is added; that relies on review (the constant's doc says when to bump it).

- **#16 fixed.** `context_boundary` now emits `CompactionNotified { id }` before each notification's `MessageItem`, and runs both through `fold::apply_event`, so the runtime and the fold can't drift apart.
  - The fold inserts the id. It already cleared the set on state-rewriting compaction. `compaction_notifications` is now a covered field in `covered_state_matches`.
  - Tests: `strategy_seed_segments_and_boundary_notifications_reach_the_session` and `milestone_notifications_precede_the_runtime_compaction` now fold the replayed log and compare it with the checkpoint; the milestone test covers insert, then clear, then re-insert across a compaction. Both failed before the fix (`{}` vs `{"half-full"}`).
  - `SESSION_LOG_FORMAT` stays at 1 because format 1 is unreleased. Its doc now says to bump it once per release.

- **#21 fixed.** Diagnosis corrected: the turn's own in-memory set is irrelevant (SubagentStart, SubagentStop and Notification handlers fire only out of turn). The gap was dispatches queued behind the lease and not yet in the checkpoint.
  - `HalterSession::load_for_out_of_turn_hooks` unions the checkpoint's `fired_hook_ids` with `SessionLeases::queued_hook_ids`. It reads the queue before the load, so a release that commits in between is seen by the load. `notify` and both subagent hook paths use it; `notify` no longer hydrates just to read the fired set.
  - Tests:
    - `once_hooks_fire_once_across_out_of_turn_dispatches` (no writer, or queued behind a turn); it failed with 2 runs before the fix;
    - `queued_hook_ids_unions_the_queued_dispatches`.
  - Residual: two out-of-turn dispatches that run at the same time can still both fire a `once` hook, because the check and the mark aren't atomic (e.g. two children finishing at once, each running SubagentStop). `spawn_agent` is `Exclusive`, so spawns in one turn are ordered and fixed.

- **#14 fixed.** Diagnosis narrowed: panel sessions are persisted as child sessions (`parent_session_id` = parent) and keep their own usage, as subagents do. The actual gaps were the synthesis inference and the advisory itself.
  - `run_panel_synthesis` returns `(text, usage)`, summed over its rounds, with the last `UsageUpdate` of each message counting.
  - `run_full_turn_deliberation` returns the synthesis as an `AssistantMessage` carrying that usage, plus the guidance `UserMessage`. `run_turn` logs both as `MessageRecorded` through `apply_event`, so the fold's `usage_so_far` counts the synthesis, and adds the usage to `turn_usage`.
  - Tests:
    - `full_turn_judge_logs_its_synthesis_and_guidance` (guided or fallback; checks the logged messages are not in the transcript, the usage, and fold/checkpoint parity). It replaces the two earlier judge tests, and a mutation that drops the recording fails it.
    - `run_panel_synthesis_sums_the_usage_of_every_round`.
  - Residual: the OneShot judge (provider seam) still drops the usage of its panel and synthesis calls, because a provider stream has only one usage per message. The parent log doesn't name the panel session ids; they are found through their `parent_session_id`.

- **#17 fixed.** Confirmed dead: nothing inserted into `file_view_cache`. It was only copied into `ContextPlan::file_views`, which nothing read. Removed the field, the plan field, the five types behind them, and the fold's clear on rollover. Legacy checkpoints that carry the field still deserialize (serde ignores it), and the compacted-context test's legacy JSON keeps it to show that.
  - Residual: pre-#9 binaries can't load checkpoints written from now on, since the field was required. They already can't read the newer event kinds, so nothing new is lost.

- **#18 fixed.** Confirmed only across processes: within one process the writer is never closed (`EvictionGuard` no longer closes it), so a same-process resume kept tracing. A new process lost every event after resume, and the only open path, `create_session_seeded`, used `File::create`, which would have truncated.
  - `TraceRecorder::open_session` does nothing if the session already has a writer. It opens a root trace for appending and writes `trace_header` only when the file is empty.
  - `mark_resumed` calls it.
  - `SubagentControl::restore` calls the new `attach` to alias each restored child to the parent's writer without writing another header. Restored children run turns through `HalterSession::new`, not `resume`.
  - Tests:
    - `resumed_sessions_keep_tracing` (same process, or after a restart; one header, and traced sequences equal the log). The restart case failed before the fix with sequences 9–14 missing.
    - `reopening_a_session_appends_without_a_second_header` (same or fresh recorder). The truncating mutation fails it.
    - `attach_aliases_a_child_only_when_its_parent_is_open`.
    - `restored_subagents_take_input_and_close` now also asserts that the child's post-restart events reach the parent's trace. The attach mutation fails it.
  - Residual: events committed while no process had the session open (for example a store-level `commit` by another tool) are still missing from the live trace. `export_trace()` stays the source of truth.

- **#13 fixed.** Diagnosis narrowed: an explicit `sqlite_path` already put notes beside the database. The gap was `backend = "sqlite"` with no path: the store opened the default database in the data dir, but the notes went to the temp dir, so a reboot lost the notes of every surviving session.
  - `clean_window_notes_root` now derives notes from `halter_session::default_db_path()` in that case. That function was private and is now exported.
  - It is fallible (missing `HOME`), and the builder propagates the error. `open_default` would fail the same way.
  - Memory sessions keep temp-dir notes, since the notes don't outlive the sessions.
  - Test: `clean_window_notes_root_follows_configured_precedence` gained a backend column and a "default sqlite" row. A mutation that drops the new arm fails it.
  - Residual: a custom store passed through `with_session_store` still gets temp-dir notes, because the builder can't tell whether that store is durable. This is documented in the README and should be set via `notes_root`. I considered defaulting everything to the data dir and rejected it: in-memory tests and embedders would leave notes in `$HOME`.
