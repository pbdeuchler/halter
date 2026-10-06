# Vendored brush crates

Halter's shell tool embeds the [brush](https://github.com/reubeno/brush) shell
as a library. The two crates in this directory are vendored copies of the
upstream crates.io releases, renamed for publishing, with two functional
additions: cancellation plumbing (a `tokio_util::sync::CancellationToken`
threaded through execution so a running script and its child processes can be
interrupted) and a process tracker (so halter can see whether background work
is still running and force-stop it). A handful of mechanical Clippy fixes are
also carried.

| Directory | Package | Upstream base |
| --- | --- | --- |
| `brush-core-vendored` | `halter-brush-core` 0.6.0 (lib name `brush_core`) | crates.io `brush-core` 0.5.0 |
| `brush-builtins-vendored` | `halter-brush-builtins` 0.3.0 (lib name `brush_builtins`) | crates.io `brush-builtins` 0.2.0 |

Everything not listed below is byte-for-byte identical to the upstream crate
contents (`src/`, `README.md`, `LICENSE`). Upstream `examples/` are not
vendored. Upstream 0.5.0 compiles for `x86_64-pc-windows-msvc` on stable
out of the box, so no Windows patches are carried (the pre-0.5.0 fork's
bespoke `sys/windows` layer is gone).

## Intentional divergences

### Both `Cargo.toml`s (packaging + lint accommodation)

- `name`, `description`, `repository` renamed for the `halter-*` packages.
- `brush-builtins`' `brush-core` dependency points at
  `{ package = "halter-brush-core", path = "../brush-core-vendored" }`.
- `[[example]]` sections removed from brush-core (examples not vendored).
- `[dependencies.tokio-util]` added to brush-core (cancellation).
- `"time"` added to brush-core's non-wasm `tokio` features (the process
  tracker's TERM grace period).
- `[lints.clippy]`: `cargo_common_metadata = "allow"` (the lint trips on
  halter's sibling workspace packages) and `unwrap_used = "allow"` (upstream's
  own unit tests use `unwrap` and upstream does not run clippy with
  `--all-targets`; halter's CI does). `brush-builtins` additionally sets
  `unused_async_trait_impl = "allow"`: clippy 1.98 added that pedantic lint,
  and upstream's builtins implement the async `Execute` trait without
  awaiting. Both crates already allow `unknown_lints`, so older toolchains
  accept the entry.


### `brush-core-vendored/src` (cancellation plumbing)

| File | Change |
| --- | --- |
| `interp.rs` | `ExecutionParameters` gains a private `cancel_token: Option<CancellationToken>` field and `set_cancel_token` / `cancel_token` / `is_cancelled` methods (`set_cancel_token` is the halter-facing entry point); `ensure_not_cancelled` helper; cancellation checks at the entry of every `Execute`/`ExecuteInPipeline` impl and inside each loop body; pipeline and coprocess waits pass the token through. |
| `processes.rs` | `ChildProcess::wait` takes `Option<CancellationToken>`; new `ProcessWaitResult::Cancelled` variant returned when the token fires first. |
| `results.rs` | `ExecutionSpawnResult::wait` takes `Option<CancellationToken>`; `Cancelled` maps to exit code 130 (128 + SIGINT). |
| `jobs.rs` | `JobTask::wait` passes `None` (background jobs are not cancellable via token) and defensively maps `Cancelled` to 130. |
| `commands.rs` | `ExecutionContext::cancel_token` / `is_cancelled` accessors (consumed by the builtins crate). |
| `shell/funcs.rs` | Function invocation wait passes the params' token. |
| `sys/tokio_process.rs` | `kill_on_drop(true)` so children abandoned after cancellation are killed instead of leaked. |

### `brush-builtins-vendored/src` (cancellation plumbing)

| File | Change |
| --- | --- |
| `command.rs` | `command` builtin captures the context's cancel token and passes it to `ExecutionSpawnResult::wait`. |
| `read.rs` | `InputReader` carries an `is_cancelled` closure; input waits poll in bounded (100 ms) slices on Unix so cancellation interrupts a blocked `read`; cancellation surfaces as `ErrorKind::Interrupted`. Non-Unix platforms keep upstream behavior plus a pre-read cancellation check. Two unit tests cover the cancelled and not-cancelled paths. |

### `brush-core-vendored/src` (process tracker)

Lets halter tell whether a shell session still has background work running
and stop it without taking the shell's execution lock.

| File | Change |
| --- | --- |
| `processes.rs` | New public `ProcessTracker` (`Clone`/`Default`, `with_activity(watch::Sender<()>)`, `has_running()`, `force_stop()`) recording owned child PIDs/PGIDs and in-flight async shell tasks. `ChildProcess` registers with it (`track`) and deregisters on reap (`retire`). A cancelled `wait` now stops the child instead of abandoning it: SIGTERM to its process group (never the harness's own group), SIGKILL after a 500 ms grace period, then reap; Windows uses `taskkill /T /F`. `Drop` SIGKILLs a still-owned child. On macOS, `kill` can return `EPERM` for a group TERM has already emptied, so `process_group_is_gone` checks with libproc (`proc_listpids`/`proc_pidinfo`) that only zombies remain before treating it as gone; on other Unix platforms it always returns `false`. `force_stop()` itself is an immediate SIGKILL (no grace period). |
| `interp.rs` | `ExecutionParameters` gains a `process_tracker` field, public `set_process_tracker` (halter-facing) and crate-private `process_tracker`. Async `&` lists and coprocesses hold a `start_task()` guard for the life of their spawned task. |
| `commands.rs` | `execute_external_command` calls `child.track(context.params.process_tracker())` on each spawned child. |

### Clippy cleanups (no behavior change)

Mechanical rewrites so `cargo clippy --all-targets -- -D warnings` stays clean
on newer toolchains (Rust 1.99 at last re-check). Drop any that upstream has
already fixed.

| File | Change |
| --- | --- |
| `brush-core-vendored/src/history.rs` | `Search::next` uses `let index = self.next_index?;` instead of an `if let … else { return None }` block. |
| `brush-core-vendored/src/completion.rs` | `try_get_variable_completions` uses `token.strip_prefix('$')?` instead of an `else if let … else { return None }` chain. |
| `brush-core-vendored/src/interp.rs` | `case` tracing formats `self.value`, not `&self.value`. |
| `brush-core-vendored/src/expansion.rs`, `extendedtests.rs` | `#[allow(clippy::double_must_use)]` on the `#[async_recursion]` fns `expand_word_piece` and `eval_extended_test_expr`. |
| `brush-builtins-vendored/src/bind.rs` | `bind_key_sequence_to_readline_target` returns one `Ok(())` after the `match` instead of one per arm. |
| `brush-builtins-vendored/src/trap.rs` | Trap listing formats `handler.command`, not `&handler.command`. |
| `brush-builtins-vendored/src/read.rs` | `test_build_array_fields_none_input` uses `assert_eq!` against an empty `Vec` instead of `assert!(…is_empty())`. |

## How halter consumes the divergence

`crates/halter-tools/src/builtin/shell/session.rs` calls
`params.set_cancel_token(token)` on the `ExecutionParameters` it passes to
`Shell::run_string`. Everything else is internal propagation so that
`while : ; do : ; done`, blocked `read`s, and running child processes all
terminate promptly when halter cancels or times out a shell tool call.

The same file builds one `ProcessTracker::with_activity(..)` per shell session
and passes it with `params.set_process_tracker(..)`. halter calls
`has_running()` to decide whether a session still has background jobs
(`crates/halter-tools/src/session_store.rs`) and `force_stop()` to tear them
down when a session is force-stopped or dropped.

## Re-vendoring procedure (upgrading upstream)

1. Download and unpack the new releases:
   `curl -L https://static.crates.io/crates/brush-core/brush-core-<V>.crate | tar xz`
   (same for `brush-builtins`; their versions must be mutually compatible —
   check `brush-builtins`' `brush-core` requirement).
2. Replace `src/`, `README.md`, and `LICENSE` in each vendored directory
   wholesale with the upstream contents.
3. Replace each `Cargo.toml` with the upstream (crates.io-normalized) one and
   reapply the packaging + lint edits listed above; bump the versions in the
   root `Cargo.toml` `[workspace.dependencies]` (including `brush-parser`).
4. Reapply the cancellation plumbing, the process tracker, and any Clippy
   cleanups still needed, per the tables above. The patch history is
   `git log -- vendor/` (cancellation came with the commit that introduced
   this file, the tracker in later session-lifecycle commits);
   reimplement idiomatically if upstream restructured.
5. Gates: `cargo fmt --all`,
   `cargo clippy --workspace --all-features --all-targets -- -D warnings`,
   `cargo test --workspace --all-features`, `cargo check --workspace`, and
   `cargo check -p halter-brush-core -p halter-brush-builtins --all-features
   --target x86_64-pc-windows-msvc`.
