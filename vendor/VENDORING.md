# Vendored brush crates

Halter's shell tool embeds the [brush](https://github.com/reubeno/brush) shell
as a library. The two crates in this directory are vendored copies of the
upstream crates.io releases, renamed for publishing, with one functional
addition: cancellation plumbing (a `tokio_util::sync::CancellationToken`
threaded through execution so a running script and its child processes can be
interrupted).

| Directory | Package | Upstream base |
| --- | --- | --- |
| `brush-core-vendored` | `halter-brush-core` 0.5.0 (lib name `brush_core`) | crates.io `brush-core` 0.5.0 |
| `brush-builtins-vendored` | `halter-brush-builtins` 0.2.0 (lib name `brush_builtins`) | crates.io `brush-builtins` 0.2.0 |

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
- `[lints.clippy]`: `cargo_common_metadata = "allow"` (the lint trips on
  halter's sibling workspace packages) and `unwrap_used = "allow"` (upstream's
  own unit tests use `unwrap` and upstream does not run clippy with
  `--all-targets`; halter's CI does).

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

## How halter consumes the divergence

`crates/halter-tools/src/builtin/shell/session.rs` calls
`params.set_cancel_token(token)` on the `ExecutionParameters` it passes to
`Shell::run_string`. Everything else is internal propagation so that
`while : ; do : ; done`, blocked `read`s, and running child processes all
terminate promptly when halter cancels or times out a shell tool call.

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
4. Reapply the cancellation plumbing to the files listed above. The full
   patch set is the commit that introduced this file (git log -- vendor/);
   reimplement idiomatically if upstream restructured.
5. Gates: `cargo fmt --all`,
   `cargo clippy --workspace --all-features --all-targets -- -D warnings`,
   `cargo test --workspace --all-features`, `cargo check --workspace`, and
   `cargo check -p halter-brush-core -p halter-brush-builtins --all-features
   --target x86_64-pc-windows-msvc`.
