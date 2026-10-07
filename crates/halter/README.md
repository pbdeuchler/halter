# halter

`halter` is a **simple and configurable agent harness and SDK** for building and
operating thoroughbred agents. It assembles config loading, resource compilation,
providers, tools, hooks, policy, runtime sessions, and persistence behind a small
builder API.

## Example

```rust,no_run
use futures::StreamExt;
use halter::prelude::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let harness = Halter::from_config_file("halter.toml").await?;
    let (session, mut events) = harness.new_session(SessionInit::default()).await?;

    let listener = tokio::spawn(async move {
        while let Some(event) = events.next().await {
            println!("{:?}", event?.payload);
        }
        Ok::<(), anyhow::Error>(())
    });

    session.submit(Message::user("Summarize this repository")).await?;
    tokio::signal::ctrl_c().await?;
    session.shutdown(None).await?;
    listener.await??;

    Ok(())
}
```

---

## Features

The `halter` crate keeps optional capabilities out of the default build. No feature is enabled by default. Enable the feature at compile time, then make sure the corresponding tool or session backend is enabled by config and policy.

| Feature          | What it enables                                                                                                                          | Dependencies                                                                                                      | Runtime notes                                                                                                                                                                     |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `advanced-tools` | Enables the advanced `grep` execution path: parallel content searches when possible and `memmap2`-backed reads for larger regular files. | `rayon`, `memmap2`                                                                                                | Applies to the existing `grep` tool. It does not register a new tool name.                                                                                                        |
| `ast-tools`      | Adds the syntax-aware `ast_grep` built-in tool for code search and rewrites.                                                             | `ast-grep-core`, `ast-grep-language`                                                                              | Tool name: `ast_grep`. Actions: `find`, `replace`.                                                                                                                                |
| `browser-tools`  | Adds the `browser` built-in tool for remote browser automation over Chrome DevTools Protocol (CDP).                                      | `playwright-rs`, `reqwest`                                                                                        | Tool name: `browser`. Requires provider configuration, currently `BROWSERBASE_API_KEY` and `BROWSERBASE_PROJECT_ID`, plus Playwright runtime setup. Network policy still applies. |
| `image-tools`    | Adds the `image` built-in tool for local image inspection and transforms.                                                                | `image`                                                                                                           | Tool name: `image`. Actions: `info`, `resize`, `convert`. File reads and writes remain subject to tool policy.                                                                    |
| `pty`            | Adds the `pty` built-in tool for bounded interactive terminal sessions.                                                                  | `portable-pty`                                                                                                    | Tool name: `pty`. Actions: `start`, `write`, `resize`, `kill`. Use this when a plain `shell` command is not enough.                                                               |
| `profiling`      | Adds the `profile` built-in tool for profiling and instrumentation workflows.                                                            | `inferno`                                                                                                         | Tool name exposed to the model: `profile`.                                                                                                                                        |
| `full`           | Convenience rollup for the optional built-in tool families.                                                                              | Same extra dependencies as `advanced-tools`, `ast-tools`, `browser-tools`, `image-tools`, `pty`, and `profiling`. | Does not include `sqlite`; enable `sqlite` separately when persistent session storage is needed.                                                                                  |
| `sqlite`         | Enables SQLite-backed session persistence and the matching config schema.                                                                | `rusqlite`                                                                                                        | Allows `sessions.backend = "sqlite"` and exposes `halter::session::SqliteSessionStore`. The default backend remains memory unless config selects SQLite.                          |
| `telemetry`      | Adds `halter::telemetry`, the opt-in `tracing` subscriber helper the CLI uses (env filter, noisy-target suppression, compact/JSON output). | `tracing-subscriber`                                                                                              | No subscriber is installed unless `try_init`/`try_init_with` is called.                                                                                                           |

`submit` returns `Submission { message_id, sequence }` after input is committed
to the configured session store. Further submissions queue for the next safe
boundary during active execution. `InputDelivered` records when accepted input
enters history; `InputRejected` records removal without delivery. `InputDeferred`
means the input remains queued. These records do not promise a final reply or
that background work has finished.

An input deferred with `ExecutionFailed` requires explicit same-ID retry;
unrelated messages skip it. Untouched followers marked `ExecutionStopped` remain
eligible when a later submission starts work.

`session.status()` reads current foreground activity. `session.subscribe_status()`
returns a watch receiver initialized with the current `Idle`, `Running`, or
`Closed` value. Activity belongs to the live handle and is not restored from the
log. Watch updates may coalesce; an idle session can still own background processes.

A clean stream closure emits one transient `SessionStatusChanged { status: Closed }`
after the final committed event. Its sequence is zero; replay does not contain it.

`session.discard(&message_id)` removes queued input while idle and records
`InputRejected`. It returns whether an entry was removed. Interrupt active
foreground work before discarding; retained deferred entries count toward inbox
capacity.

`interrupt(None)` waits for cancellation and cleanup without closing the handle.
`shutdown(None)` closes the live driver and stream; `harness.resume_session(id)`
reopens the stored conversation idle with fresh handles.

Keep a handle to submit more input across idle periods. Dropping the last clone
releases the session after foreground work, runnable input, background jobs, and
subagents finish. It does not cancel active work. Event streams and status receivers
do not retain the session; dropping a stream does not stop execution. Release runs
cleanup and `SessionEnd` with reason `session_released`. Stored history and deferred
input remain available for resume.

Use `Some(duration)` to bound the caller's wait for cleanup. Expiry returns
`SessionError::TimedOut` and requests forced recovery; call again with `None`
to await settlement. Storage writes or blocking code can delay final settlement.
Queued same-ID retries emit a fresh `InputAccepted` whose
sequence is returned in the receipt, without duplicating input. Already delivered
or rejected IDs start no work; their records remain available through `replay()`.

## More documentation

- Rustdoc API reference: <https://docs.rs/halter>
- Full project README: <https://github.com/pbdeuchler/halter/blob/main/README.md>
