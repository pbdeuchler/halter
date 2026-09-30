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

    let submission = session.submit(Message::user("Summarize this repository")).await?;
    while let Some(event) = events.next().await {
        let event = event?;
        if event.session_id != *session.id() || event.sequence() < submission.sequence {
            continue;
        }
        println!("{:?}", event.payload);
        match event.payload {
            SessionEventPayload::InputSettled { message_id: id, .. }
            | SessionEventPayload::InputRejected { message_id: id, .. }
            | SessionEventPayload::InputDeferred { message_id: id, .. } if id == submission.message_id => break,
            _ => {}
        }
    }
    session.shutdown(None).await?;

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

`submit` returns `Submission { message_id, sequence }` after input is committed
to the configured session store. While
execution is active, further submissions queue for the next safe boundary.
Ignore events before `submission.sequence`, then match `submission.message_id`
in `InputSettled` for its execution outcome. `InputRejected`
reports hook rejection; `InputDeferred` means the input is still queued.
`interrupt(None)` waits for cancellation and cleanup without closing the handle.
`shutdown(None)` closes the live driver and stream; `harness.resume_session(id)`
reopens the stored conversation idle with fresh handles.
Use `Some(duration)` to bound the caller's wait for cleanup. Expiry returns
`SessionError::TimedOut` and requests forced recovery; call again with `None`
to await settlement. Storage writes or blocking code can delay final settlement.
Queued same-ID retries emit a fresh `InputAccepted` whose
sequence is returned in the receipt, without duplicating input. Already settled
or rejected IDs start no work; their outcomes remain available through `replay()`.

## More documentation

- Rustdoc API reference: <https://docs.rs/halter>
- Full project README: <https://github.com/pbdeuchler/halter/blob/main/README.md>
