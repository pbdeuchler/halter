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
    let session = harness.new_session(SessionInit::default()).await?;

    let mut events = session
        .submit_turn(Turn::user("Summarize this repository"))
        .await?;

    while let Some(event) = events.next().await {
        println!("{:?}", event?.payload);
    }

    Ok(())
}
```

## More documentation

- Rustdoc API reference: <https://docs.rs/halter>
- Full project README: <https://github.com/pbdeuchler/halter/blob/main/README.md>
