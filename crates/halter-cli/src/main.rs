// pattern: Imperative Shell

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    halter_cli::run().await
}
