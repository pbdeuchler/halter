// pattern: Imperative Shell

#[path = "../cli_app.rs"]
mod cli_app;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    cli_app::run().await
}
