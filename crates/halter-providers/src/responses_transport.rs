// pattern: Imperative Shell

use async_openai::{Client, config::OpenAIConfig, types::responses::ResponseStream};
use reqwest::Client as ReqwestClient;
use serde_json::Value;
use tokio::select;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone)]
pub(crate) struct ResponsesTransport {
    client: Client<OpenAIConfig>,
}

impl ResponsesTransport {
    #[must_use]
    pub(crate) fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        let config = OpenAIConfig::new()
            .with_api_key(api_key)
            .with_api_base(api_base_url(base_url.into()));
        let http_client = ReqwestClient::builder()
            .user_agent(concat!("halter/", env!("CARGO_PKG_VERSION")))
            .build()
            .expect("openai transport client must build");
        let client = Client::with_config(config).with_http_client(http_client);
        Self { client }
    }

    pub(crate) async fn responses_stream(
        &self,
        request: Value,
        cancel: CancellationToken,
    ) -> anyhow::Result<ResponseStream> {
        let responses = self.client.responses();
        let response = select! {
            _ = cancel.cancelled() => anyhow::bail!("failed to execute provider request: request cancelled"),
            result = responses.create_stream_byot(request) => result,
        };
        response.map_err(|error| anyhow::anyhow!("failed to execute provider request: {error}"))
    }
}

fn api_base_url(base_url: String) -> String {
    format!("{}/v1", base_url.trim_end_matches('/'))
}
