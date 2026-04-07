// pattern: Imperative Shell

use async_openai::error::OpenAIError;
use futures::{StreamExt, channel::mpsc, stream::BoxStream};
use halter_protocol::{ApiKind, ProviderCapabilities, ProviderError, ProviderRequest, StreamEvent};
use tokio::select;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::openai_codec::{self, ResponsesRequestOptions};
use crate::responses_transport::ResponsesTransport;

#[derive(Debug, Clone, Copy)]
pub(crate) struct ResponsesProviderRequestConfig {
    pub store: Option<bool>,
    pub include_prompt_cache_key: bool,
    pub include_encrypted_reasoning: bool,
    pub reasoning_summary: Option<&'static str>,
}

#[derive(Debug, Clone)]
pub(crate) struct ResponsesProviderConfig {
    pub label: &'static str,
    pub capabilities: ProviderCapabilities,
    pub request: ResponsesProviderRequestConfig,
}

#[derive(Debug, Clone)]
pub(crate) struct ResponsesProvider {
    config: ResponsesProviderConfig,
    transport: ResponsesTransport,
}

impl ResponsesProvider {
    #[must_use]
    pub(crate) fn new(
        config: ResponsesProviderConfig,
        api_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            config,
            transport: ResponsesTransport::new(api_key, base_url),
        }
    }

    #[must_use]
    pub(crate) fn capabilities(&self) -> ProviderCapabilities {
        self.config.capabilities.clone()
    }

    pub(crate) async fn stream(
        &self,
        request: ProviderRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
        validate_responses_request(self.config.label, &request)?;
        info!(
            provider = self.config.label,
            session_id = %request.session_id,
            turn_id = %request.turn_id,
            model = %request.model.model,
            message_count = request.messages.len(),
            tool_count = request.tools.len(),
            "starting responses request"
        );

        let request_body = openai_codec::encode_responses_request(
            &request,
            ResponsesRequestOptions {
                stream: true,
                store: self.config.request.store,
                prompt_cache_key: self
                    .config
                    .request
                    .include_prompt_cache_key
                    .then_some(request.prompt.prefix_cache_key.as_str()),
                include_encrypted_reasoning: self.config.request.include_encrypted_reasoning,
                reasoning_summary: self.config.request.reasoning_summary,
            },
        )?;
        debug!(
            provider = self.config.label,
            request_bytes = request_body.to_string().len(),
            "encoded responses request"
        );
        let mut response_stream = self
            .transport
            .responses_stream(request_body, cancel.child_token())
            .await?;
        let mut decoder = openai_codec::ResponsesStreamDecoder::new(&request);
        let (tx, rx) = mpsc::unbounded();
        let provider_label = self.config.label;

        tokio::spawn(async move {
            loop {
                select! {
                    _ = cancel.cancelled() => {
                        warn!(provider = provider_label, "responses request cancelled");
                        let _ = tx.unbounded_send(Err(ProviderError::new(
                            "failed to execute provider request: request cancelled",
                            false,
                        )));
                        return;
                    }
                    item = response_stream.next() => {
                        match item {
                            Some(Ok(event)) => match decoder.decode(event) {
                                Ok(events) => {
                                    for event in events {
                                        if tx.unbounded_send(Ok(event)).is_err() {
                                            return;
                                        }
                                    }
                                }
                                Err(error) => {
                                    error!(provider = provider_label, error = %error, "failed to decode responses stream");
                                    let _ = tx.unbounded_send(Err(ProviderError::new(error.to_string(), false)));
                                    return;
                                }
                            },
                            Some(Err(error)) => {
                                warn!(provider = provider_label, error = %error, "responses stream returned provider error");
                                let _ = tx.unbounded_send(Err(provider_error_from_stream_error(error)));
                                return;
                            }
                            None => {
                                debug!(provider = provider_label, "responses stream completed");
                                return;
                            }
                        }
                    }
                }
            }
        });

        Ok(rx.boxed())
    }
}

fn validate_responses_request(label: &str, request: &ProviderRequest) -> anyhow::Result<()> {
    if request.model.api_kind != ApiKind::OpenAiResponses {
        anyhow::bail!(
            "failed to execute provider request: {label} provider requires openai_responses api kind"
        );
    }

    Ok(())
}

fn provider_error_from_stream_error(error: OpenAIError) -> ProviderError {
    let retryable = matches!(error, OpenAIError::Reqwest(_) | OpenAIError::StreamError(_));
    ProviderError::new(
        format!("failed to execute provider request: {error}"),
        retryable,
    )
}
