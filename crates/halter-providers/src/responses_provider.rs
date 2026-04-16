// pattern: Imperative Shell

use async_openai::error::OpenAIError;
use futures::{StreamExt, channel::mpsc, stream::BoxStream};
use halter_protocol::{
    ApiKind, ProviderCapabilities, ProviderCompactionRequest, ProviderCompactionResponse,
    ProviderError, ProviderRequest, StreamEvent,
};
use tokio::select;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};

use crate::openai_codec::{self, ResponsesRequestOptions};
use crate::openai_error::{
    openai_api_error_is_rate_limit, openai_message_is_rate_limit, parse_openai_stream_error,
};
use crate::openai_rate_limit_policy::estimate_openai_request_cost;
use crate::responses_transport::{
    ResponsesRateLimitStrategy, ResponsesTransport, ResponsesTransportRequest,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CompactStrategy {
    DedicatedEndpoint,
    InlineResponses,
}

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
    pub compact_strategy: Option<CompactStrategy>,
    pub rate_limit_strategy: Option<ResponsesRateLimitStrategy>,
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
        let mut capabilities = self.config.capabilities.clone();
        capabilities.supports_compaction = self.config.compact_strategy.is_some();
        capabilities
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
        let request_bytes = request_body.to_string().len();
        debug!(
            provider = self.config.label,
            request_bytes, "encoded responses request"
        );
        let request_meta = ResponsesTransportRequest {
            provider_label: self.config.label,
            model: request.model.model.clone(),
            reservation: estimate_openai_request_cost(
                request_bytes,
                request.model.max_output_tokens,
            ),
            rate_limit_strategy: self.config.rate_limit_strategy,
            tokens_per_minute: request.model.tokens_per_minute,
        };
        let track_response_id = self.config.request.store != Some(false);
        let (tx, rx) = mpsc::unbounded();
        let provider_label = self.config.label;
        let transport = self.transport.clone();

        tokio::spawn(async move {
            loop {
                let mut response_stream = match transport
                    .responses_stream(
                        request_body.clone(),
                        request_meta.clone(),
                        cancel.child_token(),
                    )
                    .await
                {
                    Ok(stream) => stream,
                    Err(error) => {
                        let message = error.to_string();
                        if openai_message_is_rate_limit(&message) {
                            warn!(
                                provider = provider_label,
                                error = %message,
                                "retrying responses request after rate limit"
                            );
                            continue;
                        }
                        let _ = tx.unbounded_send(Err(provider_error_from_transport_error(error)));
                        return;
                    }
                };
                let mut decoder =
                    openai_codec::ResponsesStreamDecoder::new(&request, track_response_id);
                let mut pending_events = Vec::new();
                let mut committed = false;

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
                                        if committed {
                                            for event in events {
                                                if tx.unbounded_send(Ok(event)).is_err() {
                                                    return;
                                                }
                                            }
                                            continue;
                                        }

                                        pending_events.extend(events);
                                        if pending_events.iter().any(stream_event_commits_attempt) {
                                            committed = true;
                                            for event in pending_events.drain(..) {
                                                if tx.unbounded_send(Ok(event)).is_err() {
                                                    return;
                                                }
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
                                    if !committed && stream_error_is_retryable(&error) {
                                        warn!(
                                            provider = provider_label,
                                            error = %error,
                                            "retrying responses stream after rate limit"
                                        );
                                        break;
                                    }
                                    warn!(provider = provider_label, error = %error, "responses stream returned provider error");
                                    let _ = tx.unbounded_send(Err(provider_error_from_stream_error(error)));
                                    return;
                                }
                                None => {
                                    if !pending_events.is_empty() {
                                        for event in pending_events.drain(..) {
                                            if tx.unbounded_send(Ok(event)).is_err() {
                                                return;
                                            }
                                        }
                                    }
                                    debug!(provider = provider_label, "responses stream completed");
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        });

        Ok(rx.boxed())
    }

    pub(crate) async fn compact(
        &self,
        request: ProviderCompactionRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<ProviderCompactionResponse> {
        validate_responses_compaction_request(self.config.label, &self.config, &request)?;
        info!(
            provider = self.config.label,
            session_id = %request.session_id,
            model = %request.model.model,
            compacted_prefix_items = request.compacted_prefix.len(),
            message_count = request.messages.len(),
            "starting responses compaction request"
        );

        match self.config.compact_strategy {
            Some(CompactStrategy::DedicatedEndpoint) => {
                self.compact_via_endpoint(&request, cancel).await
            }
            Some(CompactStrategy::InlineResponses) => {
                self.compact_via_responses(&request, cancel).await
            }
            None => anyhow::bail!(
                "failed to compact session: {} provider does not support compaction",
                self.config.label
            ),
        }
    }
}

impl ResponsesProvider {
    async fn compact_via_endpoint(
        &self,
        request: &ProviderCompactionRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<ProviderCompactionResponse> {
        let request_body = openai_codec::encode_responses_compact_request(request)?;
        let request_bytes = request_body.to_string().len();
        let response = self
            .transport
            .responses_compact(
                request_body,
                self.compaction_transport_request(request, request_bytes),
                cancel,
            )
            .await?;
        openai_codec::decode_responses_compact_response(&response)
    }

    async fn compact_via_responses(
        &self,
        request: &ProviderCompactionRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<ProviderCompactionResponse> {
        let request_body = openai_codec::encode_openrouter_compact_request(request)?;
        let request_bytes = request_body.to_string().len();
        let response = self
            .transport
            .responses_json(
                request_body,
                self.compaction_transport_request(request, request_bytes),
                cancel,
            )
            .await?;
        openai_codec::decode_openrouter_compact_response(&response)
    }

    fn compaction_transport_request(
        &self,
        request: &ProviderCompactionRequest,
        request_bytes: usize,
    ) -> ResponsesTransportRequest {
        ResponsesTransportRequest {
            provider_label: self.config.label,
            model: request.model.model.clone(),
            reservation: estimate_openai_request_cost(
                request_bytes,
                request.model.max_output_tokens,
            ),
            rate_limit_strategy: self.config.rate_limit_strategy,
            tokens_per_minute: request.model.tokens_per_minute,
        }
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

fn validate_responses_compaction_request(
    label: &str,
    config: &ResponsesProviderConfig,
    request: &ProviderCompactionRequest,
) -> anyhow::Result<()> {
    if request.model.api_kind != ApiKind::OpenAiResponses {
        anyhow::bail!(
            "failed to compact session: {label} provider requires openai_responses api kind"
        );
    }
    if config.compact_strategy.is_none() {
        anyhow::bail!("failed to compact session: {label} provider does not support compaction");
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use halter_protocol::{
        ModelId, ModelRole, ProviderKind, ProviderName, ResolvedModel, SessionId,
    };

    use super::*;

    #[tokio::test]
    async fn responses_provider_without_compaction_strategy_rejects_compaction() {
        let provider = ResponsesProvider::new(
            ResponsesProviderConfig {
                label: "responses-test",
                capabilities: ProviderCapabilities {
                    supports_compaction: true,
                    ..ProviderCapabilities::default()
                },
                request: ResponsesProviderRequestConfig {
                    store: None,
                    include_prompt_cache_key: false,
                    include_encrypted_reasoning: false,
                    reasoning_summary: None,
                },
                compact_strategy: None,
                rate_limit_strategy: None,
            },
            "test-key",
            "http://127.0.0.1:1",
        );

        let error = provider
            .compact(sample_compaction_request(), CancellationToken::new())
            .await
            .expect_err("compaction should fail without a strategy");

        assert!(
            error
                .to_string()
                .contains("responses-test provider does not support compaction")
        );
        assert!(!provider.capabilities().supports_compaction);
    }

    fn sample_compaction_request() -> ProviderCompactionRequest {
        ProviderCompactionRequest {
            session_id: SessionId::new(),
            model: ResolvedModel {
                role: ModelRole::default(),
                id: ModelId::from("default"),
                provider: ProviderName::from("responses-test"),
                provider_kind: ProviderKind::OpenAi,
                api_kind: ApiKind::OpenAiResponses,
                model: "gpt-5".to_owned(),
                max_input_tokens: Some(200_000),
                max_output_tokens: Some(8_192),
                reasoning: None,
                tokens_per_minute: None,
            },
            compacted_prefix: Vec::new(),
            messages: Vec::new(),
            tools: Vec::new(),
            instructions: "Summarize".to_owned(),
        }
    }
}

fn provider_error_from_stream_error(error: OpenAIError) -> ProviderError {
    match error {
        OpenAIError::ApiError(api_error) => ProviderError::new(
            format!("failed to execute provider request: {}", api_error.message),
            openai_api_error_is_rate_limit(&api_error),
        ),
        OpenAIError::JSONDeserialize(json_error, content) => {
            if let Some(api_error) = parse_openai_stream_error(&content) {
                return ProviderError::new(
                    format!("failed to execute provider request: {}", api_error.message),
                    openai_api_error_is_rate_limit(&api_error),
                );
            }
            ProviderError::new(
                format!(
                    "failed to execute provider request: failed to deserialize api response: error:{json_error} content:{content}"
                ),
                false,
            )
        }
        other => {
            let retryable = matches!(other, OpenAIError::Reqwest(_) | OpenAIError::StreamError(_));
            ProviderError::new(
                format!("failed to execute provider request: {other}"),
                retryable,
            )
        }
    }
}

fn provider_error_from_transport_error(error: anyhow::Error) -> ProviderError {
    let message = error.to_string();
    ProviderError::new(message.clone(), openai_message_is_rate_limit(&message))
}

fn stream_error_is_retryable(error: &OpenAIError) -> bool {
    match error {
        OpenAIError::ApiError(api_error) => openai_api_error_is_rate_limit(api_error),
        OpenAIError::JSONDeserialize(_, content) => parse_openai_stream_error(content)
            .as_ref()
            .is_some_and(openai_api_error_is_rate_limit),
        OpenAIError::Reqwest(_)
        | OpenAIError::StreamError(_)
        | OpenAIError::FileSaveError(_)
        | OpenAIError::FileReadError(_)
        | OpenAIError::InvalidArgument(_) => false,
    }
}

fn stream_event_commits_attempt(event: &StreamEvent) -> bool {
    matches!(
        event,
        StreamEvent::TextDelta { .. }
            | StreamEvent::ThinkingDelta { .. }
            | StreamEvent::ToolCallStart { .. }
            | StreamEvent::ToolArgsDelta { .. }
            | StreamEvent::ToolCallEnd { .. }
            | StreamEvent::MessageEnd { .. }
            | StreamEvent::ProviderWarning { .. }
            | StreamEvent::Error { .. }
    )
}
