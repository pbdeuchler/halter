// pattern: Imperative Shell

use async_trait::async_trait;
use futures::stream::BoxStream;
use halter_protocol::{
    ProviderCapabilities, ProviderError, ProviderRequest, StreamEvent, ToolCallIdPolicy,
};
use tokio_util::sync::CancellationToken;

use crate::Provider;
use crate::responses_provider::{
    ResponsesProvider, ResponsesProviderConfig, ResponsesProviderRequestConfig,
};

#[derive(Debug, Clone)]
pub struct OpenRouterProvider {
    inner: ResponsesProvider,
}

impl OpenRouterProvider {
    #[must_use]
    pub fn new(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            inner: ResponsesProvider::new(config(), api_key, base_url),
        }
    }
}

#[async_trait]
impl Provider for OpenRouterProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        self.inner.capabilities()
    }

    async fn stream(
        &self,
        request: ProviderRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
        self.inner.stream(request, cancel).await
    }
}

fn config() -> ResponsesProviderConfig {
    ResponsesProviderConfig {
        label: "openrouter",
        capabilities: ProviderCapabilities {
            supports_tools: true,
            supports_streaming: true,
            supports_reasoning: true,
            supports_interleaved_reasoning: false,
            supports_images: true,
            supports_documents: false,
            supports_prompt_cache: false,
            supports_compaction: false,
            supports_tool_result_media: false,
            requires_non_empty_assistant_content: false,
            tool_call_id_policy: ToolCallIdPolicy::ProviderSupplied,
            max_input_tokens: 200_000,
            max_output_tokens: 8_192,
        },
        request: ResponsesProviderRequestConfig {
            store: None,
            include_prompt_cache_key: false,
            include_encrypted_reasoning: false,
            reasoning_summary: None,
        },
        rate_limit_strategy: None,
    }
}

#[cfg(test)]
mod tests {
    use halter_protocol::{
        ApiKind, AssembledPrompt, ModelId, ModelRole, ProviderKind, ProviderName, ResolvedModel,
        SessionId, TurnId,
    };

    use super::*;

    #[tokio::test]
    async fn openrouter_provider_rejects_chat_api_kind() {
        let provider = OpenRouterProvider::new("test-key", "https://openrouter.ai/api");
        let error = match provider
            .stream(
                sample_request(ApiKind::OpenAiChat),
                CancellationToken::new(),
            )
            .await
        {
            Ok(_) => panic!("openrouter provider should reject chat requests"),
            Err(error) => error,
        };

        assert!(
            error
                .to_string()
                .contains("openrouter provider requires openai_responses api kind")
        );
    }

    fn sample_request(api_kind: ApiKind) -> ProviderRequest {
        ProviderRequest {
            session_id: SessionId::new(),
            turn_id: TurnId::new(),
            model: ResolvedModel {
                role: ModelRole::default(),
                id: ModelId::from("default"),
                provider: ProviderName::from("openrouter"),
                provider_kind: ProviderKind::OpenRouter,
                api_kind,
                model: "gpt-5".to_owned(),
                max_input_tokens: Some(200_000),
                max_output_tokens: Some(8_192),
                reasoning: None,
                tokens_per_minute: None,
            },
            prompt: AssembledPrompt {
                segments: Vec::new(),
                transcript: Vec::new(),
                ordered_segments: Vec::new(),
                prefix_cache_key: "cache-key".to_owned(),
                rendered_prefix: String::new(),
                rendered_transcript: String::new(),
                rendered: String::new(),
            },
            compacted_prefix: Vec::new(),
            messages: Vec::new(),
            tools: Vec::new(),
            previous_response_id: None,
            new_messages_start: 0,
        }
    }
}
