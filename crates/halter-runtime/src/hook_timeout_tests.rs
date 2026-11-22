use super::*;

use std::time::Duration;

use async_trait::async_trait;
use futures::{StreamExt, stream, stream::BoxStream};
use halter_protocol::{
    ApiKind, BlockId, MessageId, ModelRole, ProviderCapabilities, ProviderError, ProviderKind,
    ProviderName, ProviderRequest, ResolvedModel, StopReason, StreamEvent,
};
use halter_providers::{ModelRegistry, Provider, ProviderTimeouts, ResiliencePolicy, RetryPolicy};

const BLOCK: &str = r#"{"decision":"block","reason":"denied by prompt hook"}"#;

/// Answers [`BLOCK`] after `delay`, reporting `policy` as its own timers.
struct SlowBlockingProvider {
    delay: Duration,
    policy: Option<ResiliencePolicy>,
}

#[async_trait]
impl Provider for SlowBlockingProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::default()
    }

    fn resilience_policy(&self) -> Option<ResiliencePolicy> {
        self.policy
    }

    async fn stream(
        &self,
        _request: ProviderRequest,
        _cancel: CancellationToken,
    ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
        tokio::time::sleep(self.delay).await;
        let (message, block) = (MessageId::new(), BlockId::new());
        Ok(stream::iter(vec![
            Ok(StreamEvent::MessageStart {
                id: message.clone(),
            }),
            Ok(StreamEvent::TextStart { id: block.clone() }),
            Ok(StreamEvent::TextDelta {
                id: block.clone(),
                delta: BLOCK.to_owned(),
            }),
            Ok(StreamEvent::TextEnd { id: block }),
            Ok(StreamEvent::MessageEnd {
                id: message,
                stop_reason: StopReason::EndTurn,
                response_id: None,
            }),
        ])
        .boxed())
    }
}

fn policy(request: Duration, stream_idle: Duration) -> ResiliencePolicy {
    ResiliencePolicy {
        timeouts: ProviderTimeouts {
            connect: Duration::from_millis(5),
            request,
            stream_idle,
            compact: request,
        },
        request_retry: RetryPolicy {
            max_attempts: 2,
            base_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_millis(1_000),
            deadline: Duration::from_secs(5),
            jitter_pct: 50,
        },
    }
}

#[test]
fn recovery_window_covers_one_detected_stall_and_one_retry() {
    let ms = Duration::from_millis;
    let single = ResiliencePolicy {
        request_retry: RetryPolicy {
            max_attempts: 1,
            ..policy(ms(10), ms(10)).request_retry
        },
        ..policy(ms(10), ms(10))
    };
    let capped = ResiliencePolicy {
        request_retry: RetryPolicy {
            max_backoff: ms(120),
            ..policy(ms(10), ms(10)).request_retry
        },
        ..policy(ms(10), ms(10))
    };
    for (name, policy, want) in [
        // 2 * max(200, 50) + 100 * 1.5
        ("request dominates", policy(ms(200), ms(50)), ms(550)),
        ("stream_idle dominates", policy(ms(50), ms(300)), ms(750)),
        ("no retry: one attempt, no backoff", single, ms(10)),
        ("backoff capped by max_backoff", capped, ms(140)),
        (
            "defaults",
            ResiliencePolicy::default(),
            Duration::from_secs(120) + ms(625),
        ),
    ] {
        assert_eq!(policy.recovery_window(), want, "{name}");
    }
}

#[test]
fn model_hook_timeout_is_raised_to_the_provider_recovery_window() {
    let ms = Duration::from_millis;
    let window = policy(ms(200), ms(50)); // 550ms
    for (name, configured, policy, want) in [
        (
            "no provider policy keeps the configured timeout",
            ms(30),
            None,
            ms(30),
        ),
        ("shorter timeout is raised", ms(30), Some(window), ms(550)),
        ("longer timeout is kept", ms(900), Some(window), ms(900)),
    ] {
        assert_eq!(model_hook_timeout(configured, policy), want, "{name}");
    }
}

/// Finding 1: a 50ms prompt hook around a provider whose own timers are far
/// longer used to time out (failing open and dropping the block) before the
/// provider could answer or recover. Rows: the provider reports its policy,
/// so the hook waits out the 150ms answer and the block is delivered; the
/// provider reports none, so the configured 50ms stands and the hook times
/// out. Mutation killed: `model_hook_timeout` returning `configured`
/// (row 1 times out).
#[tokio::test]
async fn prompt_hook_deadline_leaves_room_for_the_provider() {
    let ms = Duration::from_millis;
    for (name, provider_policy, blocks) in [
        (
            "provider policy known",
            Some(policy(ms(200), ms(200))),
            true,
        ),
        ("no provider policy", None, false),
    ] {
        let mut services = RuntimeServices::default();
        let mut models = ModelRegistry::new();
        let model = |id: &str, role| ResolvedModel {
            role,
            id: ModelId::from(id),
            provider: ProviderName::from("slow"),
            provider_kind: ProviderKind::Fake,
            api_kind: ApiKind::Fake,
            model: "slow".to_owned(),
            max_input_tokens: Some(32_000),
            max_output_tokens: Some(4_096),
            reasoning: None,
            tokens_per_minute: None,
        };
        models.set_default_model(model("default", ModelRole::default()));
        models.set_subagent_model(model("subagent", ModelRole::subagent()));
        models.set_small_model(model("small", ModelRole::small()));
        models.register_provider(
            ProviderName::from("slow"),
            Arc::new(SlowBlockingProvider {
                delay: ms(150),
                policy: provider_policy,
            }),
        );
        services.models = Arc::new(models);
        let resources = services.resources.snapshot();
        let session = create_session_seeded(
            Arc::new(services),
            SessionInit::default(),
            SessionState::default(),
            resources,
        )
        .await
        .expect("create session");
        let request = HookDispatchRequest {
            event_name: HookEventName::PreToolUse,
            matcher_value: None,
            payload: json!({"tool_name": "shell"}),
            fired_hook_ids: BTreeSet::new(),
        };
        let config = PromptHookConfig {
            prompt: "Decide: $ARGUMENTS".to_owned(),
            model: None,
        };

        let result = run_prompt(
            &session,
            ms(50),
            &config,
            &request,
            CancellationToken::new(),
        )
        .await;

        match (result, blocks) {
            (Ok(HandlerExecution::Completed(output)), true) => assert!(
                matches!(output.decision, Some(halter_hooks::HookDecision::Block)),
                "{name}: {output:?}"
            ),
            (Err(error), false) => {
                assert!(error.to_string().contains("timed out"), "{name}: {error:#}")
            }
            (other, _) => panic!("{name}: unexpected {other:?}"),
        }
    }
}
