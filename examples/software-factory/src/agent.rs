use std::error::Error as StdError;
use std::fmt;
use std::path::Path;
use std::time::Duration;

use anyhow::{Context, bail};
use futures::StreamExt;
use halter::prelude::*;
use halter_protocol::{AssistantPart, Message, SessionEventPayload, ToolResult, Turn, Usage};
use serde_json::Value;
use tracing::{debug, info, warn};

use crate::prompts::{
    CODE_REVIEW_MAX_TURNS, FACTORY_TURN_USER_MESSAGE, FactorySystemPrompt,
    session_init_with_appended_context,
};

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct AgentRun {
    pub(crate) text: String,
}

pub(crate) fn json_preview(value: &Value, max_chars: usize) -> String {
    single_line_preview(&value.to_string(), max_chars)
}

pub(crate) fn single_line_preview(text: &str, max_chars: usize) -> String {
    let normalized = text.replace('\n', "\\n").replace('\r', "\\r");
    let mut preview = String::new();
    let mut truncated = false;
    for (index, ch) in normalized.chars().enumerate() {
        if index == max_chars {
            truncated = true;
            break;
        }
        preview.push(ch);
    }
    if truncated {
        preview.push_str("...");
    }
    preview
}

pub(crate) fn tool_result_kind(result: &ToolResult) -> &'static str {
    match result {
        ToolResult::Empty => "empty",
        ToolResult::Text { .. } => "text",
        ToolResult::Json { .. } => "json",
    }
}

pub(crate) fn tool_result_size(result: &ToolResult) -> usize {
    match result {
        ToolResult::Empty => 0,
        ToolResult::Text { text } => text.len(),
        ToolResult::Json { value } => value.to_string().len(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AgentTextRequirement {
    Required,
    Optional,
}

pub(crate) const CODING_STAGE_RETRY_POLICY: AgentStageRetryPolicy = AgentStageRetryPolicy {
    max_attempts: 3,
    base_backoff: Duration::from_secs(5),
    max_backoff: Duration::from_secs(30),
};
pub(crate) const INFERRED_AGENT_STAGE_CAPACITY_BACKOFF: Duration = Duration::from_secs(5);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AgentStageRetryPolicy {
    pub(crate) max_attempts: u32,
    pub(crate) base_backoff: Duration,
    pub(crate) max_backoff: Duration,
}

impl AgentStageRetryPolicy {
    pub(crate) fn delay_after_failure(self, failed_attempt: u32, error: &str) -> Option<Duration> {
        if failed_attempt >= self.max_attempts {
            return None;
        }
        let delay = inferred_agent_stage_backoff_hint(error)
            .unwrap_or_else(|| exponential_agent_stage_backoff(self, failed_attempt));
        Some(delay.min(self.max_backoff))
    }
}

pub(crate) fn exponential_agent_stage_backoff(
    policy: AgentStageRetryPolicy,
    failed_attempt: u32,
) -> Duration {
    let exponent = failed_attempt.saturating_sub(1).min(31);
    let multiplier = 1u128 << exponent;
    let millis = policy
        .base_backoff
        .as_millis()
        .saturating_mul(multiplier)
        .min(policy.max_backoff.as_millis())
        .min(u128::from(u64::MAX));
    Duration::from_millis(millis as u64)
}

pub(crate) fn inferred_agent_stage_backoff_hint(error: &str) -> Option<Duration> {
    let lower = error.to_ascii_lowercase();
    (lower.contains("overloaded")
        || lower.contains("capacity")
        || lower.contains("rate limit")
        || lower.contains("rate_limit")
        || lower.contains("too many requests"))
    .then_some(INFERRED_AGENT_STAGE_CAPACITY_BACKOFF)
}

pub(crate) fn agent_stage_failure_is_retryable(
    retryable: bool,
    cancelled: bool,
    error: &str,
) -> bool {
    !cancelled && (retryable || inferred_agent_stage_backoff_hint(error).is_some())
}

#[derive(Debug)]
pub(crate) struct AgentStageTurnFailure {
    pub(crate) label: String,
    pub(crate) error: String,
    pub(crate) retryable: bool,
    pub(crate) cancelled: bool,
}

impl AgentStageTurnFailure {
    pub(crate) fn should_retry(&self) -> bool {
        agent_stage_failure_is_retryable(self.retryable, self.cancelled, &self.error)
    }
}

impl fmt::Display for AgentStageTurnFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "agent stage {} failed: {}",
            self.label, self.error
        )
    }
}

impl StdError for AgentStageTurnFailure {}

pub(crate) fn agent_stage_error_is_retryable(error: &anyhow::Error) -> bool {
    error.downcast_ref::<AgentStageTurnFailure>().map_or_else(
        || inferred_agent_stage_backoff_hint(&error.to_string()).is_some(),
        AgentStageTurnFailure::should_retry,
    )
}

pub(crate) async fn run_code_review_agent_with_system_prompt(
    harness: &Halter,
    worktree: &Path,
    label: &str,
    prompt: impl Into<String>,
    project_system_prompt: Option<&str>,
) -> anyhow::Result<AgentRun> {
    run_agent_with_prompt_kind(
        harness,
        worktree,
        label,
        prompt,
        FactorySystemPrompt::Coding,
        project_system_prompt,
        AgentTextRequirement::Required,
        Some(CODE_REVIEW_MAX_TURNS),
    )
    .await
}

pub(crate) async fn run_coding_action_with_system_prompt(
    harness: &Halter,
    worktree: &Path,
    label: &str,
    prompt: impl Into<String>,
    project_system_prompt: Option<&str>,
) -> anyhow::Result<()> {
    run_agent_with_prompt_kind_with_retry(
        harness,
        worktree,
        label,
        prompt,
        FactorySystemPrompt::Coding,
        project_system_prompt,
        AgentTextRequirement::Optional,
        CODING_STAGE_RETRY_POLICY,
    )
    .await?;
    Ok(())
}

pub(crate) async fn run_agent_with_system_prompt(
    harness: &Halter,
    worktree: &Path,
    label: &str,
    prompt: impl Into<String>,
    project_system_prompt: Option<&str>,
) -> anyhow::Result<AgentRun> {
    run_agent_with_prompt_kind(
        harness,
        worktree,
        label,
        prompt,
        FactorySystemPrompt::General,
        project_system_prompt,
        AgentTextRequirement::Required,
        None,
    )
    .await
}

pub(crate) async fn run_agent_with_prompt_kind_with_retry(
    harness: &Halter,
    worktree: &Path,
    label: &str,
    prompt: impl Into<String>,
    system_prompt: FactorySystemPrompt,
    project_system_prompt: Option<&str>,
    text_requirement: AgentTextRequirement,
    retry_policy: AgentStageRetryPolicy,
) -> anyhow::Result<AgentRun> {
    let prompt = prompt.into();
    let mut attempt = 1;

    loop {
        match run_agent_with_prompt_kind(
            harness,
            worktree,
            label,
            prompt.clone(),
            system_prompt,
            project_system_prompt,
            text_requirement,
            None,
        )
        .await
        {
            Ok(run) => return Ok(run),
            Err(error) if agent_stage_error_is_retryable(&error) => {
                let message = error.to_string();
                let Some(delay) = retry_policy.delay_after_failure(attempt, &message) else {
                    warn!(
                        stage = label,
                        attempt,
                        max_attempts = retry_policy.max_attempts,
                        error = %message,
                        "agent stage retry budget exhausted"
                    );
                    return Err(error);
                };
                warn!(
                    stage = label,
                    attempt,
                    max_attempts = retry_policy.max_attempts,
                    retry_in_ms = delay.as_millis() as u64,
                    error = %message,
                    "retrying agent stage after transient failure"
                );
                tokio::time::sleep(delay).await;
                attempt += 1;
            }
            Err(error) => return Err(error),
        }
    }
}

pub(crate) async fn run_agent_with_prompt_kind(
    harness: &Halter,
    worktree: &Path,
    label: &str,
    prompt: impl Into<String>,
    system_prompt: FactorySystemPrompt,
    project_system_prompt: Option<&str>,
    text_requirement: AgentTextRequirement,
    max_turns: Option<u32>,
) -> anyhow::Result<AgentRun> {
    let turn_instructions = prompt.into();
    info!(
        stage = label,
        system_prompt = ?system_prompt,
        max_turns = ?max_turns,
        prompt_bytes = turn_instructions.len(),
        project_guidance = project_system_prompt.is_some_and(|prompt| !prompt.trim().is_empty()),
        "starting agent turn"
    );
    let session = harness
        .new_session(session_init_with_appended_context(
            worktree,
            system_prompt,
            &turn_instructions,
            project_system_prompt,
            max_turns,
        )?)
        .await?;
    let mut events = session
        .submit_turn(Turn::user(FACTORY_TURN_USER_MESSAGE))
        .await?;
    let mut latest_text = None;
    let mut delta_text = String::new();
    let mut usage = Usage::default();

    while let Some(event) = events.next().await {
        let event =
            event.with_context(|| format!("failed to read event for agent stage {label}"))?;
        match event.payload {
            SessionEventPayload::SessionStarted => {
                info!(stage = label, "agent session started");
            }
            SessionEventPayload::Warning { message } => {
                warn!(stage = label, warning = %message, "agent warning");
            }
            SessionEventPayload::TurnStarted { turn_id } => {
                info!(stage = label, turn_id = %turn_id, "agent turn started");
            }
            SessionEventPayload::DeltaItem { delta } => {
                debug!(stage = label, bytes = delta.text.len(), "assistant delta");
                delta_text.push_str(&delta.text);
            }
            SessionEventPayload::MessageItem {
                message: Message::Assistant(message),
            } => {
                latest_text = Some(
                    message
                        .parts
                        .iter()
                        .filter_map(|part| match part {
                            AssistantPart::Text { text } => Some(text.as_str()),
                            _ => None,
                        })
                        .collect::<String>(),
                );
            }
            SessionEventPayload::ToolExecutionStarted { call } => {
                info!(
                    stage = label,
                    tool = %call.name,
                    call_id = %call.id,
                    arguments = %json_preview(&call.arguments, 500),
                    "tool started"
                );
            }
            SessionEventPayload::ToolOutput {
                call_id,
                tool_name,
                chunk,
            } => {
                debug!(
                    stage = label,
                    tool = %tool_name,
                    call_id = %call_id,
                    bytes = chunk.len(),
                    preview = %single_line_preview(&chunk, 300),
                    "tool output"
                );
            }
            SessionEventPayload::HookStarted { run } => {
                info!(
                    stage = label,
                    hook = %run.event_name,
                    plugin = %run.plugin_id,
                    handler = ?run.handler_type,
                    "hook started"
                );
            }
            SessionEventPayload::HookCompleted { run } => {
                info!(
                    stage = label,
                    hook = %run.event_name,
                    plugin = %run.plugin_id,
                    status = ?run.status,
                    duration_ms = ?run.duration_ms,
                    entries = run.entries.len(),
                    message = ?run.status_message,
                    "hook completed"
                );
            }
            SessionEventPayload::ToolExecutionCompleted { outcome } => {
                let tool = outcome.call.name;
                let call_id = outcome.call.id;
                match outcome.result {
                    Ok(result) => {
                        info!(
                            stage = label,
                            tool = %tool,
                            call_id = %call_id,
                            result_kind = tool_result_kind(&result),
                            result_bytes = tool_result_size(&result),
                            "tool completed"
                        );
                    }
                    Err(error) => {
                        warn!(
                            stage = label,
                            tool = %tool,
                            call_id = %call_id,
                            error = %error,
                            "tool failed"
                        );
                    }
                }
            }
            SessionEventPayload::ApprovalRequested { tool_name, reason } => {
                warn!(
                    stage = label,
                    tool = %tool_name,
                    reason = %reason,
                    "tool approval requested"
                );
            }
            SessionEventPayload::ContextCompacted { summary } => {
                info!(
                    stage = label,
                    summary_bytes = summary.len(),
                    "context compacted"
                );
            }
            SessionEventPayload::TurnCompleted {
                turn_id,
                usage: turn_usage,
            } => {
                usage = turn_usage;
                info!(
                    stage = label,
                    turn_id = %turn_id,
                    input_tokens = usage.input_tokens,
                    output_tokens = usage.output_tokens,
                    cache_creation_input_tokens = usage.cache_creation_input_tokens,
                    cache_read_input_tokens = usage.cache_read_input_tokens,
                    "agent turn completed"
                );
                break;
            }
            SessionEventPayload::TurnFailed {
                turn_id,
                error,
                cancelled,
                retryable,
                ..
            } => {
                warn!(
                    stage = label,
                    turn_id = %turn_id,
                    cancelled,
                    retryable,
                    error = %error,
                    "agent turn failed"
                );
                let _ = session.shutdown("turn_failed").await;
                return Err(anyhow::Error::new(AgentStageTurnFailure {
                    label: label.to_owned(),
                    error,
                    retryable,
                    cancelled,
                }));
            }
            SessionEventPayload::Lagged { dropped_events } => {
                warn!(stage = label, dropped_events, "agent event stream lagged");
            }
            SessionEventPayload::SessionShutdownComplete => {
                info!(stage = label, "agent session shutdown complete");
            }
            _ => {}
        }
    }

    info!(stage = label, "shutting down agent session");
    session.shutdown(label).await?;
    let run = agent_run_from_text(label, latest_text, delta_text, text_requirement)?;
    if text_requirement == AgentTextRequirement::Optional && run.text.is_empty() {
        info!(
            stage = label,
            "agent turn produced no assistant text; continuing because text is optional"
        );
    }
    info!(
        stage = label,
        input_tokens = usage.input_tokens,
        output_tokens = usage.output_tokens,
        "completed agent turn"
    );
    Ok(run)
}

pub(crate) fn agent_run_from_text(
    label: &str,
    latest_text: Option<String>,
    delta_text: String,
    text_requirement: AgentTextRequirement,
) -> anyhow::Result<AgentRun> {
    if let Some(text) = latest_text
        .filter(|text| !text.trim().is_empty())
        .or_else(|| (!delta_text.trim().is_empty()).then_some(delta_text))
    {
        return Ok(AgentRun { text });
    }

    match text_requirement {
        AgentTextRequirement::Required => {
            bail!("agent stage {label} produced no assistant text");
        }
        AgentTextRequirement::Optional => Ok(AgentRun {
            text: String::new(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::Duration;

    use halter_protocol::ToolResult;
    use serde_json::json;

    #[test]
    pub(crate) fn single_line_preview_covers_normalized_truncated_and_empty_text() {
        pub(crate) struct Case {
            pub(crate) name: &'static str,
            pub(crate) text: &'static str,
            pub(crate) max_chars: usize,
            pub(crate) expected: &'static str,
        }

        let cases = [
            Case {
                name: "empty",
                text: "",
                max_chars: 10,
                expected: "",
            },
            Case {
                name: "short unchanged",
                text: "abc",
                max_chars: 10,
                expected: "abc",
            },
            Case {
                name: "newlines are escaped",
                text: "a\nb\rc",
                max_chars: 20,
                expected: "a\\nb\\rc",
            },
            Case {
                name: "truncated at character boundary",
                text: "abéde",
                max_chars: 3,
                expected: "abé...",
            },
            Case {
                name: "zero limit truncates non-empty text",
                text: "abc",
                max_chars: 0,
                expected: "...",
            },
        ];

        for case in cases {
            assert_eq!(
                single_line_preview(case.text, case.max_chars),
                case.expected,
                "{}",
                case.name
            );
        }
    }

    #[test]
    pub(crate) fn json_preview_serializes_before_previewing() {
        let value = json!({
            "message": "hello\nworld",
            "ok": true,
        });

        let preview = json_preview(&value, 16);

        assert!(preview.starts_with('{'));
        assert!(preview.ends_with("..."));
        assert!(!preview.contains('\n'));
    }

    #[test]
    pub(crate) fn tool_result_logging_helpers_cover_each_result_kind() {
        let json_value = json!({"a": 1});
        let cases = [
            (ToolResult::Empty, "empty", 0),
            (
                ToolResult::Text {
                    text: "hello".to_owned(),
                },
                "text",
                5,
            ),
            (
                ToolResult::Json {
                    value: json_value.clone(),
                },
                "json",
                json_value.to_string().len(),
            ),
        ];

        for (result, expected_kind, expected_size) in cases {
            assert_eq!(tool_result_kind(&result), expected_kind);
            assert_eq!(tool_result_size(&result), expected_size);
        }
    }

    #[test]
    pub(crate) fn agent_run_from_text_covers_required_and_optional_outputs() {
        pub(crate) struct Case {
            pub(crate) name: &'static str,
            pub(crate) latest_text: Option<&'static str>,
            pub(crate) delta_text: &'static str,
            pub(crate) requirement: AgentTextRequirement,
            pub(crate) expected_text: Option<&'static str>,
            pub(crate) expected_error: Option<&'static str>,
        }

        let cases = [
            Case {
                name: "required latest assistant text",
                latest_text: Some("final message"),
                delta_text: "partial delta",
                requirement: AgentTextRequirement::Required,
                expected_text: Some("final message"),
                expected_error: None,
            },
            Case {
                name: "required delta fallback",
                latest_text: None,
                delta_text: "streamed message",
                requirement: AgentTextRequirement::Required,
                expected_text: Some("streamed message"),
                expected_error: None,
            },
            Case {
                name: "required ignores blank latest and uses delta",
                latest_text: Some(" \n"),
                delta_text: "streamed message",
                requirement: AgentTextRequirement::Required,
                expected_text: Some("streamed message"),
                expected_error: None,
            },
            Case {
                name: "required empty output fails",
                latest_text: None,
                delta_text: " \n",
                requirement: AgentTextRequirement::Required,
                expected_text: None,
                expected_error: Some("produced no assistant text"),
            },
            Case {
                name: "optional empty output succeeds",
                latest_text: None,
                delta_text: " \n",
                requirement: AgentTextRequirement::Optional,
                expected_text: Some(""),
                expected_error: None,
            },
        ];

        for case in cases {
            let result = agent_run_from_text(
                "test stage",
                case.latest_text.map(ToOwned::to_owned),
                case.delta_text.to_owned(),
                case.requirement,
            );

            match (case.expected_text, case.expected_error, result) {
                (Some(expected), None, Ok(run)) => {
                    assert_eq!(run.text, expected, "{}", case.name);
                }
                (None, Some(expected), Err(error)) => {
                    assert!(
                        error.to_string().contains(expected),
                        "{}: {error}",
                        case.name
                    );
                }
                (_, _, other) => panic!("{}: unexpected result {other:?}", case.name),
            }
        }
    }

    #[test]
    pub(crate) fn agent_stage_failure_retry_detection_covers_flags_and_transient_text() {
        pub(crate) struct Case {
            pub(crate) name: &'static str,
            pub(crate) retryable: bool,
            pub(crate) cancelled: bool,
            pub(crate) error: &'static str,
            pub(crate) expected: bool,
        }

        let cases = [
            Case {
                name: "provider retryable flag",
                retryable: true,
                cancelled: false,
                error: "provider asked for retry",
                expected: true,
            },
            Case {
                name: "cancelled provider retryable flag",
                retryable: true,
                cancelled: true,
                error: "provider asked for retry",
                expected: false,
            },
            Case {
                name: "overloaded text fallback",
                retryable: false,
                cancelled: false,
                error: r#"Provider returned error (Parasail: {"error":{"message":"The engine is currently overloaded. Please try again later."}})"#,
                expected: true,
            },
            Case {
                name: "rate limit text fallback",
                retryable: false,
                cancelled: false,
                error: "provider returned 429 Too Many Requests",
                expected: true,
            },
            Case {
                name: "fatal validation error",
                retryable: false,
                cancelled: false,
                error: "invalid request: missing required field",
                expected: false,
            },
        ];

        for case in cases {
            assert_eq!(
                agent_stage_failure_is_retryable(case.retryable, case.cancelled, case.error),
                case.expected,
                "{}",
                case.name
            );
        }
    }

    #[test]
    pub(crate) fn agent_stage_retry_policy_covers_hint_exponential_cap_and_exhaustion() {
        pub(crate) struct Case {
            pub(crate) name: &'static str,
            pub(crate) policy: AgentStageRetryPolicy,
            pub(crate) failed_attempt: u32,
            pub(crate) error: &'static str,
            pub(crate) expected: Option<Duration>,
        }

        let policy = AgentStageRetryPolicy {
            max_attempts: 3,
            base_backoff: Duration::from_millis(100),
            max_backoff: Duration::from_millis(250),
        };
        let cases = [
            Case {
                name: "first retry uses base backoff",
                policy,
                failed_attempt: 1,
                error: "retryable provider failure",
                expected: Some(Duration::from_millis(100)),
            },
            Case {
                name: "second retry doubles backoff",
                policy,
                failed_attempt: 2,
                error: "retryable provider failure",
                expected: Some(Duration::from_millis(200)),
            },
            Case {
                name: "hint is capped to max backoff",
                policy,
                failed_attempt: 1,
                error: "upstream model is overloaded",
                expected: Some(Duration::from_millis(250)),
            },
            Case {
                name: "budget exhaustion returns none",
                policy,
                failed_attempt: 3,
                error: "retryable provider failure",
                expected: None,
            },
        ];

        for case in cases {
            assert_eq!(
                case.policy
                    .delay_after_failure(case.failed_attempt, case.error),
                case.expected,
                "{}",
                case.name
            );
        }
    }

    #[test]
    pub(crate) fn agent_stage_turn_failure_display_and_retry_metadata_match() {
        let retryable = AgentStageTurnFailure {
            label: "kimi implementation".to_owned(),
            error: "provider overloaded".to_owned(),
            retryable: false,
            cancelled: false,
        };
        assert_eq!(
            retryable.to_string(),
            "agent stage kimi implementation failed: provider overloaded"
        );
        assert!(retryable.should_retry());

        let cancelled = AgentStageTurnFailure {
            cancelled: true,
            ..retryable
        };
        assert!(!cancelled.should_retry());
    }
}
