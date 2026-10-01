// pattern: Functional Core

use clap::Args;
use halter_protocol::{
    AssistantMessage, AssistantPart, InputDeferredReason, Message, MessageId, SessionEvent,
    SessionEventPayload,
};

#[cfg(test)]
use halter_protocol::{ReplayMeta, StopReason, TurnId, Usage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RunOutputMode {
    JsonResult,
    StreamingJson,
}

#[derive(Debug, Clone, Args, Default)]
pub struct RunOutputArgs {
    #[arg(
        long,
        conflicts_with = "json_result",
        help = "Stream each session event as newline-delimited JSON"
    )]
    pub streaming_json: bool,
    #[arg(
        long,
        conflicts_with = "streaming_json",
        help = "Print the final assistant result as JSON (default)"
    )]
    pub json_result: bool,
}

impl RunOutputArgs {
    #[must_use]
    pub fn mode(&self) -> RunOutputMode {
        if self.streaming_json {
            RunOutputMode::StreamingJson
        } else {
            RunOutputMode::JsonResult
        }
    }
}

#[derive(Debug)]
/// The CLI drives one foreground execution at a time and renders its events
/// before returning to the prompt. Background lifetime belongs to the session.
pub struct ForegroundRun {
    final_result: Option<AssistantMessage>,
    message_id: MessageId,
    delivered: bool,
}

impl ForegroundRun {
    pub fn new(message_id: MessageId) -> Self {
        Self {
            final_result: None,
            message_id,
            delivered: false,
        }
    }

    pub fn observe(&mut self, payload: &SessionEventPayload) -> Result<bool, String> {
        match payload {
            SessionEventPayload::InputDelivered { message_id }
                if message_id == &self.message_id =>
            {
                self.delivered = true;
            }
            // Parent-session execution is serialized. Once this input enters
            // history, the next terminal event belongs to its execution.
            SessionEventPayload::TurnCompleted { .. } if self.delivered => {
                return Ok(true);
            }
            SessionEventPayload::TurnFailed { error, .. } if self.delivered => {
                return Err(error.clone());
            }
            SessionEventPayload::MessageItem {
                message: Message::Assistant(message),
            } if self.delivered => {
                self.final_result = Some(message.clone());
            }
            SessionEventPayload::InputRejected { message_id, reason }
                if message_id == &self.message_id =>
            {
                return Err(format!("input was rejected: {reason}"));
            }
            SessionEventPayload::InputDeferred { message_id, reason }
                if message_id == &self.message_id =>
            {
                let reason = match reason {
                    InputDeferredReason::Interrupted => "execution was interrupted".to_owned(),
                    InputDeferredReason::ExecutionFailed { error, .. } => {
                        format!("execution failed: {error}")
                    }
                    InputDeferredReason::ExecutionStopped => "earlier execution failed".to_owned(),
                    InputDeferredReason::Shutdown => "the session was shut down".to_owned(),
                    InputDeferredReason::Resumed => "the session was resumed idle".to_owned(),
                };
                return Err(format!("input remains queued because {reason}"));
            }
            _ => {}
        }
        Ok(false)
    }

    pub fn final_result(&self) -> Result<&AssistantMessage, String> {
        self.final_result
            .as_ref()
            .ok_or_else(|| "failed to capture final assistant result".to_owned())
    }
}

#[must_use]
pub fn strip_signatures_from_session_event(event: &SessionEvent) -> SessionEvent {
    let mut event = event.clone();
    if let SessionEventPayload::MessageItem { message }
    | SessionEventPayload::MessageRecorded { message } = &mut event.payload
    {
        strip_signatures_from_message(message);
    }
    event
}

#[must_use]
pub fn strip_signatures_from_assistant_message(message: &AssistantMessage) -> AssistantMessage {
    let mut message = message.clone();
    strip_signatures_from_assistant_parts(&mut message.parts);
    message
}

fn strip_signatures_from_message(message: &mut Message) {
    if let Message::Assistant(message) = message {
        strip_signatures_from_assistant_parts(&mut message.parts);
    }
}

fn strip_signatures_from_assistant_parts(parts: &mut [AssistantPart]) {
    for part in parts {
        if let AssistantPart::Thinking(block) = part {
            block.signature = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use clap::Parser;
    use halter_protocol::{SessionEvent, SessionId, ThinkingBlock};

    use super::*;

    #[derive(Debug, Parser)]
    struct TestCli {
        #[command(flatten)]
        output: RunOutputArgs,
        task: String,
    }

    #[test]
    fn run_output_mode_defaults_to_json_result() {
        let cli = TestCli::try_parse_from(["halter", "task"]).expect("parse");
        assert_eq!(cli.output.mode(), RunOutputMode::JsonResult);
        assert!(!cli.output.json_result);
        assert!(!cli.output.streaming_json);
    }

    #[test]
    fn run_output_mode_accepts_explicit_json_result() {
        let cli = TestCli::try_parse_from(["halter", "--json-result", "task"]).expect("parse");
        assert_eq!(cli.output.mode(), RunOutputMode::JsonResult);
        assert!(cli.output.json_result);
    }

    #[test]
    fn run_output_mode_accepts_streaming_json() {
        let cli = TestCli::try_parse_from(["halter", "--streaming-json", "task"]).expect("parse");
        assert_eq!(cli.output.mode(), RunOutputMode::StreamingJson);
        assert!(cli.output.streaming_json);
    }

    #[test]
    fn run_output_mode_rejects_conflicting_flags() {
        let error =
            TestCli::try_parse_from(["halter", "--json-result", "--streaming-json", "task"])
                .expect_err("conflicting flags should fail");
        assert_eq!(error.kind(), clap::error::ErrorKind::ArgumentConflict);
    }

    #[test]
    fn foreground_run_keeps_latest_assistant_until_its_execution_stops() {
        let input = MessageId::from("input-1");
        let turn = TurnId::from("foreground-turn");
        let mut foreground = ForegroundRun::new(input.clone());
        let final_result = assistant_message("done", Some(StopReason::EndTurn));
        for payload in [
            SessionEventPayload::TurnStarted {
                turn_id: turn.clone(),
                default_model: None,
                subagent_model: None,
            },
            SessionEventPayload::InputDelivered { message_id: input },
            SessionEventPayload::MessageItem {
                message: Message::Assistant(assistant_message(
                    "call tool",
                    Some(StopReason::ToolUse),
                )),
            },
            SessionEventPayload::MessageItem {
                message: Message::Assistant(final_result.clone()),
            },
        ] {
            assert!(!foreground.observe(&payload).unwrap(), "{payload:?}");
        }
        assert!(
            foreground
                .observe(&SessionEventPayload::TurnCompleted {
                    turn_id: turn,
                    usage: Usage::default(),
                })
                .unwrap()
        );
        assert_eq!(foreground.final_result().unwrap(), &final_result);
    }

    #[test]
    fn foreground_run_waits_past_rejection_of_earlier_queued_input() {
        let target = MessageId::from("fresh-input");
        let mut foreground = ForegroundRun::new(target.clone());
        let prior_turn = TurnId::from("prior-turn");
        for payload in [
            SessionEventPayload::TurnStarted {
                turn_id: prior_turn.clone(),
                default_model: None,
                subagent_model: None,
            },
            SessionEventPayload::MessageItem {
                message: Message::Assistant(assistant_message(
                    "earlier hook output",
                    Some(StopReason::EndTurn),
                )),
            },
            SessionEventPayload::InputRejected {
                message_id: MessageId::from("earlier-deferred-input"),
                reason: "blocked by hook".to_owned(),
            },
            SessionEventPayload::TurnCompleted {
                turn_id: prior_turn,
                usage: Usage::default(),
            },
        ] {
            assert!(!foreground.observe(&payload).unwrap(), "{payload:?}");
        }
        assert!(foreground.final_result().is_err());

        let target_turn = TurnId::from("target-turn");
        let expected = assistant_message("target result", Some(StopReason::EndTurn));
        for payload in [
            SessionEventPayload::TurnStarted {
                turn_id: target_turn.clone(),
                default_model: None,
                subagent_model: None,
            },
            SessionEventPayload::InputDelivered { message_id: target },
            SessionEventPayload::MessageItem {
                message: Message::Assistant(expected.clone()),
            },
        ] {
            assert!(!foreground.observe(&payload).unwrap(), "{payload:?}");
        }
        assert!(
            foreground
                .observe(&SessionEventPayload::TurnCompleted {
                    turn_id: target_turn,
                    usage: Usage::default(),
                })
                .unwrap()
        );
        assert_eq!(foreground.final_result().unwrap(), &expected);
    }

    #[test]
    fn foreground_run_observes_steering_delivery_without_an_execution_start() {
        let target = MessageId::from("steering-input");
        let expected = assistant_message("steered result", Some(StopReason::EndTurn));
        let mut foreground = ForegroundRun::new(target.clone());
        // The caller skips TurnStarted when it precedes the steering input's
        // acceptance sequence in the retained session stream.
        for payload in [
            SessionEventPayload::MessageItem {
                message: Message::Assistant(assistant_message(
                    "before steering",
                    Some(StopReason::EndTurn),
                )),
            },
            SessionEventPayload::InputDelivered { message_id: target },
            SessionEventPayload::MessageItem {
                message: Message::Assistant(expected.clone()),
            },
        ] {
            assert!(!foreground.observe(&payload).unwrap(), "{payload:?}");
        }
        assert!(
            foreground
                .observe(&SessionEventPayload::TurnCompleted {
                    turn_id: TurnId::from("already-running-turn"),
                    usage: Usage::default(),
                })
                .unwrap()
        );
        assert_eq!(foreground.final_result().unwrap(), &expected);
    }

    #[test]
    fn foreground_run_reports_its_deferral_after_an_earlier_execution_fails() {
        let target = MessageId::from("unattempted-input");
        let prior_turn = TurnId::from("prior-turn");
        let mut foreground = ForegroundRun::new(target.clone());
        for payload in [
            SessionEventPayload::TurnStarted {
                turn_id: prior_turn.clone(),
                default_model: None,
                subagent_model: None,
            },
            SessionEventPayload::TurnFailed {
                turn_id: prior_turn,
                error: "earlier input hook failed".to_owned(),
                cancelled: false,
                retryable: true,
            },
        ] {
            assert!(!foreground.observe(&payload).unwrap(), "{payload:?}");
        }
        assert_eq!(
            foreground.observe(&SessionEventPayload::InputDeferred {
                message_id: target,
                reason: InputDeferredReason::ExecutionStopped,
            }),
            Err("input remains queued because earlier execution failed".to_owned())
        );
    }

    #[test]
    fn foreground_run_ignores_unrelated_execution_and_input_events() {
        let mut foreground = ForegroundRun::new(MessageId::from("input-1"));
        let turn = TurnId::from("foreground-turn");
        foreground
            .observe(&SessionEventPayload::TurnStarted {
                turn_id: turn,
                default_model: None,
                subagent_model: None,
            })
            .unwrap();
        for payload in [
            SessionEventPayload::TurnFailed {
                turn_id: TurnId::from("earlier-turn"),
                error: "unrelated failure".to_owned(),
                cancelled: false,
                retryable: false,
            },
            SessionEventPayload::TurnCompleted {
                turn_id: TurnId::from("earlier-turn"),
                usage: Usage::default(),
            },
            SessionEventPayload::InputDelivered {
                message_id: MessageId::from("another-input"),
            },
            SessionEventPayload::InputRejected {
                message_id: MessageId::from("another-input"),
                reason: "blocked".to_owned(),
            },
            SessionEventPayload::InputDeferred {
                message_id: MessageId::from("another-input"),
                reason: InputDeferredReason::Interrupted,
            },
        ] {
            assert!(!foreground.observe(&payload).unwrap(), "{payload:?}");
        }
    }

    #[test]
    fn foreground_run_reports_execution_and_input_failures() {
        let id = MessageId::from("input-1");
        let turn = TurnId::from("foreground-turn");
        let cases = [
            (
                SessionEventPayload::TurnFailed {
                    turn_id: turn.clone(),
                    error: "provider unavailable".to_owned(),
                    cancelled: false,
                    retryable: true,
                },
                "provider unavailable",
            ),
            (
                SessionEventPayload::InputRejected {
                    message_id: id.clone(),
                    reason: "blocked by hook".to_owned(),
                },
                "input was rejected: blocked by hook",
            ),
            (
                SessionEventPayload::InputDeferred {
                    message_id: id.clone(),
                    reason: InputDeferredReason::Interrupted,
                },
                "input remains queued because execution was interrupted",
            ),
            (
                SessionEventPayload::InputDeferred {
                    message_id: id.clone(),
                    reason: InputDeferredReason::ExecutionFailed {
                        error: "hook failed".to_owned(),
                        retryable: false,
                    },
                },
                "input remains queued because execution failed: hook failed",
            ),
            (
                SessionEventPayload::InputDeferred {
                    message_id: id.clone(),
                    reason: InputDeferredReason::ExecutionStopped,
                },
                "input remains queued because earlier execution failed",
            ),
            (
                SessionEventPayload::InputDeferred {
                    message_id: id.clone(),
                    reason: InputDeferredReason::Shutdown,
                },
                "input remains queued because the session was shut down",
            ),
            (
                SessionEventPayload::InputDeferred {
                    message_id: id.clone(),
                    reason: InputDeferredReason::Resumed,
                },
                "input remains queued because the session was resumed idle",
            ),
        ];
        for (payload, expected) in cases {
            let mut foreground = ForegroundRun::new(id.clone());
            if matches!(payload, SessionEventPayload::TurnFailed { .. }) {
                foreground
                    .observe(&SessionEventPayload::InputDelivered {
                        message_id: id.clone(),
                    })
                    .unwrap();
            }
            assert_eq!(
                foreground.observe(&payload),
                Err(expected.to_owned()),
                "{payload:?}"
            );
        }
    }

    #[test]
    fn foreground_json_output_requires_an_assistant_message() {
        let foreground = ForegroundRun::new(MessageId::from("input-1"));
        assert_eq!(
            foreground.final_result().unwrap_err(),
            "failed to capture final assistant result"
        );
    }

    #[test]
    fn strip_signatures_from_assistant_message_clears_thinking_signatures() {
        let message = AssistantMessage {
            id: MessageId::from("assistant-thinking"),
            created_at: Utc::now(),
            parts: vec![
                AssistantPart::Thinking(ThinkingBlock {
                    text: "reasoning".to_owned(),
                    signature: Some("sig-123".to_owned()),
                }),
                AssistantPart::Text {
                    text: "done".to_owned(),
                },
            ],
            stop_reason: Some(StopReason::EndTurn),
            usage: Some(Usage::default()),
            replay_meta: ReplayMeta::default(),
        };

        let stripped = strip_signatures_from_assistant_message(&message);

        assert_eq!(
            stripped.parts,
            vec![
                AssistantPart::Thinking(ThinkingBlock {
                    text: "reasoning".to_owned(),
                    signature: None,
                }),
                AssistantPart::Text {
                    text: "done".to_owned(),
                },
            ]
        );
        assert_eq!(
            message.parts,
            vec![
                AssistantPart::Thinking(ThinkingBlock {
                    text: "reasoning".to_owned(),
                    signature: Some("sig-123".to_owned()),
                }),
                AssistantPart::Text {
                    text: "done".to_owned(),
                },
            ]
        );
    }

    #[test]
    fn strip_signatures_from_session_event_clears_assistant_message_signatures() {
        let event = SessionEvent::new_committed(
            SessionId::from("session-1"),
            7,
            halter_protocol::Delivery::Lossless,
            SessionEventPayload::MessageItem {
                message: Message::Assistant(AssistantMessage {
                    id: MessageId::from("assistant-thinking"),
                    created_at: Utc::now(),
                    parts: vec![AssistantPart::Thinking(ThinkingBlock {
                        text: "reasoning".to_owned(),
                        signature: Some("sig-456".to_owned()),
                    })],
                    stop_reason: Some(StopReason::EndTurn),
                    usage: Some(Usage::default()),
                    replay_meta: ReplayMeta::default(),
                }),
            },
        );

        let stripped = strip_signatures_from_session_event(&event);

        assert_eq!(
            stripped,
            SessionEvent::new_committed(
                SessionId::from("session-1"),
                7,
                halter_protocol::Delivery::Lossless,
                SessionEventPayload::MessageItem {
                    message: Message::Assistant(AssistantMessage {
                        id: MessageId::from("assistant-thinking"),
                        created_at: event.clone().payload_message_created_at(),
                        parts: vec![AssistantPart::Thinking(ThinkingBlock {
                            text: "reasoning".to_owned(),
                            signature: None,
                        })],
                        stop_reason: Some(StopReason::EndTurn),
                        usage: Some(Usage::default()),
                        replay_meta: ReplayMeta::default(),
                    }),
                },
            )
        );
    }

    fn assistant_message(text: &str, stop_reason: Option<StopReason>) -> AssistantMessage {
        AssistantMessage {
            id: MessageId::from(format!("assistant-{text}")),
            created_at: Utc::now(),
            parts: vec![AssistantPart::Text {
                text: text.to_owned(),
            }],
            stop_reason,
            usage: Some(Usage::default()),
            replay_meta: ReplayMeta::default(),
        }
    }

    trait SessionEventTestExt {
        fn payload_message_created_at(&self) -> chrono::DateTime<Utc>;
    }

    impl SessionEventTestExt for SessionEvent {
        fn payload_message_created_at(&self) -> chrono::DateTime<Utc> {
            match &self.payload {
                SessionEventPayload::MessageItem {
                    message: Message::Assistant(message),
                } => message.created_at,
                _ => panic!("expected assistant message payload"),
            }
        }
    }
}
