// pattern: Functional Core

use clap::Args;
use halter_protocol::{
    AssistantMessage, AssistantPart, InputDeferredReason, InputOutcome, Message, MessageId,
    SessionEvent, SessionEventPayload,
};

#[cfg(test)]
use halter_protocol::{ReplayMeta, SessionStatus, StopReason, Usage};

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
pub struct JsonResultTracker {
    final_result: Option<AssistantMessage>,
    message_id: MessageId,
}

impl JsonResultTracker {
    pub fn new(message_id: MessageId) -> Self {
        Self {
            final_result: None,
            message_id,
        }
    }

    pub fn observe(
        &mut self,
        payload: &SessionEventPayload,
    ) -> Result<Option<&AssistantMessage>, String> {
        if input_completion(payload, &self.message_id)? {
            return self
                .final_result
                .as_ref()
                .map(Some)
                .ok_or_else(|| "failed to capture final assistant result".to_owned());
        }
        match payload {
            SessionEventPayload::MessageItem {
                message: Message::Assistant(message),
            } => {
                self.final_result = Some(message.clone());
                Ok(None)
            }
            _ => Ok(None),
        }
    }
}

/// Match a submitted input rather than unrelated execution or activity events.
pub fn input_completion(
    payload: &SessionEventPayload,
    message_id: &MessageId,
) -> Result<bool, String> {
    match payload {
        SessionEventPayload::InputSettled {
            message_id: id,
            outcome,
        } if id == message_id => match outcome {
            InputOutcome::Completed => Ok(true),
            InputOutcome::Failed { error, .. } => Err(error.clone()),
            InputOutcome::Interrupted => Err("input was interrupted".to_owned()),
        },
        SessionEventPayload::InputRejected {
            message_id: id,
            reason,
        } if id == message_id => Err(format!("input was rejected: {reason}")),
        SessionEventPayload::InputDeferred {
            message_id: id,
            reason,
        } if id == message_id => {
            let reason = match reason {
                InputDeferredReason::Interrupted => "execution was interrupted".to_owned(),
                InputDeferredReason::ExecutionFailed { error, .. } => {
                    format!("execution failed: {error}")
                }
                InputDeferredReason::Shutdown => "the session was shut down".to_owned(),
                InputDeferredReason::Resumed => "the session was resumed idle".to_owned(),
            };
            Err(format!("input remains queued because {reason}"))
        }
        _ => Ok(false),
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
    fn json_result_tracker_returns_latest_assistant_message_on_completion() {
        let id = MessageId::from("input-1");
        let mut tracker = JsonResultTracker::new(id.clone());
        let tool_request = assistant_message("call tool", Some(StopReason::ToolUse));
        let final_result = assistant_message("done", Some(StopReason::EndTurn));

        assert!(
            tracker
                .observe(&SessionEventPayload::MessageItem {
                    message: Message::Assistant(tool_request),
                })
                .expect("observe tool request")
                .is_none()
        );
        assert!(
            tracker
                .observe(&SessionEventPayload::MessageItem {
                    message: Message::Tool(halter_protocol::ToolResultMessage {
                        id: MessageId::from("tool-message"),
                        call_id: halter_protocol::ToolCallId::from("call-1"),
                        content: halter_protocol::ToolResult::Text {
                            text: "ok".to_owned(),
                        },
                        error: None,
                        created_at: Utc::now(),
                    }),
                })
                .expect("observe tool result")
                .is_none()
        );
        assert!(
            tracker
                .observe(&SessionEventPayload::MessageItem {
                    message: Message::Assistant(final_result.clone()),
                })
                .expect("observe final result")
                .is_none()
        );

        let result = tracker
            .observe(&SessionEventPayload::InputSettled {
                message_id: id,
                outcome: InputOutcome::Completed,
            })
            .expect("turn completed")
            .expect("assistant result");

        assert_eq!(result, &final_result);
    }

    #[test]
    fn json_result_tracker_errors_on_its_input_failure() {
        let id = MessageId::from("input-1");
        let mut tracker = JsonResultTracker::new(id.clone());
        let error = tracker
            .observe(&SessionEventPayload::InputSettled {
                message_id: id,
                outcome: InputOutcome::Failed {
                    error: "provider exploded".to_owned(),
                    retryable: false,
                },
            })
            .expect_err("turn failure should surface");
        assert_eq!(error, "provider exploded");
    }

    #[test]
    fn json_result_tracker_requires_a_final_assistant_message() {
        let id = MessageId::from("input-1");
        let mut tracker = JsonResultTracker::new(id.clone());
        let error = tracker
            .observe(&SessionEventPayload::InputSettled {
                message_id: id,
                outcome: InputOutcome::Completed,
            })
            .expect_err("turn completion without assistant result should fail");
        assert_eq!(error, "failed to capture final assistant result");
    }

    #[test]
    fn json_result_tracker_ignores_activity_and_other_input_outcomes() {
        let mut tracker = JsonResultTracker::new(MessageId::from("input-1"));
        for payload in [
            SessionEventPayload::SessionStatusChanged {
                status: SessionStatus::Running,
            },
            SessionEventPayload::SessionStatusChanged {
                status: SessionStatus::Idle,
            },
            SessionEventPayload::TurnFailed {
                turn_id: halter_protocol::TurnId::from("earlier-turn"),
                error: "unrelated failure".to_owned(),
                cancelled: false,
                retryable: false,
            },
            SessionEventPayload::InputSettled {
                message_id: MessageId::from("another-input"),
                outcome: InputOutcome::Completed,
            },
            SessionEventPayload::InputDeferred {
                message_id: MessageId::from("another-input"),
                reason: InputDeferredReason::Interrupted,
            },
        ] {
            assert!(tracker.observe(&payload).unwrap().is_none(), "{payload:?}");
        }
    }

    #[test]
    fn input_completion_distinguishes_settlement_rejection_and_pending_input() {
        let id = MessageId::from("input-1");
        let cases = [
            (
                SessionEventPayload::InputSettled {
                    message_id: id.clone(),
                    outcome: InputOutcome::Completed,
                },
                Ok(true),
            ),
            (
                SessionEventPayload::InputSettled {
                    message_id: id.clone(),
                    outcome: InputOutcome::Failed {
                        error: "provider unavailable".to_owned(),
                        retryable: true,
                    },
                },
                Err("provider unavailable"),
            ),
            (
                SessionEventPayload::InputSettled {
                    message_id: id.clone(),
                    outcome: InputOutcome::Interrupted,
                },
                Err("input was interrupted"),
            ),
            (
                SessionEventPayload::InputRejected {
                    message_id: id.clone(),
                    reason: "blocked by hook".to_owned(),
                },
                Err("input was rejected: blocked by hook"),
            ),
            (
                SessionEventPayload::InputDeferred {
                    message_id: id.clone(),
                    reason: InputDeferredReason::Interrupted,
                },
                Err("input remains queued because execution was interrupted"),
            ),
            (
                SessionEventPayload::InputDeferred {
                    message_id: id.clone(),
                    reason: InputDeferredReason::ExecutionFailed {
                        error: "hook failed".to_owned(),
                        retryable: false,
                    },
                },
                Err("input remains queued because execution failed: hook failed"),
            ),
            (
                SessionEventPayload::InputDeferred {
                    message_id: id.clone(),
                    reason: InputDeferredReason::Shutdown,
                },
                Err("input remains queued because the session was shut down"),
            ),
            (
                SessionEventPayload::InputDeferred {
                    message_id: id.clone(),
                    reason: InputDeferredReason::Resumed,
                },
                Err("input remains queued because the session was resumed idle"),
            ),
        ];
        for (payload, expected) in cases {
            assert_eq!(
                input_completion(&payload, &id),
                expected.map_err(str::to_owned),
                "{payload:?}"
            );
        }
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
