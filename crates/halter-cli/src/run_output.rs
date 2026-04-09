// pattern: Functional Core

use clap::Args;
use halter_protocol::{
    AssistantMessage, AssistantPart, Message, SessionEvent, SessionEventPayload,
};

#[cfg(test)]
use halter_protocol::{MessageId, ReplayMeta, StopReason, Usage};

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

#[derive(Debug, Default)]
pub struct JsonResultTracker {
    final_result: Option<AssistantMessage>,
}

impl JsonResultTracker {
    pub fn observe(
        &mut self,
        payload: &SessionEventPayload,
    ) -> Result<Option<&AssistantMessage>, String> {
        match payload {
            SessionEventPayload::MessageItem {
                message: Message::Assistant(message),
            } => {
                self.final_result = Some(message.clone());
                Ok(None)
            }
            SessionEventPayload::TurnCompleted { .. } => self
                .final_result
                .as_ref()
                .map(Some)
                .ok_or_else(|| "failed to capture final assistant result".to_owned()),
            SessionEventPayload::TurnFailed { error, .. } => Err(error.clone()),
            _ => Ok(None),
        }
    }
}

#[must_use]
pub fn strip_signatures_from_session_event(event: &SessionEvent) -> SessionEvent {
    let mut event = event.clone();
    if let SessionEventPayload::MessageItem { message } = &mut event.payload {
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
        let mut tracker = JsonResultTracker::default();
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
            .observe(&SessionEventPayload::TurnCompleted {
                turn_id: halter_protocol::TurnId::from("turn-1"),
                usage: Usage::default(),
            })
            .expect("turn completed")
            .expect("assistant result");

        assert_eq!(result, &final_result);
    }

    #[test]
    fn json_result_tracker_errors_on_turn_failure() {
        let mut tracker = JsonResultTracker::default();
        let error = tracker
            .observe(&SessionEventPayload::TurnFailed {
                turn_id: halter_protocol::TurnId::from("turn-1"),
                error: "provider exploded".to_owned(),
            })
            .expect_err("turn failure should surface");
        assert_eq!(error, "provider exploded");
    }

    #[test]
    fn json_result_tracker_requires_a_final_assistant_message() {
        let mut tracker = JsonResultTracker::default();
        let error = tracker
            .observe(&SessionEventPayload::TurnCompleted {
                turn_id: halter_protocol::TurnId::from("turn-1"),
                usage: Usage::default(),
            })
            .expect_err("turn completion without assistant result should fail");
        assert_eq!(error, "failed to capture final assistant result");
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
        let event = SessionEvent {
            session_id: SessionId::from("session-1"),
            sequence: 7,
            delivery: halter_protocol::Delivery::Lossless,
            payload: SessionEventPayload::MessageItem {
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
        };

        let stripped = strip_signatures_from_session_event(&event);

        assert_eq!(
            stripped,
            SessionEvent {
                session_id: SessionId::from("session-1"),
                sequence: 7,
                delivery: halter_protocol::Delivery::Lossless,
                payload: SessionEventPayload::MessageItem {
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
            }
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
