//! Cache-prefix stability at the layer that assembles requests.
//!
//! A provider prompt cache is a byte prefix of the request: tools, the system
//! prompt, then the messages in order. Within one context window, each
//! request the runtime sends must therefore repeat the previous request's
//! cache-relevant fields byte for byte and extend its message list without
//! rewriting any earlier message. Compaction and rollover start a new window
//! and are the only allowed resets.
//!
//! This guard checks [`ProviderRequest`]s, the runtime's output. Codecs are
//! pure functions of a request, and `halter-providers`' property test
//! `appending_a_step_never_rewrites_the_encoded_prefix` shows that extending
//! the message list never rewrites the encoded prefix on any dialect. The two
//! together cover the bytes on the wire.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::stream::{self, BoxStream};
use futures::{StreamExt, TryStreamExt};
use halter_hooks::{Hook, HookEventName, HookResponse, RegisteredHookPriority, RegisteredHooks};
use halter_protocol::{
    BlockId, Message, MessageId, PluginId, ProviderCapabilities, ProviderError, ProviderRequest,
    SessionEventPayload, StopReason, StreamEvent, ToolCallId, ToolCapabilities, ToolConcurrency,
    ToolName, ToolResult, ToolSpec, Turn, Usage,
};
use halter_providers::Provider;
use halter_tools::{Tool, ToolContext};
use serde_json::json;
use tokio_util::sync::CancellationToken;

use crate::{
    CompactionContext, CompactionEffects, CompactionStrategy, SessionInit, SessionRuntime,
};

/// The fields of a request a provider caches ahead of the messages. Segment
/// ids are left out: no codec sends them.
fn cached_fields(request: &ProviderRequest) -> String {
    let segments = request
        .prompt
        .ordered_segments
        .iter()
        .map(|segment| json!([segment.kind, segment.cache_scope, segment.text]))
        .collect::<Vec<_>>();
    json!({
        "model": request.model,
        "segments": segments,
        "rendered_prefix": request.prompt.rendered_prefix,
        "cache_breakpoints": request.prompt.cache_breakpoints,
        "compacted_prefix": request.compacted_prefix,
        "tools": request.tools,
    })
    .to_string()
}

/// Check that every request repeats the previous one's cached fields and
/// extends its messages, except the requests at `resets`, which open a new
/// window.
fn check_prefix_stability(requests: &[ProviderRequest], resets: &[usize]) -> Result<(), String> {
    let encode = |messages: &[Message]| serde_json::to_string(messages).expect("messages encode");
    (1..requests.len())
        .filter(|index| !resets.contains(index))
        .try_for_each(|index| {
            let (before, after) = (&requests[index - 1], &requests[index]);
            let (cached_before, cached_after) = (cached_fields(before), cached_fields(after));
            if cached_before != cached_after {
                return Err(format!(
                    "request {index} rewrote the cached fields\nbefore: {cached_before}\nafter:  {cached_after}"
                ));
            }
            let seen = before.messages.len();
            if after.messages.len() < seen
                || encode(&before.messages) != encode(&after.messages[..seen])
            {
                return Err(format!(
                    "request {index} rewrote earlier messages\nbefore: {}\nafter:  {}",
                    encode(&before.messages),
                    encode(&after.messages)
                ));
            }
            Ok(())
        })
}

/// One scripted reply: text, or a batch of `echo` calls.
#[derive(Clone, Copy)]
enum Reply {
    Text,
    Calls(usize),
}

struct ScriptedProvider {
    replies: Mutex<VecDeque<Reply>>,
    requests: Mutex<Vec<ProviderRequest>>,
}

#[async_trait]
impl Provider for ScriptedProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        ProviderCapabilities::default()
    }

    async fn stream(
        &self,
        request: ProviderRequest,
        _cancel: CancellationToken,
    ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
        self.requests.lock().unwrap().push(request);
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected provider request");
        let id = MessageId::new();
        let block = BlockId::new();
        let (body, stop_reason) = match reply {
            Reply::Text => (
                vec![
                    StreamEvent::TextStart { id: block.clone() },
                    StreamEvent::TextDelta {
                        id: block.clone(),
                        delta: "done".to_owned(),
                    },
                    StreamEvent::TextEnd { id: block },
                ],
                StopReason::EndTurn,
            ),
            Reply::Calls(count) => (
                (0..count)
                    .flat_map(|_| {
                        let block = BlockId::new();
                        [
                            StreamEvent::ToolCallStart {
                                id: block.clone(),
                                tool_call_id: ToolCallId::new(),
                                name: "echo".into(),
                            },
                            StreamEvent::ToolArgsDelta {
                                id: block.clone(),
                                delta: "{}".to_owned(),
                            },
                            StreamEvent::ToolCallEnd { id: block },
                        ]
                    })
                    .collect(),
                StopReason::ToolUse,
            ),
        };
        let events = std::iter::once(StreamEvent::MessageStart { id: id.clone() })
            .chain(body)
            .chain([StreamEvent::MessageEnd {
                id,
                stop_reason,
                response_id: None,
            }])
            .map(Ok);
        Ok(stream::iter(events.collect::<Vec<_>>()).boxed())
    }
}

#[derive(Debug)]
struct Echo;

#[async_trait]
impl Tool for Echo {
    fn spec(&self) -> ToolSpec {
        ToolSpec {
            name: ToolName::from("echo"),
            description: "Echo".to_owned(),
            input_schema: json!({"type": "object", "properties": {}}),
            concurrency: ToolConcurrency::ReadOnly,
            capabilities: ToolCapabilities::default(),
            provider_aliases: Default::default(),
        }
    }

    async fn execute(
        &self,
        _context: ToolContext,
        _input: serde_json::Value,
    ) -> anyhow::Result<ToolResult> {
        Ok(ToolResult::Text {
            text: "echoed".to_owned(),
        })
    }
}

/// Keeps only the window's system messages. The next request opens a new
/// window whose system prompt carries them, `PreCompact` context included.
struct Wipe;

#[async_trait]
impl CompactionStrategy for Wipe {
    async fn compact(
        &self,
        ctx: CompactionContext<'_>,
    ) -> anyhow::Result<Option<CompactionEffects>> {
        Ok(Some(CompactionEffects {
            messages: ctx
                .state()
                .messages
                .iter()
                .filter(|message| matches!(message, Message::System(_)))
                .cloned()
                .collect(),
            compacted_context: Default::default(),
            result: halter_protocol::CompactionResult {
                compacted_count: ctx.state().messages.len(),
                summary: "wiped".to_owned(),
            },
            usage: Usage::default(),
        }))
    }
}

/// Every hook event the turn loop and manual compaction dispatch, each
/// returning additional context and a system message.
fn chatty_hooks() -> RegisteredHooks {
    let mut hooks = RegisteredHooks::default();
    let events = [
        HookEventName::SessionStart,
        HookEventName::UserPromptSubmit,
        HookEventName::PreToolUse,
        HookEventName::PostToolUse,
        HookEventName::Stop,
        HookEventName::PreCompact,
        HookEventName::PostCompact,
    ];
    let respond = |event: HookEventName| {
        HookResponse::passthrough()
            .with_additional_context(format!("context from {}", event.canonical_name()))
            .with_system_message(format!("message from {}", event.canonical_name()))
    };
    for event in events {
        hooks.register(
            PluginId::from("chatty"),
            RegisteredHookPriority::AfterPlugins,
            Hook::callback(event, move |_input| async move { respond(event) }),
        );
    }
    hooks
}

/// A session where hooks return context at every point they run: tool
/// batches, the end of each turn, several turns, a manual compaction and a
/// resume. Returns the requests the provider received and the index of the
/// one that opens the compacted window.
async fn chatty_session() -> (Vec<ProviderRequest>, usize) {
    let temp = tempfile::tempdir().expect("tempdir");
    let script = [
        // Turn 1: a two-call batch, then its answer.
        Reply::Calls(2),
        Reply::Text,
        // Turn 2: Stop context from turn 1 arrives first.
        Reply::Text,
        // Turn 3, after the compaction: one call, then its answer.
        Reply::Calls(1),
        Reply::Text,
        // Turn 4, after resuming.
        Reply::Text,
    ];
    let provider = Arc::new(ScriptedProvider {
        replies: Mutex::new(script.into()),
        requests: Mutex::new(Vec::new()),
    });
    let mut services = crate::session_driver_tests::services(provider.clone());
    let unique = Arc::get_mut(&mut services).expect("unique services");
    unique.tools.register(Arc::new(Echo));
    unique.registered_hooks = Arc::new(chatty_hooks());
    unique.compaction = Arc::new(Wipe);
    let runtime = SessionRuntime::new(services.clone());
    let session = runtime
        .new_session(SessionInit {
            working_dir: temp.path().to_path_buf(),
            ..SessionInit::default()
        })
        .await
        .expect("session");
    let turn = async |session: &crate::SessionExecutor, text: &str| {
        let events = session
            .submit_turn(Turn::user(text))
            .await
            .expect("submit turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("turn events");
        assert!(
            events
                .iter()
                .any(|event| matches!(event.payload, SessionEventPayload::TurnCompleted { .. })),
            "turn {text:?} did not complete: {events:#?}"
        );
    };

    turn(&session, "one").await;
    turn(&session, "two").await;
    session.compact("manual", None).await.expect("compact");
    let compacted_at = provider.requests.lock().unwrap().len();
    turn(&session, "three").await;
    let resumed = SessionRuntime::new(services)
        .resume(session.session_id())
        .await
        .expect("resume")
        .expect("stored session");
    turn(&resumed, "four").await;

    let requests = provider.requests.lock().unwrap().clone();
    assert_eq!(
        requests.len(),
        script.len(),
        "every scripted reply was used"
    );
    (requests, compacted_at)
}

#[tokio::test]
async fn hook_context_never_rewrites_the_prefix_within_a_window() {
    let (requests, compacted_at) = chatty_session().await;

    check_prefix_stability(&requests, &[compacted_at]).unwrap_or_else(|error| panic!("{error}"));
}

#[tokio::test]
async fn hook_output_lands_where_the_hook_ran() {
    let (requests, compacted_at) = chatty_session().await;

    // Each request's new system messages: the hook events they came from,
    // and whether they lead the window (and so join its system prompt).
    let all = [
        "SessionStart",
        "UserPromptSubmit",
        "PreToolUse",
        "PostToolUse",
        "Stop",
        "PreCompact",
        "PostCompact",
    ];
    let cases: [(usize, bool, &[&str]); 6] = [
        (0, true, &["SessionStart", "UserPromptSubmit"]),
        (1, false, &["PreToolUse", "PostToolUse"]),
        (2, false, &["Stop", "UserPromptSubmit"]),
        // The compacted window keeps the old window's system messages.
        (compacted_at, true, &all),
        (4, false, &["PreToolUse", "PostToolUse"]),
        (5, false, &["Stop", "SessionStart", "UserPromptSubmit"]),
    ];
    for (index, leading, events) in cases {
        let messages = &requests[index].messages;
        let start = match index {
            0 => 0,
            index if index == compacted_at => 0,
            index => requests[index - 1].messages.len(),
        };
        let first_other = messages
            .iter()
            .position(|message| !matches!(message, Message::System(_)))
            .expect("a user message");
        let added = messages
            .iter()
            .enumerate()
            .skip(start)
            .filter_map(|(at, message)| match message {
                Message::System(system) => Some((at < first_other, system.text.as_str())),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert!(
            added.iter().all(|(leads, _)| *leads == leading),
            "request {index}: expected leading = {leading}: {added:?}"
        );
        let texts = added
            .iter()
            .map(|(_, text)| *text)
            .collect::<std::collections::BTreeSet<_>>();
        let expected = events
            .iter()
            .flat_map(|event| ["context", "message"].map(|kind| format!("{kind} from {event}")))
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            texts,
            expected.iter().map(String::as_str).collect(),
            "request {index}"
        );
    }
}

#[test]
fn check_prefix_stability_accepts_appends_and_rejects_rewrites() {
    let base = || {
        let tool = |name: &str| ToolSpec {
            name: name.into(),
            ..Echo.spec()
        };
        let segment = crate::system_prompt_segment("follow plan");
        ProviderRequest {
            session_id: Default::default(),
            turn_id: Default::default(),
            model: crate::session_driver_tests::services(Arc::new(
                halter_providers::FakeProvider::default(),
            ))
            .models
            .default_model()
            .expect("default model"),
            prompt: halter_protocol::AssembledPrompt {
                segments: vec![segment.clone()],
                transcript: Vec::new(),
                ordered_segments: vec![segment],
                prefix_cache_key: String::new(),
                rendered_prefix: "follow plan".to_owned(),
                rendered_transcript: String::new(),
                rendered: String::new(),
                cache_breakpoints: Default::default(),
                system_segment_count: 1,
                skill_segment_count: 0,
            },
            compacted_prefix: Vec::new(),
            messages: vec![Message::User(halter_protocol::UserMessage::text("hi"))],
            tools: vec![tool("echo"), tool("read")],
            previous_response_id: None,
            new_messages_start: 0,
        }
    };
    let base = base();
    let extended = || {
        let mut request = base.clone();
        request
            .messages
            .push(Message::User(halter_protocol::UserMessage::text("more")));
        request
    };
    type Change = fn(&mut ProviderRequest);
    let cases: [(&str, Change, &[usize], Option<&str>); 7] = [
        ("an append keeps the prefix", |_| {}, &[], None),
        (
            "a system prompt edit rewrites the cached fields",
            |request| request.prompt.rendered_prefix.push_str("\n\nmore context"),
            &[],
            Some("rewrote the cached fields"),
        ),
        (
            "a tool description edit rewrites the cached fields",
            |request| request.tools[0].description.push('!'),
            &[],
            Some("rewrote the cached fields"),
        ),
        (
            "a tool reorder rewrites the cached fields",
            |request| request.tools.reverse(),
            &[],
            Some("rewrote the cached fields"),
        ),
        (
            "an earlier message edit rewrites the messages",
            |request| {
                request.messages[0] = Message::User(halter_protocol::UserMessage::text("bye"))
            },
            &[],
            Some("rewrote earlier messages"),
        ),
        (
            "a shrunk window rewrites the messages",
            |request| request.messages.clear(),
            &[],
            Some("rewrote earlier messages"),
        ),
        (
            "a reset allows a new window",
            |request| {
                request.messages.clear();
                request.prompt.rendered_prefix.push_str("\n\nsummary");
            },
            &[1],
            None,
        ),
    ];
    for (name, change, resets, expected) in cases {
        let mut after = extended();
        change(&mut after);
        let outcome = check_prefix_stability(&[base.clone(), after], resets);
        match expected {
            None => assert_eq!(outcome, Ok(()), "{name}"),
            Some(error) => assert!(
                outcome
                    .as_ref()
                    .is_err_and(|message| message.contains(error)),
                "{name}: {outcome:?}"
            ),
        }
    }
}
