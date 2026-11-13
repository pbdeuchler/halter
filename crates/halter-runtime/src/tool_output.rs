// pattern: Imperative Shell

use std::io::Write;
use std::path::{Path, PathBuf};

use halter_protocol::{MessageId, SessionId, ToolError, ToolOutputElision, ToolResult};
use serde_json::Value;
use tracing::warn;

use crate::trace_recorder::sanitize_session_id;

#[derive(Debug, Clone, PartialEq, Eq)]
/// Bound on the text one tool result puts in the context.
pub struct ToolOutputSettings {
    /// Largest model text committed whole, in bytes. `0` disables the bound.
    pub max_bytes: usize,
    /// Where the full text of an elided result is written, as
    /// `<spill_dir>/<sanitized session id>/<message id>.txt`, readable by
    /// its owner only. Must be absolute: the model reads the file back from
    /// the session's working directory. The runtime never deletes these files.
    pub spill_dir: PathBuf,
}

impl Default for ToolOutputSettings {
    fn default() -> Self {
        Self {
            max_bytes: 50 * 1024,
            spill_dir: default_spill_dir(std::env::var("USER").ok().as_deref()),
        }
    }
}

/// `halter-<user>/tool-output` under the system temporary directory, which
/// the default read roots cover. The directory is created owner-only, so on
/// a temporary directory shared between users each needs a name of their own.
fn default_spill_dir(user: Option<&str>) -> PathBuf {
    let name = user.map_or_else(
        || "halter".to_owned(),
        |user| format!("halter-{}", sanitize_session_id(user)),
    );
    std::env::temp_dir().join(name).join("tool-output")
}

#[derive(Debug, PartialEq, Eq)]
/// The two ends kept of a text that is over budget.
pub(crate) struct Elided<'a> {
    head: &'a str,
    middle: &'a str,
    tail: &'a str,
}

/// Split `text` into the head and tail that fit `budget` bytes, or `None`
/// when the text fits or `budget` is `0`. Each end gets half the budget and
/// is cut back to a line boundary when that keeps at least half of its
/// share, otherwise to a character boundary.
pub(crate) fn elide_middle(text: &str, budget: usize) -> Option<Elided<'_>> {
    if budget == 0 || text.len() <= budget {
        return None;
    }
    let head_budget = budget / 2;
    let tail_budget = budget - head_budget;

    let head_end = (0..=head_budget)
        .rev()
        .find(|&index| text.is_char_boundary(index))
        .unwrap_or(0);
    let head_end = text[..head_end]
        .rfind('\n')
        .map(|newline| newline + 1)
        .filter(|&end| end >= head_budget / 2)
        .unwrap_or(head_end);

    let tail_start = (text.len() - tail_budget..=text.len())
        .find(|&index| text.is_char_boundary(index))
        .unwrap_or(text.len());
    let tail_start = if text[..tail_start].ends_with('\n') {
        tail_start
    } else {
        text[tail_start..]
            .find('\n')
            .map(|newline| tail_start + newline + 1)
            .filter(|&start| text.len() - start >= tail_budget / 2)
            .unwrap_or(tail_start)
    };

    Some(Elided {
        head: &text[..head_end],
        middle: &text[head_end..tail_start],
        tail: &text[tail_start..],
    })
}

impl Elided<'_> {
    /// The head and tail around one marker line. `spill` is the file
    /// holding the whole output; the sizes are those of the text cut here,
    /// which is not always the file's (a JSON value is written reformatted).
    pub(crate) fn render(&self, spill: Option<&Path>) -> String {
        let bytes = self.middle.len();
        let lines = self.middle.matches('\n').count();
        let total = self.head.len() + bytes + self.tail.len();
        let kept = match spill {
            Some(path) => format!(
                "full output: {} (grep it, or read it with offset and limit)",
                path.display()
            ),
            None => "the full output was not kept".to_owned(),
        };
        let separator = if self.head.is_empty() || self.head.ends_with('\n') {
            ""
        } else {
            "\n"
        };
        format!(
            "{}{separator}[{bytes} bytes ({lines} lines) of this {total}-byte result elided here; {kept}]\n{}",
            self.head, self.tail
        )
    }

    /// The rendered text with the record of what was removed from it.
    fn bounded(&self, spill_path: Option<PathBuf>) -> (String, ToolOutputElision) {
        let original_bytes = self.head.len() + self.middle.len() + self.tail.len();
        (
            self.render(spill_path.as_deref()),
            ToolOutputElision {
                original_bytes: original_bytes as u64,
                spill_path,
            },
        )
    }
}

/// Write `full` to a new file only its owner can read. An existing file or
/// symlink at `path` is refused, never followed or replaced.
fn write_private(path: &Path, full: &str) -> std::io::Result<()> {
    let mut dir = std::fs::DirBuilder::new();
    let mut file = std::fs::OpenOptions::new();
    dir.recursive(true);
    file.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
        dir.mode(0o700);
        file.mode(0o600);
    }
    if let Some(parent) = path.parent() {
        dir.create(parent)?;
    }
    file.open(path)?.write_all(full.as_bytes())
}

/// Keep `full` in a new file at `path`. A failed write is not an error: it
/// is logged and the caller's marker says the full output was not kept.
async fn spill(path: PathBuf, session_id: &SessionId, full: String) -> Option<PathBuf> {
    let written = tokio::task::spawn_blocking({
        let path = path.clone();
        move || write_private(&path, &full)
    })
    .await
    .map_err(std::io::Error::other)
    .and_then(|written| written);
    written
        .inspect_err(|error| {
            warn!(
                session_id = %session_id,
                path = %path.display(),
                error = %error,
                "failed to keep full tool output"
            );
        })
        .is_ok()
        .then_some(path)
}

/// `text` cut to `budget` around a marker naming `spill`, or `None` when it
/// fits, `budget` is `0`, or the marker would cost as much as the cut saves.
pub(crate) fn elide_to<'a>(text: &'a str, budget: usize, spill: &Path) -> Option<Elided<'a>> {
    elide_middle(text, budget).filter(|elided| elided.render(Some(spill)).len() < text.len())
}

/// `text` bounded to the output budget, with `full()` kept in the spill
/// file of tool message `message_id`; `None` leaves the text as it is.
async fn bound_text(
    settings: &ToolOutputSettings,
    session_id: &SessionId,
    message_id: &MessageId,
    text: &str,
    full: impl FnOnce() -> String,
) -> Option<(String, ToolOutputElision)> {
    // Session ids come from the host and may be any string; message ids are
    // minted here, one per tool message, so a file is never written twice.
    let path = settings
        .spill_dir
        .join(sanitize_session_id(&session_id.0))
        .join(format!("{message_id}.txt"));
    let elided = elide_to(text, settings.max_bytes, &path)?;
    Some(elided.bounded(spill(path, session_id, full()).await))
}

/// Bound a tool error's message to the output budget, keeping the full
/// message in the spill file of tool message `message_id`.
pub(crate) async fn bound_error(
    settings: &ToolOutputSettings,
    session_id: &SessionId,
    message_id: &MessageId,
    error: ToolError,
) -> ToolError {
    let message = &error.message;
    match bound_text(settings, session_id, message_id, message, || {
        message.clone()
    })
    .await
    {
        Some((message, _)) => ToolError::new(message),
        None => error,
    }
}

/// Bound a tool result's model text, keeping its structured value whole.
/// Over budget, the full output goes to the spill file of tool message
/// `message_id` and the result records what was removed.
pub(crate) async fn bound_result(
    settings: &ToolOutputSettings,
    session_id: &SessionId,
    message_id: &MessageId,
    result: ToolResult,
) -> ToolResult {
    let text = result.to_model_string();
    let full = || match &result {
        // One line of JSON cannot be searched or paged; pretty-printed it can.
        ToolResult::Json { value } if !value.is_string() => format!("{value:#}"),
        _ => text.to_string(),
    };
    let Some((model_text, elision)) =
        bound_text(settings, session_id, message_id, &text, full).await
    else {
        return result;
    };
    ToolResult::Rendered {
        value: match result {
            ToolResult::Json { value } | ToolResult::Rendered { value, .. } => value,
            ToolResult::Text { text } => Value::String(text),
            ToolResult::Empty => Value::Null,
        },
        model_text,
        elision: Some(elision),
    }
}

/// What the transcript holds of a committed result: the text the model
/// reads. A rendered result's structured value and elision record stay in
/// the completion event alone, so they are stored once and never carried
/// in session state.
pub(crate) fn transcript_content(result: &ToolResult) -> ToolResult {
    match result {
        ToolResult::Rendered { model_text, .. } => ToolResult::Text {
            text: model_text.clone(),
        },
        result => result.clone(),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use futures::stream::{self, BoxStream};
    use futures::{StreamExt, TryStreamExt};
    use halter_hooks::{
        Hook, HookEventName, HookResponse, RegisteredHookPriority, RegisteredHooks,
    };
    use halter_protocol::{
        BlockId, Message, PluginId, ProviderCapabilities, ProviderError, ProviderRequest,
        SessionEventPayload, StopReason, StreamEvent, ToolCallId, ToolCapabilities,
        ToolConcurrency, ToolName, ToolResultMessage, ToolSpec, Turn,
    };
    use halter_providers::{FakeProvider, Provider};
    use halter_session::SqliteSessionStore;
    use halter_tools::{Tool, ToolContext, register_builtin_tools};
    use proptest::prelude::*;
    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::session_driver_tests::services;
    use crate::{RuntimeServices, SessionInit, SessionRuntime};

    fn elided(text: &str, budget: usize) -> (&str, &str, &str) {
        let Elided { head, middle, tail } = elide_middle(text, budget).expect("over budget");
        (head, middle, tail)
    }

    #[test]
    fn elide_middle_keeps_text_within_budget_whole() {
        for (text, budget) in [
            ("", 8),
            ("12345678", 8),
            ("é\nline\n", 100),
            ("far over the budget", 0),
        ] {
            assert_eq!(elide_middle(text, budget), None, "{text:?} at {budget}");
        }
    }

    #[test]
    fn elide_middle_cuts_on_lines_then_characters() {
        for (text, budget, expected) in [
            // Both ends fall back to a whole line.
            (
                "aaaa\nbbbb\ncccc\ndddd\neeee\n",
                14,
                ("aaaa\n", "bbbb\ncccc\ndddd\n", "eeee\n"),
            ),
            // One giant line has no line boundary to keep.
            ("0123456789", 4, ("01", "234567", "89")),
            // A newline that would give up more than half an end is ignored.
            (
                "a\nbcdefghijklmnopqrstuvw\nz",
                12,
                ("a\nbcde", "fghijklmnopqrs", "tuvw\nz"),
            ),
            // The cut never lands inside a character: `é` is two bytes.
            ("ééééé", 5, ("é", "ééé", "é")),
            ("aéééé", 4, ("a", "ééé", "é")),
            // A budget too small to hold a character on one side keeps none.
            ("éé", 1, ("", "éé", "")),
            ("abc", 1, ("", "ab", "c")),
        ] {
            assert_eq!(elided(text, budget), expected, "{text:?} at {budget}");
        }
    }

    #[test]
    fn elided_text_renders_one_marker_line_between_the_ends() {
        let text = "aaaa\nbbbb\ncccc\ndddd\neeee\n";
        let lines = elide_middle(text, 14).expect("over budget");
        assert_eq!(
            lines.render(Some(Path::new("/tmp/out.txt"))),
            "aaaa\n[15 bytes (3 lines) of this 25-byte result elided here; full output: /tmp/out.txt \
             (grep it, or read it with offset and limit)]\neeee\n"
        );
        let giant = elide_middle("0123456789", 4).expect("over budget");
        assert_eq!(
            giant.render(None),
            "01\n[6 bytes (0 lines) of this 10-byte result elided here; the full output was not kept]\n89"
        );
    }

    proptest! {
        #[test]
        fn elided_ends_are_a_prefix_and_suffix_within_budget(
            text in "[a-cé\n]{0,200}",
            budget in 0usize..120,
            spill in proptest::option::of("[a-z/]{1,40}"),
        ) {
            match elide_middle(&text, budget) {
                None => prop_assert!(budget == 0 || text.len() <= budget),
                Some(ends) => {
                    prop_assert!(text.len() > budget);
                    prop_assert_eq!([ends.head, ends.middle, ends.tail].concat(), text.as_str());
                    prop_assert!(ends.head.len() <= budget / 2);
                    prop_assert!(ends.tail.len() <= budget - budget / 2);
                    prop_assert!(!ends.middle.is_empty());
                    // The marker line, with the path it names, is the only
                    // allowance over the budget.
                    let rendered = ends.render(spill.as_deref().map(Path::new));
                    let marker = rendered.len() - ends.head.len() - ends.tail.len();
                    let path = spill.map_or(0, |path| path.len());
                    prop_assert!(marker <= 124 + path, "marker is {marker} bytes");
                    prop_assert_eq!(rendered.lines().filter(|line| line.contains("elided here")).count(), 1);
                }
            }

            // What is committed is never longer than what the tool returned,
            // whether or not the spill file could be written.
            let path = Path::new("/tmp/halter-user/tool-output/session/message.txt");
            if let Some(ends) = elide_to(&text, budget, path) {
                prop_assert!(ends.render(Some(path)).len() < text.len());
                prop_assert!(ends.render(None).len() < text.len());
            }
        }
    }

    #[test]
    fn default_spill_dir_is_per_user_under_the_temp_dir() {
        for (user, name) in [
            (Some("alice"), "halter-alice"),
            (Some("../a b"), "halter-___a_b"),
            (None, "halter"),
        ] {
            assert_eq!(
                default_spill_dir(user),
                std::env::temp_dir().join(name).join("tool-output")
            );
        }
    }

    const BUDGET: usize = 64;
    const CALL_ID: &str = "call/1";

    /// Calls `tool` once per request, under the same call id, until `calls`
    /// results are in the conversation, then answers. Records every request.
    struct Calls {
        tool: &'static str,
        arguments: Value,
        calls: usize,
        requests: Mutex<Vec<ProviderRequest>>,
    }

    #[async_trait]
    impl Provider for Calls {
        fn capabilities(&self) -> ProviderCapabilities {
            FakeProvider::default().capabilities()
        }

        async fn stream(
            &self,
            request: ProviderRequest,
            cancel: CancellationToken,
        ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
            let called = tool_messages(&request.messages).count();
            self.requests
                .lock()
                .expect("requests")
                .push(request.clone());
            if called >= self.calls {
                return FakeProvider::default().stream(request, cancel).await;
            }
            let (message, block) = (MessageId::new(), BlockId::new());
            Ok(stream::iter([
                Ok(StreamEvent::MessageStart {
                    id: message.clone(),
                }),
                Ok(StreamEvent::ToolCallStart {
                    id: block.clone(),
                    tool_call_id: ToolCallId::from(CALL_ID),
                    name: ToolName::from(self.tool),
                }),
                Ok(StreamEvent::ToolArgsDelta {
                    id: block.clone(),
                    delta: self.arguments.to_string(),
                }),
                Ok(StreamEvent::ToolCallEnd { id: block }),
                Ok(StreamEvent::MessageEnd {
                    id: message,
                    stop_reason: StopReason::ToolUse,
                    response_id: None,
                }),
            ])
            .boxed())
        }
    }

    /// A host tool that returns what it was built with.
    struct Fixed(Result<ToolResult, String>);

    #[async_trait]
    impl Tool for Fixed {
        fn spec(&self) -> ToolSpec {
            ToolSpec {
                name: ToolName::from("fixed"),
                description: "Return a fixed result".to_owned(),
                input_schema: json!({ "type": "object", "properties": {} }),
                concurrency: ToolConcurrency::Exclusive,
                capabilities: ToolCapabilities::default(),
                provider_aliases: Default::default(),
            }
        }

        async fn execute(
            &self,
            _context: ToolContext,
            _input: Value,
        ) -> anyhow::Result<ToolResult> {
            self.0.clone().map_err(anyhow::Error::msg)
        }
    }

    type Outcome = Result<ToolResult, ToolError>;

    fn tool_messages(messages: &[Message]) -> impl Iterator<Item = &ToolResultMessage> {
        messages.iter().filter_map(|message| match message {
            Message::Tool(tool) => Some(tool),
            _ => None,
        })
    }

    /// The tool messages in a session's log, in order.
    async fn messages(services: &RuntimeServices, session_id: &SessionId) -> Vec<Message> {
        let log = services.sessions.replay(session_id).await.expect("replay");
        let messages = log.into_iter().filter_map(|event| match event.payload {
            SessionEventPayload::MessageItem { message } => Some(message),
            _ => None,
        });
        messages.collect()
    }

    /// Each committed tool message with the completion event committed with
    /// it: the transcript holds the event's model text and nothing else.
    async fn committed(
        services: &RuntimeServices,
        session_id: &SessionId,
    ) -> Vec<(ToolResultMessage, Outcome)> {
        let log = services.sessions.replay(session_id).await.expect("replay");
        let outcomes = log.iter().filter_map(|event| match &event.payload {
            SessionEventPayload::ToolExecutionCompleted { outcome } => Some(outcome.result.clone()),
            _ => None,
        });
        let messages = messages(services, session_id).await;
        let committed = tool_messages(&messages).cloned().zip(outcomes);
        let committed = committed.collect::<Vec<_>>();
        assert_eq!(committed.len(), tool_messages(&messages).count());
        for (message, outcome) in &committed {
            match outcome {
                Ok(result) => {
                    assert_eq!(message.error, None);
                    assert_eq!(message.to_model_string(), result.to_model_string());
                    assert_eq!(
                        matches!(message.content, ToolResult::Text { .. }),
                        matches!(
                            result,
                            ToolResult::Text { .. } | ToolResult::Rendered { .. }
                        )
                    );
                }
                Err(error) => {
                    assert_eq!(message.content, ToolResult::Empty);
                    assert_eq!(message.error.as_ref(), Some(error));
                }
            }
        }
        committed
    }

    struct Run {
        services: Arc<RuntimeServices>,
        provider: Arc<Calls>,
        session_id: SessionId,
        spill_dir: PathBuf,
        temp: tempfile::TempDir,
    }

    impl Run {
        /// The text the provider was sent for the tool result in its latest
        /// request.
        fn sent(&self) -> String {
            let requests = self.provider.requests.lock().expect("requests");
            let sent = tool_messages(&requests.last().expect("request").messages);
            let sent = sent.last().expect("tool result sent");
            sent.to_model_string().into_owned()
        }

        fn spill_path(&self, message: &ToolResultMessage) -> PathBuf {
            let session = self.spill_dir.join(&self.session_id.0);
            session.join(format!("{}.txt", message.id))
        }

        /// The one committed tool message and its completion event.
        async fn committed(&self) -> (ToolResultMessage, Outcome) {
            let committed = committed(&self.services, &self.session_id).await;
            let [committed] = <[_; 1]>::try_from(committed).expect("one tool result");
            committed
        }
    }

    fn settings(dir: &Path) -> ToolOutputSettings {
        ToolOutputSettings {
            max_bytes: BUDGET,
            spill_dir: dir.join("spill"),
        }
    }

    /// Services under a 64-byte output budget spilling below `dir`.
    fn configured(
        provider: Arc<Calls>,
        dir: &Path,
        configure: impl FnOnce(&mut RuntimeServices, &Path),
    ) -> Arc<RuntimeServices> {
        let mut services = services(provider);
        let unique = Arc::get_mut(&mut services).expect("unique services");
        unique.tool_output = settings(dir);
        configure(unique, dir);
        services
    }

    /// Run one turn in which the model calls `tool` `calls` times.
    async fn run_calls(
        calls: usize,
        tool: &'static str,
        arguments: Value,
        configure: impl FnOnce(&mut RuntimeServices, &Path),
    ) -> Run {
        let temp = tempfile::tempdir().expect("tempdir");
        let provider = Arc::new(Calls {
            tool,
            arguments,
            calls,
            requests: Mutex::new(Vec::new()),
        });
        let services = configured(provider.clone(), temp.path(), configure);
        let session = SessionRuntime::new(services.clone())
            .new_session(SessionInit {
                working_dir: temp.path().to_path_buf(),
                ..SessionInit::default()
            })
            .await
            .expect("session");
        session
            .submit_turn(Turn::user("go"))
            .await
            .expect("submit turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("turn completes");
        Run {
            session_id: session.session_id().clone(),
            spill_dir: services.tool_output.spill_dir.clone(),
            services,
            provider,
            temp,
        }
    }

    async fn run(
        tool: &'static str,
        arguments: Value,
        configure: impl FnOnce(&mut RuntimeServices, &Path),
    ) -> Run {
        run_calls(1, tool, arguments, configure).await
    }

    fn fixed(result: Result<ToolResult, String>) -> impl FnOnce(&mut RuntimeServices, &Path) {
        |services, _| services.tools.register(Arc::new(Fixed(result)))
    }

    fn hook(event: HookEventName, response: fn() -> HookResponse) -> Arc<RegisteredHooks> {
        let mut hooks = RegisteredHooks::default();
        hooks.register(
            PluginId::from("test"),
            RegisteredHookPriority::AfterPlugins,
            Hook::callback(event, move |_| async move { response() }),
        );
        Arc::new(hooks)
    }

    fn big() -> String {
        (0..40).map(|line| format!("line \"{line}\" é\n")).collect()
    }

    fn bounded(full: &str, spill: Option<&Path>) -> String {
        let ends = elide_middle(full, BUDGET).expect("over budget");
        ends.render(spill)
    }

    fn not_kept(value: Value) -> ToolResult {
        ToolResult::Rendered {
            value,
            model_text: bounded(&big(), None),
            elision: Some(ToolOutputElision {
                original_bytes: big().len() as u64,
                spill_path: None,
            }),
        }
    }

    /// The structured value of an elided result whose model text was cut
    /// from `full`, was sent as committed, and names a file holding `spilled`.
    async fn assert_elided(run: &Run, full: &str, spilled: &str) -> Value {
        let (message, outcome) = run.committed().await;
        let spill = run.spill_path(&message);
        let Ok(ToolResult::Rendered {
            value,
            model_text,
            elision: Some(elision),
        }) = outcome
        else {
            panic!("expected an elided result");
        };
        assert_eq!(model_text, bounded(full, Some(&spill)));
        assert_eq!(run.sent(), model_text);
        assert_eq!(message.content, ToolResult::Text { text: model_text });
        assert_eq!(elision.original_bytes, full.len() as u64);
        assert_eq!(std::fs::read_to_string(&spill).expect("spill"), spilled);
        assert_eq!(elision.spill_path, Some(spill));
        value
    }

    #[tokio::test]
    async fn oversized_builtin_output_is_committed_elided_with_its_value_whole() {
        let content = big();
        let run = run("read", json!({ "path": "big.txt" }), |services, dir| {
            register_builtin_tools(&services.tools, &["read".to_owned()]);
            std::fs::write(dir.join("big.txt"), big()).expect("write file");
        })
        .await;

        let (message, _) = run.committed().await;
        let full = std::fs::read_to_string(run.spill_path(&message)).expect("spill");
        assert!(full.starts_with('[') && full.ends_with(&content), "{full}");
        let value = assert_elided(&run, &full, &full).await;
        assert_eq!(value["content"], content);
    }

    #[tokio::test]
    async fn oversized_legacy_json_output_is_spilled_pretty_printed() {
        let original = json!({ "exit_code": 0, "output": big() });
        let result = ToolResult::Json {
            value: original.clone(),
        };
        let run = run("fixed", json!({}), fixed(Ok(result))).await;

        let pretty = format!("{original:#}");
        assert!(pretty.lines().count() > 2, "{pretty}");
        let value = assert_elided(&run, &original.to_string(), &pretty).await;
        assert_eq!(value, original);
    }

    #[tokio::test]
    async fn oversized_hook_rewritten_output_is_committed_elided() {
        let run = run("fixed", json!({}), |services, _| {
            services
                .tools
                .register(Arc::new(Fixed(Ok(ToolResult::Empty))));
            services.registered_hooks = hook(HookEventName::PostToolUse, || {
                HookResponse::passthrough().with_updated_output(json!(big()))
            });
        })
        .await;

        let value = assert_elided(&run, &big(), &big()).await;
        assert_eq!(value, json!(big()));
    }

    #[tokio::test]
    async fn oversized_hook_block_reason_is_committed_elided() {
        let run = run("fixed", json!({}), |services, _| {
            services
                .tools
                .register(Arc::new(Fixed(Ok(ToolResult::Empty))));
            services.registered_hooks =
                hook(HookEventName::PreToolUse, || HookResponse::block(big()));
        })
        .await;

        let messages = messages(&run.services, &run.session_id).await;
        let [message] = tool_messages(&messages).collect::<Vec<_>>()[..] else {
            panic!("expected one tool message");
        };
        let spill = run.spill_path(message);
        let full = std::fs::read_to_string(&spill).expect("spill");
        assert!(full.contains(big().trim_end()), "{full}");
        let error = message.error.as_ref().expect("blocked call is an error");
        assert_eq!(error.message, bounded(&full, Some(&spill)));
        assert_eq!(run.sent(), error.message);
    }

    #[tokio::test]
    async fn output_within_budget_or_unbounded_is_committed_unchanged() {
        let small = ToolResult::Json {
            value: json!({ "ok": true }),
        };
        let large = ToolResult::Text { text: big() };
        // Over budget by less than the marker would add.
        let barely = ToolResult::Text {
            text: "x".repeat(BUDGET + 1),
        };
        let within = run("fixed", json!({}), fixed(Ok(small.clone()))).await;
        let over = run("fixed", json!({}), fixed(Ok(barely.clone()))).await;
        let unbounded = run("fixed", json!({}), |services, _| {
            services.tools.register(Arc::new(Fixed(Ok(large.clone()))));
            services.tool_output.max_bytes = 0;
        })
        .await;

        for (run, result) in [(within, small), (over, barely), (unbounded, large)] {
            let (message, outcome) = run.committed().await;
            assert_eq!(message.content, result);
            assert_eq!(run.sent(), result.to_model_string());
            assert_eq!(outcome, Ok(result));
            assert!(!run.spill_dir.exists());
        }
    }

    #[tokio::test]
    async fn same_call_id_twice_in_a_session_spills_to_two_files() {
        let result = ToolResult::Text { text: big() };
        let run = run_calls(2, "fixed", json!({}), fixed(Ok(result))).await;

        let committed = committed(&run.services, &run.session_id).await;
        let [(first, _), (second, _)] = &committed[..] else {
            panic!("expected two tool results");
        };
        assert_eq!(first.call_id, second.call_id);
        assert_ne!(run.spill_path(first), run.spill_path(second));
        for message in [first, second] {
            let spill = run.spill_path(message);
            let text = bounded(&big(), Some(&spill));
            assert_eq!(message.content, ToolResult::Text { text });
            assert_eq!(std::fs::read_to_string(spill).expect("spill"), big());
        }
    }

    #[tokio::test]
    async fn failed_spill_write_still_commits_the_elided_result() {
        let run = run("fixed", json!({}), |services, dir| {
            services
                .tools
                .register(Arc::new(Fixed(Ok(ToolResult::Text { text: big() }))));
            // A file where the directory should be makes every write fail.
            std::fs::write(dir.join("spill"), "").expect("occupy spill dir");
        })
        .await;

        let (message, outcome) = run.committed().await;
        let text = bounded(&big(), None);
        assert_eq!(outcome, Ok(not_kept(json!(big()))));
        assert_eq!(run.sent(), text);
        assert_eq!(message.content, ToolResult::Text { text });
    }

    #[tokio::test]
    async fn existing_spill_file_is_refused_and_left_alone() {
        let temp = tempfile::tempdir().expect("tempdir");
        let settings = settings(temp.path());
        let (session, message) = (SessionId::from("session"), MessageId::new());
        let path = settings.spill_dir.join("session");
        std::fs::create_dir_all(&path).expect("session dir");
        let path = path.join(format!("{message}.txt"));
        std::fs::write(&path, "mine").expect("occupy spill file");

        let result = ToolResult::Text { text: big() };
        let result = bound_result(&settings, &session, &message, result).await;
        let error = bound_error(&settings, &session, &message, ToolError::new(big())).await;

        assert_eq!(result, not_kept(json!(big())));
        assert_eq!(error, ToolError::new(bounded(&big(), None)));
        assert_eq!(std::fs::read_to_string(path).expect("spill"), "mine");
    }

    #[tokio::test]
    async fn hostile_session_id_cannot_move_the_spill_file() {
        let temp = tempfile::tempdir().expect("tempdir");
        let settings = settings(temp.path());
        let outside = temp.path().join("outside");
        let absolute = outside.to_string_lossy().into_owned();

        for id in ["../outside", "../../outside", &absolute, "..", ""] {
            let session = SessionId::from(id);
            let (result, failure) = (MessageId::new(), MessageId::new());
            let text = ToolResult::Text { text: big() };
            bound_result(&settings, &session, &result, text).await;
            bound_error(&settings, &session, &failure, ToolError::new(big())).await;

            let dir = settings.spill_dir.join(sanitize_session_id(id));
            assert_eq!(dir.parent(), Some(&*settings.spill_dir), "{id:?}");
            for message in [result, failure] {
                let spilled = std::fs::read_to_string(dir.join(format!("{message}.txt")));
                assert_eq!(spilled.expect("spill"), big(), "{id:?}");
            }
        }
        let beside = std::fs::read_dir(temp.path()).expect("read dir");
        let beside = beside.map(|entry| entry.expect("entry").file_name());
        assert_eq!(beside.collect::<Vec<_>>(), ["spill"]);
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn spilled_output_is_readable_by_its_owner_only() {
        use std::os::unix::fs::PermissionsExt;

        let temp = tempfile::tempdir().expect("tempdir");
        let settings = settings(temp.path());
        let (session, message) = (SessionId::from("session"), MessageId::new());
        let result = ToolResult::Text { text: big() };
        bound_result(&settings, &session, &message, result).await;

        let dir = settings.spill_dir.join("session");
        for (path, mode) in [
            (dir.join(format!("{message}.txt")), 0o600),
            (dir, 0o700),
            (settings.spill_dir, 0o700),
        ] {
            let permissions = std::fs::metadata(&path).expect("metadata").permissions();
            assert_eq!(permissions.mode() & 0o777, mode, "{}", path.display());
        }
    }

    #[tokio::test]
    async fn failing_tool_keeps_its_error_and_an_oversized_one_is_bounded() {
        let short = run("fixed", json!({}), fixed(Err("boom".to_owned()))).await;
        let (message, outcome) = short.committed().await;
        assert_eq!(message.error, Some(ToolError::new("boom")));
        assert_eq!(outcome, Err(ToolError::new("boom")));
        assert_eq!(short.sent(), "boom");

        let long = run("fixed", json!({}), fixed(Err(big()))).await;
        let (message, _) = long.committed().await;
        let spill = long.spill_path(&message);
        let bounded = bounded(&big(), Some(&spill));
        assert_eq!(message.error, Some(ToolError::new(bounded.as_str())));
        assert_eq!(long.sent(), bounded);
        assert_eq!(std::fs::read_to_string(spill).expect("spill"), big());
    }

    #[tokio::test]
    async fn session_restored_from_disk_sends_the_committed_message_unchanged() {
        fn sqlite(services: &mut RuntimeServices, dir: &Path) {
            let result = ToolResult::Json {
                value: json!({ "output": big() }),
            };
            services.tools.register(Arc::new(Fixed(Ok(result))));
            let store = SqliteSessionStore::open(dir.join("sessions.db"));
            services.sessions = Arc::new(store.expect("open sqlite store"));
        }
        fn encode(value: &impl serde::Serialize) -> String {
            serde_json::to_string(value).expect("encode")
        }
        let run = run("fixed", json!({}), sqlite).await;
        let before = committed(&run.services, &run.session_id).await;
        let sent = run.sent();
        assert!(sent.contains("elided here"));
        std::fs::remove_dir_all(&run.spill_dir).expect("remove spill dir");

        // A second runtime over the reopened database has only what the
        // first one stored.
        let reopened = configured(run.provider.clone(), run.temp.path(), sqlite);
        let runtime = SessionRuntime::new(reopened.clone());
        let resumed = runtime.resume(&run.session_id).await;
        let resumed = resumed.expect("resume").expect("session exists");
        resumed
            .submit_turn(Turn::user("again"))
            .await
            .expect("submit turn")
            .try_collect::<Vec<_>>()
            .await
            .expect("turn completes");

        let after = committed(&reopened, &run.session_id).await;
        assert_eq!(encode(&after), encode(&before));
        {
            let requests = run.provider.requests.lock().expect("requests");
            assert_eq!(requests.len(), 3);
            let resent = tool_messages(&requests[2].messages).collect::<Vec<_>>();
            assert_eq!(encode(&resent), encode(&[&before[0].0]));
        }
        assert_eq!(run.sent(), sent);
        assert!(!run.spill_dir.exists(), "restore must not write output");
    }
}
