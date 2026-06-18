// pattern: Imperative Shell
//
// The model-judge provider multiplexes one model call across a panel of models,
// asks a synthesis model to stack-rank and synthesize their responses, and then
// hands that synthesis to a default model whose stream is what the caller sees.
// From the runtime's perspective it is an ordinary `Provider`; all multiplexing
// stays inside this provider.
//
// Each panelist is given the original context plus a constant framing prefix
// (`PANEL_PREFIX`) so it answers as one advisory voice on a judged panel —
// comparable prose rather than an executed action or raw tool call. The prefix
// is injected only on the panel path; the synthesis judge and default member
// start from the unmodified request, so it never leaks past the panel.

use std::sync::Arc;

use async_trait::async_trait;
use chrono::Utc;
use futures::stream::{BoxStream, StreamExt};
use halter_protocol::{
    AssistantMessage, AssistantPart, BlockId, CompactionWindow, Message, MessageId,
    ProviderCapabilities, ProviderCompactionRequest, ProviderCompactionResponse, ProviderError,
    ProviderRequest, ReplayMeta, ResolvedModel, StopReason, StreamEvent, ToolCall,
    ToolCapabilities, ToolConcurrency, ToolName, ToolResult, ToolResultMessage, ToolSpec,
    UserMessage,
};
use serde_json::{Value, json};
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

use crate::Provider;

/// Tracing target shared by every model-judge telemetry event.
pub const MODEL_JUDGE_TRACE_TARGET: &str = "halter::model_judge";
/// Name of the synthesis-only tool used to submit panel stack rankings.
pub const MODEL_JUDGE_RANK_TOOL: &str = "rank_responses";

/// Maximum synthesis round trips (ranking call + synthesis text) before the
/// model-judge provider stops looping and uses whatever text it has.
const MAX_SYNTHESIS_ROUNDS: usize = 3;
const EMPTY_SYNTHESIS_ERROR: &str =
    "model-judge synthesis produced neither a ranking nor synthesis text";

/// Framing prepended as the first user message of every panel request. It tells
/// each panelist it is one advisory voice on a judged panel so it returns
/// comparable natural-language analysis instead of executing the task or
/// communicating through raw tool calls. Injected only by [`panel_messages`] on
/// the panel path — the synthesis judge and default member never see it — and
/// always at the same position so panel prompt caches stay warm across a turn.
const PANEL_PREFIX: &str = "You are one of several expert models on a review \
    panel. The conversation that follows is an in-progress agent session. An \
    independent synthesis judge will read your response alongside the other \
    panelists', stack-rank them, and distill the best guidance for the model \
    that actually acts next — so your job is to advise, not to execute.\n\nThink \
    deeply about how you would approach what is being asked: your first \
    inclination, the next one to three actions or thoughts you would commit to, \
    why you would take them, and the decision tree branching from that first \
    move — including the alternatives you weighed and the risks or unknowns that \
    would change your mind.\n\nRespond in self-contained prose so the judge can \
    compare you against the other panelists. Do not rely on tool calls to \
    communicate; if you would use a tool, name it and describe what you would \
    pass and why. Be specific and decisive — hedged or vague guidance is hard to \
    rank. Do not address the user; you are producing advisory analysis, not a \
    reply.";

/// One participating model plus its adapter.
#[derive(Clone)]
pub struct ModelJudgeMember {
    pub provider: Arc<dyn Provider>,
    pub model: ResolvedModel,
}

/// A model abstraction that judges and synthesizes a panel of responses before
/// answering through its default model.
#[derive(Clone)]
pub struct ModelJudgeProvider {
    default: ModelJudgeMember,
    synthesis: ModelJudgeMember,
    panel: Vec<ModelJudgeMember>,
}

impl ModelJudgeProvider {
    /// Build a model-judge provider from its three roles.
    #[must_use]
    pub fn new(
        default: ModelJudgeMember,
        synthesis: ModelJudgeMember,
        panel: Vec<ModelJudgeMember>,
    ) -> Self {
        Self {
            default,
            synthesis,
            panel,
        }
    }
}

#[async_trait]
impl Provider for ModelJudgeProvider {
    fn capabilities(&self) -> ProviderCapabilities {
        // The user-visible stream comes from the default member, so mirror its
        // capabilities (tool support, compaction strategy, token limits, ...).
        self.default.provider.capabilities()
    }

    fn compaction_window(&self, messages: &[Message]) -> Option<CompactionWindow> {
        self.default.provider.compaction_window(messages)
    }

    async fn stream(
        &self,
        request: ProviderRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
        let candidates = self.run_panels(&request, &cancel).await;

        if candidates.is_empty() {
            warn!(
                target: MODEL_JUDGE_TRACE_TARGET,
                "model-judge produced no panel responses; falling back to the default model alone"
            );
            return self.run_default(&request, None, cancel).await;
        }

        match self.run_synthesis(&request, &candidates, &cancel).await {
            Ok(synthesis) => self.run_default(&request, Some(synthesis), cancel).await,
            Err(error) => {
                warn!(
                    target: MODEL_JUDGE_TRACE_TARGET,
                    %error,
                    "model-judge synthesis failed; falling back to the default model alone"
                );
                self.run_default(&request, None, cancel).await
            }
        }
    }

    async fn compact(
        &self,
        request: ProviderCompactionRequest,
        cancel: CancellationToken,
    ) -> anyhow::Result<ProviderCompactionResponse> {
        let inner = ProviderCompactionRequest {
            session_id: request.session_id,
            model: self.default.model.clone(),
            compacted_prefix: request.compacted_prefix,
            messages: request.messages,
            tools: request.tools,
            instructions: request.instructions,
        };
        self.default.provider.compact(inner, cancel).await
    }
}

/// A successfully collected panel response.
struct Candidate {
    /// Stable, unique identifier shown to the judge and logged as telemetry.
    id: String,
    /// Underlying model name, for telemetry.
    model: String,
    /// Rendered response body (text plus any proposed tool calls).
    body: String,
}

impl ModelJudgeProvider {
    async fn run_panels(
        &self,
        request: &ProviderRequest,
        cancel: &CancellationToken,
    ) -> Vec<Candidate> {
        let ids = candidate_ids(&self.panel);
        let panel_context = panel_messages(&request.messages);
        let futures = self.panel.iter().enumerate().map(|(index, panelist)| {
            let inner = inner_request(
                request,
                panelist.model.clone(),
                panel_context.clone(),
                request.tools.clone(),
            );
            let provider = panelist.provider.clone();
            let cancel = cancel.child_token();
            async move {
                let collected = match provider.stream(inner, cancel).await {
                    Ok(events) => collect_message(events).await,
                    Err(error) => Err(ProviderError::new(error.to_string(), false)),
                };
                (index, collected)
            }
        });

        let mut results = futures::future::join_all(futures).await;
        results.sort_by_key(|(index, _)| *index);

        let mut candidates = Vec::new();
        for (index, collected) in results {
            let id = ids[index].clone();
            let model = self.panel[index].model.model.clone();
            match collected {
                Ok(message) => {
                    let body = render_candidate_body(&message);
                    info!(
                        target: MODEL_JUDGE_TRACE_TARGET,
                        event = "panel_response",
                        candidate_id = %id,
                        model = %model,
                        tool_calls = message.tool_calls.len(),
                        response = %body,
                        "model-judge panel response"
                    );
                    candidates.push(Candidate { id, model, body });
                }
                Err(error) => {
                    warn!(
                        target: MODEL_JUDGE_TRACE_TARGET,
                        event = "panel_error",
                        candidate_id = %id,
                        model = %model,
                        %error,
                        "model-judge panel model failed"
                    );
                }
            }
        }
        candidates
    }

    async fn run_synthesis(
        &self,
        request: &ProviderRequest,
        candidates: &[Candidate],
        cancel: &CancellationToken,
    ) -> anyhow::Result<String> {
        let mut messages = request.messages.clone();
        messages.push(Message::User(UserMessage::text(synthesis_instructions(
            candidates,
        ))));

        let rank_tool = rank_tool_spec();
        let mut recorded = false;

        for round in 0..MAX_SYNTHESIS_ROUNDS {
            let inner = inner_request(
                request,
                self.synthesis.model.clone(),
                messages.clone(),
                vec![rank_tool.clone()],
            );
            let events = self
                .synthesis
                .provider
                .stream(inner, cancel.child_token())
                .await?;
            let collected = collect_message(events)
                .await
                .map_err(|error| anyhow::anyhow!("model-judge synthesis stream failed: {error}"))?;

            let rank_call = collected
                .tool_calls
                .iter()
                .find(|call| call.name.0 == MODEL_JUDGE_RANK_TOOL)
                .cloned();

            if let Some(call) = &rank_call
                && !recorded
            {
                record_rankings(&call.arguments);
                recorded = true;
            }

            let synthesis = collected.text.trim();
            if !synthesis.is_empty() {
                info!(
                    target: MODEL_JUDGE_TRACE_TARGET,
                    event = "synthesis",
                    synthesis = %synthesis,
                    "model-judge synthesis message"
                );
                return Ok(synthesis.to_owned());
            }

            // No synthesis text yet. If the model ranked, acknowledge the tool
            // call and let it continue. If it produced neither a ranking nor
            // text, retry with an explicit correction before falling back.
            let Some(call) = rank_call else {
                if round + 1 >= MAX_SYNTHESIS_ROUNDS {
                    anyhow::bail!("{EMPTY_SYNTHESIS_ERROR} after {MAX_SYNTHESIS_ROUNDS} attempts");
                }
                warn!(
                    target: MODEL_JUDGE_TRACE_TARGET,
                    event = "synthesis_retry",
                    attempt = round + 1,
                    max_rounds = MAX_SYNTHESIS_ROUNDS,
                    rank_recorded = recorded,
                    "{EMPTY_SYNTHESIS_ERROR}; retrying synthesis model"
                );
                messages.push(Message::User(synthesis_retry_message(recorded)));
                continue;
            };
            messages.push(Message::Assistant(assistant_with_tool_call(
                &collected.text,
                call.clone(),
            )));
            messages.push(Message::Tool(ToolResultMessage {
                id: MessageId::new(),
                call_id: call.id.clone(),
                content: ToolResult::Json {
                    value: json!({ "status": "recorded" }),
                },
                error: None,
                created_at: Utc::now(),
            }));
        }

        anyhow::bail!("model-judge synthesis did not produce synthesis text within the round limit")
    }

    async fn run_default(
        &self,
        request: &ProviderRequest,
        synthesis: Option<String>,
        cancel: CancellationToken,
    ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
        let mut messages = request.messages.clone();
        if let Some(synthesis) = synthesis {
            messages.push(Message::User(synthesis_guidance_message(&synthesis)));
        }
        // The default member streams to the user, so preserve provider-native
        // state (response chaining, compacted prefix) from the original request.
        let inner = ProviderRequest {
            session_id: request.session_id.clone(),
            turn_id: request.turn_id.clone(),
            model: self.default.model.clone(),
            prompt: request.prompt.clone(),
            compacted_prefix: request.compacted_prefix.clone(),
            messages,
            tools: request.tools.clone(),
            previous_response_id: request.previous_response_id.clone(),
            new_messages_start: request.new_messages_start,
        };
        self.default.provider.stream(inner, cancel).await
    }
}

/// A single model response collected from a provider stream.
struct CollectedMessage {
    text: String,
    tool_calls: Vec<ToolCall>,
}

struct PartialToolCall {
    block_id: BlockId,
    tool_call_id: halter_protocol::ToolCallId,
    name: ToolName,
    arguments: String,
}

async fn collect_message(
    mut events: BoxStream<'static, Result<StreamEvent, ProviderError>>,
) -> Result<CollectedMessage, ProviderError> {
    let mut text = String::new();
    let mut partials: Vec<PartialToolCall> = Vec::new();

    while let Some(item) = events.next().await {
        match item? {
            StreamEvent::TextDelta { delta, .. } => text.push_str(&delta),
            StreamEvent::ToolCallStart {
                id,
                tool_call_id,
                name,
            } => partials.push(PartialToolCall {
                block_id: id,
                tool_call_id,
                name,
                arguments: String::new(),
            }),
            StreamEvent::ToolArgsDelta { id, delta } => {
                if let Some(partial) = partials.iter_mut().rev().find(|p| p.block_id == id) {
                    partial.arguments.push_str(&delta);
                }
            }
            StreamEvent::Error { error } => return Err(error),
            _ => {}
        }
    }

    let tool_calls = partials
        .into_iter()
        .map(|partial| ToolCall {
            id: partial.tool_call_id,
            name: partial.name,
            arguments: parse_tool_arguments(&partial.arguments),
        })
        .collect();

    Ok(CollectedMessage { text, tool_calls })
}

fn parse_tool_arguments(raw: &str) -> Value {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return json!({});
    }
    serde_json::from_str(trimmed).unwrap_or_else(|_| json!({ "_raw": raw }))
}

/// Build a non-chaining, stateless inner request for a panel or synthesis call.
fn inner_request(
    base: &ProviderRequest,
    model: ResolvedModel,
    messages: Vec<Message>,
    tools: Vec<ToolSpec>,
) -> ProviderRequest {
    ProviderRequest {
        session_id: base.session_id.clone(),
        turn_id: base.turn_id.clone(),
        model,
        prompt: base.prompt.clone(),
        // Cleared: provider-native state (response chaining, compacted reasoning
        // items) belongs to the default member and is not portable across the
        // panel/synthesis members, which may be different providers entirely.
        compacted_prefix: Vec::new(),
        messages,
        tools,
        previous_response_id: None,
        new_messages_start: 0,
    }
}

/// Prepend [`PANEL_PREFIX`] as the first user message so each panelist answers
/// as an advisory voice on a judged panel. The prefix sits right after the
/// system block and before the (append-only) conversation, so its position is
/// stable across a turn and panel prompt caches stay warm.
fn panel_messages(messages: &[Message]) -> Vec<Message> {
    let mut prefixed = Vec::with_capacity(messages.len() + 1);
    prefixed.push(Message::User(UserMessage::text(PANEL_PREFIX)));
    prefixed.extend(messages.iter().cloned());
    prefixed
}

fn assistant_with_tool_call(text: &str, call: ToolCall) -> AssistantMessage {
    let mut parts = Vec::new();
    if !text.trim().is_empty() {
        parts.push(AssistantPart::Text {
            text: text.to_owned(),
        });
    }
    parts.push(AssistantPart::ToolCall(call));
    AssistantMessage {
        id: MessageId::new(),
        created_at: Utc::now(),
        parts,
        stop_reason: Some(StopReason::ToolUse),
        usage: None,
        replay_meta: ReplayMeta::default(),
    }
}

/// Compute stable, unique candidate ids from the panelist models. Duplicate
/// model names are disambiguated with a positional suffix so rankings remain
/// unambiguous.
fn candidate_ids(panel: &[ModelJudgeMember]) -> Vec<String> {
    panel
        .iter()
        .enumerate()
        .map(|(index, panelist)| {
            let name = &panelist.model.model;
            let duplicate = panel
                .iter()
                .enumerate()
                .any(|(other, candidate)| other != index && candidate.model.model == *name);
            if duplicate {
                format!("{name}#{index}")
            } else {
                name.clone()
            }
        })
        .collect()
}

fn render_candidate_body(message: &CollectedMessage) -> String {
    let mut body = message.text.trim().to_owned();
    if !message.tool_calls.is_empty() {
        if !body.is_empty() {
            body.push_str("\n\n");
        }
        body.push_str("[proposed tool calls]");
        for call in &message.tool_calls {
            body.push_str(&format!("\n- {}({})", call.name, call.arguments));
        }
    }
    if body.is_empty() {
        "[empty response]".to_owned()
    } else {
        body
    }
}

fn synthesis_instructions(candidates: &[Candidate]) -> String {
    let mut prompt = String::new();
    prompt.push_str(
        "You are a synthesis judge. Below are candidate responses to the most recent user \
         message, each produced by a different model.\n\n\
         Workflow:\n\
         1. In your first response, call the `rank_responses` tool exactly once with a stack \
         ranking of every candidate from best (rank 1) to worst, keyed by the candidate's \
         `model_id`.\n\
         2. After the tool result is returned, write a synthesis that does NOT merge the \
         candidates but JUDGES and SYNTHESIZES them: cover each candidate's strengths, weaknesses, pros, and cons, \
         and finish with an overall synthesis of all responses that is greater than the sum of it's parts.\n\n\
         Valid outputs are only a `rank_responses` tool call or non-empty synthesis text. Never \
         return an empty assistant message. If you are uncertain, make a best-effort ranking and \
         concise synthesis. Do not address the user directly; your synthesis is advisory context \
         for another model.\n\nCandidates:\n",
    );
    for candidate in candidates {
        prompt.push_str(&format!(
            "\n--- model_id: {} (model: {}) ---\n{}\n",
            candidate.id, candidate.model, candidate.body
        ));
    }
    prompt
}

fn synthesis_retry_message(rank_recorded: bool) -> UserMessage {
    let next_action = if rank_recorded {
        "The ranking has already been recorded. Write the synthesis text now."
    } else {
        "Call `rank_responses` exactly once with every candidate, then continue to synthesis text."
    };
    UserMessage::text(format!(
        "Your previous model-judge synthesis response was invalid because it contained neither a \
         `rank_responses` tool call nor synthesis text.\n\n{next_action}\n\nDo not return an \
         empty message. If uncertain, make a best-effort ranking and concise synthesis."
    ))
}

fn synthesis_guidance_message(synthesis: &str) -> UserMessage {
    UserMessage::text(format!(
        "The following model-judge synthesis evaluates candidate responses to \
         the most recent user message. Treat it as advisory context for your \
         answer, not as a new request from the user.\n\n\
         <model_judge_synthesis>\n{}\n</model_judge_synthesis>",
        synthesis.trim()
    ))
}

fn rank_tool_spec() -> ToolSpec {
    ToolSpec {
        name: ToolName::from(MODEL_JUDGE_RANK_TOOL),
        description: "Submit a stack ranking of the candidate panel responses for quality \
             telemetry. Provide every candidate's model_id together with its rank, where rank 1 \
             is the best response."
            .to_owned(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "rankings": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "model_id": { "type": "string" },
                            "rank": { "type": "integer", "minimum": 1 }
                        },
                        "required": ["model_id", "rank"]
                    }
                }
            },
            "required": ["rankings"]
        }),
        concurrency: ToolConcurrency::ReadOnly,
        capabilities: ToolCapabilities::default(),
        provider_aliases: Default::default(),
    }
}

fn record_rankings(arguments: &Value) {
    let Some(rankings) = arguments.get("rankings").and_then(Value::as_array) else {
        warn!(
            target: MODEL_JUDGE_TRACE_TARGET,
            event = "ranking_invalid",
            arguments = %arguments,
            "model-judge ranking call had no rankings array"
        );
        return;
    };

    info!(
        target: MODEL_JUDGE_TRACE_TARGET,
        event = "ranking",
        rankings = %arguments,
        count = rankings.len(),
        "model-judge stack ranking"
    );
    for entry in rankings {
        let model_id = entry.get("model_id").and_then(Value::as_str).unwrap_or("");
        let rank = entry.get("rank").and_then(Value::as_i64).unwrap_or(0);
        info!(
            target: MODEL_JUDGE_TRACE_TARGET,
            event = "ranking_entry",
            model_id = %model_id,
            rank,
            "model-judge ranking entry"
        );
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::VecDeque, sync::Mutex};

    use chrono::Utc;
    use futures::stream;
    use halter_protocol::{
        ApiKind, AssembledPrompt, BlockId, CacheBreakpoints, MessageId, ModelId, ModelRole,
        ProviderCapabilities, ProviderKind, ProviderName, StopReason, ToolCallId, UserPart,
    };

    use super::*;

    /// Provider that records every request it receives and replays a fixed set
    /// of stream events (optional ranking tool call plus optional text).
    struct RecordingProvider {
        requests: Arc<Mutex<Vec<ProviderRequest>>>,
        text: Option<String>,
        tool: Option<(String, Value)>,
    }

    #[derive(Clone)]
    struct ScriptedResponse {
        text: Option<String>,
        tool: Option<(String, Value)>,
    }

    struct SequencedRecordingProvider {
        requests: Arc<Mutex<Vec<ProviderRequest>>>,
        responses: Mutex<VecDeque<ScriptedResponse>>,
    }

    #[async_trait]
    impl Provider for RecordingProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }

        async fn stream(
            &self,
            request: ProviderRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
            self.requests.lock().expect("record request").push(request);

            let events = scripted_events(ScriptedResponse {
                text: self.text.clone(),
                tool: self.tool.clone(),
            });
            Ok(stream::iter(events).boxed())
        }
    }

    #[async_trait]
    impl Provider for SequencedRecordingProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }

        async fn stream(
            &self,
            request: ProviderRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
            self.requests.lock().expect("record request").push(request);
            let response = self
                .responses
                .lock()
                .expect("scripted responses")
                .pop_front()
                .expect("scripted provider response");
            Ok(stream::iter(scripted_events(response)).boxed())
        }
    }

    fn scripted_events(response: ScriptedResponse) -> Vec<Result<StreamEvent, ProviderError>> {
        let message_id = MessageId::new();
        let mut events = vec![Ok(StreamEvent::MessageStart {
            id: message_id.clone(),
        })];

        if let Some((name, arguments)) = response.tool {
            let block = BlockId::new();
            events.push(Ok(StreamEvent::ToolCallStart {
                id: block.clone(),
                tool_call_id: ToolCallId::new(),
                name: ToolName::from(name),
            }));
            events.push(Ok(StreamEvent::ToolArgsDelta {
                id: block.clone(),
                delta: arguments.to_string(),
            }));
            events.push(Ok(StreamEvent::ToolCallEnd { id: block }));
        }

        if let Some(text) = response.text {
            let block = BlockId::new();
            events.push(Ok(StreamEvent::TextStart { id: block.clone() }));
            events.push(Ok(StreamEvent::TextDelta {
                id: block.clone(),
                delta: text,
            }));
            events.push(Ok(StreamEvent::TextEnd { id: block }));
        }

        events.push(Ok(StreamEvent::MessageEnd {
            id: message_id,
            stop_reason: StopReason::EndTurn,
            response_id: None,
        }));
        events
    }

    struct CompactRecordingProvider {
        requests: Arc<Mutex<Vec<ProviderCompactionRequest>>>,
    }

    #[async_trait]
    impl Provider for CompactRecordingProvider {
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities {
                supports_compaction: true,
                ..ProviderCapabilities::default()
            }
        }

        async fn stream(
            &self,
            _request: ProviderRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<BoxStream<'static, Result<StreamEvent, ProviderError>>> {
            anyhow::bail!("stream should not be called by this test provider")
        }

        async fn compact(
            &self,
            request: ProviderCompactionRequest,
            _cancel: CancellationToken,
        ) -> anyhow::Result<ProviderCompactionResponse> {
            self.requests
                .lock()
                .expect("record compaction request")
                .push(request);
            Ok(ProviderCompactionResponse {
                output: Vec::new(),
                usage: halter_protocol::Usage::default(),
            })
        }
    }

    fn resolved(model: &str) -> ResolvedModel {
        ResolvedModel {
            role: ModelRole::Default,
            id: ModelId::from(model),
            provider: ProviderName::from("fake"),
            provider_kind: ProviderKind::OpenAi,
            api_kind: ApiKind::OpenAiResponses,
            model: model.to_owned(),
            max_input_tokens: None,
            max_output_tokens: None,
            reasoning: None,
            tokens_per_minute: None,
        }
    }

    fn member(
        model: &str,
        text: Option<&str>,
        tool: Option<(&str, Value)>,
    ) -> (ModelJudgeMember, Arc<Mutex<Vec<ProviderRequest>>>) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let provider = Arc::new(RecordingProvider {
            requests: requests.clone(),
            text: text.map(ToOwned::to_owned),
            tool: tool.map(|(name, arguments)| (name.to_owned(), arguments)),
        });
        (
            ModelJudgeMember {
                provider,
                model: resolved(model),
            },
            requests,
        )
    }

    fn sequenced_member(
        model: &str,
        responses: Vec<ScriptedResponse>,
    ) -> (ModelJudgeMember, Arc<Mutex<Vec<ProviderRequest>>>) {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let provider = Arc::new(SequencedRecordingProvider {
            requests: requests.clone(),
            responses: Mutex::new(VecDeque::from(responses)),
        });
        (
            ModelJudgeMember {
                provider,
                model: resolved(model),
            },
            requests,
        )
    }

    fn sample_request() -> ProviderRequest {
        ProviderRequest {
            session_id: Default::default(),
            turn_id: halter_protocol::TurnId::new(),
            model: resolved("model_judge"),
            prompt: AssembledPrompt {
                segments: Vec::new(),
                transcript: Vec::new(),
                ordered_segments: Vec::new(),
                prefix_cache_key: String::new(),
                rendered_prefix: String::new(),
                rendered_transcript: String::new(),
                rendered: String::new(),
                cache_breakpoints: CacheBreakpoints::default(),
                system_segment_count: 0,
                skill_segment_count: 0,
            },
            compacted_prefix: Vec::new(),
            messages: vec![Message::User(UserMessage {
                id: MessageId::new(),
                created_at: Utc::now(),
                parts: vec![UserPart::Text {
                    text: "what is the answer?".to_owned(),
                }],
            })],
            tools: Vec::new(),
            previous_response_id: None,
            new_messages_start: 0,
        }
    }

    async fn collect_text(
        stream: BoxStream<'static, Result<StreamEvent, ProviderError>>,
    ) -> String {
        collect_message(stream).await.expect("collect").text
    }

    /// True if any user message in the request contains `needle`. Used to assert
    /// the panel framing prefix reaches panels but not the synthesis/default
    /// members.
    fn request_mentions(request: &ProviderRequest, needle: &str) -> bool {
        request.messages.iter().any(|message| match message {
            Message::User(user) => user.plain_text().contains(needle),
            _ => false,
        })
    }

    #[tokio::test]
    async fn model_judge_judges_panel_and_streams_default_with_synthesis() {
        let (panel_a, panel_a_reqs) = member("panel-a", Some("answer A"), None);
        let (panel_b, panel_b_reqs) = member("panel-b", Some("answer B"), None);
        let rankings = json!({
            "rankings": [
                { "model_id": "panel-a", "rank": 1 },
                { "model_id": "panel-b", "rank": 2 }
            ]
        });
        let (synthesis, synthesis_reqs) = member(
            "synthesis",
            Some("A is stronger than B"),
            Some((MODEL_JUDGE_RANK_TOOL, rankings)),
        );
        let (default, default_reqs) = member("default", Some("final answer"), None);

        let model_judge = ModelJudgeProvider::new(default, synthesis, vec![panel_a, panel_b]);
        let events = model_judge
            .stream(sample_request(), CancellationToken::new())
            .await
            .expect("model_judge stream");
        let output = collect_text(events).await;

        // The user sees the default member's stream.
        assert_eq!(output, "final answer");

        // Both panels were called once with the original user message.
        let panel_a_reqs = panel_a_reqs.lock().unwrap();
        assert_eq!(panel_a_reqs.len(), 1);
        assert_eq!(panel_b_reqs.lock().unwrap().len(), 1);

        // The panel framing is the first message, and the original user message
        // still reaches the panelist after it.
        let panel_first = match &panel_a_reqs[0].messages[0] {
            Message::User(user) => user.plain_text(),
            other => {
                panic!("panel context should open with the framing user message, got {other:?}")
            }
        };
        assert!(panel_first.contains("one of several expert models on a review panel"));
        assert!(request_mentions(&panel_a_reqs[0], "what is the answer?"));

        // The synthesis member saw the candidate responses.
        let synthesis_reqs = synthesis_reqs.lock().unwrap();
        assert_eq!(synthesis_reqs.len(), 1);
        // The panel framing prefix never leaks to the synthesis judge.
        assert!(!request_mentions(&synthesis_reqs[0], "review panel"));
        let synthesis_user = synthesis_reqs[0]
            .messages
            .iter()
            .rev()
            .find_map(|message| match message {
                Message::User(user) => Some(user.plain_text()),
                _ => None,
            })
            .expect("synthesis user message");
        assert!(synthesis_user.contains("answer A"));
        assert!(synthesis_user.contains("answer B"));
        assert!(synthesis_user.contains(MODEL_JUDGE_RANK_TOOL));
        // The synthesis model is offered only the ranking tool.
        assert_eq!(synthesis_reqs[0].tools.len(), 1);
        assert_eq!(synthesis_reqs[0].tools[0].name.0, MODEL_JUDGE_RANK_TOOL);

        // The default member receives the synthesis as internal guidance, not
        // as a protocol-level transcript variant.
        let default_reqs = default_reqs.lock().unwrap();
        assert_eq!(default_reqs.len(), 1);
        let guidance = default_reqs[0]
            .messages
            .iter()
            .skip(1)
            .find_map(|message| match message {
                Message::User(user) => Some(user.plain_text()),
                _ => None,
            })
            .expect("guidance user message");
        assert!(guidance.contains("model-judge synthesis"));
        assert!(guidance.contains("A is stronger than B"));
        // ...and the panel framing prefix never leaks to the default member.
        assert!(!request_mentions(&default_reqs[0], "review panel"));
    }

    #[tokio::test]
    async fn model_judge_retries_empty_synthesis_response() {
        let (panel, _) = member("panel-a", Some("answer A"), None);
        let rankings = json!({
            "rankings": [
                { "model_id": "panel-a", "rank": 1 }
            ]
        });
        let (synthesis, synthesis_reqs) = sequenced_member(
            "synthesis",
            vec![
                ScriptedResponse {
                    text: None,
                    tool: None,
                },
                ScriptedResponse {
                    text: Some("A is the best available answer".to_owned()),
                    tool: Some((MODEL_JUDGE_RANK_TOOL.to_owned(), rankings)),
                },
            ],
        );
        let (default, default_reqs) = member("default", Some("final answer"), None);

        let model_judge = ModelJudgeProvider::new(default, synthesis, vec![panel]);
        let events = model_judge
            .stream(sample_request(), CancellationToken::new())
            .await
            .expect("model_judge stream");
        let output = collect_text(events).await;

        assert_eq!(output, "final answer");

        let synthesis_reqs = synthesis_reqs.lock().unwrap();
        assert_eq!(synthesis_reqs.len(), 2);
        let initial_prompt = synthesis_reqs[0]
            .messages
            .iter()
            .rev()
            .find_map(|message| match message {
                Message::User(user) => Some(user.plain_text()),
                _ => None,
            })
            .expect("initial synthesis prompt");
        assert!(initial_prompt.contains("Never return an empty assistant message"));

        let retry_prompt = synthesis_reqs[1]
            .messages
            .iter()
            .rev()
            .find_map(|message| match message {
                Message::User(user) => Some(user.plain_text()),
                _ => None,
            })
            .expect("retry synthesis prompt");
        assert!(retry_prompt.contains("previous model-judge synthesis response was invalid"));

        let default_reqs = default_reqs.lock().unwrap();
        let guidance = default_reqs[0]
            .messages
            .iter()
            .skip(1)
            .find_map(|message| match message {
                Message::User(user) => Some(user.plain_text()),
                _ => None,
            })
            .expect("guidance user message");
        assert!(guidance.contains("A is the best available answer"));
    }

    #[tokio::test]
    async fn model_judge_falls_back_to_default_when_panel_fails() {
        // A synthesis member that never produces text would error, but with no
        // panels there is nothing to judge, so we fall straight through.
        let (synthesis, synthesis_reqs) = member("synthesis", Some("unused"), None);
        let (default, default_reqs) = member("default", Some("fallback answer"), None);

        let model_judge = ModelJudgeProvider::new(default, synthesis, Vec::new());
        let events = model_judge
            .stream(sample_request(), CancellationToken::new())
            .await
            .expect("model_judge stream");
        let output = collect_text(events).await;

        assert_eq!(output, "fallback answer");
        // The synthesis model is skipped entirely when there are no panel responses.
        assert_eq!(synthesis_reqs.lock().unwrap().len(), 0);
        // The default request carries only the original user message in the
        // fallback path.
        let default_reqs = default_reqs.lock().unwrap();
        assert_eq!(default_reqs.len(), 1);
        assert_eq!(default_reqs[0].messages.len(), 1);
    }

    #[tokio::test]
    async fn model_judge_compaction_uses_default_member_model() {
        let compact_reqs = Arc::new(Mutex::new(Vec::new()));
        let default = ModelJudgeMember {
            provider: Arc::new(CompactRecordingProvider {
                requests: compact_reqs.clone(),
            }),
            model: resolved("gpt-5.5"),
        };
        let (synthesis, _) = member("synthesis", Some("unused"), None);
        let model_judge = ModelJudgeProvider::new(default, synthesis, Vec::new());

        let mut synthetic = resolved("model-judge:gpt-5.5");
        synthetic.provider = ProviderName::from("model-judge-default");
        let request = ProviderCompactionRequest {
            session_id: Default::default(),
            model: synthetic,
            compacted_prefix: Vec::new(),
            messages: Vec::new(),
            tools: Vec::new(),
            instructions: "compact".to_owned(),
        };

        model_judge
            .compact(request, CancellationToken::new())
            .await
            .expect("model-judge compaction");

        let compact_reqs = compact_reqs.lock().expect("compaction requests");
        assert_eq!(compact_reqs.len(), 1);
        assert_eq!(compact_reqs[0].model.provider.0, "fake");
        assert_eq!(compact_reqs[0].model.model, "gpt-5.5");
        assert!(!compact_reqs[0].model.model.starts_with("model-judge:"));
    }
}
