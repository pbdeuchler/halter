//! Shared wire types and runtime contracts for the halter workspace.
//!
//! This crate contains the serializable protocol structs that the runtime,
//! providers, hooks, tools, and session stores exchange. It intentionally
//! stays dependency-light and mostly data-oriented so higher-level crates can
//! agree on event, message, provider, and resource shapes without depending on
//! each other.
// pattern: Functional Core

use std::fmt;
use std::path::PathBuf;
use std::time::Duration;

use bytes::Bytes;
use chrono::{DateTime, Utc};
use indexmap::IndexMap;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use uuid::Uuid;

/// Shared string payloads stay as `String` for now; this is the swap point for any future `Arc<str>` migration.
pub type SharedStr = String;

/// Historical sampling temperature used by older config resolution. Provider
/// requests now omit temperature unless `[providers.<name>].temperature` is
/// configured explicitly.
pub const DEFAULT_TEMPERATURE: f32 = 0.7;
/// MIME/media type label used for binary message parts.
pub type MediaType = String;
/// Provider-issued signature attached to replayable reasoning blocks.
pub type ReplaySignature = String;
/// Stable hash of prompt, resource, file-view, or context content.
pub type ContentHash = String;
/// UTC timestamp used throughout session and event records.
pub type Timestamp = DateTime<Utc>;

macro_rules! id_type {
    ($name:ident) => {
        #[derive(
            Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
        )]
        #[doc = concat!("Opaque identifier for a protocol `", stringify!($name), "`.")]
        pub struct $name(pub String);

        impl $name {
            /// Generate a new random identifier.
            #[must_use]
            pub fn new() -> Self {
                Self(Uuid::new_v4().to_string())
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self(value.to_owned())
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self(value)
            }
        }
    };
}

macro_rules! string_wrapper {
    ($name:ident) => {
        #[derive(
            Debug,
            Clone,
            PartialEq,
            Eq,
            PartialOrd,
            Ord,
            Hash,
            Default,
            Serialize,
            Deserialize,
            JsonSchema,
        )]
        #[doc = concat!("String newtype for a protocol `", stringify!($name), "`.")]
        pub struct $name(pub String);

        impl fmt::Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str(&self.0)
            }
        }

        impl From<&str> for $name {
            fn from(value: &str) -> Self {
                Self(value.to_owned())
            }
        }

        impl From<String> for $name {
            fn from(value: String) -> Self {
                Self(value)
            }
        }
    };
}

id_type!(MessageId);
id_type!(BlockId);
id_type!(ToolCallId);
id_type!(PromptId);
id_type!(PromptSegmentId);
id_type!(SessionId);
id_type!(TurnId);
id_type!(SkillId);
id_type!(PluginId);
id_type!(AgentId);

string_wrapper!(Revision);
string_wrapper!(ModelId);
string_wrapper!(ToolName);
string_wrapper!(ToolAlias);
string_wrapper!(SkillName);
string_wrapper!(AgentName);
string_wrapper!(ProviderName);

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
#[serde(rename_all = "snake_case")]
/// Logical model slot selected by a turn or subagent request.
pub enum ModelRole {
    /// General model used for normal turn execution.
    Default,
    /// Planning model.
    Plan,
    /// Model used by spawned subagents.
    Subagent,
    /// Cheaper or faster model for small supporting tasks.
    Small,
}

impl ModelRole {
    /// Role used when no role-specific override is requested.
    #[must_use]
    pub const fn default_role() -> Self {
        Self::Default
    }

    /// Planning role.
    #[must_use]
    pub const fn plan() -> Self {
        Self::Plan
    }

    /// Subagent role.
    #[must_use]
    pub const fn subagent() -> Self {
        Self::Subagent
    }

    /// Small-task role.
    #[must_use]
    pub const fn small() -> Self {
        Self::Small
    }

    /// Stable config and wire-format spelling for the role.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Default => "default",
            Self::Plan => "plan",
            Self::Subagent => "subagent",
            Self::Small => "small",
        }
    }
}

impl Default for ModelRole {
    fn default() -> Self {
        Self::default_role()
    }
}

impl std::str::FromStr for ModelRole {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "default" => Ok(Self::Default),
            "plan" => Ok(Self::Plan),
            "subagent" => Ok(Self::Subagent),
            "small" => Ok(Self::Small),
            other => Err(format!(
                "unknown ModelRole '{other}'; expected one of: default, plan, subagent, small"
            )),
        }
    }
}

impl fmt::Display for ModelRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    JsonSchema,
)]
#[serde(rename_all = "snake_case")]
/// Controls whether child subagent events are forwarded into the parent stream.
pub enum SubagentEventForwarding {
    /// Keep subagent events in the subagent session only.
    #[default]
    Off,
    /// Forward subagent events into the parent session event stream.
    All,
}

impl SubagentEventForwarding {
    /// Whether forwarding is active.
    #[must_use]
    pub const fn is_enabled(self) -> bool {
        matches!(self, Self::All)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
/// Provider family used for capability selection and registry lookup.
pub enum ProviderKind {
    /// Anthropic Messages API.
    Anthropic,
    /// OpenAI APIs.
    OpenAi,
    /// OpenRouter passthrough APIs.
    OpenRouter,
    /// Deterministic local test provider.
    Fake,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// Wire API shape a resolved model expects its provider to use.
pub enum ApiKind {
    /// Anthropic `/v1/messages`.
    AnthropicMessages,
    /// OpenAI-compatible Responses API.
    OpenAiResponses,
    /// OpenAI-compatible Chat Completions API.
    OpenAiChat,
    /// Local fake provider.
    Fake,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// Provider reasoning budget requested for a model.
pub enum ReasoningEffort {
    /// Low reasoning budget.
    Low,
    /// Medium reasoning budget.
    Medium,
    /// High reasoning budget.
    High,
    /// Extra-high reasoning budget, for providers that expose it.
    Xhigh,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
/// How a FullTurn model-judge panelist's sub-session is sandboxed while it runs
/// a complete agentic turn. Only meaningful for FullTurn judges; OneShot
/// panelists never execute tools.
pub enum PanelIsolation {
    /// Panelists share the parent working directory but get a tool set with
    /// every mutating tool (write/edit/shell/process/task) filtered out. They
    /// can read, search, and reason but cannot change the workspace. Safe under
    /// concurrency; the default.
    #[default]
    ReadOnly,
    /// Panelists share the parent working directory with the parent's full tool
    /// set. Maximum fidelity, but concurrent panelists can clobber each other's
    /// writes — the caller owns that risk.
    SharedFull,
    /// Each panelist runs in its own git worktree with the full tool set, so it
    /// can mutate freely without colliding. Requires a git repository; falls
    /// back to [`PanelIsolation::SharedFull`] (with a warning) otherwise.
    Worktree,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
/// Token accounting reported by providers and accumulated by sessions.
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub cache_read_input_tokens: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// Why an assistant message ended.
pub enum StopReason {
    /// The model completed the turn normally.
    EndTurn,
    /// The model requested tool execution.
    ToolUse,
    /// The turn was interrupted before natural completion.
    Interrupted,
    /// The provider stopped after reaching the output-token limit.
    MaxTokens,
    /// The provider or runtime reported an error.
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
/// Provider metadata preserved on assistant messages for replay and diagnostics.
pub struct ReplayMeta {
    pub provider_name: Option<ProviderName>,
    pub model: Option<ModelId>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
/// Severity for non-fatal hook loading problems.
pub enum HookWarningSeverity {
    /// Warning that does not block resource compilation.
    #[default]
    Warning,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
/// Warning emitted while loading plugin hook files.
pub struct HookWarning {
    pub severity: HookWarningSeverity,
    pub category: SharedStr,
    pub plugin_id: Option<PluginId>,
    pub plugin_name: Option<SharedStr>,
    pub source_path: Option<PathBuf>,
    pub message: SharedStr,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// System instruction message carried in the transcript.
pub struct SystemMessage {
    pub id: MessageId,
    pub created_at: Timestamp,
    pub text: SharedStr,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// User message, including text and optional media parts.
pub struct UserMessage {
    pub id: MessageId,
    pub created_at: Timestamp,
    pub parts: Vec<UserPart>,
}

impl UserMessage {
    /// Build a text-only user message with a fresh id and current timestamp.
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            id: MessageId::new(),
            created_at: Utc::now(),
            parts: vec![UserPart::Text { text: text.into() }],
        }
    }

    /// Concatenate text parts with newlines, ignoring image and document parts.
    #[must_use]
    pub fn plain_text(&self) -> String {
        self.parts
            .iter()
            .filter_map(|part| match part {
                UserPart::Text { text } => Some(text.as_str()),
                UserPart::Image { .. } | UserPart::Document { .. } => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
/// Part of a user message.
pub enum UserPart {
    /// Plain text input.
    Text { text: SharedStr },
    /// Binary image payload plus media type.
    Image {
        media_type: MediaType,
        #[schemars(with = "Vec<u8>")]
        data: Bytes,
    },
    /// Binary document payload plus media type.
    Document {
        media_type: MediaType,
        #[schemars(with = "Vec<u8>")]
        data: Bytes,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Assistant message assembled from provider stream events.
pub struct AssistantMessage {
    pub id: MessageId,
    pub created_at: Timestamp,
    pub parts: Vec<AssistantPart>,
    pub stop_reason: Option<StopReason>,
    pub usage: Option<Usage>,
    pub replay_meta: ReplayMeta,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
/// Part of an assistant message.
pub enum AssistantPart {
    /// Text visible to the user.
    Text { text: SharedStr },
    /// Reasoning or thinking content, optionally replay-signed.
    Thinking(ThinkingBlock),
    /// Tool invocation requested by the model.
    ToolCall(ToolCall),
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Provider thinking block with optional replay signature.
pub struct ThinkingBlock {
    pub text: SharedStr,
    pub signature: Option<ReplaySignature>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Tool invocation requested by an assistant message.
pub struct ToolCall {
    pub id: ToolCallId,
    pub name: ToolName,
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Result message that answers a prior [`ToolCall`].
pub struct ToolResultMessage {
    pub id: MessageId,
    pub call_id: ToolCallId,
    pub content: ToolResult,
    pub error: Option<ToolError>,
    pub created_at: Timestamp,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(tag = "role", rename_all = "snake_case")]
/// A transcript item visible to providers and session stores.
pub enum Message {
    /// System instructions.
    System(SystemMessage),
    /// User input.
    User(UserMessage),
    /// Assistant output.
    Assistant(AssistantMessage),
    /// Tool result.
    Tool(ToolResultMessage),
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
/// Incremental provider output event.
pub enum StreamEvent {
    /// Start of an assistant message.
    MessageStart { id: MessageId },
    /// Start of a text block.
    TextStart { id: BlockId },
    /// Text block delta.
    TextDelta { id: BlockId, delta: SharedStr },
    /// End of a text block.
    TextEnd { id: BlockId },
    /// Start of a thinking block.
    ThinkingStart { id: BlockId },
    /// Thinking block delta.
    ThinkingDelta { id: BlockId, delta: SharedStr },
    /// End of a thinking block.
    ThinkingEnd {
        id: BlockId,
        signature: Option<ReplaySignature>,
    },
    /// Start of a tool call block.
    ToolCallStart {
        id: BlockId,
        tool_call_id: ToolCallId,
        name: ToolName,
    },
    /// Tool arguments delta, usually a JSON fragment.
    ToolArgsDelta { id: BlockId, delta: SharedStr },
    /// End of a tool call block.
    ToolCallEnd { id: BlockId },
    /// Provider token usage update.
    UsageUpdate { usage: Usage },
    /// End of an assistant message.
    MessageEnd {
        id: MessageId,
        stop_reason: StopReason,
        /// The provider's response ID, used for `previous_response_id` chaining.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        response_id: Option<String>,
    },
    /// Non-fatal warning surfaced by the provider adapter.
    ProviderWarning { message: SharedStr },
    /// Provider error surfaced through the stream.
    Error { error: ProviderError },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// User-submitted work unit for a session.
pub struct Turn {
    pub id: TurnId,
    pub user_message: UserMessage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_model: Option<ModelId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subagent_model: Option<ModelId>,
}

impl Turn {
    /// Build a turn from a text-only user message.
    #[must_use]
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            id: TurnId::new(),
            user_message: UserMessage::text(text),
            default_model: None,
            subagent_model: None,
        }
    }

    /// Override the default model for this turn.
    #[must_use]
    pub fn with_default_model(mut self, model: impl Into<ModelId>) -> Self {
        self.default_model = Some(model.into());
        self
    }

    /// Override the subagent model for subagents spawned during this turn.
    #[must_use]
    pub fn with_subagent_model(mut self, model: impl Into<ModelId>) -> Self {
        self.subagent_model = Some(model.into());
        self
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// Runtime state of a spawned subagent.
pub enum SubagentState {
    /// The subagent is still executing.
    Running,
    /// The subagent finished successfully.
    Completed,
    /// The subagent failed.
    Failed,
    /// The subagent was cancelled.
    Cancelled,
    /// The subagent was closed by the parent.
    Closed,
}

impl SubagentState {
    /// Whether the state cannot transition back to running.
    #[must_use]
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Cancelled | Self::Closed
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Request payload for the `spawn_subagent` tool.
pub struct SpawnSubagentRequest {
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_type: Option<AgentName>,
    #[serde(default)]
    pub fork_context: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Request payload for sending additional input to a subagent.
pub struct SendSubagentInputRequest {
    pub target: AgentId,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Request payload for waiting on one or more subagents.
pub struct WaitSubagentRequest {
    pub targets: Vec<AgentId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Request payload for closing a subagent.
pub struct CloseSubagentRequest {
    pub target: AgentId,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Snapshot of a subagent's visible state.
pub struct SubagentStatus {
    pub agent_id: AgentId,
    pub session_id: SessionId,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_type: Option<AgentName>,
    pub task: String,
    pub state: SubagentState,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_message: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

impl SubagentStatus {
    /// Whether the subagent is no longer running.
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Response from a subagent wait operation.
///
/// `status` is populated when one requested subagent reaches a terminal state.
/// `target_statuses` is populated on timeout with the current state of every
/// requested target.
pub struct WaitSubagentResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<SubagentStatus>,
    pub timed_out: bool,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub target_statuses: Vec<SubagentStatus>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Response returned after closing a subagent.
pub struct CloseSubagentResponse {
    pub previous_status: SubagentStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Minimal subagent spec used by session commands.
pub struct SubagentSpecWire {
    pub role: Option<ModelRole>,
    pub task: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
/// Command accepted by a session control plane.
pub enum SessionCommand {
    /// Submit a new turn.
    SubmitTurn { turn: Turn },
    /// Interrupt the active turn.
    InterruptTurn,
    /// Append session-scoped system guidance.
    AppendSystemPrompt { id: PromptId, text: SharedStr },
    /// Switch the active model role.
    SetModelRole { role: ModelRole },
    /// Switch to a concrete model id.
    SetModel { model: ModelId },
    /// Spawn a subagent.
    SpawnSubagent { spec: SubagentSpecWire },
    /// Reload resources before continuing.
    ReloadResources,
    /// Shut down the session.
    Shutdown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Delivery semantics for committed session events.
pub enum Delivery {
    /// Must be persisted and delivered in order.
    Lossless,
    /// May be dropped under pressure.
    BestEffort,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Text delta emitted into the public session event stream.
pub struct DeltaItem {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Completed tool execution paired with the original call.
pub struct ToolExecutionOutcome {
    pub call: ToolCall,
    pub result: Result<ToolResult, ToolError>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// Backend used to execute a configured hook handler.
pub enum HookHandlerType {
    Command,
    Http,
    Prompt,
    Agent,
    Callback,
    Function,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// Lifecycle state of one hook run.
pub enum HookRunStatus {
    Running,
    Completed,
    Failed,
    Blocked,
    Stopped,
    Cancelled,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// Category assigned to a hook output entry.
pub enum HookOutputKind {
    Warning,
    Stop,
    Feedback,
    Context,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Human-readable hook output shown on a run summary.
pub struct HookOutputEntry {
    pub kind: HookOutputKind,
    pub text: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// Source reason passed to session-start hooks.
pub enum HookSessionStartSource {
    Startup,
    Resume,
    Clear,
    Compact,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Public record of one hook handler execution.
pub struct HookRunSummary {
    pub run_id: String,
    pub event_name: String,
    pub handler_type: HookHandlerType,
    pub plugin_id: PluginId,
    pub plugin_root: PathBuf,
    pub status: HookRunStatus,
    pub status_message: Option<String>,
    pub started_at: Timestamp,
    pub completed_at: Option<Timestamp>,
    pub duration_ms: Option<u64>,
    pub entries: Vec<HookOutputEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
/// Event payload emitted by the session runtime.
pub enum SessionEventPayload {
    SessionStarted,
    Warning {
        message: String,
    },
    TurnStarted {
        turn_id: TurnId,
    },
    MessageItem {
        message: Message,
    },
    DeltaItem {
        delta: DeltaItem,
    },
    ToolExecutionStarted {
        call: ToolCall,
    },
    ToolOutput {
        call_id: ToolCallId,
        tool_name: ToolName,
        chunk: SharedStr,
    },
    HookStarted {
        run: HookRunSummary,
    },
    HookCompleted {
        run: HookRunSummary,
    },
    ToolExecutionCompleted {
        outcome: ToolExecutionOutcome,
    },
    ApprovalRequested {
        tool_name: ToolName,
        reason: String,
    },
    ContextCompacted {
        summary: String,
    },
    TurnCompleted {
        turn_id: TurnId,
        usage: Usage,
    },
    TurnFailed {
        turn_id: TurnId,
        error: String,
        /// Whether the failure came from explicit user/runtime cancellation.
        #[serde(default)]
        cancelled: bool,
        /// Whether the underlying provider error advertised itself as
        /// retryable. Defaults to `false` so historical replays without this
        /// field deserialize cleanly.
        #[serde(default)]
        retryable: bool,
    },
    Lagged {
        dropped_events: u64,
    },
    SessionShutdownComplete,
}

/// An event that has been committed to the session store and therefore has
/// been assigned a monotonic `sequence` by the commit boundary. Construct a
/// `SessionEvent` only via `PendingEvent::into_committed`, `SessionEvent::new_committed`,
/// or deserialization — the `sequence` field is intentionally not publicly
/// settable.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct SessionEvent {
    pub session_id: SessionId,
    pub(crate) sequence: u64,
    pub delivery: Delivery,
    pub payload: SessionEventPayload,
}

impl SessionEvent {
    /// Construct a committed event with an explicit sequence. This is the
    /// only public constructor that sets the `sequence` field; call sites
    /// outside commit boundaries must use `PendingEvent`.
    #[must_use]
    pub fn new_committed(
        session_id: SessionId,
        sequence: u64,
        delivery: Delivery,
        payload: SessionEventPayload,
    ) -> Self {
        Self {
            session_id,
            sequence,
            delivery,
            payload,
        }
    }

    /// Monotonic sequence assigned by the session store.
    #[must_use]
    pub fn sequence(&self) -> u64 {
        self.sequence
    }
}

/// An event produced during turn execution, before the session store has
/// assigned a sequence. Convert to `SessionEvent` via `into_committed` once
/// the store has allocated a sequence number. Holding `sequence`-less events
/// until commit makes the commit-then-publish invariant type-enforced.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PendingEvent {
    pub session_id: SessionId,
    pub delivery: Delivery,
    pub payload: SessionEventPayload,
}

impl PendingEvent {
    /// Build an uncommitted event. The session store assigns its sequence.
    #[must_use]
    pub fn new(session_id: SessionId, delivery: Delivery, payload: SessionEventPayload) -> Self {
        Self {
            session_id,
            delivery,
            payload,
        }
    }

    /// Attach the commit sequence and convert into a committed event.
    #[must_use]
    pub fn into_committed(self, sequence: u64) -> SessionEvent {
        SessionEvent {
            session_id: self.session_id,
            sequence,
            delivery: self.delivery,
            payload: self.payload,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Scheduler hint for how tool calls may be batched.
pub enum ToolConcurrency {
    /// Run alone and preserve strict ordering.
    Exclusive,
    /// Can run with other non-mutating tools.
    ReadOnly,
    /// Can run concurrently with any other parallel-safe tool.
    ParallelSafe,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Capabilities exposed by a tool specification.
pub struct ToolCapabilities {
    pub mutating: bool,
    pub requires_approval: bool,
    pub cancellable: bool,
    pub long_running: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Provider-visible tool declaration.
pub struct ToolSpec {
    pub name: ToolName,
    pub description: SharedStr,
    pub input_schema: Value,
    pub concurrency: ToolConcurrency,
    pub capabilities: ToolCapabilities,
    pub provider_aliases: IndexMap<ProviderKind, ToolAlias>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
/// Output returned by a tool execution.
pub enum ToolResult {
    /// No content.
    Empty,
    /// Plain text content.
    Text { text: String },
    /// Structured JSON content.
    Json { value: Value },
}

#[derive(Debug, Clone, Error, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[error("{message}")]
/// Error returned by a tool execution.
pub struct ToolError {
    pub message: String,
}

impl ToolError {
    /// Build a tool error from a displayable message.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
/// Provider error taxonomy used by retry policy and consumers.
pub enum ProviderErrorKind {
    /// 5xx, overload, transport, stream, timeout, and otherwise unknown
    /// provider failures that may succeed on a later attempt.
    Transient,
    /// Provider rate limiting. `ProviderError::backoff_hint` may carry a
    /// server-supplied retry-after value.
    RateLimited,
    /// Explicitly non-recoverable failures such as auth, malformed requests,
    /// unknown models, context-window errors, or policy refusals.
    #[default]
    Fatal,
    /// Runtime or user cancellation.
    Cancelled,
}

impl ProviderErrorKind {
    #[must_use]
    pub const fn retryable(self) -> bool {
        matches!(self, Self::Transient | Self::RateLimited)
    }
}

#[derive(Debug, Clone, Error, Serialize, JsonSchema, PartialEq, Eq)]
#[error("{message}")]
/// Error surfaced by a provider adapter.
pub struct ProviderError {
    pub message: String,
    #[serde(default)]
    pub kind: ProviderErrorKind,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backoff_hint: Option<Duration>,
}

impl ProviderError {
    /// Sentinel message produced by `ProviderError::cancelled` and recognized
    /// by `is_cancelled`. New consumers should prefer the constructor /
    /// predicate over inline message comparison.
    pub const CANCELLED_MESSAGE: &str = "failed to execute provider request: request cancelled";

    /// Build a provider error with an explicit retryability flag.
    ///
    /// Prefer [`ProviderError::with_kind`] in new code so the error's nature is
    /// preserved after a local retry budget is exhausted.
    #[must_use]
    pub fn new(message: impl Into<String>, retryable: bool) -> Self {
        let kind = if retryable {
            ProviderErrorKind::Transient
        } else {
            ProviderErrorKind::Fatal
        };
        Self::with_kind(message, kind)
    }

    /// Build a provider error with a typed classification and no backoff hint.
    #[must_use]
    pub fn with_kind(message: impl Into<String>, kind: ProviderErrorKind) -> Self {
        Self {
            message: message.into(),
            kind,
            backoff_hint: None,
        }
    }

    /// Attach a server-supplied or inferred backoff hint.
    #[must_use]
    pub fn with_backoff_hint(mut self, backoff_hint: Option<Duration>) -> Self {
        self.backoff_hint = backoff_hint;
        self
    }

    /// Whether this error represents a retryable provider condition.
    #[must_use]
    pub const fn retryable(&self) -> bool {
        self.kind.retryable()
    }

    /// Construct a non-retryable cancellation error with the canonical
    /// message. Existing consumers that match on message text continue to
    /// work; new consumers should use `is_cancelled()` to distinguish.
    #[must_use]
    pub fn cancelled() -> Self {
        Self {
            message: Self::CANCELLED_MESSAGE.to_owned(),
            kind: ProviderErrorKind::Cancelled,
            backoff_hint: None,
        }
    }

    /// Whether this error is the canonical cancellation sentinel.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.kind == ProviderErrorKind::Cancelled || self.message == Self::CANCELLED_MESSAGE
    }
}

impl<'de> Deserialize<'de> for ProviderError {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct ProviderErrorWire {
            message: String,
            #[serde(default)]
            kind: Option<ProviderErrorKind>,
            #[serde(default)]
            backoff_hint: Option<Duration>,
            #[serde(default)]
            retryable: Option<bool>,
        }

        let wire = ProviderErrorWire::deserialize(deserializer)?;
        let kind = wire.kind.unwrap_or_else(|| {
            if wire.message == ProviderError::CANCELLED_MESSAGE {
                ProviderErrorKind::Cancelled
            } else if wire.retryable.unwrap_or(false) {
                ProviderErrorKind::Transient
            } else {
                ProviderErrorKind::Fatal
            }
        });
        Ok(Self {
            message: wire.message,
            kind,
            backoff_hint: wire.backoff_hint,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Prompt fragment assembled before provider encoding.
pub struct PromptSegment {
    pub id: PromptSegmentId,
    pub text: SharedStr,
    pub volatility: Volatility,
    pub cache_scope: CacheScope,
    pub content_hash: ContentHash,
    /// Logical section the segment belongs to. The prompt assembler groups
    /// segments by kind so the wire layout (system, then skills, then the
    /// turn) is independent of insertion order, and so codecs can emit
    /// cache breakpoints on stable boundaries.
    #[serde(default)]
    pub kind: PromptSegmentKind,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
/// Logical prompt section for cache breakpoint placement.
pub enum PromptSegmentKind {
    /// System prompt section.
    #[default]
    System,
    /// Loaded skill section.
    Skill,
    /// Runtime-appended context section.
    Append,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// How often a prompt segment is expected to change.
pub enum Volatility {
    /// Stable across all sessions for a given build/config.
    Static,
    /// Stable for the lifetime of a session.
    SessionStable,
    /// May change every turn.
    TurnDynamic,
    /// Always treated as dynamic.
    AlwaysDynamic,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Whether a segment can participate in prefix caching.
pub enum CacheScope {
    /// Eligible for provider prefix-cache placement.
    PrefixCacheable,
    /// Not eligible for prefix caching.
    Dynamic,
}

/// Marks the four section boundaries the runtime asks codecs to expose as
/// cache breakpoints when the underlying provider supports them.
///
/// The order is fixed: system prompt, tool descriptions, skills, then the
/// most recent user prompt. The "rest of the session" follows the last
/// breakpoint and is therefore the only window eligible for in-band
/// compaction by non-dedicated providers.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
pub struct CacheBreakpoints {
    pub after_system: bool,
    pub after_tools: bool,
    pub after_skills: bool,
    pub after_user_prompt: bool,
}

impl CacheBreakpoints {
    /// All four breakpoints active. The prompt assembler emits this layout
    /// for any session that has a non-empty system prompt and at least one
    /// user message; codecs may downgrade as needed.
    #[must_use]
    pub fn all() -> Self {
        Self {
            after_system: true,
            after_tools: true,
            after_skills: true,
            after_user_prompt: true,
        }
    }

    /// Number of active breakpoints.
    #[must_use]
    pub fn count_active(&self) -> usize {
        usize::from(self.after_system)
            + usize::from(self.after_tools)
            + usize::from(self.after_skills)
            + usize::from(self.after_user_prompt)
    }
}

/// Per-session cache of file ranges already shown to the model.
pub type FileViewCache = IndexMap<PathBuf, FileViewEntry>;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Cached metadata for a file view.
pub struct FileViewEntry {
    pub path: PathBuf,
    pub full_hash: ContentHash,
    pub mtime: Timestamp,
    pub size: u64,
    pub viewed_ranges: Vec<ViewedRange>,
    pub last_shown_turn: TurnId,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Inclusive range of file lines previously shown to the model.
pub struct ViewedRange {
    pub start_line: u32,
    pub end_line: u32,
    pub line_anchors: Vec<LineAnchor>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Small content anchor used to detect shifted viewed ranges.
pub struct LineAnchor {
    pub line: u32,
    pub anchor: [u8; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Tool call that has been emitted but not yet answered.
pub struct PendingToolCall {
    pub call: ToolCall,
    pub submitted_at: Timestamp,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Summary retained after older context has been compacted.
pub struct SummarySlice {
    pub id: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
/// Active transcript window after pruning and compaction planning.
pub struct TranscriptWindow {
    pub messages: Vec<Message>,
    pub elided_message_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
#[serde(transparent)]
/// Provider-native compacted context items carried across turns.
pub struct CompactedContext(pub Vec<Value>);

impl CompactedContext {
    /// Wrap provider-native compacted context items.
    #[must_use]
    pub fn new(items: Vec<Value>) -> Self {
        Self(items)
    }

    /// Borrow the compacted context items.
    #[must_use]
    pub fn items(&self) -> &[Value] {
        &self.0
    }

    /// Consume the wrapper and return the raw items.
    #[must_use]
    pub fn into_items(self) -> Vec<Value> {
        self.0
    }

    /// Whether there are no compacted context items.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Number of compacted context items.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl From<Vec<Value>> for CompactedContext {
    fn from(value: Vec<Value>) -> Self {
        Self(value)
    }
}

impl AsRef<[Value]> for CompactedContext {
    fn as_ref(&self) -> &[Value] {
        self.items()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
/// Split of messages selected for provider compaction.
pub struct CompactionWindow {
    pub eligible_messages: Vec<Message>,
    pub preserved_messages: Vec<Message>,
    pub reserved_response_block: bool,
}

impl CompactionWindow {
    /// Preserve the latest assistant response block and compact the older
    /// prefix. Providers with a first-class compaction endpoint use this
    /// broader window because the provider restores the compacted context as
    /// provider-native content.
    #[must_use]
    pub fn preserve_latest_assistant_response_block(messages: &[Message]) -> Self {
        let Some(last_assistant_index) = messages
            .iter()
            .rposition(|message| matches!(message, Message::Assistant(_)))
        else {
            return Self {
                eligible_messages: messages.to_vec(),
                preserved_messages: Vec::new(),
                reserved_response_block: false,
            };
        };

        Self {
            eligible_messages: messages[..last_assistant_index].to_vec(),
            preserved_messages: messages[last_assistant_index..].to_vec(),
            reserved_response_block: true,
        }
    }

    /// Preserve every message through the latest user message and compact
    /// only the post-user tail. Inline compaction providers use this narrower
    /// window so system, tool, skill, and latest-user cache anchors remain
    /// verbatim.
    #[must_use]
    pub fn preserve_through_latest_user(messages: &[Message]) -> Self {
        let Some(last_user_index) = messages
            .iter()
            .rposition(|message| matches!(message, Message::User(_)))
        else {
            return Self {
                eligible_messages: Vec::new(),
                preserved_messages: messages.to_vec(),
                reserved_response_block: false,
            };
        };
        let pivot = last_user_index + 1;
        Self {
            eligible_messages: messages[pivot..].to_vec(),
            preserved_messages: messages[..pivot].to_vec(),
            reserved_response_block: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// File-view data included in a context plan.
pub struct FileViewSlice {
    pub path: PathBuf,
    pub full_hash: ContentHash,
    pub viewed_ranges: Vec<ViewedRange>,
    pub last_shown_turn: TurnId,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
/// Marker describing content omitted from the active context.
pub struct ElisionMarker {
    pub kind: String,
    pub count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
/// Long-lived memory item available to context planning.
pub struct MemoryItem {
    pub key: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Link from a parent session to a spawned subagent session.
pub struct SubagentRef {
    pub session_id: SessionId,
    pub task: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Immutable session metadata created at session start.
pub struct SessionBlueprint {
    pub session_id: SessionId,
    pub parent_session_id: Option<SessionId>,
    pub default_model: ModelId,
    pub subagent_model: ModelId,
    #[serde(default)]
    pub subagent_event_forwarding: SubagentEventForwarding,
    pub snapshot_revision: Revision,
    pub working_dir: PathBuf,
    pub system_prompt_seed: Vec<PromptSegment>,
    pub max_turns: Option<u32>,
    pub subagent_depth: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
/// Mutable state persisted for a session.
pub struct SessionState {
    pub messages: Vec<Message>,
    #[serde(default)]
    pub compacted_prefix: Vec<Value>,
    pub file_view_cache: FileViewCache,
    pub appended_prompt_segments: Vec<PromptSegment>,
    pub pending_tool_calls: IndexMap<ToolCallId, PendingToolCall>,
    pub usage_so_far: Usage,
    pub summaries: Vec<SummarySlice>,
    pub lineage: Vec<SubagentRef>,
    pub fired_hook_ids: Vec<String>,
    pub pending_session_start_source: Option<HookSessionStartSource>,
    pub pending_warning_messages: Vec<HookWarning>,
    /// The OpenAI Responses API response ID from the last successful turn.
    /// Used for `previous_response_id` chaining to avoid re-sending full history.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_response_id: Option<String>,
    /// Number of messages the model has already seen via `previous_response_id`.
    /// Messages at indices `[0..messages_seen_by_provider)` don't need re-sending.
    #[serde(default)]
    pub messages_seen_by_provider: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Environment facts captured while building a context plan.
pub struct ObservedState {
    pub cwd: PathBuf,
    pub git_branch: Option<String>,
    pub git_dirty: Option<bool>,
    pub now_utc: Timestamp,
    pub env_facts: IndexMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
/// Instruction file loaded into a resource snapshot.
pub struct InstructionFile {
    pub path: PathBuf,
    pub body: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
/// Loaded skill definition made available to prompt assembly.
pub struct SkillDef {
    pub id: SkillId,
    pub name: String,
    pub description: String,
    pub body: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
/// Loaded agent definition used by subagent tools.
pub struct AgentDef {
    pub id: AgentId,
    pub name: String,
    pub prompt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
/// Manifest data loaded from a plugin root.
pub struct PluginManifest {
    pub name: String,
    pub version: String,
    pub skills: Vec<String>,
    pub agents: Vec<String>,
    pub hooks: Option<String>,
    pub mcp_servers: Option<String>,
    pub lsp_servers: Option<String>,
    pub allowed_http_hosts: Vec<String>,
    pub allowed_env_vars: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
/// Named prompt segments loaded from resources.
pub struct PromptRegistry {
    pub prompts: IndexMap<String, Vec<PromptSegment>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Complete resource set visible to a session runtime.
pub struct ResourceSnapshot {
    pub revision: Revision,
    pub tools: IndexMap<ToolName, ToolSpec>,
    pub skills: IndexMap<SkillName, SkillDef>,
    pub agents: IndexMap<AgentName, AgentDef>,
    pub prompts: PromptRegistry,
    pub plugins: IndexMap<PluginId, PluginManifest>,
    pub instruction_files: Vec<InstructionFile>,
}

impl ResourceSnapshot {
    /// Build an empty snapshot for tests and custom SDK assembly.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            revision: Revision("empty".to_owned()),
            tools: IndexMap::new(),
            skills: IndexMap::new(),
            agents: IndexMap::new(),
            prompts: PromptRegistry::default(),
            plugins: IndexMap::new(),
            instruction_files: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Capability flags advertised by a provider adapter.
pub struct ProviderCapabilities {
    pub supports_tools: bool,
    pub supports_streaming: bool,
    pub supports_reasoning: bool,
    pub supports_interleaved_reasoning: bool,
    pub supports_images: bool,
    pub supports_documents: bool,
    pub supports_prompt_cache: bool,
    pub supports_compaction: bool,
    /// How the provider implements compaction. This remains exposed for
    /// diagnostics and external callers, but runtime planning asks the
    /// provider for a `CompactionWindow` instead of branching on this value.
    #[serde(default)]
    pub compaction_strategy: Option<ProviderCompactionStrategy>,
    pub supports_tool_result_media: bool,
    pub requires_non_empty_assistant_content: bool,
    pub tool_call_id_policy: ToolCallIdPolicy,
    pub max_input_tokens: u64,
    pub max_output_tokens: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProviderCompactionStrategy {
    /// A first-class compaction endpoint (e.g. OpenAI Responses
    /// `/v1/responses/compact`) that returns encrypted content for safe
    /// reinjection. The runtime can compact aggressively because the
    /// provider preserves anchor invariants.
    Dedicated,
    /// In-band compaction via the regular completions endpoint
    /// (e.g. OpenRouter's responses passthrough). Lossy: the runtime
    /// only compacts the trailing window after the last cache breakpoint
    /// and wraps the result in explicit compaction tags so the model can
    /// distinguish it from authoritative system content.
    Inline,
}

impl Default for ProviderCapabilities {
    fn default() -> Self {
        Self {
            supports_tools: true,
            supports_streaming: true,
            supports_reasoning: false,
            supports_interleaved_reasoning: false,
            supports_images: false,
            supports_documents: false,
            supports_prompt_cache: false,
            supports_compaction: false,
            compaction_strategy: None,
            supports_tool_result_media: false,
            requires_non_empty_assistant_content: false,
            tool_call_id_policy: ToolCallIdPolicy::ProviderSupplied,
            max_input_tokens: 0,
            max_output_tokens: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// How provider tool-call ids are assigned and normalized.
pub enum ToolCallIdPolicy {
    /// Provider supplies ids and they can be used directly.
    ProviderSupplied,
    /// Runtime must synthesize ids when the provider omits them.
    RuntimeSynthesized,
    /// Runtime normalizes ids for stable replay across providers.
    StableReplayNormalized,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Fully resolved model and provider configuration.
pub struct ResolvedModel {
    pub role: ModelRole,
    pub id: ModelId,
    pub provider: ProviderName,
    pub provider_kind: ProviderKind,
    pub api_kind: ApiKind,
    pub model: String,
    pub max_input_tokens: Option<u32>,
    pub max_output_tokens: Option<u32>,
    pub reasoning: Option<ReasoningEffort>,
    #[serde(default)]
    pub tokens_per_minute: Option<u64>,
}

#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq, PartialOrd, Ord,
)]
#[serde(rename_all = "snake_case")]
/// Relative value of a message when pruning context before compaction.
pub enum MessageSignal {
    /// Compact first -- orientation commands, empty results, duplicate failures.
    VeryLow = 0,
    /// Low signal -- failed tool calls, stale reads.
    Low = 1,
    /// Default for most messages.
    Normal = 2,
    /// Active file reads and system guidance.
    High = 3,
    /// Assistant text or reasoning content.
    VeryHigh = 4,
    /// Never compact -- user messages.
    Anchor = 5,
}

#[derive(
    Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq, PartialOrd, Ord,
)]
#[serde(rename_all = "snake_case")]
/// Highest message-signal tier eligible for pre-compaction pruning.
pub enum PruneSignalThreshold {
    VeryLow,
    Low,
    #[default]
    Normal,
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Result of applying a compaction output to session state.
pub struct CompactionResult {
    /// Number of messages compacted into the raw prefix.
    pub compacted_count: usize,
    /// Human-readable summary for events and hooks.
    pub summary: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Context manager output consumed by prompt assembly and provider codecs.
pub struct ContextPlan {
    pub prompt_segments: Vec<PromptSegment>,
    pub transcript_window: TranscriptWindow,
    #[serde(default)]
    pub compacted_prefix: Vec<Value>,
    pub file_views: Vec<FileViewSlice>,
    pub carried_summaries: Vec<SummarySlice>,
    pub elided_tool_results: Vec<ElisionMarker>,
    pub memory_items: Vec<MemoryItem>,
    pub tool_specs: Vec<ToolSpec>,
    pub observed_state: ObservedState,
    pub projected_input_tokens: u64,
    pub cache_boundary_hash: ContentHash,
    pub messages: Vec<Message>,
    pub estimated_tokens: u64,
    /// If the planner compacted messages this turn, the result is here.
    /// The caller should apply it to `SessionState` after using the plan.
    pub compaction: Option<CompactionResult>,
    /// When set, the codec should chain via `previous_response_id`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    /// Index into `messages` where new messages start (for chained requests).
    #[serde(default)]
    pub new_messages_start: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Fully assembled prompt and transcript material ready for provider encoding.
pub struct AssembledPrompt {
    pub segments: Vec<PromptSegment>,
    pub transcript: Vec<Message>,
    pub ordered_segments: Vec<PromptSegment>,
    pub prefix_cache_key: String,
    pub rendered_prefix: String,
    pub rendered_transcript: String,
    pub rendered: String,
    /// Section boundaries that the assembler asks the codec to expose as
    /// cache breakpoints. Codecs that do not support explicit breakpoints
    /// (e.g. OpenAI Responses, which uses prefix-prefix caching) ignore
    /// this; codecs that do (Anthropic) emit `cache_control` on the last
    /// content block of each marked section.
    #[serde(default)]
    pub cache_breakpoints: CacheBreakpoints,
    /// Index into `ordered_segments` after which the system-prompt
    /// breakpoint applies. `None` when there are no system segments.
    #[serde(default)]
    pub system_segment_count: usize,
    /// Number of segments at the head of `ordered_segments` that belong
    /// to the skills section. Always immediately follows the system block.
    #[serde(default)]
    pub skill_segment_count: usize,
}

impl AssembledPrompt {
    /// Slice of segments that constitute the system-prompt section.
    #[must_use]
    pub fn system_segments(&self) -> &[PromptSegment] {
        let end = self.system_segment_count.min(self.ordered_segments.len());
        &self.ordered_segments[..end]
    }

    /// Slice of segments that constitute the skills section.
    #[must_use]
    pub fn skill_segments(&self) -> &[PromptSegment] {
        let start = self.system_segment_count.min(self.ordered_segments.len());
        let end = (start + self.skill_segment_count).min(self.ordered_segments.len());
        &self.ordered_segments[start..end]
    }

    /// Slice of segments that follow both the system and skills sections —
    /// hook-appended context, etc. These never receive a cache breakpoint
    /// because they may change turn-to-turn.
    #[must_use]
    pub fn append_segments(&self) -> &[PromptSegment] {
        let start =
            (self.system_segment_count + self.skill_segment_count).min(self.ordered_segments.len());
        &self.ordered_segments[start..]
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Request sent from the runtime to a provider for normal generation.
pub struct ProviderRequest {
    pub session_id: SessionId,
    pub turn_id: TurnId,
    pub model: ResolvedModel,
    pub prompt: AssembledPrompt,
    #[serde(default)]
    pub compacted_prefix: Vec<Value>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSpec>,
    /// When set, the provider can chain onto the previous response instead of
    /// re-sending the full conversation history. The codec should send only
    /// messages after `new_messages_start` when this is present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub previous_response_id: Option<String>,
    /// Index into `messages` where new (unseen-by-provider) messages begin.
    /// Only meaningful when `previous_response_id` is `Some`.
    #[serde(default)]
    pub new_messages_start: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Request sent to a provider for context compaction.
pub struct ProviderCompactionRequest {
    pub session_id: SessionId,
    pub model: ResolvedModel,
    #[serde(default)]
    pub compacted_prefix: Vec<Value>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSpec>,
    pub instructions: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Provider response containing compacted context items.
pub struct ProviderCompactionResponse {
    pub output: Vec<Value>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
/// Final output returned from a completed subagent.
pub struct SubagentResult {
    pub session_id: SessionId,
    pub output: String,
    pub usage: Usage,
}

#[cfg(test)]
mod tests {
    use bytes::Bytes;

    use super::*;

    #[test]
    fn message_roundtrip() {
        let message = Message::User(UserMessage::text("hello"));
        let encoded = serde_json::to_string(&message).expect("serialize message");
        let decoded: Message = serde_json::from_str(&encoded).expect("deserialize message");
        assert_eq!(decoded, message);
    }

    #[test]
    fn session_event_roundtrip() {
        let event = SessionEvent::new_committed(
            SessionId::new(),
            1,
            Delivery::Lossless,
            SessionEventPayload::TurnCompleted {
                turn_id: TurnId::new(),
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 0,
                },
            },
        );
        let encoded = serde_json::to_string(&event).expect("serialize event");
        let decoded: SessionEvent = serde_json::from_str(&encoded).expect("deserialize event");
        assert_eq!(decoded, event);
    }

    #[test]
    fn pending_event_into_committed_preserves_fields() {
        let session_id = SessionId::from("session-42");
        let payload = SessionEventPayload::ContextCompacted {
            summary: "summary".to_owned(),
        };
        let pending = PendingEvent::new(session_id.clone(), Delivery::Lossless, payload.clone());

        let committed = pending.clone().into_committed(7);

        assert_eq!(committed.session_id, session_id);
        assert_eq!(committed.sequence(), 7);
        assert_eq!(committed.delivery, Delivery::Lossless);
        assert_eq!(committed.payload, payload);

        // PendingEvent is still unsequenced; we reject post-hoc mutation of
        // committed events by keeping the sequence field crate-private.
        let encoded = serde_json::to_string(&pending).expect("serialize pending");
        assert!(!encoded.contains("sequence"));
    }

    #[test]
    fn turn_roundtrip_preserves_model_overrides() {
        let turn = Turn::user("hello")
            .with_default_model("default")
            .with_subagent_model("subagent");

        let encoded = serde_json::to_string(&turn).expect("serialize turn");
        let decoded: Turn = serde_json::from_str(&encoded).expect("deserialize turn");

        assert_eq!(decoded, turn);
    }

    #[test]
    fn compacted_context_serializes_as_existing_prefix_array() {
        let context = CompactedContext::new(vec![
            serde_json::json!({"type": "reasoning", "encrypted_content": "summary"}),
        ]);

        let encoded = serde_json::to_string(&context).expect("serialize compacted context");
        assert!(encoded.starts_with('['));

        let decoded: CompactedContext =
            serde_json::from_str(&encoded).expect("deserialize compacted context");
        assert_eq!(decoded, context);

        let state: SessionState = serde_json::from_value(serde_json::json!({
            "messages": [],
            "compacted_prefix": [
                {"type": "reasoning", "encrypted_content": "summary"}
            ],
            "file_view_cache": {},
            "appended_prompt_segments": [],
            "pending_tool_calls": {},
            "usage_so_far": {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0
            },
            "summaries": [],
            "lineage": [],
            "fired_hook_ids": [],
            "pending_session_start_source": null,
            "pending_warning_messages": [],
            "messages_seen_by_provider": 0
        }))
        .expect("deserialize existing session state");
        assert_eq!(state.compacted_prefix.len(), 1);
    }

    #[test]
    fn compaction_window_preserves_latest_assistant_response_block() {
        let messages = vec![
            Message::User(UserMessage::text("first")),
            assistant_text("answer"),
            Message::User(UserMessage::text("follow up")),
        ];

        let window = CompactionWindow::preserve_latest_assistant_response_block(&messages);

        assert_eq!(window.eligible_messages.len(), 1);
        assert_eq!(window.preserved_messages.len(), 2);
        assert!(window.reserved_response_block);
    }

    #[test]
    fn compaction_window_preserves_through_latest_user() {
        let messages = vec![
            Message::User(UserMessage::text("first")),
            assistant_text("answer"),
            Message::User(UserMessage::text("follow up")),
            assistant_text("tail"),
        ];

        let window = CompactionWindow::preserve_through_latest_user(&messages);

        assert_eq!(window.preserved_messages.len(), 3);
        assert!(matches!(
            window.preserved_messages.last(),
            Some(Message::User(_))
        ));
        assert_eq!(window.eligible_messages.len(), 1);
        assert!(!window.reserved_response_block);
    }

    #[test]
    fn user_message_with_media_roundtrips() {
        let message = Message::User(UserMessage {
            id: MessageId::new(),
            created_at: Utc::now(),
            parts: vec![
                UserPart::Text {
                    text: "hello".to_owned(),
                },
                UserPart::Image {
                    media_type: "image/png".to_owned(),
                    data: Bytes::from_static(b"png"),
                },
                UserPart::Document {
                    media_type: "application/pdf".to_owned(),
                    data: Bytes::from_static(b"pdf"),
                },
            ],
        });

        let encoded = serde_json::to_string(&message).expect("serialize message");
        let decoded: Message = serde_json::from_str(&encoded).expect("deserialize message");
        assert_eq!(decoded, message);
    }

    #[test]
    fn stream_event_with_signature_roundtrips() {
        let event = StreamEvent::ThinkingEnd {
            id: BlockId::new(),
            signature: Some("sig-123".to_owned()),
        };

        let encoded = serde_json::to_string(&event).expect("serialize event");
        let decoded: StreamEvent = serde_json::from_str(&encoded).expect("deserialize event");
        assert_eq!(decoded, event);
    }

    fn assistant_text(text: &str) -> Message {
        Message::Assistant(AssistantMessage {
            id: MessageId::new(),
            created_at: Utc::now(),
            parts: vec![AssistantPart::Text {
                text: text.to_owned(),
            }],
            stop_reason: None,
            usage: None,
            replay_meta: ReplayMeta::default(),
        })
    }
}
