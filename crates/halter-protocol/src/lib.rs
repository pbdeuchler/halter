// pattern: Functional Core

use std::fmt;
use std::path::PathBuf;

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
pub type MediaType = String;
pub type ReplaySignature = String;
pub type ContentHash = String;
pub type Timestamp = DateTime<Utc>;

macro_rules! id_type {
    ($name:ident) => {
        #[derive(
            Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
        )]
        pub struct $name(pub String);

        impl $name {
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
    Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, JsonSchema,
)]
pub struct ModelRole(pub String);

impl ModelRole {
    #[must_use]
    pub fn default_role() -> Self {
        Self("default".to_owned())
    }

    #[must_use]
    pub fn plan() -> Self {
        Self("plan".to_owned())
    }

    #[must_use]
    pub fn subagent() -> Self {
        Self("subagent".to_owned())
    }

    #[must_use]
    pub fn small() -> Self {
        Self("small".to_owned())
    }
}

impl Default for ModelRole {
    fn default() -> Self {
        Self::default_role()
    }
}

impl From<&str> for ModelRole {
    fn from(value: &str) -> Self {
        Self(value.to_owned())
    }
}

impl fmt::Display for ModelRole {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Anthropic,
    OpenAi,
    OpenRouter,
    Fake,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ApiKind {
    AnthropicMessages,
    OpenAiResponses,
    OpenAiChat,
    Fake,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReasoningEffort {
    Low,
    Medium,
    High,
    Xhigh,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
pub struct Usage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub cache_read_input_tokens: u64,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StopReason {
    EndTurn,
    ToolUse,
    Interrupted,
    MaxTokens,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
pub struct ReplayMeta {
    pub provider_name: Option<ProviderName>,
    pub model: Option<ModelId>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum HookWarningSeverity {
    #[default]
    Warning,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
pub struct HookWarning {
    pub severity: HookWarningSeverity,
    pub category: SharedStr,
    pub plugin_id: Option<PluginId>,
    pub plugin_name: Option<SharedStr>,
    pub source_path: Option<PathBuf>,
    pub message: SharedStr,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct SystemMessage {
    pub id: MessageId,
    pub created_at: Timestamp,
    pub text: SharedStr,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct UserMessage {
    pub id: MessageId,
    pub created_at: Timestamp,
    pub parts: Vec<UserPart>,
}

impl UserMessage {
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            id: MessageId::new(),
            created_at: Utc::now(),
            parts: vec![UserPart::Text { text: text.into() }],
        }
    }

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
pub enum UserPart {
    Text {
        text: SharedStr,
    },
    Image {
        media_type: MediaType,
        #[schemars(with = "Vec<u8>")]
        data: Bytes,
    },
    Document {
        media_type: MediaType,
        #[schemars(with = "Vec<u8>")]
        data: Bytes,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
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
pub enum AssistantPart {
    Text { text: SharedStr },
    Thinking(ThinkingBlock),
    ToolCall(ToolCall),
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ThinkingBlock {
    pub text: SharedStr,
    pub signature: Option<ReplaySignature>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ToolCall {
    pub id: ToolCallId,
    pub name: ToolName,
    pub arguments: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ToolResultMessage {
    pub id: MessageId,
    pub call_id: ToolCallId,
    pub content: ToolResult,
    pub error: Option<ToolError>,
    pub created_at: Timestamp,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(tag = "role", rename_all = "snake_case")]
pub enum Message {
    System(SystemMessage),
    User(UserMessage),
    Assistant(AssistantMessage),
    Tool(ToolResultMessage),
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StreamEvent {
    MessageStart {
        id: MessageId,
    },
    TextStart {
        id: BlockId,
    },
    TextDelta {
        id: BlockId,
        delta: SharedStr,
    },
    TextEnd {
        id: BlockId,
    },
    ThinkingStart {
        id: BlockId,
    },
    ThinkingDelta {
        id: BlockId,
        delta: SharedStr,
    },
    ThinkingEnd {
        id: BlockId,
        signature: Option<ReplaySignature>,
    },
    ToolCallStart {
        id: BlockId,
        tool_call_id: ToolCallId,
        name: ToolName,
    },
    ToolArgsDelta {
        id: BlockId,
        delta: SharedStr,
    },
    ToolCallEnd {
        id: BlockId,
    },
    UsageUpdate {
        usage: Usage,
    },
    MessageEnd {
        id: MessageId,
        stop_reason: StopReason,
    },
    ProviderWarning {
        message: SharedStr,
    },
    Error {
        error: ProviderError,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct Turn {
    pub id: TurnId,
    pub user_message: UserMessage,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub default_model: Option<ModelId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subagent_model: Option<ModelId>,
}

impl Turn {
    #[must_use]
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            id: TurnId::new(),
            user_message: UserMessage::text(text),
            default_model: None,
            subagent_model: None,
        }
    }

    #[must_use]
    pub fn with_default_model(mut self, model: impl Into<ModelId>) -> Self {
        self.default_model = Some(model.into());
        self
    }

    #[must_use]
    pub fn with_subagent_model(mut self, model: impl Into<ModelId>) -> Self {
        self.subagent_model = Some(model.into());
        self
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SubagentState {
    Running,
    Completed,
    Failed,
    Cancelled,
    Closed,
}

impl SubagentState {
    #[must_use]
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::Failed | Self::Cancelled | Self::Closed
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct SpawnSubagentRequest {
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_type: Option<AgentName>,
    #[serde(default = "default_true")]
    pub fork_context: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<ModelId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct SendSubagentInputRequest {
    pub target: AgentId,
    pub message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct WaitSubagentRequest {
    pub targets: Vec<AgentId>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct CloseSubagentRequest {
    pub target: AgentId,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
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
    #[must_use]
    pub fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct WaitSubagentResponse {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub status: Option<SubagentStatus>,
    pub timed_out: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct CloseSubagentResponse {
    pub previous_status: SubagentStatus,
}

const fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct SubagentSpecWire {
    pub role: Option<ModelRole>,
    pub task: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SessionCommand {
    SubmitTurn { turn: Turn },
    InterruptTurn,
    AppendSystemPrompt { id: PromptId, text: SharedStr },
    SetModelRole { role: ModelRole },
    SetModel { model: ModelId },
    SpawnSubagent { spec: SubagentSpecWire },
    ReloadResources,
    Shutdown,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub enum Delivery {
    Lossless,
    BestEffort,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct DeltaItem {
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ToolExecutionOutcome {
    pub call: ToolCall,
    pub result: Result<ToolResult, ToolError>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
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
pub enum HookOutputKind {
    Warning,
    Stop,
    Feedback,
    Context,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct HookOutputEntry {
    pub kind: HookOutputKind,
    pub text: String,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum HookSessionStartSource {
    Startup,
    Resume,
    Clear,
    Compact,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
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
    },
    Lagged {
        dropped_events: u64,
    },
    SessionShutdownComplete,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct SessionEvent {
    pub session_id: SessionId,
    pub sequence: u64,
    pub delivery: Delivery,
    pub payload: SessionEventPayload,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub enum ToolConcurrency {
    Exclusive,
    ReadOnly,
    ParallelSafe,
}

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ToolCapabilities {
    pub mutating: bool,
    pub requires_approval: bool,
    pub cancellable: bool,
    pub long_running: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
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
pub enum ToolResult {
    Empty,
    Text { text: String },
    Json { value: Value },
}

#[derive(Debug, Clone, Error, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[error("{message}")]
pub struct ToolError {
    pub message: String,
}

impl ToolError {
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, Error, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[error("{message}")]
pub struct ProviderError {
    pub message: String,
    pub retryable: bool,
}

impl ProviderError {
    #[must_use]
    pub fn new(message: impl Into<String>, retryable: bool) -> Self {
        Self {
            message: message.into(),
            retryable,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PromptSegment {
    pub id: PromptSegmentId,
    pub text: SharedStr,
    pub volatility: Volatility,
    pub cache_scope: CacheScope,
    pub content_hash: ContentHash,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub enum Volatility {
    Static,
    SessionStable,
    TurnDynamic,
    AlwaysDynamic,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub enum CacheScope {
    PrefixCacheable,
    Dynamic,
}

pub type FileViewCache = IndexMap<PathBuf, FileViewEntry>;

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct FileViewEntry {
    pub path: PathBuf,
    pub full_hash: ContentHash,
    pub mtime: Timestamp,
    pub size: u64,
    pub viewed_ranges: Vec<ViewedRange>,
    pub last_shown_turn: TurnId,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ViewedRange {
    pub start_line: u32,
    pub end_line: u32,
    pub line_anchors: Vec<LineAnchor>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct LineAnchor {
    pub line: u32,
    pub anchor: [u8; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct PendingToolCall {
    pub call: ToolCall,
    pub submitted_at: Timestamp,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct SummarySlice {
    pub id: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
pub struct TranscriptWindow {
    pub messages: Vec<Message>,
    pub elided_message_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct FileViewSlice {
    pub path: PathBuf,
    pub full_hash: ContentHash,
    pub viewed_ranges: Vec<ViewedRange>,
    pub last_shown_turn: TurnId,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
pub struct ElisionMarker {
    pub kind: String,
    pub count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
pub struct MemoryItem {
    pub key: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct SubagentRef {
    pub session_id: SessionId,
    pub task: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct SessionBlueprint {
    pub session_id: SessionId,
    pub parent_session_id: Option<SessionId>,
    pub default_model: ModelId,
    pub subagent_model: ModelId,
    pub snapshot_revision: Revision,
    pub working_dir: PathBuf,
    pub system_prompt_seed: Vec<PromptSegment>,
    pub max_turns: Option<u32>,
    // pub max_tool_calls_per_turn: u32,
    pub subagent_depth: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
pub struct SessionState {
    pub messages: Vec<Message>,
    pub file_view_cache: FileViewCache,
    pub appended_prompt_segments: Vec<PromptSegment>,
    pub pending_tool_calls: IndexMap<ToolCallId, PendingToolCall>,
    pub usage_so_far: Usage,
    pub summaries: Vec<SummarySlice>,
    pub lineage: Vec<SubagentRef>,
    pub fired_hook_ids: Vec<String>,
    pub pending_session_start_source: Option<HookSessionStartSource>,
    pub pending_warning_messages: Vec<HookWarning>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ObservedState {
    pub cwd: PathBuf,
    pub git_branch: Option<String>,
    pub git_dirty: Option<bool>,
    pub now_utc: Timestamp,
    pub env_facts: IndexMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
pub struct InstructionFile {
    pub path: PathBuf,
    pub body: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
pub struct SkillDef {
    pub id: SkillId,
    pub name: String,
    pub description: String,
    pub body: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
pub struct AgentDef {
    pub id: AgentId,
    pub name: String,
    pub prompt: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
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
pub struct PromptRegistry {
    pub prompts: IndexMap<String, Vec<PromptSegment>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
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
pub struct ProviderCapabilities {
    pub supports_tools: bool,
    pub supports_streaming: bool,
    pub supports_reasoning: bool,
    pub supports_interleaved_reasoning: bool,
    pub supports_images: bool,
    pub supports_documents: bool,
    pub supports_prompt_cache: bool,
    pub supports_tool_result_media: bool,
    pub requires_non_empty_assistant_content: bool,
    pub tool_call_id_policy: ToolCallIdPolicy,
    pub max_input_tokens: u64,
    pub max_output_tokens: u64,
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
pub enum ToolCallIdPolicy {
    ProviderSupplied,
    RuntimeSynthesized,
    StableReplayNormalized,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
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
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ContextPlan {
    pub prompt_segments: Vec<PromptSegment>,
    pub transcript_window: TranscriptWindow,
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
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct AssembledPrompt {
    pub segments: Vec<PromptSegment>,
    pub transcript: Vec<Message>,
    pub ordered_segments: Vec<PromptSegment>,
    pub prefix_cache_key: String,
    pub rendered_prefix: String,
    pub rendered_transcript: String,
    pub rendered: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
pub struct ProviderRequest {
    pub session_id: SessionId,
    pub turn_id: TurnId,
    pub model: ResolvedModel,
    pub prompt: AssembledPrompt,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
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
        let event = SessionEvent {
            session_id: SessionId::new(),
            sequence: 1,
            delivery: Delivery::Lossless,
            payload: SessionEventPayload::TurnCompleted {
                turn_id: TurnId::new(),
                usage: Usage {
                    input_tokens: 10,
                    output_tokens: 5,
                    cache_creation_input_tokens: 0,
                    cache_read_input_tokens: 0,
                },
            },
        };
        let encoded = serde_json::to_string(&event).expect("serialize event");
        let decoded: SessionEvent = serde_json::from_str(&encoded).expect("deserialize event");
        assert_eq!(decoded, event);
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
}
