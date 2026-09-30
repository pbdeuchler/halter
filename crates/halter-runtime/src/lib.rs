//! Session runtime, context planning, prompt assembly, hooks, and event flow.
//!
//! `halter-runtime` is the orchestration layer between providers, tools,
//! hooks, session storage, and resource snapshots. Embedders usually access it
//! through `halter::Halter`, but the exported types are useful for custom SDK
//! assembly and tests.
// pattern: Functional Core

mod clean_window;
mod compaction;
mod session_search;
mod skills;
pub use clean_window::{
    CLEAN_WINDOW_BOOTSTRAP, CLEAN_WINDOW_PROMPT, CleanWindow, ROLLOVER_REMINDER,
};
pub use session_search::{SessionSearchBackend, SessionSearchRequest, StoreSearch};
pub use skills::{SKILL_TOOL_NAME, skill_index_segment};
mod compaction_strategy;
mod context;
mod event_bus;
mod hooks_runtime;
mod model_judge;
mod model_selection;
mod model_summary;
mod prompt;
mod provider_default;
mod session;
mod session_driver;
#[cfg(test)]
mod session_driver_tests;
mod session_lease;
mod subagent_session;
mod subagents;
mod trace_export;
mod trace_format;
mod trace_recorder;
mod turn_registry;

pub use compaction::{ContextCapExceeded, ContextSettings};
pub use compaction_strategy::{
    CompactionBoundary, CompactionContext, CompactionNotification, CompactionStrategy,
    CompactionTrigger, WindowPolicy, compaction_instructions,
};
pub use context::{
    CompactionEffects, ContextManager, DefaultContextManager, prompt_segments,
    resolve_response_chain,
};
pub use model_summary::{CHECKPOINT_PREFIX, ModelSummary, TODO_NUDGE, TODO_REMINDER};
pub use provider_default::ProviderDefault;

pub use event_bus::EventBus;
pub use halter_protocol::SubagentEventForwarding;
#[cfg(test)]
pub(crate) use hooks_runtime::run_notification;
pub use hooks_runtime::{ExecutedHookDispatch, HookInvocationContext};
pub(crate) use hooks_runtime::{
    run_post_compact, run_post_tool_use, run_post_tool_use_failure, run_pre_compact,
    run_pre_tool_use, run_session_end, run_session_start, run_stop, run_subagent_start,
    run_subagent_stop, run_user_prompt_submit,
};
pub use prompt::{
    DefaultPromptAssembler, PromptAssembler, appended_system_prompt_segment,
    coding_agent_prompt_segment, default_coding_agent_prompt, default_compaction_prompt,
    default_system_prompt, default_system_prompt_segment, system_prompt_segment,
};
pub use session::SessionEventStream;
pub(crate) use session::SessionExecutor;
pub use session::{
    ParentStreamRegistry, ResourceHandle, RuntimeServices, SessionInit, SessionRuntime,
};
pub use session_driver::{SessionError, SessionHandle, Submission};
pub use session_lease::SessionLeases;
pub use trace_export::export_session_trace;
pub use trace_recorder::TraceRecorder;
pub use turn_registry::{ShutdownReport, TurnRegistry, TurnRegistryError};
