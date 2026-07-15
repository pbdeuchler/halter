//! Session runtime, context planning, prompt assembly, hooks, and event flow.
//!
//! `halter-runtime` is the orchestration layer between providers, tools,
//! hooks, session storage, and resource snapshots. Embedders usually access it
//! through `halter::Halter`, but the exported types are useful for custom SDK
//! assembly and tests.
// pattern: Functional Core

mod compaction;
mod context;
mod event_bus;
mod hooks_runtime;
mod model_judge;
mod model_selection;
mod prompt;
mod session;
mod subagent_session;
mod subagents;
mod trace_export;
mod trace_format;
mod trace_recorder;
mod turn_registry;

pub use compaction::{ContextSettings, score_message};
pub use context::{
    CompactionEffects, CompactionOutcome, ContextManager, DefaultContextManager,
    resolve_response_chain,
};
pub use event_bus::EventBus;
pub use halter_protocol::SubagentEventForwarding;
pub use hooks_runtime::{
    ExecutedHookDispatch, HookInvocationContext, run_notification, run_post_compact,
    run_post_tool_use, run_post_tool_use_failure, run_pre_compact, run_pre_tool_use,
    run_session_end, run_session_start, run_stop, run_subagent_start, run_subagent_stop,
    run_user_prompt_submit,
};
pub use prompt::{
    DefaultPromptAssembler, PromptAssembler, appended_system_prompt_segment,
    coding_agent_prompt_segment, default_coding_agent_prompt, default_compaction_prompt,
    default_system_prompt, default_system_prompt_segment, skill_prompt_segment,
    system_prompt_segment,
};
pub use session::{
    HalterSession, ParentStreamRegistry, ResourceHandle, RuntimeServices, SessionInit,
    SessionRuntime,
};
pub use trace_export::export_session_trace;
pub use trace_recorder::TraceRecorder;
pub use turn_registry::{ShutdownReport, TurnRegistry, TurnRegistryError};
