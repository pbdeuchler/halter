// pattern: Functional Core

mod compaction;
mod context;
mod event_bus;
mod hooks_runtime;
mod model_selection;
mod prompt;
mod session;
mod subagent_session;
mod subagents;
mod trace_recorder;
mod turn_registry;

pub use compaction::{ContextSettings, score_message};
pub use context::{
    CompactionOutcome, ContextManager, DefaultContextManager, resolve_response_chain,
};
pub use event_bus::EventBus;
pub use hooks_runtime::{
    ExecutedHookDispatch, HookInvocationContext, run_notification, run_post_compact,
    run_post_tool_use, run_post_tool_use_failure, run_pre_compact, run_pre_tool_use,
    run_session_end, run_session_start, run_stop, run_subagent_start, run_subagent_stop,
    run_user_prompt_submit,
};
pub use prompt::{DefaultPromptAssembler, PromptAssembler, skill_prompt_segment};
pub use session::{HalterSession, ResourceHandle, RuntimeServices, SessionInit, SessionRuntime};
pub use trace_recorder::TraceRecorder;
pub use turn_registry::{ShutdownReport, TurnRegistry, TurnRegistryError};
