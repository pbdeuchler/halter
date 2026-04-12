// pattern: Functional Core

mod context;
mod event_bus;
mod hooks_runtime;
mod model_selection;
mod prompt;
mod session;
mod subagent_session;
mod subagents;

pub use context::{ContextManager, ContextSettings, DefaultContextManager};
pub use event_bus::EventBus;
pub use hooks_runtime::{
    ExecutedHookDispatch, HookInvocationContext, run_notification, run_post_compact,
    run_post_tool_use, run_post_tool_use_failure, run_pre_compact, run_pre_tool_use,
    run_session_end, run_session_start, run_stop, run_subagent_start, run_subagent_stop,
    run_user_prompt_submit,
};
pub use prompt::{DefaultPromptAssembler, PromptAssembler};
pub use session::{HalterSession, ResourceHandle, RuntimeServices, SessionInit, SessionRuntime};
