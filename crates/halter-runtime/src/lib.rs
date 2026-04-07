// pattern: Functional Core

mod context;
mod event_bus;
mod model_selection;
mod prompt;
mod session;
mod subagent_session;
mod subagents;

pub use context::{ContextManager, ContextSettings, DefaultContextManager};
pub use event_bus::EventBus;
pub use prompt::{DefaultPromptAssembler, PromptAssembler};
pub use session::{HalterSession, ResourceHandle, RuntimeServices, SessionInit, SessionRuntime};
