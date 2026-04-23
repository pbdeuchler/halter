// pattern: Functional Core

mod builtin;
mod policy;
mod runtime;
mod session_store;
mod subagent;

#[cfg(feature = "ast-tools")]
pub use builtin::AstGrepTool;
#[cfg(feature = "browser-tools")]
pub use builtin::BrowserTool;
#[cfg(feature = "image-tools")]
pub use builtin::ImageTool;
#[cfg(feature = "profiling")]
pub use builtin::ProfilingTool;
#[cfg(feature = "pty")]
pub use builtin::PtyTool;
pub use builtin::fs_lock::PathLockMap;
pub use builtin::{
    EditTool, GlobTool, GrepTool, ProcessTool, ReadTool, ShellTool, WriteTool,
    register_builtin_tools,
};
pub use policy::{
    CanonicalPath, DefaultToolPolicy, LoopbackAllow, Pid, PolicyError, PolicySettings, ShellMode,
    ToolPolicy,
};
pub use runtime::{
    NoopSubagentControl, NoopToolEventSink, SubagentControl, SubagentParentContext, Tool,
    ToolContext, ToolEventSink, ToolRuntime, ToolRuntimeEvent,
};
pub use session_store::ToolSessionStore;
pub use subagent::register_subagent_tools;
