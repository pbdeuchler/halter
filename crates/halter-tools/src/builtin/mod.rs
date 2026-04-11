// pattern: Imperative Shell

pub mod common;
pub mod edit;
pub mod fs_lock;
pub mod grep;
pub mod glob;
#[cfg(feature = "ast-tools")]
pub mod ast;
#[cfg(feature = "image-tools")]
pub mod image;
pub mod process;
pub mod profiling;
#[cfg(feature = "pty")]
pub mod pty;
pub mod read;
pub mod shell;
pub mod text;
pub mod write;

use std::sync::Arc;

use crate::{Tool, ToolRuntime};

pub use glob::GlobTool;
pub use grep::GrepTool;
pub use process::ProcessTool;
#[cfg(feature = "profiling")]
pub use profiling::ProfilingTool;
#[cfg(feature = "pty")]
pub use pty::PtyTool;
pub use read::ReadTool;
pub use shell::ShellTool;
pub use write::WriteTool;
pub use edit::EditTool;
#[cfg(feature = "ast-tools")]
pub use ast::AstGrepTool;
#[cfg(feature = "image-tools")]
pub use image::ImageTool;

/// Registers the built-in tool set against a runtime that is already being assembled.
pub fn register_builtin_tools(runtime: &ToolRuntime, enabled: &[String]) {
    let register_all = enabled.is_empty();
    for tool in [
        Arc::new(ReadTool) as Arc<dyn Tool>,
        Arc::new(WriteTool),
        Arc::new(EditTool),
        Arc::new(GlobTool),
        Arc::new(GrepTool),
        Arc::new(ShellTool),
        Arc::new(ProcessTool),
    ] {
        let tool_name = tool.spec().name.0;
        if register_all || enabled.iter().any(|name| name == &tool_name) {
            runtime.register(tool);
        }
    }

    #[cfg(feature = "pty")]
    {
        let tool = Arc::new(PtyTool) as Arc<dyn Tool>;
        let tool_name = tool.spec().name.0;
        if register_all || enabled.iter().any(|name| name == &tool_name) {
            runtime.register(tool);
        }
    }

    #[cfg(feature = "ast-tools")]
    {
        let tool = Arc::new(AstGrepTool) as Arc<dyn Tool>;
        let tool_name = tool.spec().name.0;
        if register_all || enabled.iter().any(|name| name == &tool_name) {
            runtime.register(tool);
        }
    }

    #[cfg(feature = "image-tools")]
    {
        let tool = Arc::new(ImageTool) as Arc<dyn Tool>;
        let tool_name = tool.spec().name.0;
        if register_all || enabled.iter().any(|name| name == &tool_name) {
            runtime.register(tool);
        }
    }

    #[cfg(feature = "profiling")]
    {
        let tool = Arc::new(ProfilingTool) as Arc<dyn Tool>;
        let tool_name = tool.spec().name.0;
        if register_all || enabled.iter().any(|name| name == &tool_name) {
            runtime.register(tool);
        }
    }
}
