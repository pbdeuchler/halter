// pattern: Imperative Shell

#[cfg(feature = "ast-tools")]
pub mod ast;
#[cfg(feature = "browser-tools")]
pub mod browser;
pub mod common;
pub mod edit;
pub mod fs_lock;
pub mod glob;
pub mod grep;
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

#[cfg(feature = "ast-tools")]
pub use ast::AstGrepTool;
#[cfg(feature = "browser-tools")]
pub use browser::BrowserTool;
pub use edit::EditTool;
pub use glob::GlobTool;
pub use grep::GrepTool;
#[cfg(feature = "image-tools")]
pub use image::ImageTool;
pub use process::ProcessTool;
#[cfg(feature = "profiling")]
pub use profiling::ProfilingTool;
#[cfg(feature = "pty")]
pub use pty::PtyTool;
pub use read::ReadTool;
pub use shell::ShellTool;
pub use write::WriteTool;

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

    #[cfg(feature = "browser-tools")]
    {
        let tool = Arc::new(BrowserTool) as Arc<dyn Tool>;
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;
    use tokio_util::sync::CancellationToken;

    use crate::{
        DefaultToolPolicy, NoopToolEventSink, PathLockMap, PolicySettings, ToolContext, ToolPolicy,
        ToolSessionStore,
    };

    use super::*;

    fn tool_context(root: &std::path::Path, policy: Arc<dyn ToolPolicy>) -> ToolContext {
        ToolContext {
            session_id: halter_protocol::SessionId::new(),
            working_dir: root.to_path_buf(),
            path_locks: Arc::new(PathLockMap::default()),
            tool_sessions: Arc::new(ToolSessionStore::default()),
            snapshot: Arc::new(halter_protocol::ResourceSnapshot::empty()),
            cancel: CancellationToken::new(),
            emit: Arc::new(NoopToolEventSink),
            policy,
            shell_timeout_secs: 30,
            subagent_parent: None,
        }
    }

    #[tokio::test]
    async fn write_then_read_roundtrips_file() {
        let temp = tempfile::tempdir().expect("tempdir");
        let runtime = ToolRuntime::new();
        register_builtin_tools(&runtime, &[]);
        let policy: Arc<dyn ToolPolicy> = Arc::new(DefaultToolPolicy::new(PolicySettings {
            allowed_write_roots: vec![temp.path().to_path_buf()],
            ..PolicySettings::default()
        }));
        let context = tool_context(temp.path(), policy);

        runtime
            .execute(
                "write",
                context.clone(),
                json!({ "path": "note.txt", "content": "hello" }),
            )
            .await
            .expect("write file");

        let result = runtime
            .execute("read", context, json!({ "path": "note.txt", "limit": 500 }))
            .await
            .expect("read file");

        let halter_protocol::ToolResult::Json { value } = result else {
            panic!("expected json result");
        };
        assert_eq!(value["content"], "hello");
    }

    #[test]
    fn builtin_registration_respects_enabled_list() {
        let runtime = ToolRuntime::new();
        register_builtin_tools(&runtime, &["read".to_owned(), "glob".to_owned()]);
        let specs = runtime.specs();

        assert!(specs.iter().any(|spec| spec.name.0 == "read"));
        assert!(specs.iter().any(|spec| spec.name.0 == "glob"));
        assert!(!specs.iter().any(|spec| spec.name.0 == "write"));
        assert!(!specs.iter().any(|spec| spec.name.0 == "shell"));
    }

    #[test]
    fn specs_are_returned_in_alphabetical_order() {
        let runtime = ToolRuntime::new();
        register_builtin_tools(
            &runtime,
            &[
                "write".to_owned(),
                "edit".to_owned(),
                "read".to_owned(),
                "glob".to_owned(),
                "grep".to_owned(),
            ],
        );

        let names: Vec<String> = runtime
            .specs()
            .into_iter()
            .map(|spec| spec.name.0)
            .collect();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
    }

    #[tokio::test]
    async fn write_tool_respects_policy_denials() {
        let temp = tempfile::tempdir().expect("tempdir");
        let allowed = temp.path().join("allowed");
        std::fs::create_dir(&allowed).expect("create allowed root");
        let policy: Arc<dyn ToolPolicy> = Arc::new(DefaultToolPolicy::new(PolicySettings {
            allowed_write_roots: vec![allowed],
            ..PolicySettings::default()
        }));
        let context = tool_context(temp.path(), policy);

        let error = WriteTool
            .execute(context, json!({ "path": "denied.txt", "content": "nope" }))
            .await
            .expect_err("write outside the allowed root must be denied");

        let message = error.to_string();
        assert!(
            message.contains("not under any allowed root"),
            "expected NotInRoot, got: {message}"
        );
    }

    #[tokio::test]
    async fn glob_tool_respects_gitignore() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join(".gitignore"), "ignored.txt\n").expect("write gitignore");
        std::fs::write(temp.path().join("ignored.txt"), "ignored").expect("write ignored file");
        std::fs::write(temp.path().join("visible.txt"), "visible").expect("write visible file");
        let policy: Arc<dyn ToolPolicy> =
            Arc::new(DefaultToolPolicy::new(PolicySettings::default()));
        let context = tool_context(temp.path(), policy);

        let result = GlobTool
            .execute(context, json!({ "pattern": "*.txt" }))
            .await
            .expect("glob should succeed");

        let halter_protocol::ToolResult::Json { value } = result else {
            panic!("expected json result");
        };
        let matches = value["matches"].as_array().expect("matches array");
        assert_eq!(matches.len(), 1);
        assert!(
            matches[0]["path"]
                .as_str()
                .expect("match path")
                .ends_with("visible.txt")
        );
    }

    #[tokio::test]
    async fn grep_tool_respects_type_filters_and_count_mode() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("main.rs"), "needle\nneedle\n").expect("write rust file");
        std::fs::write(temp.path().join("notes.txt"), "needle\n").expect("write text file");
        let policy: Arc<dyn ToolPolicy> =
            Arc::new(DefaultToolPolicy::new(PolicySettings::default()));
        let context = tool_context(temp.path(), policy);

        let result = GrepTool
            .execute(
                context,
                json!({
                    "pattern": "needle",
                    "type": "rust",
                    "output_mode": "count"
                }),
            )
            .await
            .expect("grep should succeed");

        let halter_protocol::ToolResult::Json { value } = result else {
            panic!("expected json result");
        };
        let matches = value["matches"].as_array().expect("matches array");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0]["path"], "main.rs");
        assert_eq!(matches[0]["match_count"], 2);
        assert_eq!(value["total_matches"], 2);
        assert_eq!(value["files_with_matches"], 1);
    }

    #[tokio::test]
    async fn grep_tool_skips_binary_files() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join("text.txt"), "needle\n").expect("write text file");
        std::fs::write(temp.path().join("binary.bin"), b"needle\0suffix").expect("write binary");
        let policy: Arc<dyn ToolPolicy> =
            Arc::new(DefaultToolPolicy::new(PolicySettings::default()));
        let context = tool_context(temp.path(), policy);

        let result = GrepTool
            .execute(context, json!({ "pattern": "needle" }))
            .await
            .expect("grep should succeed");

        let halter_protocol::ToolResult::Json { value } = result else {
            panic!("expected json result");
        };
        let matches = value["matches"].as_array().expect("matches array");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0]["path"], "text.txt");
        assert_eq!(value["files_searched"], 1);
        assert_eq!(value["files_with_matches"], 1);
    }

    #[tokio::test]
    async fn grep_tool_supports_multiline_and_context() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(
            temp.path().join("main.rs"),
            "alpha\nstart(\n  beta\n)\nomega\n",
        )
        .expect("write");
        let policy: Arc<dyn ToolPolicy> =
            Arc::new(DefaultToolPolicy::new(PolicySettings::default()));
        let context = tool_context(temp.path(), policy);

        let result = GrepTool
            .execute(
                context,
                json!({
                    "pattern": "start\\(\\n  beta\\n\\)",
                    "multiline": true,
                    "context_before": 1,
                    "context_after": 1
                }),
            )
            .await
            .expect("grep should succeed");

        let halter_protocol::ToolResult::Json { value } = result else {
            panic!("expected json result");
        };
        let matches = value["matches"].as_array().expect("matches array");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0]["line_number"], 2);
        assert_eq!(matches[0]["context_before"][0]["line"], "alpha");
        assert_eq!(matches[0]["context_after"][0]["line"], "omega");
    }

    #[tokio::test]
    async fn grep_tool_respects_gitignore() {
        let temp = tempfile::tempdir().expect("tempdir");
        std::fs::write(temp.path().join(".gitignore"), "ignored.txt\n").expect("write gitignore");
        std::fs::write(temp.path().join("ignored.txt"), "needle\n").expect("write ignored");
        std::fs::write(temp.path().join("visible.txt"), "needle\n").expect("write visible");
        let policy: Arc<dyn ToolPolicy> =
            Arc::new(DefaultToolPolicy::new(PolicySettings::default()));
        let context = tool_context(temp.path(), policy);

        let result = GrepTool
            .execute(
                context,
                json!({ "pattern": "needle", "output_mode": "files_with_matches" }),
            )
            .await
            .expect("grep should succeed");

        let halter_protocol::ToolResult::Json { value } = result else {
            panic!("expected json result");
        };
        let matches = value["matches"].as_array().expect("matches array");
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0]["path"], "visible.txt");
        assert_eq!(value["files_with_matches"], 1);
    }
}
