//! # halter
//!
//! `halter` is a **simple and configurable agent harness and SDK** for building and
//! operating thoroughbred agents. It assembles config loading, resource compilation,
//! providers, tools, hooks, policy, runtime sessions, and persistence behind a small
//! builder API.
//!
//! ## Example
//!
//! ```rust,no_run
//! use futures::StreamExt;
//! use halter::prelude::*;
//!
//! #[tokio::main]
//! async fn main() -> anyhow::Result<()> {
//!     let harness = Halter::from_config_file("halter.toml").await?;
//!     let session = harness.new_session(SessionInit::default()).await?;
//!
//!     let mut events = session
//!         .submit_turn(Turn::user("Summarize this repository"))
//!         .await?;
//!
//!     while let Some(event) = events.next().await {
//!         println!("{:?}", event?.payload);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## More documentation
//!
//! - Rustdoc API reference: <https://docs.rs/halter>
//! - Full project README: <https://github.com/pbdeuchler/halter/blob/main/README.md>

// pattern: Functional Core

mod builder;
mod resources;

pub use builder::{Halter, HalterBuilder};
pub use halter_config::{
    LoadedAgent, LoadedExecutable, LoadedHooksFile, LoadedLspServer, LoadedMcpServer,
    LoadedOutputStyle, LoadedPlugin, LoadedResourceFile, LoadedSkill, PluginDefaults, PluginLoader,
    SkillLoader,
};
pub use resources::{CompiledResources, ResourceCompiler};

pub mod session {
    pub use halter_session::{InMemorySessionStore, SessionStore, StoredSession};

    #[cfg(feature = "sqlite")]
    pub use halter_session::SqliteSessionStore;
}

pub mod providers {
    pub use halter_providers::{
        DefaultProviderErrorClassifier, ProviderErrorClassifier, ProviderErrorKind,
        ProviderTimeouts, ResiliencePolicy, RetryPolicy,
    };
}

/// Built-in default prompts and helpers for installing them.
///
/// Read or compose the defaults, then seed a session with one via
/// [`prelude::SessionInit::with_system_prompt`] or `[prompts]` config. For a
/// batteries-included coding agent, use [`prompts::default_coding_agent_prompt`]
/// (or set `prompts.preset = "coding"` in config).
pub mod prompts {
    pub use halter_config::SystemPromptPreset;
    pub use halter_runtime::{
        appended_system_prompt_segment, coding_agent_prompt_segment, default_coding_agent_prompt,
        default_compaction_prompt, default_system_prompt, default_system_prompt_segment,
        system_prompt_segment,
    };
}

pub mod prelude {
    pub use halter_config::{HarnessConfig, PromptsConfig, SystemPromptPreset};
    pub use halter_protocol::{
        Message, ResourceSnapshot, SessionEvent, SessionEventPayload, SessionId, Turn,
    };
    pub use halter_runtime::{HalterSession, SessionInit, SessionRuntime, SubagentEventForwarding};

    pub use crate::prompts;
    pub use crate::providers;
    // Re-exported from the `providers` module so the flattened names have a
    // single maintenance point.
    pub use crate::providers::*;
    pub use crate::{Halter, HalterBuilder, PluginLoader, ResourceCompiler, SkillLoader};
}
