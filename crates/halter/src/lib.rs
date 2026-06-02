// pattern: Functional Core
//!
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

mod builder;
mod resources;

pub use builder::{Halter, HalterBuilder};
pub use resources::{
    CompiledResources, LoadedAgent, LoadedExecutable, LoadedHooksFile, LoadedLspServer,
    LoadedMcpServer, LoadedOutputStyle, LoadedPlugin, LoadedResourceFile, LoadedSkill,
    PluginDefaults, PluginLoader, ResourceCompiler, SkillLoader,
};

pub mod session {
    pub use halter_session::{InMemorySessionStore, SessionStore, StoredSession};

    #[cfg(feature = "sqlite")]
    pub use halter_session::SqliteSessionStore;
}

pub mod prelude {
    pub use halter_config::HarnessConfig;
    pub use halter_protocol::{
        Message, ResourceSnapshot, SessionEvent, SessionEventPayload, SessionId, Turn,
    };
    pub use halter_runtime::{HalterSession, SessionInit, SessionRuntime, SubagentEventForwarding};

    pub use crate::{Halter, HalterBuilder, PluginLoader, ResourceCompiler, SkillLoader};
}
