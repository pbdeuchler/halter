// pattern: Functional Core

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
    pub use halter_runtime::{HalterSession, SessionInit, SessionRuntime};

    pub use crate::{Halter, HalterBuilder, PluginLoader, ResourceCompiler, SkillLoader};
}
