//! Configuration schema, loading, and provider resolution.
//!
//! `halter-config` owns the TOML-facing configuration model and the layered
//! loader used by the CLI and SDK. It validates static config values here,
//! then resolves provider runtime requirements such as API keys or OpenAI
//! OAuth credentials at load time.
// pattern: Functional Core

mod loader;
mod resources;
mod schema;

pub use halter_protocol::{PanelIsolation, SubagentEventForwarding};
pub use loader::{
    LayeredConfigPaths, apply_env_overrides, config_fingerprint, expand_path, export_json_schema,
    generate_starter_config, load_layered, load_path, schema_as_json_value,
};
pub use resources::{
    LoadedAgent, LoadedExecutable, LoadedHooksFile, LoadedLspServer, LoadedMcpServer,
    LoadedOutputStyle, LoadedPlugin, LoadedResourceFile, LoadedSkill, PluginDefaults, PluginLoader,
    SkillLoader,
};
pub use schema::{
    ConfiguredProvider, ContextConfig, DEFAULT_MODEL_ID, DEFAULT_PROVIDER_CONNECT_TIMEOUT_SECS,
    DEFAULT_PROVIDER_REQUEST_TIMEOUT_SECS, DEFAULT_PROVIDER_RETRY_BASE_BACKOFF_MS,
    DEFAULT_PROVIDER_RETRY_DEADLINE_SECS, DEFAULT_PROVIDER_RETRY_JITTER_PCT,
    DEFAULT_PROVIDER_RETRY_MAX_ATTEMPTS, DEFAULT_PROVIDER_RETRY_MAX_BACKOFF_SECS,
    DEFAULT_PROVIDER_STREAM_IDLE_TIMEOUT_SECS, HarnessConfig, LoopbackAllowConfig, ModelConfig,
    ModelJudgeConfig, ModelJudgeMode, ModelSlot, ModelSlotRef, ModelsConfig, NetworkPolicyConfig,
    OpenAiOAuthConfig, PolicyConfig, PromptsConfig, ProviderConfig, ProvidersConfig,
    RequestRetryConfig, RequestRetryOverrideConfig, ResilienceConfig, ResilienceOverrideConfig,
    ResilienceTimeoutsConfig, ResilienceTimeoutsOverrideConfig, ResolvedProviderAuth,
    ResolvedProviderConfig, ResourcesConfig, RuntimeConfig, SMALL_MODEL_ID, SUBAGENT_MODEL_ID,
    SearchRoots, SessionBackend, SessionsConfig, ShellPolicyConfig, SystemPromptPreset,
    ToolsConfig, resolve_provider_runtime_config,
};

#[cfg(feature = "remote-plugins")]
pub mod github;
