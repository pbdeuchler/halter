//! Configuration schema, loading, and provider resolution.
//!
//! `halter-config` owns the TOML-facing configuration model and the layered
//! loader used by the CLI and SDK. It validates static config values here,
//! then resolves provider runtime requirements such as API keys or OpenAI
//! OAuth credentials at load time.
// pattern: Functional Core

mod loader;
mod schema;

pub use halter_protocol::SubagentEventForwarding;
pub use loader::{
    LayeredConfigPaths, apply_env_overrides, config_fingerprint, expand_path, export_json_schema,
    generate_starter_config, load_layered, load_path, schema_as_json_value,
};
pub use schema::{
    ConfiguredProvider, ContextConfig, DEFAULT_MODEL_ID, HarnessConfig, LoopbackAllowConfig,
    ModelConfig, ModelJudgeConfig, ModelSlot, ModelSlotRef, ModelsConfig, NetworkPolicyConfig,
    OpenAiOAuthConfig, PolicyConfig, PromptsConfig, ProviderConfig, ProvidersConfig,
    ResolvedProviderAuth, ResolvedProviderConfig, ResourcesConfig, RuntimeConfig, SMALL_MODEL_ID,
    SUBAGENT_MODEL_ID, SearchRoots, SessionBackend, SessionsConfig, ShellPolicyConfig,
    SystemPromptPreset, ToolsConfig, resolve_provider_runtime_config,
};
