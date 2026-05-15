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
    ModelConfig, ModelsConfig, NetworkPolicyConfig, PolicyConfig, PromptsConfig, ProviderConfig,
    ProvidersConfig, ResolvedProviderConfig, ResourcesConfig, RuntimeConfig, SMALL_MODEL_ID,
    SUBAGENT_MODEL_ID, SearchRoots, SessionBackend, SessionsConfig, ShellPolicyConfig, ToolsConfig,
    resolve_provider_runtime_config,
};
