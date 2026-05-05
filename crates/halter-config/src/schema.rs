// pattern: Functional Core

use std::fmt;
use std::path::PathBuf;

use anyhow::Context;
use halter_protocol::{
    ApiKind, DEFAULT_TEMPERATURE, ProviderKind, PruneSignalThreshold, ReasoningEffort,
};
use indexmap::IndexMap;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub const DEFAULT_MODEL_ID: &str = "default";
pub const SMALL_MODEL_ID: &str = "small";
pub const SUBAGENT_MODEL_ID: &str = "subagent";

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HarnessConfig {
    pub version: u32,
    #[serde(default)]
    pub providers: ProvidersConfig,
    #[serde(default)]
    pub models: ModelsConfig,
    #[serde(default)]
    pub resources: ResourcesConfig,
    #[serde(default)]
    pub prompts: PromptsConfig,
    #[serde(default)]
    pub context: ContextConfig,
    #[serde(default)]
    pub tools: ToolsConfig,
    #[serde(default)]
    pub policy: PolicyConfig,
    #[serde(default)]
    pub sessions: SessionsConfig,
    #[serde(default)]
    pub runtime: RuntimeConfig,
}

impl Default for HarnessConfig {
    fn default() -> Self {
        Self {
            version: 1,
            providers: ProvidersConfig::default(),
            models: ModelsConfig::default(),
            resources: ResourcesConfig::default(),
            prompts: PromptsConfig::default(),
            context: ContextConfig::default(),
            tools: ToolsConfig::default(),
            policy: PolicyConfig::default(),
            sessions: SessionsConfig::default(),
            runtime: RuntimeConfig::default(),
        }
    }
}

impl HarnessConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.version != 1 {
            anyhow::bail!(
                "invalid configuration: unsupported version {}",
                self.version
            );
        }

        self.providers.validate()?;
        self.sessions.validate()?;

        let model = self.default_model()?;
        validate_model_config("models.default", model)?;
        if let Some(model) = self.small_model() {
            validate_model_config("models.small", model)?;
        }
        if let Some(model) = self.subagent_model() {
            validate_model_config("models.subagent", model)?;
        }

        if self.policy.max_read_bytes == 0 {
            anyhow::bail!("invalid configuration: max_read_bytes must be greater than zero");
        }

        self.context.validate()?;
        self.runtime.validate()?;

        Ok(())
    }

    pub fn default_model(&self) -> anyhow::Result<&ModelConfig> {
        self.models
            .default
            .as_ref()
            .context("invalid configuration: [models.default] is required")
    }

    #[must_use]
    pub fn subagent_model(&self) -> Option<&ModelConfig> {
        self.models.subagent.as_ref()
    }

    #[must_use]
    pub fn small_model(&self) -> Option<&ModelConfig> {
        self.models.small.as_ref()
    }

    #[must_use]
    pub fn provider_config(&self, provider: ConfiguredProvider) -> Option<&ProviderConfig> {
        self.providers.get(provider)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct ProvidersConfig {
    #[serde(default)]
    pub openai: Option<ProviderConfig>,
    #[serde(default)]
    pub anthropic: Option<ProviderConfig>,
    #[serde(default)]
    pub openrouter: Option<ProviderConfig>,
}

impl ProvidersConfig {
    fn validate(&self) -> anyhow::Result<()> {
        for (name, config) in [
            ("openai", self.openai.as_ref()),
            ("anthropic", self.anthropic.as_ref()),
            ("openrouter", self.openrouter.as_ref()),
        ] {
            if let Some(config) = config {
                validate_provider_config(name, config)?;
            }
        }

        Ok(())
    }

    #[must_use]
    pub fn get(&self, provider: ConfiguredProvider) -> Option<&ProviderConfig> {
        match provider {
            ConfiguredProvider::OpenAi => self.openai.as_ref(),
            ConfiguredProvider::Anthropic => self.anthropic.as_ref(),
            ConfiguredProvider::OpenRouter => self.openrouter.as_ref(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
pub struct ModelsConfig {
    #[serde(default)]
    pub default: Option<ModelConfig>,
    #[serde(default)]
    pub small: Option<ModelConfig>,
    #[serde(default)]
    pub subagent: Option<ModelConfig>,
}

#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
pub enum ConfiguredProvider {
    #[serde(rename = "anthropic")]
    Anthropic,
    #[serde(rename = "openai")]
    OpenAi,
    #[serde(rename = "openrouter")]
    OpenRouter,
}

impl ConfiguredProvider {
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenAi => "openai",
            Self::OpenRouter => "openrouter",
        }
    }

    #[must_use]
    pub const fn provider_kind(self) -> ProviderKind {
        match self {
            Self::Anthropic => ProviderKind::Anthropic,
            Self::OpenAi => ProviderKind::OpenAi,
            Self::OpenRouter => ProviderKind::OpenRouter,
        }
    }

    #[must_use]
    pub const fn api_kind(self) -> ApiKind {
        match self {
            Self::Anthropic => ApiKind::AnthropicMessages,
            Self::OpenAi => ApiKind::OpenAiResponses,
            Self::OpenRouter => ApiKind::OpenAiResponses,
        }
    }

    #[must_use]
    pub const fn api_key_env_var(self) -> &'static str {
        match self {
            Self::Anthropic => "ANTHROPIC_API_KEY",
            Self::OpenAi => "OPENAI_API_KEY",
            Self::OpenRouter => "OPENROUTER_API_KEY",
        }
    }

    #[must_use]
    pub const fn default_base_url(self) -> &'static str {
        match self {
            Self::Anthropic => "https://api.anthropic.com",
            Self::OpenAi => "https://api.openai.com",
            Self::OpenRouter => "https://openrouter.ai/api",
        }
    }
}

impl fmt::Display for ConfiguredProvider {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
#[serde(deny_unknown_fields)]
pub struct ProviderConfig {
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub api_key: Option<String>,
    /// Optional HTTP headers applied to every request the provider emits.
    /// Names collide case-insensitively; configured entries override any
    /// default or hardcoded provider header (Authorization, x-api-key, etc.).
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub headers: IndexMap<String, String>,
    /// Optional override for the sampling temperature. Falls back to the
    /// global `DEFAULT_TEMPERATURE` (0.7) when unset. Must be in `0.0..=2.0`.
    #[serde(default)]
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub provider: ConfiguredProvider,
    pub model: String,
    #[serde(default)]
    pub max_input_tokens: Option<u32>,
    #[serde(default)]
    pub max_output_tokens: Option<u32>,
    #[serde(default)]
    pub reasoning: Option<ReasoningEffort>,
    #[serde(default = "default_tokens_per_minute")]
    pub tokens_per_minute: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResolvedProviderConfig {
    pub provider: ConfiguredProvider,
    pub base_url: String,
    pub api_key: String,
    /// Ordered list of user-configured headers. The runtime applies these
    /// over provider defaults using case-insensitive name matching.
    pub headers: Vec<(String, String)>,
    /// Sampling temperature forwarded to every request this provider emits.
    /// Defaults to `DEFAULT_TEMPERATURE` (0.7) when the user does not override
    /// `[providers.<name>].temperature`.
    pub temperature: f32,
}

pub fn resolve_provider_runtime_config<F>(
    provider: ConfiguredProvider,
    configured: Option<&ProviderConfig>,
    mut lookup_env: F,
) -> anyhow::Result<ResolvedProviderConfig>
where
    F: FnMut(&str) -> anyhow::Result<Option<String>>,
{
    let base_url = configured
        .and_then(|config| config.base_url.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(provider.default_base_url())
        .to_owned();

    let configured_api_key = configured
        .and_then(|config| config.api_key.as_deref())
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let env_api_key = if configured_api_key.is_some() {
        None
    } else {
        lookup_env(provider.api_key_env_var())?.and_then(|value| {
            let trimmed = value.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_owned())
        })
    };
    let api_key = configured_api_key.or(env_api_key).with_context(|| {
        format!(
            "missing api key for provider '{}': set [providers.{}].api_key or {}",
            provider,
            provider,
            provider.api_key_env_var()
        )
    })?;

    let headers = configured
        .map(|config| {
            config
                .headers
                .iter()
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect()
        })
        .unwrap_or_default();

    let temperature = configured
        .and_then(|config| config.temperature)
        .unwrap_or(DEFAULT_TEMPERATURE);

    Ok(ResolvedProviderConfig {
        provider,
        base_url,
        api_key,
        headers,
        temperature,
    })
}

fn validate_provider_config(name: &str, provider: &ProviderConfig) -> anyhow::Result<()> {
    validate_optional_string(
        &format!("providers.{name}.base_url"),
        provider.base_url.as_deref(),
    )?;
    validate_optional_string(
        &format!("providers.{name}.api_key"),
        provider.api_key.as_deref(),
    )?;
    for (header_name, header_value) in &provider.headers {
        validate_required_string(&format!("providers.{name}.headers.<key>"), header_name)?;
        if !header_name
            .bytes()
            .all(|b| b.is_ascii_graphic() && b != b':')
        {
            anyhow::bail!(
                "invalid configuration: providers.{name}.headers name '{header_name}' is not a valid HTTP header name"
            );
        }
        validate_optional_string(
            &format!("providers.{name}.headers.{header_name}"),
            Some(header_value),
        )?;
    }
    validate_optional_temperature(
        &format!("providers.{name}.temperature"),
        provider.temperature,
    )?;
    Ok(())
}

fn validate_optional_temperature(path: &str, value: Option<f32>) -> anyhow::Result<()> {
    let Some(temperature) = value else {
        return Ok(());
    };
    if !temperature.is_finite() || !(0.0..=2.0).contains(&temperature) {
        anyhow::bail!("invalid configuration: {path} must be a finite value in 0.0..=2.0");
    }
    Ok(())
}

fn validate_model_config(path: &str, model: &ModelConfig) -> anyhow::Result<()> {
    validate_required_string(&format!("{path}.model"), &model.model)?;
    validate_optional_positive_u32(&format!("{path}.max_input_tokens"), model.max_input_tokens)?;
    validate_optional_positive_u32(
        &format!("{path}.max_output_tokens"),
        model.max_output_tokens,
    )?;
    Ok(())
}

fn validate_required_string(path: &str, value: &str) -> anyhow::Result<()> {
    if value.trim().is_empty() {
        anyhow::bail!("invalid configuration: {path} must not be empty");
    }
    Ok(())
}

fn validate_optional_string(path: &str, value: Option<&str>) -> anyhow::Result<()> {
    if let Some(value) = value {
        validate_required_string(path, value)?;
    }
    Ok(())
}

fn validate_optional_positive_u32(path: &str, value: Option<u32>) -> anyhow::Result<()> {
    if matches!(value, Some(0)) {
        anyhow::bail!("invalid configuration: {path} must be greater than zero");
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
pub struct ResourcesConfig {
    #[serde(default)]
    pub skills: SearchRoots,
    #[serde(default)]
    pub plugins: SearchRoots,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
pub struct SearchRoots {
    #[serde(default)]
    pub roots: Vec<PathBuf>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
pub struct PromptsConfig {
    #[serde(default)]
    pub system_prompt: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ContextConfig {
    /// Trigger compaction when the estimated input reaches this threshold, with a 100-token buffer.
    #[serde(default = "default_compaction_threshold")]
    pub compaction_threshold: u64,
    /// Evict low-signal history until the estimated prefix is below this target before compaction.
    #[serde(default = "default_pre_compaction_target")]
    pub pre_compaction_target: u64,
    /// Highest signal tier eligible for pre-compaction eviction.
    #[serde(default)]
    pub prune_signal_threshold: PruneSignalThreshold,
}

impl ContextConfig {
    fn validate(&self) -> anyhow::Result<()> {
        if self.compaction_threshold == 0 {
            anyhow::bail!(
                "invalid configuration: context.compaction_threshold must be greater than zero"
            );
        }
        if self.pre_compaction_target >= self.compaction_threshold {
            anyhow::bail!(
                "invalid configuration: context.pre_compaction_target must be less than context.compaction_threshold"
            );
        }

        Ok(())
    }
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            compaction_threshold: default_compaction_threshold(),
            pre_compaction_target: default_pre_compaction_target(),
            prune_signal_threshold: PruneSignalThreshold::default(),
        }
    }
}

const fn default_tokens_per_minute() -> Option<u64> {
    Some(500_000)
}

const fn default_compaction_threshold() -> u64 {
    80_000
}

const fn default_pre_compaction_target() -> u64 {
    60_000
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
pub struct ToolsConfig {
    #[serde(default)]
    pub enabled: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct PolicyConfig {
    #[serde(default = "default_write_roots")]
    pub allowed_write_roots: Vec<PathBuf>,
    #[serde(default = "default_max_read_bytes")]
    pub max_read_bytes: usize,
    #[serde(default = "default_max_subagent_depth")]
    pub max_subagent_depth: u32,
    #[serde(default = "default_max_concurrent_subagents")]
    pub max_concurrent_subagents: usize,
    #[serde(default)]
    pub shell: ShellPolicyConfig,
    #[serde(default)]
    pub network: NetworkPolicyConfig,
}

impl Default for PolicyConfig {
    fn default() -> Self {
        Self {
            allowed_write_roots: default_write_roots(),
            max_read_bytes: default_max_read_bytes(),
            max_subagent_depth: default_max_subagent_depth(),
            max_concurrent_subagents: default_max_concurrent_subagents(),
            shell: ShellPolicyConfig::default(),
            network: NetworkPolicyConfig::default(),
        }
    }
}

fn default_write_roots() -> Vec<PathBuf> {
    vec![PathBuf::from("."), PathBuf::from("/tmp/halter")]
}

const fn default_max_read_bytes() -> usize {
    1_048_576
}

const fn default_max_subagent_depth() -> u32 {
    3
}

const fn default_max_concurrent_subagents() -> usize {
    8
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ShellPolicyConfig {
    #[serde(default = "default_shell_enabled")]
    pub enabled: bool,
    #[serde(default = "default_shell_allowlist")]
    pub allow: Vec<String>,
    #[serde(default = "default_shell_timeout_secs")]
    pub timeout_secs: u64,
}

impl Default for ShellPolicyConfig {
    fn default() -> Self {
        Self {
            enabled: default_shell_enabled(),
            allow: default_shell_allowlist(),
            timeout_secs: default_shell_timeout_secs(),
        }
    }
}

const fn default_shell_enabled() -> bool {
    true
}

fn default_shell_allowlist() -> Vec<String> {
    vec![
        "git".to_owned(),
        "cargo".to_owned(),
        "rg".to_owned(),
        "ls".to_owned(),
        "find".to_owned(),
    ]
}

const fn default_shell_timeout_secs() -> u64 {
    30
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
pub struct NetworkPolicyConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub allowed_hosts: Vec<String>,
    /// Loopback sidecar allowlist. Accepts the legacy TOML key
    /// `allowed_loopback_services` as an alias (Phase 1 rename).
    #[serde(default, alias = "allowed_loopback_services")]
    pub allowed_loopback: Vec<LoopbackAllowConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct LoopbackAllowConfig {
    pub host: String,
    #[serde(default)]
    pub port: Option<u16>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SessionBackend {
    #[default]
    Memory,
    #[cfg(feature = "sqlite")]
    Sqlite,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SessionsConfig {
    #[serde(default)]
    pub backend: SessionBackend,
    #[serde(default)]
    pub sqlite_path: Option<PathBuf>,
}

impl Default for SessionsConfig {
    fn default() -> Self {
        Self {
            backend: SessionBackend::Memory,
            sqlite_path: None,
        }
    }
}

impl SessionsConfig {
    fn validate(&self) -> anyhow::Result<()> {
        if let Some(path) = &self.sqlite_path
            && path.as_os_str().is_empty()
        {
            anyhow::bail!("invalid configuration: sessions.sqlite_path must not be empty");
        }

        #[cfg(feature = "sqlite")]
        if self.sqlite_path.is_some() && self.backend != SessionBackend::Sqlite {
            anyhow::bail!(
                "invalid configuration: sessions.sqlite_path requires sessions.backend = 'sqlite'"
            );
        }

        #[cfg(not(feature = "sqlite"))]
        if self.sqlite_path.is_some() {
            anyhow::bail!(
                "invalid configuration: sessions.sqlite_path requires the 'sqlite' cargo feature"
            );
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
pub struct RuntimeConfig {
    #[serde(default)]
    pub working_dir: Option<PathBuf>,
    /// Directory where full per-session traces are written as `<session_id>.txt`
    /// JSONL files. Each file begins with a header line describing the session
    /// blueprint and is followed by one JSON-encoded `SessionEvent` per line —
    /// enough to debug a run and to rebuild session state offline.
    #[serde(default)]
    pub traces_dir: Option<PathBuf>,
}

impl RuntimeConfig {
    fn validate(&self) -> anyhow::Result<()> {
        if let Some(path) = &self.traces_dir
            && path.as_os_str().is_empty()
        {
            anyhow::bail!("invalid configuration: runtime.traces_dir must not be empty");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_requires_models_default() {
        let error = HarnessConfig::default()
            .validate()
            .expect_err("validation should fail");

        assert!(
            error
                .to_string()
                .contains("invalid configuration: [models.default] is required")
        );
    }

    #[test]
    fn provider_resolution_uses_configured_api_key_before_env() {
        let resolved = resolve_provider_runtime_config(
            ConfiguredProvider::OpenAi,
            Some(&ProviderConfig {
                base_url: Some("https://proxy.example.com".to_owned()),
                api_key: Some("configured-key".to_owned()),
                headers: IndexMap::new(),
                temperature: None,
            }),
            |_| Ok(Some("env-key".to_owned())),
        )
        .expect("resolve provider");

        assert_eq!(resolved.base_url, "https://proxy.example.com");
        assert_eq!(resolved.api_key, "configured-key");
        assert_eq!(resolved.temperature, DEFAULT_TEMPERATURE);
    }

    #[test]
    fn provider_resolution_applies_configured_temperature_override() {
        let resolved = resolve_provider_runtime_config(
            ConfiguredProvider::OpenRouter,
            Some(&ProviderConfig {
                base_url: None,
                api_key: Some("configured-key".to_owned()),
                headers: IndexMap::new(),
                temperature: Some(0.2),
            }),
            |_| Ok(None),
        )
        .expect("resolve provider");

        assert!((resolved.temperature - 0.2).abs() < f32::EPSILON);
    }

    #[test]
    fn provider_resolution_defaults_temperature_when_unset() {
        let resolved = resolve_provider_runtime_config(ConfiguredProvider::Anthropic, None, |_| {
            Ok(Some("env-key".to_owned()))
        })
        .expect("resolve provider");

        assert_eq!(resolved.temperature, DEFAULT_TEMPERATURE);
    }

    #[test]
    fn provider_config_rejects_out_of_range_temperature() {
        let error = validate_provider_config(
            "openrouter",
            &ProviderConfig {
                base_url: None,
                api_key: Some("configured-key".to_owned()),
                headers: IndexMap::new(),
                temperature: Some(2.5),
            },
        )
        .expect_err("validation should fail");

        assert!(
            error
                .to_string()
                .contains("providers.openrouter.temperature must be a finite value in 0.0..=2.0")
        );
    }

    #[test]
    fn provider_config_rejects_nan_temperature() {
        let error = validate_provider_config(
            "openai",
            &ProviderConfig {
                base_url: None,
                api_key: Some("configured-key".to_owned()),
                headers: IndexMap::new(),
                temperature: Some(f32::NAN),
            },
        )
        .expect_err("validation should fail");

        assert!(error.to_string().contains("temperature"));
    }

    #[test]
    fn provider_resolution_requires_api_key_for_selected_provider() {
        let error =
            resolve_provider_runtime_config(ConfiguredProvider::OpenRouter, None, |_| Ok(None))
                .expect_err("resolution should fail");

        assert!(error.to_string().contains("OPENROUTER_API_KEY"));
    }

    #[test]
    fn openai_models_use_openai_responses_api_kind() {
        let model = ModelConfig {
            provider: ConfiguredProvider::OpenAi,
            model: "gpt-5".to_owned(),
            max_input_tokens: None,
            max_output_tokens: None,
            reasoning: Some(ReasoningEffort::Medium),
            tokens_per_minute: None,
        };

        assert_eq!(model.provider.api_kind(), ApiKind::OpenAiResponses);
    }

    #[test]
    fn openrouter_models_use_openai_responses_api_kind() {
        let model = ModelConfig {
            provider: ConfiguredProvider::OpenRouter,
            model: "openai/gpt-5".to_owned(),
            max_input_tokens: None,
            max_output_tokens: None,
            reasoning: Some(ReasoningEffort::Medium),
            tokens_per_minute: None,
        };

        assert_eq!(model.provider.api_kind(), ApiKind::OpenAiResponses);
    }

    #[test]
    fn sqlite_path_requires_supported_sqlite_backend() {
        let mut config = HarnessConfig::default();
        config.models.default = Some(ModelConfig {
            provider: ConfiguredProvider::OpenAi,
            model: "gpt-5".to_owned(),
            max_input_tokens: None,
            max_output_tokens: None,
            reasoning: None,
            tokens_per_minute: None,
        });
        config.providers.openai = Some(ProviderConfig {
            base_url: None,
            api_key: Some("test-key".to_owned()),
            headers: IndexMap::new(),
            temperature: None,
        });
        config.sessions.sqlite_path = Some(PathBuf::from("/tmp/halter.db"));

        let error = config.validate().expect_err("sqlite path should fail");

        #[cfg(feature = "sqlite")]
        assert!(
            error
                .to_string()
                .contains("sessions.sqlite_path requires sessions.backend = 'sqlite'")
        );

        #[cfg(not(feature = "sqlite"))]
        assert!(
            error
                .to_string()
                .contains("sessions.sqlite_path requires the 'sqlite' cargo feature")
        );
    }

    #[test]
    fn runtime_traces_dir_round_trips_through_toml() {
        let parsed: HarnessConfig = toml::from_str(
            r#"
version = 1

[models.default]
provider = "openai"
model = "gpt-5"

[providers.openai]
api_key = "test-key"

[runtime]
traces_dir = "/tmp/halter/traces"
"#,
        )
        .expect("parse config");

        assert_eq!(
            parsed.runtime.traces_dir,
            Some(PathBuf::from("/tmp/halter/traces"))
        );
        parsed.validate().expect("config should validate");
    }

    #[test]
    fn runtime_traces_dir_rejects_empty_path() {
        let mut config = HarnessConfig::default();
        config.models.default = Some(ModelConfig {
            provider: ConfiguredProvider::OpenAi,
            model: "gpt-5".to_owned(),
            max_input_tokens: None,
            max_output_tokens: None,
            reasoning: None,
            tokens_per_minute: None,
        });
        config.providers.openai = Some(ProviderConfig {
            base_url: None,
            api_key: Some("test-key".to_owned()),
            headers: IndexMap::new(),
            temperature: None,
        });
        config.runtime.traces_dir = Some(PathBuf::from(""));

        let error = config.validate().expect_err("empty traces_dir should fail");
        assert!(
            error
                .to_string()
                .contains("runtime.traces_dir must not be empty"),
            "unexpected error: {error}"
        );
    }

    // AC2.8: the legacy key `allowed_loopback_services` deserializes into the
    // new `allowed_loopback` field via a serde alias.
    #[test]
    fn review_hook_runtime_ac2_8_loopback_alias_migrates() {
        let toml = r#"
enabled = true
allowed_hosts = []

[[allowed_loopback_services]]
host = "localhost"
port = 9090
"#;
        let parsed: NetworkPolicyConfig = toml::from_str(toml).expect("parse alias");
        assert!(parsed.enabled);
        assert_eq!(parsed.allowed_loopback.len(), 1);
        assert_eq!(parsed.allowed_loopback[0].host, "localhost");
        assert_eq!(parsed.allowed_loopback[0].port, Some(9090));
    }
}
