// pattern: Functional Core

use std::fmt;
use std::path::PathBuf;

use anyhow::Context;
use halter_protocol::{ApiKind, ProviderKind, ReasoningEffort};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

pub const DEFAULT_MODEL_ID: &str = "default";
pub const SUBAGENT_MODEL_ID: &str = "subagent";

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
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

        let model = self.default_model()?;
        validate_model_config("models.default", model)?;
        if let Some(model) = self.subagent_model() {
            validate_model_config("models.subagent", model)?;
        }

        if self.policy.max_tool_output_bytes == 0 {
            anyhow::bail!("invalid configuration: max_tool_output_bytes must be greater than zero");
        }

        if self.policy.max_read_bytes == 0 {
            anyhow::bail!("invalid configuration: max_read_bytes must be greater than zero");
        }

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
    pub fn provider_config(&self, provider: ConfiguredProvider) -> Option<&ProviderConfig> {
        self.providers.get(provider)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
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

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
pub struct ProviderConfig {
    #[serde(default)]
    pub base_url: Option<String>,
    #[serde(default)]
    pub api_key: Option<String>,
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
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedProviderConfig {
    pub provider: ConfiguredProvider,
    pub base_url: String,
    pub api_key: String,
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

    Ok(ResolvedProviderConfig {
        provider,
        base_url,
        api_key,
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
    #[serde(default = "default_max_context_messages")]
    pub max_context_messages: usize,
}

impl Default for ContextConfig {
    fn default() -> Self {
        Self {
            max_context_messages: default_max_context_messages(),
        }
    }
}

const fn default_max_context_messages() -> usize {
    24
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
    #[serde(default = "default_max_tool_output_bytes")]
    pub max_tool_output_bytes: usize,
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
            max_tool_output_bytes: default_max_tool_output_bytes(),
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

const fn default_max_tool_output_bytes() -> usize {
    262_144
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
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SessionBackend {
    Memory,
    FlatFile,
    Sqlite,
    Postgres,
}

impl Default for SessionBackend {
    fn default() -> Self {
        Self::Memory
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SessionsConfig {
    #[serde(default)]
    pub backend: SessionBackend,
}

impl Default for SessionsConfig {
    fn default() -> Self {
        Self {
            backend: SessionBackend::Memory,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
pub struct RuntimeConfig {
    #[serde(default)]
    pub working_dir: Option<PathBuf>,
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
            }),
            |_| Ok(Some("env-key".to_owned())),
        )
        .expect("resolve provider");

        assert_eq!(resolved.base_url, "https://proxy.example.com");
        assert_eq!(resolved.api_key, "configured-key");
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
        };

        assert_eq!(model.provider.api_kind(), ApiKind::OpenAiResponses);
    }
}
