// pattern: Functional Core

use std::fmt;
use std::path::PathBuf;

use anyhow::Context;
use halter_protocol::{
    ApiKind, PanelIsolation, ProviderKind, PruneSignalThreshold, ReasoningEffort,
    SubagentEventForwarding,
};
use indexmap::IndexMap;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Built-in id for the required default model slot.
pub const DEFAULT_MODEL_ID: &str = "default";
/// Built-in id for the optional small-task model slot.
pub const SMALL_MODEL_ID: &str = "small";
/// Built-in id for the optional subagent model slot.
pub const SUBAGENT_MODEL_ID: &str = "subagent";

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(deny_unknown_fields)]
/// Top-level TOML configuration for a halter runtime.
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
    /// Validate cross-field requirements that serde cannot express.
    pub fn validate(&self) -> anyhow::Result<()> {
        if self.version != 1 {
            anyhow::bail!(
                "invalid configuration: unsupported version {}",
                self.version
            );
        }

        self.providers.validate()?;
        self.sessions.validate()?;

        self.models.validate()?;

        if self.policy.max_read_bytes == 0 {
            anyhow::bail!("invalid configuration: max_read_bytes must be greater than zero");
        }

        self.context.validate()?;
        self.runtime.validate()?;

        Ok(())
    }

    /// Required default model slot.
    pub fn default_slot(&self) -> anyhow::Result<&ModelSlot> {
        self.models
            .default
            .as_ref()
            .context("invalid configuration: [models.default] is required")
    }

    /// Optional subagent model slot override.
    #[must_use]
    pub fn subagent_slot(&self) -> Option<&ModelSlot> {
        self.models.subagent.as_ref()
    }

    /// Shared model-judge definition referenced by `"model_judge"` model slots.
    #[must_use]
    pub fn model_judge(&self) -> Option<&ModelJudgeConfig> {
        self.models.model_judge.as_ref()
    }

    /// Representative default leaf model (the inline model, or the
    /// model-judge default model when the default slot references
    /// `[models.model_judge]`).
    pub fn default_model(&self) -> anyhow::Result<&ModelConfig> {
        self.default_slot()?.primary(self.model_judge())
    }

    /// Representative subagent leaf model, if a concrete subagent slot is configured.
    pub fn subagent_model(&self) -> Option<&ModelConfig> {
        self.subagent_slot()
            .and_then(|slot| slot.primary(self.model_judge()).ok())
    }

    /// Optional small-task model override.
    #[must_use]
    pub fn small_model(&self) -> Option<&ModelConfig> {
        self.models.small.as_ref()
    }

    /// Distinct provider families referenced across all model slots, expanding
    /// model-judge slots into their leaf models. Order is deterministic and
    /// deduplicated.
    #[must_use]
    pub fn referenced_providers(&self) -> Vec<ConfiguredProvider> {
        let mut providers = Vec::new();
        let mut push = |provider: ConfiguredProvider| {
            if !providers.contains(&provider) {
                providers.push(provider);
            }
        };
        for slot in [self.models.default.as_ref(), self.models.subagent.as_ref()]
            .into_iter()
            .flatten()
        {
            match slot {
                ModelSlot::Inline(model) => push(model.provider),
                ModelSlot::Reference(ModelSlotRef::ModelJudge) => {
                    if let Some(model_judge) = self.model_judge() {
                        for model in model_judge.models() {
                            push(model.provider);
                        }
                    }
                }
                ModelSlot::Reference(ModelSlotRef::AutoResolve) => {}
            }
        }
        if let Some(small) = self.small_model() {
            push(small.provider);
        }
        providers
    }

    /// Provider config for a known provider family.
    #[must_use]
    pub fn provider_config(&self, provider: ConfiguredProvider) -> Option<&ProviderConfig> {
        self.providers.get(provider)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
#[serde(deny_unknown_fields)]
/// Provider-specific connection settings keyed by provider family.
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
    /// Return the configured provider block for a provider family.
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
/// Model slots used by the runtime.
pub struct ModelsConfig {
    #[serde(default)]
    pub default: Option<ModelSlot>,
    #[serde(default)]
    pub small: Option<ModelConfig>,
    #[serde(default)]
    pub subagent: Option<ModelSlot>,
    /// Shared definition referenced when a slot is set to `"model_judge"`.
    #[serde(default)]
    pub model_judge: Option<ModelJudgeConfig>,
}

impl ModelsConfig {
    fn validate(&self) -> anyhow::Result<()> {
        let default = self
            .default
            .as_ref()
            .context("invalid configuration: [models.default] is required")?;
        validate_model_slot("models.default", default, self.model_judge.as_ref())?;
        if let Some(subagent) = &self.subagent {
            validate_model_slot("models.subagent", subagent, self.model_judge.as_ref())?;
        }
        if let Some(small) = &self.small {
            validate_model_config("models.small", small)?;
        }
        if let Some(model_judge) = &self.model_judge {
            validate_model_judge_config("models.model_judge", model_judge)?;
        }
        Ok(())
    }
}

/// A model slot: either an inline model or a symbolic reference.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(untagged)]
pub enum ModelSlot {
    /// Reference to the shared `[models.model_judge]` definition via the bare
    /// string `"model_judge"`.
    Reference(ModelSlotRef),
    /// Inline model configuration (the historical form).
    Inline(ModelConfig),
}

impl ModelSlot {
    /// Whether this slot resolves through `[models.model_judge]`.
    #[must_use]
    pub fn is_model_judge(&self) -> bool {
        matches!(self, Self::Reference(ModelSlotRef::ModelJudge))
    }

    /// Whether this slot resolves to the parent session's active model.
    #[must_use]
    pub fn is_auto_resolve(&self) -> bool {
        matches!(self, Self::Reference(ModelSlotRef::AutoResolve))
    }

    /// Representative leaf model for the slot: the inline model, or the
    /// model-judge default model when the slot references `[models.model_judge]`.
    pub fn primary<'a>(
        &'a self,
        model_judge: Option<&'a ModelJudgeConfig>,
    ) -> anyhow::Result<&'a ModelConfig> {
        match self {
            Self::Inline(model) => Ok(model),
            Self::Reference(ModelSlotRef::ModelJudge) => model_judge
                .map(|model_judge| &model_judge.default)
                .context(
                    "invalid configuration: a model slot is set to \"model_judge\" but [models.model_judge] is not defined",
                ),
            Self::Reference(ModelSlotRef::AutoResolve) => anyhow::bail!(
                "invalid configuration: \"auto_resolve\" does not have a standalone leaf model"
            ),
        }
    }
}

/// Symbolic references usable in a model slot.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ModelSlotRef {
    /// Resolve the slot through the shared `[models.model_judge]` definition.
    ModelJudge,
    /// Resolve spawned subagents to the parent session's active model config.
    AutoResolve,
}

/// How a model-judge slot turns one decision into a panel-judged one.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ModelJudgeMode {
    /// Each panelist answers with a single message (one inference, no tools),
    /// and the panel/synthesis/default cycle runs on every model call within a
    /// turn. Cheap, fast, step-by-step second opinions. The historical default.
    #[default]
    OneShot,
    /// Each panelist runs a complete agentic turn (inference + tool loop) on the
    /// user's message once per turn, and the synthesis model judges the
    /// *outcomes* of those turns before handing guidance to the default model,
    /// which owns the real, user-visible execution. See [`PanelIsolation`] for
    /// how panelist tool execution is sandboxed.
    FullTurn,
}

/// Model-judge definition: a default model, a synthesis model, and the panel of
/// models whose responses are judged.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ModelJudgeConfig {
    /// Whether panelists answer in one shot or run full agentic turns.
    #[serde(default)]
    pub mode: ModelJudgeMode,
    /// Model that produces the final, user-visible response from the synthesis.
    pub default: ModelConfig,
    /// Model that ranks the panel responses and writes the synthesis message.
    pub synthesis: ModelConfig,
    /// Panel of models the user message is multiplexed to.
    #[serde(default)]
    pub panel: Vec<ModelConfig>,
    /// How FullTurn panelist sub-sessions are sandboxed. Ignored under
    /// [`ModelJudgeMode::OneShot`] (one-shot panelists never execute tools).
    #[serde(default)]
    pub panel_isolation: PanelIsolation,
}

impl ModelJudgeConfig {
    /// Iterate every leaf model referenced by this model-judge definition.
    pub fn models(&self) -> impl Iterator<Item = &ModelConfig> {
        std::iter::once(&self.default)
            .chain(std::iter::once(&self.synthesis))
            .chain(self.panel.iter())
    }
}

#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq, PartialOrd, Ord, Hash,
)]
/// Provider family accepted by user config.
pub enum ConfiguredProvider {
    #[serde(rename = "anthropic")]
    Anthropic,
    #[serde(rename = "openai")]
    OpenAi,
    #[serde(rename = "openrouter")]
    OpenRouter,
}

impl ConfiguredProvider {
    /// TOML section spelling for this provider.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Anthropic => "anthropic",
            Self::OpenAi => "openai",
            Self::OpenRouter => "openrouter",
        }
    }

    /// Protocol provider kind used after config resolution.
    #[must_use]
    pub const fn provider_kind(self) -> ProviderKind {
        match self {
            Self::Anthropic => ProviderKind::Anthropic,
            Self::OpenAi => ProviderKind::OpenAi,
            Self::OpenRouter => ProviderKind::OpenRouter,
        }
    }

    /// Default API kind used by this provider.
    #[must_use]
    pub const fn api_kind(self) -> ApiKind {
        match self {
            Self::Anthropic => ApiKind::AnthropicMessages,
            Self::OpenAi => ApiKind::OpenAiResponses,
            Self::OpenRouter => ApiKind::OpenAiResponses,
        }
    }

    /// Environment variable used as a fallback API key source.
    #[must_use]
    pub const fn api_key_env_var(self) -> &'static str {
        match self {
            Self::Anthropic => "ANTHROPIC_API_KEY",
            Self::OpenAi => "OPENAI_API_KEY",
            Self::OpenRouter => "OPENROUTER_API_KEY",
        }
    }

    /// Default upstream base URL.
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
/// Runtime connection settings for one provider.
pub struct ProviderConfig {
    #[serde(default)]
    pub base_url: Option<String>,
    /// Provider API key. For OpenAI, this is mutually exclusive with `oauth`.
    #[serde(default)]
    pub api_key: Option<String>,
    /// OpenAI OAuth credentials. Only accepted for `[providers.openai]`.
    /// When present, the provider uses `access_token` as the bearer token and
    /// routes `/v1/responses`, every path below that prefix, and
    /// `/chat/completions` through ChatGPT's Codex backend with top-level
    /// `instructions` and `store: false`.
    #[serde(default)]
    pub oauth: Option<OpenAiOAuthConfig>,
    /// Optional HTTP headers applied to every request the provider emits.
    /// Names collide case-insensitively; configured entries override any
    /// default or hardcoded provider header (Authorization, x-api-key, etc.).
    #[serde(default, skip_serializing_if = "IndexMap::is_empty")]
    pub headers: IndexMap<String, String>,
    /// Optional override for the sampling temperature. When unset, no
    /// temperature is sent to the provider. Must be in `0.0..=2.0`.
    #[serde(default)]
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// OpenAI OAuth credential bundle supplied by the user.
pub struct OpenAiOAuthConfig {
    /// Public OAuth client id that issued the token bundle.
    pub client_id: String,
    /// Bearer token sent to OpenAI OAuth provider requests.
    pub access_token: String,
    /// ID token retained with the bundle for caller-managed refresh/exchange flows.
    pub id_token: String,
    /// Refresh token retained with the bundle for caller-managed refresh flows.
    pub refresh_token: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// Model name plus provider and optional runtime limits.
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
    #[schemars(range(min = 1))]
    pub tokens_per_minute: Option<u64>,
}

#[derive(Debug, Clone, PartialEq)]
/// Provider settings after defaults, env fallbacks, and validation are applied.
pub struct ResolvedProviderConfig {
    pub provider: ConfiguredProvider,
    pub base_url: String,
    pub auth: ResolvedProviderAuth,
    /// Ordered list of user-configured headers. The runtime applies these
    /// over provider defaults using case-insensitive name matching.
    pub headers: Vec<(String, String)>,
    /// Sampling temperature forwarded to every request this provider emits.
    /// When unset, request bodies omit temperature and defer to the provider.
    pub temperature: Option<f32>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
/// Provider credential selected after config/env resolution.
pub enum ResolvedProviderAuth {
    /// Provider API key resolved from config or environment.
    ApiKey(String),
    /// OpenAI OAuth credentials resolved from `[providers.openai].oauth`.
    OpenAiOAuth(OpenAiOAuthConfig),
}

/// Resolve provider runtime settings from config plus an environment lookup.
///
/// Configured API keys win over environment variables. Empty strings are
/// treated as missing so accidental whitespace does not mask a fallback. For
/// OpenAI, configured OAuth credentials also win over the environment and are
/// mutually exclusive with a configured API key.
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
    let configured_oauth = configured.and_then(|config| config.oauth.as_ref());
    if configured_oauth.is_some() && provider != ConfiguredProvider::OpenAi {
        anyhow::bail!(
            "invalid configuration: providers.{provider}.oauth is only supported for providers.openai"
        );
    }
    if configured_api_key.is_some() && configured_oauth.is_some() {
        anyhow::bail!(
            "invalid configuration: configure either providers.openai.api_key or providers.openai.oauth, not both"
        );
    }
    if let Some(oauth) = configured_oauth {
        validate_openai_oauth_config(&format!("providers.{provider}.oauth"), oauth)?;
    }
    let env_api_key = if configured_api_key.is_some() || configured_oauth.is_some() {
        None
    } else {
        lookup_env(provider.api_key_env_var())?.and_then(|value| {
            let trimmed = value.trim();
            (!trimmed.is_empty()).then(|| trimmed.to_owned())
        })
    };
    let auth = match (configured_api_key, configured_oauth) {
        (Some(api_key), None) => ResolvedProviderAuth::ApiKey(api_key),
        (None, Some(oauth)) => ResolvedProviderAuth::OpenAiOAuth(trimmed_openai_oauth(oauth)),
        (None, None) => ResolvedProviderAuth::ApiKey(
            env_api_key.with_context(|| missing_provider_credentials_message(provider))?,
        ),
        (Some(_), Some(_)) => unreachable!("configured api_key and oauth checked above"),
    };

    let headers = configured
        .map(|config| {
            config
                .headers
                .iter()
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect()
        })
        .unwrap_or_default();

    let temperature = configured.and_then(|config| config.temperature);

    Ok(ResolvedProviderConfig {
        provider,
        base_url,
        auth,
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
    if provider.oauth.is_some() && name != ConfiguredProvider::OpenAi.as_str() {
        anyhow::bail!(
            "invalid configuration: providers.{name}.oauth is only supported for providers.openai"
        );
    }
    if provider
        .api_key
        .as_deref()
        .is_some_and(|value| !value.trim().is_empty())
        && provider.oauth.is_some()
    {
        anyhow::bail!(
            "invalid configuration: configure either providers.openai.api_key or providers.openai.oauth, not both"
        );
    }
    if let Some(oauth) = &provider.oauth {
        validate_openai_oauth_config(&format!("providers.{name}.oauth"), oauth)?;
    }
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

fn trimmed_openai_oauth(oauth: &OpenAiOAuthConfig) -> OpenAiOAuthConfig {
    OpenAiOAuthConfig {
        client_id: oauth.client_id.trim().to_owned(),
        access_token: oauth.access_token.trim().to_owned(),
        id_token: oauth.id_token.trim().to_owned(),
        refresh_token: oauth.refresh_token.trim().to_owned(),
    }
}

fn missing_provider_credentials_message(provider: ConfiguredProvider) -> String {
    if provider == ConfiguredProvider::OpenAi {
        return format!(
            "missing credentials for provider '{}': set [providers.{}].api_key, [providers.{}].oauth, or {}",
            provider,
            provider,
            provider,
            provider.api_key_env_var()
        );
    }

    format!(
        "missing api key for provider '{}': set [providers.{}].api_key or {}",
        provider,
        provider,
        provider.api_key_env_var()
    )
}

fn validate_openai_oauth_config(path: &str, oauth: &OpenAiOAuthConfig) -> anyhow::Result<()> {
    validate_required_string(&format!("{path}.client_id"), &oauth.client_id)?;
    validate_required_string(&format!("{path}.access_token"), &oauth.access_token)?;
    validate_required_string(&format!("{path}.id_token"), &oauth.id_token)?;
    validate_required_string(&format!("{path}.refresh_token"), &oauth.refresh_token)?;
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

fn validate_model_slot(
    path: &str,
    slot: &ModelSlot,
    model_judge: Option<&ModelJudgeConfig>,
) -> anyhow::Result<()> {
    match slot {
        ModelSlot::Inline(model) => validate_model_config(path, model),
        ModelSlot::Reference(ModelSlotRef::ModelJudge) => {
            if model_judge.is_none() {
                anyhow::bail!(
                    "invalid configuration: {path} is set to \"model_judge\" but [models.model_judge] is not defined"
                );
            }
            Ok(())
        }
        ModelSlot::Reference(ModelSlotRef::AutoResolve) => {
            if path != "models.subagent" {
                anyhow::bail!(
                    "invalid configuration: {path} is set to \"auto_resolve\" but \"auto_resolve\" is only valid for models.subagent"
                );
            }
            Ok(())
        }
    }
}

fn validate_model_judge_config(path: &str, model_judge: &ModelJudgeConfig) -> anyhow::Result<()> {
    validate_model_config(&format!("{path}.default"), &model_judge.default)?;
    validate_model_config(&format!("{path}.synthesis"), &model_judge.synthesis)?;
    if model_judge.panel.is_empty() {
        anyhow::bail!("invalid configuration: {path}.panel must not be empty");
    }
    for (index, panelist) in model_judge.panel.iter().enumerate() {
        validate_model_config(&format!("{path}.panel[{index}]"), panelist)?;
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
    validate_optional_positive_u64(
        &format!("{path}.tokens_per_minute"),
        model.tokens_per_minute,
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

fn validate_optional_positive_u64(path: &str, value: Option<u64>) -> anyhow::Result<()> {
    if matches!(value, Some(0)) {
        anyhow::bail!("invalid configuration: {path} must be greater than zero");
    }
    Ok(())
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
/// Resource search roots.
pub struct ResourcesConfig {
    #[serde(default)]
    pub skills: SearchRoots,
    #[serde(default)]
    pub plugins: SearchRoots,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
/// Ordered filesystem roots searched by a loader.
pub struct SearchRoots {
    #[serde(default)]
    pub roots: Vec<PathBuf>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
/// Which built-in system prompt a session starts from when no explicit
/// `system_prompt` override is set.
pub enum SystemPromptPreset {
    /// The general-purpose agent prompt (the default).
    #[default]
    General,
    /// The batteries-included coding-agent prompt — a quick on-ramp for SDK
    /// users who want a working coding agent.
    Coding,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
/// Prompt-related config.
pub struct PromptsConfig {
    /// Which built-in system prompt to start from. Ignored when
    /// `system_prompt` is set.
    #[serde(default)]
    pub preset: SystemPromptPreset,
    /// Full override of the session system prompt. Wins over `preset`. When
    /// unset, the built-in prompt named by `preset` is used.
    #[serde(default)]
    pub system_prompt: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// Context window and compaction thresholds.
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
/// Built-in tool selection.
pub struct ToolsConfig {
    #[serde(default)]
    pub enabled: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// Policy settings applied by built-in tools.
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
/// Shell tool policy.
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
        "true".to_owned(),
        "cd".to_owned(),
    ]
}

const fn default_shell_timeout_secs() -> u64 {
    30
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Default)]
#[serde(deny_unknown_fields)]
/// Network tool policy.
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
/// One loopback host/port exception.
pub struct LoopbackAllowConfig {
    pub host: String,
    #[serde(default)]
    pub port: Option<u16>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
/// Session persistence backend selected by config.
pub enum SessionBackend {
    #[default]
    Memory,
    #[cfg(feature = "sqlite")]
    Sqlite,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// Session persistence settings.
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

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
/// Runtime process and tracing settings.
pub struct RuntimeConfig {
    #[serde(default)]
    pub working_dir: Option<PathBuf>,
    /// Directory where full per-session traces are written as `<session_id>.txt`
    /// JSONL files. Each file begins with a header line describing the session
    /// blueprint and is followed by one JSON-encoded `SessionEvent` per line —
    /// enough to debug a run and to rebuild session state offline.
    #[serde(default)]
    pub traces_dir: Option<PathBuf>,
    #[serde(default)]
    pub subagent_event_forwarding: SubagentEventForwarding,
    #[serde(default = "default_subagent_event_forwarding_cap")]
    pub subagent_event_forwarding_cap: u64,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            working_dir: None,
            traces_dir: None,
            subagent_event_forwarding: SubagentEventForwarding::Off,
            subagent_event_forwarding_cap: default_subagent_event_forwarding_cap(),
        }
    }
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

const fn default_subagent_event_forwarding_cap() -> u64 {
    100_000
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
    fn shell_policy_defaults_allow_basic_noop_and_directory_change_commands() {
        let shell = ShellPolicyConfig::default();

        assert!(shell.allow.iter().any(|command| command == "true"));
        assert!(shell.allow.iter().any(|command| command == "cd"));
    }

    #[test]
    fn inline_model_slot_round_trips_through_toml() {
        let parsed: HarnessConfig = toml::from_str(
            r#"
version = 1

[models.default]
provider = "openai"
model = "gpt-5"

[providers.openai]
api_key = "test-key"
"#,
        )
        .expect("parse config");

        assert!(matches!(parsed.models.default, Some(ModelSlot::Inline(_))));
        assert_eq!(
            parsed.default_model().expect("default model").model,
            "gpt-5"
        );
        parsed.validate().expect("config should validate");
    }

    #[test]
    fn model_judge_model_slot_round_trips_through_toml() {
        let parsed: HarnessConfig = toml::from_str(
            r#"
version = 1

[models]
default = "model_judge"
subagent = "model_judge"

[models.model_judge.default]
provider = "anthropic"
model = "claude-default"

[models.model_judge.synthesis]
provider = "openai"
model = "synthesis-5"

[[models.model_judge.panel]]
provider = "openai"
model = "panel-a"

[[models.model_judge.panel]]
provider = "openrouter"
model = "panel-b"

[providers.openai]
api_key = "openai-key"

[providers.anthropic]
api_key = "anthropic-key"

[providers.openrouter]
api_key = "openrouter-key"
"#,
        )
        .expect("parse config");

        let default = parsed.default_slot().expect("default slot");
        assert!(default.is_model_judge());
        assert!(
            parsed
                .subagent_slot()
                .is_some_and(ModelSlot::is_model_judge)
        );

        let model_judge = parsed.model_judge().expect("model_judge config");
        assert_eq!(model_judge.default.model, "claude-default");
        assert_eq!(model_judge.synthesis.model, "synthesis-5");
        assert_eq!(model_judge.panel.len(), 2);
        // Omitted mode/panel_isolation fall back to the backward-compatible
        // OneShot + ReadOnly defaults.
        assert_eq!(model_judge.mode, ModelJudgeMode::OneShot);
        assert_eq!(model_judge.panel_isolation, PanelIsolation::ReadOnly);

        // Representative leaf model is the model-judge default model.
        assert_eq!(
            parsed.default_model().expect("default model").model,
            "claude-default"
        );
        // Every leaf family is surfaced for credential resolution.
        assert_eq!(
            parsed.referenced_providers(),
            vec![
                ConfiguredProvider::Anthropic,
                ConfiguredProvider::OpenAi,
                ConfiguredProvider::OpenRouter,
            ]
        );

        parsed.validate().expect("config should validate");
    }

    #[test]
    fn auto_resolve_subagent_slot_round_trips_through_toml() {
        let parsed: HarnessConfig = toml::from_str(
            r#"
version = 1

[models]
subagent = "auto_resolve"

[models.default]
provider = "openai"
model = "gpt-default"

[providers.openai]
api_key = "openai-key"
"#,
        )
        .expect("parse config");

        let subagent = parsed.subagent_slot().expect("subagent slot");
        assert!(subagent.is_auto_resolve());
        assert!(parsed.subagent_model().is_none());
        assert_eq!(
            parsed.referenced_providers(),
            vec![ConfiguredProvider::OpenAi]
        );

        parsed.validate().expect("config should validate");
    }

    #[test]
    fn auto_resolve_is_rejected_for_default_slot() {
        let parsed: HarnessConfig = toml::from_str(
            r#"
version = 1

[models]
default = "auto_resolve"
"#,
        )
        .expect("parse config");

        let error = parsed
            .validate()
            .expect_err("default auto_resolve should fail");

        assert!(error.to_string().contains("models.default"));
        assert!(error.to_string().contains("models.subagent"));
    }

    #[test]
    fn full_turn_model_judge_round_trips_through_toml() {
        let parsed: HarnessConfig = toml::from_str(
            r#"
version = 1

[models]
default = "model_judge"

[models.model_judge]
mode = "full_turn"
panel_isolation = "worktree"

[models.model_judge.default]
provider = "anthropic"
model = "claude-default"

[models.model_judge.synthesis]
provider = "openai"
model = "synthesis-5"

[[models.model_judge.panel]]
provider = "openai"
model = "panel-a"

[providers.openai]
api_key = "openai-key"

[providers.anthropic]
api_key = "anthropic-key"
"#,
        )
        .expect("parse config");

        let model_judge = parsed.model_judge().expect("model_judge config");
        assert_eq!(model_judge.mode, ModelJudgeMode::FullTurn);
        assert_eq!(model_judge.panel_isolation, PanelIsolation::Worktree);
        parsed.validate().expect("config should validate");
    }

    #[test]
    fn model_judge_reference_requires_model_judge_block() {
        let parsed: HarnessConfig = toml::from_str(
            r#"
version = 1

[models]
default = "model_judge"

[providers.openai]
api_key = "test-key"
"#,
        )
        .expect("parse config");

        let error = parsed
            .validate()
            .expect_err("missing model_judge block should fail");
        assert!(
            error.to_string().contains(
                "models.default is set to \"model_judge\" but [models.model_judge] is not defined"
            ),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn model_judge_requires_non_empty_panel() {
        let parsed: HarnessConfig = toml::from_str(
            r#"
version = 1

[models]
default = "model_judge"

[models.model_judge.default]
provider = "openai"
model = "gpt-5"

[models.model_judge.synthesis]
provider = "openai"
model = "synthesis-5"

[providers.openai]
api_key = "test-key"
"#,
        )
        .expect("parse config");

        let error = parsed.validate().expect_err("empty panel should fail");
        assert!(
            error
                .to_string()
                .contains("models.model_judge.panel must not be empty"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn model_config_validates_tokens_per_minute() {
        let cases = [
            ("unset", None, None),
            ("one", Some(1), None),
            ("default", Some(500_000), None),
            (
                "zero",
                Some(0),
                Some("models.default.tokens_per_minute must be greater than zero"),
            ),
        ];

        for (name, tokens_per_minute, want_error) in cases {
            let model = ModelConfig {
                provider: ConfiguredProvider::OpenAi,
                model: "gpt-5".to_owned(),
                max_input_tokens: None,
                max_output_tokens: None,
                reasoning: None,
                tokens_per_minute,
            };

            let result = validate_model_config("models.default", &model);

            match want_error {
                Some(want_error) => {
                    let error = result.expect_err("validation should fail");
                    assert!(
                        error.to_string().contains(want_error),
                        "{name}: unexpected error: {error}"
                    );
                }
                None => result.expect("validation should pass"),
            }
        }
    }

    #[test]
    fn provider_resolution_uses_configured_api_key_before_env() {
        let resolved = resolve_provider_runtime_config(
            ConfiguredProvider::OpenAi,
            Some(&ProviderConfig {
                base_url: Some("https://proxy.example.com".to_owned()),
                api_key: Some("configured-key".to_owned()),
                ..ProviderConfig::default()
            }),
            |_| Ok(Some("env-key".to_owned())),
        )
        .expect("resolve provider");

        assert_eq!(resolved.base_url, "https://proxy.example.com");
        assert_eq!(
            resolved.auth,
            ResolvedProviderAuth::ApiKey("configured-key".to_owned())
        );
        assert_eq!(resolved.temperature, None);
    }

    #[test]
    fn provider_resolution_uses_configured_openai_oauth_before_env() {
        let resolved = resolve_provider_runtime_config(
            ConfiguredProvider::OpenAi,
            Some(&ProviderConfig {
                base_url: Some("https://proxy.example.com".to_owned()),
                oauth: Some(OpenAiOAuthConfig {
                    client_id: " client ".to_owned(),
                    access_token: " access-token ".to_owned(),
                    id_token: " id-token ".to_owned(),
                    refresh_token: " refresh-token ".to_owned(),
                }),
                ..ProviderConfig::default()
            }),
            |_| Ok(Some("env-key".to_owned())),
        )
        .expect("resolve provider");

        assert_eq!(resolved.base_url, "https://proxy.example.com");
        assert_eq!(
            resolved.auth,
            ResolvedProviderAuth::OpenAiOAuth(OpenAiOAuthConfig {
                client_id: "client".to_owned(),
                access_token: "access-token".to_owned(),
                id_token: "id-token".to_owned(),
                refresh_token: "refresh-token".to_owned(),
            })
        );
    }

    #[test]
    fn provider_resolution_rejects_openai_api_key_and_oauth() {
        let error = resolve_provider_runtime_config(
            ConfiguredProvider::OpenAi,
            Some(&ProviderConfig {
                api_key: Some("configured-key".to_owned()),
                oauth: Some(openai_oauth_config()),
                ..ProviderConfig::default()
            }),
            |_| Ok(None),
        )
        .expect_err("conflicting credentials should fail");

        assert!(
            error
                .to_string()
                .contains("configure either providers.openai.api_key or providers.openai.oauth")
        );
    }

    #[test]
    fn provider_resolution_rejects_oauth_for_non_openai_provider() {
        let error = resolve_provider_runtime_config(
            ConfiguredProvider::OpenRouter,
            Some(&ProviderConfig {
                oauth: Some(openai_oauth_config()),
                ..ProviderConfig::default()
            }),
            |_| Ok(None),
        )
        .expect_err("unsupported oauth should fail");

        assert!(
            error
                .to_string()
                .contains("providers.openrouter.oauth is only supported for providers.openai")
        );
    }

    #[test]
    fn provider_resolution_rejects_empty_openai_oauth_field() {
        let error = resolve_provider_runtime_config(
            ConfiguredProvider::OpenAi,
            Some(&ProviderConfig {
                oauth: Some(OpenAiOAuthConfig {
                    access_token: " ".to_owned(),
                    ..openai_oauth_config()
                }),
                ..ProviderConfig::default()
            }),
            |_| Ok(Some("env-key".to_owned())),
        )
        .expect_err("empty OAuth access token should fail");

        assert!(
            error
                .to_string()
                .contains("providers.openai.oauth.access_token must not be empty")
        );
    }

    #[test]
    fn provider_resolution_applies_configured_temperature_override() {
        let resolved = resolve_provider_runtime_config(
            ConfiguredProvider::OpenRouter,
            Some(&ProviderConfig {
                api_key: Some("configured-key".to_owned()),
                temperature: Some(0.2),
                ..ProviderConfig::default()
            }),
            |_| Ok(None),
        )
        .expect("resolve provider");

        assert_eq!(resolved.temperature, Some(0.2));
    }

    #[test]
    fn provider_resolution_leaves_temperature_unset_when_unconfigured() {
        let resolved = resolve_provider_runtime_config(ConfiguredProvider::Anthropic, None, |_| {
            Ok(Some("env-key".to_owned()))
        })
        .expect("resolve provider");

        assert_eq!(resolved.temperature, None);
    }

    #[test]
    fn provider_config_rejects_out_of_range_temperature() {
        let error = validate_provider_config(
            "openrouter",
            &ProviderConfig {
                api_key: Some("configured-key".to_owned()),
                temperature: Some(2.5),
                ..ProviderConfig::default()
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
                api_key: Some("configured-key".to_owned()),
                temperature: Some(f32::NAN),
                ..ProviderConfig::default()
            },
        )
        .expect_err("validation should fail");

        assert!(error.to_string().contains("temperature"));
    }

    #[test]
    fn provider_config_accepts_openai_oauth_without_api_key() {
        validate_provider_config(
            "openai",
            &ProviderConfig {
                oauth: Some(openai_oauth_config()),
                ..ProviderConfig::default()
            },
        )
        .expect("valid oauth config should pass");
    }

    #[test]
    fn provider_config_rejects_openai_api_key_and_oauth() {
        let error = validate_provider_config(
            "openai",
            &ProviderConfig {
                api_key: Some("configured-key".to_owned()),
                oauth: Some(openai_oauth_config()),
                ..ProviderConfig::default()
            },
        )
        .expect_err("conflicting credentials should fail");

        assert!(
            error
                .to_string()
                .contains("configure either providers.openai.api_key or providers.openai.oauth")
        );
    }

    #[test]
    fn provider_config_rejects_empty_openai_oauth_fields() {
        let cases = [
            (
                "client_id",
                OpenAiOAuthConfig {
                    client_id: " ".to_owned(),
                    ..openai_oauth_config()
                },
            ),
            (
                "access_token",
                OpenAiOAuthConfig {
                    access_token: " ".to_owned(),
                    ..openai_oauth_config()
                },
            ),
            (
                "id_token",
                OpenAiOAuthConfig {
                    id_token: " ".to_owned(),
                    ..openai_oauth_config()
                },
            ),
            (
                "refresh_token",
                OpenAiOAuthConfig {
                    refresh_token: " ".to_owned(),
                    ..openai_oauth_config()
                },
            ),
        ];

        for (field, oauth) in cases {
            let error = validate_provider_config(
                "openai",
                &ProviderConfig {
                    oauth: Some(oauth),
                    ..ProviderConfig::default()
                },
            )
            .expect_err("empty OAuth field should fail");

            assert!(
                error
                    .to_string()
                    .contains(&format!("providers.openai.oauth.{field} must not be empty")),
                "{field}: unexpected error: {error}"
            );
        }
    }

    #[test]
    fn provider_config_rejects_oauth_for_non_openai_provider() {
        let error = validate_provider_config(
            "anthropic",
            &ProviderConfig {
                oauth: Some(openai_oauth_config()),
                ..ProviderConfig::default()
            },
        )
        .expect_err("unsupported oauth should fail");

        assert!(
            error
                .to_string()
                .contains("providers.anthropic.oauth is only supported for providers.openai")
        );
    }

    #[test]
    fn provider_resolution_requires_api_key_or_oauth_for_openai() {
        let error = resolve_provider_runtime_config(ConfiguredProvider::OpenAi, None, |_| Ok(None))
            .expect_err("resolution should fail");

        assert!(error.to_string().contains("[providers.openai].oauth"));
    }

    #[test]
    fn provider_resolution_requires_api_key_for_selected_non_openai_provider() {
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
        config.models.default = Some(ModelSlot::Inline(ModelConfig {
            provider: ConfiguredProvider::OpenAi,
            model: "gpt-5".to_owned(),
            max_input_tokens: None,
            max_output_tokens: None,
            reasoning: None,
            tokens_per_minute: None,
        }));
        config.providers.openai = Some(ProviderConfig {
            api_key: Some("test-key".to_owned()),
            ..ProviderConfig::default()
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
    fn runtime_subagent_event_forwarding_round_trips_through_toml() {
        let parsed: HarnessConfig = toml::from_str(
            r#"
version = 1

[models.default]
provider = "openai"
model = "gpt-5"

[providers.openai]
api_key = "test-key"

[runtime]
subagent_event_forwarding = "all"
subagent_event_forwarding_cap = 42
"#,
        )
        .expect("parse config");

        assert_eq!(
            parsed.runtime.subagent_event_forwarding,
            SubagentEventForwarding::All
        );
        assert_eq!(parsed.runtime.subagent_event_forwarding_cap, 42);
        parsed.validate().expect("config should validate");
    }

    #[test]
    fn runtime_subagent_event_forwarding_defaults_to_off_with_cap() {
        let runtime = RuntimeConfig::default();

        assert_eq!(
            runtime.subagent_event_forwarding,
            SubagentEventForwarding::Off
        );
        assert_eq!(runtime.subagent_event_forwarding_cap, 100_000);
    }

    #[test]
    fn runtime_traces_dir_rejects_empty_path() {
        let mut config = HarnessConfig::default();
        config.models.default = Some(ModelSlot::Inline(ModelConfig {
            provider: ConfiguredProvider::OpenAi,
            model: "gpt-5".to_owned(),
            max_input_tokens: None,
            max_output_tokens: None,
            reasoning: None,
            tokens_per_minute: None,
        }));
        config.providers.openai = Some(ProviderConfig {
            api_key: Some("test-key".to_owned()),
            ..ProviderConfig::default()
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

    fn openai_oauth_config() -> OpenAiOAuthConfig {
        OpenAiOAuthConfig {
            client_id: "client".to_owned(),
            access_token: "access-token".to_owned(),
            id_token: "id-token".to_owned(),
            refresh_token: "refresh-token".to_owned(),
        }
    }
}
