// pattern: Imperative Shell

use std::collections::HashMap;
use std::env;
use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use halter_config::{
    ConfiguredProvider, DEFAULT_MODEL_ID, HarnessConfig, ModelConfig, ModelJudgeConfig, ModelSlot,
    ModelSlotRef, OpenAiOAuthConfig, PolicyConfig, PromptsConfig, ResolvedProviderAuth,
    ResolvedProviderConfig, SMALL_MODEL_ID, SUBAGENT_MODEL_ID, SessionBackend, SessionsConfig,
    SystemPromptPreset, expand_path, load_path, resolve_provider_runtime_config,
};
use halter_hooks::{Hook, Hooks, RegisteredHookPriority, RegisteredHooks};
use halter_protocol::{
    HookWarning, ModelId, ModelRole, ProviderName, PromptSegmentKind, ResolvedModel,
    ResourceSnapshot,
};
use halter_providers::{
    AnthropicProvider, ModelJudgeMember, ModelJudgeProvider, ModelRegistry, OpenAiOAuthCredentials,
    OpenAiProvider, OpenRouterProvider, Provider,
};
use halter_runtime::{
    DefaultContextManager, DefaultPromptAssembler, EventBus, HalterSession, ResourceHandle,
    RuntimeServices, SessionInit, SessionRuntime, TraceRecorder,
};
use halter_session::{InMemorySessionStore, SessionStore};
use halter_tools::{
    DefaultToolPolicy, LoopbackAllow, PathLockMap, PolicySettings, Tool, ToolRuntime,
    ToolSessionStore, register_builtin_tools, register_subagent_tools,
};
use tracing::{debug, info};

use crate::{CompiledResources, LoadedPlugin, LoadedSkill, ResourceCompiler};

#[cfg(feature = "sqlite")]
use halter_session::SqliteSessionStore;

#[derive(Default)]
/// Builder for assembling a [`Halter`] runtime from config, resources, tools, and stores.
pub struct HalterBuilder {
    config: HarnessConfig,
    resource_snapshot: Option<ResourceSnapshot>,
    resource_hooks: Option<Arc<Hooks>>,
    resource_hook_warnings: Vec<HookWarning>,
    registered_hooks: RegisteredHooks,
    loaded_skills: Vec<LoadedSkill>,
    loaded_plugins: Vec<LoadedPlugin>,
    tools: Vec<Arc<dyn Tool>>,
    session_store: Option<Arc<dyn SessionStore>>,
}

impl HalterBuilder {
    /// Start a builder with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Replace the default harness configuration.
    #[must_use]
    pub fn with_config(mut self, config: HarnessConfig) -> Self {
        self.config = config;
        self
    }

    /// Provide a precompiled resource snapshot without hook metadata.
    #[must_use]
    pub fn with_resource_snapshot(mut self, snapshot: ResourceSnapshot) -> Self {
        self.resource_snapshot = Some(snapshot);
        self
    }

    /// Provide resources compiled by [`ResourceCompiler`].
    #[must_use]
    pub fn with_compiled_resources(mut self, resources: CompiledResources) -> Self {
        self.resource_snapshot = Some(resources.snapshot);
        self.resource_hooks = Some(resources.hooks);
        self.resource_hook_warnings = resources.hook_warnings;
        self
    }

    /// Provide loaded skills to compile during [`HalterBuilder::build`].
    #[must_use]
    pub fn with_loaded_skills(mut self, skills: Vec<LoadedSkill>) -> Self {
        self.loaded_skills = skills;
        self
    }

    /// Provide loaded plugins to compile during [`HalterBuilder::build`].
    #[must_use]
    pub fn with_loaded_plugins(mut self, plugins: Vec<LoadedPlugin>) -> Self {
        self.loaded_plugins = plugins;
        self
    }

    /// Register an SDK hook that runs after plugin-file hooks.
    #[must_use]
    pub fn with_plugin_hook(mut self, plugin_id: halter_protocol::PluginId, hook: Hook) -> Self {
        self.registered_hooks
            .register(plugin_id, RegisteredHookPriority::AfterPlugins, hook);
        self
    }

    /// Register an SDK hook with explicit priority relative to plugin-file hooks.
    #[must_use]
    pub fn with_plugin_hook_priority(
        mut self,
        plugin_id: halter_protocol::PluginId,
        priority: RegisteredHookPriority,
        hook: Hook,
    ) -> Self {
        self.registered_hooks.register(plugin_id, priority, hook);
        self
    }

    /// Add a custom tool to the runtime.
    #[must_use]
    pub fn with_tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Use a custom session store instead of the configured built-in backend.
    #[must_use]
    pub fn with_session_store(mut self, store: Arc<dyn SessionStore>) -> Self {
        self.session_store = Some(store);
        self
    }

    /// Validate configuration, register tools/providers/hooks, and build the runtime.
    pub async fn build(self) -> anyhow::Result<Halter> {
        let HalterBuilder {
            config,
            resource_snapshot,
            resource_hooks,
            resource_hook_warnings,
            registered_hooks,
            loaded_skills,
            loaded_plugins,
            tools: custom_tools,
            session_store,
        } = self;
        debug!("validating halter builder config");
        config.validate()?;
        registered_hooks.validate()?;

        if resource_snapshot.is_some() && (!loaded_skills.is_empty() || !loaded_plugins.is_empty())
        {
            anyhow::bail!(
                "failed to build halter runtime: cannot combine a prebuilt resource snapshot with loaded skills/plugins"
            );
        }

        let compiled_resources = if resource_snapshot.is_none()
            && (!loaded_skills.is_empty() || !loaded_plugins.is_empty())
        {
            Some(
                ResourceCompiler::from_config(&config)
                    .with_loaded_skills(loaded_skills)
                    .with_loaded_plugins(loaded_plugins)
                    .compile()
                    .await?,
            )
        } else {
            None
        };

        let snapshot = resource_snapshot
            .or_else(|| compiled_resources.as_ref().map(|resources| resources.snapshot.clone()))
            .with_context(|| {
                "failed to build halter runtime: missing resource snapshot; use Halter::from_config_file or HalterBuilder::with_resource_snapshot"
            })?;
        let hooks = resource_hooks
            .or_else(|| {
                compiled_resources
                    .as_ref()
                    .map(|resources| resources.hooks.clone())
            })
            .unwrap_or_else(|| Arc::new(Hooks::default()));
        let hook_warnings = compiled_resources
            .as_ref()
            .map_or(resource_hook_warnings, |resources| {
                resources.hook_warnings.clone()
            });

        let models = Arc::new(build_model_registry(&config)?);
        let tools = Arc::new(ToolRuntime::new());
        register_builtin_tools(&tools, &config.tools.enabled);
        for tool in custom_tools {
            tools.register(tool);
        }

        let policy = Arc::new(DefaultToolPolicy::new(policy_from_config(&config.policy)));
        let session_backend = session_store
            .as_ref()
            .map(|_| "custom".to_owned())
            .unwrap_or_else(|| describe_session_backend(&config.sessions).to_owned());
        let sessions = match session_store {
            Some(store) => store,
            None => build_session_store(&config.sessions)?,
        };
        let trace_recorder = config
            .runtime
            .traces_dir
            .as_ref()
            .map(|dir| TraceRecorder::open(expand_path(dir)).map(Arc::new))
            .transpose()?;
        let services = Arc::new(RuntimeServices {
            resources: Arc::new(ResourceHandle::new(snapshot, hooks, hook_warnings)),
            registered_hooks: Arc::new(registered_hooks),
            session_hook_store: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            models,
            tools,
            path_locks: Arc::new(PathLockMap::default()),
            tool_sessions: Arc::new(ToolSessionStore::default()),
            sessions,
            policy: policy.clone(),
            prompt_assembler: Arc::new(DefaultPromptAssembler),
            context_manager: Arc::new(DefaultContextManager::new(
                config.context.compaction_threshold,
                config.context.pre_compaction_target,
                config.context.prune_signal_threshold,
            )),
            event_bus: Arc::new(EventBus::default()),
            parent_streams: Arc::new(halter_runtime::ParentStreamRegistry::default()),
            turn_registry: Arc::new(halter_runtime::TurnRegistry::new()),
            subagent_event_forwarding: config.runtime.subagent_event_forwarding,
            subagent_event_forwarding_cap: config.runtime.subagent_event_forwarding_cap,
            shell_timeout_secs: config.policy.shell.timeout_secs,
            trace_recorder,
        });
        let runtime = SessionRuntime::new(services.clone());
        register_subagent_tools(
            &services.tools,
            runtime.subagent_control(),
            &config.tools.enabled,
            services.resources.snapshot().as_ref(),
            &services
                .models
                .model_ids()
                .into_iter()
                .map(|model_id| model_id.0)
                .collect::<Vec<_>>(),
        );
        let default_model = config.default_model()?;
        let subagent_model = config.subagent_model().unwrap_or(default_model);
        info!(
            default_provider = %default_model.provider,
            default_model = %default_model.model,
            subagent_provider = %subagent_model.provider,
            subagent_model = %subagent_model.model,
            session_backend,
            tool_count = services.tools.specs().len(),
            snapshot_revision = %services.resources.snapshot().revision,
            "built halter runtime"
        );

        Ok(Halter { config, runtime })
    }
}

#[derive(Clone)]
/// High-level SDK handle for creating and managing halter sessions.
pub struct Halter {
    config: HarnessConfig,
    runtime: SessionRuntime,
}

impl Halter {
    /// Start a [`HalterBuilder`].
    #[must_use]
    pub fn builder() -> HalterBuilder {
        HalterBuilder::default()
    }

    /// Load config and resources from a TOML file, then build a harness.
    pub async fn from_config_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        debug!(path = %path.as_ref().display(), "building halter from config file");
        let config = load_path(path).await?;
        let resources = ResourceCompiler::from_config(&config).compile().await?;
        Self::from_compiled_resources(config, resources).await
    }

    /// Build from config and a resource snapshot.
    pub async fn from_config(
        config: HarnessConfig,
        snapshot: ResourceSnapshot,
    ) -> anyhow::Result<Self> {
        HalterBuilder::default()
            .with_config(config)
            .with_resource_snapshot(snapshot)
            .build()
            .await
    }

    /// Build from config and compiled resources, including hooks and warnings.
    pub async fn from_compiled_resources(
        config: HarnessConfig,
        resources: CompiledResources,
    ) -> anyhow::Result<Self> {
        HalterBuilder::default()
            .with_config(config)
            .with_compiled_resources(resources)
            .build()
            .await
    }

    /// Create a new session.
    ///
    /// The session's system prompt is resolved from `[prompts]` config unless
    /// the caller installed an explicit one on `init` (see
    /// [`SessionInit::with_system_prompt`]). Precedence, most specific first:
    /// an explicit per-session prompt > `prompts.system_prompt` >
    /// `prompts.preset` > the built-in general-purpose default.
    pub async fn new_session(&self, init: SessionInit) -> anyhow::Result<HalterSession> {
        let init = apply_prompt_config(&self.config.prompts, init);
        self.runtime.new_session(init).await
    }

    /// Replace the live resource snapshot and hook registry for future work.
    pub fn replace_resources(&self, resources: CompiledResources) {
        self.runtime.replace_resources(
            resources.snapshot,
            resources.hooks,
            resources.hook_warnings,
        );
    }

    /// Borrow the underlying session runtime.
    #[must_use]
    pub fn runtime(&self) -> &SessionRuntime {
        &self.runtime
    }

    /// Borrow the effective harness configuration.
    #[must_use]
    pub fn config(&self) -> &HarnessConfig {
        &self.config
    }

    /// Drain all in-flight turns and refuse new submissions. Bounded by
    /// `drain` — tasks still running when the deadline elapses are
    /// aborted via `JoinHandle::abort`.
    ///
    /// Wire this into your process-level signal handler (e.g.
    /// `tokio::signal::ctrl_c`) so that Ctrl-C does not orphan
    /// half-committed turns.
    pub async fn shutdown(&self, drain: std::time::Duration) -> halter_runtime::ShutdownReport {
        self.runtime.shutdown(drain).await
    }
}

/// Resolve the system prompt selected by `[prompts]` config, or `None` when
/// config selects the built-in general default (so the seed needs no change).
/// An explicit `system_prompt` override wins over `preset`.
fn configured_system_prompt(prompts: &PromptsConfig) -> Option<String> {
    if let Some(custom) = prompts
        .system_prompt
        .as_deref()
        .map(str::trim)
        .filter(|text| !text.is_empty())
    {
        return Some(custom.to_owned());
    }
    match prompts.preset {
        SystemPromptPreset::General => None,
        SystemPromptPreset::Coding => Some(halter_runtime::default_coding_agent_prompt().to_owned()),
    }
}

/// Apply `[prompts]` config to a session seed. Config only replaces the
/// built-in general default that `SessionInit::default()` installs; a
/// caller-supplied system prompt is left untouched so explicit init wins.
fn apply_prompt_config(prompts: &PromptsConfig, mut init: SessionInit) -> SessionInit {
    let Some(configured) = configured_system_prompt(prompts) else {
        return init;
    };
    let default_text = halter_runtime::default_system_prompt();
    for segment in &mut init.system_prompt_seed {
        if segment.kind == PromptSegmentKind::System && segment.text == default_text {
            *segment = halter_runtime::system_prompt_segment(&configured);
        }
    }
    init
}

#[cfg(feature = "sqlite")]
fn build_session_store(config: &SessionsConfig) -> anyhow::Result<Arc<dyn SessionStore>> {
    match config.backend {
        SessionBackend::Memory => Ok(Arc::new(InMemorySessionStore::default())),
        SessionBackend::Sqlite => build_sqlite_session_store(config),
    }
}

#[cfg(not(feature = "sqlite"))]
fn build_session_store(config: &SessionsConfig) -> anyhow::Result<Arc<dyn SessionStore>> {
    match config.backend {
        SessionBackend::Memory => Ok(Arc::new(InMemorySessionStore::default())),
    }
}

#[cfg(feature = "sqlite")]
fn build_sqlite_session_store(config: &SessionsConfig) -> anyhow::Result<Arc<dyn SessionStore>> {
    let store = match config.sqlite_path.as_ref() {
        Some(path) => SqliteSessionStore::open(path)?,
        None => SqliteSessionStore::open_default()?,
    };
    Ok(Arc::new(store))
}

#[cfg(feature = "sqlite")]
fn describe_session_backend(config: &SessionsConfig) -> &'static str {
    match config.backend {
        SessionBackend::Memory => "memory",
        SessionBackend::Sqlite => "sqlite",
    }
}

#[cfg(not(feature = "sqlite"))]
fn describe_session_backend(config: &SessionsConfig) -> &'static str {
    match config.backend {
        SessionBackend::Memory => "memory",
    }
}

fn build_model_registry(config: &HarnessConfig) -> anyhow::Result<ModelRegistry> {
    let mut registry = ModelRegistry::new();
    // Each provider family is resolved and constructed at most once, then shared
    // across every role and model-judge member that references it.
    let mut family_providers: HashMap<ConfiguredProvider, Arc<dyn Provider>> = HashMap::new();

    let default_slot = config.default_slot()?;
    let default_model = build_slot_model(
        config,
        &mut registry,
        &mut family_providers,
        default_slot,
        ModelRole::default_role(),
        DEFAULT_MODEL_ID,
        "default",
    )?;

    let subagent_slot = config.subagent_slot().unwrap_or(default_slot);
    let subagent_model = build_slot_model(
        config,
        &mut registry,
        &mut family_providers,
        subagent_slot,
        ModelRole::subagent(),
        SUBAGENT_MODEL_ID,
        "subagent",
    )?;

    // The small slot is always a single concrete model. When unset it falls
    // back to the representative leaf of the default slot (the model-judge
    // default model for model-judge slots) rather than fanning out.
    let small_config = match config.small_model() {
        Some(model) => model.clone(),
        None => config.default_model()?.clone(),
    };
    let small_model = build_inline_model(
        config,
        &mut registry,
        &mut family_providers,
        &small_config,
        ModelRole::small(),
        SMALL_MODEL_ID,
    )?;

    debug!(
        default_provider = %default_model.provider,
        default_model = %default_model.model,
        small_provider = %small_model.provider,
        small_model = %small_model.model,
        subagent_provider = %subagent_model.provider,
        subagent_model = %subagent_model.model,
        "building model registry"
    );

    registry.set_default_model(default_model);
    registry.set_small_model(small_model);
    registry.set_subagent_model(subagent_model);

    Ok(registry)
}

/// Build the [`ResolvedModel`] for a model slot, registering whatever providers
/// it needs (a single family provider for inline slots, or a synthetic
/// model-judge provider plus its members' family providers for model-judge
/// slots).
fn build_slot_model(
    config: &HarnessConfig,
    registry: &mut ModelRegistry,
    family_providers: &mut HashMap<ConfiguredProvider, Arc<dyn Provider>>,
    slot: &ModelSlot,
    role: ModelRole,
    id: &str,
    slot_label: &str,
) -> anyhow::Result<ResolvedModel> {
    match slot {
        ModelSlot::Inline(model) => {
            build_inline_model(config, registry, family_providers, model, role, id)
        }
        ModelSlot::Reference(ModelSlotRef::ModelJudge) => {
            let model_judge = config.model_judge().with_context(|| {
                format!(
                    "invalid configuration: models.{slot_label} is set to \"model_judge\" but [models.model_judge] is not defined"
                )
            })?;
            build_model_judge_model(
                config,
                registry,
                family_providers,
                model_judge,
                role,
                id,
                slot_label,
            )
        }
    }
}

fn build_inline_model(
    config: &HarnessConfig,
    registry: &mut ModelRegistry,
    family_providers: &mut HashMap<ConfiguredProvider, Arc<dyn Provider>>,
    model: &ModelConfig,
    role: ModelRole,
    id: &str,
) -> anyhow::Result<ResolvedModel> {
    ensure_family_provider(config, registry, family_providers, model.provider)?;
    Ok(resolved_model(
        model,
        role,
        ModelId::from(id),
        ProviderName::from(model.provider.to_string()),
    ))
}

fn build_model_judge_model(
    config: &HarnessConfig,
    registry: &mut ModelRegistry,
    family_providers: &mut HashMap<ConfiguredProvider, Arc<dyn Provider>>,
    model_judge: &ModelJudgeConfig,
    role: ModelRole,
    id: &str,
    slot_label: &str,
) -> anyhow::Result<ResolvedModel> {
    let default_member =
        build_model_judge_member(config, registry, family_providers, &model_judge.default)?;
    let synthesis_member =
        build_model_judge_member(config, registry, family_providers, &model_judge.synthesis)?;
    let mut panel = Vec::with_capacity(model_judge.panel.len());
    for panelist in &model_judge.panel {
        panel.push(build_model_judge_member(
            config,
            registry,
            family_providers,
            panelist,
        )?);
    }

    // Mirror the default leaf so capability/compaction queries (which the
    // model-judge provider delegates to its default member) behave
    // consistently, but route the slot through a synthetic provider name.
    let provider_name = ProviderName::from(format!("model-judge-{slot_label}"));
    let default_leaf = default_member.model.clone();
    let model_judge_provider = Arc::new(ModelJudgeProvider::new(
        default_member,
        synthesis_member,
        panel,
    ));
    registry.register_provider(provider_name.clone(), model_judge_provider);

    Ok(ResolvedModel {
        role,
        id: ModelId::from(id),
        provider: provider_name,
        provider_kind: default_leaf.provider_kind,
        api_kind: default_leaf.api_kind,
        model: default_leaf.model,
        max_input_tokens: default_leaf.max_input_tokens,
        max_output_tokens: default_leaf.max_output_tokens,
        reasoning: default_leaf.reasoning,
        tokens_per_minute: default_leaf.tokens_per_minute,
    })
}

fn build_model_judge_member(
    config: &HarnessConfig,
    registry: &mut ModelRegistry,
    family_providers: &mut HashMap<ConfiguredProvider, Arc<dyn Provider>>,
    model: &ModelConfig,
) -> anyhow::Result<ModelJudgeMember> {
    let provider = ensure_family_provider(config, registry, family_providers, model.provider)?;
    let resolved = resolved_model(
        model,
        ModelRole::default_role(),
        ModelId::from(model.model.clone()),
        ProviderName::from(model.provider.to_string()),
    );
    Ok(ModelJudgeMember {
        provider,
        model: resolved,
    })
}

/// Resolve, construct, register, and cache the provider for a family exactly
/// once.
fn ensure_family_provider(
    config: &HarnessConfig,
    registry: &mut ModelRegistry,
    family_providers: &mut HashMap<ConfiguredProvider, Arc<dyn Provider>>,
    family: ConfiguredProvider,
) -> anyhow::Result<Arc<dyn Provider>> {
    if let Some(provider) = family_providers.get(&family) {
        return Ok(provider.clone());
    }
    let resolved = resolve_selected_provider_config(config, family)?;
    let provider = build_provider(&resolved)?;
    registry.register_provider(ProviderName::from(family.to_string()), provider.clone());
    family_providers.insert(family, provider.clone());
    Ok(provider)
}

fn resolved_model(
    model: &ModelConfig,
    role: ModelRole,
    id: ModelId,
    provider: ProviderName,
) -> ResolvedModel {
    ResolvedModel {
        role,
        id,
        provider,
        provider_kind: model.provider.provider_kind(),
        api_kind: model.provider.api_kind(),
        model: model.model.clone(),
        max_input_tokens: model.max_input_tokens,
        max_output_tokens: model.max_output_tokens,
        reasoning: model.reasoning,
        tokens_per_minute: model.tokens_per_minute,
    }
}

fn build_provider(
    provider: &ResolvedProviderConfig,
) -> anyhow::Result<Arc<dyn halter_providers::Provider>> {
    debug!(
        provider = %provider.provider,
        base_url = %provider.base_url,
        header_overrides = provider.headers.len(),
        "constructing provider client"
    );
    let provider: Arc<dyn halter_providers::Provider> = match provider.provider {
        ConfiguredProvider::Anthropic => Arc::new(AnthropicProvider::new_with_headers(
            api_key_auth(provider)?,
            provider.base_url.clone(),
            &provider.headers,
            provider.temperature,
        )?),
        ConfiguredProvider::OpenAi => match &provider.auth {
            ResolvedProviderAuth::ApiKey(api_key) => Arc::new(OpenAiProvider::new_with_headers(
                api_key.clone(),
                provider.base_url.clone(),
                &provider.headers,
                provider.temperature,
            )?),
            ResolvedProviderAuth::OpenAiOAuth(oauth) => {
                Arc::new(OpenAiProvider::new_with_oauth_and_headers(
                    openai_oauth_credentials(oauth),
                    provider.base_url.clone(),
                    &provider.headers,
                    provider.temperature,
                )?)
            }
        },
        ConfiguredProvider::OpenRouter => Arc::new(OpenRouterProvider::new_with_headers(
            api_key_auth(provider)?,
            provider.base_url.clone(),
            &provider.headers,
            provider.temperature,
        )?),
    };
    Ok(provider)
}

fn api_key_auth(provider: &ResolvedProviderConfig) -> anyhow::Result<String> {
    match &provider.auth {
        ResolvedProviderAuth::ApiKey(api_key) => Ok(api_key.clone()),
        ResolvedProviderAuth::OpenAiOAuth(_) => {
            anyhow::bail!(
                "provider '{}' does not support OpenAI OAuth credentials",
                provider.provider
            )
        }
    }
}

fn openai_oauth_credentials(config: &OpenAiOAuthConfig) -> OpenAiOAuthCredentials {
    OpenAiOAuthCredentials::new(
        config.client_id.clone(),
        config.access_token.clone(),
        config.id_token.clone(),
        config.refresh_token.clone(),
    )
}

fn resolve_selected_provider_config(
    config: &HarnessConfig,
    provider: ConfiguredProvider,
) -> anyhow::Result<ResolvedProviderConfig> {
    resolve_selected_provider_config_with(config, provider, |name| {
        let Some(raw) = env::var_os(name) else {
            return Ok(None);
        };
        let value = raw
            .into_string()
            .map_err(|_| anyhow::anyhow!("invalid utf-8 in {}", name))?;
        Ok(Some(value))
    })
}

fn resolve_selected_provider_config_with<F>(
    config: &HarnessConfig,
    provider: ConfiguredProvider,
    lookup_env: F,
) -> anyhow::Result<ResolvedProviderConfig>
where
    F: FnMut(&str) -> anyhow::Result<Option<String>>,
{
    resolve_provider_runtime_config(provider, config.provider_config(provider), lookup_env)
}

fn policy_from_config(config: &PolicyConfig) -> PolicySettings {
    // `process_tree_root` is anchored to the live halter PID at builder
    // time so process-signal checks (AC1.6 / AC1.7) can reject signals
    // aimed at PIDs that aren't descendants of this process. Other newer
    // fields (`allowed_read_roots`, `sensitive_path_patterns`,
    // `shell_mode`) still inherit from `PolicySettings::default()` until
    // the surface lands in user config.
    let defaults = PolicySettings::default();
    let allowed_hosts = if config.network.allowed_hosts.is_empty() {
        defaults.allowed_hosts.clone()
    } else {
        config.network.allowed_hosts.clone()
    };
    PolicySettings {
        allowed_write_roots: config.allowed_write_roots.clone(),
        max_read_bytes: config.max_read_bytes,
        shell_enabled: config.shell.enabled,
        allowed_shell_commands: config.shell.allow.clone(),
        shell_timeout_secs: config.shell.timeout_secs,
        network_enabled: config.network.enabled,
        allowed_hosts,
        allowed_loopback: config
            .network
            .allowed_loopback
            .iter()
            .map(|entry| LoopbackAllow {
                host: entry.host.clone(),
                port: entry.port,
            })
            .collect(),
        max_subagent_depth: config.max_subagent_depth,
        max_concurrent_subagents: config.max_concurrent_subagents,
        process_tree_root: Some(std::process::id() as i32),
        ..defaults
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use halter_config::{ModelConfig, OpenAiOAuthConfig, ProviderConfig};
    use halter_protocol::{PluginManifest, ReasoningEffort, SkillId};
    use tempfile::tempdir;

    use super::*;

    #[tokio::test]
    async fn builder_requires_default_model_configuration() {
        let error = match HalterBuilder::default()
            .with_resource_snapshot(ResourceSnapshot::empty())
            .build()
            .await
        {
            Ok(_) => panic!("build should fail"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("[models.default] is required"));
    }

    #[tokio::test]
    async fn builder_requires_precompiled_resources() {
        let error = match HalterBuilder::default()
            .with_config(openai_config(Some("test-key")))
            .build()
            .await
        {
            Ok(_) => panic!("build should fail"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("missing resource snapshot"));
    }

    #[test]
    fn policy_from_config_anchors_process_tree_root_to_live_pid() {
        // AC1.6 / AC1.7 rely on `process_tree_root` being populated so the
        // policy can reject signals aimed at PIDs outside the halter process
        // tree. The capability surface defaults to `None`; the builder is the
        // single place where the live PID gets stitched in.
        let config = openai_config(Some("test-key")).policy.clone();
        let settings = policy_from_config(&config);
        assert_eq!(
            settings.process_tree_root,
            Some(std::process::id() as i32),
            "process_tree_root should be anchored to std::process::id() at builder time"
        );
    }

    #[tokio::test]
    async fn builder_requires_provider_credentials_when_not_configured() {
        let error = resolve_selected_provider_config_with(
            &openai_config(None),
            ConfiguredProvider::OpenAi,
            |_| Ok(None),
        )
        .expect_err("provider resolution should fail");

        assert!(error.to_string().contains("OPENAI_API_KEY"));
        assert!(error.to_string().contains("[providers.openai].oauth"));
    }

    #[tokio::test]
    async fn builder_accepts_openai_oauth_without_api_key() {
        let mut config = openai_config(None);
        config.providers.openai = Some(ProviderConfig {
            oauth: Some(openai_oauth_config()),
            ..ProviderConfig::default()
        });

        HalterBuilder::default()
            .with_config(config)
            .with_resource_snapshot(ResourceSnapshot::empty())
            .build()
            .await
            .expect("build should succeed with OpenAI OAuth credentials");
    }

    #[tokio::test]
    async fn builder_constructs_model_judge_default_slot() {
        let mut config = openai_config(Some("test-key"));
        config.models.default = Some(ModelSlot::Reference(ModelSlotRef::ModelJudge));
        let leaf = |model: &str| ModelConfig {
            provider: ConfiguredProvider::OpenAi,
            model: model.to_owned(),
            max_input_tokens: None,
            max_output_tokens: None,
            reasoning: None,
            tokens_per_minute: None,
        };
        config.models.model_judge = Some(ModelJudgeConfig {
            default: leaf("gpt-default"),
            synthesis: leaf("gpt-synthesis"),
            panel: vec![leaf("gpt-panel-a"), leaf("gpt-panel-b")],
        });

        let registry = build_model_registry(&config).expect("model registry");
        let default_model = registry.default_model().expect("default model");
        assert_eq!(default_model.provider.0, "model-judge-default");
        assert_eq!(default_model.model, "gpt-default");
        assert!(!default_model.model.starts_with("model-judge:"));

        let halter = HalterBuilder::default()
            .with_config(config)
            .with_resource_snapshot(ResourceSnapshot::empty())
            .build()
            .await
            .expect("build should succeed for a model-judge default slot");

        // The model-judge slot mirrors its default leaf model.
        assert_eq!(
            halter.config.default_model().expect("default model").model,
            "gpt-default"
        );
    }

    #[tokio::test]
    async fn h12_builder_accepts_shared_provider_across_roles() {
        // All three roles share ConfiguredProvider::OpenAi, which resolves to
        // the same ResolvedProviderConfig — the H12 collision check must
        // accept identical registrations and not false-positive.
        let mut config = openai_config(Some("test-key"));
        config.models.small = Some(ModelConfig {
            provider: ConfiguredProvider::OpenAi,
            model: "gpt-5-mini".to_owned(),
            max_input_tokens: Some(64_000),
            max_output_tokens: Some(4_096),
            reasoning: None,
            tokens_per_minute: None,
        });
        config.models.subagent = Some(ModelSlot::Inline(ModelConfig {
            provider: ConfiguredProvider::OpenAi,
            model: "gpt-5".to_owned(),
            max_input_tokens: Some(200_000),
            max_output_tokens: Some(8_192),
            reasoning: None,
            tokens_per_minute: None,
        }));

        HalterBuilder::default()
            .with_config(config)
            .with_resource_snapshot(ResourceSnapshot::empty())
            .build()
            .await
            .expect("build should succeed when roles share a provider");
    }

    #[tokio::test]
    async fn builder_uses_injected_session_store() {
        let temp = tempdir().expect("tempdir");
        let store = CountingSessionStore::default();
        let store_for_builder: Arc<dyn SessionStore> = Arc::new(store.clone());
        let halter = HalterBuilder::default()
            .with_config(openai_config(Some("test-key")))
            .with_resource_snapshot(ResourceSnapshot::empty())
            .with_session_store(store_for_builder)
            .build()
            .await
            .expect("build halter");

        halter
            .new_session(SessionInit {
                working_dir: temp.path().to_path_buf(),
                ..SessionInit::default()
            })
            .await
            .expect("create session");

        assert_eq!(store.create_calls(), 1);
        assert_eq!(store.commit_calls(), 1);
    }

    #[tokio::test]
    async fn builder_accepts_loaded_skills_without_prebuilt_snapshot() {
        let temp = tempdir().expect("tempdir");

        HalterBuilder::default()
            .with_config(openai_config(Some("test-key")))
            .with_loaded_skills(vec![LoadedSkill {
                id: SkillId::from("skill-1"),
                name: "helper".to_owned(),
                description: "Loaded helper skill".to_owned(),
                root: temp.path().join("helper"),
                body: "Do the helpful thing.".to_owned(),
                supporting_files: Vec::new(),
                scripts: Vec::new(),
                revision: "skill-revision".to_owned(),
            }])
            .build()
            .await
            .expect("build halter");
    }

    #[tokio::test]
    async fn builder_accepts_loaded_plugins_without_prebuilt_snapshot() {
        let temp = tempdir().expect("tempdir");

        HalterBuilder::default()
            .with_config(openai_config(Some("test-key")))
            .with_loaded_plugins(vec![LoadedPlugin {
                id: "plugin-1".into(),
                root: temp.path().join("plugin"),
                manifest: PluginManifest {
                    name: "plugin".to_owned(),
                    version: "0.1.0".to_owned(),
                    ..PluginManifest::default()
                },
                skills: Vec::new(),
                agents: Vec::new(),
                hooks: Vec::new(),
                mcp_servers: Vec::new(),
                lsp_servers: Vec::new(),
                output_styles: Vec::new(),
                bin_paths: Vec::new(),
                defaults: crate::PluginDefaults::default(),
            }])
            .build()
            .await
            .expect("build halter");
    }

    #[tokio::test]
    async fn builder_writes_per_session_trace_file_when_traces_dir_configured() {
        let temp = tempdir().expect("tempdir");
        let traces_dir = temp.path().join("traces");
        let mut config = openai_config(Some("test-key"));
        config.runtime.traces_dir = Some(traces_dir.clone());

        let halter = HalterBuilder::default()
            .with_config(config)
            .with_resource_snapshot(ResourceSnapshot::empty())
            .build()
            .await
            .expect("build halter");

        let session = halter
            .new_session(SessionInit {
                working_dir: temp.path().to_path_buf(),
                ..SessionInit::default()
            })
            .await
            .expect("create session");

        let session_id = session.session_id().0.clone();
        drop(session);

        let trace_path = traces_dir.join(format!("{session_id}.txt"));
        let contents = std::fs::read_to_string(&trace_path).expect("read trace file");
        let mut lines = contents.lines();
        let header: serde_json::Value =
            serde_json::from_str(lines.next().expect("header")).expect("header json");
        assert_eq!(header["kind"], "trace_header");
        assert_eq!(header["session_id"], session_id);
        let started: halter_protocol::SessionEvent =
            serde_json::from_str(lines.next().expect("session-started event")).expect("event json");
        assert!(matches!(
            started.payload,
            halter_protocol::SessionEventPayload::SessionStarted
        ));
    }

    #[tokio::test]
    async fn builder_skips_trace_file_when_traces_dir_unset() {
        let temp = tempdir().expect("tempdir");
        let halter = HalterBuilder::default()
            .with_config(openai_config(Some("test-key")))
            .with_resource_snapshot(ResourceSnapshot::empty())
            .build()
            .await
            .expect("build halter");

        halter
            .new_session(SessionInit {
                working_dir: temp.path().to_path_buf(),
                ..SessionInit::default()
            })
            .await
            .expect("create session");

        let entries: Vec<_> = std::fs::read_dir(temp.path())
            .expect("read tempdir")
            .map(|entry| entry.expect("entry").file_name())
            .collect();
        assert!(
            entries.is_empty(),
            "no trace files should exist outside traces_dir: {entries:?}"
        );
    }

    #[tokio::test]
    async fn builder_fails_when_traces_dir_points_at_a_file() {
        let temp = tempdir().expect("tempdir");
        let bogus = temp.path().join("not-a-dir.txt");
        std::fs::write(&bogus, b"hi").expect("seed file");
        let mut config = openai_config(Some("test-key"));
        config.runtime.traces_dir = Some(bogus.clone());

        let error = match HalterBuilder::default()
            .with_config(config)
            .with_resource_snapshot(ResourceSnapshot::empty())
            .build()
            .await
        {
            Ok(_) => panic!("build should fail when traces_dir points at a file"),
            Err(error) => error,
        };

        assert!(
            error.to_string().contains("not a directory"),
            "unexpected error: {error}"
        );
    }

    #[cfg(feature = "sqlite")]
    #[tokio::test]
    async fn builder_uses_sqlite_store_from_config() {
        let temp = tempdir().expect("tempdir");
        let db_path = temp.path().join("sessions.db");
        let mut config = openai_config(Some("test-key"));
        config.sessions.backend = SessionBackend::Sqlite;
        config.sessions.sqlite_path = Some(db_path.clone());

        let halter = HalterBuilder::default()
            .with_config(config)
            .with_resource_snapshot(ResourceSnapshot::empty())
            .build()
            .await
            .expect("build halter");

        halter
            .new_session(SessionInit {
                working_dir: temp.path().to_path_buf(),
                ..SessionInit::default()
            })
            .await
            .expect("create session");

        let persisted = SqliteSessionStore::open(&db_path)
            .expect("open persisted sqlite store")
            .list_sessions()
            .await
            .expect("list persisted sessions");
        assert_eq!(persisted.len(), 1);
    }

    fn openai_config(api_key: Option<&str>) -> HarnessConfig {
        let mut config = HarnessConfig::default();
        config.models.default = Some(ModelSlot::Inline(ModelConfig {
            provider: ConfiguredProvider::OpenAi,
            model: "gpt-5".to_owned(),
            max_input_tokens: Some(200_000),
            max_output_tokens: Some(8_192),
            reasoning: Some(ReasoningEffort::Medium),
            tokens_per_minute: None,
        }));
        config.providers.openai = Some(ProviderConfig {
            api_key: api_key.map(ToOwned::to_owned),
            ..ProviderConfig::default()
        });
        config
    }

    fn openai_oauth_config() -> OpenAiOAuthConfig {
        OpenAiOAuthConfig {
            client_id: "client".to_owned(),
            access_token: "access-token".to_owned(),
            id_token: "id-token".to_owned(),
            refresh_token: "refresh-token".to_owned(),
        }
    }

    #[derive(Clone, Default)]
    struct CountingSessionStore {
        inner: InMemorySessionStore,
        create_calls: Arc<AtomicUsize>,
        commit_calls: Arc<AtomicUsize>,
    }

    impl CountingSessionStore {
        fn create_calls(&self) -> usize {
            self.create_calls.load(Ordering::SeqCst)
        }

        fn commit_calls(&self) -> usize {
            self.commit_calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl SessionStore for CountingSessionStore {
        async fn create_session(
            &self,
            session: halter_session::StoredSession,
        ) -> anyhow::Result<()> {
            self.create_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.create_session(session).await
        }

        async fn load_session(
            &self,
            session_id: &halter_protocol::SessionId,
        ) -> anyhow::Result<Option<halter_session::StoredSession>> {
            self.inner.load_session(session_id).await
        }

        async fn commit(
            &self,
            session_id: &halter_protocol::SessionId,
            snapshot: Option<Arc<halter_protocol::ResourceSnapshot>>,
            expected_state: Option<halter_protocol::SessionState>,
            state: Option<halter_protocol::SessionState>,
            events: Vec<halter_protocol::PendingEvent>,
        ) -> anyhow::Result<Vec<halter_protocol::SessionEvent>> {
            self.commit_calls.fetch_add(1, Ordering::SeqCst);
            self.inner
                .commit(session_id, snapshot, expected_state, state, events)
                .await
        }

        async fn replay(
            &self,
            session_id: &halter_protocol::SessionId,
        ) -> anyhow::Result<Vec<halter_protocol::SessionEvent>> {
            self.inner.replay(session_id).await
        }

        async fn list_sessions(&self) -> anyhow::Result<Vec<halter_protocol::SessionBlueprint>> {
            self.inner.list_sessions().await
        }
    }

    fn seed_text(init: &SessionInit) -> &str {
        init.system_prompt_seed
            .first()
            .map(|segment| segment.text.as_str())
            .expect("seed has at least one segment")
    }

    #[test]
    fn configured_system_prompt_general_preset_is_noop() {
        let prompts = PromptsConfig::default();
        assert_eq!(prompts.preset, SystemPromptPreset::General);
        assert_eq!(configured_system_prompt(&prompts), None);
    }

    #[test]
    fn configured_system_prompt_coding_preset_selects_coding_prompt() {
        let prompts = PromptsConfig {
            preset: SystemPromptPreset::Coding,
            system_prompt: None,
        };
        assert_eq!(
            configured_system_prompt(&prompts).as_deref(),
            Some(halter_runtime::default_coding_agent_prompt())
        );
    }

    #[test]
    fn configured_system_prompt_override_wins_over_preset() {
        let prompts = PromptsConfig {
            preset: SystemPromptPreset::Coding,
            system_prompt: Some("custom override".to_owned()),
        };
        assert_eq!(
            configured_system_prompt(&prompts).as_deref(),
            Some("custom override")
        );
    }

    #[test]
    fn configured_system_prompt_blank_override_falls_back_to_preset() {
        // A whitespace-only override is ignored; the preset is used instead.
        let prompts = PromptsConfig {
            preset: SystemPromptPreset::General,
            system_prompt: Some("   ".to_owned()),
        };
        assert_eq!(configured_system_prompt(&prompts), None);
    }

    #[test]
    fn apply_prompt_config_swaps_default_seed_for_coding_preset() {
        let prompts = PromptsConfig {
            preset: SystemPromptPreset::Coding,
            system_prompt: None,
        };
        let init = apply_prompt_config(&prompts, SessionInit::default());
        assert_eq!(seed_text(&init), halter_runtime::default_coding_agent_prompt());
    }

    #[test]
    fn apply_prompt_config_general_preset_leaves_default_seed() {
        let prompts = PromptsConfig::default();
        let init = apply_prompt_config(&prompts, SessionInit::default());
        assert_eq!(seed_text(&init), halter_runtime::default_system_prompt());
    }

    #[test]
    fn apply_prompt_config_respects_explicit_session_prompt() {
        // An explicit per-session prompt is never overridden by config.
        let prompts = PromptsConfig {
            preset: SystemPromptPreset::Coding,
            system_prompt: Some("config wins?".to_owned()),
        };
        let init = SessionInit::default().with_system_prompt("explicit session prompt");
        let init = apply_prompt_config(&prompts, init);
        assert_eq!(seed_text(&init), "explicit session prompt");
    }
}
