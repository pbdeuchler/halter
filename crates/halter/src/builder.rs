// pattern: Imperative Shell

use std::collections::HashMap;
use std::env;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Context;
use halter_config::{
    ConfiguredProvider, DEFAULT_MODEL_ID, HarnessConfig, ModelConfig, ModelJudgeConfig,
    ModelJudgeMode, ModelSlot, ModelSlotRef, OpenAiOAuthConfig, PolicyConfig, PromptsConfig,
    ResilienceConfig, ResolvedProviderAuth, ResolvedProviderConfig, SMALL_MODEL_ID,
    SUBAGENT_MODEL_ID, SessionBackend, SessionsConfig, SystemPromptPreset, expand_path, load_path,
    resolve_provider_runtime_config,
};
use halter_hooks::{Hook, Hooks, RegisteredHookPriority, RegisteredHooks};
use halter_protocol::{
    HookWarning, ModelId, ModelRole, PromptSegmentKind, ProviderName, ResolvedModel,
    ResourceSnapshot,
};
use halter_providers::{
    AnthropicProvider, DefaultProviderErrorClassifier, FullTurnJudgePlan, FullTurnPanelist,
    ModelJudgeMember, ModelJudgeProvider, ModelRegistry, OpenAiOAuthCredentials, OpenAiProvider,
    OpenRouterProvider, Provider, ProviderErrorClassifier, ProviderTimeouts, ResiliencePolicy,
    RetryPolicy,
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
    resilience_policy: Option<ResiliencePolicy>,
    provider_error_classifier: Option<Arc<dyn ProviderErrorClassifier>>,
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

    /// Override provider request timeouts and retry policy for all provider
    /// families built by this harness.
    #[must_use]
    pub fn with_resilience_policy(mut self, policy: ResiliencePolicy) -> Self {
        self.resilience_policy = Some(policy);
        self
    }

    /// Install a provider error classifier used by resilient providers after
    /// provider-native classification and before retry decisions.
    #[must_use]
    pub fn with_provider_error_classifier(
        mut self,
        classifier: Arc<dyn ProviderErrorClassifier>,
    ) -> Self {
        self.provider_error_classifier = Some(classifier);
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
            resilience_policy,
            provider_error_classifier,
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

        let provider_options = ProviderBuildOptions {
            resilience_policy,
            provider_error_classifier: provider_error_classifier
                .unwrap_or_else(|| Arc::new(DefaultProviderErrorClassifier)),
        };
        let models = Arc::new(build_model_registry(&config, &provider_options)?);
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
    /// The session's base system prompt is resolved from `[prompts]` config
    /// unless the caller installed an explicit one on `init` (see
    /// [`SessionInit::with_system_prompt`]). Base precedence, most specific
    /// first: an explicit per-session prompt > `prompts.system_prompt` >
    /// `prompts.preset` > the built-in general-purpose default.
    ///
    /// `prompts.append_system_prompt` is additive: when present, it is inserted
    /// after the resolved base prompt and before any per-session appended
    /// system-prompt segments.
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
        SystemPromptPreset::Coding => {
            Some(halter_runtime::default_coding_agent_prompt().to_owned())
        }
    }
}

fn trimmed_prompt(value: Option<&str>) -> Option<String> {
    value
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(ToOwned::to_owned)
}

/// Apply `[prompts]` config to a session seed. Config only replaces the
/// built-in general default that `SessionInit::default()` installs; a
/// caller-supplied system prompt is left untouched so explicit init wins.
/// Config-level appended text is additive and lands immediately after the
/// resolved base prompt, before any session-level appended system prompt.
fn apply_prompt_config(prompts: &PromptsConfig, mut init: SessionInit) -> SessionInit {
    if let Some(configured) = configured_system_prompt(prompts) {
        let default_text = halter_runtime::default_system_prompt();
        for segment in &mut init.system_prompt_seed {
            if segment.kind == PromptSegmentKind::System && segment.text == default_text {
                *segment = halter_runtime::system_prompt_segment(&configured);
            }
        }
    }

    if let Some(append) = trimmed_prompt(prompts.append_system_prompt.as_deref()) {
        let segment = halter_runtime::appended_system_prompt_segment(&append);
        let insert_index = init
            .system_prompt_seed
            .iter()
            .position(|segment| segment.kind == PromptSegmentKind::System)
            .map_or(0, |index| index + 1);
        init.system_prompt_seed.insert(insert_index, segment);
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

#[derive(Clone)]
struct ProviderBuildOptions {
    resilience_policy: Option<ResiliencePolicy>,
    provider_error_classifier: Arc<dyn ProviderErrorClassifier>,
}

fn build_model_registry(
    config: &HarnessConfig,
    provider_options: &ProviderBuildOptions,
) -> anyhow::Result<ModelRegistry> {
    let mut registry = ModelRegistry::new();
    // Each provider family is resolved and constructed at most once, then shared
    // across every role and model-judge member that references it.
    let mut family_providers: HashMap<ConfiguredProvider, Arc<dyn Provider>> = HashMap::new();

    let default_slot = config.default_slot()?;
    let default_model = build_slot_model(
        config,
        &mut registry,
        &mut family_providers,
        provider_options,
        default_slot,
        ModelRole::default_role(),
        DEFAULT_MODEL_ID,
        "default",
    )?;

    let subagent_model = match config.subagent_slot() {
        Some(ModelSlot::Reference(ModelSlotRef::AutoResolve)) => default_model.clone(),
        Some(subagent_slot) => build_slot_model(
            config,
            &mut registry,
            &mut family_providers,
            provider_options,
            subagent_slot,
            ModelRole::subagent(),
            SUBAGENT_MODEL_ID,
            "subagent",
        )?,
        None => build_slot_model(
            config,
            &mut registry,
            &mut family_providers,
            provider_options,
            default_slot,
            ModelRole::subagent(),
            SUBAGENT_MODEL_ID,
            "subagent",
        )?,
    };

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
        provider_options,
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
    provider_options: &ProviderBuildOptions,
    slot: &ModelSlot,
    role: ModelRole,
    id: &str,
    slot_label: &str,
) -> anyhow::Result<ResolvedModel> {
    match slot {
        ModelSlot::Inline(model) => build_inline_model(
            config,
            registry,
            family_providers,
            provider_options,
            model,
            role,
            id,
        ),
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
                provider_options,
                model_judge,
                role,
                id,
                slot_label,
            )
        }
        ModelSlot::Reference(ModelSlotRef::AutoResolve) => anyhow::bail!(
            "invalid configuration: models.{slot_label} is set to \"auto_resolve\" but auto-resolve must be handled before building a concrete slot"
        ),
    }
}

fn build_inline_model(
    config: &HarnessConfig,
    registry: &mut ModelRegistry,
    family_providers: &mut HashMap<ConfiguredProvider, Arc<dyn Provider>>,
    provider_options: &ProviderBuildOptions,
    model: &ModelConfig,
    role: ModelRole,
    id: &str,
) -> anyhow::Result<ResolvedModel> {
    ensure_family_provider(
        config,
        registry,
        family_providers,
        provider_options,
        model.provider,
    )?;
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
    provider_options: &ProviderBuildOptions,
    model_judge: &ModelJudgeConfig,
    role: ModelRole,
    id: &str,
    slot_label: &str,
) -> anyhow::Result<ResolvedModel> {
    // The mode is encoded structurally here so the runtime never has to branch
    // on it: OneShot registers a synthetic `Provider` the slot routes through;
    // FullTurn leaves the slot pointing at a plain default model and records a
    // panel plan the turn loop consults.
    match model_judge.mode {
        ModelJudgeMode::OneShot => build_one_shot_judge(
            config,
            registry,
            family_providers,
            provider_options,
            model_judge,
            role,
            id,
            slot_label,
        ),
        ModelJudgeMode::FullTurn => build_full_turn_judge(
            config,
            registry,
            family_providers,
            provider_options,
            model_judge,
            role,
            id,
            slot_label,
        ),
    }
}

/// OneShot judge: wrap the three roles in a [`ModelJudgeProvider`] and route the
/// slot through a synthetic provider name. The panel/synthesis/default cycle
/// runs inside the provider on every model call.
fn build_one_shot_judge(
    config: &HarnessConfig,
    registry: &mut ModelRegistry,
    family_providers: &mut HashMap<ConfiguredProvider, Arc<dyn Provider>>,
    provider_options: &ProviderBuildOptions,
    model_judge: &ModelJudgeConfig,
    role: ModelRole,
    id: &str,
    slot_label: &str,
) -> anyhow::Result<ResolvedModel> {
    let default_member = build_model_judge_member(
        config,
        registry,
        family_providers,
        provider_options,
        &model_judge.default,
    )?;
    let synthesis_member = build_model_judge_member(
        config,
        registry,
        family_providers,
        provider_options,
        &model_judge.synthesis,
    )?;
    let mut panel = Vec::with_capacity(model_judge.panel.len());
    for panelist in &model_judge.panel {
        panel.push(build_model_judge_member(
            config,
            registry,
            family_providers,
            provider_options,
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

/// FullTurn judge: the slot resolves to the plain default model (normal per-step
/// inference), each panelist is registered as its own resolvable model, and a
/// [`FullTurnJudgePlan`] is recorded so the runtime fans the user's turn out to
/// the panel as full sub-sessions before running the default.
fn build_full_turn_judge(
    config: &HarnessConfig,
    registry: &mut ModelRegistry,
    family_providers: &mut HashMap<ConfiguredProvider, Arc<dyn Provider>>,
    provider_options: &ProviderBuildOptions,
    model_judge: &ModelJudgeConfig,
    role: ModelRole,
    id: &str,
    slot_label: &str,
) -> anyhow::Result<ResolvedModel> {
    // The default model is an ordinary registry model; its per-step inference is
    // never judged. The judging happens once per turn, in the runtime.
    ensure_family_provider(
        config,
        registry,
        family_providers,
        provider_options,
        model_judge.default.provider,
    )?;
    let slot_model = resolved_model(
        &model_judge.default,
        role,
        ModelId::from(id),
        ProviderName::from(model_judge.default.provider.to_string()),
    );

    // Synthesis is a single inner inference, identical to the OneShot path.
    let synthesis = build_model_judge_member(
        config,
        registry,
        family_providers,
        provider_options,
        &model_judge.synthesis,
    )?;

    // Each panelist gets a unique, slot-scoped model id so the runtime can start
    // a sub-session against it; the label is the model name (disambiguated) used
    // as the candidate id shown to the synthesis judge.
    let labels = panel_labels(&model_judge.panel);
    let mut panel = Vec::with_capacity(model_judge.panel.len());
    for (index, panelist) in model_judge.panel.iter().enumerate() {
        ensure_family_provider(
            config,
            registry,
            family_providers,
            provider_options,
            panelist.provider,
        )?;
        let model_id = ModelId::from(format!("model-judge-panel:{slot_label}:{index}"));
        registry.register_model(resolved_model(
            panelist,
            ModelRole::default_role(),
            model_id.clone(),
            ProviderName::from(panelist.provider.to_string()),
        ));
        panel.push(FullTurnPanelist {
            model_id,
            label: labels[index].clone(),
        });
    }

    registry.register_full_turn_judge(
        &ModelId::from(id),
        Arc::new(FullTurnJudgePlan {
            synthesis,
            panel,
            isolation: model_judge.panel_isolation,
        }),
    );

    Ok(slot_model)
}

/// Candidate labels for FullTurn panelists: the model name, disambiguated with a
/// positional suffix when the same model appears more than once.
fn panel_labels(panel: &[ModelConfig]) -> Vec<String> {
    panel
        .iter()
        .enumerate()
        .map(|(index, panelist)| {
            let name = &panelist.model;
            let duplicate = panel
                .iter()
                .enumerate()
                .any(|(other, candidate)| other != index && candidate.model == *name);
            if duplicate {
                format!("{name}#{index}")
            } else {
                name.clone()
            }
        })
        .collect()
}

fn build_model_judge_member(
    config: &HarnessConfig,
    registry: &mut ModelRegistry,
    family_providers: &mut HashMap<ConfiguredProvider, Arc<dyn Provider>>,
    provider_options: &ProviderBuildOptions,
    model: &ModelConfig,
) -> anyhow::Result<ModelJudgeMember> {
    let provider = ensure_family_provider(
        config,
        registry,
        family_providers,
        provider_options,
        model.provider,
    )?;
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
    provider_options: &ProviderBuildOptions,
    family: ConfiguredProvider,
) -> anyhow::Result<Arc<dyn Provider>> {
    if let Some(provider) = family_providers.get(&family) {
        return Ok(provider.clone());
    }
    let resolved = resolve_selected_provider_config(config, family)?;
    let provider = build_provider(
        &resolved,
        selected_resilience_policy(config, family, provider_options),
        provider_options.provider_error_classifier.clone(),
    )?;
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
    resilience_policy: ResiliencePolicy,
    classifier: Arc<dyn ProviderErrorClassifier>,
) -> anyhow::Result<Arc<dyn halter_providers::Provider>> {
    debug!(
        provider = %provider.provider,
        base_url = %provider.base_url,
        header_overrides = provider.headers.len(),
        "constructing provider client"
    );
    let provider: Arc<dyn halter_providers::Provider> = match provider.provider {
        ConfiguredProvider::Anthropic => {
            Arc::new(AnthropicProvider::new_with_headers_and_timeouts(
                api_key_auth(provider)?,
                provider.base_url.clone(),
                &provider.headers,
                provider.temperature,
                resilience_policy.timeouts,
            )?)
        }
        ConfiguredProvider::OpenAi => match &provider.auth {
            ResolvedProviderAuth::ApiKey(api_key) => {
                Arc::new(OpenAiProvider::new_with_headers_and_resilience(
                    api_key.clone(),
                    provider.base_url.clone(),
                    &provider.headers,
                    provider.temperature,
                    resilience_policy,
                    classifier,
                )?)
            }
            ResolvedProviderAuth::OpenAiOAuth(oauth) => {
                Arc::new(OpenAiProvider::new_with_oauth_headers_and_resilience(
                    openai_oauth_credentials(oauth),
                    provider.base_url.clone(),
                    &provider.headers,
                    provider.temperature,
                    resilience_policy,
                    classifier,
                )?)
            }
        },
        ConfiguredProvider::OpenRouter => {
            Arc::new(OpenRouterProvider::new_with_headers_and_resilience(
                api_key_auth(provider)?,
                provider.base_url.clone(),
                &provider.headers,
                provider.temperature,
                resilience_policy,
                classifier,
            )?)
        }
    };
    Ok(provider)
}

fn selected_resilience_policy(
    config: &HarnessConfig,
    family: ConfiguredProvider,
    provider_options: &ProviderBuildOptions,
) -> ResiliencePolicy {
    provider_options
        .resilience_policy
        .unwrap_or_else(|| resilience_policy_from_config(config.resilience_for(family)))
}

fn resilience_policy_from_config(config: ResilienceConfig) -> ResiliencePolicy {
    ResiliencePolicy {
        timeouts: ProviderTimeouts {
            connect: std::time::Duration::from_secs(config.timeouts.connect_secs),
            request: std::time::Duration::from_secs(config.timeouts.request_secs),
            stream_idle: std::time::Duration::from_secs(config.timeouts.stream_idle_secs),
        },
        request_retry: RetryPolicy {
            max_attempts: config.request_retry.max_attempts,
            base_backoff: std::time::Duration::from_millis(config.request_retry.base_backoff_ms),
            max_backoff: std::time::Duration::from_secs(config.request_retry.max_backoff_secs),
            deadline: std::time::Duration::from_secs(config.request_retry.deadline_secs),
            jitter_pct: config.request_retry.jitter_pct,
        },
    }
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
    // fields (`sensitive_path_patterns`, `shell_mode`) still inherit from
    // `PolicySettings::default()` until the surface lands in user config.
    let defaults = PolicySettings::default();
    let allowed_hosts = if config.network.allowed_hosts.is_empty() {
        defaults.allowed_hosts.clone()
    } else {
        config.network.allowed_hosts.clone()
    };
    PolicySettings {
        allowed_write_roots: config.allowed_write_roots.clone(),
        allowed_read_roots: allowed_read_roots_from_config(config, &defaults),
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

fn allowed_read_roots_from_config(
    config: &PolicyConfig,
    defaults: &PolicySettings,
) -> Vec<PathBuf> {
    let mut roots = defaults.allowed_read_roots.clone();
    for root in &config.allowed_write_roots {
        if !roots.iter().any(|existing| existing == root) {
            roots.push(root.clone());
        }
    }
    roots
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use async_trait::async_trait;
    use halter_config::{
        ModelConfig, OpenAiOAuthConfig, ProviderConfig, RequestRetryConfig, ResilienceConfig,
        ResilienceTimeoutsConfig,
    };
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
    fn resilience_policy_from_config_maps_config_units_to_durations() {
        let policy = resilience_policy_from_config(ResilienceConfig {
            timeouts: ResilienceTimeoutsConfig {
                connect_secs: 2,
                request_secs: 3,
                stream_idle_secs: 4,
            },
            request_retry: RequestRetryConfig {
                max_attempts: 7,
                deadline_secs: 8,
                base_backoff_ms: 250,
                max_backoff_secs: 9,
                jitter_pct: 10,
            },
        });

        assert_eq!(policy.timeouts.connect, Duration::from_secs(2));
        assert_eq!(policy.timeouts.request, Duration::from_secs(3));
        assert_eq!(policy.timeouts.stream_idle, Duration::from_secs(4));
        assert_eq!(policy.request_retry.max_attempts, 7);
        assert_eq!(policy.request_retry.deadline, Duration::from_secs(8));
        assert_eq!(
            policy.request_retry.base_backoff,
            Duration::from_millis(250)
        );
        assert_eq!(policy.request_retry.max_backoff, Duration::from_secs(9));
        assert_eq!(policy.request_retry.jitter_pct, 10);
    }

    #[test]
    fn selected_resilience_policy_prefers_sdk_override() {
        let override_policy = ResiliencePolicy {
            timeouts: ProviderTimeouts {
                connect: Duration::from_secs(11),
                request: Duration::from_secs(12),
                stream_idle: Duration::from_secs(13),
            },
            request_retry: RetryPolicy {
                max_attempts: 2,
                base_backoff: Duration::from_millis(30),
                max_backoff: Duration::from_secs(4),
                deadline: Duration::from_secs(5),
                jitter_pct: 0,
            },
        };
        let options = ProviderBuildOptions {
            resilience_policy: Some(override_policy),
            provider_error_classifier: Arc::new(DefaultProviderErrorClassifier),
        };
        let mut config = HarnessConfig::default();
        config.resilience.timeouts.request_secs = 99;

        let selected = selected_resilience_policy(&config, ConfiguredProvider::OpenAi, &options);

        assert_eq!(selected, override_policy);
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

    #[test]
    fn policy_from_config_reads_configured_write_roots() {
        let mut config = openai_config(Some("test-key")).policy.clone();
        let worktree = PathBuf::from("/tmp/halter-factory-worktree");
        config.allowed_write_roots = vec![PathBuf::from("."), worktree.clone(), worktree.clone()];

        let settings = policy_from_config(&config);

        assert!(settings.allowed_read_roots.contains(&worktree));
        assert_eq!(
            settings
                .allowed_read_roots
                .iter()
                .filter(|root| *root == &worktree)
                .count(),
            1,
            "configured write roots should be added to read roots once"
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
            mode: ModelJudgeMode::OneShot,
            default: leaf("gpt-default"),
            synthesis: leaf("gpt-synthesis"),
            panel: vec![leaf("gpt-panel-a"), leaf("gpt-panel-b")],
            panel_isolation: Default::default(),
        });

        let options = default_provider_build_options();
        let registry = build_model_registry(&config, &options).expect("model registry");
        let default_model = registry.default_model().expect("default model");
        assert_eq!(default_model.provider.0, "model-judge-default");
        assert_eq!(default_model.model, "gpt-default");
        assert!(!default_model.model.starts_with("model-judge:"));
        // OneShot routes through the synthetic provider, not a FullTurn plan.
        assert!(registry.full_turn_judge(&default_model.id).is_none());

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
    async fn builder_constructs_full_turn_model_judge_slot() {
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
            mode: ModelJudgeMode::FullTurn,
            default: leaf("gpt-default"),
            synthesis: leaf("gpt-synthesis"),
            panel: vec![leaf("gpt-panel-a"), leaf("gpt-panel-b")],
            panel_isolation: Default::default(),
        });

        let options = default_provider_build_options();
        let registry = build_model_registry(&config, &options).expect("model registry");
        let default_model = registry.default_model().expect("default model");

        // FullTurn leaves the slot pointing at a plain default provider (no
        // synthetic model-judge provider) and records a panel plan instead.
        assert_eq!(default_model.provider.0, "openai");
        assert_eq!(default_model.model, "gpt-default");
        let plan = registry
            .full_turn_judge(&default_model.id)
            .expect("full-turn plan registered for the default slot");
        assert_eq!(plan.panel.len(), 2);
        assert_eq!(plan.panel[0].label, "gpt-panel-a");
        assert_eq!(plan.panel[1].label, "gpt-panel-b");

        // Each panelist is resolvable as its own registry model.
        for panelist in &plan.panel {
            registry
                .model(&panelist.model_id)
                .expect("panelist model resolvable");
        }

        HalterBuilder::default()
            .with_config(config)
            .with_resource_snapshot(ResourceSnapshot::empty())
            .build()
            .await
            .expect("build should succeed for a full-turn model-judge slot");
    }

    #[tokio::test]
    async fn builder_auto_resolve_subagent_reuses_default_model_slot() {
        let mut config = openai_config(Some("test-key"));
        config.models.default = Some(ModelSlot::Reference(ModelSlotRef::ModelJudge));
        config.models.subagent = Some(ModelSlot::Reference(ModelSlotRef::AutoResolve));
        let leaf = |model: &str| ModelConfig {
            provider: ConfiguredProvider::OpenAi,
            model: model.to_owned(),
            max_input_tokens: None,
            max_output_tokens: None,
            reasoning: None,
            tokens_per_minute: None,
        };
        config.models.model_judge = Some(ModelJudgeConfig {
            mode: ModelJudgeMode::FullTurn,
            default: leaf("gpt-default"),
            synthesis: leaf("gpt-synthesis"),
            panel: vec![leaf("gpt-panel-a"), leaf("gpt-panel-b")],
            panel_isolation: Default::default(),
        });

        let options = default_provider_build_options();
        let registry = build_model_registry(&config, &options).expect("model registry");
        let default_model = registry.default_model().expect("default model");
        let subagent_model = registry.subagent_model().expect("subagent model");

        assert_eq!(subagent_model.id, default_model.id);
        assert!(
            !registry
                .model_ids()
                .contains(&ModelId::from(SUBAGENT_MODEL_ID))
        );
        assert!(registry.full_turn_judge(&default_model.id).is_some());
        assert!(
            registry
                .full_turn_judge(&ModelId::from(SUBAGENT_MODEL_ID))
                .is_none()
        );

        HalterBuilder::default()
            .with_config(config)
            .with_resource_snapshot(ResourceSnapshot::empty())
            .build()
            .await
            .expect("build should succeed for auto-resolved subagents");
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

    fn default_provider_build_options() -> ProviderBuildOptions {
        ProviderBuildOptions {
            resilience_policy: None,
            provider_error_classifier: Arc::new(DefaultProviderErrorClassifier),
        }
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
            append_system_prompt: None,
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
            append_system_prompt: None,
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
            append_system_prompt: None,
        };
        assert_eq!(configured_system_prompt(&prompts), None);
    }

    #[test]
    fn apply_prompt_config_swaps_default_seed_for_coding_preset() {
        let prompts = PromptsConfig {
            preset: SystemPromptPreset::Coding,
            system_prompt: None,
            append_system_prompt: None,
        };
        let init = apply_prompt_config(&prompts, SessionInit::default());
        assert_eq!(
            seed_text(&init),
            halter_runtime::default_coding_agent_prompt()
        );
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
            append_system_prompt: None,
        };
        let init = SessionInit::default().with_system_prompt("explicit session prompt");
        let init = apply_prompt_config(&prompts, init);
        assert_eq!(seed_text(&init), "explicit session prompt");
    }

    #[test]
    fn apply_prompt_config_appends_after_default_base() {
        let prompts = PromptsConfig {
            append_system_prompt: Some("  house rules  ".to_owned()),
            ..PromptsConfig::default()
        };
        let init = apply_prompt_config(&prompts, SessionInit::default());

        assert_eq!(init.system_prompt_seed.len(), 2);
        assert_eq!(
            init.system_prompt_seed[0].text,
            halter_runtime::default_system_prompt()
        );
        assert_eq!(init.system_prompt_seed[1].text, "house rules");
        assert_eq!(init.system_prompt_seed[1].kind, PromptSegmentKind::System);
    }

    #[test]
    fn apply_prompt_config_ignores_blank_append() {
        let prompts = PromptsConfig {
            append_system_prompt: Some(" \n\t ".to_owned()),
            ..PromptsConfig::default()
        };
        let init = apply_prompt_config(&prompts, SessionInit::default());

        assert_eq!(init.system_prompt_seed.len(), 1);
        assert_eq!(seed_text(&init), halter_runtime::default_system_prompt());
    }

    #[test]
    fn apply_prompt_config_appends_after_coding_preset_base() {
        let prompts = PromptsConfig {
            preset: SystemPromptPreset::Coding,
            append_system_prompt: Some("house rules".to_owned()),
            ..PromptsConfig::default()
        };
        let init = apply_prompt_config(&prompts, SessionInit::default());

        assert_eq!(init.system_prompt_seed.len(), 2);
        assert_eq!(
            init.system_prompt_seed[0].text,
            halter_runtime::default_coding_agent_prompt()
        );
        assert_eq!(init.system_prompt_seed[1].text, "house rules");
    }

    #[test]
    fn apply_prompt_config_appends_after_system_prompt_override() {
        let prompts = PromptsConfig {
            system_prompt: Some("config base".to_owned()),
            append_system_prompt: Some("house rules".to_owned()),
            ..PromptsConfig::default()
        };
        let init = apply_prompt_config(&prompts, SessionInit::default());

        assert_eq!(init.system_prompt_seed.len(), 2);
        assert_eq!(init.system_prompt_seed[0].text, "config base");
        assert_eq!(init.system_prompt_seed[1].text, "house rules");
    }

    #[test]
    fn apply_prompt_config_stacks_config_append_before_session_append() {
        let prompts = PromptsConfig {
            append_system_prompt: Some("config rules".to_owned()),
            ..PromptsConfig::default()
        };
        let init = SessionInit::default().append_system_prompt("session rules");
        let init = apply_prompt_config(&prompts, init);
        let texts = init
            .system_prompt_seed
            .iter()
            .map(|segment| segment.text.as_str())
            .collect::<Vec<_>>();

        assert_eq!(
            texts,
            vec![
                halter_runtime::default_system_prompt(),
                "config rules",
                "session rules",
            ]
        );
    }

    #[test]
    fn apply_prompt_config_keeps_explicit_base_and_adds_config_append() {
        let prompts = PromptsConfig {
            system_prompt: Some("config base".to_owned()),
            append_system_prompt: Some("config rules".to_owned()),
            ..PromptsConfig::default()
        };
        let init = SessionInit::default()
            .with_system_prompt("explicit base")
            .append_system_prompt("session rules");
        let init = apply_prompt_config(&prompts, init);
        let texts = init
            .system_prompt_seed
            .iter()
            .map(|segment| segment.text.as_str())
            .collect::<Vec<_>>();

        assert_eq!(
            texts,
            vec!["explicit base", "config rules", "session rules"]
        );
    }
}
