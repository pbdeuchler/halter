// pattern: Imperative Shell

use std::collections::HashMap;
use std::env;
use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use halter_config::{
    ConfiguredProvider, DEFAULT_MODEL_ID, HarnessConfig, PolicyConfig, ResolvedProviderConfig,
    SMALL_MODEL_ID, SUBAGENT_MODEL_ID, SessionBackend, SessionsConfig, load_path,
    resolve_provider_runtime_config,
};
use halter_hooks::{Hook, Hooks, RegisteredHookPriority, RegisteredHooks};
use halter_protocol::{
    HookWarning, ModelId, ModelRole, ProviderName, ResolvedModel, ResourceSnapshot,
};
use halter_providers::{AnthropicProvider, ModelRegistry, OpenAiProvider, OpenRouterProvider};
use halter_runtime::{
    DefaultContextManager, DefaultPromptAssembler, EventBus, HalterSession, ResourceHandle,
    RuntimeServices, SessionInit, SessionRuntime,
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
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_config(mut self, config: HarnessConfig) -> Self {
        self.config = config;
        self
    }

    #[must_use]
    pub fn with_resource_snapshot(mut self, snapshot: ResourceSnapshot) -> Self {
        self.resource_snapshot = Some(snapshot);
        self
    }

    #[must_use]
    pub fn with_compiled_resources(mut self, resources: CompiledResources) -> Self {
        self.resource_snapshot = Some(resources.snapshot);
        self.resource_hooks = Some(resources.hooks);
        self.resource_hook_warnings = resources.hook_warnings;
        self
    }

    #[must_use]
    pub fn with_loaded_skills(mut self, skills: Vec<LoadedSkill>) -> Self {
        self.loaded_skills = skills;
        self
    }

    #[must_use]
    pub fn with_loaded_plugins(mut self, plugins: Vec<LoadedPlugin>) -> Self {
        self.loaded_plugins = plugins;
        self
    }

    #[must_use]
    pub fn with_plugin_hook(mut self, plugin_id: halter_protocol::PluginId, hook: Hook) -> Self {
        self.registered_hooks
            .register(plugin_id, RegisteredHookPriority::AfterPlugins, hook);
        self
    }

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

    #[must_use]
    pub fn with_tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    #[must_use]
    pub fn with_session_store(mut self, store: Arc<dyn SessionStore>) -> Self {
        self.session_store = Some(store);
        self
    }

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
            turn_registry: Arc::new(halter_runtime::TurnRegistry::new()),
            shell_timeout_secs: config.policy.shell.timeout_secs,
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
pub struct Halter {
    config: HarnessConfig,
    runtime: SessionRuntime,
}

impl Halter {
    #[must_use]
    pub fn builder() -> HalterBuilder {
        HalterBuilder::default()
    }

    pub async fn from_config_file(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        debug!(path = %path.as_ref().display(), "building halter from config file");
        let config = load_path(path).await?;
        let resources = ResourceCompiler::from_config(&config).compile().await?;
        Self::from_compiled_resources(config, resources).await
    }

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

    pub async fn new_session(&self, init: SessionInit) -> anyhow::Result<HalterSession> {
        self.runtime.new_session(init).await
    }

    pub fn replace_resources(&self, resources: CompiledResources) {
        self.runtime.replace_resources(
            resources.snapshot,
            resources.hooks,
            resources.hook_warnings,
        );
    }

    #[must_use]
    pub fn runtime(&self) -> &SessionRuntime {
        &self.runtime
    }

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
    let default_config = config.default_model()?;
    let small_config = config.small_model().unwrap_or(default_config);
    let subagent_config = config.subagent_model().unwrap_or(default_config);
    let default_model = ResolvedModel {
        role: ModelRole::default(),
        id: ModelId::from(DEFAULT_MODEL_ID),
        provider: ProviderName::from(default_config.provider.to_string()),
        provider_kind: default_config.provider.provider_kind(),
        api_kind: default_config.provider.api_kind(),
        model: default_config.model.clone(),
        max_input_tokens: default_config.max_input_tokens,
        max_output_tokens: default_config.max_output_tokens,
        reasoning: default_config.reasoning,
        tokens_per_minute: default_config.tokens_per_minute,
    };
    let small_model = ResolvedModel {
        role: ModelRole::small(),
        id: ModelId::from(SMALL_MODEL_ID),
        provider: ProviderName::from(small_config.provider.to_string()),
        provider_kind: small_config.provider.provider_kind(),
        api_kind: small_config.provider.api_kind(),
        model: small_config.model.clone(),
        max_input_tokens: small_config.max_input_tokens,
        max_output_tokens: small_config.max_output_tokens,
        reasoning: small_config.reasoning,
        tokens_per_minute: small_config.tokens_per_minute,
    };
    let subagent_model = ResolvedModel {
        role: ModelRole::subagent(),
        id: ModelId::from(SUBAGENT_MODEL_ID),
        provider: ProviderName::from(subagent_config.provider.to_string()),
        provider_kind: subagent_config.provider.provider_kind(),
        api_kind: subagent_config.provider.api_kind(),
        model: subagent_config.model.clone(),
        max_input_tokens: subagent_config.max_input_tokens,
        max_output_tokens: subagent_config.max_output_tokens,
        reasoning: subagent_config.reasoning,
        tokens_per_minute: subagent_config.tokens_per_minute,
    };

    debug!(
        default_provider = %default_model.provider,
        default_model = %default_model.model,
        small_provider = %small_model.provider,
        small_model = %small_model.model,
        subagent_provider = %subagent_model.provider,
        subagent_model = %subagent_model.model,
        "building model registry"
    );

    let mut registered_providers: HashMap<String, ResolvedProviderConfig> = HashMap::new();
    for (role_label, provider_name, configured_provider) in [
        ("default", &default_model.provider, default_config.provider),
        ("small", &small_model.provider, small_config.provider),
        (
            "subagent",
            &subagent_model.provider,
            subagent_config.provider,
        ),
    ] {
        let resolved = resolve_selected_provider_config(config, configured_provider)?;
        if let Some(existing) = registered_providers.get(&provider_name.0) {
            anyhow::ensure!(
                existing == &resolved,
                "provider '{name}' is used by multiple roles with divergent per-role configuration; \
                 the {role_label} role resolved to base_url '{new_base}' but an earlier role \
                 resolved the same provider to base_url '{old_base}' (api key differences also \
                 trigger this error). Consolidate the config or use distinct providers.",
                name = provider_name,
                role_label = role_label,
                new_base = resolved.base_url,
                old_base = existing.base_url,
            );
            continue;
        }
        registered_providers.insert(provider_name.0.clone(), resolved.clone());
        registry.register_provider(provider_name.clone(), build_provider(&resolved)?);
    }
    registry.set_default_model(default_model);
    registry.set_small_model(small_model);
    registry.set_subagent_model(subagent_model);

    Ok(registry)
}

fn build_provider(
    provider: &ResolvedProviderConfig,
) -> anyhow::Result<Arc<dyn halter_providers::Provider>> {
    debug!(
        provider = %provider.provider,
        base_url = %provider.base_url,
        "constructing provider client"
    );
    let provider: Arc<dyn halter_providers::Provider> = match provider.provider {
        ConfiguredProvider::Anthropic => Arc::new(AnthropicProvider::new(
            provider.api_key.clone(),
            provider.base_url.clone(),
        )),
        ConfiguredProvider::OpenAi => Arc::new(OpenAiProvider::new(
            provider.api_key.clone(),
            provider.base_url.clone(),
        )),
        ConfiguredProvider::OpenRouter => Arc::new(OpenRouterProvider::new(
            provider.api_key.clone(),
            provider.base_url.clone(),
        )),
    };
    Ok(provider)
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
    use halter_config::{ModelConfig, ProviderConfig};
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
    async fn builder_requires_provider_api_key_when_not_configured() {
        let error = resolve_selected_provider_config_with(
            &openai_config(None),
            ConfiguredProvider::OpenAi,
            |_| Ok(None),
        )
        .expect_err("provider resolution should fail");

        assert!(error.to_string().contains("OPENAI_API_KEY"));
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
        config.models.subagent = Some(ModelConfig {
            provider: ConfiguredProvider::OpenAi,
            model: "gpt-5".to_owned(),
            max_input_tokens: Some(200_000),
            max_output_tokens: Some(8_192),
            reasoning: None,
            tokens_per_minute: None,
        });

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
        config.models.default = Some(ModelConfig {
            provider: ConfiguredProvider::OpenAi,
            model: "gpt-5".to_owned(),
            max_input_tokens: Some(200_000),
            max_output_tokens: Some(8_192),
            reasoning: Some(ReasoningEffort::Medium),
            tokens_per_minute: None,
        });
        config.providers.openai = Some(ProviderConfig {
            base_url: None,
            api_key: api_key.map(ToOwned::to_owned),
        });
        config
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
}
