// pattern: Imperative Shell

use std::collections::HashSet;
use std::env;
use std::path::Path;
use std::sync::Arc;

use anyhow::Context;
use halter_config::{
    ConfiguredProvider, DEFAULT_MODEL_ID, HarnessConfig, PolicyConfig, ResolvedProviderConfig,
    SUBAGENT_MODEL_ID, SessionBackend, SessionsConfig, load_path, resolve_provider_runtime_config,
};
use halter_protocol::{ModelId, ModelRole, ProviderName, ResolvedModel, ResourceSnapshot};
use halter_providers::{AnthropicProvider, ModelRegistry, OpenAiProvider, OpenRouterProvider};
use halter_runtime::{
    DefaultContextManager, DefaultPromptAssembler, EventBus, HalterSession, ResourceHandle,
    RuntimeServices, SessionInit, SessionRuntime,
};
use halter_session::{InMemorySessionStore, SessionStore};
use halter_tools::{
    DefaultToolPolicy, PolicySettings, Tool, ToolRuntime, register_builtin_tools,
    register_subagent_tools,
};
use tracing::{debug, info};

use crate::{LoadedPlugin, LoadedSkill, ResourceCompiler};

#[cfg(feature = "sqlite")]
use halter_session::SqliteSessionStore;

#[derive(Default)]
pub struct HalterBuilder {
    config: HarnessConfig,
    resource_snapshot: Option<ResourceSnapshot>,
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
        let config = self.config;
        debug!("validating halter builder config");
        config.validate()?;

        if !self.loaded_skills.is_empty() || !self.loaded_plugins.is_empty() {
            anyhow::bail!(
                "failed to build halter runtime: loaded skills/plugins require a prebuilt resource snapshot"
            );
        }

        let snapshot = self.resource_snapshot.with_context(|| {
            "failed to build halter runtime: missing resource snapshot; use Halter::from_config_file or HalterBuilder::with_resource_snapshot"
        })?;

        let models = Arc::new(build_model_registry(&config)?);
        let tools = Arc::new(ToolRuntime::new());
        register_builtin_tools(&tools, &config.tools.enabled);
        for tool in self.tools {
            tools.register(tool);
        }

        let policy = Arc::new(DefaultToolPolicy::new(policy_from_config(&config.policy)));
        let session_backend = self
            .session_store
            .as_ref()
            .map(|_| "custom".to_owned())
            .unwrap_or_else(|| describe_session_backend(&config.sessions).to_owned());
        let sessions = match self.session_store {
            Some(store) => store,
            None => build_session_store(&config.sessions)?,
        };
        let services = Arc::new(RuntimeServices {
            resources: Arc::new(ResourceHandle::new(snapshot)),
            models,
            tools,
            sessions,
            policy: policy.clone(),
            prompt_assembler: Arc::new(DefaultPromptAssembler),
            context_manager: Arc::new(DefaultContextManager::new(
                config.context.max_context_messages,
            )),
            event_bus: Arc::new(EventBus::default()),
            max_tool_output_bytes: config.policy.max_tool_output_bytes,
            shell_timeout_secs: config.policy.shell.timeout_secs,
        });
        let runtime = SessionRuntime::new(services.clone());
        register_subagent_tools(
            &services.tools,
            runtime.subagent_control(),
            &config.tools.enabled,
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
        let snapshot = ResourceCompiler::from_config(&config).compile().await?;
        Self::from_config(config, snapshot).await
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

    pub async fn from_resource_snapshot(_snapshot: ResourceSnapshot) -> anyhow::Result<Self> {
        anyhow::bail!(
            "failed to build halter runtime: missing config; use Halter::from_config_file or Halter::from_config"
        )
    }

    pub async fn new_session(&self, init: SessionInit) -> anyhow::Result<HalterSession> {
        self.runtime.new_session(init).await
    }

    pub fn replace_resources(&self, snapshot: ResourceSnapshot) {
        self.runtime.replace_resources(snapshot);
    }

    #[must_use]
    pub fn runtime(&self) -> &SessionRuntime {
        &self.runtime
    }

    #[must_use]
    pub fn config(&self) -> &HarnessConfig {
        &self.config
    }
}

fn build_session_store(config: &SessionsConfig) -> anyhow::Result<Arc<dyn SessionStore>> {
    match config.backend {
        SessionBackend::Memory => Ok(Arc::new(InMemorySessionStore::default())),
        SessionBackend::Sqlite => build_sqlite_session_store(config),
        SessionBackend::FlatFile => {
            anyhow::bail!(
                "failed to build halter runtime: session backend 'flat_file' is not implemented"
            );
        }
        SessionBackend::Postgres => {
            anyhow::bail!(
                "failed to build halter runtime: session backend 'postgres' is not implemented"
            );
        }
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

#[cfg(not(feature = "sqlite"))]
fn build_sqlite_session_store(_config: &SessionsConfig) -> anyhow::Result<Arc<dyn SessionStore>> {
    anyhow::bail!(
        "failed to build halter runtime: session backend 'sqlite' requires the 'sqlite' cargo feature"
    );
}

fn describe_session_backend(config: &SessionsConfig) -> &'static str {
    match config.backend {
        SessionBackend::Memory => "memory",
        SessionBackend::FlatFile => "flat_file",
        SessionBackend::Sqlite => "sqlite",
        SessionBackend::Postgres => "postgres",
    }
}

fn build_model_registry(config: &HarnessConfig) -> anyhow::Result<ModelRegistry> {
    let mut registry = ModelRegistry::new();
    let default_config = config.default_model()?;
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
    };

    debug!(
        default_provider = %default_model.provider,
        default_model = %default_model.model,
        subagent_provider = %subagent_model.provider,
        subagent_model = %subagent_model.model,
        "building model registry"
    );

    let mut registered_providers = HashSet::new();
    for (provider_name, configured_provider) in [
        (&default_model.provider, default_config.provider),
        (&subagent_model.provider, subagent_config.provider),
    ] {
        if !registered_providers.insert(provider_name.0.clone()) {
            continue;
        }
        let resolved = resolve_selected_provider_config(config, configured_provider)?;
        registry.register_provider(provider_name.clone(), build_provider(&resolved)?);
    }
    registry.set_default_model(default_model);
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
    PolicySettings {
        allowed_write_roots: config.allowed_write_roots.clone(),
        max_read_bytes: config.max_read_bytes,
        max_tool_output_bytes: config.max_tool_output_bytes,
        shell_enabled: config.shell.enabled,
        allowed_shell_commands: config.shell.allow.clone(),
        shell_timeout_secs: config.shell.timeout_secs,
        network_enabled: config.network.enabled,
        allowed_hosts: config.network.allowed_hosts.clone(),
        max_subagent_depth: config.max_subagent_depth,
        max_concurrent_subagents: config.max_concurrent_subagents,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use halter_config::{ModelConfig, ProviderConfig};
    use halter_protocol::ReasoningEffort;
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
            state: Option<halter_protocol::SessionState>,
            events: Vec<halter_protocol::SessionEvent>,
        ) -> anyhow::Result<Vec<halter_protocol::SessionEvent>> {
            self.commit_calls.fetch_add(1, Ordering::SeqCst);
            self.inner.commit(session_id, snapshot, state, events).await
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
