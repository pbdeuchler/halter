// pattern: Imperative Shell

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Context;
use halter_protocol::{ModelId, ProviderName, ResolvedModel};
use tracing::debug;

use crate::Provider;

#[derive(Default, Clone)]
pub struct ModelRegistry {
    models: HashMap<String, ResolvedModel>,
    default_model: Option<ResolvedModel>,
    small_model: Option<ResolvedModel>,
    subagent_model: Option<ResolvedModel>,
    plan_model: Option<ResolvedModel>,
    providers: HashMap<String, Arc<dyn Provider>>,
}

impl ModelRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_default_model(&mut self, model: ResolvedModel) {
        debug!(model_id = %model.id, provider = %model.provider, "setting default model");
        self.models.insert(model.id.0.clone(), model.clone());
        self.default_model = Some(model);
    }

    pub fn default_model(&self) -> anyhow::Result<ResolvedModel> {
        self.default_model
            .clone()
            .context("failed to resolve model: default model is not configured")
    }

    pub fn set_small_model(&mut self, model: ResolvedModel) {
        debug!(model_id = %model.id, provider = %model.provider, "setting small model");
        self.models.insert(model.id.0.clone(), model.clone());
        self.small_model = Some(model);
    }

    pub fn small_model(&self) -> anyhow::Result<ResolvedModel> {
        self.small_model
            .clone()
            .or_else(|| self.default_model.clone())
            .context("failed to resolve model: small model is not configured")
    }

    pub fn set_subagent_model(&mut self, model: ResolvedModel) {
        debug!(
            model_id = %model.id,
            provider = %model.provider,
            "setting subagent model"
        );
        self.models.insert(model.id.0.clone(), model.clone());
        self.subagent_model = Some(model);
    }

    pub fn subagent_model(&self) -> anyhow::Result<ResolvedModel> {
        self.subagent_model
            .clone()
            .or_else(|| self.default_model.clone())
            .context("failed to resolve model: subagent model is not configured")
    }

    pub fn set_plan_model(&mut self, model: ResolvedModel) {
        debug!(model_id = %model.id, provider = %model.provider, "setting plan model");
        self.models.insert(model.id.0.clone(), model.clone());
        self.plan_model = Some(model);
    }

    pub fn plan_model(&self) -> anyhow::Result<ResolvedModel> {
        self.plan_model
            .clone()
            .or_else(|| self.default_model.clone())
            .context("failed to resolve model: plan model is not configured")
    }

    pub fn model(&self, model_id: &ModelId) -> anyhow::Result<ResolvedModel> {
        debug!(model_id = %model_id, "resolving model");
        self.models
            .get(&model_id.0)
            .cloned()
            .with_context(|| format!("failed to resolve model: unknown model '{}'", model_id.0))
    }

    #[must_use]
    pub fn model_ids(&self) -> Vec<ModelId> {
        let mut model_ids = self
            .models
            .keys()
            .cloned()
            .map(ModelId::from)
            .collect::<Vec<_>>();
        model_ids.sort_by(|left, right| left.0.cmp(&right.0));
        model_ids
    }

    pub fn register_provider(&mut self, name: ProviderName, provider: Arc<dyn Provider>) {
        debug!(provider = %name, "registering provider");
        self.providers.insert(name.0, provider);
    }

    pub fn provider(&self, name: &ProviderName) -> anyhow::Result<Arc<dyn Provider>> {
        debug!(provider = %name, "resolving provider");
        self.providers
            .get(&name.0)
            .cloned()
            .with_context(|| format!("failed to resolve provider: unknown provider '{}'", name.0))
    }
}
