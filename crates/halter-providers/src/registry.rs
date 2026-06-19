// pattern: Imperative Shell

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Context;
use halter_protocol::{ModelId, ProviderName, ResolvedModel};
use tracing::debug;

use crate::{FullTurnJudgePlan, Provider};

#[derive(Default, Clone)]
/// Registry of resolved models and provider adapters.
pub struct ModelRegistry {
    models: HashMap<String, ResolvedModel>,
    default_model: Option<ResolvedModel>,
    small_model: Option<ResolvedModel>,
    subagent_model: Option<ResolvedModel>,
    plan_model: Option<ResolvedModel>,
    providers: HashMap<String, Arc<dyn Provider>>,
    /// FullTurn model-judge plans keyed by the registry model id of the slot
    /// they back. A `Some` entry tells the runtime to fan a turn out to the
    /// panel as full sub-sessions before running this model.
    full_turn_judges: HashMap<String, Arc<FullTurnJudgePlan>>,
}

impl ModelRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the default model and make it resolvable by id.
    pub fn set_default_model(&mut self, model: ResolvedModel) {
        debug!(model_id = %model.id, provider = %model.provider, "setting default model");
        self.models.insert(model.id.0.clone(), model.clone());
        self.default_model = Some(model);
    }

    /// Resolve the default model.
    pub fn default_model(&self) -> anyhow::Result<ResolvedModel> {
        self.default_model
            .clone()
            .context("failed to resolve model: default model is not configured")
    }

    /// Set the optional small-task model and make it resolvable by id.
    pub fn set_small_model(&mut self, model: ResolvedModel) {
        debug!(model_id = %model.id, provider = %model.provider, "setting small model");
        self.models.insert(model.id.0.clone(), model.clone());
        self.small_model = Some(model);
    }

    /// Resolve the small-task model, falling back to the default model.
    pub fn small_model(&self) -> anyhow::Result<ResolvedModel> {
        self.small_model
            .clone()
            .or_else(|| self.default_model.clone())
            .context("failed to resolve model: small model is not configured")
    }

    /// Set the subagent model and make it resolvable by id.
    pub fn set_subagent_model(&mut self, model: ResolvedModel) {
        debug!(
            model_id = %model.id,
            provider = %model.provider,
            "setting subagent model"
        );
        self.models.insert(model.id.0.clone(), model.clone());
        self.subagent_model = Some(model);
    }

    /// Resolve the subagent model, falling back to the default model.
    pub fn subagent_model(&self) -> anyhow::Result<ResolvedModel> {
        self.subagent_model
            .clone()
            .or_else(|| self.default_model.clone())
            .context("failed to resolve model: subagent model is not configured")
    }

    /// Set the planning model and make it resolvable by id.
    pub fn set_plan_model(&mut self, model: ResolvedModel) {
        debug!(model_id = %model.id, provider = %model.provider, "setting plan model");
        self.models.insert(model.id.0.clone(), model.clone());
        self.plan_model = Some(model);
    }

    /// Resolve the planning model, falling back to the default model.
    pub fn plan_model(&self) -> anyhow::Result<ResolvedModel> {
        self.plan_model
            .clone()
            .or_else(|| self.default_model.clone())
            .context("failed to resolve model: plan model is not configured")
    }

    /// Make a model resolvable by id without binding it to a named role. Used
    /// for FullTurn model-judge panelists, which the runtime starts sub-sessions
    /// against but which are not the default/small/subagent model of any slot.
    pub fn register_model(&mut self, model: ResolvedModel) {
        debug!(model_id = %model.id, provider = %model.provider, "registering model");
        self.models.insert(model.id.0.clone(), model);
    }

    /// Resolve a concrete model id.
    pub fn model(&self, model_id: &ModelId) -> anyhow::Result<ResolvedModel> {
        debug!(model_id = %model_id, "resolving model");
        self.models
            .get(&model_id.0)
            .cloned()
            .with_context(|| format!("failed to resolve model: unknown model '{}'", model_id.0))
    }

    /// List registered model ids in stable order.
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

    /// Register a provider adapter by provider name.
    pub fn register_provider(&mut self, name: ProviderName, provider: Arc<dyn Provider>) {
        debug!(provider = %name, "registering provider");
        self.providers.insert(name.0, provider);
    }

    /// Resolve a provider adapter by provider name.
    pub fn provider(&self, name: &ProviderName) -> anyhow::Result<Arc<dyn Provider>> {
        debug!(provider = %name, "resolving provider");
        self.providers
            .get(&name.0)
            .cloned()
            .with_context(|| format!("failed to resolve provider: unknown provider '{}'", name.0))
    }

    /// Register a FullTurn model-judge plan for a slot's model id.
    pub fn register_full_turn_judge(&mut self, model_id: &ModelId, plan: Arc<FullTurnJudgePlan>) {
        debug!(
            model_id = %model_id,
            panel = plan.panel.len(),
            "registering full-turn model judge"
        );
        self.full_turn_judges.insert(model_id.0.clone(), plan);
    }

    /// Resolve the FullTurn model-judge plan backing a model id, if any.
    #[must_use]
    pub fn full_turn_judge(&self, model_id: &ModelId) -> Option<Arc<FullTurnJudgePlan>> {
        self.full_turn_judges.get(&model_id.0).cloned()
    }
}
