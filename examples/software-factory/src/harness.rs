use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use halter::prelude::*;
use halter_config::{HarnessConfig, ModelSlot};
use halter_protocol::ReasoningEffort;
use halter_tools::Tool;
use tracing::info;

use crate::config::add_worktree_policy;
use crate::core::ModelSpec;

pub(crate) async fn build_judge_harness(
    config: &HarnessConfig,
    worktree: &Path,
    issue_tool: Arc<dyn Tool>,
) -> anyhow::Result<Halter> {
    info!("building model judge harness");
    let mut config = config.clone();
    add_worktree_policy(&mut config, worktree);
    if !config
        .tools
        .enabled
        .iter()
        .any(|tool| tool == "github_issue")
    {
        config.tools.enabled.push("github_issue".to_owned());
    }
    let resources = ResourceCompiler::from_config(&config).compile().await?;
    let harness = Halter::builder()
        .with_config(config)
        .with_compiled_resources(resources)
        .with_tool(issue_tool)
        .build()
        .await?;
    info!("built model judge harness");
    Ok(harness)
}

pub(crate) async fn build_model_harness(
    config: &HarnessConfig,
    role: &str,
    model: ModelSpec,
    reasoning: ReasoningEffort,
    worktree: &Path,
) -> anyhow::Result<Halter> {
    info!(
        role,
        provider = ?model.provider,
        model = %model.model,
        reasoning = ?reasoning,
        "building model harness"
    );
    let mut config = config.clone();
    add_worktree_policy(&mut config, worktree);
    let model = model.into_model_config(reasoning, Some(230_000), Some(16_384));
    config.models.default = Some(ModelSlot::Inline(model.clone()));
    config.models.small = Some(model.clone());
    config.models.subagent = Some(ModelSlot::Inline(model));
    let resources = ResourceCompiler::from_config(&config).compile().await?;
    let harness = Halter::from_compiled_resources(config, resources).await?;
    info!(role, "built model harness");
    Ok(harness)
}

pub(crate) async fn shutdown_all<'a>(harnesses: impl IntoIterator<Item = &'a Halter>) {
    for harness in harnesses {
        let _ = harness.shutdown(Duration::from_secs(10)).await;
    }
}
