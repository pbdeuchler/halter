use std::path::{Component, Path, PathBuf};

use halter_config::{
    ConfiguredProvider, ContextConfig, HarnessConfig, ModelConfig, ModelJudgeConfig,
    ModelJudgeMode, ModelSlot, ModelSlotRef, ModelsConfig, NetworkPolicyConfig, PanelIsolation,
    PolicyConfig, ProviderConfig, ProvidersConfig, ResourcesConfig, RuntimeConfig, SearchRoots,
    SessionsConfig, ShellPolicyConfig, ToolsConfig,
};
use halter_protocol::{PruneSignalThreshold, ReasoningEffort};

pub(crate) fn default_factory_config() -> HarnessConfig {
    HarnessConfig {
        version: 1,
        providers: ProvidersConfig {
            openai: Some(ProviderConfig::default()),
            anthropic: None,
            openrouter: Some(ProviderConfig::default()),
        },
        models: factory_models(),
        resources: ResourcesConfig {
            skills: SearchRoots {
                roots: vec![PathBuf::from("./.agent/skills")],
            },
            plugins: SearchRoots {
                roots: vec![PathBuf::from("./.agent/plugins")],
            },
        },
        context: ContextConfig {
            compaction_threshold: 230_000,
            pre_compaction_target: 150_000,
            prune_signal_threshold: PruneSignalThreshold::Low,
        },
        tools: ToolsConfig {
            enabled: JUDGE_EXAMPLE_TOOLS.iter().map(|&s| s.to_owned()).collect(),
        },
        policy: PolicyConfig {
            allowed_write_roots: vec![PathBuf::from("./"), PathBuf::from("/tmp/halter")],
            max_read_bytes: 1_048_576,
            max_subagent_depth: 3,
            max_concurrent_subagents: 8,
            shell: ShellPolicyConfig {
                enabled: true,
                allow: JUDGE_EXAMPLE_SHELL_ALLOWLIST
                    .iter()
                    .map(|&s| s.to_owned())
                    .collect(),
                timeout_secs: 30,
            },
            network: NetworkPolicyConfig {
                enabled: true,
                ..NetworkPolicyConfig::default()
            },
        },
        sessions: SessionsConfig::default(),
        runtime: RuntimeConfig {
            traces_dir: Some(PathBuf::from("~/.halter/traces/")),
            ..RuntimeConfig::default()
        },
        ..HarnessConfig::default()
    }
}

pub(crate) fn factory_models() -> ModelsConfig {
    ModelsConfig {
        default: Some(ModelSlot::Reference(ModelSlotRef::ModelJudge)),
        subagent: Some(ModelSlot::Reference(ModelSlotRef::AutoResolve)),
        small: Some(ModelConfig {
            provider: ConfiguredProvider::OpenRouter,
            model: "z-ai/glm-5.2".to_owned(),
            max_input_tokens: None,
            max_output_tokens: None,
            reasoning: Some(ReasoningEffort::Medium),
            tokens_per_minute: Some(500_000),
        }),
        model_judge: Some(ModelJudgeConfig {
            mode: ModelJudgeMode::FullTurn,
            default: ModelConfig {
                provider: ConfiguredProvider::OpenRouter,
                model: "z-ai/glm-5.2".to_owned(),
                max_input_tokens: None,
                max_output_tokens: None,
                reasoning: Some(ReasoningEffort::Xhigh),
                tokens_per_minute: Some(500_000),
            },
            synthesis: ModelConfig {
                provider: ConfiguredProvider::OpenRouter,
                model: "google/gemma-4-31b-it".to_owned(),
                max_input_tokens: None,
                max_output_tokens: None,
                reasoning: Some(ReasoningEffort::High),
                tokens_per_minute: Some(500_000),
            },
            panel: vec![
                ModelConfig {
                    provider: ConfiguredProvider::OpenRouter,
                    model: "minimax/minimax-m3".to_owned(),
                    max_input_tokens: None,
                    max_output_tokens: None,
                    reasoning: Some(ReasoningEffort::Xhigh),
                    tokens_per_minute: Some(500_000),
                },
                ModelConfig {
                    provider: ConfiguredProvider::OpenRouter,
                    model: "nvidia/nemotron-3-ultra-550b-a55b".to_owned(),
                    max_input_tokens: None,
                    max_output_tokens: None,
                    reasoning: Some(ReasoningEffort::Xhigh),
                    tokens_per_minute: Some(500_000),
                },
                // ModelConfig {
                //     provider: ConfiguredProvider::OpenRouter,
                //     model: "moonshotai/kimi-k2.6".to_owned(),
                //     max_input_tokens: None,
                //     max_output_tokens: None,
                //     reasoning: Some(ReasoningEffort::Xhigh),
                //     tokens_per_minute: Some(500_000),
                // },
                ModelConfig {
                    provider: ConfiguredProvider::OpenRouter,
                    model: "qwen/qwen3.6-27b".to_owned(),
                    max_input_tokens: None,
                    max_output_tokens: None,
                    reasoning: Some(ReasoningEffort::Xhigh),
                    tokens_per_minute: Some(500_000),
                },
            ],
            panel_isolation: PanelIsolation::ReadOnly,
        }),
    }
}

pub(crate) const JUDGE_EXAMPLE_TOOLS: [&str; 17] = [
    "read",
    "glob",
    "grep",
    "profile",
    "write",
    "edit",
    "shell",
    "process",
    "task",
    "pty",
    "ast_grep",
    "image",
    "wait_agent",
    "spawn_agent",
    "send_input",
    "close_agent",
    "browser",
];

pub(crate) const JUDGE_EXAMPLE_SHELL_ALLOWLIST: [&str; 17] = [
    "git", "cargo", "rg", "ls", "find", "python", "python3", "pwd", "echo", "date", "gh", "which",
    "sort", "nl", "sed", "wc", "head",
];

pub(crate) fn add_worktree_policy(config: &mut HarnessConfig, worktree: &Path) {
    absolutize_relative_roots(&mut config.policy.allowed_write_roots, worktree);
    if !config
        .policy
        .allowed_write_roots
        .iter()
        .any(|root| root == worktree)
    {
        config
            .policy
            .allowed_write_roots
            .push(worktree.to_path_buf());
    }
    absolutize_relative_roots(&mut config.resources.skills.roots, worktree);
    absolutize_relative_roots(&mut config.resources.plugins.roots, worktree);
}

pub(crate) fn absolutize_relative_roots(roots: &mut [PathBuf], worktree: &Path) {
    for root in roots {
        if root.is_relative() && !path_starts_with_tilde(root) {
            *root = if root == Path::new(".") || root == Path::new("./") {
                worktree.to_path_buf()
            } else {
                worktree.join(&root)
            };
        }
    }
}

pub(crate) fn path_starts_with_tilde(path: &Path) -> bool {
    path.components()
        .next()
        .is_some_and(|component| matches!(component, Component::Normal(value) if value == "~"))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    #[test]
    pub(crate) fn factory_models_equals_pre_refactor_values() {
        let expected = ModelsConfig {
            default: Some(ModelSlot::Reference(ModelSlotRef::ModelJudge)),
            subagent: Some(ModelSlot::Reference(ModelSlotRef::AutoResolve)),
            small: Some(ModelConfig {
                provider: ConfiguredProvider::OpenRouter,
                model: "z-ai/glm-5.2".to_owned(),
                max_input_tokens: None,
                max_output_tokens: None,
                reasoning: Some(ReasoningEffort::Medium),
                tokens_per_minute: Some(500_000),
            }),
            model_judge: Some(ModelJudgeConfig {
                mode: ModelJudgeMode::FullTurn,
                default: ModelConfig {
                    provider: ConfiguredProvider::OpenRouter,
                    model: "z-ai/glm-5.2".to_owned(),
                    max_input_tokens: None,
                    max_output_tokens: None,
                    reasoning: Some(ReasoningEffort::Xhigh),
                    tokens_per_minute: Some(500_000),
                },
                synthesis: ModelConfig {
                    provider: ConfiguredProvider::OpenRouter,
                    model: "google/gemma-4-31b-it".to_owned(),
                    max_input_tokens: None,
                    max_output_tokens: None,
                    reasoning: Some(ReasoningEffort::High),
                    tokens_per_minute: Some(500_000),
                },
                panel: vec![
                    ModelConfig {
                        provider: ConfiguredProvider::OpenRouter,
                        model: "minimax/minimax-m3".to_owned(),
                        max_input_tokens: None,
                        max_output_tokens: None,
                        reasoning: Some(ReasoningEffort::Xhigh),
                        tokens_per_minute: Some(500_000),
                    },
                    ModelConfig {
                        provider: ConfiguredProvider::OpenRouter,
                        model: "nvidia/nemotron-3-ultra-550b-a55b".to_owned(),
                        max_input_tokens: None,
                        max_output_tokens: None,
                        reasoning: Some(ReasoningEffort::Xhigh),
                        tokens_per_minute: Some(500_000),
                    },
                    ModelConfig {
                        provider: ConfiguredProvider::OpenRouter,
                        model: "qwen/qwen3.6-27b".to_owned(),
                        max_input_tokens: None,
                        max_output_tokens: None,
                        reasoning: Some(ReasoningEffort::Xhigh),
                        tokens_per_minute: Some(500_000),
                    },
                ],
                panel_isolation: PanelIsolation::ReadOnly,
            }),
        };
        assert_eq!(factory_models(), expected);
    }

    #[test]
    pub(crate) fn default_factory_config_matches_judge_example_tool_and_shell_lists() {
        let config = default_factory_config();
        let tools: HashSet<String> = config.tools.enabled.into_iter().collect();
        let expected_tools: HashSet<String> =
            JUDGE_EXAMPLE_TOOLS.iter().map(|s| s.to_string()).collect();
        assert_eq!(tools, expected_tools);

        let allow: HashSet<String> = config.policy.shell.allow.into_iter().collect();
        let expected_allow: HashSet<String> = JUDGE_EXAMPLE_SHELL_ALLOWLIST
            .iter()
            .map(|s| s.to_string())
            .collect();
        assert_eq!(allow, expected_allow);
        assert!(config.policy.network.enabled);
    }

    #[tokio::test]
    pub(crate) async fn add_worktree_policy_absolutizes_relative_resource_roots_idempotently() {
        let worktree = tempfile::tempdir().unwrap();
        let worktree_path = worktree.path().canonicalize().unwrap();
        let mut config = default_factory_config();
        config.resources.skills.roots = vec![PathBuf::from("~/skills")];
        add_worktree_policy(&mut config, &worktree_path);

        assert_eq!(
            config.resources.skills.roots,
            vec![PathBuf::from("~/skills")],
            "tilde-prefixed resource roots must not be absolutized"
        );

        for root in &config.policy.allowed_write_roots {
            assert!(
                root.is_absolute(),
                "allowed_write_roots must be absolute: {root:?}"
            );
        }
        assert!(
            config.policy.allowed_write_roots.contains(&worktree_path),
            "worktree must be present in allowed_write_roots"
        );
        for root in &config.resources.plugins.roots {
            assert!(
                root.is_absolute(),
                "plugins roots must be absolute: {root:?}"
            );
        }

        // Running a second time must be idempotent.
        add_worktree_policy(&mut config, &worktree_path);
        assert_eq!(
            config
                .policy
                .allowed_write_roots
                .iter()
                .filter(|root| **root == worktree_path)
                .count(),
            1,
            "worktree must appear exactly once"
        );
    }
}
