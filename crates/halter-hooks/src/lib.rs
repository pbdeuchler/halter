// pattern: Functional Core

mod config;
mod engine;
mod merge;

pub use config::{
    AgentHookConfig, CommandHookConfig, HookEventName, HookHandler, HookHandlerConfig,
    HookMatcherGroup, HookShell, HooksFile, HooksLoadWarning, HttpHookConfig, PromptHookConfig,
};
pub use engine::{
    ConfiguredHandler, HOOK_PROTOCOL_VERSION, HookDispatchOutcome, HookDispatchRequest,
    HookRegistrySource, Hooks, PreparedHookDispatch,
};
pub use merge::{
    HandlerPriority, HookDecision, HookMergedOutcome, HookOutput, HookSpecificOutput, MergeInput,
    PermissionDecision, merge_outputs, summary_entries,
};
