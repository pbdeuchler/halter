// pattern: Functional Core

mod config;
mod engine;
mod merge;
mod sdk;

pub use config::{
    AgentHookConfig, CommandHookConfig, HookEventName, HookHandler, HookHandlerConfig,
    HookMatcherGroup, HookShell, HooksFile, HooksLoadWarning, HttpHookConfig, PromptHookConfig,
};
pub use engine::{
    ConfiguredHandler, ConfiguredHandlerConfig, HOOK_PROTOCOL_VERSION, HookDispatchOutcome,
    HookDispatchRequest, HookRegistrySource, Hooks, PreparedHookDispatch,
};
pub use merge::{
    HandlerPriority, HandlerPriorityGroup, HookDecision, HookMergedOutcome, HookOutput,
    HookSpecificOutput, MergeInput, PermissionDecision, merge_outputs, summary_entries,
};
pub use sdk::{
    Hook, HookCallback, HookCallbackFuture, HookFunctionFactory, HookInput, HookKind, HookResponse,
    RegisteredHook, RegisteredHookPriority, RegisteredHooks,
};
