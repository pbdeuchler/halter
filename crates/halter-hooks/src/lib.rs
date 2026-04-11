// pattern: Functional Core

mod config;
mod engine;
mod merge;

pub use config::{
    CommandHookConfig, HookEventName, HookHandler, HookHandlerConfig, HookMatcherGroup,
    HookShell, HooksFile, HooksLoadWarning,
};
pub use engine::{
    HookDispatchOutcome, HookDispatchRequest, HookRegistrySource, Hooks, PreparedHookDispatch,
};
pub use merge::{HookMergedOutcome, PermissionDecision};
