// pattern: Functional Core

use std::fmt;
use std::future::Future;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use futures::future::BoxFuture;
use halter_protocol::{HookHandlerType, PluginId};
use serde::de::DeserializeOwned;
use serde_json::Value;

use crate::config::HookEventName;
use crate::merge::{HookDecision, HookOutput, HookSpecificOutput, PermissionDecision};

/// Boxed future returned by an SDK hook callback.
pub type HookCallbackFuture = BoxFuture<'static, anyhow::Result<HookResponse>>;
/// Shared callback used by SDK-registered hooks.
pub type HookCallback = Arc<dyn Fn(HookInput) -> HookCallbackFuture + Send + Sync>;
/// Factory for hooks that need a fresh callback instance per dispatch.
pub type HookFunctionFactory = Arc<dyn Fn() -> HookCallback + Send + Sync>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
/// Relative priority for SDK hooks compared with plugin-file hooks.
pub enum RegisteredHookPriority {
    /// Run before hooks loaded from plugin files.
    BeforePlugins,
    /// Run after hooks loaded from plugin files.
    #[default]
    AfterPlugins,
}

#[derive(Debug, Clone)]
/// Input passed to an SDK hook callback.
pub struct HookInput {
    pub event_name: HookEventName,
    pub matcher_value: Option<String>,
    pub payload: Value,
}

impl HookInput {
    /// Return a raw JSON payload field.
    #[must_use]
    pub fn field(&self, key: &str) -> Option<&Value> {
        self.payload.get(key)
    }

    /// Return a string payload field.
    #[must_use]
    pub fn string_field(&self, key: &str) -> Option<&str> {
        self.field(key).and_then(Value::as_str)
    }

    /// Tool name for tool-related hook events.
    #[must_use]
    pub fn tool_name(&self) -> Option<&str> {
        self.string_field("tool_name")
    }

    /// Tool use id for tool-related hook events.
    #[must_use]
    pub fn tool_use_id(&self) -> Option<&str> {
        self.string_field("tool_use_id")
    }

    /// Decode the entire hook payload into a typed struct.
    pub fn decode<T: DeserializeOwned>(&self) -> anyhow::Result<T> {
        serde_json::from_value(self.payload.clone()).context("failed to decode hook input")
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
/// Builder-style response returned by SDK hooks.
pub struct HookResponse {
    output: HookOutput,
}

impl HookResponse {
    /// Return no changes and allow execution to continue.
    #[must_use]
    pub fn passthrough() -> Self {
        Self::default()
    }

    /// Block the current operation with a reason.
    #[must_use]
    pub fn block(reason: impl Into<String>) -> Self {
        Self {
            output: HookOutput {
                decision: Some(HookDecision::Block),
                reason: Some(reason.into()),
                ..HookOutput::default()
            },
        }
    }

    /// Stop the current turn with a reason.
    #[must_use]
    pub fn stop(reason: impl Into<String>) -> Self {
        Self {
            output: HookOutput {
                continue_execution: Some(false),
                stop_reason: Some(reason.into()),
                ..HookOutput::default()
            },
        }
    }

    /// Add a system message to the merged hook outcome.
    #[must_use]
    pub fn with_system_message(mut self, message: impl Into<String>) -> Self {
        self.output.system_message = Some(message.into());
        self
    }

    /// Add context to the next model request.
    #[must_use]
    pub fn with_additional_context(mut self, context: impl Into<String>) -> Self {
        self.output
            .hook_specific_output
            .get_or_insert_with(HookSpecificOutput::default)
            .additional_context = Some(context.into());
        self
    }

    /// Replace the tool input seen by downstream execution.
    #[must_use]
    pub fn with_updated_input(mut self, input: Value) -> Self {
        self.output
            .hook_specific_output
            .get_or_insert_with(HookSpecificOutput::default)
            .updated_input = Some(input);
        self
    }

    /// Replace the tool output seen by downstream execution.
    #[must_use]
    pub fn with_updated_output(mut self, output: Value) -> Self {
        self.output
            .hook_specific_output
            .get_or_insert_with(HookSpecificOutput::default)
            .updated_mcp_tool_output = Some(output);
        self
    }

    /// Set a permission decision for permission-request hooks.
    #[must_use]
    pub fn with_permission(
        mut self,
        decision: PermissionDecision,
        reason: Option<impl Into<String>>,
    ) -> Self {
        let specific = self
            .output
            .hook_specific_output
            .get_or_insert_with(HookSpecificOutput::default);
        specific.permission_decision = Some(decision);
        specific.permission_decision_reason = reason.map(Into::into);
        self
    }

    /// Suppress user-visible tool output when supported by the caller.
    #[must_use]
    pub fn with_suppress_output(mut self, suppress_output: bool) -> Self {
        self.output.suppress_output = Some(suppress_output);
        self
    }

    /// Convert into the wire-compatible hook output.
    #[must_use]
    pub fn into_output(self) -> HookOutput {
        self.output
    }
}

impl From<HookOutput> for HookResponse {
    fn from(output: HookOutput) -> Self {
        Self { output }
    }
}

/// Accepted return types for SDK hook callbacks.
pub trait IntoHookResponse {
    /// Convert a callback result into a [`HookResponse`].
    fn into_hook_response(self) -> anyhow::Result<HookResponse>;
}

impl IntoHookResponse for HookResponse {
    fn into_hook_response(self) -> anyhow::Result<HookResponse> {
        Ok(self)
    }
}

impl IntoHookResponse for HookOutput {
    fn into_hook_response(self) -> anyhow::Result<HookResponse> {
        Ok(HookResponse::from(self))
    }
}

impl IntoHookResponse for anyhow::Result<HookResponse> {
    fn into_hook_response(self) -> anyhow::Result<HookResponse> {
        self
    }
}

#[derive(Clone)]
/// Executable SDK hook backend.
pub enum HookKind {
    /// Reuse the same callback for every matching hook dispatch.
    Callback(HookCallback),
    /// Build a callback per dispatch.
    Function(HookFunctionFactory),
}

impl fmt::Debug for HookKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Callback(_) => f.write_str("Callback(..)"),
            Self::Function(_) => f.write_str("Function(..)"),
        }
    }
}

impl HookKind {
    /// Hook handler type reported in run summaries.
    #[must_use]
    pub fn handler_type(&self) -> HookHandlerType {
        match self {
            Self::Callback(_) => HookHandlerType::Callback,
            Self::Function(_) => HookHandlerType::Function,
        }
    }
}

#[derive(Debug, Clone)]
/// SDK hook definition registered into a builder.
pub struct Hook {
    pub event: HookEventName,
    pub matcher: Option<String>,
    pub timeout: Duration,
    pub status_message: Option<String>,
    pub if_condition: Option<String>,
    pub once: bool,
    pub kind: HookKind,
}

impl Hook {
    /// Create a callback hook for an event.
    #[must_use]
    pub fn callback<F, Fut, R>(event: HookEventName, callback: F) -> Self
    where
        F: Fn(HookInput) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = R> + Send + 'static,
        R: IntoHookResponse + 'static,
    {
        Self {
            event,
            matcher: None,
            timeout: Duration::from_secs(30),
            status_message: None,
            if_condition: None,
            once: false,
            kind: HookKind::Callback(Arc::new(move |input| {
                let fut = callback(input);
                Box::pin(async move { fut.await.into_hook_response() })
            })),
        }
    }

    /// Create a hook that builds its callback per dispatch.
    #[must_use]
    pub fn function<Factory, F, Fut, R>(event: HookEventName, factory: Factory) -> Self
    where
        Factory: Fn() -> F + Send + Sync + 'static,
        F: Fn(HookInput) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = R> + Send + 'static,
        R: IntoHookResponse + 'static,
    {
        Self {
            event,
            matcher: None,
            timeout: Duration::from_secs(30),
            status_message: None,
            if_condition: None,
            once: false,
            kind: HookKind::Function(Arc::new(move || {
                let callback = factory();
                Arc::new(move |input| {
                    let fut = callback(input);
                    Box::pin(async move { fut.await.into_hook_response() })
                })
            })),
        }
    }

    /// Restrict the hook to matching event values.
    #[must_use]
    pub fn with_matcher(mut self, matcher: impl Into<String>) -> Self {
        self.matcher = Some(matcher.into());
        self
    }

    /// Override the hook timeout.
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// Set the status message shown while the hook runs.
    #[must_use]
    pub fn with_status_message(mut self, status_message: impl Into<String>) -> Self {
        self.status_message = Some(status_message.into());
        self
    }

    /// Set a simple hook condition expression.
    #[must_use]
    pub fn with_if_condition(mut self, if_condition: impl Into<String>) -> Self {
        self.if_condition = Some(if_condition.into());
        self
    }

    /// Run this hook only once per session when true.
    #[must_use]
    pub fn with_once(mut self, once: bool) -> Self {
        self.once = once;
        self
    }
}

#[derive(Debug, Clone)]
/// SDK hook with plugin identity and priority metadata.
pub struct RegisteredHook {
    pub plugin_id: PluginId,
    pub plugin_root: PathBuf,
    pub priority: RegisteredHookPriority,
    pub hook: Hook,
}

#[derive(Debug, Clone, Default)]
/// Collection of SDK hooks registered before runtime construction.
pub struct RegisteredHooks {
    hooks: Vec<RegisteredHook>,
}

impl RegisteredHooks {
    /// Whether no SDK hooks are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty()
    }

    /// Register a hook for one plugin id.
    pub fn register(&mut self, plugin_id: PluginId, priority: RegisteredHookPriority, hook: Hook) {
        self.hooks.push(RegisteredHook {
            plugin_id,
            plugin_root: PathBuf::new(),
            priority,
            hook,
        });
    }

    /// Validate SDK hook matchers before runtime construction.
    pub fn validate(&self) -> anyhow::Result<()> {
        for hook in &self.hooks {
            if let Some(matcher) = hook
                .hook
                .matcher
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                crate::matcher::CompiledMatcher::compile_regex(matcher).with_context(|| {
                    format!(
                        "failed to compile sdk hook matcher for plugin '{}' event '{}'",
                        hook.plugin_id,
                        hook.hook.event.canonical_name()
                    )
                })?;
            }
        }
        Ok(())
    }

    /// Convert registered SDK hooks into a runtime hook registry.
    pub fn instantiate(&self) -> anyhow::Result<crate::Hooks> {
        self.validate()?;
        crate::Hooks::from_registered(self.hooks.clone())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use serde_json::json;

    use super::*;
    use crate::{ConfiguredHandlerConfig, HookDispatchRequest, Hooks};

    #[test]
    fn registered_hooks_validate_rejects_invalid_matcher() {
        let mut hooks = RegisteredHooks::default();
        hooks.register(
            PluginId::from("plugin"),
            RegisteredHookPriority::AfterPlugins,
            Hook::callback(HookEventName::Stop, |_input| async {
                HookResponse::passthrough()
            })
            .with_matcher("["),
        );

        let error = hooks.validate().expect_err("invalid matcher should fail");
        assert!(
            error
                .to_string()
                .contains("failed to compile sdk hook matcher")
        );
    }

    #[test]
    fn hook_response_builders_populate_output() {
        let output = HookResponse::block("blocked")
            .with_system_message("system")
            .with_additional_context("context")
            .with_updated_input(json!({"command": "echo hi"}))
            .with_updated_output(json!({"ok": true}))
            .with_permission(PermissionDecision::Deny, Some("nope"))
            .with_suppress_output(true)
            .into_output();

        assert_eq!(output.decision, Some(HookDecision::Block));
        assert_eq!(output.reason.as_deref(), Some("blocked"));
        assert_eq!(output.system_message.as_deref(), Some("system"));
        assert_eq!(output.suppress_output, Some(true));

        let specific = output.hook_specific_output.expect("hook specific output");
        assert_eq!(specific.additional_context.as_deref(), Some("context"));
        assert_eq!(specific.updated_input, Some(json!({"command": "echo hi"})));
        assert_eq!(specific.updated_mcp_tool_output, Some(json!({"ok": true})));
        assert_eq!(specific.permission_decision, Some(PermissionDecision::Deny));
        assert_eq!(specific.permission_decision_reason.as_deref(), Some("nope"));
    }

    #[tokio::test]
    async fn hook_function_factory_creates_fresh_callback_per_instantiate() {
        let factory_calls = Arc::new(AtomicUsize::new(0));
        let counter = factory_calls.clone();
        let hook = Hook::function(HookEventName::Stop, move || {
            let instance = counter.fetch_add(1, Ordering::SeqCst) + 1;
            move |_input| async move {
                Ok(HookResponse::passthrough()
                    .with_system_message(format!("factory-instance-{instance}")))
            }
        });

        let mut registered = RegisteredHooks::default();
        registered.register(
            PluginId::from("plugin"),
            RegisteredHookPriority::AfterPlugins,
            hook,
        );

        let first_output =
            invoke_function_handler(&registered.instantiate().expect("instantiate")).await;
        let second_output =
            invoke_function_handler(&registered.instantiate().expect("instantiate")).await;

        assert_eq!(factory_calls.load(Ordering::SeqCst), 2);
        assert_eq!(first_output.as_deref(), Some("factory-instance-1"));
        assert_eq!(second_output.as_deref(), Some("factory-instance-2"));
    }

    async fn invoke_function_handler(hooks: &Hooks) -> Option<String> {
        let prepared = hooks.prepare(HookDispatchRequest {
            event_name: HookEventName::Stop,
            matcher_value: None,
            payload: json!({}),
            fired_hook_ids: BTreeSet::new(),
        });
        let handler = prepared
            .matched_handlers()
            .first()
            .cloned()
            .expect("function handler");

        let ConfiguredHandlerConfig::Function(callback) = handler.config else {
            panic!("expected function handler");
        };
        let response = callback(HookInput {
            event_name: HookEventName::Stop,
            matcher_value: None,
            payload: json!({}),
        })
        .await
        .expect("callback response");

        response.into_output().system_message
    }
}
