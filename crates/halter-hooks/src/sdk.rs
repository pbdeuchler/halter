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

pub type HookCallbackFuture = BoxFuture<'static, anyhow::Result<HookResponse>>;
pub type HookCallback = Arc<dyn Fn(HookInput) -> HookCallbackFuture + Send + Sync>;
pub type HookFunctionFactory = Arc<dyn Fn() -> HookCallback + Send + Sync>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RegisteredHookPriority {
    BeforePlugins,
    #[default]
    AfterPlugins,
}

#[derive(Debug, Clone)]
pub struct HookInput {
    pub event_name: HookEventName,
    pub matcher_value: Option<String>,
    pub payload: Value,
}

impl HookInput {
    #[must_use]
    pub fn field(&self, key: &str) -> Option<&Value> {
        self.payload.get(key)
    }

    #[must_use]
    pub fn string_field(&self, key: &str) -> Option<&str> {
        self.field(key).and_then(Value::as_str)
    }

    #[must_use]
    pub fn tool_name(&self) -> Option<&str> {
        self.string_field("tool_name")
    }

    #[must_use]
    pub fn tool_use_id(&self) -> Option<&str> {
        self.string_field("tool_use_id")
    }

    pub fn decode<T: DeserializeOwned>(&self) -> anyhow::Result<T> {
        serde_json::from_value(self.payload.clone()).context("failed to decode hook input")
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct HookResponse {
    output: HookOutput,
}

impl HookResponse {
    #[must_use]
    pub fn passthrough() -> Self {
        Self::default()
    }

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

    #[must_use]
    pub fn with_system_message(mut self, message: impl Into<String>) -> Self {
        self.output.system_message = Some(message.into());
        self
    }

    #[must_use]
    pub fn with_additional_context(mut self, context: impl Into<String>) -> Self {
        self.output
            .hook_specific_output
            .get_or_insert_with(HookSpecificOutput::default)
            .additional_context = Some(context.into());
        self
    }

    #[must_use]
    pub fn with_updated_input(mut self, input: Value) -> Self {
        self.output
            .hook_specific_output
            .get_or_insert_with(HookSpecificOutput::default)
            .updated_input = Some(input);
        self
    }

    #[must_use]
    pub fn with_updated_output(mut self, output: Value) -> Self {
        self.output
            .hook_specific_output
            .get_or_insert_with(HookSpecificOutput::default)
            .updated_mcp_tool_output = Some(output);
        self
    }

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

    #[must_use]
    pub fn with_suppress_output(mut self, suppress_output: bool) -> Self {
        self.output.suppress_output = Some(suppress_output);
        self
    }

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

pub trait IntoHookResponse {
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
pub enum HookKind {
    Callback(HookCallback),
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
    #[must_use]
    pub fn handler_type(&self) -> HookHandlerType {
        match self {
            Self::Callback(_) => HookHandlerType::Callback,
            Self::Function(_) => HookHandlerType::Function,
        }
    }
}

#[derive(Debug, Clone)]
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

    #[must_use]
    pub fn with_matcher(mut self, matcher: impl Into<String>) -> Self {
        self.matcher = Some(matcher.into());
        self
    }

    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    #[must_use]
    pub fn with_status_message(mut self, status_message: impl Into<String>) -> Self {
        self.status_message = Some(status_message.into());
        self
    }

    #[must_use]
    pub fn with_if_condition(mut self, if_condition: impl Into<String>) -> Self {
        self.if_condition = Some(if_condition.into());
        self
    }

    #[must_use]
    pub fn with_once(mut self, once: bool) -> Self {
        self.once = once;
        self
    }
}

#[derive(Debug, Clone)]
pub struct RegisteredHook {
    pub plugin_id: PluginId,
    pub plugin_root: PathBuf,
    pub priority: RegisteredHookPriority,
    pub hook: Hook,
}

#[derive(Debug, Clone, Default)]
pub struct RegisteredHooks {
    hooks: Vec<RegisteredHook>,
}

impl RegisteredHooks {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.hooks.is_empty()
    }

    pub fn register(&mut self, plugin_id: PluginId, priority: RegisteredHookPriority, hook: Hook) {
        self.hooks.push(RegisteredHook {
            plugin_id,
            plugin_root: PathBuf::new(),
            priority,
            hook,
        });
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        for hook in &self.hooks {
            if let Some(matcher) = hook
                .hook
                .matcher
                .as_deref()
                .map(str::trim)
                .filter(|value| !value.is_empty())
            {
                regex::Regex::new(matcher).with_context(|| {
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

    pub fn instantiate(&self) -> anyhow::Result<crate::Hooks> {
        self.validate()?;
        crate::Hooks::from_registered(self.hooks.clone())
    }
}
