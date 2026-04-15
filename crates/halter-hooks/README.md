# halter-hooks

`halter-hooks` is halter's policy and extension surface for intercepting runtime events, modifying inputs and outputs, attaching extra context, enforcing approvals, and integrating external hook executables or in-process hook implementations.

If you need to shape runtime behavior without forking the runtime, this is the crate you reach for.

---

## Who this crate is for

### Primary: programmers extending the runtime

Use `halter-hooks` when you want to:

- observe important runtime lifecycle events
- block unsafe or undesired actions
- require approvals or permission decisions
- inject system messages or contextual guidance
- rewrite tool inputs or outputs
- integrate script-based hooks discovered from resources
- register SDK hooks directly in Rust

### Secondary: CLI users and platform operators

If you run the CLI, hooks affect you even if you never import this crate directly. They can:

- block tool calls
- request permission
- add extra system guidance
- emit notifications
- stop sessions on policy violations

For user-facing command behavior, also read `../halter-cli/README.md`.

---

## Mental model

A hook in halter is an event-driven policy function.

At runtime, the system:

1. emits a typed hook event
2. constructs a dispatch request with payload and context
3. runs registered hooks and/or external hook executables
4. merges their outputs deterministically
5. feeds the merged result back into the runtime

Hooks can do more than just observe. They can actively influence execution.

They can:

- approve or block
- stop execution
- attach system messages
- add additional structured context
- modify inbound or outbound payloads
- make permission decisions
- suppress output visibility

---

## Public API at a glance

Core types exported by the crate include:

- `HookEventName`
- `HookDecision`
- `PermissionDecision`
- `HookResponse`
- `HookOutput`
- `HookDispatchRequest`
- `PreparedHookDispatch`
- `HookDispatchOutcome`
- `Hooks`
- `HooksEngine`
- `MergeError`
- `HookConfig`
- `HooksConfig`
- `ConfiguredHook`
- `register_runtime_hook`
- `clear_runtime_hook_registry`

The SDK layer in `halter` uses these to install plugin and in-process hooks.

---

## Event model

## `HookEventName`

The runtime exposes a broad event surface. Important events include:

- `SessionStart`
- `SessionEnd`
- `UserPromptSubmit`
- `PreToolUse`
- `PostToolUse`
- `PostToolUseFailure`
- `Notification`
- `Stop`
- `SubagentStart`
- `SubagentStop`
- `PreCompact`
- `PostCompact`
- `PermissionRequest`
- `PermissionDenied`
- `Elicitation`
- `ElicitationResult`
- `WorktreeCreate`
- `WorktreeRemove`
- `FileChanged`
- `CwdChanged`
- `InstructionsLoaded`
- `ConfigChange`
- `Setup`
- `TeammateIdle`
- `TaskCreated`
- `TaskCompleted`
- `StopFailure`
- `PostSampling`

In practice, the most operationally important events are usually:

- `UserPromptSubmit`
- `PreToolUse`
- `PostToolUse`
- `PostToolUseFailure`
- `PermissionRequest`
- `SubagentStart`
- `SubagentStop`
- `PreCompact`
- `PostCompact`

---

## Hook responses

## `HookResponse`

A hook produces a response, which can then be converted into a `HookOutput`.

Convenience constructors/helpers include:

- `HookResponse::passthrough()`
- `HookResponse::block(...)`
- `HookResponse::stop(...)`
- `.with_system_message(...)`
- `.with_additional_context(...)`
- `.with_updated_input(...)`
- `.with_updated_output(...)`
- `.with_permission(...)`
- `.with_suppress_output(...)`
- `.into_output(...)`

### Core capabilities

A response can express:

- a decision (`Approve` or `Block`)
- an optional permission decision (`Allow`, `Ask`, `Deny`, `Passthrough`)
- additional system messages
- appended additional context
- transformed input
- transformed output
- whether tool/output display should be suppressed
- a stop condition

---

## Merge semantics

Multiple hooks may respond to the same event.

The crate merges them using explicit rules, rather than "last one wins" guessing.

Key types:

- `HookDecision::{Approve, Block}`
- `PermissionDecision::{Deny, Ask, Allow, Passthrough}`
- `merge_outputs(...)`

### Practical consequences

- any blocking hook matters
- permission decisions are merged deliberately
- contextual additions may accumulate
- input/output mutations must merge coherently or fail

This matters when you combine:

- repo-local script hooks
- installed plugin hooks
- in-process SDK hooks

---

## Creating hooks in Rust

The exact hook registration surface is intentionally simple from the consumer side: register a runtime hook, then let the runtime dispatch it.

A typical policy hook looks like this conceptually:

```rust
use halter_hooks::{HookEventName, HookResponse};

fn deny_rm_rf(event: HookEventName, payload: serde_json::Value) -> HookResponse {
    if event == HookEventName::PreToolUse {
        let tool_name = payload.get("tool_name").and_then(|v| v.as_str()).unwrap_or("");
        let input = payload.get("input").cloned().unwrap_or_default();

        if tool_name == "shell" && input.to_string().contains("rm -rf /") {
            return HookResponse::block("dangerous shell command blocked by policy");
        }
    }

    HookResponse::passthrough()
}
```

Then register it with the runtime-facing registry using the crate's registration helpers.

### When to use SDK hooks vs external hooks

Use SDK hooks when:

- you want typed Rust integration
- you need internal state or shared process access
- you package policy with an application embedding halter

Use external hooks when:

- you want repo-local policy scripts
- you want non-Rust implementations
- you need easy operator customization without recompiling

---

## Loading hooks from resources

`Hooks::from_sources(...)` builds a hook set from discovered hook definitions.

This is how repo/resource-level hooks become active.

Typical source categories include:

- hooks declared by resources/plugins
- hooks loaded from configuration
- hooks resolved from runtime registries

`Hooks::from_registered(...)` builds from registered in-process hooks.

The `halter` crate bridges plugin resource loading and these dispatch structures.

---

## Dispatch pipeline

Important runtime-facing types:

- `HookDispatchRequest`
- `PreparedHookDispatch`
- `HookDispatchOutcome`

### Conceptual flow

1. build a `HookDispatchRequest`
2. normalize/prepare it into `PreparedHookDispatch`
3. execute matching hooks
4. collect responses
5. merge them into final `HookDispatchOutcome`

The runtime then interprets the outcome.

That may mean:

- continue as normal
- block an operation
- ask the user/operator for permission
- inject additional instructions before the next model call
- stop the session entirely

---

## Practical patterns

## Pattern: add safety guidance before tool use

A `PreToolUse` hook can attach a system message reminding the model about local policy.

```rust
use halter_hooks::HookResponse;

let response = HookResponse::passthrough()
    .with_system_message("Only modify files under the active repository root.")
    .with_additional_context(serde_json::json!({
        "policy": { "write_scope": "repo-root-only" }
    }));
```

This is useful when you want soft guidance rather than hard blocking.

---

## Pattern: hard-block dangerous file writes

Example policy idea:

- on `PreToolUse`
- inspect `tool_name == "write"` or `tool_name == "edit"`
- reject paths outside approved roots

In practice, the tool policy layer already enforces filesystem scope. Hooks are best for:

- additional business rules
- approval workflows
- annotation and audit context

---

## Pattern: permission mediation

A hook can say "this should be asked" rather than directly allowed or denied.

```rust
use halter_hooks::{HookResponse, PermissionDecision};

let response = HookResponse::passthrough()
    .with_permission(PermissionDecision::Ask)
    .with_system_message("Escalated operation requires explicit approval.");
```

This is appropriate for:

- privileged shell commands
- high-risk network operations
- writes outside the main workspace
- long-running or destructive subprocesses

---

## Pattern: redact or suppress noisy output

For verbose tools, a post-tool hook can request output suppression.

```rust
use halter_hooks::HookResponse;

let response = HookResponse::passthrough()
    .with_suppress_output(true);
```

That is useful when:

- output contains secrets
- output is huge and not valuable to the transcript
- you want to preserve audit metadata without rendering raw payloads

---

## Pattern: transform input/output

Hooks can also rewrite data.

Examples:

- normalize shell commands before execution
- sanitize arguments
- rewrite generated paths
- wrap command output with metadata
- redact tokens and keys before transcript inclusion

Use this carefully. Transformative hooks can be powerful but hard to debug.

---

## Example: repo-local policy stack

A realistic policy arrangement for an enterprise workspace might include:

1. a repo-local pre-tool hook to detect destructive commands
2. a post-tool hook to annotate outputs with ticket or policy IDs
3. a session-start hook to inject organization-wide operating guidelines
4. a permission-request hook to auto-deny network access outside an allowlist
5. a subagent-start hook to clamp allowed task classes for delegated work

This lets you adapt halter to local governance without changing core runtime logic.

---

## User-facing implications in the CLI

Even if you're only using `halter run` or `halter chat`, hooks can materially change behavior.

You may see:

- commands blocked that would otherwise be allowed by tool policy
- injected guidance that changes model behavior
- permission prompts or denials
- notifications emitted at significant lifecycle moments
- different outputs because a hook rewrote or suppressed them

This is expected. Hooks are part of the runtime contract.

---

## Ordering and precedence

The crate is designed to support ordered hook execution and merging.

In practical terms, when building systems on top of this crate:

- keep high-priority hard-safety hooks early and simple
- keep advisory/enrichment hooks separate from deny hooks
- avoid having multiple hooks compete to rewrite the same field
- document your hook stack clearly, especially if both scripts and SDK hooks are active

---

## Error handling

Important failure classes include:

- malformed hook payloads
- merge conflicts between incompatible outputs
- unavailable external hook commands
- serialization/deserialization errors
- unexpected runtime hook panics or adapter failures

If a hook architecture is mission-critical, test it like code, not like config.

Recommended practices:

- keep hook outputs deterministic
- prefer explicit block/allow decisions over ambiguous transformations
- log enough metadata to understand why a hook fired
- avoid hidden side effects in hooks

---

## Testing strategy

If you embed halter and rely on hooks, test at three layers.

### Unit tests

Test your hook logic in isolation.

### Merge tests

If multiple hooks apply to the same event, test merged outcomes explicitly.

### Integration tests

Drive the runtime through actual events and assert observed behavior:

- tool call blocked
- permission requested
- context injected
- output suppressed
- session stopped

The `fake` provider from `halter-providers` is useful for fast runtime tests.

---

## When not to use hooks

Hooks are the wrong tool when you simply need:

- a new model provider → use `halter-providers`
- a new tool → use `halter-tools`
- durable session persistence → use `halter-session`
- custom resource loading → use the `halter` resource/compiler layer

Use hooks when the problem is policy, interception, annotation, or workflow control.

---

## Recommended design guidelines

- Keep hooks narrow and event-specific.
- Prefer explicit hard blocks for truly unsafe operations.
- Prefer system messages and additional context for guidance.
- Use permission decisions for operations requiring human escalation.
- Treat output rewriting as a high-power feature and keep it well-tested.
- Avoid business logic that depends on undocumented payload shapes.

---

## Related docs

- `../halter/README.md` — builder APIs for installing plugin and SDK hooks
- `../halter-runtime/README.md` — where hook dispatch is invoked during execution
- `../halter-tools/README.md` — common policy targets for pre/post tool hooks
- `../halter-cli/README.md` — how hook effects surface to end users
