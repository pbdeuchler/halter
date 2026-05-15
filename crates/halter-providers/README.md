# halter-providers

`halter-providers` adapts model providers to the halter runtime.

It is the transport and normalization layer between provider-native APIs and halter's canonical protocol.

If the runtime asks for "stream a turn using this model role", the provider crate is what turns that request into OpenAI, Anthropic, OpenRouter, or fake/test traffic.

---

## Who this crate is for

### Primary: programmers integrating or extending model backends

Use this crate when you need to:

- instantiate providers from config
- register providers in a model registry
- understand capability differences between providers
- add a new provider implementation
- test runtime behavior without a real remote model

### Secondary: advanced CLI and operations users

If you only run the CLI, you still care about this crate because provider choice affects:

- which API key you need
- whether streaming is supported
- whether compaction is supported
- tool-call replay semantics
- model routing for default vs subagent workloads

---

## Public API at a glance

Core exports include:

- `Provider` trait
- `ProviderCapabilities`
- `ProviderRequest`
- `ProviderChunk`
- `ProviderError`
- `ModelRegistry`
- `ModelDefinition`
- `ModelReference`
- `ConfiguredProvider`
- `ProviderKind`
- `ApiKind`
- provider implementations:
  - `OpenAiProvider`
  - `AnthropicProvider`
  - `OpenRouterProvider`
  - `FakeProvider`
  - `UnsupportedProvider`

There is also a `responses_provider` adapter layer used by the OpenAI/OpenRouter side of the implementation.

---

## Mental model

A provider implementation is responsible for four things:

1. advertise capabilities
2. accept a normalized halter request
3. stream normalized halter chunks back
4. optionally support session compaction

The runtime should not need to know whether the backend is OpenAI, Anthropic, or a fake test provider.

That is the core abstraction boundary.

---

## The `Provider` trait

At the center of the crate is the `Provider` trait.

Important methods:

- `capabilities()`
- `stream(request, cancel)`
- `compact(request, cancel)` — optional, with a default error implementation

### `capabilities()`

Returns a `ProviderCapabilities` descriptor used by the rest of the system.

This tells the runtime things like:

- whether streaming is supported
- whether compaction is supported
- whether assistant content must be non-empty
- how tool-call IDs should be handled during replay

### `stream(...)`

Streams a model turn as canonical halter chunks.

### `compact(...)`

If supported, compacts session history for context management.

The default behavior is an error:

> `failed to compact session: provider does not support compaction`

This means compaction support is explicit, not assumed.

---

## Model registry

`ModelRegistry` is the runtime-facing index of:

- model role definitions
- model aliases/handles
- provider instances

Important methods include:

- `new`
- `set_default_model`
- `default_model`
- `set_small_model`
- `small_model`
- `set_subagent_model`
- `subagent_model`
- `model`
- `register_provider`
- `provider`

### Typical usage

A runtime builder will:

1. create a `ModelRegistry`
2. register provider instances
3. register model role definitions from config
4. ask for default or subagent models later during session execution

Example sketch:

```rust
use halter_providers::{ModelRegistry, OpenAiProvider};

let mut registry = ModelRegistry::new();
registry.register_provider(
    "openai".into(),
    std::sync::Arc::new(OpenAiProvider::new("sk-...", None)?),
)?;

// then register model definitions and set logical roles
```

The higher-level `halter` crate does this assembly for you.

---

## Built-in providers

## OpenAI

`OpenAiProvider::new(api_key, base_url)`

Characteristics:

- uses the Responses API pathing/semantics
- supports compaction
- suitable as a default provider for both primary and subagent models

Typical usage:

```rust
use halter_providers::OpenAiProvider;

let provider = OpenAiProvider::new(std::env::var("OPENAI_API_KEY")?, None)?;
```

If you need a custom endpoint:

```rust
let provider = OpenAiProvider::new(
    std::env::var("OPENAI_API_KEY")?,
    Some("https://api.openai.com".into()),
)?;
```

---

## OpenRouter

`OpenRouterProvider::new(api_key, base_url)`

Characteristics:

- also uses the Responses-style adapter path
- supports streaming
- supports inline compaction through the regular Responses endpoint

OpenRouter does not expose OpenAI's dedicated `/v1/responses/compact` endpoint,
so the runtime treats OpenRouter compaction as inline/lossy and narrows the
eligible window before summarizing.

Typical use:

```rust
use halter_providers::OpenRouterProvider;

let provider = OpenRouterProvider::new(
    std::env::var("OPENROUTER_API_KEY")?,
    None,
)?;
```

---

## Anthropic

`AnthropicProvider::new(api_key, base_url)`

Notable reported capabilities:

- `supports_streaming: true`
- `supports_prompt_cache: true`
- `supports_reasoning: true`
- `supports_interleaved_reasoning: true`
- `supports_compaction: true`
- `compaction_strategy: Inline`
- `requires_non_empty_assistant_content: true`
- `tool_call_id_policy: StableReplayNormalized`

Anthropic uses the Messages API rather than the Responses-style adapter path.
It has feature parity at the runtime capability level, but its replay and
compaction wire shapes remain Anthropic-native.

Typical use:

```rust
use halter_providers::AnthropicProvider;

let provider = AnthropicProvider::new(
    std::env::var("ANTHROPIC_API_KEY")?,
    None,
)?;
```

### Practical implication

If you switch a session to Anthropic, parts of your assumptions may need to change:

- do not assume streaming output behavior matches OpenAI
- do not assume compaction is available
- be aware of assistant-content normalization rules

---

## HTTP header overrides

All three built-in providers expose a `new_with_headers(api_key, base_url, &[(name, value)])`
constructor. The supplied overrides are applied per request with insert
semantics: entries replace any default or hardcoded header (`Authorization`,
`x-api-key`, `anthropic-version`, `Content-Type`) case-insensitively, and any
unrelated header names are forwarded as-is.

```rust
use halter_providers::OpenAiProvider;

let provider = OpenAiProvider::new_with_headers(
    std::env::var("OPENAI_API_KEY")?,
    "https://api.openai.com",
    &[
        ("Authorization".into(), "Bearer org-specific-token".into()),
        ("X-Trace-Id".into(), "halter-dev".into()),
    ],
)?;
```

Config-driven use goes through `[providers.<name>.headers]` in `halter-config`.

---

## Fake provider

`FakeProvider` is for tests and local verification.

Characteristics:

- deterministic/test-oriented behavior
- supports compaction
- removes dependency on external API credentials or network availability

This is useful for:

- runtime integration tests
- hook tests
- transcript/persistence tests
- CI environments where you want full control

---

## Unsupported provider

`UnsupportedProvider::new(kind)` is a deliberate stub used when a provider kind is recognized logically, but the transport is not wired into the current build.

It returns a provider error stating the transport is not wired in this build.

This is better than silently pretending support exists.

---

## Capability differences matter

A technically correct provider integration requires capability-aware code.

### Streaming

OpenAI, OpenRouter, Anthropic, and Fake stream canonical halter chunks.

### Compaction

Do not assume every provider supports compaction.

- OpenAI: yes
- OpenRouter: yes, inline
- Anthropic: yes, inline
- Fake: yes

### Tool-call replay rules

Providers may have different tool-call ID expectations. The `tool_call_id_policy` capability exists for a reason.

### Assistant content constraints

Some providers require non-empty assistant content. The runtime and protocol layers normalize accordingly.

---

## Realistic assembly example

If you were assembling a registry yourself:

```rust
use std::sync::Arc;
use halter_providers::{
    ApiKind, ConfiguredProvider, ModelDefinition, ModelRegistry, OpenAiProvider, ProviderKind,
};

fn build_registry() -> anyhow::Result<ModelRegistry> {
    let mut registry = ModelRegistry::new();

    registry.register_provider(
        "openai".into(),
        Arc::new(OpenAiProvider::new(std::env::var("OPENAI_API_KEY")?, None)?),
    )?;

    registry.set_default_model(ModelDefinition {
        name: "default".into(),
        provider: ConfiguredProvider::OpenAi,
        provider_name: "openai".into(),
        provider_kind: ProviderKind::OpenAi,
        api_kind: ApiKind::Responses,
        model: "gpt-5".into(),
        reasoning: Some("high".into()),
        max_input_tokens: None,
        max_output_tokens: None,
        tokens_per_minute: Some(500_000),
    });

    Ok(registry)
}
```

In practice, the `halter` crate builds this from `HarnessConfig`, but this shows the conceptual layering.

---

## Implementing a new provider

If you want to add a provider backend, the rough checklist is:

1. implement `Provider`
2. define correct `ProviderCapabilities`
3. translate halter requests to your provider's API
4. translate provider responses/chunks back into canonical protocol chunks
5. handle cancellation correctly
6. decide whether compaction is supported
7. add tests for streaming, tool calls, failures, and replay behavior

### Design advice

- Normalize provider-native weirdness inside the provider crate, not in the runtime.
- Be conservative and accurate with capabilities.
- If a feature is not implemented, advertise that honestly rather than approximating.
- Add a fake or fixture-driven test path before hitting the real API repeatedly.

---

## Error handling

Provider code is where many time-sensitive or network-sensitive failures surface.

Typical classes of failure:

- authentication errors
- invalid base URLs
- rate limiting
- request timeouts
- unsupported provider features
- schema mismatches during chunk translation
- replay normalization issues

If a provider doesn't support compaction and the runtime asks anyway, you will see an error like:

> `failed to compact session: provider does not support compaction`

That is expected, not necessarily a bug.

---

## Operational guidance for CLI users

Even if you never import this crate directly, you should choose providers intentionally.

### Choose OpenAI when you want

- straightforward default behavior
- streaming-friendly usage
- provider-backed compaction support

### Choose OpenRouter when you want

- alternative model routing
- Responses-style integration
- broad upstream model access

But remember: compaction is inline/lossy rather than dedicated endpoint backed.

### Choose Anthropic when you want

- Anthropic-native backend behavior
- Messages API streaming
- prompt caching and interleaved thinking

But remember:

- compaction is inline/lossy rather than dedicated endpoint backed
- some additional replay and content-normalization constraints

---

## Testing recommendations

When building on this crate:

- use `FakeProvider` for runtime contract tests
- run one small set of smoke tests against real providers
- verify capability-sensitive branches explicitly
- test replay and tool-call stability if your workload uses tools heavily

A common mistake is to test only "assistant text comes back" and ignore:

- chunk ordering
- tool call IDs
- cancellation
- compaction support
- provider-specific content constraints

---

## Related docs

- `../halter-config/README.md` — provider selection and credential resolution
- `../halter-runtime/README.md` — where providers are used in sessions
- `../halter-protocol/README.md` — canonical request/chunk/event types
- `../halter/README.md` — top-level harness assembly from config
