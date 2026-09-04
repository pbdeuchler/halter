# halter-config

`halter-config` defines halter's configuration schema and the utilities for loading, validating, merging, fingerprinting, and exporting it.

If `halter` is the assembly layer, `halter-config` is the contract that tells the rest of the workspace what to build.

---

## Who this crate is for

### Primary: programmers and operators configuring halter

Use this crate when you need to:

- define or inspect `HarnessConfig`
- load `halter.toml`
- merge layered configuration
- apply environment overrides
- validate provider credentials and runtime requirements
- generate starter configs or JSON Schema

### Secondary: CLI users

The `halter` CLI uses this crate internally for:

- `halter init`
- `halter validate`
- `halter config schema`
- `halter run` / `chat` / `resources` config loading

If you only want the command-line workflow, read `../halter-cli/README.md` too.

---

## Public API at a glance

This crate exports two major groups of APIs.

### Schema types

- `HarnessConfig`
- `ProvidersConfig`, `ProviderConfig`, `OpenAiOAuthConfig`, `ConfiguredProvider`
- `ResolvedProviderConfig`, `ResolvedProviderAuth`
- `ModelsConfig`, `ModelConfig`, `ModelSlot`, `ModelSlotRef`, `ModelJudgeConfig`
- `ResourcesConfig`, `SearchRoots`
- `PromptsConfig`
- `ContextConfig`
- `ToolsConfig`
- `PolicyConfig`, `ShellPolicyConfig`, `NetworkPolicyConfig`
- `SessionsConfig`, `SessionBackend`
- `RuntimeConfig`

### Loader / utility functions

- `load_path(...)`
- `load_layered(...)`
- `apply_env_overrides(...)`
- `export_json_schema()`
- `schema_as_json_value()`
- `generate_starter_config()`
- `config_fingerprint(...)`
- `expand_path(...)`
- `resolve_provider_runtime_config(...)`

---

## Minimal valid config

A `HarnessConfig` requires at least:

- `version = 1`
- `[models.default]`
- a compaction threshold source: `models.default.max_input_tokens` or
  `context.compaction_threshold`
- credentials for the selected provider, either in config or the environment

Minimal example:

```toml
version = 1

[models.default]
provider = "openai"
model = "gpt-5"
reasoning = "medium"
max_input_tokens = 200_000

[resources.skills]

roots = ["./.agent/skills"]

[resources.plugins]
roots = ["./.agent/plugins"]
```

With credentials via environment:

```bash
export OPENAI_API_KEY=...
```

Or inline in config:

```toml
[providers.openai]
api_key = "sk-..."
```

OpenAI also accepts OAuth credentials instead of an API key:

```toml
[providers.openai.oauth]
client_id = "..."
access_token = "..."
id_token = "..."
refresh_token = "..."
```

---

## Schema walkthrough

## `version`

Currently only this is accepted:

```toml
version = 1
```

Any other version is rejected.

---

## Providers

Supported configured providers are:

- `openai`
- `anthropic`
- `openrouter`

Schema shape:

```toml
[providers.openai]
base_url = "https://api.openai.com"
api_key = "sk-..."

# Instead of api_key, OpenAI can use OAuth credentials.
[providers.openai.oauth]
client_id = "..."
access_token = "..."
id_token = "..."
refresh_token = "..."

[providers.anthropic]
base_url = "https://api.anthropic.com"
api_key = "..."

[providers.openrouter]
base_url = "https://openrouter.ai/api"
api_key = "..."

# OpenRouter only: which upstream provider should serve the model. Sent as the
# `provider` object of every OpenRouter request body, compaction included.
[providers.openrouter.routing]
order = ["anthropic", "google-vertex"]
allow_fallbacks = false
```

`routing.order` must name at least one upstream provider and its slugs must be
non-empty and distinct; slugs are trimmed. `allow_fallbacks` alone is not a
preference, because `true` is already OpenRouter's default and `false` would
permit nothing. Omitting `allow_fallbacks` defers to OpenRouter's default,
which lets it route outside `order`. Configuring `routing` for any other
provider fails validation.

The rules themselves live on `halter_protocol::OpenRouterRouting::normalized`,
which this crate applies while resolving the config and `OpenRouterProvider`
applies to values supplied directly.

Each provider also accepts an optional `headers` sub-table. Entries override
the provider's default or hardcoded HTTP headers (`Authorization`, `x-api-key`,
`anthropic-version`, `Content-Type`) case-insensitively, and any unrelated
headers are appended verbatim:

```toml
[providers.openai.headers]
Authorization = "Bearer org-specific-token"
X-Trace-Id = "halter-dev"
```

Header names must be ASCII graphic characters with no colons; values must be
non-empty. Validation rejects malformed entries at load time.

Provider-level resilience overrides inherit from the global `[resilience]`
block and can override only the fields that need to differ:

```toml
[resilience.timeouts]
connect_secs = 10
request_secs = 60
stream_idle_secs = 60

[resilience.request_retry]
max_attempts = 5
deadline_secs = 60
base_backoff_ms = 500
max_backoff_secs = 30
jitter_pct = 25

[providers.openrouter.resilience.request_retry]
max_attempts = 3
```

For streaming calls, `request_secs` bounds request setup through response
headers and `stream_idle_secs` bounds the idle gap between stream items; it is
not a total stream lifetime cap. `max_attempts` includes the initial provider
request. Timeout and retry values must be positive, and `jitter_pct` must be in
`0..=100`.

### Resolution rules

`resolve_provider_runtime_config(...)` resolves provider settings this way:

1. `base_url`
   - use configured `base_url` if present and non-empty
   - otherwise fall back to the provider default
2. credentials
   - prefer configured `[providers.<name>].api_key`
   - for OpenAI only, configured `[providers.openai].oauth` is accepted instead of `api_key`
   - otherwise look up the provider-specific environment variable
   - fail if neither is available

OpenAI `api_key` and `oauth` are mutually exclusive. OAuth config must include
`client_id`, `access_token`, `id_token`, and `refresh_token`; the OpenAI
provider sends `access_token` as the bearer token. OAuth traffic for
`/v1/responses`, every path below that prefix such as
`/v1/responses/compact`, and `/chat/completions` rewrites to
`https://chatgpt.com/backend-api/codex/responses`, so `base_url` is ignored for
those paths. OAuth requests send the assembled system prompt as top-level
`instructions`, omit the developer/system input item, and set `store` to
`false`.

Environment variables used:

- OpenAI → `OPENAI_API_KEY`
- Anthropic → `ANTHROPIC_API_KEY`
- OpenRouter → `OPENROUTER_API_KEY`

### Runtime validation behavior

Only providers that are actually selected by the config are required at runtime.

That means if your default model uses OpenAI and your subagent model uses OpenRouter, both credentials must be resolvable.

---

## Models

`models` defines logical roles rather than every possible model in the world.

Supported roles:

- `default` — required
- `small` — optional
- `subagent` — optional

Example:

```toml
[models.default]
provider = "openai"
model = "gpt-5"
reasoning = "high"
max_input_tokens = 200000
max_output_tokens = 8192
tokens_per_minute = 500000

[models.small]
provider = "openai"
model = "gpt-5-mini"
reasoning = "medium"

[models.subagent]
provider = "openrouter"
model = "openai/gpt-5-mini"
reasoning = "medium"
```

### Model-judge slots

`models.default` and `models.subagent` accept either an inline model (above) or
the string `"model_judge"`, which resolves through a shared
`[models.model_judge]` block holding a `default` model, a `synthesis` model, and a
non-empty `panel`:

```toml
[models]
default = "model_judge"

[models.model_judge.default]
provider = "anthropic"
model = "claude-opus-4-8"

[models.model_judge.synthesis]
provider = "openai"
model = "gpt-5"

[[models.model_judge.panel]]
provider = "openai"
model = "gpt-5"

[[models.model_judge.panel]]
provider = "anthropic"
model = "claude-sonnet-4-6"
```

At runtime a model-judge slot multiplexes each call to the `panel`, asks the
`synthesis` model to stack-rank and synthesize their responses, and hands the
synthesis to the `default` model to produce the visible reply. See the
workspace README for the full flow and telemetry.

`models.subagent` also accepts `"auto_resolve"` when spawned subagents should use
the parent session's active model configuration instead of a separately
configured subagent model:

```toml
[models]
default = "model_judge"
subagent = "auto_resolve"
```

This is useful when the default slot is a full-turn model judge: panel members
spawn child sessions with their concrete panel model instead of recursively
re-entering the global `model_judge` slot.

### Notes

- `models.default` is required
- a `"model_judge"` slot requires a `[models.model_judge]` block with non-empty `panel`
- `"auto_resolve"` is valid only for `models.subagent`
- `models.small` is always a single inline model (it does not fan out); when
  unset it falls back to the default slot's representative leaf model
- `reasoning` is optional and accepts `none`, `minimal`, `low`, `medium`,
  `high`, `xhigh`, and `max`; provider/model support varies
- `tokens_per_minute` defaults to `500_000` and must be positive when set
- `max_input_tokens` / `max_output_tokens` must be positive when set
- the provider choice also determines the provider kind and API kind used elsewhere in the workspace

---

## Resource discovery

`resources` controls where halter searches for skills and plugins.

```toml
[resources.skills]
roots = ["./.agent/skills", "~/shared-skills"]

[resources.plugins]
roots = ["./.agent/plugins"]
```

These roots are consumed by `halter::ResourceCompiler`.

### Path expansion

`expand_path(...)` expands `~/...` using `$HOME`. Internally the `halter-config`
crate resolves `$HOME` through an injectable environment lookup (the same `_with`
convention used by env overrides), so the expansion rules are unit-testable;
the public `expand_path` wrapper reads the real process environment.

It does **not** perform general shell expansion.

---

## Prompts

Current prompt config surface:

```toml
[prompts]
preset = "general" # "general" (default) | "coding"
system_prompt = "optional override text"
append_system_prompt = """
## House rules
- Stable extra instructions appended after the resolved base prompt.
"""
```

`system_prompt` replaces the selected base prompt. `append_system_prompt` keeps the selected base prompt and adds static, prefix-cacheable System text after it.

---

## Context / compaction policy

`ContextConfig` controls when the runtime compacts session history and where
it caps it. Every threshold is optional.

```toml
[context]
compaction_threshold = 180000    # default: models.default.max_input_tokens - 20000
pre_compaction_target = 135000   # default: three quarters of compaction_threshold
max_tokens = 200000              # default: models.default.max_input_tokens
prune_signal_threshold = "normal"
```

Derivation, applied by `HarnessConfig::resolved_context` (`ContextConfig::resolve`):

- `compaction_threshold` derives from `models.default.max_input_tokens` minus
  `COMPACTION_HEADROOM_TOKENS` (20,000). If neither is set, loading the config
  and building the harness fail with
  `context.compaction_threshold is unset and models.default.max_input_tokens is unset`.
  There is no built-in model-to-window table and no silent fallback.
- `pre_compaction_target` derives as three quarters of the resolved threshold.
- `max_tokens` derives from `max_input_tokens`; without a window there is no cap.
  The runtime fails a turn that would exceed it, after compaction has had its
  chance, instead of sending the context to the provider.

Validation rules, checked on the resolved values (`HarnessConfig::validate`
runs them whenever the threshold is derivable):

- `compaction_threshold` must be greater than zero; a derived one needs
  `max_input_tokens` above 20,000
- `pre_compaction_target` must be less than `compaction_threshold`
- `max_tokens` must be at least `compaction_threshold`

A more aggressive setup:

```toml
[context]
compaction_threshold = 200000
pre_compaction_target = 150000
prune_signal_threshold = "low"
```

---

## Tool enablement

`ToolsConfig` is intentionally simple:

```toml
[tools]
enabled = ["read", "glob", "grep", "write", "edit", "shell", "process"]
```

Behavior:

- if `enabled` is empty, the runtime registers all built-in tools available in the build
- if `enabled` is non-empty, only listed tools are registered

Feature-gated tools must still be compiled in via Cargo features, even if named here.

For the full tool catalog, see `../halter-tools/README.md`.

---

## Policy

`PolicyConfig` controls file writes, reads, shell usage, network policy, and subagent limits.

Example:

```toml
[policy]
allowed_write_roots = ["./", "/tmp/halter"]
allowed_read_roots = []       # empty keeps the built-in read roots
max_read_bytes = 1048576
max_subagent_depth = 3
max_concurrent_subagents = 8
# sensitive_path_patterns = ["**/.ssh/**"]   # absent keeps the built-in globs

[policy.shell]
enabled = true
allow = ["git", "cargo", "rg", "ls", "find", "true", "cd"]
mode = "strict"
timeout_secs = 30

[policy.network]
enabled = false
allowed_hosts = []
```

Defaults:

- `allowed_write_roots = [".", "/tmp/halter"]`
- `allowed_read_roots = []` — runtime read roots then start from
  `[ ".", $TMPDIR | "/tmp" ]`. A non-empty list replaces those built-ins.
  Either way the runtime also reads under every configured
  `allowed_write_roots` entry
- `sensitive_path_patterns` unset — the runtime denies the built-in globs
  (`**/.ssh/**`, `**/.aws/**`, `**/.secrets`, `/etc/shadow`, `/etc/shadow.*`)
  to every file tool. An explicit list replaces them; `[]` disables the check
- `max_read_bytes = 1_048_576`
- `max_subagent_depth = 3`
- `max_concurrent_subagents = 8`
- shell enabled by default
- shell allowlist defaults to `git`, `cargo`, `rg`, `ls`, `find`, `true`, `cd`.
  `["*"]` allows every program; `[]` denies every external command
- `policy.shell.mode = "strict"` — also rejects function definitions and
  `eval`/`exec`/`source`/`.`. `"relaxed"` applies only the allowlist
- network disabled by default

Validation rules include:

- `max_read_bytes > 0`

---

## Sessions

Session persistence is configured here.

```toml
[sessions]
backend = "memory"
# sqlite_path = "/tmp/halter/sessions.db"
```

Possible backends:

- `memory`
- `sqlite` (only when the `sqlite` feature is compiled in)

Validation rules:

- `sqlite_path` must not be empty when set
- with `sqlite` support enabled, `sqlite_path` requires `backend = "sqlite"`
- without the Cargo `sqlite` feature, setting `sqlite_path` is rejected

---

## Runtime

Current runtime-specific config surface:

```toml
[runtime]
working_dir = "/path/to/project"
traces_dir = "/tmp/halter/traces"
subagent_event_forwarding = "off"
subagent_event_forwarding_cap = 100_000
```

`subagent_event_forwarding` defaults to `"off"`. Set it to `"all"` when the caller consuming a parent turn stream also needs raw committed events from subagents spawned under that parent. Forwarded events keep their source `session_id`; consumers must filter by session when attributing tool calls or messages.

`subagent_event_forwarding_cap` limits forwarded subagent events per parent turn. The default is `100_000`; `0` disables the cap.

---

## Loading APIs

## `load_path(path)`

Loads, parses, overrides, validates, and runtime-checks a single TOML file.

```rust
use halter_config::load_path;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = load_path("halter.toml").await?;
    println!("default model = {}", config.default_model()?.model);
    Ok(())
}
```

What it does:

1. read file text
2. parse TOML
3. apply supported environment overrides
4. deserialize into `HarnessConfig`
5. validate schema rules
6. validate runtime requirements such as provider credentials

---

## `load_layered(paths)`

Loads multiple config layers and merges them in order:

1. `user_config`
2. `project_config`
3. `explicit_config`

Later layers override earlier ones.

```rust
use std::path::PathBuf;
use halter_config::{LayeredConfigPaths, load_layered};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = load_layered(LayeredConfigPaths {
        user_config: Some(PathBuf::from("~/.config/halter/config.toml")),
        project_config: Some(PathBuf::from("./halter.toml")),
        explicit_config: Some(PathBuf::from("./ci/halter.toml")),
    }).await?;

    println!("fingerprint = {}", halter_config::config_fingerprint(&config));
    Ok(())
}
```

### Array merge semantics

Not all arrays merge the same way.

#### Append + dedupe

These are appended with duplicates removed:

- `resources.skills.roots`
- `resources.plugins.roots`

#### Replace

These are replaced by later layers:

- `policy.shell.allow`
- `policy.allowed_write_roots`
- `policy.network.allowed_hosts`
- `tools.enabled`
- everything else unless explicitly handled otherwise

This is important operationally: adding a later config layer with `tools.enabled = [...]` replaces the entire enabled tool list.

---

## Environment overrides

`apply_env_overrides(...)` supports a small, explicit set of override variables.

Currently supported:

- `HALTER_SESSION_BACKEND`
- `HALTER_POLICY_SHELL_ENABLED`
- `HALTER_POLICY_NETWORK_ENABLED`
- `HALTER_SKILL_ROOTS`
- `HALTER_PLUGIN_ROOTS`
- `HALTER_POLICY_SHELL_ALLOW`
- `HALTER_POLICY_ALLOWED_HOSTS`
- `HALTER_TOOLS_ENABLED`

### Parsing behavior

- `HALTER_SKILL_ROOTS` and `HALTER_PLUGIN_ROOTS` use `:` as the separator
- allow/host/tool lists use `,` as the separator
- booleans use Rust-style boolean parsing (`true` / `false`)

Examples:

```bash
export HALTER_SESSION_BACKEND=memory
export HALTER_POLICY_SHELL_ENABLED=false
export HALTER_SKILL_ROOTS=./skills:./vendor/skills
export HALTER_POLICY_SHELL_ALLOW=git,just,rg
export HALTER_TOOLS_ENABLED=read,glob,grep
```

Programmatic example:

```rust
use halter_config::apply_env_overrides;

let mut value: toml::Value = toml::from_str("version = 1")?;
apply_env_overrides(&mut value)?;
```

---

## JSON Schema export

You can render the config schema as JSON text or as a serde value.

```rust
use halter_config::{export_json_schema, schema_as_json_value};

fn main() -> anyhow::Result<()> {
    let text = export_json_schema()?;
    let value = schema_as_json_value();

    println!("{}", text);
    println!("schema title = {:?}", value.get("title"));
    Ok(())
}
```

This is what the CLI prints for `halter config schema`.

---

## Starter config generation

`generate_starter_config()` returns a parseable starter TOML string.

```rust
use halter_config::generate_starter_config;

fn main() {
    println!("{}", generate_starter_config());
}
```

This is what powers `halter init`.

---

## Fingerprinting

`config_fingerprint(...)` returns a stable SHA-256 hash of the effective config serialized as JSON.

```rust
use halter_config::{config_fingerprint, load_path};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let config = load_path("halter.toml").await?;
    println!("{}", config_fingerprint(&config));
    Ok(())
}
```

Typical use cases:

- cache keys
- reload detection
- telemetry / provenance
- config drift detection across environments

---

## Realistic configs

### Local coding-agent style setup

```toml
version = 1

[models.default]
provider = "openai"
model = "gpt-5"
reasoning = "high"

[models.subagent]
provider = "openai"
model = "gpt-5-mini"
reasoning = "medium"

[resources.skills]
roots = ["./.agent/skills"]

[resources.plugins]
roots = ["./.agent/plugins"]

[tools]
enabled = [
  "read",
  "glob",
  "grep",
  "write",
  "edit",
  "shell",
  "process",
  "wait_agent",
  "spawn_agent",
  "send_input",
  "close_agent",
]

[policy]
allowed_write_roots = ["./", "/tmp/halter"]
max_read_bytes = 1048576
max_subagent_depth = 3
max_concurrent_subagents = 8

[policy.shell]
enabled = true
allow = ["git", "cargo", "rg", "ls", "find", "true", "cd", "python"]
timeout_secs = 30

[sessions]
backend = "memory"
```

### OpenRouter-backed subagents with restricted shell

```toml
version = 1

[models.default]
provider = "openai"
model = "gpt-5"
reasoning = "high"

[models.subagent]
provider = "openrouter"
model = "openai/gpt-5-mini"
reasoning = "medium"

[policy]
allowed_write_roots = ["./"]
max_read_bytes = 524288
max_subagent_depth = 2
max_concurrent_subagents = 4

[policy.shell]
enabled = true
allow = ["git", "rg", "ls"]
timeout_secs = 15
```

Remember: that config requires OpenAI credentials and OpenRouter credentials.
OpenAI can use `OPENAI_API_KEY` or `[providers.openai].oauth`; OpenRouter uses
`OPENROUTER_API_KEY` or `[providers.openrouter].api_key`.

---

## Common failure modes

### "[models.default] is required"

You loaded a config with no default model.

### Missing provider credentials

The chosen provider was selected by a model role, but neither config nor
environment supplied credentials. For OpenAI, set `[providers.openai].api_key`,
`[providers.openai].oauth`, or `OPENAI_API_KEY`.

### Invalid SQLite config

You set `sessions.sqlite_path` without enabling the `sqlite` feature or without `backend = "sqlite"`.

### Context thresholds reversed

`pre_compaction_target` must be lower than `compaction_threshold`, and
`max_tokens` must be at least `compaction_threshold`. Derived values count:
an explicit `pre_compaction_target` above a threshold derived from
`max_input_tokens` is rejected too.

### No compaction threshold source

Neither `context.compaction_threshold` nor `models.default.max_input_tokens`
is set. Set one; the threshold derives from the window when only the window is
present.


### Empty strings where values are required

Provider base URLs, API keys, OpenAI OAuth fields, and model names are trimmed
and validated.

---

## Practical recommendations

- Keep `halter.toml` small and environment-specific secrets out of source control.
- Use `load_layered(...)` if you need a user → project → explicit override model.
- Use env overrides sparingly and only for the supported keys.
- Treat `tools.enabled` and shell allowlists as explicit policy, not convenience defaults.
- Fingerprint the final merged config if you cache resource or runtime artifacts.

---

## Related docs

- `../halter/README.md` — high-level harness assembly
- `../halter-cli/README.md` — CLI behavior on top of this schema
- `../halter-tools/README.md` — tool names, policies, feature-gated tool families
