# halter-cli

`halter-cli` packages the halter SDK as a portable command-line binary.

This is the crate for people who want to **run** halter rather than embed it. It exposes a thin interface over the `halter` crate and supports a predictable config-driven workflow:

- initialize a starter config
- validate config and credentials
- inspect compiled resources
- run a single task
- open an interactive chat loop
- emit JSON suitable for automation

---

## Who this crate is for

### Primary: users operating the CLI

Use `halter` when you want to:

- keep agent configuration in `halter.toml`
- invoke a task from a terminal or script
- stream JSON events into another program
- debug resource loading and configuration

### Secondary: programmers integrating the SDK

This crate is also useful as a reference for how to wire the SDK together. `src/main.rs` shows a minimal but complete application using:

- `Halter::from_config_file(...)`
- `SessionInit::default()`
- `session.submit_turn(...)`
- event streaming and final result extraction

If you are embedding halter in Rust, start with `../halter/README.md`.

---

## Binaries

This crate currently exposes two binary names pointing at the same entrypoint:

- `halter`
- `halter-cli`

The default run target is `halter`.

---

## Installation / running

From the workspace root:

```bash
cargo run -p halter-cli -- --help
```

Install the binary locally:

```bash
cargo install --path crates/halter-cli
```

With extra features enabled:

```bash
cargo install --path crates/halter-cli --features full,sqlite
```

Feature forwarding:

- `advanced-tools`
- `ast-tools`
- `image-tools`
- `pty`
- `profiling`
- `full`
- `sqlite`

`full` enables the optional tool families exposed by the SDK.

---

## Global options

All commands support:

```text
--config <CONFIG>
--output-file <OUTPUT_FILE>
```

### `--config`

Defaults to:

```text
halter.toml
```

Example:

```bash
halter --config ./examples/halter.example.toml validate
```

### `--output-file`

Writes normal command output to a file instead of stdout.

Important behavior: when `--output-file` is set, tracing/log output is also directed to the **same file**.

That makes it useful for automation and debugging in a single artifact.

Example:

```bash
halter --output-file run.jsonl run --streaming-json "Summarize this repo"
```

---

## Command overview

```text
halter init
halter chat
halter run [--json-result | --streaming-json] (<TASK> | --prompt-file <PROMPT_FILE>)
halter resources
halter validate
halter config schema
```

---

## `halter init`

Creates a starter config at the `--config` path.

```bash
halter init
```

Typical output:

```text
wrote halter.toml
```

If the target file already exists, the command fails rather than overwriting it.

### What gets generated

The starter config includes:

- `version = 1`
- a required `[models.default]`
- default skill/plugin roots
- default tool and policy settings
- memory-backed sessions by default
- SQLite comments when the binary was built with the `sqlite` feature

---

## `halter validate`

Validates config structure and runtime prerequisites.

```bash
halter validate
```

Typical output:

```text
config valid
```

Validation covers more than TOML syntax. It also checks things like:

- `[models.default]` exists
- selected providers have credentials available
- context thresholds are internally consistent
- `sessions.sqlite_path` is only used when supported

Example with explicit config path:

```bash
halter --config ./examples/halter.example.toml validate
```

---

## `halter resources`

Compiles resources and prints a small summary.

```bash
halter resources
```

Typical output:

```text
revision: 8d4d3f...
skills: 12
agents: 3
plugins: 2
```

Use this when you want to verify:

- your skill roots are being discovered
- your plugin manifests are valid
- agents and hooks are being pulled in as expected
- a resource reload would change the snapshot revision

---

## `halter run`

Runs a single user task in a fresh session.

```bash
halter run "Review the staged diff and summarize the main risk"
```

For longer prompts, keep the prompt in a file and pass the file path:

```bash
halter run --prompt-file ./prompt.md
```

Internally this command does the following:

1. loads the config
2. builds a `Halter`
3. creates `SessionInit::default()`
4. reads `--prompt-file` when provided
5. submits one `Turn::user(task)`
6. prints either the final assistant message or the full event stream

### Output modes

#### Default: `--json-result`

The default behavior is to emit the **final assistant message** as JSON.

```bash
halter run "Summarize this repository"
```

Equivalent explicit form:

```bash
halter run --json-result "Summarize this repository"
```

The CLI strips reasoning signatures from assistant thinking blocks before printing.

This mode is best for:

- shell scripts
- CI integration
- capturing the final answer only

#### Streaming: `--streaming-json`

Emit each `SessionEvent` as newline-delimited JSON.

```bash
halter run --streaming-json "Explain how subagents work here"
```

This is best for:

- custom UIs
- telemetry pipelines
- event-based integrations
- debugging turn execution, hooks, and tool usage

Example shell pipeline:

```bash
halter run --streaming-json "Summarize current repo status" \
  | jq -c '.payload.kind'
```

### Examples

Single-shot human use:

```bash
halter run "List the crates in this workspace and what each one does"
```

File-backed prompt:

```bash
halter run --prompt-file ./prompts/release-review.md
```

Write final JSON result to a file:

```bash
halter run --output-file result.json "Produce a release checklist"
```

Write streaming events plus logs into one file:

```bash
RUST_LOG=info halter run \
  --streaming-json \
  --output-file trace.jsonl \
  "Review the current branch"
```

### Failure behavior

If the runtime emits `TurnFailed`, the command exits with an error.

If streaming mode is used, emitted events are flushed as they arrive.

---

## `halter chat`

Starts a simple stdin/stdout chat loop backed by a single session.

```bash
halter chat
```

Startup banner:

```text
halter chat; submit an empty line or press ctrl-d to exit
```

Behavior:

- each non-empty input line becomes a `Turn::user(...)`
- assistant text deltas are printed as they stream
- tool output chunks are also printed inline
- the loop exits on an empty line or EOF

This mode is intentionally minimal. It is useful for quick operator workflows, not for building a rich terminal UI.

### Example session

```text
$ halter chat
halter chat; submit an empty line or press ctrl-d to exit
what resources are loaded?
...assistant output streams here...
```

---

## `halter config schema`

Prints the `HarnessConfig` JSON Schema as formatted JSON.

```bash
halter config schema
```

Use cases:

- editor/schema integration
- validation in external tools
- generating docs for your own internal operator platform

Example:

```bash
halter config schema > halter.schema.json
```

---

## Output redirection details

`--output-file` is intentionally global and works before or after the subcommand.

These are both accepted:

```bash
halter --output-file out.jsonl run "task"
halter run --output-file out.jsonl "task"
```

When writing to a file:

- command output is written there
- trace output is written there too
- writes are line-buffered and flushed explicitly

That means one file can contain both JSON/event payloads and trace logs. For strict machine parsing, set `RUST_LOG=off` or leave it unset.

---

## Logging behavior

Logging is disabled by default unless `RUST_LOG` is set.

Examples:

```bash
RUST_LOG=info halter validate
RUST_LOG=debug halter resources
RUST_LOG=halter_runtime=debug,halter_tools=debug halter run "inspect repo"
```

Formatting behavior:

- normal terminal output: compact tracing formatter
- file output mode: JSON tracing formatter

If `RUST_LOG` is missing, the CLI installs an `off` filter.

---

## Realistic workflows

### 1. Bootstrap a new workspace

```bash
halter init
$EDITOR halter.toml
export OPENAI_API_KEY=...
halter validate
halter resources
halter run "Describe the loaded agent environment"
```

### 2. Use the CLI in automation

```bash
halter run --json-result "Generate a changelog summary" > result.json
jq .parts result.json
```

### 3. Stream events into another program

```bash
halter run --streaming-json "Review the repository" \
  | python scripts/consume_events.py
```

### 4. Debug config and resource loading

```bash
RUST_LOG=debug halter validate
RUST_LOG=debug halter resources
```

---

## Config expectations

By default the CLI looks for `halter.toml` in the current working directory.

The config must define at least:

- `version = 1`
- `[models.default]`
- provider credentials, either in config or environment

A typical minimal setup is:

```toml
version = 1

[models.default]
provider = "openai"
model = "gpt-5"
reasoning = "medium"

[resources.skills]
roots = ["./.agent/skills"]

[resources.plugins]
roots = ["./.agent/plugins"]
```

For full config documentation, see `../halter-config/README.md`.

---

## How the CLI maps onto the SDK

Each top-level command is a straightforward SDK call:

- `init` → `generate_starter_config()`
- `validate` → `load_path(...)`
- `resources` → `ResourceCompiler::from_config(&config).compile().await`
- `run` → `Halter::from_config_file(...)`, `new_session(...)`, `submit_turn(...)`
- `chat` → same as `run`, but loops on stdin
- `config schema` → `export_json_schema()`

That makes this crate a good reference implementation for your own thin wrapper.

---

## What this CLI intentionally does not do

This binary is intentionally lightweight. It does **not** currently provide:

- a full-screen TUI
- persistent chat history commands
- built-in resume/list session commands
- advanced approval UX
- multi-profile config management

Those are all possible higher-level applications on top of the workspace crates, but they are not the responsibility of this binary today.

---

## Related crate docs

- `../halter/README.md` — embedding API
- `../halter-config/README.md` — config schema and loading
- `../halter-tools/README.md` — built-in tools and tool policy
- `../halter-runtime/README.md` — event streams, sessions, compaction, subagents
