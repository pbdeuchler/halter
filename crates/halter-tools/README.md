# halter-tools

`halter-tools` provides halter's tool runtime, built-in tool catalog, policy enforcement, and subagent-control tools.

This crate is unusually important because it matters to both core personas:

- **programmers** embedding the halter runtime and extending tool behavior
- **users/operators** interacting with the CLI and expecting predictable tool semantics

If you want to know what `read`, `shell`, `spawn_agent`, `wait_agent`, `edit`, or `glob` actually do, this is the crate.

---

## Who this crate is for

### Primary: programmers extending or embedding halter

Use this crate when you need to:

- register built-in tools in a runtime
- add your own custom tool implementations
- control filesystem/shell/network/subagent policy
- understand tool execution events
- plug in a session store for tool-side session awareness
- enable optional tool families via Cargo features

### Secondary: CLI users and technical operators

If you run `halter run` or `halter chat`, this crate defines what the agent can actually do.

It determines:

- which tools are available
- which arguments they accept
- policy errors and limits
- shell allowlisting behavior
- how subagents are spawned and managed

For many users, this crate is the practical heart of the system.

---

## Mental model

A halter tool is a named operation callable by the model at runtime.

The tool system has four layers:

1. **tool definitions** — names, schemas, handlers
2. **tool runtime** — invocation and event plumbing
3. **policy** — what is allowed, where, how much, and how often
4. **tool catalog** — built-in tools plus optional feature-gated tools

The runtime asks for a tool by name. This crate decides:

- whether it exists
- whether it is enabled
- whether policy allows it
- how to execute it
- what output and events come back

---

## Public API at a glance

### Core runtime abstractions

- `Tool`
- `ToolContext`
- `ToolRuntime`
- `ToolRuntimeEvent`
- `ToolEventSink`
- `SubagentControl`
- `SubagentParentContext`
- `ToolSessionStore`

### Policy surface

- `ToolPolicy`
- `DefaultToolPolicy`
- `PolicySettings`

### Registration helpers

- `register_builtin_tools`
- `register_subagent_tools`

### Built-in tools

- `ReadTool`
- `WriteTool`
- `EditTool`
- `GlobTool`
- `GrepTool`
- `ShellTool`
- `ProcessTool`

### Optional tools by Cargo feature

- `PtyTool` (`pty`)
- `AstGrepTool` (`ast-tools`)
- `BrowserTool` (`browser-tools`)
- `ImageTool` (`image-tools`)
- `ProfilingTool` (`profiling`)

### Subagent-control tools

- `spawn_agent`
- `send_input`
- `wait_agent`
- `close_agent`

---

## Cargo features

This crate supports capability slicing via features:

- `advanced-tools`
- `ast-tools`
- `browser-tools`
- `image-tools`
- `pty`
- `profiling`
- `full`

Practical interpretation:

- a tool must be compiled into the binary **and** enabled by config/policy to be usable
- naming a feature-gated tool in config does not magically compile it in
- `full` is the easiest way to get the broadest built-in catalog

---

## Policy model

## `PolicySettings`

This struct defines the default policy envelope.

Defaults:

- `allowed_write_roots = [".", "/tmp/halter"]`
- `allowed_read_roots = [".", $TMPDIR | "/tmp"]`
- `sensitive_path_patterns = ["**/.ssh/**", "**/.aws/**", "**/.env", "**/.env.*", "/etc/shadow", "/etc/shadow.*"]`
- `max_read_bytes = 1_048_576`
- shell enabled = `true`
- `shell_mode = Strict` (rejects `eval`, `exec`, `source`, `.`, and function definitions at the AST level)
- shell allowlist = `git`, `cargo`, `rg`, `ls`, `find`
- shell timeout = `30`
- network enabled = `false`
- `allowed_loopback = []` (loopback addresses require an explicit entry to be reached)
- `allowed_hosts = ["*"]` (wildcard — remote hosts allowed when `network_enabled = true`)
- `process_tree_root = None` (Phase 2 threads the live session's root)
- `max_subagent_depth = 3`
- `max_concurrent_subagents = 8`

When the high-level `halter` builder constructs policy from `PolicyConfig`, it
also adds configured `allowed_write_roots` to the runtime read roots. This lets
agents inspect files under any root they are allowed to modify, including
generated worktrees.

This is the main security and operability boundary for tool use.

`ToolPolicy` exposes a capability-oriented surface
(`check_read_path`, `check_write_path`, `check_process_signal`,
`check_shell_enabled`, `check_shell_command_strict`, `check_network`,
`check_subagent_spawn_typed`, `shell_mode`) that returns `PolicyError`
and binds resolved paths to a parent-directory fd via `CanonicalPath`.
Name-based bypass methods (`check_shell("shell")`,
`check_shell("process")`) and the older string-error surface have been
removed; built-in tools call the typed methods directly.

### Why policy lives here

The model may *request* a tool action, but the tool layer decides whether that action actually happens.

That separation is essential.

---

## Important policy failures

You will commonly see errors like:

- `failed to execute shell tool: shell usage is disabled by policy`
- `failed to execute shell tool: program '{}' is not in the allowlist`
- `failed to execute write tool: path '{}' is outside allowed_write_roots`
- `failed to execute spawn_agent tool: subagent depth {} exceeds max_subagent_depth {}`
- `failed to execute spawn_agent tool: active subagents {} exceed max_concurrent_subagents {}`

These are not bugs. They are explicit policy enforcement.

---

## Built-in tools

## `read`

Reads UTF-8 text from a file.

Behavior:

- reads by lines, not arbitrary bytes
- default and maximum `limit = 500`
- subject to `max_read_bytes` policy

Typical input:

```json
{
  "path": "src/main.rs",
  "offset": 1,
  "limit": 200
}
```

Typical use cases:

- inspect source files
- read configs
- peek at logs or generated artifacts

### User guidance

If you're prompting the CLI agent, ask it to read specific files or ranges rather than vaguely saying "look around". You'll get better results.

---

## `write`

Writes a UTF-8 file atomically.

Typical input:

```json
{
  "path": "README.md",
  "content": "# New file\n"
}
```

Characteristics:

- whole-file write
- atomic replacement semantics
- constrained by `allowed_write_roots`

Use `write` when you want to create or fully replace a file.

---

## `edit`

Performs exact string replacement in a UTF-8 file using an atomic write.

Typical input:

```json
{
  "path": "Cargo.toml",
  "old_string": "edition = \"2021\"",
  "new_string": "edition = \"2024\"",
  "replace_all": false
}
```

Optional safety field:

- `expected_sha256`

This is useful when you want optimistic concurrency protection against editing stale file contents.

Use `edit` rather than `write` when:

- you want a targeted patch
- you need change locality
- you want to avoid rewriting unrelated content

---

## `glob`

Expands glob patterns relative to the working directory.

Typical input:

```json
{
  "pattern": "crates/*/src/**/*.rs",
  "file_type": "file",
  "max_results": 200,
  "sort_by_mtime": false
}
```

Useful for:

- codebase discovery
- finding manifests, configs, docs, tests
- locating generated or recently touched files

---

## `grep`

Searches file contents with regex filters and optional context.

Typical input:

```json
{
  "path": ".",
  "glob": "**/*.rs",
  "pattern": "impl SessionStore",
  "type": "rust",
  "ignore_case": false,
  "context_before": 2,
  "context_after": 2,
  "output_mode": "content",
  "multiline": false,
  "max_matches": 50,
  "offset": 0,
  "max_columns": 200
}
```

Output modes:

- `content`
- `count`
- `files_with_matches`

This is one of the most valuable tools for real repository work.

---

## `shell`

Runs a command in a persistent shell session.

Typical input:

```json
{
  "command": "cargo test -p halter-tools",
  "cwd": "/workspace/halter",
  "timeout_ms": 30000
}
```

Optional fields can include environment variables, depending on the calling integration.

### Important semantics

- persistent shell session, not one isolated process per call
- gated by `policy.shell.enabled`
- executable must be on the allowlist
- command runtime constrained by timeout policy

### Common allowlist surprise

If the model tries to use a program not on the allowlist, you'll get:

> `failed to execute shell tool: program '{}' is not in the allowlist`

### CLI-user advice

If your workflow needs `python`, `just`, `make`, `npm`, `docker`, or `sort`, add them explicitly to the shell allowlist.

---

## `process`

Inspects or terminates process trees.

Actions:

- `kill_tree`
- `list_descendants`

Typical input:

```json
{
  "action": "list_descendants",
  "pid": 12345,
  "signal": 15
}
```

Notes:

- `kill_tree` requires process control to be allowed
- internally this is checked through shell-style policy gating
- useful for cleaning up runaway tasks or inspecting spawned subprocess trees
- `kill_tree` response includes `per_pid: [{pid, killed}]` alongside the
  aggregate `killed` count, so callers can see which descendant refused the
  signal rather than just "3 of 5 killed"

---

## Optional built-in tools

## `pty`

Feature: `pty`

Actions:

- `start`
- `write`
- `resize`
- `kill`

Use this for interactive terminal workflows where a plain shell command is not enough.

Examples:

- running REPLs
- driving interactive installers
- handling TUI apps in a bounded way

---

## `ast_grep`

Feature: `ast-tools`

Actions:

- `find`
- `replace`

Use this for syntax-aware code search and rewrites where regex is too brittle.

This is especially useful for large-scale refactors.

---

## `image`

Feature: `image-tools`

Actions:

- `info`
- `resize`
- `convert`

Use this when the agent needs to inspect or transform local image files.

---

## `browser`

Feature: `browser-tools`

Drives a remote web browser via [playwright-rs](https://docs.rs/playwright-rs)
connected over CDP to a cloud provider. One persistent browser session per
halter `SessionId`, dispatched by `action`.

Actions:

- `navigate` — load a URL; returns title + ARIA snapshot with ref ids
- `snapshot` — re-fetch the ARIA snapshot
- `click` / `type` / `press` — act on a `ref` from the snapshot, or a raw `selector`
- `scroll` — `direction: "up" | "down"`
- `back` — history navigation
- `screenshot` — PNG; saved to `output_path` or returned base64
- `eval` — JavaScript expression, returns the value
- `console` — accumulated console messages + uncaught errors
- `close` — release the cloud session

Element addressing prefers ARIA refs that Playwright assigns natively in the
snapshot YAML (e.g. `ref="s1e3"`). Bare selectors work as fallback for cases
where ref-based addressing isn't enough (CSS, `role=…`, `text=…`).

URL navigation goes through `ToolPolicy::check_network`, including the
post-redirect URL — a 302 to a private address is rejected before snapshot
returns.

### Provider configuration

Currently one provider is shipped: `BrowserbaseProvider`. It activates when
both env vars are present:

- `BROWSERBASE_API_KEY`
- `BROWSERBASE_PROJECT_ID`

Optional knobs:

- `BROWSERBASE_BASE_URL` (default `https://api.browserbase.com`)
- `BROWSERBASE_PROXIES` (default `true` — falls back gracefully on plans without proxy support)
- `BROWSERBASE_KEEP_ALIVE` (default `true` — falls back on free plans)
- `BROWSERBASE_SESSION_TIMEOUT` (in milliseconds)

The trait `BrowserProvider` lets new adapters drop in without touching the
tool itself.

### Runtime requirements

playwright-rs spawns the Playwright Node driver. First-time setup needs:

```sh
npm install -g playwright
npx playwright install
```

The driver is launched once per process and shared across all halter
sessions.

---

## `profile`

Feature: `profiling`

Tool name exposed to the model is `profile`.

Use this when you want profiling/instrumentation workflows available to the agent.

---

## Subagent-control tools

These tools are central to halter's delegated-work model.

## `spawn_agent`

Starts a child session to handle a delegated task.

Typical input:

```json
{
  "message": "Audit the session persistence layer for conflict handling.",
  "fork_context": true,
  "model": "default"
}
```

Key semantics:

- starts a child session
- optionally inherits parent context
- subject to `max_subagent_depth`
- subject to `max_concurrent_subagents`

Common policy failures:

- depth too high
- too many active subagents

### User guidance

This tool works best when the delegated task is concrete and independently verifiable.

Good tasks:

- document one crate
- audit one subsystem
- gather test failures from one package

Bad tasks:

- "solve everything"
- vague multi-phase efforts with no crisp output contract

---

## `send_input`

Sends follow-up input to a child session after its current turn has reached a
terminal state.

Typical input:

```json
{
  "target": "agent-uuid",
  "message": "Focus only on public APIs and write the README to disk."
}
```

Use this when:

- the child completed or failed and needs a follow-up turn
- requirements changed after the child produced a result
- you want to tighten scope after collecting the current result

If the child is still running, use `wait_agent` to collect completion or
`close_agent` to stop it.

---

## `wait_agent`

Waits for one or more child sessions to reach terminal state.

Typical input:

```json
{
  "targets": ["agent-1", "agent-2"],
  "timeout_ms": 30000
}
```

This is essential for orchestrated parallel work.

If `timeout_ms` expires before any target reaches terminal state, the response
has `timed_out: true`, `status: null`, and `target_statuses` containing the
current status of each requested target.

---

## `close_agent`

Closes a child session. If the child is running, this cancels in-progress work;
closed children no longer accept `send_input`.

Typical input:

```json
{
  "target": "agent-uuid"
}
```

Use it to clean up control surfaces once delegated work is done.

---

## Registering tools programmatically

### Register built-ins

Use `register_builtin_tools(...)` to populate a runtime with standard tools.

### Register subagent tools

Use `register_subagent_tools(...)` when the runtime has subagent control wired in.

### Add your own tools

Implement `Tool` and register it with the runtime.

Conceptual example:

```rust
use halter_tools::{Tool, ToolContext};

struct MyTool;

// implement Tool for MyTool
```

A well-behaved custom tool should:

- have a clear, narrow purpose
- validate inputs strictly
- emit bounded outputs
- honor context/policy expectations
- fail clearly when preconditions are not met

---

## Tool runtime and events

`ToolRuntime` is the engine that looks up tools, enforces policy, executes handlers, and emits tool-level events.

Key associated types:

- `ToolRuntimeEvent`
- `ToolEventSink`
- `ToolContext`

### What belongs in `ToolContext`

Typical tool execution context includes things like:

- working directory
- policy handle
- event sink
- session metadata
- resource or runtime references
- subagent control access

This is what makes tools runtime-aware rather than mere pure functions.

---

## Tool session store integration

`ToolSessionStore` exists so tool execution can coordinate with session-related state when needed.

This matters most for:

- subagent lifecycle tools
- transcript-aware tooling
- operations that need session lineage or audit context

---

## User-facing workflows

## Workflow: inspect and patch code

Typical sequence:

1. `glob` to find files
2. `grep` to locate relevant symbols
3. `read` targeted sections
4. `edit` precise replacements
5. `shell` to run tests

This is the default bread-and-butter coding-agent loop.

---

## Workflow: generate documentation

Typical sequence:

1. `glob` for crates/manifests/docs
2. `read` source and config files
3. `grep` public types and commands
4. `write` README drafts
5. `edit` for iterative refinement
6. `spawn_agent` for per-subsystem parallelization

---

## Workflow: interactive debugging

Typical sequence:

1. `shell` run failing tests
2. `read` stack traces or logs
3. `process` inspect stray subprocesses if needed
4. `edit` fixes
5. `shell` rerun verification
6. optionally `pty` for interactive reproductions

---

## Security and operational guidance

- Keep shell allowlists tight.
- Keep write roots narrow.
- Prefer `edit` over broad `write` when patching existing files.
- Limit subagent depth and concurrency deliberately.
- Enable optional tool families only if your deployment really needs them.

### Hooks vs tool policy

Tool policy is the hard mechanical boundary.
Hooks are the event-driven overlay.

Use policy for:

- filesystem scope
- shell program allowlisting
- output/read limits
- subagent ceilings

Use hooks for:

- approvals
- redaction
- workflow annotations
- event-specific custom rules

---

## Practical config examples

### Conservative local coding setup

```toml
[tools]
enabled = ["read", "glob", "grep", "write", "edit", "shell", "process"]

[policy]
allowed_write_roots = ["./"]
max_read_bytes = 1048576
max_subagent_depth = 1
max_concurrent_subagents = 2

[policy.shell]
enabled = true
allow = ["git", "cargo", "rg", "ls", "find"]
timeout_secs = 20
```

### Delegation-heavy documentation or analysis setup

```toml
[tools]
enabled = [
  "read", "glob", "grep", "write", "edit", "shell",
  "spawn_agent", "wait_agent", "send_input", "close_agent"
]

[policy]
allowed_write_roots = ["./", "/tmp/halter"]
max_read_bytes = 1048576
max_subagent_depth = 3
max_concurrent_subagents = 8

[policy.shell]
enabled = true
allow = ["git", "cargo", "rg", "ls", "find", "python", "echo"]
timeout_secs = 30
```

---

## Common mistakes

### Confusing config enablement with feature compilation

If `pty` is not compiled in, listing it under `tools.enabled` is not enough.

### Allowlisting too little for real workflows

A shell allowlist of only `git` and `ls` will frustrate most coding flows.

### Allowlisting too much without review

Adding `rm`, `sudo`, `docker`, or arbitrary network tools changes your risk posture significantly.

### Using `write` when `edit` would be safer

Whole-file rewrites are more error-prone when only a small patch is needed.

### Delegating without synchronization

If you `spawn_agent` repeatedly and never `wait_agent`, you can exhaust concurrency limits or lose track of task completion.

---

## Recommendations for tool authors

- Keep argument schemas small and obvious.
- Return structured, bounded outputs.
- Make failure messages operationally useful.
- Treat working directory and path handling carefully.
- Emit events consistently so users can debug what happened.
- Avoid hidden network access or undeclared side effects.

---

## Recommendations for CLI users

- Be explicit about files, directories, and commands you want the agent to inspect.
- Tune your shell allowlist to your actual stack.
- Prefer repo-local `allowed_write_roots` whenever possible.
- Use subagents for parallelizable, narrow tasks.
- If a tool call fails, read the policy error literally; it usually tells you exactly what to fix.

---

## Related docs

- `../halter-cli/README.md` — user-facing commands that exercise these tools
- `../halter-config/README.md` — policy and `tools.enabled` configuration
- `../halter-runtime/README.md` — runtime integration of the tool engine
- `../halter-hooks/README.md` — event-driven policy overlays around tool usage
- `../halter/README.md` — top-level builder APIs for registering tools and policies
