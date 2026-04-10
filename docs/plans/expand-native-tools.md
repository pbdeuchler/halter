# Expand Native Tools

Port the native tool surface from oh-my-pi's `pi-natives` crate into halter, adapting the N-API/JS patterns into halter's async `Tool` trait. Basic tools ship by default; advanced optimizations (rayon parallel grep, memmap2) live behind an `advanced-tools` feature flag. Adds file-level locking to make memmap2 safe under concurrent agent tool use.

## Context

Halter currently has 4 native tools: `read`, `write`, `glob`, `shell`. This plan ports the following from `../../can1357/oh-my-pi/crates/pi-natives/src/`:

| Module | LOC | Halter mapping |
|--------|-----|----------------|
| `grep.rs` | 1780 | `GrepTool` — ripgrep content search |
| `ast.rs` + `language/` | 1064+ | `AstGrepTool` — structural search/replace |
| `shell.rs` | 1169 | `ShellTool` replacement — persistent session |
| `pty.rs` | 483 | `PtyTool` — interactive PTY execution |
| `ps.rs` | 295 | Internal `ps` module — process tree mgmt for shell/pty |
| `image.rs` | 189 | `ImageTool` — decode/resize/encode |
| `prof.rs` | 249 | Internal profiling infrastructure |
| `task.rs` | 348 | Cancellation patterns (merge with existing `CancellationToken`) |
| `text.rs` | 1507 | Internal `text` module — ANSI-aware output truncation |

## Architecture Decisions

- **Module restructure:** `builtin.rs` → `builtin/` directory. Each tool gets its own submodule. Shared infrastructure lives in `builtin/common.rs` and new internal modules (`fs_lock.rs`, `profiling.rs`, `text.rs`, `process.rs`).
- **File-level locking for memmap2 safety:** A `PathLockMap` provides per-path `RwLock` semantics. Read tools (grep, read) acquire shared read locks. Write tools (write, edit, ast_replace) acquire exclusive write locks. This prevents SIGBUS from concurrent mmap+write on the same file. Implemented as `Arc<DashMap<PathBuf, Arc<RwLock<()>>>>` with LRU eviction.
- **Feature flags:** `advanced-tools` enables rayon + memmap2 for grep. `ast-tools` enables ast-grep-core + tree-sitter grammars. `image-tools` enables image crate. `pty` enables portable-pty. Base halter ships with grep (sequential/fs::read), edit, enhanced read/glob, and persistent shell — no heavy optional deps.
- **Port oh-my-pi's ripgrep implementation directly:** Use `grep-regex`, `grep-matcher`, `grep-searcher` (same as oh-my-pi's `fff-grep` wraps). Port `sanitize_braces`, `escape_unescaped_parentheses`, `resolve_type_filter`, `MatchCollector` sink, `run_parallel_search`, `read_file_bytes` (with memmap2), binary detection, and context line extraction. Adapt from N-API blocking tasks to `tokio::task::spawn_blocking`.
- **Port oh-my-pi's shell pattern:** Replace current `ShellTool` (subprocess with captured output) with a persistent `brush-core` shell session that survives across tool calls within a session. Streaming output via `ToolEventSink`. PTY support for interactive commands.
- **Profiling:** Always-on circular buffer profiler (port `prof.rs`). Each tool execution records timing. `get_work_profile()` returns folded stacks + flamegraph SVG. Exposed as a tool so agents can self-diagnose performance.
- **Edit tool uses hashline method:** SHA256 of full file verified between read and write. Atomic write via temp+rename. Error on no-match or ambiguous match.
- **Read offset/limit in lines.** Grep capped by `max_matches` count (default 100).

---

## Step 1: Infrastructure Foundation

Restructure `halter-tools` into a module directory and add the shared infrastructure that all subsequent steps depend on.

**Changes:**

1. **Module restructure:** `builtin.rs` → `builtin/mod.rs` + `builtin/common.rs` (shared helpers) + `builtin/{read,write,glob,shell}.rs` (existing tools, moved verbatim)

2. **File-level locking (`builtin/fs_lock.rs`):** Port the concurrency safety layer.
   ```
   PathLockMap:
     acquire_read(path) -> ReadGuard   // shared, allows concurrent reads/mmaps
     acquire_write(path) -> WriteGuard  // exclusive, blocks reads and other writes
   ```
   Internally: `DashMap<CanonicalPath, Arc<tokio::sync::RwLock<()>>>` with weak-ref eviction when refcount drops to 1. Read tools call `acquire_read` before file access. Write tools call `acquire_write` before mutation. This makes memmap2 safe.

3. **Text utilities (`builtin/text.rs`):** Port oh-my-pi's `text.rs` ANSI-aware truncation. Needed by grep and read for output capping. Key functions: `visible_width(line)`, `truncate_to_width(line, max_cols)`. Strip the N-API/UTF-16 machinery — work directly with `&str`.

4. **Process management (`builtin/process.rs`):** Port oh-my-pi's `ps.rs` verbatim (platform-specific `collect_descendants`, `kill_tree`, `kill_pid`, `kill_process_group`). Used by shell/pty for cleanup. Cross-platform: Linux `/proc`, macOS `libproc`, Windows toolhelp snapshot.

5. **Profiling (`builtin/profiling.rs`):** Port oh-my-pi's `prof.rs` circular buffer profiler. `profile_region(tag) -> ProfileGuard` (RAII timing). `get_work_profile(last_seconds) -> WorkProfile` (folded stacks + summary + optional SVG via `inferno`). Wire into `emit_started`/`emit_completed` in `common.rs` so all tool executions are profiled automatically.

6. **Enhanced `ToolContext`:** Add `path_locks: Arc<PathLockMap>` field. Populate from `HalterBuilder`. All tools access locking through context.

**New dependencies:** `dashmap`, `parking-lot`, `inferno` (optional, behind `profiling` feature), `libc` (unix), `smallvec`

**Files:** `halter-tools/src/builtin.rs` → `halter-tools/src/builtin/{mod,common,read,write,glob,shell,fs_lock,text,process,profiling}.rs`, `halter-tools/src/lib.rs`, `halter-tools/Cargo.toml`, `halter-protocol` (add `PathLockMap` to `ToolContext`)

**Verify:** All existing tests pass unchanged. New tests: `PathLockMap` allows concurrent reads; `PathLockMap` blocks write during read; `kill_tree` on a spawned process group terminates children; `visible_width` handles ANSI escape sequences; profiling records timing for registered regions.

---

## Step 2: Grep + Enhanced Read + Edit + Enhanced Glob

Port oh-my-pi's full ripgrep implementation. Add enhanced read, edit tool, and enhanced glob.

**New dependencies:** `grep-regex`, `grep-matcher`, `grep-searcher`, `memmap2` (optional, behind `advanced-tools`), `rayon` (optional, behind `advanced-tools`)

### GrepTool (`builtin/grep/`)

Port oh-my-pi's grep.rs directly, adapting N-API patterns to halter's `Tool` trait:

- **Module structure:** `grep/mod.rs` (Tool impl), `grep/types.rs` (shared types), `grep/basic.rs` (sequential search), `grep/advanced.rs` (rayon parallel + memmap2, behind `advanced-tools`)
- **`grep/mod.rs`** re-exports via `#[cfg(feature = "advanced-tools")]` selecting between basic/advanced
- **Port directly from oh-my-pi:** `MatchCollector` sink impl, `SearchParams`, `run_search`, `build_searcher`, `read_file_bytes` (with memmap2 under feature flag, `fs::read` otherwise), `extract_context_lines`, `run_parallel_search` (rayon `par_iter` with `map_init`), `run_sequential_search`, `collect_files`, `resolve_type_filter` (full extension table), `sanitize_braces`, `escape_unescaped_parentheses`, `build_matcher` with fallback retry
- **Locking:** Call `path_locks.acquire_read(path)` before `read_file_bytes` for each file
- **Bridge:** `tokio::task::spawn_blocking` wraps the entire grep operation (CPU-bound regex)
- **Safety:** Pattern length cap 1000 chars, `size_limit(10MB)`, `dfa_size_limit(10MB)` on regex builder

**Schema:**
```
grep({ pattern, path?, glob?, type?, ignore_case?, multiline?,
       context_before?, context_after?, max_matches?, offset?, max_columns?,
       output_mode?: "content" | "count" | "files_with_matches" })
```

**Returns:** `{ matches: [{path, line_number, line, context_before?, context_after?, truncated?, match_count?}], total_matches, files_searched, files_with_matches, truncated }`

### Enhanced ReadTool

- Add `offset` (1-indexed line) and `limit` (max lines) optional params
- Call `path_locks.acquire_read` before reading
- Call `check_read` with actual returned byte count
- Return `total_lines` in response

### EditTool (`builtin/edit.rs`)

- **Schema:** `edit({ path, old_string, new_string, replace_all? })`
- **Flow:** `check_write` → `path_locks.acquire_write` → read file → compute SHA256 → find `old_string` → error if 0 matches or >1 without `replace_all` → replace → write to tempfile in same dir → `rename` atomic → return `{ path, matches_replaced, file_hash_before, file_hash_after }`
- **Concurrency:** `Exclusive`, `mutating: true`

### Enhanced GlobTool

- Add `max_results`, `file_type` ("file"|"dir"|"symlink"), `sort_by_mtime` optional params
- Collect metadata during walk when mtime sorting requested
- Early-break on max_results when not sorting; sort+truncate when sorting
- Return `{ matches: [{path, file_type?, mtime?}], total_matches }`

**Files:** `halter-tools/Cargo.toml`, `halter-tools/src/builtin/{grep/{mod,types,basic,advanced},edit,read,glob}.rs`

**Verify:** Grep: single-file match, directory recursive, type filter, glob filter, ignore_case, multiline, context lines, max_matches truncation, count mode, files_with_matches mode, binary skipped, regex DoS rejected, gitignore respected, cancellation honored, file locking prevents SIGBUS. Edit: single replace, replace_all, no-match error, ambiguous error, atomic write, policy denial. Read: offset/limit, beyond-file offset. Glob: max_results, file_type filter, mtime sort.

---

## Step 3: Persistent Shell + PTY + Process Tool

Replace the existing `ShellTool` with oh-my-pi's persistent shell session pattern and add PTY support.

**New dependencies:** `brush-core` (workspace, from vendored crate), `brush-builtins` (workspace, from vendored crate), `portable-pty` (optional, behind `pty` feature)

### Persistent ShellTool replacement (`builtin/shell/`)

Port oh-my-pi's `shell.rs` persistent session pattern:

- **Module structure:** `shell/mod.rs` (Tool impl), `shell/session.rs` (brush-core session management), `shell/streaming.rs` (output streaming via `ToolEventSink`)
- **Key change from current ShellTool:** Session persists across tool calls within a `HalterSession`. Environment, working directory, and shell state carry over. This matches how oh-my-pi's `Shell` class holds a `TokioMutex<Option<ShellSessionCore>>`.
- **Port directly:** `create_session`, `run_shell_command`, `run_shell_session` with cancellation/timeout/abort semantics, `session_keepalive` logic
- **ToolContext addition:** `shell_session: Arc<TokioMutex<Option<ShellSessionCore>>>` — shared across calls within a session
- **Streaming:** Output chunks emitted via `ToolEventSink` as `ToolRuntimeEvent::ShellOutput { chunk }` so callers can display progressively
- **Cleanup:** On session end or timeout, kill process tree using `process.rs` `kill_tree`

**Schema (unchanged name, enhanced behavior):**
```
shell({ command, cwd?, env?, timeout_ms? })
```
Returns: `{ exit_code, stdout, stderr, timed_out, cancelled }`

### PtyTool (`builtin/pty.rs`) — behind `pty` feature

Port oh-my-pi's `pty.rs` for interactive command execution:

- **Schema:** `pty_start({ command, cwd?, env?, timeout_ms?, cols?, rows? })` — starts PTY session
- `pty_write({ input })` — write stdin
- `pty_resize({ cols, rows })` — resize terminal
- `pty_kill()` — force kill
- Alternatively, implement as a single stateful `PtyTool` with an `action` field discriminator
- Output streamed via `ToolEventSink`
- Uses `portable-pty` crate for cross-platform PTY allocation
- Process cleanup via `kill_tree`

### ProcessTool (`builtin/process.rs` exposed as tool)

Expose process management as an agent-facing tool:
- **Schema:** `process({ action: "kill_tree" | "list_descendants", pid, signal? })`
- Wraps the internal `ps` module ported in Step 1
- Policy: `requires_approval: true` (killing processes is destructive)

**Files:** `halter-tools/Cargo.toml`, `halter-tools/src/builtin/shell/{mod,session,streaming}.rs`, `halter-tools/src/builtin/pty.rs`, update `builtin/process.rs` to also be a Tool, `halter-protocol` (new event variants), `halter-runtime/src/session.rs` (wire shell session into ToolContext)

**Verify:** Shell: persistent state across calls (cd in one call visible in next), timeout kills process tree, streaming output emitted, cancellation works. PTY: start/write/kill lifecycle, resize propagates, timeout cleans up. Process: kill_tree terminates children, list_descendants returns correct PIDs, policy blocks unauthorized kills.

---

## Step 4: AST Structural Search + Replace

Port oh-my-pi's `ast.rs` and `language/` module for AST-aware code search and rewrite.

**New dependencies:** `ast-grep-core` (behind `ast-tools` feature), tree-sitter grammar crates for supported languages (vendored or via `ast-grep-core`'s built-in set)

### AstGrepTool (`builtin/ast/`)

Port oh-my-pi's AST find and replace operations:

- **Module structure:** `ast/mod.rs` (Tool impl with `action` discriminator), `ast/find.rs` (structural search), `ast/replace.rs` (structural rewrite), `ast/language.rs` (port `language/mod.rs` — `SupportLang` enum with extension→language mapping and `impl Language` for each)
- **Port directly:** `compile_pattern`, `collect_candidates`, `is_supported_file`, `resolve_language`, `apply_edits` (non-overlapping edit application), `normalize_pattern_list`, `infer_single_replace_lang`
- **Find schema:** `ast_grep({ action: "find", patterns, path?, glob?, lang?, strictness?, limit?, offset?, include_meta? })`
- **Replace schema:** `ast_grep({ action: "replace", rewrites: {pattern: replacement}, path?, glob?, lang?, strictness?, dry_run?, max_replacements?, max_files? })`
- **Locking:** Find uses `acquire_read`. Replace uses `acquire_write` per file + `check_write` policy.
- **Replace safety:** `dry_run` defaults to `true` (safe by default). Agent must explicitly set `dry_run: false` to write. `max_replacements` and `max_files` cap blast radius.

**Find returns:** `{ matches: [{path, text, byte_start, byte_end, start_line, start_column, end_line, end_column, meta_variables?}], total_matches, files_searched, files_with_matches, limit_reached, parse_errors? }`

**Replace returns:** `{ changes: [{path, before, after, start_line, end_line}], file_changes: [{path, count}], total_replacements, files_touched, files_searched, applied, limit_reached, parse_errors? }`

**Files:** `halter-tools/Cargo.toml`, `halter-tools/src/builtin/ast/{mod,find,replace,language}.rs`

**Verify:** Find: match Rust function by pattern, match with meta-variables, multi-pattern OR, glob filter, language inference from extension, strictness modes, limit/offset pagination. Replace: dry_run produces changes without writing, applied mode writes atomically, overlapping edits rejected, max_replacements caps work, policy gates writes, parse errors collected non-fatally.

---

## Step 5: Image Tool + Profiling Tool + Feature Flag Wiring

Final tools and feature flag integration.

### ImageTool (`builtin/image.rs`) — behind `image-tools` feature

Port oh-my-pi's `image.rs`:

- **Schema:** `image({ action: "info" | "resize" | "convert", path, width?, height?, format?, quality?, filter? })`
- `info`: decode and return dimensions + format
- `resize`: resize to target dimensions with configurable filter (nearest, triangle, catmull-rom, gaussian, lanczos3)
- `convert`: re-encode to target format (PNG, JPEG, WebP, GIF) with quality param
- Output: write result to specified output path or return base64-encoded bytes in result
- Uses `image` crate (same as oh-my-pi)
- Wraps in `spawn_blocking` (image decode/encode is CPU-bound)

### ProfilingTool (`builtin/profiling.rs` exposed as tool)

Expose the always-on profiler as an agent-facing tool:
- **Schema:** `profile({ last_seconds? })` — returns work profile for the last N seconds
- Returns: `{ folded, summary, svg?, total_ms, sample_count }`
- Agent can self-diagnose which tools are slow and adjust strategy

### Feature Flag Wiring (`Cargo.toml`)

```toml
[features]
default = []
advanced-tools = ["dep:rayon", "dep:memmap2"]
ast-tools = ["dep:ast-grep-core"]
image-tools = ["dep:image", "dep:icy-sixel"]
pty = ["dep:portable-pty"]
profiling = ["dep:inferno"]
full = ["advanced-tools", "ast-tools", "image-tools", "pty", "profiling"]
```

**Registration:** `register_builtin_tools` registers base tools always (read, write, edit, glob, grep-basic, shell, process). Feature-gated tools registered conditionally:
```
#[cfg(feature = "ast-tools")]     register AstGrepTool
#[cfg(feature = "image-tools")]   register ImageTool
#[cfg(feature = "pty")]           register PtyTool
#[cfg(feature = "profiling")]     register ProfilingTool
```

### Workspace Cargo.toml updates

Add all new workspace dependencies:
- `grep-regex`, `grep-matcher`, `grep-searcher` (always)
- `dashmap`, `smallvec` (always)
- `rayon`, `memmap2` (optional)
- `ast-grep-core` (optional)
- `image`, `icy-sixel` (optional)
- `portable-pty` (optional)
- `inferno` (optional)
- `brush-core`, `brush-builtins` (vendored from oh-my-pi)
- `libc` (unix)

**Files:** `Cargo.toml` (workspace), `halter-tools/Cargo.toml`, `halter-tools/src/builtin/{image,profiling}.rs`, `halter-tools/src/builtin/mod.rs` (conditional registration), `halter-tools/src/lib.rs` (conditional exports)

**Verify:** `cargo build` (no features) compiles with base tools only. `cargo build --features full` compiles everything. `cargo test --all-features` passes. Image tool decodes/resizes/encodes roundtrip. Profiling tool returns valid folded stacks after tool executions. No unused dependency warnings in any feature combination.

---

## Definition of Done

- All 5 steps complete, tests passing under `cargo test` and `cargo test --all-features`
- No regressions on existing tests
- `register_builtin_tools` registers base tools (read, write, edit, glob, grep, shell, process) + feature-gated tools
- New tools exported from `halter-tools`: `EditTool`, `GrepTool`, `AstGrepTool`, `ImageTool`, `PtyTool`, `ProcessTool`, `ProfilingTool`
- File-level locking prevents data races between concurrent tool calls
- `halter-tools` compiles cleanly under: no features, `advanced-tools`, `ast-tools`, `image-tools`, `pty`, `profiling`, `full`
- `cargo clippy --all-features` clean
- Oh-my-pi's grep, shell, pty, ast, image, ps, prof, task, and text patterns are recognizably ported (same algorithms, same data structures, adapted interfaces)
