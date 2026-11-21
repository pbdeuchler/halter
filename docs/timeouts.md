# Timeouts and retries

This is the inventory of every timeout, retry and wait in halter, what
encloses what, and every way each wait can end. When you add or change a
wait, update this file in the same change and list each of its exits. A wait
that ends only on success or a deadline holds a stale answer open for the
whole timeout.

## What encloses what

```text
caller deadline = interrupt/shutdown deadline + FORCED_RECOVERY_GRACE (5s)
└─ graceful window (the deadline the caller passed)
   ├─ turn
   │  ├─ provider call (one retry sequence)
   │  │  ├─ rate-limit admission          no timer; outside the request timer
   │  │  ├─ attempt 1..max_attempts
   │  │  │  ├─ connect_secs               inside request_secs / compact_secs
   │  │  │  ├─ request_secs               stream setup through response headers
   │  │  │  └─ stream_idle_secs           per gap between stream items
   │  │  │     (compaction: compact_secs bounds the whole unary request)
   │  │  └─ backoff between attempts      total capped by deadline_secs
   │  ├─ tool call
   │  │  ├─ shell: timeout_ms or policy.shell.timeout_secs, then TERM_GRACE
   │  │  ├─ pty: timeout_ms, or none
   │  │  ├─ wait_agent / close_agent: timeout_ms, or until cancelled
   │  │  └─ model timeout_ms <= policy.max_tool_timeout_secs
   │  └─ hooks: per-handler timeout
   │     └─ prompt/agent hooks: max(timeout, provider recovery window)
   └─ SessionEnd hooks during shutdown: cancelled when the graceful window ends
```

Every enclosing timer must be at least as long as what it encloses, or the
inner timer and its retries never run. Validation rejects `connect_secs` that
is not less than `request_secs` and `compact_secs`. Prompt and agent hooks are
raised to `ResiliencePolicy::recovery_window()` (two attempts plus the first
backoff) so a hook timeout cannot pre-empt the provider's own retry.

## Provider requests

Configured under `[resilience]`, with per-provider overrides under
`[providers.<name>.resilience]`. Every value must be greater than zero.

| Wait | Default | Ends on |
| --- | --- | --- |
| `timeouts.connect_secs` | 10s | connected; elapsed (retryable network error); cancel |
| `timeouts.request_secs` | 60s | response headers; elapsed (retryable); cancel |
| `timeouts.stream_idle_secs` | 60s | next stream item; stream end; elapsed (retryable until the first content event); cancel |
| `timeouts.compact_secs` | 300s | full compaction response; elapsed (retryable); cancel |
| `request_retry.max_attempts` | 5 | includes the first attempt; a non-retryable error or the first delivered content event stops retrying |
| `request_retry.deadline_secs` | 60s | caps total backoff slept, not time inside attempts |
| `base_backoff_ms` / `max_backoff_secs` / `jitter_pct` | 500ms / 30s / 25% | `Retry-After` and "try again in" hints are honoured, capped at `max_backoff_secs` |
| Rate-limit admission (OpenAI header-driven) | none | permit granted; cancel. Paused out of `request_secs` |

A retry sequence lasts at most `max_attempts` attempts, each bounded by its
own timeouts, plus `deadline_secs` of backoff.

## Tools

| Wait | Default | Ends on |
| --- | --- | --- |
| `shell` command | `timeout_ms`, else `policy.shell.timeout_secs` (30s) | exit; elapsed (SIGTERM, then SIGKILL after 500ms); cancel |
| `pty` session | `timeout_ms`, else none | exit; elapsed; `kill` control; cancel |
| `wait_agent` | `timeout_ms`, else none | a target settles; elapsed (returns statuses); cancel; unknown target (immediate error) |
| `close_agent` | `timeout_ms`, else none | cleanup done; elapsed (force-closes, then waits up to `FORCED_RECOVERY_GRACE`); cancel |
| Model-supplied `timeout_ms` ceiling | `policy.max_tool_timeout_secs` (3600s) | a larger value is rejected before anything runs |
| `read` | `timeout_secs`, else 10s; not under the ceiling | read done; elapsed, checked between lines only, so a blocking open or read is not interrupted; size is capped by `max_read_bytes` |
| Browser navigation | 30s | loaded; elapsed |
| Browserbase connect / request | 10s / 60s | response; elapsed |
| Background job stop | 500ms TERM grace, 2s output drain | exit; elapsed (killed) |

## Hooks

| Wait | Default | Ends on |
| --- | --- | --- |
| Command and HTTP hooks | 600s (`timeout` / `timeout_sec`) | handler result; elapsed; cancel |
| Prompt and agent hooks | 60s, raised to the provider recovery window | handler result; elapsed; cancel |
| SessionEnd hooks during a bounded shutdown | as above | handler result; elapsed; the shutdown's graceful window ends |
| Subagent Stop hook resubmissions | cap of 8 | hook allows the stop; cap reached (error) |

Hooks fail open: a timed-out or failed hook allows the action.

## Session lifecycle

| Wait | Default | Ends on |
| --- | --- | --- |
| `interrupt(deadline)` / `shutdown(deadline)` graceful window | the caller's deadline, or none | turn or cleanup settles; deadline (force: cancel and abort) |
| Caller wait after forcing | `FORCED_RECOVERY_GRACE` (5s) | forced recovery completes; elapsed (`TimedOut`) |
| Forced interrupt store conflicts | 8 attempts, 10ms x attempt backoff | write succeeds; non-conflict error; cap reached (error) |
| Shutdown lease | none | lease acquired; cancel (a free lease wins over a fired cancel) |
| SQLite busy | 5s | lock acquired; elapsed |
| OpenTelemetry shutdown | 3s | flushed; elapsed |

## CLI login (`halter login openai`)

| Wait | Default | Ends on |
| --- | --- | --- |
| Whole login | 600s (`--timeout-secs`) | code exchanged; provider error; elapsed |
| One callback connection read | 5s | request read; elapsed (counts as a stray) |
| Stray callback connections | 16 | a valid callback; budget spent (error) |

A malformed request, a bad connection or a stale `state` counts as a stray. A
provider error or a missing code under the expected state ends the login.

## Model-judge panels

Panels run concurrently with no panel-level deadline. Each judge call is a
provider request bounded by the provider's resilience policy above.

## Zero and `None`

| Value | Meaning |
| --- | --- |
| `[resilience]` timeouts and retry counts of 0 | rejected by validation |
| `policy.shell.timeout_secs = 0`, `policy.max_tool_timeout_secs = 0` | rejected by validation |
| Hook `timeout = 0` | load warning `zero_timeout`; the handler default is used |
| `wait_agent` `timeout_ms: 0` | poll: return current statuses now |
| `close_agent` `timeout_ms: 0` | force-close now, then wait up to `FORCED_RECOVERY_GRACE` |
| `interrupt` / `shutdown` with `Some(Duration::ZERO)` | force now, then wait up to `FORCED_RECOVERY_GRACE` |
| `interrupt` / `shutdown` with `None`; tool `timeout_ms` omitted (except `shell`) | wait until done or cancelled |
