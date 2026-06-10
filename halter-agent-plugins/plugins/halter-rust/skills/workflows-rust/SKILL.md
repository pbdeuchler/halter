---
name: workflows-rust
description: Use when building scripted, deterministic multi-agent workflows in Rust on top of the Halter SDK. Applies to fan-out/fan-in (map-reduce), staged pipelines, adversarial verification, judge panels, produce-review-repair loops, loop-until-count / loop-until-dry / loop-until-budget loops, multi-modal sweeps, structured agent output, side-effect isolation, concurrency control, token budgeting, progress reporting, and resumable orchestration. Pairs with the basic-rust skill, which covers harness construction.
---

# Scripted Halter workflows

Use this skill to help users orchestrate **many agent steps from Rust** with the Halter SDK. The mental model is the inverse of model-driven delegation: the **Rust program is the orchestrator**, and every "agent" is one bounded session turn that the program drives, waits on, and aggregates. Loops, conditionals, fan-out, retries, and verification live in deterministic Rust — not in a model's own tool calls.

This skill assumes a built `Halter`. For config, providers, custom tools, hooks, policy, and persistence construction, use the **basic-rust** skill first.

## When to use scripted workflows

Pick the orchestration style deliberately:

- **Single turn** (`session.submit_turn`): one agent, one answer. Use basic-rust.
- **Model-driven subagents** (`spawn_agent`/`wait_agent` tools): let *the model* decide when to delegate. Use when the decomposition is unknown until runtime. Covered in basic-rust.
- **Scripted workflow** (this skill): *you* decide the decomposition in Rust. Use when the structure is known — N parallel readers, a fixed verify-then-synthesize pipeline, a bounded discovery loop. Deterministic control flow makes runs reproducible, testable, cache-friendly, and cheap to reason about.

Scripted workflows are the right tool for "be comprehensive," "review across these dimensions," "do X for each of these N items," "transform each item and integrate the results," and "keep looking until you stop finding things."

## The atomic unit: one agent step

Every workflow is built from one primitive — run a fresh session for a single turn and drain its event stream to the final assistant text. Capture the **latest** assistant `MessageItem` and stop at `TurnCompleted`; always handle `TurnFailed`, or failures become silent hangs. Make this the canonical, metered helper and derive the simple one from it:

```rust
use anyhow::{anyhow, bail};
use futures::StreamExt;
use halter::prelude::*;
use halter_protocol::{AssistantPart, Usage};

/// Result of one agent step: its final text plus the turn's token usage.
pub struct AgentRun {
    pub text: String,
    pub usage: Usage,
}

/// Canonical agent step. A fresh session, one turn, drained to its final
/// assistant message. The orchestration logic lives in the Rust caller.
pub async fn run_agent_metered(
    harness: &Halter,
    init: SessionInit,
    prompt: impl Into<String>,
) -> anyhow::Result<AgentRun> {
    let session = harness.new_session(init).await?;
    let mut events = session.submit_turn(Turn::user(prompt)).await?;

    let mut text: Option<String> = None;
    let mut usage = Usage::default();
    while let Some(event) = events.next().await {
        match event?.payload {
            SessionEventPayload::MessageItem {
                message: Message::Assistant(msg),
            } => {
                text = Some(
                    msg.parts
                        .iter()
                        .filter_map(|p| match p {
                            AssistantPart::Text { text } => Some(text.to_string()),
                            _ => None,
                        })
                        .collect(),
                );
            }
            SessionEventPayload::TurnCompleted { usage: u, .. } => {
                usage = u;
                break;
            }
            SessionEventPayload::TurnFailed { error, .. } => bail!("agent turn failed: {error}"),
            _ => {}
        }
    }

    Ok(AgentRun {
        text: text.ok_or_else(|| anyhow!("agent produced no assistant text"))?,
        usage,
    })
}

/// The 90% case: just the text.
pub async fn run_agent(harness: &Halter, prompt: impl Into<String>) -> anyhow::Result<String> {
    run_agent_metered(harness, SessionInit::default(), prompt)
        .await
        .map(|run| run.text)
}
```

Notes that keep this correct:

- The handle (`SessionHandle`) is cheaply cloneable and `Send`; the stream is a `'static` boxed stream. Both move into `tokio::spawn` if you need detachment.
- A `SessionHandle` dropping at end of scope evicts its hooks via the runtime's eviction guard, so ephemeral fan-out sessions do not leak. Per-session `shutdown` is optional hygiene; the **harness** still needs a final `harness.shutdown(...)`.
- One session serializes its turns. For concurrent agents, use **one session per concurrent step** (as above). Reuse a single handle only for sequential multi-turn continuity.

## Structured agent output

Halter has no provider-forced output schema. Get typed data one of two ways.

**Parse-and-repair (lightweight).** Ask for bare JSON, slice it defensively, deserialize, and on failure re-ask with the parse error fed back so the model can self-correct. Bound the attempts, and surface the raw output on the final failure — it is the only way to debug a step that never parsed:

```rust
use serde::de::DeserializeOwned;

const JSON_ATTEMPTS: usize = 3;

pub async fn run_agent_json<T: DeserializeOwned>(
    harness: &Halter,
    prompt: impl Into<String>,
) -> anyhow::Result<T> {
    let base = format!(
        "{}\n\nReturn ONLY a single JSON value matching the requested shape. \
         No prose, no markdown fences.",
        prompt.into()
    );
    let mut ask = base.clone();
    let mut last: Option<(String, serde_json::Error)> = None;
    for _ in 0..JSON_ATTEMPTS {
        let raw = run_agent(harness, ask.clone()).await?;
        match serde_json::from_str(json_slice(&raw)) {
            Ok(value) => return Ok(value),
            Err(e) => {
                ask = format!("{base}\n\nYour previous reply did not parse: {e}\nReturn corrected JSON only.");
                last = Some((raw, e));
            }
        }
    }
    let (raw, e) = last.expect("loop runs at least once");
    Err(anyhow!("agent output was not valid JSON after {JSON_ATTEMPTS} attempts ({e}):\n{raw}"))
}

/// Tolerate accidental prose or ```json fences around the payload.
fn json_slice(raw: &str) -> &str {
    match (raw.find(['{', '[']), raw.rfind(['}', ']'])) {
        (Some(s), Some(e)) if e >= s => &raw[s..=e],
        _ => raw.trim(),
    }
}
```

**Capture tool (reliable).** Register a custom `submit_result` tool whose `execute` returns `ToolResult::Json` and stores the validated value into an `Arc<Mutex<Option<T>>>` (or a `oneshot`), and instruct the agent to call it exactly once. This validates at the tool boundary and survives chatty models. Build the tool with `HalterBuilder::with_tool(...)` per the basic-rust "Custom tools" section. Prefer this when malformed output is costly.

## Composition primitives

These four shapes cover almost everything. Map the Rust to the orchestration intent.

**Capped fan-out (default).** Bound concurrency yourself — scripted `new_session` fan-out is **not** limited by `max_concurrent_subagents` (that only bounds model-driven `spawn_agent`):

```rust
use futures::stream::{self, StreamExt};

const CONCURRENCY: usize = 8;

let results: Vec<anyhow::Result<String>> = stream::iter(prompts)
    .map(|p| run_agent(harness, p))
    .buffer_unordered(CONCURRENCY) // `.buffered(N)` instead to preserve input order
    .collect()
    .await;
```

**Barrier fan-out.** Use only when a later stage genuinely needs *all* prior results together (dedup, merge, early-exit on zero):

```rust
use futures::future::join_all;

let runs = join_all(prompts.iter().map(|p| run_agent(harness, p.clone()))).await;
let oks: Vec<String> = runs.into_iter().flatten().collect(); // drops the Err steps
```

**Per-item pipeline (prefer over a chain of barriers).** Each item flows through every stage independently, so item B's stage 1 runs while item A is in stage 3 — wall-clock is the slowest single chain, not the sum of per-stage maxima:

```rust
async fn refine(harness: &Halter, topic: String) -> anyhow::Result<String> {
    let draft = run_agent(harness, format!("Draft a section on {topic}.")).await?;
    let flaws = run_agent(harness, format!("List concrete flaws in this draft:\n{draft}")).await?;
    run_agent(
        harness,
        format!("Rewrite the draft, fixing every flaw.\n\nDRAFT:\n{draft}\n\nFLAWS:\n{flaws}"),
    )
    .await
}

let finals: Vec<anyhow::Result<String>> = stream::iter(topics)
    .map(|t| refine(harness, t))
    .buffer_unordered(CONCURRENCY)
    .collect()
    .await;
```

**Detached tasks.** For CPU-bound post-processing or true parallelism across cores, clone the harness into `tokio::spawn`:

```rust
let handles: Vec<_> = items
    .into_iter()
    .map(|item| {
        let h = harness.clone();
        tokio::spawn(async move { run_agent(&h, prompt_for(&item)).await })
    })
    .collect();
let results = futures::future::join_all(handles).await; // Vec<Result<Result<String>, JoinError>>
```

Guidance: default to the per-item pipeline + `buffer_unordered`. Reach for a `join_all` barrier only when the next step references "all the other results." A middle `flatten`/`map`/`filter` is not a reason for a barrier — do it inside a pipeline stage.

## Side effects and isolation

The fan-out primitives above assume each step is **independent**, and the safe default is read-only agents whose output the Rust orchestrator aggregates. The moment parallel steps *write* shared state — files, a scratch dir, a database, an external resource — they can clobber each other and the run stops being reproducible. Two moves keep side-effecting fan-out safe; use them only when a step actually mutates.

**Give each agent its own workspace, then merge in Rust.** Point each session at a private directory with `SessionInit { working_dir, .. }`; file tools resolve relative to it, so two agents editing "the same" path touch different copies. The orchestrator — not the agents — performs the merge: collect each workspace's artifact (a file, a diff, a JSON value) and apply or discard it deterministically.

```rust
// One isolated workspace per item. `working_dir` has no builder helper — set it in the struct literal.
let init = SessionInit { working_dir: workspace_dir.clone(), ..Default::default() };
let run = run_agent_metered(harness, init, prompt).await?;
// ... orchestrator inspects `workspace_dir` and decides whether to keep the result.
```

For **code**, the workspace is a throwaway git worktree: the orchestrator runs `git worktree add <dir> <ref>` itself (via `std::process::Command`, not the agent's shell), spawns the agent with that `working_dir`, captures `git -C <dir> diff`, then applies the diff to the main tree or drops the worktree. Use it for risky edits, broad refactors, or any step expected to return a diff. The isolated directory must sit inside `policy.allowed_write_roots`, or the write tools deny it.

**Or shard ownership and enforce it.** When a private workspace is overkill, give each parallel writer a disjoint slice of the namespace (file globs, key prefixes, partitions) and enforce the boundary at the tool layer with a `PreToolUse` hook that denies writes outside the slice — least privilege per step, not a prompt request. Grant `write`/`edit`/`shell` only to steps that mutate; readers keep `read`/`glob`/`grep`.

Steps that instead funnel into one shared resource cannot be isolated away — serialize them (see Concurrency, rate limits, and back-pressure).

## Workflow patterns

Compose the primitives into the standard quality patterns. Scale the fan-out and vote counts to the request: a quick check uses a few finders and single-vote verification; "thoroughly audit" uses a larger pool and 3–5 vote adversarial passes plus a synthesis stage.

**Adversarial verify (majority vote).** Spawn N skeptics prompted to *refute*; keep the claim only if a majority fail to:

```rust
async fn survives(harness: &Halter, claim: &str, voters: usize) -> anyhow::Result<bool> {
    let votes = futures::future::join_all((0..voters).map(|i| {
        let prompt = format!(
            "Skeptic #{i}: try hard to REFUTE the claim below. If you cannot clearly \
             refute it, or you are unsure, answer STANDS; otherwise answer REFUTED.\n\
             Claim: {claim}"
        );
        run_agent(harness, prompt)
    }))
    .await;
    let stands = votes
        .into_iter()
        .flatten()
        .filter(|v| v.to_ascii_uppercase().contains("STANDS"))
        .count();
    Ok(stands * 2 > voters)
}
```

For findings that can fail in several ways, give each verifier a distinct lens (correctness / security / does-it-reproduce) instead of N identical skeptics.

**Judge panel.** Generate N attempts from different framings (MVP-first, risk-first, user-first), score each with a judge agent (`run_agent_json::<Score>`), then synthesize from the winner while grafting the best ideas from runners-up. Beats one-attempt-iterated when the solution space is wide.

**Produce, review, repair.** Adversarial verify checks *claims you found*; this checks *artifacts you produced* — a draft, a config, a plan, a code edit. Generate the artifact, send it to two or more independent reviewers, apply their corrections, and optionally re-review. Accept only on consensus:

```rust
let accepted = reviews.len() >= 2 && reviews.iter().all(|r| r.accept);
```

Reviewers should reject more than surface errors: a result that misses the goal's root cause, takes an unsafe shortcut, drifts from the source of truth, reaches beyond its assigned scope, or papers over the problem instead of solving it (the classic tell is a comment that "explains" a hack). For code migrations this is the standard `survey → fix → two reviewers → apply corrections → re-review` loop; the same shape works for any produced artifact. Isolate the fix/produce step (see "Side effects and isolation") when it mutates state concurrently.

**Loop-until-count.** Accumulate to a target, stopping early when a round is empty:

```rust
#[derive(serde::Deserialize)]
struct Findings { items: Vec<String> }

let mut found: Vec<String> = Vec::new();
while found.len() < 10 {
    let batch: Findings = run_agent_json(harness, prompt_excluding(&found)).await?;
    if batch.items.is_empty() { break; }
    found.extend(batch.items);
}
```

**Loop-until-dry.** For unknown-size discovery, keep going until K consecutive rounds find nothing new. Dedup against everything *seen*, not just what survived verification, or it never converges:

```rust
use std::collections::HashSet;

let mut seen: HashSet<String> = HashSet::new();
let mut dry = 0;
while dry < 2 {
    let batch: Findings = run_agent_json(harness, prompt_excluding(&seen)).await?;
    let fresh: Vec<_> = batch.items.into_iter().filter(|x| seen.insert(x.clone())).collect();
    if fresh.is_empty() { dry += 1; continue; }
    dry = 0;
    // verify / record `fresh` ...
}
```

**Loop-until-budget.** Scale depth to a token ceiling using metered runs:

```rust
let ceiling_out_tokens = 500_000u64;
let mut spent = 0u64;
let mut all = Vec::new();
while spent < ceiling_out_tokens {
    let run = run_agent_metered(harness, SessionInit::default(), "Find more issues...").await?;
    spent += run.usage.output_tokens;
    all.push(run.text);
}
```

**Multi-modal sweep.** Fan out agents that each search a *different way* (by container, by content, by entity, by time); union the results. Each is blind to the others, so one angle's miss is covered by another.

**Completeness critic.** End a round with one agent asked "what is missing — a modality not run, a claim unverified, a source unread?" Feed its answer into the next round of work. Pairs naturally with loop-until-dry.

## Model roles and cost

Select the cheapest model per step that still works. Role ids (`default`, `small`, `subagent`) act as model ids; pass them per session or per turn:

```rust
// cheap, wide fan-out
run_agent_metered(harness, SessionInit::default().with_default_model("small"), prompt).await?;
// strong synthesis / judging
run_agent_metered(harness, SessionInit::default().with_default_model("default"), prompt).await?;
```

Drive wide discovery and verification with `small`/`subagent`; reserve `default` for synthesis, judging, and final write-ups. Meter with `AgentRun.usage` to keep budget loops honest.

## Concurrency, rate limits, and back-pressure

- Scripted fan-out is bounded only by your own `buffer_unordered`/`buffered` width or a `tokio::sync::Semaphore` — set it deliberately.
- Set `models.<role>.tokens_per_minute` so Halter throttles **proactively**. Without it, wide fan-out will trip provider 429s rather than queueing.
- `max_concurrent_subagents` and `max_subagent_depth` bound model-driven subagent tools, not your `new_session` calls. Do not rely on them to cap a scripted workflow.
- Concurrency is per-step, not global: discovery/produce/review steps fan out wide, but a step that funnels into one shared resource — a build, an index rebuild, a write to a single file, a rate-capped external API — must be **single-owner**, run at concurrency 1 (a `Semaphore` of one, or simply `await` it outside the fan-out). The canonical shape is a wide produce phase feeding a serialized integrate-or-verify gate.
- Add per-step deadlines with `tokio::time::timeout(dur, run_agent(...))`; dropping the future cancels the turn.

## Sessions, lineage, and resume

- Each `run_agent` is an **isolated** session: no shared transcript, maximally cache-friendly and deterministic. This is the right default for fan-out.
- For multi-turn continuity inside one agent, keep the `SessionHandle` and call `submit_turn` again sequentially.
- To record parent/child lineage (and respect depth policy for any nested model-driven subagents), set `SessionInit { parent_session_id: Some(parent_id), subagent_depth: depth + 1, ..Default::default() }`.
- For **resumable** workflows, give steps deterministic `SessionInit.session_id`s and use the SQLite session store (basic-rust "Persistence"); re-running resumes committed turns instead of repeating them. Alternatively keep stages pure and idempotent so re-running is cheap.
- Set `runtime.traces_dir` to get a JSONL trace per root session for offline inspection of a workflow run.

## Progress and rollup

A long run must be observable and auditable, not just correct.

- **Print progress as you go.** A scripted workflow is silent unless you say something. Emit a compact line per step start/finish with elapsed time (`eprintln!` or `tracing`), and optionally append a JSONL event (`{step, phase, ok, ms}`) for machine consumption. This is the difference between a run you can watch and one you can only wait on.
- **Persist each step's output, keyed by a stable label.** Write every step's prompt and parsed/raw result under a run directory (`runs/<id>/<label>.json`), where `label` is deterministic (e.g. `review:<item>`). Stable labels double as the cheapest resume key — on re-run, skip any label whose artifact already exists — and keep raw output even when parsing fails, so a bad step is debuggable after the fact. This is a filesystem-level resume that needs no session store; combine it with the deterministic `session_id` + SQLite approach above when you also need mid-turn resume.
- **Roll up usage.** `run_agent_metered` returns per-step `Usage`; sum `input_tokens`/`output_tokens` and the step count across the whole run into one final report so total cost is attributable rather than a surprise.

## Config shape for workflows

Workflow agents are usually focused, so keep each session's tool surface lean and define the cheap roles up front:

```toml
version = 1

[models.default]
provider = "openai"
model = "gpt-5"
reasoning = "high"
tokens_per_minute = 500_000

[models.small]
provider = "openai"
model = "gpt-5-mini"
reasoning = "low"
tokens_per_minute = 1_000_000

[models.subagent]
provider = "openai"
model = "gpt-5-mini"
reasoning = "medium"
tokens_per_minute = 750_000

[tools]
# Analysis/review fan-out often needs only reads; the Rust orchestrator aggregates.
enabled = ["read", "glob", "grep"]

[policy.shell]
enabled = false

[sessions]
backend = "memory" # use "sqlite" (with the sqlite feature) for resumable runs
```

Give individual agents `write`/`edit`/`shell` only when a step actually mutates state; the orchestrator holds the aggregate.

## Testing workflows

Make orchestration deterministic and offline:

- Wire the harness to `halter_providers::FakeProvider` (canned responses) and `InMemorySessionStore` — no network, reproducible.
- Pin concurrency to 1 in tests so interleaving is fixed; assert on the **aggregated** result, vote tallies, and iteration counts (not transcript wording).
- Test the loop terminators directly: empty-batch early exit, the dry counter, and the budget ceiling.
- Test `json_slice`/`run_agent_json` on fenced, prose-wrapped, and malformed payloads (Ok and Err paths).
- See basic-rust "Testing harnesses" for harness wiring and useful `cargo test -p ...` targets.

## Common failure modes

- **Unbounded fan-out → 429s.** Always cap with `buffer_unordered`/`Semaphore` and set `tokens_per_minute`.
- **Assuming `max_concurrent_subagents` caps scripted fan-out.** It does not; it governs model-driven subagent tools.
- **Sharing one session across concurrent turns.** A session serializes turns and may conflict on commit. One session per concurrent step.
- **Unhandled `TurnFailed`.** Without the match arm a failed turn yields no text and the loop hangs or mis-reports. Always handle it.
- **Loops that never terminate.** Every discovery loop needs a dry counter, count target, or budget/iteration ceiling.
- **Non-deterministic JSON.** Use `json_slice` plus a one-shot repair re-ask, or the capture-tool approach, before trusting structured output.
- **Dedup against the wrong set.** Loop-until-dry must dedup against everything *seen*, or judge-rejected items reappear each round.
- **Cost blowup.** Meter `AgentRun.usage`, pick cheap roles for breadth, and keep `default` for synthesis only.
- **Unisolated parallel writes.** Concurrent agents sharing one `working_dir` clobber each other and break reproducibility. Give each mutating step a private workspace, or shard ownership with an enforced write boundary.
- **Fanning out an exclusive step.** Builds, index rebuilds, single-file writes, and rate-capped APIs must be single-owner. Run them at concurrency 1, not inside the wide fan-out.
- **Silent long runs.** With no progress output you cannot tell a slow run from a hung one. Print a line per step and persist per-step artifacts.

## Cross-reference

Use **basic-rust** for: building `Halter`, config and credentials, custom tools (including the capture-result tool), hooks and policy, model/provider setup, session persistence, and the canonical event-payload reference. This skill layers deterministic multi-agent orchestration on top of that foundation.
