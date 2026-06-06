---
name: workflows-rust
description: Use when building scripted, deterministic multi-agent workflows in Rust on top of the Halter SDK. Applies to fan-out/fan-in (map-reduce), staged pipelines, adversarial verification, judge panels, loop-until-count / loop-until-dry / loop-until-budget loops, multi-modal sweeps, structured agent output, concurrency control, token budgeting, and resumable orchestration. Pairs with the basic-rust skill, which covers harness construction.
---

# Scripted Halter workflows

Use this skill to help users orchestrate **many agent steps from Rust** with the Halter SDK. The mental model is the inverse of model-driven delegation: the **Rust program is the orchestrator**, and every "agent" is one bounded session turn that the program drives, waits on, and aggregates. Loops, conditionals, fan-out, retries, and verification live in deterministic Rust — not in a model's own tool calls.

This skill assumes a built `Halter`. For config, providers, custom tools, hooks, policy, and persistence construction, use the **basic-rust** skill first.

## When to use scripted workflows

Pick the orchestration style deliberately:

- **Single turn** (`session.submit_turn`): one agent, one answer. Use basic-rust.
- **Model-driven subagents** (`spawn_agent`/`wait_agent` tools): let *the model* decide when to delegate. Use when the decomposition is unknown until runtime. Covered in basic-rust.
- **Scripted workflow** (this skill): *you* decide the decomposition in Rust. Use when the structure is known — N parallel readers, a fixed verify-then-synthesize pipeline, a bounded discovery loop. Deterministic control flow makes runs reproducible, testable, cache-friendly, and cheap to reason about.

Scripted workflows are the right tool for "be comprehensive," "review across these dimensions," "do X for each of these N items," and "keep looking until you stop finding things."

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

**Parse-and-repair (lightweight).** Ask for bare JSON, slice it defensively, deserialize, and optionally re-ask once with the parse error:

```rust
use serde::de::DeserializeOwned;

pub async fn run_agent_json<T: DeserializeOwned>(
    harness: &Halter,
    prompt: impl Into<String>,
) -> anyhow::Result<T> {
    let prompt = format!(
        "{}\n\nReturn ONLY a single JSON value matching the requested shape. \
         No prose, no markdown fences.",
        prompt.into()
    );
    let raw = run_agent(harness, prompt).await?;
    serde_json::from_str(json_slice(&raw))
        .map_err(|e| anyhow!("agent output was not valid JSON ({e}):\n{raw}"))
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
- Add per-step deadlines with `tokio::time::timeout(dur, run_agent(...))`; dropping the future cancels the turn.

## Sessions, lineage, and resume

- Each `run_agent` is an **isolated** session: no shared transcript, maximally cache-friendly and deterministic. This is the right default for fan-out.
- For multi-turn continuity inside one agent, keep the `SessionHandle` and call `submit_turn` again sequentially.
- To record parent/child lineage (and respect depth policy for any nested model-driven subagents), set `SessionInit { parent_session_id: Some(parent_id), subagent_depth: depth + 1, ..Default::default() }`.
- For **resumable** workflows, give steps deterministic `SessionInit.session_id`s and use the SQLite session store (basic-rust "Persistence"); re-running resumes committed turns instead of repeating them. Alternatively keep stages pure and idempotent so re-running is cheap.
- Set `runtime.traces_dir` to get a JSONL trace per root session for offline inspection of a workflow run.

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

## Cross-reference

Use **basic-rust** for: building `Halter`, config and credentials, custom tools (including the capture-result tool), hooks and policy, model/provider setup, session persistence, and the canonical event-payload reference. This skill layers deterministic multi-agent orchestration on top of that foundation.
