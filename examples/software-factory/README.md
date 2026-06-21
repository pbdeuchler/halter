# software factory example

`software-factory` is a runnable Halter SDK example that turns open GitHub issues
into an implementation pull request. It is intentionally write-capable: it can
create a branch, edit files, run commands, commit, push, and open a PR.

The example is useful when you want to see Halter used as an embedded Rust
runtime instead of as a CLI wrapper. It builds several model-specific harnesses,
adds a custom GitHub issue tool, streams canonical session events, persists
checkpoint state, and coordinates a multi-stage coding workflow.

> [!CAUTION]
> This is not a dry-run tool. Run it only in a repository where you are prepared
> for an agent to create commits and a pull request.

## what it does

The command runs this workflow:

1. Resolve the current git worktree and GitHub repository from a remote URL.
   With `--worktree`, create a detached git worktree under
   `/tmp/halter-software-factory/` and run the factory there.
2. Read project guidance from top-level `CLAUDE.md`, `AGENTS.md`, and `SOUL.md`
   files when they exist.
3. Fetch recent open GitHub issues, or fetch one issue when `--issue` is set.
4. Build issue candidates. With `--issue`, the requested issue becomes the
   candidate set. Without it, a model proposes up to three issue groups.
5. Ask a model-judge harness to select the smallest high-confidence unit of work.
6. Write an implementation plan to
   `.halter/software-factory/implementation-plan.md`.
7. Create a branch from the base branch.
8. Run an implementation agent.
9. Run a reviewer agent that inspects the branch changes and ask the
   implementation agent to repair findings until the review is clean or the
   iteration limit is hit.
10. Commit, push, draft a PR body, and open the PR.
11. With `--monitor`, poll the PR for maintainer review feedback and maintainer
    `/plsfix` comments, apply requested fixes, commit, and push again until the
    PR merges.

Only maintainer issue comments are included in the issue corpus. During PR
monitoring, only maintainer review activity and maintainer `/plsfix` comments
trigger follow-up work.

## prerequisites

Run the command inside the git repository you want it to change. The repository
must have a GitHub remote such as `git@github.com:owner/repo.git` or
`https://github.com/owner/repo.git`.

You also need:

- Rust 1.86 or newer.
- `git` and `cargo` on `PATH`.
- Git credentials that can push a branch to `origin`.
- GitHub API credentials that can read issues and create pull requests.
- Model provider credentials for the configured models.

For the default model choices, set:

```bash
export OPENROUTER_API_KEY=...
export OPENAI_API_KEY=...
```

For GitHub, set one of these:

```bash
export GITHUB_TOKEN=...
# or
export GH_TOKEN=...
```

If neither variable is set, the example tries `gh auth token`. Unauthenticated
GitHub requests may work for public read operations, but PR creation requires
credentials.

The example writes local state under `.halter/software-factory/`. Add `.halter/`
to the target repository's `.gitignore` unless you intentionally want this state
tracked. The `--commit-impl-plan` flag controls whether the implementation plan
is committed; checkpoint state is meant to stay local.

With `--worktree`, that local state is written in the temporary git worktree
under `/tmp/halter-software-factory/`, leaving the launch worktree's checkout and
`.halter/` state untouched.

## run it

From the Halter workspace, this targets the Halter repository itself:

```bash
cargo run -p halter-software-factory-example -- --issue 123
```

From another repository, point Cargo at this example's manifest while keeping the
current directory in the target repository:

```bash
cd /path/to/target/repo
cargo run --manifest-path /path/to/halter/examples/software-factory/Cargo.toml -- --issue 123
```

To let the workflow choose from recent open issues:

```bash
cargo run --manifest-path /path/to/halter/examples/software-factory/Cargo.toml
```

To keep watching the PR after it is opened:

```bash
cargo run --manifest-path /path/to/halter/examples/software-factory/Cargo.toml -- --issue 123 --monitor
```

To run the modifying stages in a dedicated git worktree under `/tmp`:

```bash
cargo run --manifest-path /path/to/halter/examples/software-factory/Cargo.toml -- --issue 123 --worktree
```

The command prints the created PR URL:

```text
created PR: https://github.com/owner/repo/pull/123
```

## options

Common options:

| option | behavior |
| --- | --- |
| `--issue <ISSUE>` | Work on one specific open issue. Without this, the workflow considers recent open issues. |
| `--remote <REMOTE>` | Git remote used to identify the GitHub repository. Defaults to `origin`. |
| `--base <BASE>` | Base branch for the PR. Defaults to the repository default branch from GitHub. |
| `--branch <BRANCH>` | Branch name to create. Defaults to `halter-factory/<repo>-<timestamp>-<title>`. |
| `--worktree` | Create a detached git worktree under `/tmp/halter-software-factory/` and run the factory there. |
| `--monitor` | Poll the opened PR for maintainer feedback and `/plsfix` comments until it merges. |
| `--allow-dirty` | Allow the run to start from a dirty worktree. Use this carefully. |
| `--commit-impl-plan` | Include `.halter/software-factory/implementation-plan.md` in commits. |
| `--resume` | Resume from `.halter/software-factory/checkpoint.json`. |
| `--reset-checkpoint` | Delete an existing checkpoint and start fresh. |
| `--max-review-iterations <N>` | Maximum implementation/review repair iterations. Defaults to `5`. |
| `--poll-seconds <N>` | PR monitor polling interval. Defaults to `60`. |

Model options use `provider/model` form, where `provider` is `openai`,
`anthropic`, or `openrouter`:

| option | default |
| --- | --- |
| `--glm-model` | `openrouter/z-ai/glm-5.2` |
| `--implementer-model` | `openrouter/moonshotai/kimi-k2.7-code` |
| `--reviewer-model` | `openai/gpt-5.5` |
| `--pr-model` | `openrouter/google/gemma-4-31b-it` |

The model-judge panel is configured in `default_factory_config()` in
`src/main.rs`.

## state and resume

The workflow checkpoints after each major stage:

- issue corpus loaded
- candidate set proposed
- judge selection and implementation plan written
- branch prepared
- implementation complete
- review loop complete
- commit complete
- push complete
- PR draft created
- PR opened

Checkpoint file:

```text
.halter/software-factory/checkpoint.json
```

A fresh run fails if this file already exists. Use `--resume` to continue the
same run, or `--reset-checkpoint` to start over.

Resume validates that the checkpoint matches the current repository, base
branch, requested issue, and `--commit-impl-plan` setting. If any of those
inputs change, start a new run with `--reset-checkpoint`.

The implementation plan is restored from the checkpoint on resume, so a resumed
run uses the same selected scope and plan.

For `--worktree` runs, resume from inside the temporary factory worktree that
contains the checkpoint. The path is logged when the run starts. Running
`--worktree --resume` from the original launch checkout fails because the
temporary worktree path is not recoverable from that checkout alone.

## project guidance

Before any model stage runs, the example reads these top-level files when they
exist:

- `CLAUDE.md`
- `AGENTS.md`
- `SOUL.md`

Each file is capped at 1 MiB. Non-empty guidance is appended to the system prompt
for planning, implementation, review, and PR drafting.

Repo-local Halter resources are also loaded from:

- `./.agent/skills`
- `./.agent/plugins`

Use those directories when you want this example to pick up repo-specific skills,
plugin agents, or hooks.

## implementation notes

The code is split by side-effect boundary:

- `src/core.rs` contains the functional core: parsing, formatting, selection
  validation, branch naming, dirty-status handling, and monitor classification.
- `src/main.rs` contains the imperative shell: CLI parsing, GitHub API calls,
  Halter harness construction, agent turns, git commands, checkpoint I/O, and PR
  monitoring.

The example builds separate harnesses for separate jobs:

- GLM: issue grouping and `/plsfix` refinement.
- Model judge: issue selection and implementation planning.
- Implementer: code changes and repair work.
- Reviewer: branch review.
- PR writer: PR title and body drafting.

The judge harness registers a custom `github_issue` tool. The tool lets the
judge fetch full issue text for issues already present in the current issue
corpus, while still keeping selection bounded to the issues the workflow loaded.

All agent turns consume `SessionEventPayload` events. The runner logs tool
starts, tool results, warnings, hook runs, compaction, usage, failures, and lagged
events instead of treating assistant text as the only output channel.

## verification while developing

Run the example's tests:

```bash
cargo test -p halter-software-factory-example
```

Check the CLI surface:

```bash
cargo run -p halter-software-factory-example -- --help
```

Set `RUST_LOG` for more detail:

```bash
RUST_LOG=debug cargo run -p halter-software-factory-example -- --issue 123
```

Halter traces are written under:

```text
~/.halter/traces/
```

## common failures

`factory checkpoint already exists`

Use `--resume` for the same run or `--reset-checkpoint` for a new run.

`worktree is dirty`

Commit, stash, or discard unrelated changes before running. If the only dirty
files are intentional, pass `--allow-dirty`.

`git remote must point at github.com`

The selected remote must be a GitHub URL. Use `--remote` if the GitHub remote is
not named `origin`.

GitHub API returns `401`, `403`, or `404`

Check `GITHUB_TOKEN`, `GH_TOKEN`, or `gh auth token`. The token must have access
to the repository and permission to create pull requests.

Model credential errors

Set the provider environment variables required by the selected models. The
defaults require OpenRouter and OpenAI credentials.
