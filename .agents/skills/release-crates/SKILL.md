---
name: release-crates
description: Use when cutting a release of the Halter crates — bumping versions, writing the CHANGELOG entry, tagging, publishing to crates.io, and creating the GitHub release. Triggers on "cut a release", "bump versions", "release v0.X", "publish the crates", "ship a new version".
---

# Releasing the Halter crates

A release is: pick a version, bump every manifest, write the CHANGELOG entry, land it
on `master`, tag it, let CI publish, verify, then create the GitHub release.

**Pushing a `v*` tag is irreversible.** `.github/workflows/publish.yml` fires on
`push: tags: ["v*"]` and publishes to crates.io, where versions can be yanked but never
replaced. Do every reversible step first, verify locally, and get CI green on the release
commit before any tag exists.

## The release graph

Ten crates publish, in this dependency order — the list lives in `bin/crate-release-candidates`
and that file is the source of truth:

```
halter-brush-core       vendor/brush-core-vendored/Cargo.toml
halter-brush-builtins   vendor/brush-builtins-vendored/Cargo.toml
halter-protocol         crates/halter-protocol/Cargo.toml
halter-hooks            crates/halter-hooks/Cargo.toml
halter-config           crates/halter-config/Cargo.toml
halter-providers        crates/halter-providers/Cargo.toml
halter-session          crates/halter-session/Cargo.toml
halter-tools            crates/halter-tools/Cargo.toml
halter-runtime          crates/halter-runtime/Cargo.toml
halter                  crates/halter/Cargo.toml
```

Facts that shape every step:

- The **eight `crates/halter-*` crates plus `halter`** move as one semver-coupled unit. They
  always share a version number. `halter-cli` (`publish = false`) is bumped in lockstep so the
  workspace reads consistently; `examples/software-factory` (`publish = false`) is not versioned.
- The **two vendored brush crates** version independently. They only move when
  `vendor/` changes, and their numbers (currently `halter-brush-core` 0.5.x,
  `halter-brush-builtins` 0.2.x) have nothing to do with the halter line. Do not sweep them
  with a blanket find-and-replace, and do not assume they are unchanged — check.
- Versions are **literal strings**, not `version.workspace`. Each manifest has its own
  `version = "X.Y.Z"` under `[package]`, *and* inter-crate dependency lines carry
  `{ path = "...", version = "X.Y.Z" }`. Both must move. `crates/halter/Cargo.toml` alone has
  seven dependency lines to update.
- `Cargo.lock` is **gitignored**, so there is no lock churn to commit. `docs/` is gitignored too.

## 1. Establish the baseline

```bash
LAST=$(git describe --tags --match 'v*' --abbrev=0)
git log --oneline "$LAST"..HEAD
git diff --stat "$LAST"..HEAD -- crates/ vendor/ examples/
```

Read the commits, not just the stat. You are looking for the story the release notes have to
tell and for anything that breaks a downstream consumer.

Then let the release tooling tell you what would actually publish:

```bash
CRATE_RELEASE_BASE_REF="$LAST" bin/crate-release-candidates
```

Run this **before** bumping (to see what the last release left behind — the vendored crates in
particular can carry an unpublished bump from a prior merge) and **again after** bumping to
confirm the final publish set. Its stderr narrates `candidate:` / `unchanged:` per crate; its
stdout is the `name|version|manifest` rows the workflow consumes.

## 2. Choose the version

Pre-1.0, so under Cargo's semver rules the **minor** position is the breaking position. Bump
minor when anything in the diff breaks a downstream compile or silently changes a documented
value:

- a new public struct field (breaks struct literals)
- a new variant on a public enum that is not `#[non_exhaustive]` (breaks exhaustive matches)
- any change to a public function signature
- a trait method signature change (breaks external implementors, e.g. custom `SessionStore`)
- a semantic change to an existing public or persisted field

Patch is for genuinely additive, compile-compatible work. When it is close, take the minor —
the cost of an extra minor on a 0.x line is nil, and a wrong patch bump breaks builds silently.

To scan the public surface for breaks:

```bash
git diff "$LAST"..HEAD -- 'crates/**/*.rs' | grep -E '^[-+][[:space:]]*pub '
```

## 3. Bump the manifests

Rewrite `version = "<old>"` → `version = "<new>"` across the nine `crates/*/Cargo.toml` files,
covering both `[package]` and inter-crate dependency lines. Confirm the vendored manifests and
the `brush-*` entries in the root `[workspace.dependencies]` were not caught by the sweep unless
you intend to move them.

```bash
grep -rn 'version = "' crates/*/Cargo.toml
grep -n 'brush-' Cargo.toml
grep -n '^version' vendor/*/Cargo.toml
```

## 4. Write the release notes

`CHANGELOG.md` is the canonical write-up; the GitHub release body is derived from it. Add a
section directly under `## [Unreleased]`, following the shape of the `0.3.0` and `0.4.0` entries:

```markdown
## [X.Y.Z] - YYYY-MM-DD

One paragraph: what this release is, and *why this version position* — name the
specific break that forces a minor, or state that the work is additive.

Published crates: <list>. `halter-cli` also moves to `X.Y.Z` but remains
`publish = false`.

### Added / ### Changed / ### Fixed

- Prose entries. Say what a consumer sees, not what the commit touched.

### Upgrading from <prev-minor>

- One bullet per thing that breaks a build, with the mechanical fix.
```

The `Upgrading from` section is the part that earns its keep. Every break identified in step 2
gets a bullet with the concrete remedy ("`estimate_context_tokens` takes a trailing
`usage_anchor_floor: usize`; pass `SessionState::usage_anchor_floor`, or `0` for the previous
behavior").

Also update prose docs the release invalidates — the root `README.md` and any affected
`crates/*/README.md`. These carry no version strings, so the edits are content-only: new config
keys, new enum values, changed provider behavior.

## 5. Verify locally

```bash
cargo fmt --all -- --check
cargo clippy --workspace --all-features --all-targets -- -D warnings
cargo test --workspace --all-features
CRATE_RELEASE_BASE_REF="$LAST" bin/crate-release-candidates | cut -d'|' -f1,2
```

That last line is the publish set. Reconcile it against the CHANGELOG's "Published crates" list —
a mismatch means one of the two is wrong, and it is usually the changelog.

## 6. Land on master and get CI green

Commit as a dedicated release commit (`Bump crate versions for vX.Y.Z`), or fold the bump into
the feature commit when the release exists to ship that one change — both patterns are in the
history. Push to `master` and wait for CI:

```bash
gh run list --branch master --limit 5 \
  --json displayTitle,status,conclusion,headSha,workflowName \
  --jq '.[] | "\(.headSha[0:7])  \(.workflowName)  \(.status)/\(.conclusion // "-")  \(.displayTitle)"'
```

**Do not tag until CI is green on the exact release commit.** Publishing from a red master
ships a broken version permanently.

## 7. Tag — hand this to the user

The tag commands are gated by the permission classifier, correctly: pushing the tag is the
irreversible publish. Do not route around it. Give the user the two commands to run with the
`!` prefix:

```
! git tag -a vX.Y.Z -m "vX.Y.Z" -m "See CHANGELOG.md for release notes."
```
```
! git push origin vX.Y.Z
```

Two things that have bitten this repo before:

- **Use two `-m` flags, never an embedded newline.** A multi-line `-m "..."` string leaves zsh in
  an unmatched-quote continuation. Git joins repeated `-m` with a blank line, producing the same
  message the previous tags carry.
- **A repository ruleset restricts tag creation.** The push succeeds for the repo owner but
  prints `Bypassed rule violations for refs/tags/vX.Y.Z ... creations being restricted`. That
  line is expected, not a failure — check for `* [new tag]` in the same output.

## 8. Watch the publish

```bash
gh run list --workflow publish.yml --limit 3 \
  --json databaseId,status,conclusion,headBranch,event,createdAt \
  --jq '.[] | "\(.databaseId)  \(.status)/\(.conclusion // "-")  ref=\(.headBranch)  \(.createdAt)"'
gh run watch <run-id> --exit-status
```

The workflow runs `bin/crate-release-candidates` against the previous tag, then
`bin/publish-crate-release-candidates`, which publishes in dependency order and blocks on the
sparse index between crates (30 attempts × 10s) so each dependency is resolvable before its
dependents publish. It skips versions already present, so it is safe to re-run.

Typical duration is under 5 minutes. Then verify against crates.io directly rather than trusting
the workflow's exit code:

```bash
for spec in halter-protocol halter-hooks halter-config halter-providers \
            halter-session halter-tools halter-runtime halter; do
  curl -fsSL "https://index.crates.io/ha/lt/$spec" \
    | grep -q "\"vers\":\"X.Y.Z\"" && echo "  ok $spec X.Y.Z" || echo "  MISSING $spec X.Y.Z"
done
```

The index path is `{name[0:2]}/{name[2:4]}/{name}` for names of four or more characters — so all
`halter*` crates live under `ha/lt/`. Check the vendored crates at their own versions when they
are part of the release.

### If the publish fails partway

Crates already on the index stay there. Fix the cause, then re-run
`gh workflow run publish.yml` (or `-f publish_all=true` to sweep every listed crate whose current
version is missing from the index). The publisher's skip-if-published check makes re-runs
idempotent — do not bump versions to work around a partial publish.

## 9. Create the GitHub release

Only after crates.io is confirmed. Use `--verify-tag` so the command can never create a ref:

```bash
gh release create vX.Y.Z --verify-tag \
  --title "vX.Y.Z - <short theme>" \
  --notes-file <notes.md>
```

Body format, following `v0.1.2`:

```
Compared against <prev-tag> (<sha>).

Published crate bumps:
- <crate> <old> -> <new>
  ...

Changes:
- <consumer-facing bullet>
  ...
```

Keep it a condensation of the CHANGELOG section, not a second independent write-up. Link the
CHANGELOG for the full `Upgrading from` detail.

Tags `v0.2.0` through `v0.4.0` were published to crates.io but never got a GitHub release. If
you are asked to backfill, `gh release create <tag> --verify-tag` works on any existing tag and
does not re-trigger the publish workflow.

## Report

Close out with: version and previous version, the exact crates published and at what versions,
the run ID and link, index verification result, and the GitHub release URL. State plainly if any
crate failed or was skipped.
