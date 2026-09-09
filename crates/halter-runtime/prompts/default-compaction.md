You are performing a CONTEXT CHECKPOINT COMPACTION. Summarize this session for your own
continuation in a fresh context window, where this summary will be the only record of the work
so far. Optimize for your ability to continue working, not for human readability.

<summary-format>
## End Objective
What the user ultimately asked for, with direct quotes for key requirements. If the goal evolved,
capture that progression.

## Progress and Key Decisions
What has been accomplished and the decisions that shaped it, with the reasoning that still
matters. Be specific: what was created, modified, or deleted; approaches that failed so they are
not retried.

## User Instructions and Input
High-priority instructions, corrections, and preferences the user stated. Reproduce
security-relevant ones verbatim: forbidden operations, sensitive files or data to avoid,
credential handling rules, and any "always/never" directive.

## Context and Constraints
Technical constraints, environment facts, and background discovered along the way that the
next window must not rediscover.

## What Remains
The remaining work in order, distinguishing what was explicitly requested from what was
implied. Do not invent steps beyond what the user asked for. Include exactly where in-progress
work left off, quoting verbatim where drift would be costly.

## Critical Data
Identifiers, paths, URLs, commands, values, error messages, and results you will need again.
Redact credentials.
</summary-format>

<preserve-rules>
Always preserve when present:
- Exact identifiers (IDs, paths, URLs, keys, names)
- Error messages verbatim
- User corrections and negative feedback
- Security-relevant instructions and constraints, verbatim, so they keep applying
- Specific values, formulas, or configurations
- The precise state of any in-progress work
</preserve-rules>

<compression-rules>
- Weight recent messages more heavily: the end of the transcript is the active context
- Omit pleasantries, acknowledgments, and filler
- Omit the system prompt and skills; they are re-injected separately
- Keep each section concise; if you must cut, preserve: security constraints > user corrections >
  errors > active work > completed work
</compression-rules>

Respond with the summary as plain text. Do not call tools.
