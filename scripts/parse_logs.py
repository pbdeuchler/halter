#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# ///
"""Parse halter log files for cache usage and performance metrics.

Extracts data from two log formats:
  - Structured tracing lines (ANSI-colored, key=value fields)
  - JSON streaming event lines (session event payloads)

Reports per-session and aggregate statistics for:
  - Token usage (input, output, cache creation, cache read)
  - Cache hit rates
  - API request latencies (time between request start and materialized response)
  - Rate limit wait time
  - Tool call counts
"""

import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# Strip ANSI escape sequences
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

# Structured log line: timestamp, level, target, message, then key=value pairs
# Example after stripping ANSI:
#   2026-04-14T23:16:11.123241Z DEBUG halter_runtime::session: materialized assistant message session_id=... input_tokens=3039 output_tokens=182
TRACING_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T[\d:.]+Z)\s+"
    r"(?P<level>\w+)\s+"
    r"(?P<target>[\w:]+):\s+"
    r"(?P<msg>.+)$"
)

KV_RE = re.compile(r"(\w+)=(\S+)")


def parse_ts(s: str) -> datetime:
    # Handle fractional seconds of varying precision
    s = s.rstrip("Z")
    if "." in s:
        base, frac = s.split(".")
        frac = frac[:6].ljust(6, "0")
        s = f"{base}.{frac}"
    return datetime.fromisoformat(s)


@dataclass
class RequestSpan:
    session_id: str
    start_ts: datetime
    model: str
    message_count: int


@dataclass
class SessionStats:
    session_id: str
    model: str = ""
    is_subagent: bool = False
    parent_session_id: str | None = None

    # Accumulated from JSON usage objects (authoritative, has cache fields)
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0

    # Counts
    api_requests: int = 0
    tool_calls: int = 0
    rate_limit_waits: int = 0
    rate_limit_total_ms: int = 0
    errors: int = 0

    # Latencies (request start -> materialized response)
    latencies_ms: list[float] = field(default_factory=list)


@dataclass
class ParseResult:
    sessions: dict[str, SessionStats]
    first_ts: datetime | None
    last_ts: datetime | None


def parse_file(path: str) -> ParseResult:
    sessions: dict[str, SessionStats] = {}
    inflight: dict[str, RequestSpan] = {}
    seen_usage_message_ids: set[str] = set()
    first_ts: datetime | None = None
    last_ts: datetime | None = None

    with open(path) as f:
        for raw_line in f:
            line = ANSI_RE.sub("", raw_line.strip())
            if not line:
                continue

            # Try structured tracing line first
            m = TRACING_RE.match(line)
            if m:
                ts = parse_ts(m.group("ts"))
                if first_ts is None:
                    first_ts = ts
                last_ts = ts
                msg = m.group("msg")
                kvs = dict(KV_RE.findall(msg))

                # --- Session creation ---
                if "creating session" in msg:
                    sid = kvs.get("session_id") or kvs.get("session")
                    if not sid:
                        continue
                    parent = kvs.get("parent_session_id")
                    if parent == "None":
                        parent = None
                    depth = int(kvs.get("subagent_depth", "0"))
                    if sid not in sessions:
                        sessions[sid] = SessionStats(
                            session_id=sid,
                            is_subagent=parent is not None or depth > 0,
                            parent_session_id=parent,
                        )

                # Also catch "created session blueprint" for session_id
                elif "created session blueprint" in msg:
                    sid = kvs.get("session_id")
                    if sid and sid not in sessions:
                        sessions[sid] = SessionStats(session_id=sid)
                    if sid:
                        sessions[sid].model = kvs.get("default_model", "")

                # --- Subagent detection via "started subagent turn" ---
                elif "started subagent turn" in msg:
                    sid = kvs.get("session_id")
                    if sid and sid in sessions:
                        sessions[sid].is_subagent = True

                # --- API request start ---
                elif "starting responses request" in msg:
                    sid = kvs.get("session_id")
                    model = kvs.get("model", "")
                    mc = int(kvs.get("message_count", "0"))
                    if sid:
                        if sid not in sessions:
                            sessions[sid] = SessionStats(session_id=sid, model=model)
                        inflight[sid] = RequestSpan(
                            session_id=sid,
                            start_ts=ts,
                            model=model,
                            message_count=mc,
                        )

                # --- Materialized response (structured log) ---
                elif "materialized assistant message" in msg:
                    sid = kvs.get("session_id")
                    if sid:
                        if sid not in sessions:
                            sessions[sid] = SessionStats(session_id=sid)
                        s = sessions[sid]
                        s.api_requests += 1
                        # Calculate latency if we have an inflight span
                        if sid in inflight:
                            span = inflight.pop(sid)
                            lat = (ts - span.start_ts).total_seconds() * 1000
                            s.latencies_ms.append(lat)

                # --- Tool call execution ---
                elif "executing tool call" in msg:
                    sid = kvs.get("session_id")
                    if sid:
                        if sid not in sessions:
                            sessions[sid] = SessionStats(session_id=sid)
                        sessions[sid].tool_calls += 1

                # --- Errors ---
                elif "turn failed" in msg or "provider stream failed" in msg:
                    sid = kvs.get("session_id")
                    if sid:
                        if sid not in sessions:
                            sessions[sid] = SessionStats(session_id=sid)
                        sessions[sid].errors += 1

                # --- Rate limit waits ---
                elif "waiting for openai rate limit" in msg:
                    model = kvs.get("model", "").strip('"')
                    delay = int(kvs.get("delay_ms", "0"))
                    # Attribute to whichever session was most recently in-flight for this model,
                    # or just accumulate globally. We'll use a simple heuristic: find the session
                    # with the most recent inflight request.
                    target_sid = None
                    latest_ts = None
                    for sid, span in inflight.items():
                        if span.model == model or f'"{span.model}"' == model:
                            if latest_ts is None or span.start_ts > latest_ts:
                                target_sid = sid
                                latest_ts = span.start_ts
                    if target_sid is None and sessions:
                        # Fallback: attribute to first session
                        target_sid = next(iter(sessions))
                    if target_sid:
                        if target_sid not in sessions:
                            sessions[target_sid] = SessionStats(session_id=target_sid)
                        sessions[target_sid].rate_limit_waits += 1
                        sessions[target_sid].rate_limit_total_ms += delay

                continue

            # Try JSON line
            if line.startswith("{"):
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                sid = obj.get("session_id")
                if not sid:
                    continue
                if sid not in sessions:
                    sessions[sid] = SessionStats(session_id=sid)

                payload = obj.get("payload", {})
                kind = payload.get("kind")

                if kind == "message_item":
                    message = payload.get("message", {})
                    msg_id = message.get("id", "")
                    usage = message.get("usage", {})
                    if usage and msg_id not in seen_usage_message_ids:
                        seen_usage_message_ids.add(msg_id)
                        s = sessions[sid]
                        s.input_tokens += usage.get("input_tokens", 0)
                        s.output_tokens += usage.get("output_tokens", 0)
                        s.cache_creation_tokens += usage.get("cache_creation_input_tokens", 0)
                        s.cache_read_tokens += usage.get("cache_read_input_tokens", 0)
                        replay = message.get("replay_meta", {})
                        if replay.get("model"):
                            s.model = replay["model"]

    return ParseResult(sessions=sessions, first_ts=first_ts, last_ts=last_ts)


def fmt_tokens(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def fmt_ms(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.2f}s"
    return f"{ms:.0f}ms"


def fmt_duration(td: timedelta) -> str:
    total = int(td.total_seconds())
    m, s = divmod(total, 60)
    if m > 0:
        return f"{m}m{s}s"
    return f"{s}s"


def report(result: ParseResult, path: str) -> None:
    sessions = result.sessions
    if not sessions:
        print("No session data found.")
        return

    print(f"=== Halter Log Analysis: {path} ===\n")

    if result.first_ts and result.last_ts:
        wall = result.last_ts - result.first_ts
        print(f"  Wall clock:  {fmt_duration(wall)}")
        print(f"  Start:       {result.first_ts.isoformat()}Z")
        print(f"  End:         {result.last_ts.isoformat()}Z")
        print()

    # Separate root vs subagent sessions
    root = {k: v for k, v in sessions.items() if not v.is_subagent}
    subs = {k: v for k, v in sessions.items() if v.is_subagent}

    all_sessions = list(sessions.values())

    # --- Aggregate token usage ---
    total_input = sum(s.input_tokens for s in all_sessions)
    total_output = sum(s.output_tokens for s in all_sessions)
    total_cache_create = sum(s.cache_creation_tokens for s in all_sessions)
    total_cache_read = sum(s.cache_read_tokens for s in all_sessions)
    total_requests = sum(s.api_requests for s in all_sessions)
    total_tools = sum(s.tool_calls for s in all_sessions)
    total_rl_waits = sum(s.rate_limit_waits for s in all_sessions)
    total_rl_ms = sum(s.rate_limit_total_ms for s in all_sessions)

    # Cache hit rate: cache_read / (cache_read + cache_create + non-cached input)
    # Non-cached input = input_tokens - cache_read_tokens
    # So denominator = input_tokens + cache_create
    total_cacheable = total_input + total_cache_create
    cache_hit_pct = (total_cache_read / total_cacheable * 100) if total_cacheable > 0 else 0.0

    print("--- Token Usage (aggregate) ---")
    print(f"  Input tokens:          {fmt_tokens(total_input):>10}  (total sent to model)")
    print(f"  Output tokens:         {fmt_tokens(total_output):>10}  (total generated)")
    print(f"  Cache creation:        {fmt_tokens(total_cache_create):>10}  (tokens written to cache)")
    print(f"  Cache read:            {fmt_tokens(total_cache_read):>10}  (tokens served from cache)")
    print(f"  Cache hit rate:        {cache_hit_pct:>9.2f}%")
    print(f"  Total tokens:          {fmt_tokens(total_input + total_output):>10}")
    print()

    # --- API performance ---
    all_latencies = []
    for s in all_sessions:
        all_latencies.extend(s.latencies_ms)

    print("--- API Performance ---")
    print(f"  API requests:          {total_requests:>10}")
    if all_latencies:
        sorted_lat = sorted(all_latencies)
        avg_lat = sum(sorted_lat) / len(sorted_lat)
        p50 = sorted_lat[len(sorted_lat) // 2]
        p90 = sorted_lat[int(len(sorted_lat) * 0.9)]
        p99 = sorted_lat[int(len(sorted_lat) * 0.99)]
        print(f"  Avg latency:           {fmt_ms(avg_lat):>10}")
        print(f"  p50 latency:           {fmt_ms(p50):>10}")
        print(f"  p90 latency:           {fmt_ms(p90):>10}")
        print(f"  p99 latency:           {fmt_ms(p99):>10}")
        print(f"  Min latency:           {fmt_ms(sorted_lat[0]):>10}")
        print(f"  Max latency:           {fmt_ms(sorted_lat[-1]):>10}")
    print(f"  Tool calls:            {total_tools:>10}")
    print()

    # --- Rate limiting ---
    print("--- Rate Limiting ---")
    print(f"  Rate limit waits:      {total_rl_waits:>10}")
    if total_rl_ms > 0:
        print(f"  Total wait time:       {fmt_duration(timedelta(milliseconds=total_rl_ms)):>10}")
        print(f"  Avg wait per event:    {fmt_ms(total_rl_ms / total_rl_waits):>10}")
    print()

    # --- Per-session breakdown ---
    print(f"--- Sessions ({len(sessions)} total, {len(root)} root, {len(subs)} subagent) ---")
    for sid, s in sorted(sessions.items(), key=lambda x: x[1].input_tokens, reverse=True):
        label = "subagent" if s.is_subagent else "root"
        model = s.model or "unknown"
        cacheable = s.input_tokens + s.cache_creation_tokens
        hit_pct = (s.cache_read_tokens / cacheable * 100) if cacheable > 0 else 0.0
        print(f"\n  [{label}] {sid[:12]}... (model: {model})")
        print(f"    Input: {fmt_tokens(s.input_tokens):>8}  Output: {fmt_tokens(s.output_tokens):>8}  "
              f"Cache create: {fmt_tokens(s.cache_creation_tokens):>8}  Cache read: {fmt_tokens(s.cache_read_tokens):>8}  "
              f"Hit: {hit_pct:.2f}%")
        print(f"    Requests: {s.api_requests}  Tools: {s.tool_calls}  Errors: {s.errors}  "
              f"RL waits: {s.rate_limit_waits} ({fmt_duration(timedelta(milliseconds=s.rate_limit_total_ms))})")
        if s.latencies_ms:
            avg = sum(s.latencies_ms) / len(s.latencies_ms)
            print(f"    Latency avg: {fmt_ms(avg)}  "
                  f"min: {fmt_ms(min(s.latencies_ms))}  max: {fmt_ms(max(s.latencies_ms))}")

    # --- Token growth curve (input tokens per request, root session only) ---
    # Re-parse to get ordered data points
    if root:
        print("\n--- Input Token Growth (root session, per API request) ---")
        root_sid = next(iter(root))
        # Collect from JSON payloads in order
        token_points = _extract_token_growth(path, root_sid)
        if token_points:
            for i, (inp, out, cc, cr) in enumerate(token_points, 1):
                bar_len = min(inp // 2000, 60)
                bar = "#" * bar_len
                print(f"  req {i:>2}: {fmt_tokens(inp):>8} in / {fmt_tokens(out):>6} out  "
                      f"[cache: +{fmt_tokens(cc)} / hit {fmt_tokens(cr)}]  {bar}")
    print()


def _extract_token_growth(path: str, target_sid: str) -> list[tuple[int, int, int, int]]:
    """Re-scan file for ordered usage snapshots for a specific session."""
    points = []
    seen = set()
    with open(path) as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line.startswith("{"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("session_id") != target_sid:
                continue
            payload = obj.get("payload", {})
            if payload.get("kind") != "message_item":
                continue
            message = payload.get("message", {})
            if message.get("role") != "assistant":
                continue
            msg_id = message.get("id", "")
            if msg_id in seen:
                continue
            seen.add(msg_id)
            usage = message.get("usage", {})
            if usage:
                points.append((
                    usage.get("input_tokens", 0),
                    usage.get("output_tokens", 0),
                    usage.get("cache_creation_input_tokens", 0),
                    usage.get("cache_read_input_tokens", 0),
                ))
    return points


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <logfile> [logfile2 ...]", file=sys.stderr)
        sys.exit(1)

    for path in sys.argv[1:]:
        result = parse_file(path)
        report(result, path)


if __name__ == "__main__":
    main()
