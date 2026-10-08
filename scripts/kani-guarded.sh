#!/usr/bin/env bash
# Run `cargo kani` with a hard memory ceiling and wall-clock timeout.
#
# CBMC has no memory limit of its own and macOS does not enforce
# `ulimit -v`, so a harness whose formula explodes will consume all RAM.
# This wrapper polls the resident memory of the whole kani process tree and
# kills it once the ceiling is crossed.
#
# Usage: scripts/kani-guarded.sh [cargo kani args...]
# Env:   KANI_MAX_RSS_MB  (default 8192)
#        KANI_TIMEOUT_S   (default 900)
set -euo pipefail

max_rss_mb="${KANI_MAX_RSS_MB:-8192}"
timeout_s="${KANI_TIMEOUT_S:-900}"

tree_pids() {
  local root="$1"
  local pids=("$root") frontier=("$root") next child
  while ((${#frontier[@]})); do
    next=()
    for pid in "${frontier[@]}"; do
      while read -r child; do
        [[ -n "$child" ]] && next+=("$child")
      done < <(pgrep -P "$pid" 2>/dev/null || true)
    done
    pids+=("${next[@]+"${next[@]}"}")
    frontier=("${next[@]+"${next[@]}"}")
  done
  echo "${pids[@]}"
}

kill_tree() {
  local pids
  pids="$(tree_pids "$1")"
  # shellcheck disable=SC2086
  kill -KILL $pids 2>/dev/null || true
}

cargo kani "$@" &
kani_pid=$!
trap 'kill_tree "$kani_pid"' INT TERM

start=$(date +%s)
peak_mb=0
while kill -0 "$kani_pid" 2>/dev/null; do
  pids="$(tree_pids "$kani_pid" | tr ' ' ',')"
  rss_kb=$(ps -o rss= -p "$pids" 2>/dev/null | awk '{s += $1} END {print s + 0}')
  rss_mb=$((rss_kb / 1024))
  ((rss_mb > peak_mb)) && peak_mb=$rss_mb
  if ((rss_mb > max_rss_mb)); then
    echo "kani-guarded: killed, process tree RSS ${rss_mb} MB exceeded ${max_rss_mb} MB" >&2
    kill_tree "$kani_pid"
    wait "$kani_pid" 2>/dev/null || true
    exit 137
  fi
  if (($(date +%s) - start > timeout_s)); then
    echo "kani-guarded: killed, exceeded ${timeout_s}s timeout" >&2
    kill_tree "$kani_pid"
    wait "$kani_pid" 2>/dev/null || true
    exit 124
  fi
  sleep 1
done

set +e
wait "$kani_pid"
status=$?
echo "kani-guarded: exit ${status}, peak RSS ${peak_mb} MB, $(($(date +%s) - start))s" >&2
exit "$status"
