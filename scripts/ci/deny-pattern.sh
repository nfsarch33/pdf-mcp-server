#!/usr/bin/env bash
# Fail when the tracked tree carries maintainer-private references: editor rule
# files, secret-manager paths, absolute home paths, private or tailnet addresses,
# and symlinks that point outside the repository.
#
# Usage: bash scripts/ci/deny-pattern.sh
# Extra patterns, one ERE per line, can be added locally with
# DENY_PATTERNS_FILE=<file>; the file is never committed.
# Exit: 0 clean, 1 at least one hit, 2 DENY_PATTERNS_FILE set but unreadable.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

patterns=(
  '[A-Za-z0-9_-]+\.mdc\b'
  'op://'
  'cursor-config/'
  '/home/[a-z][a-z0-9_-]*/'
  '\b10\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\b'
  '\b192\.168\.[0-9]{1,3}\.[0-9]{1,3}\b'
  '\b172\.(1[6-9]|2[0-9]|3[01])\.[0-9]{1,3}\.[0-9]{1,3}\b'
  '\b100\.(6[4-9]|[7-9][0-9]|1[01][0-9]|12[0-7])\.[0-9]{1,3}\.[0-9]{1,3}\b'
  '\.ts\.net\b'
)
if [[ -n "${DENY_PATTERNS_FILE:-}" ]]; then
  [[ -r "$DENY_PATTERNS_FILE" ]] || { echo "DENY_PATTERNS_FILE not readable" >&2; exit 2; }
  while IFS= read -r p; do [[ -n "$p" && "$p" != \#* ]] && patterns+=("$p"); done < "$DENY_PATTERNS_FILE"
fi

args=()
for p in "${patterns[@]}"; do args+=(-e "$p"); done
hits=0
if git grep -nIE "${args[@]}" -- . ':!scripts/ci/deny-pattern.sh' ':!uv.lock'; then
  hits=1
fi
links=$(git ls-files -s | awk '$1 == "120000" {print $4}' | while read -r f; do
  t=$(git cat-file -p ":$f"); [[ "$t" == /* || "$t" == ../* ]] && echo "$f -> $t"; done || true)
if [[ -n "$links" ]]; then echo "symlink out of repo: $links"; hits=1; fi

if (( hits )); then echo "deny-pattern: FAIL"; exit 1; fi
echo "deny-pattern: clean"
