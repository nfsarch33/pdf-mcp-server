#!/usr/bin/env python3
"""CI guard for PR hygiene.

If code changes touch pdf-handler implementation, require CHANGELOG.md update unless label override.

This is intentionally lightweight (no external deps).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def main() -> int:
    base = os.environ.get("GITHUB_BASE_REF")
    sha = os.environ.get("GITHUB_SHA")
    labels = os.environ.get("PR_LABELS", "")

    if not base or not sha:
        print("PR hygiene check skipped (not a PR context)")
        return 0

    if "skip-changelog" in labels.split(","):
        print("PR hygiene: skip-changelog label present; skipping changelog enforcement")
        return 0

    # Fetch base branch for diff.
    _run(["git", "fetch", "--no-tags", "origin", base])
    changed = _run(["git", "diff", "--name-only", f"origin/{base}...{sha}"])
    files = [f for f in changed.splitlines() if f.strip()]

    # Only enforce when implementation changed.
    impl_touched = any(
        f.startswith("pdf_mcp/") or f.startswith("scripts/") or f.startswith("tests/")
        for f in files
    )
    if not impl_touched:
        print("PR hygiene: no implementation/test changes; no changelog required")
        return 0

    if "CHANGELOG.md" not in files:
        print("ERROR: Implementation/tests changed but CHANGELOG.md was not updated.")
        print("- Add an entry under 'Unreleased' OR apply label 'skip-changelog' if truly not user-facing.")
        print("Changed files:")
        for f in files:
            print(f"- {f}")
        return 2

    print("PR hygiene: CHANGELOG.md updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
