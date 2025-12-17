#!/usr/bin/env python3
"""
Release gate checks for tag pushes (vX.Y.Z).

Ensures:
- Tag version matches pyproject.toml version
- CHANGELOG.md contains a section for that version

Fails fast to prevent creating a mismatched GitHub Release.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict


def _read_pyproject_version(pyproject_path: Path) -> str:
    """
    Read version from pyproject.toml.

    Prefer TOML parsing (PEP 621: [project].version). Fall back to regex only if
    TOML parsing is unavailable for some reason.
    """
    text = pyproject_path.read_text(encoding="utf-8")

    # Python 3.11+ ships tomllib. CI runs 3.10+ but tomllib should exist on 3.11/3.12.
    try:
        import tomllib  # type: ignore

        data: Dict[str, Any] = tomllib.loads(text)
        project = data.get("project") or {}
        version = project.get("version")
        if isinstance(version, str) and version.strip():
            return version.strip()
    except Exception:
        pass

    # Fallback: accept both double and single quoted versions.
    m = re.search(r"(?m)^version\\s*=\\s*['\\\"]([^'\\\"]+)['\\\"]\\s*$", text)
    if not m:
        raise RuntimeError("Could not find version in pyproject.toml")
    return m.group(1).strip()


def _changelog_has_version(changelog: str, version: str) -> bool:
    # Match "## 1.2.3" or "## 1.2.3 - YYYY-MM-DD"
    pat = re.compile(rf"(?m)^##\s+{re.escape(version)}(\s+-.*)?$")
    return bool(pat.search(changelog))


def main() -> int:
    tag = os.environ.get("GITHUB_REF_NAME", "")
    if not tag.startswith("v"):
        print(f"Release gate skipped (not a v* tag): {tag}")
        return 0

    version = tag[1:]
    py_version = _read_pyproject_version(Path("pyproject.toml"))
    if py_version != version:
        raise SystemExit(
            f"Release gate failed: tag {tag} does not match pyproject.toml version {py_version}"
        )

    changelog = Path("CHANGELOG.md").read_text(encoding="utf-8")
    if not _changelog_has_version(changelog, version):
        raise SystemExit(f"Release gate failed: CHANGELOG.md missing section for {version}")

    print(f"OK: release gate passed for {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


