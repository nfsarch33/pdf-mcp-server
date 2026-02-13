"""
pdf-mcp: MCP server for PDF form filling, editing, and OCR text extraction.

Version is read directly from pyproject.toml (single source of truth) so that
editable installs always reflect the current checkout, not the stale pip
metadata from the last ``pip install`` invocation (see BUG-011).
"""
from __future__ import annotations

import re as _re
from pathlib import Path as _Path


def _get_version() -> str:
    """Read version from pyproject.toml (single source of truth).

    Fallback chain:
      1. Parse ``version = "X.Y.Z"`` from pyproject.toml in the repo root.
      2. importlib.metadata (works for non-editable pip installs).
      3. ``"0.0.0-dev"`` sentinel.
    """
    # 1. Read from pyproject.toml (always current in dev / editable installs)
    try:
        pyproject = _Path(__file__).resolve().parent.parent / "pyproject.toml"
        if pyproject.is_file():
            text = pyproject.read_text(encoding="utf-8")
            match = _re.search(r'^version\s*=\s*"([^"]+)"', text, _re.MULTILINE)
            if match:
                return match.group(1)
    except Exception:
        pass

    # 2. Installed package metadata (production / non-editable installs)
    try:
        from importlib.metadata import version

        return version("pdf-mcp")
    except Exception:
        pass

    return "0.0.0-dev"


__version__: str = _get_version()
