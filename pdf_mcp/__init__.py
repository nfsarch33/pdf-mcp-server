"""
pdf-mcp: MCP server for PDF form filling, editing, and OCR text extraction.

Version is derived from pyproject.toml at build/install time via importlib.metadata.
This is the single source of truth -- bump version ONLY in pyproject.toml.
"""
from __future__ import annotations

try:
    from importlib.metadata import version

    __version__: str = version("pdf-mcp")
except Exception:
    # Fallback for editable installs or when metadata is unavailable.
    __version__ = "0.0.0-dev"
