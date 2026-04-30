"""PDF MCP Server - Model Context Protocol server for PDF operations.

This module exposes PDF tools via the MCP protocol for use with AI assistants.
Run with: ``python -m pdf_mcp.server``.

Tool registration is **registry-driven** as of v1.3.0. Every tool exposed
on the MCP surface is sourced from :mod:`pdf_mcp.registry` (the single
source of truth shared with the ``pdf-mcp`` CLI). Adding a new tool is a
single :func:`pdf_mcp.registry.register_tool` call; no changes to this
module are required.

Available tool categories (see :data:`pdf_mcp.registry._VERB_HELP` for
the canonical list):

- form        : form discovery, filling, templates, flattening (9 tools)
- pages       : merge, split, extract, rotate, reorder, insert, remove (8 tools)
- text        : annotations, redaction, watermarks, comments, page numbers, Bates (~11 tools)
- extract     : text blocks, tables, images, links, structured data (~5 tools)
- sign        : digital and visual signatures (~5 tools)
- metadata    : metadata read/write/sanitisation, type detection (~3 tools)
- ocr         : OCR helpers and image inspection (~3 tools)
- ai          : LLM-backed extraction, auto-fill, and analysis (~4 tools)
- batch       : multi-file processing and comparisons (~1 tool)
- security    : encryption and PII detection (~2 tools)

Version: see ``pdf_mcp.__version__`` (single source: ``pyproject.toml``).
License: AGPL-3.0.
"""

from __future__ import annotations

import functools
import traceback
from pathlib import Path
from typing import Any, Callable

from mcp.server.fastmcp import FastMCP

from .pdf_tools import PdfToolError
from .registry import iter_all

mcp = FastMCP("PDF Handler")


def _wrap_result(result: Any) -> Any:
    """Coerce non-JSON-serialisable result types to MCP-friendly forms."""
    if isinstance(result, Path):
        return str(result)
    return result


def _handle_errors(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap a tool function with uniform PdfToolError -> error-dict handling."""

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return _wrap_result(fn(*args, **kwargs))
        except PdfToolError as exc:
            return {"error": str(exc)}
        except Exception as exc:  # pragma: no cover - defensive
            return {
                "error": f"Unexpected error: {exc}",
                "trace": traceback.format_exc(),
            }

    return wrapper


def _register_all_tools() -> None:
    """Register every tool from :mod:`pdf_mcp.registry` on the FastMCP surface.

    Replaces 57 hand-written ``@mcp.tool() / @_handle_errors`` decorator
    pairs with a single registry-driven loop. The wrapper preserves
    ``__name__`` and ``__annotations__`` via :func:`functools.wraps` so
    FastMCP's introspection-based JSON-Schema generation continues to
    work unchanged. The MCP-facing description is taken from the
    registry's curated ``description`` field, so the tools/list payload
    is byte-identical to the pre-refactor surface.
    """
    for tool in iter_all():
        fn = tool.callable.resolve()
        wrapped = _handle_errors(fn)
        # Override docstring with the curated MCP description; functools.wraps
        # copies ``fn.__doc__``, but the registry holds the canonical text.
        wrapped.__doc__ = tool.description
        mcp.tool()(wrapped)


_register_all_tools()


if __name__ == "__main__":
    mcp.run(transport="stdio")
