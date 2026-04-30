"""TICKET-13 parity tests: server.py must mount its tool surface from the registry.

This is the canonical contract test for the single-source registry refactor
(TICKET-05). Both the MCP server (``pdf_mcp.server``) and the CLI
(``pdf_mcp.cli``) must derive their tool list, ordering, names, and
descriptions from :mod:`pdf_mcp.registry` and nowhere else.

These tests fail loudly if a future change re-introduces hand-written
``mcp.tool`` decorators in ``server.py``, drifts ordering, drops a
tool, or adds a tool without registering it via
:func:`pdf_mcp.registry.register_tool`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import pdf_mcp.registry as registry

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_PY = REPO_ROOT / "pdf_mcp" / "server.py"


# ---------------------------------------------------------------------------
# Source-level invariants
# ---------------------------------------------------------------------------


def _hand_written_mcp_tool_decorators(server_py: Path) -> list[str]:
    """AST-parse ``server.py`` and return names of functions decorated with ``mcp.tool``.

    Counting the literal string ``@mcp.tool()`` is unsafe because the
    refactored ``server.py`` mentions the historical decorator in its
    module docstring (for context). The AST walk only flags real
    decorator usage on top-level ``FunctionDef`` nodes.
    """
    tree = ast.parse(server_py.read_text())
    names: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        for deco in node.decorator_list:
            if isinstance(deco, ast.Call) and getattr(deco.func, "attr", None) == "tool":
                names.append(node.name)
                break
    return names


def test_server_uses_registry_driven_loop():
    """``server.py`` must NOT contain hand-written ``mcp.tool`` decorators.

    The registry-driven loop in :func:`pdf_mcp.server._register_all_tools`
    calls ``mcp.tool()(wrapped)`` imperatively inside a ``for`` loop.
    Any decorator-syntax usage at module scope means a tool has been
    re-introduced outside the registry contract.
    """
    bad = _hand_written_mcp_tool_decorators(SERVER_PY)
    assert not bad, (
        f"server.py contains {len(bad)} hand-written `mcp.tool` decorators "
        f"({bad[:3]}...); the registry-driven loop should use `mcp.tool()(wrapped)` "
        f"imperatively. All tool registration must flow through "
        f"`pdf_mcp.registry.register_tool()`."
    )


def test_server_imports_registry():
    """``server.py`` must import the registry module."""
    src = SERVER_PY.read_text()
    assert (
        "from .registry import" in src
        or "from pdf_mcp.registry import" in src
        or "import pdf_mcp.registry" in src
        or "from pdf_mcp import registry" in src
    ), (
        "server.py must import pdf_mcp.registry to drive its tool surface. "
        "Without this import, the registry contract is not enforced."
    )


def test_server_size_remains_bounded():
    """``server.py`` must stay small (~100 LOC max) after registry refactor.

    Pre-refactor server.py was 1175 LOC. The registry-driven version is
    ~95 LOC. A regression past 200 LOC almost certainly means hand-written
    tool definitions have been re-introduced outside the registry.
    """
    line_count = sum(1 for _ in SERVER_PY.read_text().splitlines())
    assert line_count <= 200, (
        f"server.py is {line_count} LOC; the registry-driven loop should fit "
        f"in <= 200 LOC. If you added tool wrappers, register them via "
        f"pdf_mcp.registry.register_tool() instead."
    )


# ---------------------------------------------------------------------------
# Runtime parity: server's MCP surface vs registry
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server_tools() -> list:
    """Return the FastMCP-registered tools list for ``pdf_mcp.server``."""
    import pdf_mcp.server as server  # noqa: PLC0415  (intentional lazy import)

    return list(server.mcp._tool_manager.list_tools())


def test_server_tool_count_matches_registry(server_tools):
    """Server must expose exactly as many tools as the registry registers."""
    registry_tools = list(registry.iter_all())
    assert len(server_tools) == len(registry_tools), (
        f"server has {len(server_tools)} tools but registry has "
        f"{len(registry_tools)} tools; registry contract violated."
    )


def test_server_tool_names_match_registry_in_order(server_tools):
    """Server tool ordering must match registry insertion order exactly.

    FastMCP preserves decoration order in ``ToolManager._tools``; the
    registry preserves :func:`register_tool` order in its
    ``OrderedDict``. After the registry-driven refactor these two
    sequences must be identical.
    """
    server_names = [t.name for t in server_tools]
    registry_names = [t.name for t in registry.iter_all()]
    assert server_names == registry_names, (
        "server tool names/order != registry tool names/order.\n"
        f"  In server but not registry: {set(server_names) - set(registry_names)}\n"
        f"  In registry but not server: {set(registry_names) - set(server_names)}\n"
        f"  First disagreement at index "
        f"{next((i for i, (a, b) in enumerate(zip(server_names, registry_names)) if a != b), 'n/a')}"
    )


def test_server_tool_descriptions_match_registry(server_tools):
    """Server tool descriptions (visible on tools/list) must match registry.

    The registry holds the canonical MCP-facing description string. The
    server-side wrapper sets ``wrapped.__doc__ = tool.description`` so
    FastMCP's tools/list payload is byte-identical to the pre-refactor
    surface (which used hand-written docstrings on each wrapper fn).
    """
    by_name = {t.name: t for t in server_tools}
    mismatches: list[tuple[str, str, str]] = []
    for tool in registry.iter_all():
        srv = by_name.get(tool.name)
        if srv is None:
            mismatches.append((tool.name, "<missing on server>", tool.description))
            continue
        srv_desc = (srv.description or "").strip()
        reg_desc = (tool.description or "").strip()
        if srv_desc != reg_desc:
            mismatches.append((tool.name, srv_desc, reg_desc))

    assert not mismatches, (
        f"{len(mismatches)} tool descriptions differ between server and registry. First 3 mismatches: {mismatches[:3]}"
    )


def test_no_tool_lost_in_refactor(server_tools):
    """Sentinel: every tool that was on the v1.2.x server.py surface must still be exposed.

    This is the regression net: even if the registry is later trimmed
    deliberately, the maintainer must touch this list explicitly so a
    silent tool drop cannot ship.
    """
    expected_tools = {
        # form (9)
        "get_pdf_form_fields",
        "fill_pdf_form",
        "fill_pdf_form_any",
        "create_pdf_form",
        "get_form_templates",
        "create_pdf_form_from_template",
        "flatten_pdf",
        "clear_pdf_form_fields",
        "encrypt_pdf",
        # pages
        "merge_pdfs",
        "extract_pages",
        "rotate_pages",
        "reorder_pages",
        "insert_pages",
        "remove_pages",
        "split_pdf",
        # ai
        "auto_fill_pdf_form",
        "extract_structured_data",
        "analyze_pdf_content",
    }
    server_names = {t.name for t in server_tools}
    missing = expected_tools - server_names
    assert not missing, f"Sentinel tools missing from server surface: {sorted(missing)}"
