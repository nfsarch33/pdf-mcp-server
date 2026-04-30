from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


def _server_tool_names() -> list[str]:
    """Return the live FastMCP tool names registered by ``pdf_mcp.server``.

    Pre-v1.3.0 this helper AST-parsed ``server.py`` looking for
    ``@mcp.tool()`` decorators. After the registry refactor (TICKET-05
    commit 2) ``server.py`` no longer carries decorators -- it loops
    over :func:`pdf_mcp.registry.iter_all` and calls ``mcp.tool()``
    imperatively. The runtime introspection below is cheap (server
    startup is a single pass) and gives a true contract check.
    """
    import pdf_mcp.server as server  # noqa: PLC0415  (intentional lazy import)

    return [tool.name for tool in server.mcp._tool_manager.list_tools()]


def test_registry_uniqueness(monkeypatch: pytest.MonkeyPatch) -> None:
    import pdf_mcp.registry as registry

    monkeypatch.setattr(registry, "_TOOLS", {})

    registry.register_tool(
        name="duplicate_name",
        verb="form",
        description="first",
        import_path="os:getcwd",
    )

    with pytest.raises(ValueError, match="duplicate_name"):
        registry.register_tool(
            name="duplicate_name",
            verb="form",
            description="second",
            import_path="os:getcwd",
        )


def test_registry_iter_all_stable_order(monkeypatch: pytest.MonkeyPatch) -> None:
    import pdf_mcp.registry as registry

    monkeypatch.setattr(registry, "_TOOLS", {})

    registry.register_tool(
        name="first_tool",
        verb="form",
        description="first",
        import_path="os:getcwd",
    )
    registry.register_tool(
        name="second_tool",
        verb="pages",
        description="second",
        import_path="os:getcwd",
    )
    registry.register_tool(
        name="third_tool",
        verb="text",
        description="third",
        import_path="os:getcwd",
    )

    assert [tool.name for tool in registry.iter_all()] == [
        "first_tool",
        "second_tool",
        "third_tool",
    ]


def test_registry_verb_groups_present() -> None:
    import pdf_mcp.registry as registry

    groups = {group.verb: group for group in registry.verb_groups()}

    for verb in ("form", "pages", "text", "extract", "sign"):
        assert verb in groups
        assert groups[verb].tools


def test_registry_lazy_no_pdf_tools_import() -> None:
    sys.modules.pop("pdf_mcp.registry", None)
    sys.modules.pop("pdf_mcp.pdf_tools", None)

    importlib.import_module("pdf_mcp.registry")

    assert "pdf_mcp.pdf_tools" not in sys.modules


def test_registry_get_unknown_raises() -> None:
    import pdf_mcp.registry as registry

    with pytest.raises(KeyError):
        registry.get("does_not_exist")


def test_lazy_callable_resolves() -> None:
    from pdf_mcp.registry import LazyCallable

    lazy = LazyCallable("os:getcwd")

    assert callable(lazy)
    assert lazy() == Path.cwd().as_posix()


def test_lazy_callable_caches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module_name = "lazy_callable_probe"
    probe_module = tmp_path / f"{module_name}.py"
    probe_module.write_text(
        "\n".join(
            [
                "IMPORT_COUNT = 0",
                "IMPORT_COUNT += 1",
                "CALL_COUNT = 0",
                "",
                "def probe(value):",
                "    global CALL_COUNT",
                "    CALL_COUNT += 1",
                "    return {",
                "        'value': value,",
                "        'import_count': IMPORT_COUNT,",
                "        'call_count': CALL_COUNT,",
                "    }",
                "",
            ]
        )
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(module_name, None)

    from pdf_mcp.registry import LazyCallable

    lazy = LazyCallable(f"{module_name}:probe")

    first = lazy("first")
    assert first == {"value": "first", "import_count": 1, "call_count": 1}

    with patch("importlib.import_module", wraps=importlib.import_module) as import_module:
        second = lazy("second")

    assert second == {"value": "second", "import_count": 1, "call_count": 2}
    assert import_module.call_count == 0


def test_registry_mirrors_server_tool_names_in_order() -> None:
    import pdf_mcp.registry as registry

    assert [tool.name for tool in registry.iter_all()] == _server_tool_names()


def test_registry_dataclasses_are_frozen() -> None:
    from dataclasses import FrozenInstanceError

    import pdf_mcp.registry as registry

    tool = next(iter(registry.iter_all()))
    group = registry.verb_groups()[0]

    with pytest.raises(FrozenInstanceError):
        tool.name = "mutated"  # type: ignore[misc]

    with pytest.raises(FrozenInstanceError):
        group.help = "mutated"  # type: ignore[misc]
