"""Tests for ``scripts/generate_usage_doc.py``.

The generator turns ``pdf_mcp.registry.verb_groups()`` into ``USAGE.md``.
Two contracts:

1. ``USAGE.md`` MUST be in sync with the live registry. Drift fails CI.
2. ``--check`` MUST be a pure read (does not write) and must exit non-zero
   when the file diverges.

These tests run the script as an importable module so they are fast and
do not require a subprocess.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "generate_usage_doc.py"
USAGE_PATH = REPO_ROOT / "USAGE.md"


@pytest.fixture(scope="module")
def usage_doc_module():
    """Load ``scripts/generate_usage_doc.py`` as a module.

    The script is intentionally not under a Python package, so we use
    ``importlib.util.spec_from_file_location`` (same trick as the
    release-gate test) to load it.
    """
    spec = importlib.util.spec_from_file_location("_test_generate_usage_doc", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


class TestGeneratorOutput:
    """Pin the structural contract of the rendered ``USAGE.md``."""

    def test_output_starts_with_h1(self, usage_doc_module):
        from pdf_mcp import registry

        rendered = usage_doc_module._render(registry.verb_groups())
        assert rendered.startswith("# pdf-mcp CLI Usage Reference\n")

    def test_output_lists_all_verb_groups(self, usage_doc_module):
        from pdf_mcp import registry

        rendered = usage_doc_module._render(registry.verb_groups())
        for group in registry.verb_groups():
            assert f"## `pdf-mcp {group.verb}`" in rendered, (
                f"USAGE.md is missing the {group.verb!r} verb-group section"
            )

    def test_output_lists_all_tool_cli_names(self, usage_doc_module):
        from pdf_mcp import registry

        rendered = usage_doc_module._render(registry.verb_groups())
        for tool in registry.iter_all():
            cli_name = tool.name.replace("_", "-")
            assert f"`{cli_name}`" in rendered, f"USAGE.md is missing the {cli_name!r} CLI command"

    def test_output_includes_invocation_section(self, usage_doc_module):
        from pdf_mcp import registry

        rendered = usage_doc_module._render(registry.verb_groups())
        assert "## Invocation" in rendered
        assert "[--json '{...}']" in rendered
        assert "[--json-file PATH]" in rendered
        assert "[--pretty]" in rendered
        assert "[--output PATH]" in rendered

    def test_output_includes_backwards_compat_note(self, usage_doc_module):
        from pdf_mcp import registry

        rendered = usage_doc_module._render(registry.verb_groups())
        assert "## Backwards compatibility" in rendered
        assert "pdf-mcp serve" in rendered
        assert "python -m pdf_mcp.server" in rendered


class TestCheckMode:
    """``--check`` exits 0 when in sync, 1 when drifted, never writes."""

    def test_check_passes_when_usage_md_is_current(self, usage_doc_module, capsys):
        rc = usage_doc_module.main(["--check"])
        assert rc == 0, capsys.readouterr().err

    def test_check_fails_when_file_missing(self, usage_doc_module, tmp_path):
        target = tmp_path / "USAGE.md"
        rc = usage_doc_module.main(["--check", "--out", str(target)])
        assert rc == 1
        assert not target.exists()

    def test_check_fails_when_file_drifted(self, usage_doc_module, tmp_path):
        target = tmp_path / "USAGE.md"
        target.write_text("stale content\n", encoding="utf-8")
        rc = usage_doc_module.main(["--check", "--out", str(target)])
        assert rc == 1
        assert target.read_text(encoding="utf-8") == "stale content\n"

    def test_write_mode_creates_expected_file(self, usage_doc_module, tmp_path):
        target = tmp_path / "USAGE.md"
        rc = usage_doc_module.main(["--out", str(target)])
        assert rc == 0
        body = target.read_text(encoding="utf-8")
        assert body.startswith("# pdf-mcp CLI Usage Reference\n")


class TestUsageMdInSyncWithRegistry:
    """Belt-and-braces: USAGE.md on disk must match the rendered output."""

    def test_usage_md_matches_registry(self, usage_doc_module):
        from pdf_mcp import registry

        expected = usage_doc_module._render(registry.verb_groups())
        actual = USAGE_PATH.read_text(encoding="utf-8")
        assert actual == expected, (
            "USAGE.md is out of sync with pdf_mcp.registry. Run `python scripts/generate_usage_doc.py` and commit."
        )
