"""Tests for the registry-driven CLI verb groups (TICKET-05 c3, v1.3.0).

The CLI must expose every registered tool under ``pdf-mcp <verb> <tool>``
with the verb taxonomy coming from :mod:`pdf_mcp.registry`. The mount is
driven by :func:`pdf_mcp.registry.verb_groups` so adding a tool is a
one-line edit in the registry, not a five-place duplication.

Hard contracts pinned here:

1.  Every verb in ``registry.verb_groups()`` mounts as a Typer subapp.
2.  Every tool in a verb group is reachable as
    ``pdf-mcp <verb> <tool-name-with-underscores>``.
3.  Tools accept a ``--json`` payload that is forwarded as kwargs.
4.  Successful invocations print the tool's return value as JSON to
    stdout; non-zero exit on tool exception with a short error line.
5.  ``pdf-mcp --help`` and ``pdf-mcp <verb> --help`` MUST NOT import
    :mod:`pdf_mcp.pdf_tools`. The whole point of the lazy-callable
    plumbing is that ``pdf-mcp --help`` stays sub-200 ms cold.
6.  ``pdf-mcp serve`` continues to invoke ``mcp.run(transport='stdio')``
    (see ``test_cli.py``); c3 must not break c1 / c2.

Why not auto-derive Click options from the Python signature?
    pdf_tools functions take heterogeneous shapes (positional paths,
    nested dicts of field values, optional booleans, callbacks). The
    cleanest portable contract for v1.3.0 is a single ``--json`` kwargs
    blob. A future ticket may layer hand-curated ergonomic flags on top
    for the most-used tools without breaking this contract.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import pytest


@pytest.fixture()
def cli_runner():
    typer_testing = pytest.importorskip("typer.testing")
    return typer_testing.CliRunner()


def _all_verb_names() -> tuple[str, ...]:
    """Snapshot of registry verb-group names. Re-exported for stability."""
    from pdf_mcp import registry

    return tuple(g.verb for g in registry.verb_groups())


def _all_tool_names_for(verb: str) -> tuple[str, ...]:
    from pdf_mcp import registry

    for g in registry.verb_groups():
        if g.verb == verb:
            return tuple(t.name for t in g.tools)
    raise KeyError(verb)


class TestRootHelpListsAllVerbs:
    """``pdf-mcp --help`` must list every registry verb group."""

    def test_root_help_lists_every_verb_group(self, cli_runner) -> None:
        from pdf_mcp.cli import app

        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0, result.output
        for verb in _all_verb_names():
            assert verb in result.output, f"verb {verb!r} not present in `pdf-mcp --help` output:\n{result.output}"

    def test_root_help_still_lists_serve(self, cli_runner) -> None:
        from pdf_mcp.cli import app

        result = cli_runner.invoke(app, ["--help"])
        assert result.exit_code == 0, result.output
        assert "serve" in result.output, "the existing `serve` subcommand must remain visible"


class TestVerbHelpListsTools:
    """``pdf-mcp <verb> --help`` must list every tool in that verb."""

    @pytest.mark.parametrize("verb", _all_verb_names())
    def test_verb_help_exits_zero(self, cli_runner, verb: str) -> None:
        from pdf_mcp.cli import app

        result = cli_runner.invoke(app, [verb, "--help"])
        assert result.exit_code == 0, f"`pdf-mcp {verb} --help` failed:\n{result.output}"

    @pytest.mark.parametrize("verb", _all_verb_names())
    def test_verb_help_lists_every_tool(self, cli_runner, verb: str) -> None:
        from pdf_mcp.cli import app

        result = cli_runner.invoke(app, [verb, "--help"])
        assert result.exit_code == 0, result.output
        for tool_name in _all_tool_names_for(verb):
            cli_form = tool_name.replace("_", "-")
            assert cli_form in result.output or tool_name in result.output, (
                f"tool {tool_name!r} (cli form {cli_form!r}) missing from `pdf-mcp {verb} --help`:\n{result.output}"
            )


class TestLazyImportPreserved:
    """The CLI plumbing must NOT import pdf_mcp.pdf_tools just to render help.

    This is the load-bearing invariant: ``pdf-mcp --help`` must stay
    sub-200 ms cold, which means the heavy dependency tree
    (pymupdf, pypdf, optional openai, etc.) MUST be deferred to the
    moment a tool is actually invoked.
    """

    def test_help_does_not_import_pdf_tools(self) -> None:
        """A fresh subprocess running `pdf-mcp --help` does not load pdf_tools."""
        # Spawn a fresh interpreter so we observe a clean import graph.
        # ``sys.modules`` checks inside the parent process are unreliable
        # because earlier tests may have imported pdf_tools already.
        script = textwrap.dedent(
            """
            import importlib, sys
            from typer.testing import CliRunner
            from pdf_mcp.cli import app

            assert 'pdf_mcp.pdf_tools' not in sys.modules, (
                'pdf_mcp.cli must not import pdf_tools at module load time'
            )

            runner = CliRunner()
            result = runner.invoke(app, ['--help'])
            assert result.exit_code == 0, result.output
            assert 'pdf_mcp.pdf_tools' not in sys.modules, (
                'pdf-mcp --help must not import pdf_tools'
            )
            print('OK')
            """
        )
        import subprocess

        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, f"subprocess failed:\nstdout:{proc.stdout}\nstderr:{proc.stderr}"
        assert proc.stdout.strip().endswith("OK"), proc.stdout

    def test_verb_help_does_not_import_pdf_tools(self) -> None:
        """`pdf-mcp form --help` also stays lazy."""
        script = textwrap.dedent(
            """
            import sys
            from typer.testing import CliRunner
            from pdf_mcp.cli import app

            runner = CliRunner()
            result = runner.invoke(app, ['form', '--help'])
            assert result.exit_code == 0, result.output
            assert 'pdf_mcp.pdf_tools' not in sys.modules, (
                'pdf-mcp form --help must not import pdf_tools'
            )
            print('OK')
            """
        )
        import subprocess

        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert proc.returncode == 0, f"subprocess failed:\nstdout:{proc.stdout}\nstderr:{proc.stderr}"


class TestToolInvocationViaJson:
    """``pdf-mcp <verb> <tool> --json '{...}'`` forwards kwargs to the tool."""

    def test_invokes_tool_with_json_kwargs(self, cli_runner, tmp_path) -> None:
        """A tool invocation deserialises --json and passes it as kwargs.

        We pick ``get_llm_backend_info`` (verb=ai) because it takes no
        required args and returns a dict — the simplest end-to-end shape.
        """
        from pdf_mcp.cli import app

        # No args needed for get_llm_backend_info.
        result = cli_runner.invoke(app, ["ai", "get-llm-backend-info", "--json", "{}"])
        assert result.exit_code == 0, result.output
        # Output must be valid JSON.
        try:
            payload = json.loads(result.output)
        except json.JSONDecodeError as e:
            pytest.fail(f"CLI did not produce valid JSON output: {e}\nraw:\n{result.output}")
        assert isinstance(payload, dict)
        # Backend info always reports at least a backends key.
        assert any(
            k in payload
            for k in ("backends", "available", "providers", "openai_available", "ollama_available", "vlm_available")
        )

    def test_tool_exception_yields_nonzero_exit(self, cli_runner, tmp_path) -> None:
        """If the resolved tool raises, the CLI exits non-zero with a stderr message."""
        from pdf_mcp.cli import app

        # get_pdf_metadata requires a real path; pass a non-existent one.
        bogus = tmp_path / "does-not-exist.pdf"
        result = cli_runner.invoke(
            app,
            ["metadata", "get-pdf-metadata", "--json", json.dumps({"pdf_path": str(bogus)})],
        )
        assert result.exit_code != 0, f"expected non-zero exit when tool raises, got 0:\n{result.output}"

    def test_invalid_json_payload_exits_nonzero(self, cli_runner) -> None:
        from pdf_mcp.cli import app

        result = cli_runner.invoke(
            app,
            ["ai", "get-llm-backend-info", "--json", "{not valid json"],
        )
        assert result.exit_code != 0, result.output


class TestJsonFileArgument:
    """``--json-file path/to/payload.json`` reads the kwargs from disk."""

    def test_json_file_loads_kwargs(self, cli_runner, tmp_path: Path) -> None:
        from pdf_mcp.cli import app

        payload_path = tmp_path / "payload.json"
        payload_path.write_text(json.dumps({}))
        result = cli_runner.invoke(app, ["ai", "get-llm-backend-info", "--json-file", str(payload_path)])
        assert result.exit_code == 0, result.output
        json.loads(result.output)


class TestPrettyPrint:
    """``--pretty`` indents the JSON output for human reading."""

    def test_pretty_indents_output(self, cli_runner) -> None:
        from pdf_mcp.cli import app

        result = cli_runner.invoke(
            app,
            ["ai", "get-llm-backend-info", "--json", "{}", "--pretty"],
        )
        assert result.exit_code == 0, result.output
        # A pretty-printed dict has a newline after the opening brace.
        assert "\n" in result.output


class TestOutputFlag:
    """``--output FILE`` writes JSON to the given path instead of stdout."""

    def test_output_flag_writes_to_file(self, cli_runner, tmp_path: Path) -> None:
        from pdf_mcp.cli import app

        out = tmp_path / "result.json"
        result = cli_runner.invoke(
            app,
            ["ai", "get-llm-backend-info", "--json", "{}", "--output", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert out.exists(), f"--output should have written {out}"
        # Must be valid JSON.
        json.loads(out.read_text())


class TestVerbGroupCount:
    """Sanity guard: regression check for accidental verb pruning."""

    def test_at_least_ten_verb_groups(self) -> None:
        """The v1.3.0 baseline ships >=10 verb groups (form, pages, ...)."""
        verbs = _all_verb_names()
        assert len(verbs) >= 10, (
            f"expected >=10 verb groups but got {len(verbs)}: {verbs}. "
            "Did someone delete a verb without updating this guard?"
        )
