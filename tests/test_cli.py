"""Unit tests for the pdf-mcp CLI entrypoint (v1.3.0+).

The CLI was added in v1.3.0 to ship pdf-mcp as a first-class command-line
program, not just an MCP server. The existing MCP behaviour is preserved
behind the ``serve`` subcommand for backwards compatibility with
``python -m pdf_mcp.server``.

Surface contracts pinned by these tests:
  * ``pdf_mcp.cli`` module is importable.
  * ``pdf_mcp.cli.app`` is a Typer app.
  * ``pdf-mcp --version`` prints the same string as ``pdf_mcp.__version__``.
  * ``pdf-mcp --help`` lists the ``serve`` subcommand.
  * ``pdf-mcp serve`` invokes ``pdf_mcp.server.mcp.run(transport="stdio")``.
  * ``pdf_mcp.cli.main()`` is callable (used as the ``[project.scripts]``
    entry point).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

import pdf_mcp


@pytest.fixture()
def cli_runner():
    """Return a Typer/Click CliRunner."""
    typer_testing = pytest.importorskip("typer.testing")
    return typer_testing.CliRunner()


def test_cli_module_importable() -> None:
    """``pdf_mcp.cli`` must be importable at the package level."""
    import pdf_mcp.cli  # noqa: F401


def test_cli_app_is_typer_app() -> None:
    """``pdf_mcp.cli.app`` must be a Typer app (not just a plain click.Group)."""
    typer = pytest.importorskip("typer")
    from pdf_mcp.cli import app

    assert isinstance(app, typer.Typer), f"pdf_mcp.cli.app must be typer.Typer, got {type(app).__name__}"


def test_cli_main_callable() -> None:
    """``main()`` is the ``[project.scripts]`` entry point and must be callable."""
    from pdf_mcp.cli import main

    assert callable(main)


def test_cli_version_flag_prints_version(cli_runner) -> None:
    """``pdf-mcp --version`` exits 0 and prints ``pdf_mcp.__version__``."""
    from pdf_mcp.cli import app

    result = cli_runner.invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert pdf_mcp.__version__ in result.output


def test_cli_help_lists_serve_subcommand(cli_runner) -> None:
    """``pdf-mcp --help`` must mention the ``serve`` subcommand."""
    from pdf_mcp.cli import app

    result = cli_runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "serve" in result.output


def test_cli_serve_invokes_mcp_run_stdio(cli_runner) -> None:
    """``pdf-mcp serve`` must call ``pdf_mcp.server.mcp.run(transport='stdio')``.

    We patch the FastMCP instance so the test does not actually start a
    long-running stdio loop. We expect a single call with the stdio transport.
    """
    from pdf_mcp.cli import app

    with patch("pdf_mcp.server.mcp.run") as mock_run:
        result = cli_runner.invoke(app, ["serve"])

    assert result.exit_code == 0, result.output
    assert mock_run.call_count == 1, f"expected exactly one mcp.run() call, got {mock_run.call_count}"
    _, kwargs = mock_run.call_args
    assert kwargs.get("transport") == "stdio", f"expected transport='stdio', got transport={kwargs.get('transport')!r}"
