"""Golden snapshot tests for the pdf-mcp CLI surface (TICKET-03, v1.3.0).

Catches accidental regressions in user-visible CLI output. To regenerate
all snapshots after an intentional change:

    PDF_MCP_UPDATE_SNAPSHOTS=1 pytest tests/test_cli_golden.py

See ``tests/golden_helpers.py`` for the harness contract and rationale.
"""

from __future__ import annotations

import pytest
from golden_helpers import assert_cli_output_matches


@pytest.fixture()
def cli_runner():
    typer_testing = pytest.importorskip("typer.testing")
    # ``mix_stderr=False`` would split stdout/stderr; keep them merged
    # since Typer's --help/--version both target stdout and combining
    # is fine for a verbatim snapshot.
    return typer_testing.CliRunner()


def test_cli_version_golden(cli_runner) -> None:
    """``pdf-mcp --version`` output must match tests/golden/version.txt."""
    from pdf_mcp.cli import app

    assert_cli_output_matches(cli_runner, app, ["--version"], "version")


def test_cli_help_golden(cli_runner) -> None:
    """``pdf-mcp --help`` output must match tests/golden/help.txt.

    Drift here is usually intentional (added flag, new subcommand) but
    must be reviewed and re-snapshotted explicitly via
    ``PDF_MCP_UPDATE_SNAPSHOTS=1``.
    """
    from pdf_mcp.cli import app

    assert_cli_output_matches(cli_runner, app, ["--help"], "help")


def test_cli_serve_help_golden(cli_runner) -> None:
    """``pdf-mcp serve --help`` output must match tests/golden/serve_help.txt."""
    from pdf_mcp.cli import app

    assert_cli_output_matches(cli_runner, app, ["serve", "--help"], "serve_help")
