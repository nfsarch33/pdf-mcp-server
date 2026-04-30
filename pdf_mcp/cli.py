"""Command-line interface for pdf-mcp (v1.3.0+).

Promotes pdf-mcp from "MCP server only" to a first-class CLI program.
The original MCP stdio behaviour is preserved behind the ``serve``
subcommand, so existing integrations that expect ``python -m pdf_mcp.server``
keep working unchanged.

Wiring:
  * ``pyproject.toml`` declares ``[project.scripts] pdf-mcp = "pdf_mcp.cli:main"``.
  * ``main()`` is the entry point Setuptools generates; it just delegates to
    the Typer ``app`` so tests can drive ``app`` with ``CliRunner`` directly
    (see ``tests/test_cli.py``).

Future subcommands (form / pages / text / extract / annotate / sign / ai)
will be added in subsequent sprint tickets. This module intentionally
ships a thin skeleton first, gated by tests, before the verb tree expands.
"""

from __future__ import annotations

import typer

from pdf_mcp import __version__

app = typer.Typer(
    name="pdf-mcp",
    help="pdf-mcp - PDF tooling CLI and MCP server.",
    no_args_is_help=True,
    add_completion=False,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print pdf-mcp version and exit.",
    ),
) -> None:
    """pdf-mcp entry point.

    Use a subcommand (run ``pdf-mcp --help``) for the available verbs. The
    initial release ships ``serve`` for MCP stdio mode; further verbs are
    added under TICKET-04 onwards in the v1.3.0 sprint backlog.
    """


@app.command()
def serve() -> None:
    """Run pdf-mcp as an MCP server over stdio.

    Backwards-compatible alias for ``python -m pdf_mcp.server``. Hosts
    such as Cursor and Claude Desktop launch this subcommand and then
    speak MCP over the spawned process's stdin/stdout.
    """
    # Lazy import: server.py pulls in pdf_tools and FastMCP machinery, which
    # is heavy and unnecessary for ``--version`` / ``--help`` paths.
    from pdf_mcp.server import mcp

    mcp.run(transport="stdio")


def main() -> None:
    """Setuptools console-script entry point.

    Kept as a tiny shim so ``[project.scripts]`` resolves to a stable
    callable even if the underlying Typer app structure changes.
    """
    app()


if __name__ == "__main__":
    main()
