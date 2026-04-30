"""Command-line interface for pdf-mcp (v1.3.0+).

Promotes pdf-mcp from "MCP server only" to a first-class CLI program.
The original MCP stdio behaviour is preserved behind the ``serve``
subcommand, so existing integrations that expect ``python -m pdf_mcp.server``
keep working unchanged.

Wiring:
  * ``pyproject.toml`` declares ``[project.scripts] pdf-mcp = "pdf_mcp.cli:main"``.
  * ``main()`` is the entry point Setuptools generates; it just delegates to
    the Typer ``app`` so tests can drive ``app`` with ``CliRunner`` directly
    (see ``tests/test_cli.py`` and ``tests/test_cli_verb_groups.py``).
  * ``configure_logging()`` is called during the root callback so every
    subcommand inherits the same logging context (see TICKET-02).
  * Verb groups (``pdf-mcp form fill``, ``pdf-mcp pages merge``, ...) are
    mounted from :mod:`pdf_mcp.registry`. Adding a tool is a single
    ``register_tool(...)`` call; the CLI surface updates automatically
    (see TICKET-05 c3).

Lazy-import invariant
---------------------

Importing ``pdf_mcp.cli`` MUST NOT import ``pdf_mcp.pdf_tools``. The
verb-group plumbing wraps every tool in a closure that resolves the
:class:`~pdf_mcp.registry.LazyCallable` on first invocation, so
``pdf-mcp --help`` and ``pdf-mcp <verb> --help`` stay fast (sub-200 ms
cold). This is enforced by
:func:`tests.test_cli_verb_groups.TestLazyImportPreserved`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Optional

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


def _resolve_cli_level(verbose: bool, quiet: bool) -> str | None:
    """Map CLI flags to a logging level string.

    ``--verbose`` wins over ``--quiet`` if both are set (best effort: warn,
    don't crash). Returning ``None`` defers to ``PDF_MCP_LOG_LEVEL`` /
    INFO inside ``configure_logging``.
    """
    if verbose:
        return "DEBUG"
    if quiet:
        return "WARNING"
    return None


@app.callback()
def _root(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Print pdf-mcp version and exit.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Enable DEBUG logging (overrides --quiet).",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet",
        "-q",
        help="Restrict logging to WARNING and above.",
    ),
    log_format: str = typer.Option(
        "text",
        "--log-format",
        help="Log output format. One of: text, json.",
        case_sensitive=False,
        metavar="text|json",
    ),
) -> None:
    """pdf-mcp entry point.

    Use a subcommand (run ``pdf-mcp --help``) for the available verbs. The
    initial release ships ``serve`` for MCP stdio mode; further verbs are
    added under TICKET-04 onwards in the v1.3.0 sprint backlog.
    """
    # Lazy import keeps ``pdf-mcp --version`` fast (importing pdf_mcp.logging
    # is cheap, but staying consistent with the lazy-import pattern used by
    # ``serve``).
    from pdf_mcp import logging as pdf_logging

    pdf_logging.configure_logging(
        level=_resolve_cli_level(verbose, quiet),
        fmt=(log_format or "text").lower(),
    )


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


# ---------------------------------------------------------------------------
# Registry-driven verb groups (TICKET-05 c3, v1.3.0)
# ---------------------------------------------------------------------------


def _load_kwargs(json_str: Optional[str], json_file: Optional[Path]) -> dict[str, Any]:
    """Decode CLI-provided kwargs.

    Precedence: ``--json`` wins over ``--json-file`` (matches the human
    expectation that an explicit string overrides a referenced file).
    Returns an empty dict if neither is supplied so tools that take no
    args still work without a sentinel payload.

    Raises :class:`typer.BadParameter` on malformed JSON or non-dict
    payloads so Typer renders a tidy CLI error.
    """
    if json_str is not None:
        raw = json_str
    elif json_file is not None:
        raw = json_file.read_text()
    else:
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"--json is not valid JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise typer.BadParameter(f"--json must decode to a JSON object, got {type(decoded).__name__}")
    return decoded


def _emit_result(result: Any, output: Optional[Path], pretty: bool) -> None:
    """Serialise ``result`` as JSON and route to file/stdout.

    ``default=str`` means values that are not JSON-serialisable
    (e.g. :class:`pathlib.Path`) get coerced to their string form
    instead of crashing. This matches the behaviour of the MCP
    server which already returns strings for paths.
    """
    text = json.dumps(result, indent=2 if pretty else None, default=str, ensure_ascii=False)
    if output is not None:
        output.write_text(text)
        return
    typer.echo(text)


def _make_tool_command(tool_name: str) -> Callable[..., None]:
    """Build a Typer command function for one registry tool.

    The command captures the tool *name*, not the resolved callable, so
    importing ``pdf_mcp.cli`` continues to avoid loading ``pdf_tools``.
    Resolution happens inside the command body when the user actually
    invokes the verb, at which point ``LazyCallable.__call__`` triggers
    ``importlib.import_module``.
    """

    def _cmd(
        json_str: Optional[str] = typer.Option(
            None,
            "--json",
            help='Tool kwargs as a JSON object (e.g. --json \'{"pdf_path":"a.pdf"}\').',
            metavar="JSON",
        ),
        json_file: Optional[Path] = typer.Option(
            None,
            "--json-file",
            help="Path to a JSON file whose contents are used as kwargs.",
            metavar="PATH",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
        output: Optional[Path] = typer.Option(
            None,
            "--output",
            "-o",
            help="Write JSON result to this file instead of stdout.",
            metavar="PATH",
        ),
        pretty: bool = typer.Option(
            False,
            "--pretty",
            help="Indent JSON output for human consumption.",
        ),
    ) -> None:
        # Lazy import — keeps `pdf-mcp <verb> --help` from pulling
        # in pdf_tools.
        from pdf_mcp import registry as _registry

        kwargs = _load_kwargs(json_str, json_file)
        tool = _registry.get(tool_name)
        try:
            result = tool.callable(**kwargs)
        except Exception as exc:  # noqa: BLE001 - propagate as CLI error
            typer.echo(f"error: {tool_name} failed: {exc}", err=True)
            raise typer.Exit(code=1) from exc
        _emit_result(result, output, pretty)

    return _cmd


def _register_verb_groups(parent: typer.Typer) -> None:
    """Mount one Typer subapp per registry verb group.

    This runs once at module import time. The registry import is cheap
    (no transitive ``pdf_tools`` load) so we can do it eagerly without
    breaking the lazy-import invariant.
    """
    # Local import keeps top-of-file imports tidy; identical effect.
    from pdf_mcp import registry

    for group in registry.verb_groups():
        sub = typer.Typer(
            name=group.verb,
            help=group.help,
            no_args_is_help=True,
            add_completion=False,
        )
        for tool in group.tools:
            command_fn = _make_tool_command(tool.name)
            # Click conventionally uses dashes; preserve underscores as
            # an alias by registering the canonical (dashed) form.
            cli_name = tool.name.replace("_", "-")
            sub.command(name=cli_name, help=tool.description)(command_fn)
        parent.add_typer(sub, name=group.verb, help=group.help)


# Mount everything at import time so the Typer surface is ready before
# ``CliRunner`` or the entry point poke at it.
_register_verb_groups(app)


def main() -> None:
    """Setuptools console-script entry point.

    Kept as a tiny shim so ``[project.scripts]`` resolves to a stable
    callable even if the underlying Typer app structure changes.
    """
    app()


if __name__ == "__main__":
    main()
