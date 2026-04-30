#!/usr/bin/env python3
"""Generate ``USAGE.md`` from the live :mod:`pdf_mcp.registry`.

The registry is the single source of truth for the CLI / MCP surface, so
deriving the docs from it eliminates drift entirely. Contributors who
add a new tool with ``register_tool(...)`` re-run this script and the
docs catch up in one commit.

Usage
-----

    python scripts/generate_usage_doc.py [--out USAGE.md] [--check]

* ``--out PATH``: target file (default ``USAGE.md`` at repo root).
* ``--check``: do not write; exit 1 if the on-disk file differs from
  what would be generated. Useful as a CI guard.

The output deliberately renders Markdown only, with stable section
order: alphabetical within each verb group, verb groups in
``registry.verb_groups()`` order (which mirrors insertion order in
``_seed_default_registry``).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = REPO_ROOT / "USAGE.md"


def _render(verb_groups: Iterable[object]) -> str:
    """Render the full USAGE.md body from a list of VerbGroup objects."""
    out: list[str] = []
    out.append("# pdf-mcp CLI Usage Reference")
    out.append("")
    out.append(
        "Auto-generated from `pdf_mcp.registry` by "
        "`scripts/generate_usage_doc.py`. Do not edit by hand; re-run the "
        "generator after adding or removing a tool."
    )
    out.append("")
    out.append("## Invocation")
    out.append("")
    out.append("```")
    out.append("pdf-mcp <verb> <tool-name> [--json '{...}'] [--json-file PATH]")
    out.append("                            [--pretty] [--output PATH]")
    out.append("```")
    out.append("")
    out.append("* `--json` and `--json-file` are mutually exclusive (`--json` wins if both are passed).")
    out.append("* `--pretty` indents the JSON output for human reading.")
    out.append("* `--output PATH` writes the JSON result to a file instead of stdout.")
    out.append("* Tool exceptions exit non-zero with `error: <tool> failed: <msg>` on stderr.")
    out.append("")
    out.append("Run `pdf-mcp --help` for the top-level surface and `pdf-mcp <verb> --help` for the per-verb tool list.")
    out.append("")

    out.append("## Verb groups")
    out.append("")
    out.append("| Verb | Tools | Description |")
    out.append("| ---- | ----- | ----------- |")
    for g in verb_groups:
        out.append(f"| `{g.verb}` | {len(g.tools)} | {g.help} |")  # type: ignore[attr-defined]
    out.append("")

    for g in verb_groups:
        out.append(f"## `pdf-mcp {g.verb}`")  # type: ignore[attr-defined]
        out.append("")
        out.append(g.help)  # type: ignore[attr-defined]
        out.append("")
        out.append("| Tool | Description |")
        out.append("| ---- | ----------- |")
        for tool in g.tools:  # type: ignore[attr-defined]
            cli_name = tool.name.replace("_", "-")
            out.append(f"| `{cli_name}` | {tool.description} |")
        out.append("")

    out.append("## Backwards compatibility")
    out.append("")
    out.append("* `pdf-mcp serve` runs the MCP server over stdio (drop-in replacement for `python -m pdf_mcp.server`).")
    out.append("* All tools remain reachable via the MCP protocol with their original `snake_case` names.")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Path to write USAGE.md (default: repo root)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit 1 if file is out of sync.",
    )
    args = parser.parse_args(argv)

    # Lazy import: keep this script runnable from any CWD without
    # paying the pdf_tools dependency tax. The registry import does
    # not pull in pdf_tools (that's the whole point of LazyCallable).
    sys.path.insert(0, str(REPO_ROOT))
    from pdf_mcp import registry

    rendered = _render(registry.verb_groups())

    if args.check:
        if not args.out.exists():
            print(f"USAGE.md missing at {args.out}", file=sys.stderr)
            return 1
        existing = args.out.read_text(encoding="utf-8")
        if existing != rendered:
            print(
                "USAGE.md is out of sync with pdf_mcp.registry. Run `python scripts/generate_usage_doc.py` and commit.",
                file=sys.stderr,
            )
            return 1
        return 0

    args.out.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.out} ({len(rendered)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
