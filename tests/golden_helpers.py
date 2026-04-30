"""Golden snapshot helpers for the pdf-mcp CLI (TICKET-03, v1.3.0).

A golden snapshot test compares an exact byte-for-byte (after light
normalisation) capture of CLI output against a committed file in
``tests/golden/``. This catches accidental UX regressions (e.g. removed
help text, renamed verb, swapped flag) that freeform string asserts miss.

Workflow
--------

1. Author writes a test that calls ``assert_cli_output_matches(...)``
   with a snapshot name. The first run fails because the file doesn't
   exist yet — instructions in the failure message tell you to set the
   environment variable to materialise it.

2. Run with ``PDF_MCP_UPDATE_SNAPSHOTS=1 pytest tests/test_cli_golden.py``
   to overwrite/create the snapshot from the current CLI output.

3. Commit ``tests/golden/<name>.txt`` alongside the test.

4. CI then re-runs the same comparison and fails on any drift.

Determinism rules
-----------------

* Capture STDOUT only (Typer puts ``--help`` and ``--version`` there;
  errors go through stderr where caplog/capfd can be used instead).
* Strip trailing whitespace per line and ensure a single trailing newline
  at EOF (pre-commit ``end-of-file-fixer`` would otherwise diff us into
  the dust).
* Normalise ``\\r\\n`` -> ``\\n`` so Windows-checkout developers don't
  trigger unintended diffs.
* Never include timestamps, working-directory paths, version-of-Python,
  or anything else that varies across machines.

The harness intentionally does NOT diff against ``--help`` output of
heavy verb groups while those are still under design (see TICKET-05).
For now we only pin the stable surface: ``--version``, ``--help``, and
``serve --help``. Adding more snapshots is one ``assert_cli_output_matches``
call away.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

GOLDEN_DIR: Path = Path(__file__).resolve().parent / "golden"
UPDATE_ENV: str = "PDF_MCP_UPDATE_SNAPSHOTS"


def _normalise(text: str) -> str:
    """Right-strip every line and force a single trailing newline."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    # Drop trailing empty lines, then add exactly one ``\n``.
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


def assert_cli_output_matches(
    cli_runner: object,
    app: object,
    args: Sequence[str],
    snapshot_name: str,
) -> None:
    """Run ``app`` with ``args`` and compare its STDOUT to a golden file.

    Parameters
    ----------
    cli_runner:
        ``typer.testing.CliRunner`` instance.
    app:
        The Typer ``app`` to invoke (anything CliRunner accepts).
    args:
        Command-line args to pass to ``app``.
    snapshot_name:
        File stem under ``tests/golden/`` (no extension). Will be
        materialised as ``tests/golden/<snapshot_name>.txt``.

    Behaviour
    ---------
    * If ``PDF_MCP_UPDATE_SNAPSHOTS=1``, write the captured output and
      pass. Useful for first-time snapshot generation and intentional
      updates.
    * Otherwise read the snapshot file and compare. Missing snapshot
      raises ``AssertionError`` with the exact regen command embedded
      so the developer is unblocked in one shell line.
    """
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    snapshot_path = GOLDEN_DIR / f"{snapshot_name}.txt"

    # CliRunner.invoke() returns a Result with .output / .stdout / .stderr.
    result = cli_runner.invoke(app, list(args))  # type: ignore[attr-defined]
    if result.exit_code != 0:
        raise AssertionError(
            f"snapshot {snapshot_name!r}: CLI exited with code {result.exit_code} — stdout: {result.output!r}"
        )

    captured = _normalise(result.output)

    if os.environ.get(UPDATE_ENV) in ("1", "true", "yes"):
        snapshot_path.write_text(captured, encoding="utf-8")
        return

    if not snapshot_path.exists():
        raise AssertionError(
            f"snapshot {snapshot_name!r} does not exist at {snapshot_path}.\n"
            f"To materialise it from the current CLI output, run:\n"
            f"    {UPDATE_ENV}=1 pytest tests/test_cli_golden.py::"
            f"<this test>\n"
            f"and commit the new file."
        )

    expected = _normalise(snapshot_path.read_text(encoding="utf-8"))
    if captured != expected:
        # Build a small unified-diff style preview so the failure is
        # immediately actionable.
        import difflib

        diff = "".join(
            difflib.unified_diff(
                expected.splitlines(keepends=True),
                captured.splitlines(keepends=True),
                fromfile=f"{snapshot_name}.txt (expected)",
                tofile=f"{snapshot_name}.txt (actual)",
                n=3,
            )
        )
        raise AssertionError(
            f"snapshot {snapshot_name!r} drifted.\n"
            f"To regenerate: {UPDATE_ENV}=1 pytest "
            f"tests/test_cli_golden.py\n"
            f"Then review the diff and commit if intentional.\n\n"
            f"--- diff ---\n{diff}"
        )
