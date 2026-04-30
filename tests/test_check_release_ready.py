"""Unit tests for scripts.check_release_ready._read_pyproject_version.

Regression coverage for the broken regex fallback (BUG, v1.3.0):
the fallback raw-string had double-escaped backslashes, so ``\\s`` was a
literal ``\\s`` rather than whitespace. These tests pin both the tomllib
preferred path and the regex fallback (double-quote, single-quote, missing).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from scripts.check_release_ready import _read_pyproject_version


def _write_pyproject(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "pyproject.toml"
    p.write_text(content, encoding="utf-8")
    return p


def _force_regex_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``import tomllib`` fail inside _read_pyproject_version.

    Setting ``sys.modules['tomllib'] = None`` causes Python to raise
    ImportError on subsequent ``import tomllib`` statements, which forces
    the function into its except-pass and on to the regex fallback.
    """
    import scripts.check_release_ready as mod

    monkeypatch.setitem(sys.modules, "tomllib", None)
    if hasattr(mod, "tomllib"):
        monkeypatch.delattr(mod, "tomllib", raising=False)


def test_pyproject_version_double_quoted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _force_regex_fallback(monkeypatch)
    pyproject = _write_pyproject(
        tmp_path,
        '[project]\nname = "demo"\nversion = "1.2.3"\n',
    )
    assert _read_pyproject_version(pyproject) == "1.2.3"


def test_pyproject_version_single_quoted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _force_regex_fallback(monkeypatch)
    pyproject = _write_pyproject(
        tmp_path,
        "[project]\nname = 'demo'\nversion = '1.2.3'\n",
    )
    assert _read_pyproject_version(pyproject) == "1.2.3"


def test_pyproject_version_missing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _force_regex_fallback(monkeypatch)
    pyproject = _write_pyproject(
        tmp_path,
        '[project]\nname = "demo"\n',
    )
    with pytest.raises(RuntimeError, match="Could not find version"):
        _read_pyproject_version(pyproject)


def test_pyproject_version_uses_tomllib_when_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The TOML below is valid, but the regex fallback would NOT match the
    # ``version = "1.2.3"  # comment`` line because the trailing inline
    # comment fails the ``$`` anchor. So if tomllib is preferred (as it
    # should be), parsing succeeds; if the function ever regresses to
    # regex-only, this test fails.
    pyproject = _write_pyproject(
        tmp_path,
        '[project]\nname = "demo"\nversion    =    "1.2.3"  # inline comment\n',
    )
    pytest.importorskip("tomllib")
    assert _read_pyproject_version(pyproject) == "1.2.3"
