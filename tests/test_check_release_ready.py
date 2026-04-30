"""Unit tests for scripts/check_release_ready.py::_read_pyproject_version.

Regression coverage for the broken regex fallback (BUG, v1.3.0):
the fallback raw-string had double-escaped backslashes, so ``\\s`` was a
literal ``\\s`` rather than whitespace. These tests pin both the tomllib
preferred path and the regex fallback (double-quote, single-quote, missing).

The ``scripts/`` directory is intentionally NOT a Python package
(it is excluded from setuptools' packages.find and has no ``__init__.py``).
We load the script by file path with ``importlib`` so the test works
regardless of CWD or sys.path setup.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_release_ready.py"


def _load_script_module() -> ModuleType:
    """Load scripts/check_release_ready.py as a fresh module."""
    spec = importlib.util.spec_from_file_location("check_release_ready_under_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None, f"Could not build importlib spec for {SCRIPT_PATH}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_pyproject(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "pyproject.toml"
    p.write_text(content, encoding="utf-8")
    return p


def _force_regex_fallback(monkeypatch: pytest.MonkeyPatch, module: ModuleType) -> None:
    """Make ``import tomllib`` fail inside _read_pyproject_version.

    Setting ``sys.modules['tomllib'] = None`` causes Python to raise
    ImportError on subsequent ``import tomllib`` statements, which forces
    the function into its except-pass and on to the regex fallback.
    """
    monkeypatch.setitem(sys.modules, "tomllib", None)
    if hasattr(module, "tomllib"):
        monkeypatch.delattr(module, "tomllib", raising=False)


def test_pyproject_version_double_quoted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module()
    _force_regex_fallback(monkeypatch, module)
    pyproject = _write_pyproject(
        tmp_path,
        '[project]\nname = "demo"\nversion = "1.2.3"\n',
    )
    assert module._read_pyproject_version(pyproject) == "1.2.3"


def test_pyproject_version_single_quoted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module()
    _force_regex_fallback(monkeypatch, module)
    pyproject = _write_pyproject(
        tmp_path,
        "[project]\nname = 'demo'\nversion = '1.2.3'\n",
    )
    assert module._read_pyproject_version(pyproject) == "1.2.3"


def test_pyproject_version_missing_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module()
    _force_regex_fallback(monkeypatch, module)
    pyproject = _write_pyproject(
        tmp_path,
        '[project]\nname = "demo"\n',
    )
    with pytest.raises(RuntimeError, match="Could not find version"):
        module._read_pyproject_version(pyproject)


def test_pyproject_version_uses_tomllib_when_available(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The TOML below is valid, but the regex fallback would NOT match the
    # ``version = "1.2.3"  # comment`` line because the trailing inline
    # comment fails the ``$`` anchor. So if tomllib is preferred (as it
    # should be), parsing succeeds; if the function ever regresses to
    # regex-only, this test fails.
    pytest.importorskip("tomllib")
    module = _load_script_module()
    pyproject = _write_pyproject(
        tmp_path,
        '[project]\nname = "demo"\nversion    =    "1.2.3"  # inline comment\n',
    )
    assert module._read_pyproject_version(pyproject) == "1.2.3"
