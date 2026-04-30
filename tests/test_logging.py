"""Unit tests for pdf_mcp.logging (v1.3.0 structured logging, TICKET-02).

Pinned behaviours:
1. ``configure_logging`` sets up a single handler on the package logger.
2. JSON mode emits one JSON object per record with required fields
   (``timestamp``, ``level``, ``logger``, ``message``).
3. Text mode emits a human-readable single line per record.
4. Configuring twice does NOT add duplicate handlers (idempotent).
5. The module respects the ``PDF_MCP_LOG_LEVEL`` env var when no explicit
   level is passed.
6. CLI ``--verbose`` / ``--quiet`` flags route through the same code path.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Module-level imports
# ---------------------------------------------------------------------------


def _import_module() -> Any:
    """Import pdf_mcp.logging fresh (the module reconfigures logging at import)."""
    import importlib

    import pdf_mcp.logging as mod

    return importlib.reload(mod)


def _capture_records(caplog: pytest.LogCaptureFixture, logger_name: str = "pdf_mcp") -> Any:
    """Helper: bind caplog to the pdf_mcp logger so records flow into it."""
    caplog.set_level(logging.DEBUG, logger=logger_name)
    return caplog


# ---------------------------------------------------------------------------
# 1. configure_logging exists with the expected signature
# ---------------------------------------------------------------------------


def test_configure_logging_callable() -> None:
    mod = _import_module()
    assert hasattr(mod, "configure_logging")
    assert callable(mod.configure_logging)


def test_configure_logging_default_level_is_info() -> None:
    mod = _import_module()
    mod.configure_logging()
    pkg = logging.getLogger("pdf_mcp")
    assert pkg.level <= logging.INFO


# ---------------------------------------------------------------------------
# 2. JSON formatter contract
# ---------------------------------------------------------------------------


def test_json_formatter_emits_required_fields(capsys: pytest.CaptureFixture[str]) -> None:
    mod = _import_module()
    mod.configure_logging(level="DEBUG", fmt="json")

    pkg = logging.getLogger("pdf_mcp.test_json")
    pkg.warning("hello world")

    captured = capsys.readouterr().err or capsys.readouterr().out
    # JSON mode writes one JSON object per record, terminated by newline.
    last_line = next(line for line in reversed(captured.splitlines()) if line.strip())
    payload = json.loads(last_line)
    for key in ("timestamp", "level", "logger", "message"):
        assert key in payload, f"missing {key!r} in JSON record: keys={list(payload.keys())}"
    assert payload["level"] == "WARNING"
    assert payload["logger"] == "pdf_mcp.test_json"
    assert payload["message"] == "hello world"


def test_json_formatter_handles_extras(capsys: pytest.CaptureFixture[str]) -> None:
    mod = _import_module()
    mod.configure_logging(level="DEBUG", fmt="json")

    pkg = logging.getLogger("pdf_mcp.test_extras")
    pkg.info("with extras", extra={"trace_id": "abc-123", "duration_ms": 42})

    captured = capsys.readouterr().err or capsys.readouterr().out
    last_line = next(line for line in reversed(captured.splitlines()) if line.strip())
    payload = json.loads(last_line)
    # Extras must be preserved (extras either flat or under an ``extra`` key)
    serialized = json.dumps(payload)
    assert "abc-123" in serialized
    assert "42" in serialized


# ---------------------------------------------------------------------------
# 3. Text formatter
# ---------------------------------------------------------------------------


def test_text_formatter_emits_single_line(capsys: pytest.CaptureFixture[str]) -> None:
    mod = _import_module()
    mod.configure_logging(level="INFO", fmt="text")

    pkg = logging.getLogger("pdf_mcp.test_text")
    pkg.info("readable message")

    captured = capsys.readouterr().err or capsys.readouterr().out
    line = next(line for line in reversed(captured.splitlines()) if line.strip())
    # Loose contract: must mention level + logger + message in a single line.
    assert "INFO" in line
    assert "pdf_mcp.test_text" in line
    assert "readable message" in line


# ---------------------------------------------------------------------------
# 4. Idempotency
# ---------------------------------------------------------------------------


def test_configure_logging_idempotent() -> None:
    mod = _import_module()
    mod.configure_logging(level="INFO", fmt="json")
    pkg = logging.getLogger("pdf_mcp")
    handlers_before = list(pkg.handlers)

    mod.configure_logging(level="INFO", fmt="json")
    handlers_after = list(pkg.handlers)

    assert len(handlers_after) == len(handlers_before), (
        "configure_logging must not add duplicate handlers when called twice"
    )


# ---------------------------------------------------------------------------
# 5. Env override
# ---------------------------------------------------------------------------


def test_env_var_overrides_default_level(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PDF_MCP_LOG_LEVEL", "DEBUG")
    mod = _import_module()
    mod.configure_logging()  # explicit args omitted
    pkg = logging.getLogger("pdf_mcp")
    assert pkg.level == logging.DEBUG, "PDF_MCP_LOG_LEVEL=DEBUG must lower the package logger to DEBUG"


def test_env_var_invalid_level_falls_back_to_info(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PDF_MCP_LOG_LEVEL", "NOT_A_REAL_LEVEL")
    mod = _import_module()
    mod.configure_logging()
    pkg = logging.getLogger("pdf_mcp")
    # Implementation choice: invalid level falls back to INFO (must not crash).
    assert pkg.level == logging.INFO, "invalid PDF_MCP_LOG_LEVEL must not raise and must fall back to INFO"


# ---------------------------------------------------------------------------
# 6. CLI integration: --verbose / --quiet routes through configure_logging
# ---------------------------------------------------------------------------


@pytest.fixture()
def cli_runner():
    typer_testing = pytest.importorskip("typer.testing")
    return typer_testing.CliRunner()


def _mock_serve_invocation(cli_runner, monkeypatch: pytest.MonkeyPatch, args: list[str]) -> tuple[Any, dict[str, Any]]:
    """Drive ``pdf-mcp <flags> serve`` while mocking ``mcp.run`` and capturing
    the arguments passed to ``configure_logging``.

    We use ``serve`` (not ``--version``) because ``--version`` is an eager
    Typer option that exits before the root callback body executes.
    Real subcommands always run the root body first, so this is the path
    end users hit. ``mcp.run`` is patched to a no-op so we don't actually
    spin up the stdio loop.
    """
    from pdf_mcp.cli import app

    captured: dict[str, Any] = {}

    def _spy(level: str | int = "INFO", fmt: str = "text", **_: Any) -> None:
        captured["level"] = level
        captured["fmt"] = fmt

    monkeypatch.setattr("pdf_mcp.logging.configure_logging", _spy)
    monkeypatch.setattr("pdf_mcp.server.mcp.run", lambda **_: None)

    result = cli_runner.invoke(app, [*args, "serve"])
    return result, captured


def test_cli_verbose_sets_debug_level(cli_runner, monkeypatch: pytest.MonkeyPatch) -> None:
    """``pdf-mcp --verbose serve`` must configure DEBUG-level logging."""
    monkeypatch.delenv("PDF_MCP_LOG_LEVEL", raising=False)
    result, captured = _mock_serve_invocation(cli_runner, monkeypatch, ["--verbose"])
    assert result.exit_code == 0, result.output
    assert captured.get("level") in ("DEBUG", logging.DEBUG), (
        f"--verbose must request DEBUG level, got {captured.get('level')!r}"
    )


def test_cli_quiet_sets_warning_level(cli_runner, monkeypatch: pytest.MonkeyPatch) -> None:
    """``pdf-mcp --quiet serve`` must configure WARNING-level logging."""
    monkeypatch.delenv("PDF_MCP_LOG_LEVEL", raising=False)
    result, captured = _mock_serve_invocation(cli_runner, monkeypatch, ["--quiet"])
    assert result.exit_code == 0, result.output
    assert captured.get("level") in ("WARNING", logging.WARNING), (
        f"--quiet must request WARNING level, got {captured.get('level')!r}"
    )


def test_cli_log_format_flag_routes_through(cli_runner, monkeypatch: pytest.MonkeyPatch) -> None:
    """``pdf-mcp --log-format=json serve`` must request JSON formatter."""
    result, captured = _mock_serve_invocation(cli_runner, monkeypatch, ["--log-format", "json"])
    assert result.exit_code == 0, result.output
    assert captured.get("fmt") == "json", f"--log-format=json must request fmt='json', got {captured.get('fmt')!r}"


def test_cli_default_passes_none_level_through(cli_runner, monkeypatch: pytest.MonkeyPatch) -> None:
    """``pdf-mcp serve`` (no flags) must defer level resolution to env/INFO.

    The CLI must NOT hard-code a level; instead it passes ``None`` so
    ``configure_logging`` consults ``PDF_MCP_LOG_LEVEL`` and falls back to
    INFO. This keeps env-var setups working when invoked via ``pdf-mcp``.
    """
    monkeypatch.delenv("PDF_MCP_LOG_LEVEL", raising=False)
    result, captured = _mock_serve_invocation(cli_runner, monkeypatch, [])
    assert result.exit_code == 0, result.output
    assert captured.get("level") is None, f"default invocation must pass level=None, got {captured.get('level')!r}"


# ---------------------------------------------------------------------------
# Cleanup: tests must not leak handlers across modules.
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_pdf_mcp_logger() -> None:
    """Remove any handlers attached by previous tests so each test starts fresh."""
    pkg = logging.getLogger("pdf_mcp")
    saved = list(pkg.handlers)
    saved_level = pkg.level
    saved_propagate = pkg.propagate
    saved_env = os.environ.get("PDF_MCP_LOG_LEVEL")
    yield
    for h in list(pkg.handlers):
        pkg.removeHandler(h)
    for h in saved:
        pkg.addHandler(h)
    pkg.setLevel(saved_level)
    pkg.propagate = saved_propagate
    if saved_env is not None:
        os.environ["PDF_MCP_LOG_LEVEL"] = saved_env
