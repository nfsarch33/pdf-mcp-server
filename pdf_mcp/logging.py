"""Structured logging for pdf-mcp (TICKET-02, v1.3.0).

Goals
-----
* Single configuration call from CLI / server / tests.
* JSON or human-readable output (selectable at runtime).
* Idempotent — calling twice does not double output or stack handlers.
* Lives on the ``pdf_mcp`` package logger only, never the root logger,
  so embedding pdf-mcp in a larger Python program does not steal that
  program's logging configuration.
* Respects ``PDF_MCP_LOG_LEVEL`` env var when no explicit level is passed.

Usage
-----

    from pdf_mcp import logging as pdf_logging

    # CLI / server entrypoint
    pdf_logging.configure_logging(level="INFO", fmt="json")

    # Library code
    import logging
    log = logging.getLogger(__name__)  # e.g. "pdf_mcp.tools.text"
    log.info("starting extraction", extra={"path": str(p)})

The structured (JSON) formatter emits one object per record with at least
these keys: ``timestamp`` (ISO 8601 UTC), ``level``, ``logger``, ``message``.
Anything passed via ``extra=`` is preserved verbatim under those top-level
keys (or a nested ``extra`` mapping if a key would otherwise collide).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any, Iterable, TextIO

PACKAGE_LOGGER = "pdf_mcp"
ENV_LEVEL = "PDF_MCP_LOG_LEVEL"

# Standard ``LogRecord`` attribute set; anything outside this is treated as
# a user-supplied "extra" and forwarded into structured output.
_RESERVED: set[str] = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",  # populated by ``LogRecord.getMessage()``
    "asctime",  # populated by ``Formatter.format()``
    "taskName",  # Python 3.12+
}


def _resolve_level(level: str | int | None) -> int:
    """Resolve a logging level from CLI / env / default.

    Priority: explicit argument > ``PDF_MCP_LOG_LEVEL`` env var > INFO.
    Invalid env values silently fall back to INFO (logging must never crash
    a CLI) but invalid explicit args raise — that path is a programming
    error, not a runtime input.
    """
    if level is None:
        env = os.environ.get(ENV_LEVEL, "").strip().upper()
        if not env:
            return logging.INFO
        candidate = logging.getLevelName(env)
        if isinstance(candidate, int):
            return candidate
        return logging.INFO  # invalid env → INFO, no exception
    if isinstance(level, int):
        return level
    candidate = logging.getLevelName(str(level).strip().upper())
    if isinstance(candidate, int):
        return candidate
    raise ValueError(f"unknown log level: {level!r}")


class _JsonFormatter(logging.Formatter):
    """Minimal JSON formatter — one object per record, terminated by newline."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003 - stdlib name
        payload: dict[str, Any] = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)) + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = record.stack_info

        # Forward any extras (record attributes outside the reserved set).
        extras: dict[str, Any] = {}
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            extras[key] = _safe_jsonify(value)
        if extras:
            # Avoid clobbering reserved top-level keys.
            collision = set(extras).intersection(payload)
            if collision:
                payload["extra"] = extras
            else:
                payload.update(extras)

        return json.dumps(payload, ensure_ascii=False, default=str)


def _safe_jsonify(value: Any) -> Any:
    """Best-effort JSON-safe conversion for extras (str fallback)."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


_TEXT_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(
    level: str | int | None = None,
    fmt: str = "text",
    *,
    stream: TextIO | None = None,
    extra_loggers: Iterable[str] = (),
) -> logging.Logger:
    """Configure the ``pdf_mcp`` package logger.

    Parameters
    ----------
    level:
        Logging level name (``"DEBUG"`` etc.) or numeric level. ``None`` means
        "consult ``PDF_MCP_LOG_LEVEL`` then default to INFO".
    fmt:
        ``"text"`` (default) or ``"json"``. Anything else falls back to text.
    stream:
        Output stream. Defaults to ``sys.stderr`` so logging never collides
        with stdout content the CLI is computing for the user.
    extra_loggers:
        Optional list of additional logger names that should also adopt this
        configuration (e.g. ``"mcp"`` if you want SDK chatter to flow through
        the same handler).

    Returns
    -------
    logging.Logger
        The configured ``pdf_mcp`` logger (handy for tests and call-site
        chaining; the function's primary effect is the side-effect on the
        package logger).
    """
    target_stream = stream if stream is not None else sys.stderr
    resolved_level = _resolve_level(level)

    if fmt == "json":
        formatter: logging.Formatter = _JsonFormatter()
    else:
        formatter = logging.Formatter(_TEXT_FORMAT)

    package = logging.getLogger(PACKAGE_LOGGER)
    package.setLevel(resolved_level)
    # Keep propagate=True so:
    #   * pytest's ``caplog`` (handler on root) still sees our records;
    #   * embedding apps that configured their own root handler still
    #     receive our log lines without extra wiring.
    # If a host wants to silence pdf-mcp emission to root, they can set
    # ``logging.getLogger("pdf_mcp").propagate = False`` themselves AFTER
    # calling configure_logging. Documented contract: configure_logging
    # never mutates propagate.

    # Idempotency: drop any handler we previously installed before adding
    # a fresh one. We tag ours with ``_pdf_mcp_managed`` so we never touch
    # handlers a host application attached on its own.
    for handler in list(package.handlers):
        if getattr(handler, "_pdf_mcp_managed", False):
            package.removeHandler(handler)

    handler = logging.StreamHandler(target_stream)
    handler.setLevel(resolved_level)
    handler.setFormatter(formatter)
    handler._pdf_mcp_managed = True  # type: ignore[attr-defined]
    package.addHandler(handler)

    for name in extra_loggers:
        if not name or name == PACKAGE_LOGGER:
            continue
        other = logging.getLogger(name)
        other.setLevel(resolved_level)
        for h in list(other.handlers):
            if getattr(h, "_pdf_mcp_managed", False):
                other.removeHandler(h)
        side_handler = logging.StreamHandler(target_stream)
        side_handler.setLevel(resolved_level)
        side_handler.setFormatter(formatter)
        side_handler._pdf_mcp_managed = True  # type: ignore[attr-defined]
        other.addHandler(side_handler)
        # Keep propagate=True for the same reasons as the package logger.

    return package


__all__ = ["configure_logging", "PACKAGE_LOGGER", "ENV_LEVEL"]
