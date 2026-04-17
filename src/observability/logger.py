"""Logging utilities for stderr output and JSON Lines trace persistence."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from core.settings import load_settings


_DEFAULT_SETTINGS_PATH = Path("config/settings.yaml")
_TRACE_LOGGER_NAME = "modular_rag_mcp_server.trace"


class JSONFormatter(logging.Formatter):
    """Serialize logging records as one-line JSON payloads."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any]
        if isinstance(record.msg, dict) and not record.args:
            payload = dict(record.msg)
        else:
            payload = {
                "message": record.getMessage(),
                "level": record.levelname,
                "logger": record.name,
            }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str = "modular_rag_mcp_server") -> logging.Logger:
    """Return a stderr-backed logger."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def get_trace_logger(
    *,
    trace_file: str | Path | None = None,
    settings_path: str | Path = _DEFAULT_SETTINGS_PATH,
) -> logging.Logger:
    """Return a file-backed logger that appends one JSON object per line."""

    target_file = _resolve_trace_file(trace_file=trace_file, settings_path=settings_path)
    logger_name = f"{_TRACE_LOGGER_NAME}:{target_file}"
    logger = logging.getLogger(logger_name)

    if logger.handlers:
        return logger

    target_file.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(target_file, encoding="utf-8")
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def write_trace(
    trace_dict: dict[str, Any],
    *,
    trace_file: str | Path | None = None,
    settings_path: str | Path = _DEFAULT_SETTINGS_PATH,
) -> None:
    """Append one trace payload to the configured JSON Lines file."""

    logger = get_trace_logger(trace_file=trace_file, settings_path=settings_path)
    logger.info(trace_dict)


def _resolve_trace_file(*, trace_file: str | Path | None, settings_path: str | Path) -> Path:
    if trace_file is not None:
        return Path(trace_file)

    settings = load_settings(settings_path)
    return Path(settings.observability.trace_file)


__all__ = ["JSONFormatter", "get_logger", "get_trace_logger", "write_trace"]
