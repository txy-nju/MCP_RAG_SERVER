"""Minimal logging setup for early project phases."""

from __future__ import annotations

import logging
import sys


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
