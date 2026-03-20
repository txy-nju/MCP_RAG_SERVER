"""Project entrypoint for the Modular RAG MCP Server."""

from __future__ import annotations

from pathlib import Path

from core.settings import Settings, load_settings
from observability.logger import get_logger


def main(settings_path: str | Path = "config/settings.yaml") -> int:
    """Load settings and stop immediately if configuration is invalid."""
    logger = get_logger()
    try:
        settings: Settings = load_settings(settings_path)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load settings: %s", exc)
        return 1

    logger.info("Loaded settings successfully from %s", settings_path)
    logger.info("Configured LLM provider: %s", settings.llm.provider)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
