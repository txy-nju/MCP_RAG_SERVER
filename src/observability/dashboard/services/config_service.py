"""Configuration reading service for the Dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.settings import Settings, load_settings

_DEFAULT_CONFIG_PATH = str(Path(__file__).parents[4] / "config" / "settings.yaml")


@dataclass
class ComponentCard:
    """Represents a single system component for display."""

    name: str
    provider: str
    details: dict[str, Any] = field(default_factory=dict)


class ConfigService:
    """Wraps Settings loading and formats component configuration for the Dashboard."""

    def __init__(self, settings: Settings | None = None, config_path: str | None = None) -> None:
        if settings is not None:
            self._settings = settings
        else:
            path = config_path or _DEFAULT_CONFIG_PATH
            self._settings = load_settings(path)

    def get_settings(self) -> Settings:
        """Return the loaded Settings object."""
        return self._settings

    def get_component_cards(self) -> list[ComponentCard]:
        """Return a list of component cards describing the active configuration."""
        s = self._settings
        cards: list[ComponentCard] = [
            ComponentCard(
                name="LLM",
                provider=s.llm.provider,
                details={"model": s.llm.model},
            ),
            ComponentCard(
                name="Embedding",
                provider=s.embedding.provider,
                details={"model": s.embedding.model},
            ),
            ComponentCard(
                name="Splitter",
                provider=s.splitter.provider,
                details={
                    "chunk_size": s.splitter.chunk_size,
                    "chunk_overlap": s.splitter.chunk_overlap,
                },
            ),
            ComponentCard(
                name="Vector Store",
                provider=s.vector_store.provider,
                details={
                    "collection": s.vector_store.collection,
                    "persist_path": s.vector_store.persist_path,
                },
            ),
            ComponentCard(
                name="Reranker",
                provider=s.rerank.provider,
                details={"max_candidates": s.rerank.max_candidates},
            ),
            ComponentCard(
                name="Evaluation",
                provider=s.evaluation.backend,
                details={},
            ),
        ]
        if s.vision_llm is not None:
            cards.append(
                ComponentCard(
                    name="Vision LLM",
                    provider=s.vision_llm.provider,
                    details={"model": s.vision_llm.model},
                )
            )
        return cards
