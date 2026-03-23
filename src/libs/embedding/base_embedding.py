"""Base interfaces for pluggable embedding providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class BaseEmbedding(ABC):
    """Common contract for all embedding providers used by the project."""

    def __init__(self, *, provider: str, model: str) -> None:
        """Initialize shared provider metadata for an embedding instance."""

        self.provider = provider
        self.model = model

    @abstractmethod
    def embed(self, texts: list[str], trace: TraceContext | None = None) -> list[list[float]]:
        """Convert a batch of text inputs into dense vectors."""

    @classmethod
    def from_settings(cls, settings: Any) -> "BaseEmbedding":
        """Build an embedding instance from an embedding settings object."""

        return cls(provider=str(settings.provider), model=str(settings.model))
