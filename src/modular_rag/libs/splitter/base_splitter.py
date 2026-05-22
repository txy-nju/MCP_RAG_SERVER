"""Base interfaces for pluggable text splitters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modular_rag.core.trace.trace_context import TraceContext


class BaseSplitter(ABC):
    """Common contract for all text splitting providers used by the project."""

    def __init__(self, *, provider: str, chunk_size: int, chunk_overlap: int) -> None:
        """Initialize shared splitter configuration metadata."""

        self.provider = provider
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @abstractmethod
    def split_text(self, text: str, trace: TraceContext | None = None) -> list[str]:
        """Split a text input into retrieval-ready chunks."""

    @classmethod
    def from_settings(cls, settings: Any) -> "BaseSplitter":
        """Build a splitter instance from a splitter settings object."""

        return cls(
            provider=str(settings.provider),
            chunk_size=int(settings.chunk_size),
            chunk_overlap=int(settings.chunk_overlap),
        )
