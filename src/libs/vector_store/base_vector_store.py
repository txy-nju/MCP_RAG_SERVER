"""Base interfaces for pluggable vector store providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


@dataclass(slots=True)
class VectorStoreRecord:
    """Normalized payload written to a vector store backend."""

    id: str
    embedding: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    text: str | None = None


@dataclass(slots=True)
class VectorStoreQueryResult:
    """Normalized retrieval payload returned by a vector store backend."""

    id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    text: str | None = None


class BaseVectorStore(ABC):
    """Common contract for all vector store providers used by the project."""

    def __init__(self, *, provider: str, collection: str) -> None:
        """Initialize shared vector store configuration metadata."""

        self.provider = provider
        self.collection = collection

    @abstractmethod
    def upsert(self, records: list[VectorStoreRecord], trace: TraceContext | None = None) -> None:
        """Insert or update vector records in the underlying backend."""

    @abstractmethod
    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
        trace: TraceContext | None = None,
    ) -> list[VectorStoreQueryResult]:
        """Search for the nearest vector records matching the supplied query."""

    @abstractmethod
    def get_by_ids(
        self,
        ids: list[str],
        trace: TraceContext | None = None,
    ) -> list[VectorStoreQueryResult]:
        """Fetch stored records by id while preserving retrievable text and metadata."""

    @classmethod
    def from_settings(cls, settings: Any) -> "BaseVectorStore":
        """Build a vector store instance from a vector-store settings object."""

        return cls(provider=str(settings.provider), collection=str(settings.collection))
