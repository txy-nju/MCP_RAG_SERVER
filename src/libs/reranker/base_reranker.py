"""Base interfaces for pluggable reranker providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


@dataclass(slots=True)
class RerankCandidate:
    """Normalized candidate item passed through reranker backends."""

    id: str
    score: float
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseReranker(ABC):
    """Common contract for all reranker providers used by the project."""

    def __init__(self, *, provider: str, model: str | None = None) -> None:
        """Initialize shared provider metadata for a reranker instance."""

        self.provider = provider
        self.model = model

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: TraceContext | None = None,
    ) -> list[RerankCandidate]:
        """Return candidates reordered for the supplied query."""

    @classmethod
    def from_settings(cls, settings: Any) -> "BaseReranker":
        """Build a reranker instance from a rerank settings object."""

        model = getattr(settings, "model", None)
        return cls(provider=str(settings.provider), model=None if model is None else str(model))


class NoneReranker(BaseReranker):
    """Default reranker that preserves the incoming candidate order."""

    def __init__(self, *, provider: str = "none", model: str | None = None) -> None:
        super().__init__(provider=provider, model=model)

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: TraceContext | None = None,
    ) -> list[RerankCandidate]:
        """Return a shallow copy of the candidate list without reordering it."""

        return list(candidates)
