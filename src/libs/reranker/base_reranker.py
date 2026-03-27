"""Base interfaces for pluggable reranker providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


@dataclass(slots=True)
class RerankCandidate:
    """Normalized candidate item passed through reranker backends.

    Args:
        id: Unique identifier of the candidate item.
        score: Original retrieval or ranking score associated with the candidate.
        text: Optional candidate text content used by reranker implementations.
        metadata: Optional structured metadata carried alongside the candidate.

    Returns:
        A lightweight dataclass instance representing one rerankable candidate.
    """

    id: str
    score: float
    text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RerankerFallbackError(RuntimeError):
    """Signal that the caller should fall back to the original candidate order."""


class BaseReranker(ABC):
    """Common contract for all reranker providers used by the project.

    Returns:
        A base type that defines the shared constructor metadata and rerank
        interface for concrete reranker implementations.
    """

    def __init__(self, *, provider: str, model: str | None = None) -> None:
        """Initialize shared provider metadata for a reranker instance.

        Args:
            provider: Normalized reranker backend name, such as ``none`` or a
                future provider like ``llm``.
            model: Optional model identifier used by the selected backend.

        Returns:
            None. The constructor stores common reranker metadata on the instance.
        """

        self.provider = provider
        self.model = model

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: TraceContext | None = None,
    ) -> list[RerankCandidate]:
        """Return candidates reordered for the supplied query.

        Args:
            query: User query or rewritten query used to score the candidates.
            candidates: Ordered candidate list produced by the retrieval stage.
            trace: Optional trace context used to record observability data.

        Returns:
            A candidate list ordered according to the backend-specific reranking
            strategy.
        """

    @classmethod
    def from_settings(cls, settings: Any) -> "BaseReranker":
        """Build a reranker instance from a rerank settings object.

        Args:
            settings: Configuration object that provides at least a ``provider``
                attribute and may optionally expose a ``model`` attribute.

        Returns:
            A newly constructed reranker instance initialized from the supplied
            settings object.
        """

        model = settings.model if hasattr(settings, "model") else None
        return cls(provider=str(settings.provider), model=None if model is None else str(model))


class NoneReranker(BaseReranker):
    """Default reranker that preserves the incoming candidate order.

    Returns:
        A concrete reranker implementation that acts as a no-op fallback when
        reranking is disabled.
    """

    def __init__(self, *, provider: str = "none", model: str | None = None) -> None:
        """Initialize the no-op reranker implementation.

        Args:
            provider: Backend name for the no-op reranker. Defaults to ``none``.
            model: Optional model name. Unused for the no-op backend but kept for
                interface consistency.

        Returns:
            None. The constructor delegates shared initialization to
            ``BaseReranker``.
        """

        super().__init__(provider=provider, model=model)

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: TraceContext | None = None,
    ) -> list[RerankCandidate]:
        """Return a shallow copy of the candidate list without reordering it.

        Args:
            query: User query associated with the candidates. Accepted for
                interface compatibility and ignored by this implementation.
            candidates: Candidate list that should keep its current order.
            trace: Optional trace context accepted for interface compatibility.

        Returns:
            A new list containing the same candidate items in their original
            order.
        """

        return list(candidates)
