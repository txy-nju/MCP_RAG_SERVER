"""Base interfaces for pluggable evaluator providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from modular_rag.core.trace.trace_context import TraceContext


class BaseEvaluator(ABC):
    """Common contract for all evaluator providers used by the project."""

    def __init__(self, *, backend: str) -> None:
        """Initialize shared evaluator metadata.

        Args:
            backend: Normalized evaluation backend name, such as ``custom``.

        Returns:
            None. The constructor stores the backend name on the instance.
        """

        self.backend = backend

    @abstractmethod
    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: TraceContext | None = None,
    ) -> dict[str, float]:
        """Evaluate retrieval quality for one query.

        Args:
            query: User query associated with the retrieval result.
            retrieved_ids: Ordered retrieved chunk or document identifiers.
            golden_ids: Ordered or unordered golden identifiers expected to match.
            trace: Optional trace context used to record observability data.

        Returns:
            A metrics mapping produced by the evaluator backend.
        """

    @classmethod
    def from_settings(cls, settings: Any) -> "BaseEvaluator":
        """Build an evaluator instance from an evaluation settings object.

        Args:
            settings: Configuration object that provides at least a ``backend``
                attribute.

        Returns:
            A newly constructed evaluator instance initialized from the supplied
            settings object.
        """

        return cls(backend=str(settings.backend))
