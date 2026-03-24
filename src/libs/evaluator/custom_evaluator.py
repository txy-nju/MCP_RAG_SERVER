"""Lightweight evaluator implementation for retrieval metrics."""

from __future__ import annotations

from libs.evaluator.base_evaluator import BaseEvaluator


class CustomEvaluator(BaseEvaluator):
    """Compute lightweight retrieval metrics such as hit rate and MRR."""

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: object | None = None,
    ) -> dict[str, float]:
        """Evaluate retrieval quality for one query using lightweight metrics.

        Args:
            query: User query associated with the retrieval result. Accepted for
                interface consistency and currently not used in metric formulas.
            retrieved_ids: Ordered identifiers returned by the retrieval pipeline.
            golden_ids: Relevant identifiers expected for the query.
            trace: Optional trace context accepted for interface compatibility.

        Returns:
            A metrics dictionary containing at least ``hit_rate`` and ``mrr``.
        """

        if not golden_ids:
            raise ValueError("golden_ids must contain at least one expected identifier")

        golden_set = set(golden_ids)
        hit_rate = 1.0 if any(identifier in golden_set for identifier in retrieved_ids) else 0.0

        reciprocal_rank = 0.0
        for index, identifier in enumerate(retrieved_ids, start=1):
            if identifier in golden_set:
                reciprocal_rank = 1.0 / float(index)
                break

        return {
            "hit_rate": hit_rate,
            "mrr": reciprocal_rank,
        }
