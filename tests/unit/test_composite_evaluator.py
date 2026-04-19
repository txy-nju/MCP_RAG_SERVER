"""Unit tests for the composite evaluator backend."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from core.settings import EvaluationSettings
from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.evaluator_factory import EvaluatorFactory
from observability.evaluation.composite_evaluator import CompositeEvaluator


class HitRateEvaluator(BaseEvaluator):
    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: object | None = None,
    ) -> dict[str, float]:
        return {"hit_rate": 1.0 if retrieved_ids and golden_ids and retrieved_ids[0] == golden_ids[0] else 0.0}


class FaithfulnessEvaluator(BaseEvaluator):
    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: object | None = None,
    ) -> dict[str, float]:
        return {"faithfulness": 0.93}


class DuplicateMetricEvaluator(BaseEvaluator):
    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: object | None = None,
    ) -> dict[str, float]:
        return {"hit_rate": 0.2}


@pytest.fixture(autouse=True)
def reset_evaluator_registry() -> Iterator[None]:
    original = dict(EvaluatorFactory._providers)
    try:
        EvaluatorFactory._providers.clear()
        EvaluatorFactory._providers.update(
            {
                "hit": HitRateEvaluator,
                "faith": FaithfulnessEvaluator,
            }
        )
        yield
    finally:
        EvaluatorFactory._providers.clear()
        EvaluatorFactory._providers.update(original)


@pytest.mark.unit
def test_composite_evaluator_merges_metrics_from_multiple_evaluators() -> None:
    evaluator = CompositeEvaluator(
        [
            HitRateEvaluator(backend="hit"),
            FaithfulnessEvaluator(backend="faith"),
        ]
    )

    metrics = evaluator.evaluate(
        query="what is rrf",
        retrieved_ids=["chunk-a"],
        golden_ids=["chunk-a"],
    )

    assert metrics == {"hit_rate": 1.0, "faithfulness": 0.93}


@pytest.mark.unit
def test_composite_evaluator_rejects_duplicate_metric_names() -> None:
    evaluator = CompositeEvaluator(
        [
            HitRateEvaluator(backend="hit"),
            DuplicateMetricEvaluator(backend="dup"),
        ]
    )

    with pytest.raises(ValueError, match="duplicate metric keys: hit_rate"):
        evaluator.evaluate(query="q", retrieved_ids=["a"], golden_ids=["a"])


@pytest.mark.unit
def test_factory_creates_composite_evaluator_from_multiple_backends() -> None:
    evaluator = EvaluatorFactory.create(EvaluationSettings(backend="hit", backends=("hit", "faith")))

    assert isinstance(evaluator, CompositeEvaluator)
    metrics = evaluator.evaluate(query="q", retrieved_ids=["a"], golden_ids=["a"])
    assert metrics == {"hit_rate": 1.0, "faithfulness": 0.93}


@pytest.mark.unit
def test_composite_evaluator_requires_at_least_one_child() -> None:
    with pytest.raises(ValueError, match="requires at least one evaluator"):
        CompositeEvaluator([])
