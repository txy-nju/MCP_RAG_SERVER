"""Unit tests for the lightweight custom evaluator and factory."""

from __future__ import annotations

import pytest

from core.settings import (
    EmbeddingSettings,
    EvaluationSettings,
    LLMSettings,
    ObservabilitySettings,
    RerankSettings,
    RetrievalSettings,
    Settings,
    SplitterSettings,
    VectorStoreSettings,
)
from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator
from libs.evaluator.evaluator_factory import EvaluatorFactory


class FakeEvaluator(BaseEvaluator):
    """Simple fake evaluator used to verify factory routing."""

    def evaluate(
        self,
        query: str,
        retrieved_ids: list[str],
        golden_ids: list[str],
        trace: object | None = None,
    ) -> dict[str, float]:
        return {"query_length": float(len(query)), "retrieved_count": float(len(retrieved_ids))}


def make_settings(backend: str = "custom") -> Settings:
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o-mini"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        splitter=SplitterSettings(provider="recursive", chunk_size=1000, chunk_overlap=200),
        vector_store=VectorStoreSettings(provider="chroma", collection="default"),
        retrieval=RetrievalSettings(top_k=5),
        rerank=RerankSettings(provider="none"),
        evaluation=EvaluationSettings(backend=backend),
        observability=ObservabilitySettings(log_level="INFO", trace_file="logs/traces.jsonl"),
    )


@pytest.fixture(autouse=True)
def reset_evaluator_registry() -> None:
    original = dict(EvaluatorFactory._providers)
    try:
        EvaluatorFactory._providers.clear()
        EvaluatorFactory._providers["custom"] = CustomEvaluator
        yield
    finally:
        EvaluatorFactory._providers.clear()
        EvaluatorFactory._providers.update(original)


@pytest.mark.unit
def test_custom_evaluator_returns_hit_rate_and_mrr_for_first_relevant_match() -> None:
    evaluator = CustomEvaluator(backend="custom")

    metrics = evaluator.evaluate(
        query="how to configure azure openai",
        retrieved_ids=["chunk-1", "chunk-2", "chunk-3"],
        golden_ids=["chunk-2", "chunk-9"],
    )

    assert metrics == {"hit_rate": 1.0, "mrr": 0.5}


@pytest.mark.unit
def test_custom_evaluator_returns_zero_metrics_when_no_relevant_match_is_found() -> None:
    evaluator = CustomEvaluator(backend="custom")

    metrics = evaluator.evaluate(
        query="missing",
        retrieved_ids=["chunk-1", "chunk-2"],
        golden_ids=["chunk-9"],
    )

    assert metrics == {"hit_rate": 0.0, "mrr": 0.0}


@pytest.mark.unit
def test_custom_evaluator_rejects_empty_golden_ids() -> None:
    evaluator = CustomEvaluator(backend="custom")

    with pytest.raises(ValueError, match="golden_ids must contain"):
        evaluator.evaluate(query="q", retrieved_ids=["chunk-1"], golden_ids=[])


@pytest.mark.unit
def test_factory_routes_backend_from_top_level_settings() -> None:
    evaluator = EvaluatorFactory.create(make_settings())

    assert isinstance(evaluator, CustomEvaluator)
    assert evaluator.evaluate("q", ["a"], ["a"]) == {"hit_rate": 1.0, "mrr": 1.0}


@pytest.mark.unit
def test_factory_accepts_evaluation_settings_directly() -> None:
    EvaluatorFactory.register("fake", FakeEvaluator)

    evaluator = EvaluatorFactory.create(EvaluationSettings(backend="fake"))

    assert isinstance(evaluator, FakeEvaluator)
    assert evaluator.backend == "fake"


@pytest.mark.unit
def test_factory_reports_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unsupported evaluator backend: missing"):
        EvaluatorFactory.create(make_settings(backend="missing"))


@pytest.mark.unit
def test_register_rejects_non_evaluator_classes() -> None:
    with pytest.raises(TypeError, match="must inherit from BaseEvaluator"):
        EvaluatorFactory.register("bad", object)
