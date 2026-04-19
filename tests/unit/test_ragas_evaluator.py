"""Unit tests for the Ragas evaluator backend."""

from __future__ import annotations

import importlib
import re
from collections.abc import Iterator
from typing import Any

import pytest

from core.settings import EvaluationSettings
from libs.evaluator.ragas_evaluator import RagasEvaluator
from libs.evaluator.evaluator_factory import EvaluatorFactory


@pytest.fixture(autouse=True)
def reset_evaluator_registry() -> Iterator[None]:
    original = dict(EvaluatorFactory._providers)
    try:
        EvaluatorFactory._providers.clear()
        EvaluatorFactory._providers["ragas"] = RagasEvaluator
        yield
    finally:
        EvaluatorFactory._providers.clear()
        EvaluatorFactory._providers.update(original)


@pytest.mark.unit
def test_ragas_evaluator_returns_expected_metrics_with_injected_runner() -> None:
    def fake_runner(query: str, retrieved_ids: list[str], golden_ids: list[str]) -> dict[str, Any]:
        assert query == "what is rrf"
        assert retrieved_ids == ["chunk-a", "chunk-b"]
        assert golden_ids == ["chunk-b"]
        return {
            "faithfulness": 0.91,
            "answer_relevancy": 0.88,
            "context_precision": 0.79,
        }

    evaluator = RagasEvaluator(ragas_runner=fake_runner)

    metrics = evaluator.evaluate(
        query="what is rrf",
        retrieved_ids=["chunk-a", "chunk-b"],
        golden_ids=["chunk-b"],
    )

    assert metrics == {
        "faithfulness": 0.91,
        "answer_relevancy": 0.88,
        "context_precision": 0.79,
    }


@pytest.mark.unit
def test_ragas_evaluator_accepts_answer_relevance_alias() -> None:
    evaluator = RagasEvaluator(
        ragas_runner=lambda *_: {
            "faithfulness": 0.7,
            "answer_relevance": 0.8,
            "context_precision": 0.9,
        }
    )

    metrics = evaluator.evaluate(query="q", retrieved_ids=["a"], golden_ids=["a"])

    assert metrics == {
        "faithfulness": 0.7,
        "answer_relevancy": 0.8,
        "context_precision": 0.9,
    }


@pytest.mark.unit
def test_ragas_evaluator_raises_clear_import_error_when_optional_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_message = (
        "RagasEvaluator requires optional dependencies: ragas and datasets. "
        "Install them before using evaluation backend 'ragas'."
    )

    real_import_module = importlib.import_module

    def fake_import_module(name: str):
        if name == "ragas":
            raise ModuleNotFoundError("No module named 'ragas'")
        return real_import_module(name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)
    evaluator = RagasEvaluator()

    with pytest.raises(ImportError, match=re.escape(expected_message)):
        evaluator.evaluate(query="q", retrieved_ids=["a"], golden_ids=["a"])


@pytest.mark.unit
def test_factory_creates_ragas_backend() -> None:
    evaluator = EvaluatorFactory.create(EvaluationSettings(backend="ragas"))

    assert isinstance(evaluator, RagasEvaluator)
