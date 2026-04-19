"""Unit tests for EvalRunner."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.settings import (
    EmbeddingSettings,
    EvaluationSettings,
    LLMSettings,
    ObservabilitySettings,
    RetrievalSettings,
    RerankSettings,
    Settings,
    SplitterSettings,
    VectorStoreSettings,
)
from core.types import RetrievalResult
from libs.evaluator.custom_evaluator import CustomEvaluator
from observability.evaluation import EvalRunner


class FakeHybridSearch:
    def __init__(self, results_by_query: dict[str, list[RetrievalResult]]) -> None:
        self.results_by_query = results_by_query
        self.calls: list[dict[str, object]] = []

    def search(self, query: str, top_k: int, filters: dict[str, object] | None = None) -> list[RetrievalResult]:
        self.calls.append({"query": query, "top_k": top_k, "filters": filters})
        return list(self.results_by_query.get(query, []))


@pytest.fixture
def settings() -> Settings:
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o-mini"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        splitter=SplitterSettings(provider="recursive", chunk_size=1000, chunk_overlap=200),
        vector_store=VectorStoreSettings(provider="chroma", collection="default"),
        retrieval=RetrievalSettings(top_k=3),
        rerank=RerankSettings(provider="none"),
        evaluation=EvaluationSettings(backend="custom"),
        observability=ObservabilitySettings(log_level="INFO", trace_file="logs/traces.jsonl"),
    )


@pytest.fixture
def golden_test_set_file(tmp_path: Path) -> Path:
    test_set_path = tmp_path / "golden_test_set.json"
    test_set_path.write_text(
        json.dumps(
            {
                "test_cases": [
                    {
                        "query": "what is rrf",
                        "expected_chunk_ids": ["chunk-2"],
                        "expected_sources": ["rrf.md"],
                    },
                    {
                        "query": "azure setup",
                        "expected_chunk_ids": ["chunk-9"],
                        "expected_sources": ["azure.md"],
                    },
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return test_set_path


@pytest.mark.unit
def test_eval_runner_aggregates_metrics_and_case_details(
    settings: Settings,
    golden_test_set_file: Path,
) -> None:
    hybrid_search = FakeHybridSearch(
        {
            "what is rrf": [
                RetrievalResult(
                    chunk_id="chunk-2",
                    score=0.9,
                    text="rrf explanation",
                    metadata={"source_path": "rrf.md"},
                )
            ],
            "azure setup": [
                RetrievalResult(
                    chunk_id="chunk-1",
                    score=0.8,
                    text="azure intro",
                    metadata={"source_path": "azure.md"},
                )
            ],
        }
    )
    runner = EvalRunner(settings=settings, hybrid_search=hybrid_search, evaluator=CustomEvaluator(backend="custom"))

    report = runner.run(golden_test_set_file)

    assert report.metrics == {"hit_rate": 0.5, "mrr": 0.5}
    assert len(report.cases) == 2
    assert report.cases[0].retrieved_chunk_ids == ["chunk-2"]
    assert report.cases[1].retrieved_sources == ["azure.md"]
    assert hybrid_search.calls[0]["filters"] == {"collection": "default"}


@pytest.mark.unit
def test_eval_runner_rejects_empty_test_case_list(settings: Settings, tmp_path: Path) -> None:
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"test_cases": []}), encoding="utf-8")
    runner = EvalRunner(settings=settings, hybrid_search=FakeHybridSearch({}), evaluator=CustomEvaluator(backend="custom"))

    with pytest.raises(ValueError, match="non-empty 'test_cases' list"):
        runner.run(path)
