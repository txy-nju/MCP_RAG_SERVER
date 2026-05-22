"""E2E-style recall regression test based on golden test set."""

from __future__ import annotations

from pathlib import Path

import pytest

from modular_rag.core.settings import (
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
from modular_rag.core.types import RetrievalResult
from modular_rag.libs.evaluator.custom_evaluator import CustomEvaluator
from modular_rag.observability.evaluation import EvalRunner


class FakeHybridSearch:
    """Deterministic retrieval stub used for stable recall regression checks."""

    def __init__(self, results_by_query: dict[str, list[RetrievalResult]]) -> None:
        self.results_by_query = results_by_query

    def search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievalResult]:
        del top_k, filters
        return list(self.results_by_query.get(query, []))


def _make_settings() -> Settings:
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o-mini"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        splitter=SplitterSettings(provider="recursive", chunk_size=1000, chunk_overlap=200),
        vector_store=VectorStoreSettings(provider="chroma", collection="default", persist_path="data/db/chroma"),
        retrieval=RetrievalSettings(top_k=5),
        rerank=RerankSettings(provider="none"),
        evaluation=EvaluationSettings(backend="custom"),
        observability=ObservabilitySettings(log_level="INFO", trace_file="logs/traces.jsonl"),
    )


@pytest.mark.e2e
def test_recall_hit_rate_regression_threshold() -> None:
    """Ensure recall regression does not drop below minimum hit@k threshold."""

    settings = _make_settings()
    golden_set_path = Path("tests/fixtures/golden_test_set.json")

    # 3 cases in fixture: 2 hit + 1 miss => hit_rate = 0.666..., keeps threshold meaningful.
    fake_hybrid_search = FakeHybridSearch(
        {
            "如何配置 Azure OpenAI？": [
                RetrievalResult(
                    chunk_id="chunk_azure_config_001",
                    score=0.92,
                    text="Azure OpenAI setup steps",
                    metadata={"source_path": "config_guide.pdf"},
                )
            ],
            "如何启用混合检索？": [
                RetrievalResult(
                    chunk_id="chunk_hybrid_search_001",
                    score=0.88,
                    text="Hybrid search guide",
                    metadata={"source_path": "retrieval_guide.pdf"},
                )
            ],
            "如何查看 Query Trace？": [
                RetrievalResult(
                    chunk_id="chunk_unrelated_404",
                    score=0.45,
                    text="Unrelated chunk",
                    metadata={"source_path": "other_doc.md"},
                )
            ],
        }
    )

    runner = EvalRunner(
        settings=settings,
        hybrid_search=fake_hybrid_search,
        evaluator=CustomEvaluator(backend="custom"),
    )

    report = runner.run(golden_set_path)

    min_hit_rate_threshold = 0.60
    assert report.metrics["hit_rate"] >= min_hit_rate_threshold
