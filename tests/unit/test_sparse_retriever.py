"""Unit tests for SparseRetriever (D3)."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.query_engine.sparse_retriever import SparseRetriever
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
from libs.vector_store.base_vector_store import VectorStoreQueryResult


class FakeBM25Indexer:
    def __init__(self, hits: list[dict[str, float | str]] | None = None) -> None:
        self.hits = hits or []
        self.calls: list[dict[str, object]] = []

    def query(self, keywords: list[str], top_k: int = 10) -> list[dict[str, float | str]]:
        self.calls.append({"keywords": keywords, "top_k": top_k})
        return list(self.hits)


class FakeVectorStore:
    def __init__(self, payloads: list[VectorStoreQueryResult] | None = None) -> None:
        self.payloads = payloads or []
        self.calls: list[dict[str, object]] = []

    def get_by_ids(self, ids: list[str], trace=None) -> list[VectorStoreQueryResult]:
        self.calls.append({"ids": ids, "trace": trace})
        by_id = {payload.id: payload for payload in self.payloads}
        return [by_id[record_id] for record_id in ids if record_id in by_id]


def _make_settings() -> Settings:
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o-mini"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        splitter=SplitterSettings(provider="recursive", chunk_size=1000, chunk_overlap=200),
        vector_store=VectorStoreSettings(provider="chroma", collection="demo", persist_path="data/db/chroma"),
        retrieval=RetrievalSettings(top_k=5),
        rerank=RerankSettings(provider="none"),
        evaluation=EvaluationSettings(backend="custom"),
        observability=ObservabilitySettings(log_level="INFO", trace_file="logs/traces.jsonl"),
    )


@pytest.mark.unit
def test_sparse_retriever_queries_bm25_and_fetches_payloads_by_chunk_id() -> None:
    bm25 = FakeBM25Indexer(
        hits=[
            {"chunk_id": "chunk-2", "score": 2.1},
            {"chunk_id": "chunk-1", "score": 1.7},
        ]
    )
    vector_store = FakeVectorStore(
        payloads=[
            VectorStoreQueryResult(id="chunk-1", score=0.0, text="first text", metadata={"source_path": "a.pdf"}),
            VectorStoreQueryResult(id="chunk-2", score=0.0, text="second text", metadata={"source_path": "b.pdf"}),
        ]
    )
    retriever = SparseRetriever(_make_settings(), bm25_indexer=bm25, vector_store=vector_store)

    trace = object()
    results = retriever.retrieve(["Azure", "Config"], top_k=2, trace=trace)

    assert bm25.calls == [{"keywords": ["azure", "config"], "top_k": 2}]
    assert vector_store.calls == [{"ids": ["chunk-2", "chunk-1"], "trace": trace}]
    assert [result.chunk_id for result in results] == ["chunk-2", "chunk-1"]
    assert [result.score for result in results] == [2.1, 1.7]
    assert results[0].text == "second text"
    assert results[1].metadata["source_path"] == "a.pdf"


@pytest.mark.unit
def test_sparse_retriever_skips_hits_missing_payloads() -> None:
    bm25 = FakeBM25Indexer(hits=[{"chunk_id": "missing", "score": 0.8}])
    retriever = SparseRetriever(_make_settings(), bm25_indexer=bm25, vector_store=FakeVectorStore(payloads=[]))

    results = retriever.retrieve(["missing"], top_k=5)

    assert results == []


@pytest.mark.unit
def test_sparse_retriever_returns_empty_for_empty_keywords() -> None:
    retriever = SparseRetriever(
        _make_settings(),
        bm25_indexer=FakeBM25Indexer(hits=[{"chunk_id": "chunk-1", "score": 1.0}]),
        vector_store=FakeVectorStore(),
    )

    assert retriever.retrieve([], top_k=5) == []
    assert retriever.retrieve(["   ", ""], top_k=5) == []


@pytest.mark.unit
def test_sparse_retriever_rejects_non_positive_top_k() -> None:
    retriever = SparseRetriever(_make_settings(), bm25_indexer=FakeBM25Indexer(), vector_store=FakeVectorStore())

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        retriever.retrieve(["azure"], top_k=0)
