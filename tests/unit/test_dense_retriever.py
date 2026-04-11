"""Unit tests for DenseRetriever (D2)."""

from __future__ import annotations

import pytest

from core.query_engine.dense_retriever import DenseRetriever
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


class FakeEmbedding:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def embed(self, texts: list[str], trace=None) -> list[list[float]]:
        self.calls.append({"texts": texts, "trace": trace})
        return [[0.11, 0.22, 0.33]]


class FakeVectorStore:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def query(self, vector: list[float], top_k: int, filters=None, trace=None):
        self.calls.append({"vector": vector, "top_k": top_k, "filters": filters, "trace": trace})
        return [
            VectorStoreQueryResult(
                id="chunk-1",
                score=0.91,
                text="dense retrieval text",
                metadata={"source_path": "doc.pdf", "collection": "demo"},
            )
        ]


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
def test_dense_retriever_calls_embedding_and_vector_store_in_order() -> None:
    embedding = FakeEmbedding()
    vector_store = FakeVectorStore()
    retriever = DenseRetriever(_make_settings(), embedding_client=embedding, vector_store=vector_store)

    trace = object()
    results = retriever.retrieve("find ingestion pipeline", top_k=3, filters={"collection": "demo"}, trace=trace)

    assert embedding.calls[0]["texts"] == ["find ingestion pipeline"]
    assert embedding.calls[0]["trace"] is trace

    assert vector_store.calls[0]["vector"] == [0.11, 0.22, 0.33]
    assert vector_store.calls[0]["top_k"] == 3
    assert vector_store.calls[0]["filters"] == {"collection": "demo"}
    assert vector_store.calls[0]["trace"] is trace

    assert len(results) == 1
    assert results[0].chunk_id == "chunk-1"
    assert results[0].score == 0.91
    assert results[0].text == "dense retrieval text"
    assert results[0].metadata["source_path"] == "doc.pdf"


@pytest.mark.unit
def test_dense_retriever_accepts_mapping_results_from_vector_store() -> None:
    embedding = FakeEmbedding()

    class MappingVectorStore(FakeVectorStore):
        def query(self, vector, top_k, filters=None, trace=None):
            return [
                {
                    "id": "chunk-2",
                    "score": 0.52,
                    "text": "mapped result",
                    "metadata": {"source_path": "mapped.pdf"},
                }
            ]

    retriever = DenseRetriever(_make_settings(), embedding_client=embedding, vector_store=MappingVectorStore())

    results = retriever.retrieve("mapped", top_k=1)

    assert results[0].chunk_id == "chunk-2"
    assert results[0].text == "mapped result"
    assert results[0].metadata["source_path"] == "mapped.pdf"


@pytest.mark.unit
def test_dense_retriever_rejects_empty_query() -> None:
    retriever = DenseRetriever(_make_settings(), embedding_client=FakeEmbedding(), vector_store=FakeVectorStore())

    with pytest.raises(ValueError, match="query must not be empty"):
        retriever.retrieve("   ", top_k=1)


@pytest.mark.unit
def test_dense_retriever_rejects_non_positive_top_k() -> None:
    retriever = DenseRetriever(_make_settings(), embedding_client=FakeEmbedding(), vector_store=FakeVectorStore())

    with pytest.raises(ValueError, match="top_k must be greater than 0"):
        retriever.retrieve("query", top_k=0)
