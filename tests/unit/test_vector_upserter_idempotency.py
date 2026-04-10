"""Unit tests for VectorUpserter idempotent behavior (C12)."""

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
from core.types import ChunkRecord
from ingestion.storage.vector_upserter import VectorUpserter
from libs.vector_store.base_vector_store import BaseVectorStore, VectorStoreRecord


class FakeVectorStore(BaseVectorStore):
    """In-memory vector store fake used by unit tests."""

    def __init__(self) -> None:
        super().__init__(provider="fake", collection="test")
        self.last_upsert: list[VectorStoreRecord] = []

    def upsert(self, records: list[VectorStoreRecord], trace: object | None = None) -> None:
        del trace
        self.last_upsert = list(records)

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, object] | None = None,
        trace: object | None = None,
    ) -> list[object]:
        del vector, top_k, filters, trace
        return []


def _make_settings() -> Settings:
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        splitter=SplitterSettings(provider="recursive", chunk_size=1000, chunk_overlap=200),
        vector_store=VectorStoreSettings(provider="chroma", collection="test", persist_path="data/db/chroma"),
        retrieval=RetrievalSettings(top_k=5),
        rerank=RerankSettings(provider="none"),
        evaluation=EvaluationSettings(backend="custom"),
        observability=ObservabilitySettings(log_level="INFO", trace_file="logs/trace.jsonl"),
    )


def _record(*, text: str, chunk_index: int = 0) -> ChunkRecord:
    return ChunkRecord(
        id="temp-id",
        text=text,
        metadata={"source_path": "docs/a.pdf", "chunk_index": chunk_index},
        dense_vector=[0.1, 0.2, 0.3],
    )


@pytest.mark.unit
def test_same_chunk_generates_same_id_across_repeated_upserts() -> None:
    store = FakeVectorStore()
    upserter = VectorUpserter(_make_settings(), vector_store=store)

    ids_first = upserter.upsert([_record(text="same content", chunk_index=1)])
    ids_second = upserter.upsert([_record(text="same content", chunk_index=1)])

    assert ids_first == ids_second
    assert len(ids_first) == 1
    assert store.last_upsert[0].id == ids_first[0]


@pytest.mark.unit
def test_content_change_produces_different_id() -> None:
    store = FakeVectorStore()
    upserter = VectorUpserter(_make_settings(), vector_store=store)

    old_id = upserter.upsert([_record(text="before", chunk_index=2)])[0]
    new_id = upserter.upsert([_record(text="after", chunk_index=2)])[0]

    assert old_id != new_id


@pytest.mark.unit
def test_batch_upsert_preserves_input_order() -> None:
    store = FakeVectorStore()
    upserter = VectorUpserter(_make_settings(), vector_store=store)

    records = [
        _record(text="alpha", chunk_index=0),
        _record(text="beta", chunk_index=1),
        _record(text="gamma", chunk_index=2),
    ]
    ids = upserter.upsert(records)

    assert len(ids) == 3
    assert [item.id for item in store.last_upsert] == ids
    assert [item.text for item in store.last_upsert] == ["alpha", "beta", "gamma"]


@pytest.mark.unit
def test_missing_dense_vector_raises_value_error() -> None:
    store = FakeVectorStore()
    upserter = VectorUpserter(_make_settings(), vector_store=store)

    bad = ChunkRecord(
        id="temp-id",
        text="missing vector",
        metadata={"source_path": "docs/a.pdf", "chunk_index": 0},
        dense_vector=None,
    )

    with pytest.raises(ValueError, match="missing dense_vector"):
        upserter.upsert([bad])
