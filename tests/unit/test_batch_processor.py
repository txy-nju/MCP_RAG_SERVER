"""Unit tests for BatchProcessor (C10)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from modular_rag.core.settings import (
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
from modular_rag.core.trace.trace_context import TraceContext
from modular_rag.core.types import Chunk, ChunkRecord
from modular_rag.ingestion.embedding.batch_processor import BatchProcessor


class FakeDenseEncoder:
    """Test double for DenseEncoder with deterministic output."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(
        self, chunks: list[Chunk], trace: TraceContext | None = None
    ) -> list[ChunkRecord]:
        self.calls.append([chunk.id for chunk in chunks])
        return [
            ChunkRecord.from_chunk(chunk, dense_vector=[float(index), 1.0])
            for index, chunk in enumerate(chunks)
        ]


class FakeSparseEncoder:
    """Test double for SparseEncoder with deterministic output."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def encode(
        self, chunks: list[Chunk], trace: TraceContext | None = None
    ) -> list[ChunkRecord]:
        self.calls.append([chunk.id for chunk in chunks])
        return [
            ChunkRecord.from_chunk(chunk, sparse_vector={"token": float(index + 1)})
            for index, chunk in enumerate(chunks)
        ]


def _make_settings() -> Settings:
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        splitter=SplitterSettings(provider="recursive", chunk_size=1000, chunk_overlap=200),
        vector_store=VectorStoreSettings(provider="chroma", collection="test"),
        retrieval=RetrievalSettings(top_k=5),
        rerank=RerankSettings(provider="none"),
        evaluation=EvaluationSettings(backend="custom"),
        observability=ObservabilitySettings(log_level="INFO", trace_file="logs/trace.jsonl"),
    )


def _make_chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": "test.pdf"},
        start_offset=0,
        end_offset=len(text),
        source_ref={"doc_id": "doc-001"},
    )


@pytest.mark.unit
def test_batch_size_two_on_five_chunks_splits_into_three_batches() -> None:
    dense = FakeDenseEncoder()
    sparse = FakeSparseEncoder()
    processor = BatchProcessor(_make_settings(), dense_encoder=dense, sparse_encoder=sparse)
    chunks = [_make_chunk(f"c{i}", f"text {i}") for i in range(5)]

    records = processor.process(chunks, batch_size=2)

    assert len(records) == 5
    assert dense.calls == [["c0", "c1"], ["c2", "c3"], ["c4"]]
    assert sparse.calls == [["c0", "c1"], ["c2", "c3"], ["c4"]]


@pytest.mark.unit
def test_order_is_stable_across_batches() -> None:
    dense = FakeDenseEncoder()
    sparse = FakeSparseEncoder()
    processor = BatchProcessor(_make_settings(), dense_encoder=dense, sparse_encoder=sparse)
    chunks = [_make_chunk(f"id-{i}", f"content-{i}") for i in range(5)]

    records = processor.process(chunks, batch_size=2)

    assert [record.id for record in records] == [chunk.id for chunk in chunks]


@pytest.mark.unit
def test_merged_records_contain_both_dense_and_sparse_vectors() -> None:
    processor = BatchProcessor(
        _make_settings(), dense_encoder=FakeDenseEncoder(), sparse_encoder=FakeSparseEncoder()
    )
    chunks = [_make_chunk("c1", "hello world")]

    records = processor.process(chunks, batch_size=1)

    assert records[0].dense_vector is not None
    assert records[0].sparse_vector is not None


@pytest.mark.unit
def test_empty_chunks_returns_empty_list_without_encoder_calls() -> None:
    dense = FakeDenseEncoder()
    sparse = FakeSparseEncoder()
    processor = BatchProcessor(_make_settings(), dense_encoder=dense, sparse_encoder=sparse)

    records = processor.process([], batch_size=2)

    assert records == []
    assert dense.calls == []
    assert sparse.calls == []


@pytest.mark.unit
def test_batch_size_must_be_positive() -> None:
    processor = BatchProcessor(
        _make_settings(), dense_encoder=FakeDenseEncoder(), sparse_encoder=FakeSparseEncoder()
    )
    chunks = [_make_chunk("c1", "hello")]

    with pytest.raises(ValueError, match="batch_size"):
        processor.process(chunks, batch_size=0)


@pytest.mark.unit
def test_trace_records_one_stage_per_batch() -> None:
    processor = BatchProcessor(
        _make_settings(), dense_encoder=FakeDenseEncoder(), sparse_encoder=FakeSparseEncoder()
    )
    trace = TraceContext()
    chunks = [_make_chunk(f"c{i}", f"text {i}") for i in range(5)]

    processor.process(chunks, batch_size=2, trace=trace)

    stages = [stage for stage in trace.stages if stage["stage"] == "embedding_batch"]
    assert len(stages) == 3
    assert stages[0]["data"]["batch_index"] == 1
    assert stages[2]["data"]["total_batches"] == 3


@pytest.mark.unit
def test_merge_raises_when_count_mismatch() -> None:
    dense_records = [
        ChunkRecord(
            id="a",
            text="x",
            metadata={"source_path": "test.pdf"},
            dense_vector=[1.0],
        )
    ]
    sparse_records: list[ChunkRecord] = []

    with pytest.raises(ValueError, match="counts must match"):
        BatchProcessor._merge_records(dense_records, sparse_records)


@pytest.mark.unit
def test_merge_raises_when_id_mismatch() -> None:
    dense_records = [
        ChunkRecord(
            id="dense-id",
            text="x",
            metadata={"source_path": "test.pdf"},
            dense_vector=[1.0],
        )
    ]
    sparse_records = [
        ChunkRecord(
            id="sparse-id",
            text="x",
            metadata={"source_path": "test.pdf"},
            sparse_vector={"x": 1.0},
        )
    ]

    with pytest.raises(ValueError, match="IDs must align"):
        BatchProcessor._merge_records(dense_records, sparse_records)


@pytest.mark.unit
def test_trace_can_be_mock_object() -> None:
    processor = BatchProcessor(
        _make_settings(), dense_encoder=FakeDenseEncoder(), sparse_encoder=FakeSparseEncoder()
    )
    trace = MagicMock()
    chunks = [_make_chunk("c1", "test")]

    processor.process(chunks, batch_size=1, trace=trace)

    trace.record_stage.assert_called_once()
