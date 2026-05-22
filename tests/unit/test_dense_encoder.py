"""Unit tests for DenseEncoder.

Validates:
1. Encoder output vector count matches input chunk count
2. Encoder output vector dimensions are consistent
3. ChunkRecords are properly created with dense vectors
4. DenseEncoder works with injected embedding instances (DI for testing)
5. Error handling for edge cases (empty chunks, etc.)
"""

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
from modular_rag.ingestion.embedding.dense_encoder import DenseEncoder
from modular_rag.libs.embedding.base_embedding import BaseEmbedding


# ---------------------------------------------------------------------------
# Test Fixtures & Helpers
# ---------------------------------------------------------------------------


class FakeEmbedding(BaseEmbedding):
    """Controllable fake embedding for testing."""

    def __init__(self, vector_dim: int = 2) -> None:
        super().__init__(provider="fake", model="test-model")
        self.vector_dim = vector_dim
        self.embed_calls: list[list[str]] = []

    def embed(self, texts: list[str], trace: TraceContext | None = None) -> list[list[float]]:
        """Return deterministic vectors: [text_index, text_length, ...pad to vector_dim]."""
        self.embed_calls.append(texts)
        vectors = []
        for idx, text in enumerate(texts):
            # Create vector with index and text length as first two dims
            vector = [float(idx), float(len(text))]
            # Pad to requested dimension
            while len(vector) < self.vector_dim:
                vector.append(0.0)
            vectors.append(vector[:self.vector_dim])
        return vectors


def _make_settings() -> Settings:
    """Build minimal Settings for testing."""
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
    """Create a test Chunk."""
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": "test.pdf"},
        start_offset=0,
        end_offset=len(text),
        source_ref={"doc_id": "doc-001"},
    )


# ---------------------------------------------------------------------------
# Tests: Basic Functionality
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_encoder_with_injected_embedding() -> None:
    """DenseEncoder should accept injected embedding for testing."""
    settings = _make_settings()
    fake_embedding = FakeEmbedding(vector_dim=3)
    encoder = DenseEncoder(settings, embedding=fake_embedding)

    assert encoder._embedding is fake_embedding
    assert encoder._embedding.vector_dim == 3


@pytest.mark.unit
def test_encode_single_chunk() -> None:
    """Encoding a single chunk should produce a ChunkRecord with dense vector."""
    settings = _make_settings()
    fake_embedding = FakeEmbedding(vector_dim=2)
    encoder = DenseEncoder(settings, embedding=fake_embedding)

    chunk = _make_chunk("chunk-001", "hello world")
    records = encoder.encode([chunk])

    assert len(records) == 1
    record = records[0]
    assert isinstance(record, ChunkRecord)
    assert record.id == "chunk-001"
    assert record.text == "hello world"
    assert record.dense_vector is not None
    assert len(record.dense_vector) == 2
    # First value should be 0.0 (index), second should be 11 (text length)
    assert record.dense_vector == [0.0, 11.0]


@pytest.mark.unit
def test_encode_multiple_chunks() -> None:
    """Encoding multiple chunks should produce records with matching count and dimensions."""
    settings = _make_settings()
    fake_embedding = FakeEmbedding(vector_dim=3)
    encoder = DenseEncoder(settings, embedding=fake_embedding)

    chunks = [
        _make_chunk("chunk-001", "hello"),
        _make_chunk("chunk-002", "world test"),
        _make_chunk("chunk-003", "x"),
    ]
    records = encoder.encode(chunks)

    assert len(records) == 3
    # Check each record has consistent structure
    for idx, record in enumerate(records):
        assert isinstance(record, ChunkRecord)
        assert record.id == chunks[idx].id
        assert record.text == chunks[idx].text
        assert record.dense_vector is not None
        assert len(record.dense_vector) == 3
        # Validate specific vector values
        assert record.dense_vector[0] == float(idx)  # index
        assert record.dense_vector[1] == float(len(chunks[idx].text))  # text length


@pytest.mark.unit
def test_encode_preserves_chunk_metadata() -> None:
    """Encoding should preserve chunk metadata in the resulting ChunkRecord."""
    settings = _make_settings()
    fake_embedding = FakeEmbedding(vector_dim=2)
    encoder = DenseEncoder(settings, embedding=fake_embedding)

    chunk = Chunk(
        id="chunk-001",
        text="test content",
        metadata={"source_path": "docs/file.pdf", "chunk_index": 5, "custom_field": "value"},
        start_offset=10,
        end_offset=22,
        source_ref={"doc_id": "doc-001", "page": 1},
    )
    records = encoder.encode([chunk])

    assert len(records) == 1
    record = records[0]
    assert record.metadata["source_path"] == "docs/file.pdf"
    assert record.metadata["chunk_index"] == 5
    assert record.metadata["custom_field"] == "value"
    assert record.sparse_vector is None  # Not set by DenseEncoder


@pytest.mark.unit
def test_encode_with_trace_context() -> None:
    """Encoder should pass trace context to embedding.embed()."""
    settings = _make_settings()
    fake_embedding = FakeEmbedding(vector_dim=2)
    encoder = DenseEncoder(settings, embedding=fake_embedding)

    trace = TraceContext(trace_id="trace-001", trace_type="query.test")
    chunks = [_make_chunk("chunk-001", "hello")]
    records = encoder.encode(chunks, trace=trace)

    assert len(records) == 1
    # Verify embedding was called (we don't directly check trace propagation,
    # but at minimum the operation should succeed)
    assert len(fake_embedding.embed_calls) == 1


# ---------------------------------------------------------------------------
# Tests: Error Handling
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_encode_rejects_empty_chunks() -> None:
    """Encoder should raise ValueError for empty chunks list."""
    settings = _make_settings()
    fake_embedding = FakeEmbedding(vector_dim=2)
    encoder = DenseEncoder(settings, embedding=fake_embedding)

    with pytest.raises(ValueError, match="Cannot encode an empty chunks list"):
        encoder.encode([])


@pytest.mark.unit
def test_encode_detects_embedding_mismatch() -> None:
    """Encoder should detect if embedding output count doesn't match input count."""
    settings = _make_settings()

    # Create a broken embedding that returns wrong number of vectors
    class BrokenEmbedding(BaseEmbedding):
        def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
            # Return one less vector than input texts
            return [[1.0, 2.0] for _ in texts[:-1]]

    encoder = DenseEncoder(settings, embedding=BrokenEmbedding(provider="broken", model="test"))
    chunks = [_make_chunk("chunk-001", "hello"), _make_chunk("chunk-002", "world")]

    with pytest.raises(ValueError, match="Embedding output count .* does not match input chunk count"):
        encoder.encode(chunks)


# ---------------------------------------------------------------------------
# Tests: Vector Dimension Consistency
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_encode_consistent_vector_dimensions() -> None:
    """All output vectors should have same dimension."""
    settings = _make_settings()
    fake_embedding = FakeEmbedding(vector_dim=5)
    encoder = DenseEncoder(settings, embedding=fake_embedding)

    chunks = [
        _make_chunk("c-1", "short"),
        _make_chunk("c-2", "a much longer piece of text here"),
        _make_chunk("c-3", "x"),
    ]
    records = encoder.encode(chunks)

    dimensions = [len(r.dense_vector) for r in records]
    assert all(dim == 5 for dim in dimensions)
    assert dimensions == [5, 5, 5]


# ---------------------------------------------------------------------------
# Tests: Call Tracking (for debugging/verification)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_encoder_tracks_embedding_calls() -> None:
    """FakeEmbedding should track all embed() calls for verification."""
    settings = _make_settings()
    fake_embedding = FakeEmbedding(vector_dim=2)
    encoder = DenseEncoder(settings, embedding=fake_embedding)

    chunks = [_make_chunk("c-1", "hello"), _make_chunk("c-2", "world")]
    records = encoder.encode(chunks)

    assert len(fake_embedding.embed_calls) == 1
    assert fake_embedding.embed_calls[0] == ["hello", "world"]
    assert len(records) == 2
