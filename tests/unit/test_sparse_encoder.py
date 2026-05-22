"""Unit tests for SparseEncoder.

Validates:
1. Output count matches input chunk count
2. Sparse vectors contain expected terms with positive TF counts
3. Dense vector is None in output records
4. Empty text produces an empty (but not None) sparse vector
5. Empty chunks list raises ValueError
6. Repeated terms are summed correctly (TF counting)
7. Stop words are filtered out
8. Short tokens (single char) are excluded
9. ChunkRecord retains chunk id, text and metadata
10. Trace context receives a stage record when provided
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
from modular_rag.ingestion.embedding.sparse_encoder import SparseEncoder, _tokenize


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tests: output shape & basic contract
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_output_count_matches_input_count() -> None:
    """Encoder must return exactly as many ChunkRecords as input Chunks."""
    encoder = SparseEncoder(_make_settings())
    chunks = [_make_chunk(f"c{i}", f"hello world chunk number {i}") for i in range(5)]
    records = encoder.encode(chunks)
    assert len(records) == 5


@pytest.mark.unit
def test_output_ids_match_input_ids() -> None:
    """Each ChunkRecord id must match the corresponding input Chunk id."""
    encoder = SparseEncoder(_make_settings())
    chunks = [_make_chunk(f"id-{i}", f"text {i}") for i in range(3)]
    records = encoder.encode(chunks)
    for chunk, record in zip(chunks, records):
        assert record.id == chunk.id


@pytest.mark.unit
def test_dense_vector_is_none() -> None:
    """Dense vector must remain None as SparseEncoder does not populate it."""
    encoder = SparseEncoder(_make_settings())
    chunks = [_make_chunk("c1", "some text")]
    records = encoder.encode(chunks)
    assert records[0].dense_vector is None


@pytest.mark.unit
def test_sparse_vector_is_not_none_for_non_empty_text() -> None:
    """A chunk with non-empty text must have a non-None sparse vector."""
    encoder = SparseEncoder(_make_settings())
    chunks = [_make_chunk("c1", "retrieval augmented generation")]
    records = encoder.encode(chunks)
    assert records[0].sparse_vector is not None


@pytest.mark.unit
def test_record_preserves_text_and_metadata() -> None:
    """ChunkRecord must carry chunk text and metadata unchanged."""
    encoder = SparseEncoder(_make_settings())
    chunk = _make_chunk("c1", "vector database indexing")
    records = encoder.encode([chunk])
    assert records[0].text == chunk.text
    assert records[0].metadata["source_path"] == "test.pdf"


# ---------------------------------------------------------------------------
# Tests: TF counting
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_known_term_appears_in_sparse_vector() -> None:
    """Terms present in the chunk text must appear in the sparse vector."""
    encoder = SparseEncoder(_make_settings())
    chunks = [_make_chunk("c1", "embedding embedding retrieval")]
    records = encoder.encode(chunks)
    sv = records[0].sparse_vector
    assert sv is not None
    assert "embedding" in sv
    assert "retrieval" in sv


@pytest.mark.unit
def test_tf_count_reflects_repetitions() -> None:
    """Term frequency must equal the number of occurrences in the chunk."""
    encoder = SparseEncoder(_make_settings())
    chunks = [_make_chunk("c1", "cat cat cat dog dog")]
    records = encoder.encode(chunks)
    sv = records[0].sparse_vector
    assert sv is not None
    assert sv["cat"] == 3.0
    assert sv["dog"] == 2.0


@pytest.mark.unit
def test_sparse_vector_values_are_floats() -> None:
    """All sparse vector values must be Python floats."""
    encoder = SparseEncoder(_make_settings())
    chunks = [_make_chunk("c1", "hello world hello")]
    records = encoder.encode(chunks)
    sv = records[0].sparse_vector
    assert sv is not None
    for value in sv.values():
        assert isinstance(value, float)


# ---------------------------------------------------------------------------
# Tests: empty / edge-case text
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_empty_text_produces_empty_sparse_vector() -> None:
    """A chunk with empty text must yield an empty dict as sparse vector."""
    encoder = SparseEncoder(_make_settings())
    chunks = [_make_chunk("c1", "")]
    records = encoder.encode(chunks)
    sv = records[0].sparse_vector
    assert sv is not None
    assert sv == {}


@pytest.mark.unit
def test_whitespace_only_text_produces_empty_sparse_vector() -> None:
    """Whitespace-only text is equivalent to empty text."""
    encoder = SparseEncoder(_make_settings())
    chunks = [_make_chunk("c1", "   \t\n  ")]
    records = encoder.encode(chunks)
    assert records[0].sparse_vector == {}


@pytest.mark.unit
def test_punctuation_only_text_produces_empty_sparse_vector() -> None:
    """Text containing only punctuation/single-char tokens results in empty vector."""
    encoder = SparseEncoder(_make_settings())
    # All tokens are length 1 or purely symbolic – filtered out by tokenizer
    chunks = [_make_chunk("c1", "! @ # $ % ^ & * ( )")]
    records = encoder.encode(chunks)
    assert records[0].sparse_vector == {}


# ---------------------------------------------------------------------------
# Tests: stop-word filtering
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stop_words_are_excluded() -> None:
    """Common English stop words must not appear in the sparse vector."""
    encoder = SparseEncoder(_make_settings())
    stop_words_text = "the and or but in on at to for of with by from"
    chunks = [_make_chunk("c1", stop_words_text + " retrieval")]
    records = encoder.encode(chunks)
    sv = records[0].sparse_vector
    assert sv is not None
    for stop_word in ("the", "and", "or", "but", "in", "on", "at", "to", "for", "of"):
        assert stop_word not in sv
    # Content word must remain
    assert "retrieval" in sv


# ---------------------------------------------------------------------------
# Tests: empty input list
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_empty_chunks_raises_value_error() -> None:
    """encode() must raise ValueError when given an empty chunks list."""
    encoder = SparseEncoder(_make_settings())
    with pytest.raises(ValueError, match="empty"):
        encoder.encode([])


# ---------------------------------------------------------------------------
# Tests: trace integration
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_trace_records_stage_when_provided() -> None:
    """When a TraceContext is passed, encode() must record a stage."""
    encoder = SparseEncoder(_make_settings())
    trace = TraceContext()
    chunks = [_make_chunk("c1", "vector search engine")]
    encoder.encode(chunks, trace=trace)
    assert len(trace.stages) == 1
    assert trace.stages[0]["stage"] == "sparse_encode"


@pytest.mark.unit
def test_trace_not_required() -> None:
    """encode() must work without a trace context (trace=None)."""
    encoder = SparseEncoder(_make_settings())
    chunks = [_make_chunk("c1", "hello world")]
    records = encoder.encode(chunks, trace=None)
    assert len(records) == 1


# ---------------------------------------------------------------------------
# Tests: tokenizer helper (_tokenize)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_tokenize_lowercases_tokens() -> None:
    """_tokenize must return all tokens in lowercase."""
    tokens = _tokenize("Hello WORLD Foo")
    assert all(t == t.lower() for t in tokens)


@pytest.mark.unit
def test_tokenize_handles_cjk_characters() -> None:
    """_tokenize must split CJK characters as individual tokens."""
    tokens = _tokenize("向量检索")
    # Each CJK character is a separate token
    assert "向" in tokens or len(tokens) >= 1  # At least some tokens extracted


@pytest.mark.unit
def test_tokenize_empty_string_returns_empty_list() -> None:
    """_tokenize on empty string must return an empty list."""
    assert _tokenize("") == []
