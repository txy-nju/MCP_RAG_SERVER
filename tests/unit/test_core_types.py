"""Tests for shared core data contracts."""

from __future__ import annotations

import json

import pytest

from core import Chunk, ChunkRecord, Document, ProcessedQuery, RetrievalResult


@pytest.mark.unit
def test_document_round_trips_with_image_metadata() -> None:
    document = Document(
        id="doc-1",
        text="# Title\n[IMAGE: dochash_1_0]\nhello",
        metadata={
            "source_path": "docs/example.pdf",
            "doc_type": "pdf",
            "images": [
                {
                    "id": "dochash_1_0",
                    "path": "data/images/test/dochash_1_0.png",
                    "page": 1,
                    "text_offset": 8,
                    "text_length": 20,
                    "position": {"x": 10, "y": 20},
                }
            ],
        },
    )

    payload = document.to_dict()

    assert payload["metadata"]["source_path"] == "docs/example.pdf"
    assert payload["metadata"]["images"][0]["id"] == "dochash_1_0"
    assert json.loads(json.dumps(payload)) == payload
    assert Document.from_dict(payload) == document


@pytest.mark.unit
def test_document_requires_metadata_source_path() -> None:
    with pytest.raises(ValueError, match="metadata.source_path is required"):
        Document(id="doc-1", text="hello", metadata={})


@pytest.mark.unit
def test_chunk_serializes_offsets_and_source_ref() -> None:
    chunk = Chunk(
        id="chunk-1",
        text="hello",
        metadata={"source_path": "docs/example.pdf"},
        start_offset=0,
        end_offset=5,
        source_ref={"document_id": "doc-1", "chunk_index": 0},
    )

    payload = chunk.to_dict()

    assert payload["start_offset"] == 0
    assert payload["end_offset"] == 5
    assert payload["source_ref"] == {"document_id": "doc-1", "chunk_index": 0}
    assert Chunk.from_dict(payload) == chunk


@pytest.mark.unit
def test_chunk_rejects_invalid_offsets() -> None:
    with pytest.raises(ValueError, match="end_offset must be greater than or equal to start_offset"):
        Chunk(
            id="chunk-1",
            text="hello",
            metadata={"source_path": "docs/example.pdf"},
            start_offset=10,
            end_offset=5,
        )


@pytest.mark.unit
def test_chunk_record_serializes_vectors() -> None:
    record = ChunkRecord(
        id="chunk-1",
        text="hello",
        metadata={"source_path": "docs/example.pdf"},
        dense_vector=[0, 1.5],
        sparse_vector={"hello": 2, "world": 0.5},
    )

    assert record.to_dict() == {
        "id": "chunk-1",
        "text": "hello",
        "metadata": {"source_path": "docs/example.pdf", "images": []},
        "dense_vector": [0.0, 1.5],
        "sparse_vector": {"hello": 2.0, "world": 0.5},
    }


@pytest.mark.unit
def test_chunk_record_from_chunk_preserves_shared_contract() -> None:
    chunk = Chunk(
        id="chunk-1",
        text="hello",
        metadata={"source_path": "docs/example.pdf"},
        start_offset=0,
        end_offset=5,
    )

    record = ChunkRecord.from_chunk(chunk, dense_vector=[1.0], sparse_vector={"hello": 1.0})

    assert record.id == chunk.id
    assert record.metadata == chunk.metadata
    assert record.dense_vector == [1.0]
    assert record.sparse_vector == {"hello": 1.0}


@pytest.mark.unit
def test_processed_query_and_retrieval_result_are_json_serializable() -> None:
    processed_query = ProcessedQuery(
        raw_query="How to configure Azure OpenAI?",
        keywords=["configure", "azure", "openai"],
        filters={"collection": "docs"},
    )
    retrieval_result = RetrievalResult(
        chunk_id="chunk-1",
        score=0.95,
        text="Use the azure provider settings.",
        metadata={"source_path": "docs/config.pdf", "page": 2},
    )

    assert json.loads(json.dumps(processed_query.to_dict())) == processed_query.to_dict()
    assert json.loads(json.dumps(retrieval_result.to_dict())) == retrieval_result.to_dict()
