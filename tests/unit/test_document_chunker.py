"""Unit tests for DocumentChunker.

Validates the 6 core responsibilities:
1. Deterministic chunk ID generation
2. Document metadata inheritance
3. chunk_index tracking
4. source_ref back-link
5. Per-chunk image reference distribution
6. Type contract compliance (List[Chunk])

Uses FakeSplitter to isolate from real LLM/embedding dependencies.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any
from unittest.mock import patch

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
from core.types import Chunk, Document
from ingestion.chunking.document_chunker import DocumentChunker
from libs.splitter.base_splitter import BaseSplitter


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeSplitter(BaseSplitter):
    """Controllable splitter for isolation testing."""

    def __init__(self, chunks: list[str]) -> None:
        super().__init__(provider="fake", chunk_size=100, chunk_overlap=0)
        self._chunks = chunks

    def split_text(self, text: str, trace: Any = None) -> list[str]:
        return list(self._chunks)


def _make_settings(splitter_provider: str = "fake") -> Settings:
    """Build a minimal Settings object for testing."""
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        splitter=SplitterSettings(provider=splitter_provider, chunk_size=100, chunk_overlap=0),
        vector_store=VectorStoreSettings(provider="chroma", collection="test"),
        retrieval=RetrievalSettings(top_k=5),
        rerank=RerankSettings(provider="none"),
        evaluation=EvaluationSettings(backend="custom"),
        observability=ObservabilitySettings(log_level="INFO", trace_file="logs/trace.jsonl"),
    )


def _make_chunker(chunks: list[str]) -> DocumentChunker:
    """Build a DocumentChunker backed by a FakeSplitter returning the given chunks."""
    settings = _make_settings()
    chunker = DocumentChunker.__new__(DocumentChunker)
    chunker._splitter = FakeSplitter(chunks)
    return chunker


def _simple_document(text: str = "hello world", **meta: Any) -> Document:
    base_meta: dict[str, Any] = {"source_path": "docs/sample.pdf"}
    base_meta.update(meta)
    return Document(id="doc-001", text=text, metadata=base_meta)


# ---------------------------------------------------------------------------
# 1. Chunk ID uniqueness
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_chunk_ids_are_unique_within_document() -> None:
    doc = _simple_document("paragraph one. paragraph two. paragraph three.")
    chunker = _make_chunker(["paragraph one.", "paragraph two.", "paragraph three."])
    chunks = chunker.split_document(doc)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids)), "chunk IDs must be unique within a document"


@pytest.mark.unit
def test_chunk_ids_unique_even_for_identical_texts() -> None:
    """Two chunks with the same text should still get different IDs (different index)."""
    doc = _simple_document("same text. same text.")
    chunker = _make_chunker(["same text.", "same text."])
    chunks = chunker.split_document(doc)
    assert chunks[0].id != chunks[1].id


# ---------------------------------------------------------------------------
# 2. Chunk ID determinism
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_chunk_ids_are_deterministic_across_repeated_splits() -> None:
    doc = _simple_document("hello world")
    chunker = _make_chunker(["hello world"])
    ids_first = [c.id for c in chunker.split_document(doc)]
    ids_second = [c.id for c in chunker.split_document(doc)]
    assert ids_first == ids_second


@pytest.mark.unit
def test_chunk_id_format_matches_spec() -> None:
    """ID format: {doc_id}_{index:04d}_{hash_8chars}"""
    doc = _simple_document("chunk text here")
    chunker = _make_chunker(["chunk text here"])
    chunk = chunker.split_document(doc)[0]

    expected_hash = hashlib.sha256("chunk text here".encode()).hexdigest()[:8]
    assert chunk.id == f"doc-001_0000_{expected_hash}"


# ---------------------------------------------------------------------------
# 3. Metadata inheritance
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_all_document_metadata_fields_are_inherited() -> None:
    doc = _simple_document("some text", doc_type="pdf", title="My Doc", author="Alice")
    chunker = _make_chunker(["some text"])
    chunk = chunker.split_document(doc)[0]

    assert chunk.metadata["source_path"] == "docs/sample.pdf"
    assert chunk.metadata["doc_type"] == "pdf"
    assert chunk.metadata["title"] == "My Doc"
    assert chunk.metadata["author"] == "Alice"


@pytest.mark.unit
def test_chunk_index_is_added_to_metadata() -> None:
    doc = _simple_document("a b c")
    chunker = _make_chunker(["a", "b", "c"])
    chunks = chunker.split_document(doc)

    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[1].metadata["chunk_index"] == 1
    assert chunks[2].metadata["chunk_index"] == 2


@pytest.mark.unit
def test_metadata_mutation_does_not_affect_original_document() -> None:
    doc = _simple_document("text")
    chunker = _make_chunker(["text"])
    chunks = chunker.split_document(doc)
    chunks[0].metadata["injected"] = "value"

    assert "injected" not in doc.metadata


# ---------------------------------------------------------------------------
# 4. source_ref back-link
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_source_ref_points_to_parent_document_id() -> None:
    doc = _simple_document("content")
    chunker = _make_chunker(["content"])
    chunk = chunker.split_document(doc)[0]

    assert chunk.source_ref is not None
    assert chunk.source_ref["doc_id"] == "doc-001"


@pytest.mark.unit
def test_all_chunks_have_same_source_ref_doc_id() -> None:
    doc = _simple_document("alpha beta gamma")
    chunker = _make_chunker(["alpha", "beta", "gamma"])
    chunks = chunker.split_document(doc)

    for chunk in chunks:
        assert chunk.source_ref is not None
        assert chunk.source_ref["doc_id"] == doc.id


# ---------------------------------------------------------------------------
# 5. Image reference distribution
# ---------------------------------------------------------------------------

_IMAGE_META = {
    "id": "img-001",
    "path": "data/images/img-001.png",
    "text_offset": 10,
    "text_length": 20,
    "page": 1,
}

_IMAGE_META_2 = {
    "id": "img-002",
    "path": "data/images/img-002.png",
    "text_offset": 50,
    "text_length": 20,
    "page": 2,
}


@pytest.mark.unit
def test_chunk_with_image_placeholder_gets_matching_image_metadata() -> None:
    doc = Document(
        id="doc-img",
        text="Intro [IMAGE: img-001] End",
        metadata={"source_path": "docs/report.pdf", "images": [_IMAGE_META]},
    )
    chunker = _make_chunker(["Intro [IMAGE: img-001] End"])
    chunk = chunker.split_document(doc)[0]

    assert len(chunk.metadata["images"]) == 1
    assert chunk.metadata["images"][0]["id"] == "img-001"
    assert chunk.metadata["image_refs"] == ["img-001"]


@pytest.mark.unit
def test_chunk_without_image_placeholder_has_no_image_refs() -> None:
    doc = Document(
        id="doc-img",
        text="Intro [IMAGE: img-001] normal text",
        metadata={"source_path": "docs/report.pdf", "images": [_IMAGE_META]},
    )
    chunker = _make_chunker(["Intro [IMAGE: img-001]", "normal text"])
    chunks = chunker.split_document(doc)

    no_image_chunk = chunks[1]
    assert "image_refs" not in no_image_chunk.metadata
    assert no_image_chunk.metadata.get("images") == []


@pytest.mark.unit
def test_image_refs_are_distributed_correctly_across_multiple_chunks() -> None:
    doc = Document(
        id="doc-multi",
        text="chunk A [IMAGE: img-001] --- chunk B [IMAGE: img-002]",
        metadata={
            "source_path": "docs/multi.pdf",
            "images": [_IMAGE_META, _IMAGE_META_2],
        },
    )
    chunker = _make_chunker(["chunk A [IMAGE: img-001]", "chunk B [IMAGE: img-002]"])
    chunks = chunker.split_document(doc)

    assert chunks[0].metadata["image_refs"] == ["img-001"]
    assert chunks[0].metadata["images"][0]["id"] == "img-001"
    assert chunks[1].metadata["image_refs"] == ["img-002"]
    assert chunks[1].metadata["images"][0]["id"] == "img-002"


@pytest.mark.unit
def test_image_not_in_document_images_is_silently_skipped() -> None:
    """If a placeholder references an unknown image ID, it should be skipped gracefully."""
    doc = Document(
        id="doc-missing",
        text="text [IMAGE: unknown-id]",
        metadata={"source_path": "docs/x.pdf", "images": [_IMAGE_META]},
    )
    chunker = _make_chunker(["text [IMAGE: unknown-id]"])
    chunk = chunker.split_document(doc)[0]
    # unknown-id not in images, so images list should be empty
    assert chunk.metadata.get("images", []) == []
    # but image_refs should still list what was referenced
    assert chunk.metadata.get("image_refs") == ["unknown-id"]


@pytest.mark.unit
def test_document_without_images_produces_chunks_with_empty_images() -> None:
    doc = _simple_document("plain text with no images")
    chunker = _make_chunker(["plain text with no images"])
    chunk = chunker.split_document(doc)[0]

    # images is normalized to [] when not provided
    assert chunk.metadata.get("images") == []
    assert "image_refs" not in chunk.metadata


# ---------------------------------------------------------------------------
# 6. Type contract compliance
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_split_document_returns_list_of_chunk_instances() -> None:
    doc = _simple_document("page one. page two.")
    chunker = _make_chunker(["page one.", "page two."])
    result = chunker.split_document(doc)

    assert isinstance(result, list)
    assert all(isinstance(c, Chunk) for c in result)


@pytest.mark.unit
def test_chunk_is_serializable_to_dict() -> None:
    doc = _simple_document("hello world")
    chunker = _make_chunker(["hello world"])
    chunk = chunker.split_document(doc)[0]

    payload = chunk.to_dict()
    assert payload["id"] == chunk.id
    assert payload["text"] == "hello world"
    assert isinstance(payload["metadata"], dict)
    assert payload["start_offset"] >= 0
    assert payload["end_offset"] >= payload["start_offset"]


@pytest.mark.unit
def test_empty_document_text_produces_no_chunks() -> None:
    """When the splitter returns an empty list (e.g. blank document), result is empty."""
    doc = _simple_document("   ")
    chunker = _make_chunker([])
    chunks = chunker.split_document(doc)
    assert chunks == []


@pytest.mark.unit
def test_chunker_delegates_to_splitter_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """DocumentChunker.__init__ must use SplitterFactory.create."""
    created: list[object] = []

    def fake_create(settings: object) -> FakeSplitter:
        created.append(settings)
        return FakeSplitter(["stub"])

    monkeypatch.setattr("ingestion.chunking.document_chunker.SplitterFactory.create", fake_create)
    settings = _make_settings()
    chunker = DocumentChunker(settings)

    assert len(created) == 1
    assert created[0] is settings
