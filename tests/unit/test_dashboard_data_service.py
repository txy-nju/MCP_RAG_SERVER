"""Unit tests for Dashboard DataService (G3)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from observability.dashboard.services.data_service import DataService


@dataclass(slots=True)
class _FakeVectorRecord:
    id: str
    text: str
    metadata: dict[str, Any]


class _FakeChromaStore:
    def __init__(self, records: list[_FakeVectorRecord]) -> None:
        self._records = list(records)

    def get_by_metadata(self, filters: dict[str, Any] | None = None, limit: int | None = None):
        rows = list(self._records)
        if filters:
            rows = [
                row
                for row in rows
                if all(row.metadata.get(key) == value for key, value in filters.items())
            ]
        if limit is not None:
            return rows[:limit]
        return rows


class _FakeImageStorage:
    def list_images(self, *, collection: str | None = None, doc_hash: str | None = None):
        rows = [
            {"image_id": "img-1", "collection": "alpha", "doc_hash": "hash-a", "file_path": "x"},
            {"image_id": "img-2", "collection": "beta", "doc_hash": "hash-b", "file_path": "y"},
        ]
        if collection is not None:
            rows = [row for row in rows if row["collection"] == collection]
        if doc_hash is not None:
            rows = [row for row in rows if row["doc_hash"] == doc_hash]
        return rows


class _FakeDocumentManager:
    def list_documents(self, collection: str | None = None):
        class _Doc:
            def __init__(self, source_path: str, chunk_count: int, image_count: int, file_hash: str | None) -> None:
                self.source_path = source_path
                self.chunk_count = chunk_count
                self.image_count = image_count
                self.file_hash = file_hash

        docs = [
            _Doc("a.pdf", 2, 1, "hash-a"),
            _Doc("b.pdf", 1, 0, "hash-b"),
        ]
        if collection is None:
            return docs
        if collection == "alpha":
            return [docs[0]]
        if collection == "beta":
            return [docs[1]]
        return []

    def get_document_detail(self, doc_id: str):
        class _Detail:
            def __init__(self) -> None:
                self.doc_id = doc_id
                self.source_path = doc_id
                self.file_hash = "hash-a"
                self.chunk_count = 2
                self.image_count = 1
                self.chunks = [
                    {"chunk_id": "c1", "text": "t1", "metadata": {"chunk_index": 1}},
                    {"chunk_id": "c0", "text": "t0", "metadata": {"chunk_index": 0}},
                ]
                self.images = [{"image_id": "img-1", "file_path": "x", "doc_hash": "hash-a"}]

        return _Detail()


@pytest.mark.unit
def test_list_collections_collects_unique_sorted_values() -> None:
    chroma = _FakeChromaStore(
        records=[
            _FakeVectorRecord("c1", "t1", {"collection": "beta"}),
            _FakeVectorRecord("c2", "t2", {"collection": "alpha"}),
            _FakeVectorRecord("c3", "t3", {"collection": "alpha"}),
        ]
    )

    service = DataService(
        settings=object(),
        chroma_store=chroma,
        image_storage=_FakeImageStorage(),
        file_integrity=object(),
        document_manager=_FakeDocumentManager(),
    )

    assert service.list_collections() == ["alpha", "beta"]


@pytest.mark.unit
def test_get_chunks_filters_and_sorts_by_chunk_index() -> None:
    chroma = _FakeChromaStore(
        records=[
            _FakeVectorRecord("c1", "t1", {"source_path": "a.pdf", "collection": "alpha", "chunk_index": 1}),
            _FakeVectorRecord("c0", "t0", {"source_path": "a.pdf", "collection": "alpha", "chunk_index": 0}),
            _FakeVectorRecord("c2", "t2", {"source_path": "b.pdf", "collection": "beta", "chunk_index": 0}),
        ]
    )

    service = DataService(
        settings=object(),
        chroma_store=chroma,
        image_storage=_FakeImageStorage(),
        file_integrity=object(),
        document_manager=_FakeDocumentManager(),
    )

    chunks = service.get_chunks("a.pdf", collection="alpha")

    assert [row["chunk_id"] for row in chunks] == ["c0", "c1"]


@pytest.mark.unit
def test_get_document_detail_returns_normalized_mapping() -> None:
    service = DataService(
        settings=object(),
        chroma_store=_FakeChromaStore(records=[]),
        image_storage=_FakeImageStorage(),
        file_integrity=object(),
        document_manager=_FakeDocumentManager(),
    )

    detail = service.get_document_detail("a.pdf")

    assert detail["doc_id"] == "a.pdf"
    assert detail["chunk_count"] == 2
    assert detail["images"][0]["image_id"] == "img-1"
