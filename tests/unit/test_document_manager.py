"""Unit tests for DocumentManager cross-storage lifecycle operations (G2)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from modular_rag.ingestion.document_manager import DocumentManager


@dataclass(slots=True)
class _FakeVectorRecord:
    id: str
    text: str
    metadata: dict[str, Any]


class _FakeChromaStore:
    def __init__(self, records: list[_FakeVectorRecord]) -> None:
        self._records = list(records)

    def get_by_metadata(self, filters: dict[str, Any] | None = None):
        if not filters:
            return list(self._records)
        result = []
        for record in self._records:
            if all(record.metadata.get(k) == v for k, v in filters.items()):
                result.append(record)
        return result

    def delete_by_metadata(self, filters: dict[str, Any]) -> int:
        keep: list[_FakeVectorRecord] = []
        deleted = 0
        for record in self._records:
            if all(record.metadata.get(k) == v for k, v in filters.items()):
                deleted += 1
            else:
                keep.append(record)
        self._records = keep
        return deleted


class _FakeBM25Indexer:
    def __init__(self) -> None:
        self.removed_sources: list[str] = []

    def remove_document(self, source: str) -> None:
        self.removed_sources.append(source)


class _FakeImageStorage:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = list(rows)

    def list_images(self, *, collection: str | None = None, doc_hash: str | None = None) -> list[dict[str, Any]]:
        rows = list(self._rows)
        if collection is not None:
            rows = [row for row in rows if row.get("collection") == collection]
        if doc_hash is not None:
            rows = [row for row in rows if row.get("doc_hash") == doc_hash]
        return rows

    def delete_images(self, *, collection: str | None = None, doc_hash: str | None = None) -> int:
        deleted = 0
        keep = []
        for row in self._rows:
            match_collection = collection is None or row.get("collection") == collection
            match_hash = doc_hash is None or row.get("doc_hash") == doc_hash
            if match_collection and match_hash:
                deleted += 1
            else:
                keep.append(row)
        self._rows = keep
        return deleted


class _FakeIntegrity:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = list(rows)
        self.removed_hashes: list[str] = []

    def list_processed(self) -> list[dict[str, str | None]]:
        return list(self._rows)

    def compute_sha256(self, path: str) -> str:
        return f"hash::{path}"

    def remove_record(self, file_hash: str) -> None:
        self.removed_hashes.append(file_hash)
        self._rows = [row for row in self._rows if row.get("file_hash") != file_hash]


@pytest.mark.unit
def test_list_documents_groups_by_source_and_counts_chunks_and_images(tmp_path: Path) -> None:
    source_a = str(tmp_path / "docs" / "alpha.pdf")
    source_b = str(tmp_path / "docs" / "beta.pdf")

    manager = DocumentManager(
        chroma_store=_FakeChromaStore(
            records=[
                _FakeVectorRecord("c1", "t1", {"source_path": source_a, "collection": "default", "image_refs": ["img-1"]}),
                _FakeVectorRecord("c2", "t2", {"source_path": source_a, "collection": "default", "image_refs": ["img-2"]}),
                _FakeVectorRecord("c3", "t3", {"source_path": source_b, "collection": "default"}),
            ]
        ),
        bm25_indexer=_FakeBM25Indexer(),
        image_storage=_FakeImageStorage(rows=[]),
        file_integrity=_FakeIntegrity(
            rows=[
                {"file_hash": "hash-a", "file_path": source_a, "status": "success"},
                {"file_hash": "hash-b", "file_path": source_b, "status": "success"},
            ]
        ),
    )

    docs = manager.list_documents(collection="default")

    assert [item.source_path for item in docs] == [source_a, source_b]
    assert docs[0].chunk_count == 2
    assert docs[0].image_count == 2
    assert docs[0].file_hash == "hash-a"
    assert docs[1].chunk_count == 1
    assert docs[1].image_count == 0


@pytest.mark.unit
def test_get_document_detail_returns_chunks_and_linked_images(tmp_path: Path) -> None:
    source_a = str(tmp_path / "docs" / "alpha.pdf")
    image_path = tmp_path / "images" / "img-1.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"img")

    manager = DocumentManager(
        chroma_store=_FakeChromaStore(
            records=[
                _FakeVectorRecord("c1", "chunk-a", {"source_path": source_a, "collection": "default"}),
                _FakeVectorRecord("c2", "chunk-b", {"source_path": source_a, "collection": "default"}),
            ]
        ),
        bm25_indexer=_FakeBM25Indexer(),
        image_storage=_FakeImageStorage(
            rows=[
                {
                    "image_id": "img-1",
                    "file_path": str(image_path),
                    "collection": "default",
                    "doc_hash": "hash-a",
                    "page_num": 1,
                    "created_at": "now",
                }
            ]
        ),
        file_integrity=_FakeIntegrity(rows=[{"file_hash": "hash-a", "file_path": source_a, "status": "success"}]),
    )

    detail = manager.get_document_detail(source_a)

    assert detail.source_path == source_a
    assert detail.chunk_count == 2
    assert detail.image_count == 1
    assert [chunk["chunk_id"] for chunk in detail.chunks] == ["c1", "c2"]
    assert detail.images[0]["image_id"] == "img-1"


@pytest.mark.unit
def test_delete_document_coordinates_all_storages(tmp_path: Path) -> None:
    source_a = str(tmp_path / "docs" / "alpha.pdf")

    chroma = _FakeChromaStore(
        records=[
            _FakeVectorRecord("c1", "t1", {"source_path": source_a, "collection": "default"}),
            _FakeVectorRecord("c2", "t2", {"source_path": source_a, "collection": "default"}),
        ]
    )
    bm25 = _FakeBM25Indexer()
    image_storage = _FakeImageStorage(
        rows=[
            {"image_id": "img-1", "file_path": "ignored", "collection": "default", "doc_hash": "hash-a"}
        ]
    )
    integrity = _FakeIntegrity(rows=[{"file_hash": "hash-a", "file_path": source_a, "status": "success"}])

    manager = DocumentManager(
        chroma_store=chroma,
        bm25_indexer=bm25,
        image_storage=image_storage,
        file_integrity=integrity,
    )

    result = manager.delete_document(source_a, collection="default")

    assert result.success is True
    assert result.deleted_chunks == 2
    assert result.deleted_images == 1
    assert result.removed_integrity_record is True
    assert bm25.removed_sources == [source_a]
    assert integrity.removed_hashes == ["hash-a"]
    assert manager.list_documents(collection="default") == []


@pytest.mark.unit
def test_get_collection_stats_aggregates_counts_and_sizes(tmp_path: Path) -> None:
    source_a = tmp_path / "docs" / "alpha.pdf"
    source_a.parent.mkdir(parents=True)
    source_a.write_bytes(b"alpha")

    image_path = tmp_path / "images" / "img-1.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image")

    manager = DocumentManager(
        chroma_store=_FakeChromaStore(
            records=[
                _FakeVectorRecord("c1", "t1", {"source_path": str(source_a), "collection": "default", "image_refs": ["img-1"]}),
            ]
        ),
        bm25_indexer=_FakeBM25Indexer(),
        image_storage=_FakeImageStorage(
            rows=[
                {
                    "image_id": "img-1",
                    "file_path": str(image_path),
                    "collection": "default",
                    "doc_hash": "hash-a",
                }
            ]
        ),
        file_integrity=_FakeIntegrity(rows=[{"file_hash": "hash-a", "file_path": str(source_a), "status": "success"}]),
    )

    stats = manager.get_collection_stats(collection="default")

    assert stats.document_count == 1
    assert stats.chunk_count == 1
    assert stats.image_count == 1
    assert stats.storage_size_bytes >= len(b"alpha") + len(b"image")
