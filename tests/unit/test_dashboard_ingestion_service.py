"""Unit tests for Dashboard IngestionService (G4)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from observability.dashboard.services.ingestion_service import IngestionService


@dataclass(slots=True)
class _FakeUploadedFile:
    name: str
    content: bytes

    def getbuffer(self) -> memoryview:
        return memoryview(self.content)


class _FakePipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(
        self,
        source_path: str,
        *,
        collection: str,
        force: bool,
        batch_size: int,
        on_progress: Any | None,
    ) -> dict[str, Any]:
        self.calls.append(
            {
                "source_path": source_path,
                "collection": collection,
                "force": force,
                "batch_size": batch_size,
                "on_progress": on_progress,
            }
        )
        return {"skipped": False, "chunk_count": 2}


class _FakeDataService:
    def __init__(self) -> None:
        self.default_collection = "knowledge"
        self.deleted: list[tuple[str, str | None]] = []

    def list_collections(self) -> list[str]:
        return ["alpha", "knowledge"]

    def list_documents(self, collection: str | None = None) -> list[str]:
        return ["doc"] if collection != "empty" else []

    def delete_document(self, source_path: str, collection: str | None = None) -> dict[str, Any]:
        self.deleted.append((source_path, collection))
        return {"success": True}


class _FakeSettings:
    class _VectorStore:
        collection = "knowledge"

    vector_store = _VectorStore()


@pytest.mark.unit
def test_save_uploaded_file_persists_bytes_under_collection_dir(tmp_path: Path) -> None:
    service = IngestionService(
        settings=_FakeSettings(),
        documents_root=tmp_path / "documents",
        pipeline=_FakePipeline(),
        data_service=_FakeDataService(),
    )

    saved_path = service.save_uploaded_file(_FakeUploadedFile("sample.pdf", b"%PDF-test"), "alpha")

    assert saved_path == tmp_path / "documents" / "alpha" / "sample.pdf"
    assert saved_path.read_bytes() == b"%PDF-test"


@pytest.mark.unit
def test_ingest_uploaded_file_saves_then_calls_pipeline(tmp_path: Path) -> None:
    pipeline = _FakePipeline()
    service = IngestionService(
        settings=_FakeSettings(),
        documents_root=tmp_path / "documents",
        pipeline=pipeline,
        data_service=_FakeDataService(),
    )

    progress = []
    result = service.ingest_uploaded_file(
        _FakeUploadedFile("sample.pdf", b"payload"),
        collection="alpha",
        force=True,
        on_progress=progress.append,
        batch_size=8,
    )

    assert result == {"skipped": False, "chunk_count": 2}
    assert len(pipeline.calls) == 1
    assert pipeline.calls[0]["collection"] == "alpha"
    assert pipeline.calls[0]["force"] is True
    assert pipeline.calls[0]["batch_size"] == 8
    assert Path(pipeline.calls[0]["source_path"]).exists()


@pytest.mark.unit
def test_delete_document_delegates_to_data_service(tmp_path: Path) -> None:
    data_service = _FakeDataService()
    service = IngestionService(
        settings=_FakeSettings(),
        documents_root=tmp_path / "documents",
        pipeline=_FakePipeline(),
        data_service=data_service,
    )

    result = service.delete_document("alpha/sample.pdf", collection="alpha")

    assert result == {"success": True}
    assert data_service.deleted == [("alpha/sample.pdf", "alpha")]
