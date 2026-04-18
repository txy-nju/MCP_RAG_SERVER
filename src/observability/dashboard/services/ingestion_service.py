"""Business service for Dashboard ingestion management workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, BinaryIO, Protocol

from core.settings import Settings, load_settings
from ingestion.pipeline import IngestionPipeline
from observability.dashboard.services.data_service import DataService

_DEFAULT_CONFIG_PATH = str(Path(__file__).parents[4] / "config" / "settings.yaml")
_DEFAULT_DOCUMENTS_ROOT = Path(__file__).parents[4] / "data" / "documents"


class UploadedFileLike(Protocol):
    """Minimal upload contract compatible with Streamlit UploadedFile."""

    name: str

    def getbuffer(self) -> BinaryIO | bytes | bytearray | memoryview: ...


class IngestionService:
    """Coordinate uploaded file persistence, pipeline execution, and document deletion."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        config_path: str | None = None,
        documents_root: str | Path | None = None,
        pipeline: IngestionPipeline | None = None,
        data_service: DataService | None = None,
    ) -> None:
        self._settings = settings or load_settings(config_path or _DEFAULT_CONFIG_PATH)
        self._documents_root = Path(documents_root) if documents_root is not None else _DEFAULT_DOCUMENTS_ROOT
        self._data_service = data_service or DataService(settings=self._settings)
        self._pipeline = pipeline

    @property
    def default_collection(self) -> str:
        """Return the configured default collection name."""
        collection = self._data_service.default_collection
        if collection:
            return collection
        return str(self._settings.vector_store.collection)

    def list_collections(self) -> list[str]:
        """Return known collection names for filtering or selection."""
        collections = set(self._data_service.list_collections())
        if self.default_collection:
            collections.add(self.default_collection)
        return sorted(collections)

    def list_documents(self, collection: str | None = None) -> list[Any]:
        """Return document rows for the Dashboard table."""
        return self._data_service.list_documents(collection=collection)

    def save_uploaded_file(self, uploaded_file: UploadedFileLike, collection: str) -> Path:
        """Persist an uploaded file under data/documents/<collection>/."""
        normalized_collection = str(collection).strip() or self.default_collection or "default"
        collection_dir = self._documents_root / normalized_collection
        collection_dir.mkdir(parents=True, exist_ok=True)

        target_path = collection_dir / Path(str(uploaded_file.name)).name
        payload = uploaded_file.getbuffer()
        if isinstance(payload, memoryview):
            content = payload.tobytes()
        elif isinstance(payload, bytearray):
            content = bytes(payload)
        elif isinstance(payload, bytes):
            content = payload
        else:
            content = bytes(payload.read())
        target_path.write_bytes(content)
        return target_path

    def ingest_uploaded_file(
        self,
        uploaded_file: UploadedFileLike,
        *,
        collection: str,
        force: bool = False,
        on_progress: Any | None = None,
        batch_size: int = 32,
    ) -> Any:
        """Persist and ingest one uploaded document."""
        target_path = self.save_uploaded_file(uploaded_file, collection)
        normalized_collection = str(collection).strip() or self.default_collection or "default"
        return self._get_pipeline().run(
            str(target_path),
            collection=normalized_collection,
            force=force,
            batch_size=batch_size,
            on_progress=on_progress,
        )

    def delete_document(self, source_path: str, collection: str | None = None) -> Any:
        """Delete one ingested source document."""
        return self._data_service.delete_document(source_path, collection=collection)

    def _get_pipeline(self) -> IngestionPipeline:
        if self._pipeline is None:
            self._pipeline = IngestionPipeline(self._settings)
        return self._pipeline


__all__ = ["IngestionService", "UploadedFileLike"]
