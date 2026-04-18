"""Data access service for Dashboard data browser page."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.settings import Settings, load_settings
from ingestion.document_manager import DocumentManager
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from libs.loader.file_integrity import SQLiteIntegrityChecker
from libs.vector_store.chroma_store import ChromaStore

_DEFAULT_CONFIG_PATH = str(Path(__file__).parents[4] / "config" / "settings.yaml")


@dataclass(slots=True)
class DocumentRow:
	"""Flattened document row used by Data Browser list view."""

	source_path: str
	collection: str
	chunk_count: int
	image_count: int
	file_hash: str | None


class DataService:
	"""Facade for reading collection/document/chunk/image data for Dashboard."""

	def __init__(
		self,
		*,
		settings: Settings | None = None,
		config_path: str | None = None,
		chroma_store: Any | None = None,
		image_storage: Any | None = None,
		file_integrity: Any | None = None,
		document_manager: DocumentManager | None = None,
	) -> None:
		if settings is not None:
			self._settings = settings
		else:
			self._settings = load_settings(config_path or _DEFAULT_CONFIG_PATH)

		self._chroma_store = chroma_store or ChromaStore.from_settings(self._settings.vector_store)
		self._image_storage = image_storage or ImageStorage()
		self._file_integrity = file_integrity or SQLiteIntegrityChecker()

		self._document_manager = document_manager or DocumentManager(
			chroma_store=self._chroma_store,
			bm25_indexer=BM25Indexer(),
			image_storage=self._image_storage,
			file_integrity=self._file_integrity,
		)

	@property
	def default_collection(self) -> str:
		"""Return the configured default collection name."""
		vector_store = getattr(self._settings, "vector_store", None)
		return str(getattr(vector_store, "collection", "")).strip()

	def list_collections(self) -> list[str]:
		"""Return sorted collection names discovered from chunk metadata."""
		records = self._chroma_store.get_by_metadata(filters=None)
		collections: set[str] = set()
		for record in records:
			metadata = dict(getattr(record, "metadata", {}))
			collection = str(metadata.get("collection", "")).strip()
			if collection:
				collections.add(collection)
		return sorted(collections)

	def list_documents(self, collection: str | None = None) -> list[DocumentRow]:
		"""Return document rows for table rendering."""
		normalized_collection = str(collection).strip() if collection is not None else ""
		docs = self._document_manager.list_documents(collection=normalized_collection or None)

		rows: list[DocumentRow] = []
		for doc in docs:
			rows.append(
				DocumentRow(
					source_path=doc.source_path,
					collection=normalized_collection or self._infer_collection_for_source(doc.source_path),
					chunk_count=int(doc.chunk_count),
					image_count=int(doc.image_count),
					file_hash=doc.file_hash,
				)
			)
		return rows

	def get_document_detail(self, doc_id: str) -> dict[str, Any]:
		"""Return normalized document detail payload for UI rendering."""
		detail = self._document_manager.get_document_detail(doc_id)
		return {
			"doc_id": detail.doc_id,
			"source_path": detail.source_path,
			"file_hash": detail.file_hash,
			"chunk_count": detail.chunk_count,
			"image_count": detail.image_count,
			"chunks": list(detail.chunks),
			"images": list(detail.images),
		}

	def delete_document(self, source_path: str, collection: str | None = None) -> Any:
		"""Delete one source document through the underlying DocumentManager."""
		return self._document_manager.delete_document(source_path, collection=collection)

	def get_chunks(self, source_path: str, collection: str | None = None) -> list[dict[str, Any]]:
		"""Return chunks for one source file, sorted by chunk_index then chunk id."""
		normalized_source = str(source_path).strip()
		if not normalized_source:
			return []

		filters: dict[str, Any] = {"source_path": normalized_source}
		normalized_collection = str(collection).strip() if collection is not None else ""
		if normalized_collection:
			filters["collection"] = normalized_collection

		rows: list[dict[str, Any]] = []
		for record in self._chroma_store.get_by_metadata(filters=filters):
			metadata = dict(getattr(record, "metadata", {}))
			rows.append(
				{
					"chunk_id": str(getattr(record, "id", "")),
					"text": str(getattr(record, "text", "")),
					"metadata": metadata,
				}
			)

		rows.sort(
			key=lambda item: (
				int(item["metadata"].get("chunk_index", 10**9)),
				str(item["chunk_id"]),
			)
		)
		return rows

	def get_images(self, *, collection: str | None = None, doc_hash: str | None = None) -> list[dict[str, Any]]:
		"""Return image index rows for optional collection/doc filters."""
		if hasattr(self._image_storage, "list_images"):
			return list(self._image_storage.list_images(collection=collection, doc_hash=doc_hash))
		if collection is not None and hasattr(self._image_storage, "list_by_collection"):
			rows = list(self._image_storage.list_by_collection(collection))
			if doc_hash is None:
				return rows
			return [row for row in rows if str(row.get("doc_hash", "")) == str(doc_hash)]
		return []

	def _infer_collection_for_source(self, source_path: str) -> str:
		rows = self._chroma_store.get_by_metadata(filters={"source_path": source_path}, limit=1)
		if not rows:
			return ""
		metadata = dict(getattr(rows[0], "metadata", {}))
		return str(metadata.get("collection", "")).strip()


__all__ = ["DataService", "DocumentRow"]
