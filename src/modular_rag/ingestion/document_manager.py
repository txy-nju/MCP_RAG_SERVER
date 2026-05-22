"""Cross-storage document lifecycle manager for dashboard operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DocumentInfo:
	"""Summary row used by dashboard document listings."""

	source_path: str
	chunk_count: int
	image_count: int
	file_hash: str | None = None


@dataclass(slots=True)
class DocumentDetail:
	"""Detailed payload for one source document."""

	doc_id: str
	source_path: str
	file_hash: str | None
	chunk_count: int
	image_count: int
	chunks: list[dict[str, Any]] = field(default_factory=list)
	images: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class DeleteResult:
	"""Deletion result across all managed storage backends."""

	source_path: str
	collection: str | None
	deleted_chunks: int
	deleted_images: int
	removed_integrity_record: bool
	success: bool


@dataclass(slots=True)
class CollectionStats:
	"""Collection-level aggregate stats for dashboard cards."""

	collection: str | None
	document_count: int
	chunk_count: int
	image_count: int
	storage_size_bytes: int


class DocumentManager:
	"""Coordinate list/detail/delete across vector, sparse, image, and integrity stores."""

	def __init__(self, chroma_store: Any, bm25_indexer: Any, image_storage: Any, file_integrity: Any) -> None:
		self._chroma_store = chroma_store
		self._bm25_indexer = bm25_indexer
		self._image_storage = image_storage
		self._file_integrity = file_integrity

	def list_documents(self, collection: str | None = None) -> list[DocumentInfo]:
		"""Return ingested document summaries grouped by source path."""
		records = self._fetch_chunk_records(collection=collection)
		if not records:
			return []

		path_to_hash = self._path_to_hash_map()
		grouped: dict[str, dict[str, Any]] = {}

		for record in records:
			metadata = dict(record.get("metadata", {}))
			source_path = str(metadata.get("source_path", "")).strip()
			if not source_path:
				continue

			entry = grouped.setdefault(
				source_path,
				{
					"chunk_count": 0,
					"image_ids": set(),
				},
			)
			entry["chunk_count"] += 1

			for image_id in self._extract_image_ids(metadata):
				entry["image_ids"].add(image_id)

		rows: list[DocumentInfo] = []
		for source_path, data in sorted(grouped.items(), key=lambda item: item[0]):
			rows.append(
				DocumentInfo(
					source_path=source_path,
					chunk_count=int(data["chunk_count"]),
					image_count=len(data["image_ids"]),
					file_hash=path_to_hash.get(source_path),
				)
			)
		return rows

	def get_document_detail(self, doc_id: str) -> DocumentDetail:
		"""Return all chunk/image details for a source path or file hash doc id."""
		normalized_doc_id = str(doc_id).strip()
		if not normalized_doc_id:
			raise ValueError("doc_id must be a non-empty string")

		path_to_hash = self._path_to_hash_map()
		hash_to_path = {file_hash: path for path, file_hash in path_to_hash.items() if file_hash}

		source_path = normalized_doc_id
		file_hash = path_to_hash.get(source_path)
		if file_hash is None and normalized_doc_id in hash_to_path:
			source_path = hash_to_path[normalized_doc_id]
			file_hash = normalized_doc_id

		detail_records = [
			record
			for record in self._fetch_chunk_records(collection=None)
			if str(record.get("metadata", {}).get("source_path", "")).strip() == source_path
		]
		if not detail_records:
			raise ValueError(f"document not found: {normalized_doc_id}")

		chunks = [
			{
				"chunk_id": str(record.get("id", "")),
				"text": str(record.get("text", "")),
				"metadata": dict(record.get("metadata", {})),
			}
			for record in detail_records
		]

		images = self._list_images(collection=None, doc_hash=file_hash)
		image_count = len(images)
		if image_count == 0:
			metadata_image_ids: set[str] = set()
			for record in detail_records:
				metadata_image_ids.update(self._extract_image_ids(dict(record.get("metadata", {}))))
			image_count = len(metadata_image_ids)

		return DocumentDetail(
			doc_id=normalized_doc_id,
			source_path=source_path,
			file_hash=file_hash,
			chunk_count=len(chunks),
			image_count=image_count,
			chunks=chunks,
			images=images,
		)

	def delete_document(self, source_path: str, collection: str | None = None) -> DeleteResult:
		"""Delete a source document across vector/BM25/image/integrity stores."""
		normalized_source_path = str(source_path).strip()
		if not normalized_source_path:
			raise ValueError("source_path must be a non-empty string")

		path_to_hash = self._path_to_hash_map()
		file_hash = path_to_hash.get(normalized_source_path)
		if file_hash is None and Path(normalized_source_path).exists():
			file_hash = self._file_integrity.compute_sha256(normalized_source_path)

		filters: dict[str, Any] = {"source_path": normalized_source_path}
		normalized_collection = str(collection).strip() if collection is not None else ""
		if normalized_collection:
			filters["collection"] = normalized_collection

		deleted_chunks = int(self._chroma_store.delete_by_metadata(filters))
		self._bm25_indexer.remove_document(normalized_source_path)
		deleted_images = self._delete_images(collection=normalized_collection or None, doc_hash=file_hash)

		removed_integrity_record = False
		if file_hash:
			self._file_integrity.remove_record(file_hash)
			removed_integrity_record = True

		success = bool(deleted_chunks or deleted_images or removed_integrity_record)
		return DeleteResult(
			source_path=normalized_source_path,
			collection=normalized_collection or None,
			deleted_chunks=deleted_chunks,
			deleted_images=deleted_images,
			removed_integrity_record=removed_integrity_record,
			success=success,
		)

	def get_collection_stats(self, collection: str | None = None) -> CollectionStats:
		"""Return aggregate document/chunk/image/size stats."""
		documents = self.list_documents(collection=collection)
		document_count = len(documents)
		chunk_count = sum(item.chunk_count for item in documents)
		image_count = sum(item.image_count for item in documents)

		storage_size_bytes = 0
		for doc in documents:
			source_file = Path(doc.source_path)
			if source_file.exists() and source_file.is_file():
				storage_size_bytes += source_file.stat().st_size

		for image in self._list_images(collection=collection, doc_hash=None):
			image_path = Path(str(image.get("file_path", "")))
			if image_path.exists() and image_path.is_file():
				storage_size_bytes += image_path.stat().st_size

		normalized_collection = str(collection).strip() if collection is not None else None
		if normalized_collection == "":
			normalized_collection = None

		return CollectionStats(
			collection=normalized_collection,
			document_count=document_count,
			chunk_count=chunk_count,
			image_count=image_count,
			storage_size_bytes=storage_size_bytes,
		)

	def _fetch_chunk_records(self, *, collection: str | None) -> list[dict[str, Any]]:
		filters: dict[str, Any] | None = None
		normalized_collection = str(collection).strip() if collection is not None else ""
		if normalized_collection:
			filters = {"collection": normalized_collection}

		results = self._chroma_store.get_by_metadata(filters=filters)
		rows: list[dict[str, Any]] = []
		for result in results:
			rows.append(
				{
					"id": str(getattr(result, "id", "")),
					"text": str(getattr(result, "text", "")),
					"metadata": dict(getattr(result, "metadata", {})),
				}
			)
		return rows

	def _path_to_hash_map(self) -> dict[str, str]:
		rows = self._file_integrity.list_processed()
		mapping: dict[str, str] = {}
		for row in rows:
			if str(row.get("status", "")) != "success":
				continue
			file_path = str(row.get("file_path", "")).strip()
			file_hash = str(row.get("file_hash", "")).strip()
			if file_path and file_hash:
				mapping[file_path] = file_hash
		return mapping

	def _list_images(self, *, collection: str | None, doc_hash: str | None) -> list[dict[str, Any]]:
		if hasattr(self._image_storage, "list_images"):
			return list(self._image_storage.list_images(collection=collection, doc_hash=doc_hash))
		if collection is not None and hasattr(self._image_storage, "list_by_collection"):
			rows = list(self._image_storage.list_by_collection(collection))
			if doc_hash:
				return [row for row in rows if str(row.get("doc_hash", "")) == str(doc_hash)]
			return rows
		return []

	def _delete_images(self, *, collection: str | None, doc_hash: str | None) -> int:
		if hasattr(self._image_storage, "delete_images"):
			return int(self._image_storage.delete_images(collection=collection, doc_hash=doc_hash))

		rows = self._list_images(collection=collection, doc_hash=doc_hash)
		deleted = 0
		for row in rows:
			image_path = Path(str(row.get("file_path", "")))
			if image_path.exists() and image_path.is_file():
				try:
					image_path.unlink()
				except Exception:
					pass
			deleted += 1
		return deleted

	@staticmethod
	def _extract_image_ids(metadata: dict[str, Any]) -> set[str]:
		image_ids: set[str] = set()

		image_refs = metadata.get("image_refs", [])
		if isinstance(image_refs, list):
			for image_id in image_refs:
				normalized = str(image_id).strip()
				if normalized:
					image_ids.add(normalized)

		images = metadata.get("images", [])
		if isinstance(images, list):
			for image in images:
				if not isinstance(image, dict):
					continue
				normalized = str(image.get("id", "")).strip()
				if normalized:
					image_ids.add(normalized)

		return image_ids


__all__ = [
	"CollectionStats",
	"DeleteResult",
	"DocumentDetail",
	"DocumentInfo",
	"DocumentManager",
]
