"""Vector upserter for deterministic IDs and idempotent vector-store writes."""

from __future__ import annotations

from hashlib import sha256

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import ChunkRecord
from libs.vector_store.base_vector_store import BaseVectorStore, VectorStoreRecord
from libs.vector_store.vector_store_factory import VectorStoreFactory


class VectorUpserter:
	"""Persist dense vectors with deterministic IDs for idempotent ingestion."""

	def __init__(self, settings: Settings, vector_store: BaseVectorStore | None = None) -> None:
		self._settings = settings
		self._vector_store = vector_store or VectorStoreFactory.create(settings)

	def upsert(self, records: list[ChunkRecord], trace: TraceContext | None = None) -> list[str]:
		"""Convert chunk records to vector-store records and upsert them.

		Args:
			records: Chunk records carrying dense vectors.
			trace: Optional trace context passed through to the vector store.

		Returns:
			Deterministic vector-store record IDs in input order.

		Raises:
			ValueError: If a record has no dense vector.
		"""
		if not records:
			return []

		normalized_records: list[VectorStoreRecord] = []
		normalized_ids: list[str] = []

		for position, record in enumerate(records):
			if record.dense_vector is None:
				raise ValueError(
					f"record at position {position} is missing dense_vector; "
					"run DenseEncoder before VectorUpserter"
				)

			source_path = str(record.metadata.get("source_path", ""))
			chunk_index = int(record.metadata.get("chunk_index", position))
			stable_id = self._build_chunk_id(
				source_path=source_path,
				chunk_index=chunk_index,
				text=record.text,
			)

			metadata = dict(record.metadata)
			metadata["chunk_id"] = stable_id

			normalized_records.append(
				VectorStoreRecord(
					id=stable_id,
					embedding=list(record.dense_vector),
					metadata=metadata,
					text=record.text,
				)
			)
			normalized_ids.append(stable_id)

		self._vector_store.upsert(normalized_records, trace=trace)
		return normalized_ids

	@staticmethod
	def _build_chunk_id(*, source_path: str, chunk_index: int, text: str) -> str:
		content_hash = sha256(text.encode("utf-8")).hexdigest()
		raw = f"{source_path}|{chunk_index}|{content_hash[:8]}"
		return sha256(raw.encode("utf-8")).hexdigest()
