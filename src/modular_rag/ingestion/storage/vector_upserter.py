"""Vector upserter for deterministic IDs and idempotent vector-store writes."""

from __future__ import annotations

from modular_rag.core.settings import Settings
from modular_rag.core.trace.trace_context import TraceContext
from modular_rag.core.types import ChunkRecord
from modular_rag.libs.vector_store.base_vector_store import BaseVectorStore, VectorStoreRecord
from modular_rag.libs.vector_store.vector_store_factory import VectorStoreFactory


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
			Record IDs in input order, matching DocumentChunker IDs so that
			BM25 and Chroma share the same chunk identity for hybrid search.

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

			# Use record.id directly instead of regenerating a different ID.
			# This ensures Chroma and BM25 share the same chunk identity,
			# which is critical for SparseRetriever's get_by_ids lookup.
			chunk_id = record.id
			metadata = dict(record.metadata)
			metadata["chunk_id"] = chunk_id

			normalized_records.append(
				VectorStoreRecord(
					id=chunk_id,
					embedding=list(record.dense_vector),
					metadata=metadata,
					text=record.text,
				)
			)
			normalized_ids.append(chunk_id)

		self._vector_store.upsert(normalized_records, trace=trace)
		return normalized_ids
