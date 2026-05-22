"""Batch processor orchestrating dense/sparse encoding in stable mini-batches."""

from __future__ import annotations

from time import perf_counter

from modular_rag.core.settings import Settings
from modular_rag.core.trace.trace_context import TraceContext
from modular_rag.core.types import Chunk, ChunkRecord
from modular_rag.ingestion.embedding.dense_encoder import DenseEncoder
from modular_rag.ingestion.embedding.sparse_encoder import SparseEncoder


class BatchProcessor:
	"""Split chunks into mini-batches and run dense/sparse encoders in lockstep."""

	def __init__(
		self,
		settings: Settings,
		dense_encoder: DenseEncoder | None = None,
		sparse_encoder: SparseEncoder | None = None,
	) -> None:
		self._settings = settings
		self._dense_encoder = dense_encoder or DenseEncoder(settings)
		self._sparse_encoder = sparse_encoder or SparseEncoder(settings)

	def process(
		self,
		chunks: list[Chunk],
		*,
		batch_size: int,
		trace: TraceContext | None = None,
	) -> list[ChunkRecord]:
		"""Encode chunks in batches while preserving global input order.

		Args:
			chunks: Source chunks to process.
			batch_size: Number of chunks per batch, must be > 0.
			trace: Optional trace context used to record batch-level metrics.

		Returns:
			Merged ChunkRecords containing both dense and sparse vectors.

		Raises:
			ValueError: If batch_size <= 0.
			ValueError: If dense/sparse encoder output sizes mismatch.
			ValueError: If dense/sparse encoder record IDs mismatch.
		"""
		if batch_size <= 0:
			raise ValueError("batch_size must be greater than 0")
		if not chunks:
			return []

		results: list[ChunkRecord] = []
		total_batches = (len(chunks) + batch_size - 1) // batch_size

		for batch_index, start in enumerate(range(0, len(chunks), batch_size), start=1):
			batch = chunks[start : start + batch_size]
			t0 = perf_counter()

			dense_records = self._dense_encoder.encode(batch, trace=trace)
			sparse_records = self._sparse_encoder.encode(batch, trace=trace)
			merged_batch = self._merge_records(dense_records, sparse_records)
			results.extend(merged_batch)

			if trace is not None:
				trace.record_stage(
					"embedding_batch",
					{
						"batch_index": batch_index,
						"batch_size": len(batch),
						"total_batches": total_batches,
						"elapsed_ms": round((perf_counter() - t0) * 1000, 3),
						"method": "dense_plus_sparse",
					},
				)

		return results

	@staticmethod
	def _merge_records(
		dense_records: list[ChunkRecord], sparse_records: list[ChunkRecord]
	) -> list[ChunkRecord]:
		if len(dense_records) != len(sparse_records):
			raise ValueError(
				"Dense and sparse record counts must match per batch "
				f"(dense={len(dense_records)}, sparse={len(sparse_records)})"
			)

		merged: list[ChunkRecord] = []
		for dense_record, sparse_record in zip(dense_records, sparse_records):
			if dense_record.id != sparse_record.id:
				raise ValueError(
					"Dense and sparse record IDs must align per position "
					f"(dense='{dense_record.id}', sparse='{sparse_record.id}')"
				)

			merged.append(
				ChunkRecord(
					id=dense_record.id,
					text=dense_record.text,
					metadata=dense_record.metadata,
					dense_vector=dense_record.dense_vector,
					sparse_vector=sparse_record.sparse_vector,
				)
			)

		return merged
