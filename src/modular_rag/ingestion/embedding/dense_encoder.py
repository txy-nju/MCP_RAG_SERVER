"""Dense encoder implementation using BaseEmbedding for batch vectorization."""

from __future__ import annotations

from modular_rag.core.settings import Settings
from modular_rag.core.trace.trace_context import TraceContext
from modular_rag.core.types import Chunk, ChunkRecord
from modular_rag.libs.embedding.base_embedding import BaseEmbedding
from modular_rag.libs.embedding.embedding_factory import EmbeddingFactory


class DenseEncoder:
	"""Batch encode chunks using a pluggable embedding provider.

	Responsibilities:
	1. Initialize embedding client from settings
	2. Batch embed chunk texts
	3. Pair embeddings with chunks to create ChunkRecords
	4. Maintain deterministic ordering and count
	"""

	def __init__(self, settings: Settings, embedding: BaseEmbedding | None = None) -> None:
		"""Initialize encoder with settings or injected embedding instance.

		Args:
			settings: Project settings (used if embedding is None)
			embedding: Optional pre-built embedding client (for testing/DI)
		"""
		self._settings = settings
		self._embedding = embedding or EmbeddingFactory.create(settings)

	def encode(
		self, chunks: list[Chunk], trace: TraceContext | None = None
	) -> list[ChunkRecord]:
		"""Encode chunks to dense vectors using the embedding client.

		Args:
			chunks: List of input Chunk objects
			trace: Optional trace context for recording metrics

		Returns:
			List of ChunkRecord objects paired with dense vectors

		Raises:
			ValueError: If chunks list is empty
		"""
		if not chunks:
			raise ValueError("Cannot encode an empty chunks list")

		# Extract text and embed
		texts = [chunk.text for chunk in chunks]
		dense_vectors = self._embedding.embed(texts, trace=trace)

		# Verify output dimensions match input
		if len(dense_vectors) != len(chunks):
			raise ValueError(
				f"Embedding output count ({len(dense_vectors)}) does not match "
				f"input chunk count ({len(chunks)})"
			)

		# Pair chunks with vectors to create ChunkRecords
		records: list[ChunkRecord] = []
		for chunk, vector in zip(chunks, dense_vectors):
			record = ChunkRecord.from_chunk(chunk, dense_vector=vector)
			records.append(record)

		return records
