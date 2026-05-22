"""Sparse retrieval orchestration using BM25 index + vector store payload lookup."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from modular_rag.core.settings import Settings
from modular_rag.core.types import RetrievalResult
from modular_rag.ingestion.storage.bm25_indexer import BM25Indexer
from modular_rag.libs.vector_store.base_vector_store import BaseVectorStore
from modular_rag.libs.vector_store.vector_store_factory import VectorStoreFactory

if TYPE_CHECKING:
	from modular_rag.core.trace.trace_context import TraceContext


@dataclass(slots=True)
class SparseRetriever:
	"""Retrieve keyword-matching chunks through BM25 and stored payload lookup."""

	settings: Settings
	bm25_indexer: BM25Indexer | None = None
	vector_store: BaseVectorStore | None = None

	def __post_init__(self) -> None:
		if self.bm25_indexer is None:
			self.bm25_indexer = BM25Indexer()
			if Path(self.bm25_indexer.index_path).exists():
				self.bm25_indexer.load()
		if self.vector_store is None:
			self.vector_store = VectorStoreFactory.create(self.settings)

	def retrieve(
		self,
		keywords: list[str],
		top_k: int,
		trace: TraceContext | None = None,
	) -> list[RetrievalResult]:
		"""Run sparse retrieval for pre-extracted keywords."""

		if top_k <= 0:
			raise ValueError("top_k must be greater than 0")
		normalized_keywords = [str(keyword).strip().lower() for keyword in keywords if str(keyword).strip()]
		if not normalized_keywords:
			return []

		assert self.bm25_indexer is not None
		assert self.vector_store is not None

		bm25_hits = self.bm25_indexer.query(normalized_keywords, top_k=top_k)
		if not bm25_hits:
			return []

		chunk_ids = [str(hit["chunk_id"]) for hit in bm25_hits]
		payloads = self.vector_store.get_by_ids(chunk_ids, trace=trace)
		payload_by_id = {payload.id: payload for payload in payloads}

		results: list[RetrievalResult] = []
		for hit in bm25_hits:
			chunk_id = str(hit["chunk_id"])
			payload = payload_by_id.get(chunk_id)
			if payload is None:
				continue
			results.append(
				RetrievalResult(
					chunk_id=chunk_id,
					score=float(hit["score"]),
					text=str(payload.text or ""),
					metadata=dict(payload.metadata or {}),
				)
			)

		return results


__all__ = ["SparseRetriever"]
