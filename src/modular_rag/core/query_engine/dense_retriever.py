"""Dense retrieval orchestration using embedding + vector store backends."""

from __future__ import annotations

import logging

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modular_rag.core.settings import Settings
from modular_rag.core.types import RetrievalResult
from modular_rag.libs.embedding.base_embedding import BaseEmbedding
from modular_rag.libs.embedding.embedding_factory import EmbeddingFactory
from modular_rag.libs.vector_store.base_vector_store import BaseVectorStore, VectorStoreQueryResult
from modular_rag.libs.vector_store.vector_store_factory import VectorStoreFactory

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
	from modular_rag.core.trace.trace_context import TraceContext


@dataclass(slots=True)
class DenseRetriever:
	"""Retrieve semantically similar chunks from vector store."""

	settings: Settings
	embedding_client: BaseEmbedding | None = None
	vector_store: BaseVectorStore | None = None

	def __post_init__(self) -> None:
		if self.embedding_client is None:
			self.embedding_client = EmbeddingFactory.create(self.settings)
		if self.vector_store is None:
			self.vector_store = VectorStoreFactory.create(self.settings)

	def retrieve(
		self,
		query: str,
		top_k: int,
		filters: dict[str, Any] | None = None,
		trace: TraceContext | None = None,
	) -> list[RetrievalResult]:
		"""Run dense retrieval for a single user query."""

		normalized_query = str(query or "").strip()
		if not normalized_query:
			raise ValueError("query must not be empty")
		if top_k <= 0:
			raise ValueError("top_k must be greater than 0")

		assert self.embedding_client is not None
		assert self.vector_store is not None

		embeddings = self.embedding_client.embed([normalized_query], trace=trace)
		if not embeddings or not embeddings[0]:
			raise ValueError("embedding provider returned empty query vector")

		vector = [float(value) for value in embeddings[0]]
		raw_results = self.vector_store.query(vector=vector, top_k=top_k, filters=filters, trace=trace)
		logger.debug("DenseRetriever: top_k=%d filters=%s result_count=%d", top_k, filters, len(raw_results))
		return [self._to_retrieval_result(item) for item in raw_results]

	@staticmethod
	def _to_retrieval_result(item: VectorStoreQueryResult | dict[str, Any]) -> RetrievalResult:
		if isinstance(item, dict):
			chunk_id = str(item.get("id", ""))
			score = float(item.get("score", 0.0))
			text = str(item.get("text") or "")
			metadata = dict(item.get("metadata") or {})
		else:
			chunk_id = str(item.id)
			score = float(item.score)
			text = str(item.text or "")
			metadata = dict(item.metadata or {})

		return RetrievalResult(chunk_id=chunk_id, score=score, text=text, metadata=metadata)


__all__ = ["DenseRetriever"]
