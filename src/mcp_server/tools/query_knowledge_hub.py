"""MCP tool: query the knowledge hub and return cited answers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.query_engine import DenseRetriever, HybridSearch, QueryProcessor, Reranker, SparseRetriever
from core.query_engine.reranker import RerankResult
from core.response.response_builder import ResponseBuilder
from core.settings import Settings, load_settings
from ingestion.storage.bm25_indexer import BM25Indexer
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.vector_store.vector_store_factory import VectorStoreFactory


def _build_components(settings: Settings) -> tuple[HybridSearch, Reranker]:
	"""Build search and rerank components from project settings."""

	embedding_client = EmbeddingFactory.create(settings)
	vector_store = VectorStoreFactory.create(settings)
	bm25_indexer = BM25Indexer()
	if bm25_indexer.index_path.exists():
		bm25_indexer.load()

	query_processor = QueryProcessor()
	dense_retriever = DenseRetriever(settings=settings, embedding_client=embedding_client, vector_store=vector_store)
	sparse_retriever = SparseRetriever(settings=settings, bm25_indexer=bm25_indexer, vector_store=vector_store)

	hybrid_search = HybridSearch(
		settings=settings,
		query_processor=query_processor,
		dense_retriever=dense_retriever,
		sparse_retriever=sparse_retriever,
	)
	reranker = Reranker(settings=settings)
	return hybrid_search, reranker


def query_knowledge_hub(
	query: str,
	*,
	top_k: int,
	collection: str | None,
	hybrid_search: HybridSearch,
	reranker: Reranker,
	response_builder: ResponseBuilder,
) -> dict[str, Any]:
	"""Execute retrieval + rerank and return MCP tool response payload."""

	normalized_query = str(query or "").strip()
	if not normalized_query:
		raise ValueError("query must not be empty")
	if int(top_k) <= 0:
		raise ValueError("top_k must be greater than 0")

	filters: dict[str, Any] | None = None
	if collection is not None and str(collection).strip():
		filters = {"collection": str(collection).strip()}

	candidates = hybrid_search.search(
		query=normalized_query,
		top_k=int(top_k),
		filters=filters,
	)
	rerank_result: RerankResult = reranker.rerank(query=normalized_query, candidates=candidates)

	return response_builder.build(
		retrieval_results=rerank_result.candidates,
		query=normalized_query,
		fallback_reason=rerank_result.fallback_reason if rerank_result.fallback else None,
	)


@dataclass(slots=True)
class _QueryToolRuntime:
	settings_path: str = "config/settings.yaml"
	settings: Settings | None = None
	hybrid_search: HybridSearch | None = None
	reranker: Reranker | None = None
	response_builder: ResponseBuilder | None = None

	def ensure_initialized(self) -> None:
		if self.settings is not None and self.hybrid_search is not None and self.reranker is not None:
			if self.response_builder is None:
				self.response_builder = ResponseBuilder()
			return

		self.settings = load_settings(self.settings_path)
		self.hybrid_search, self.reranker = _build_components(self.settings)
		self.response_builder = ResponseBuilder()


def build_query_tool_handler(settings_path: str = "config/settings.yaml") -> Any:
	"""Create a protocol-compatible tool handler callable."""

	runtime = _QueryToolRuntime(settings_path=settings_path)

	def _handler(arguments: dict[str, Any]) -> dict[str, Any]:
		runtime.ensure_initialized()
		assert runtime.settings is not None
		assert runtime.hybrid_search is not None
		assert runtime.reranker is not None
		assert runtime.response_builder is not None

		query = arguments.get("query")
		if not isinstance(query, str) or not query.strip():
			raise ValueError("query must be a non-empty string")

		raw_top_k = arguments.get("top_k", runtime.settings.retrieval.top_k)
		try:
			top_k = int(raw_top_k)
		except (TypeError, ValueError) as exc:
			raise ValueError("top_k must be an integer") from exc

		collection = arguments.get("collection")
		if collection is not None and not isinstance(collection, str):
			raise ValueError("collection must be a string")

		return query_knowledge_hub(
			query=query,
			top_k=top_k,
			collection=collection,
			hybrid_search=runtime.hybrid_search,
			reranker=runtime.reranker,
			response_builder=runtime.response_builder,
		)

	return _handler


__all__ = ["build_query_tool_handler", "query_knowledge_hub"]
