"""Hybrid retrieval orchestration across dense, sparse, and fusion stages."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from core.query_engine.dense_retriever import DenseRetriever
from core.query_engine.fusion import RRFFuser
from core.query_engine.query_processor import QueryProcessor
from core.query_engine.sparse_retriever import SparseRetriever
from core.settings import Settings
from core.types import RetrievalResult

if TYPE_CHECKING:
	from core.trace.trace_context import TraceContext


@dataclass(slots=True)
class HybridSearch:
	"""Coordinate dense and sparse retrieval with graceful degradation."""

	settings: Settings
	query_processor: QueryProcessor | None = None
	dense_retriever: DenseRetriever | None = None
	sparse_retriever: SparseRetriever | None = None
	fusion: RRFFuser | None = None

	def __post_init__(self) -> None:
		if self.query_processor is None:
			self.query_processor = QueryProcessor()
		if self.dense_retriever is None:
			self.dense_retriever = DenseRetriever(self.settings)
		if self.sparse_retriever is None:
			self.sparse_retriever = SparseRetriever(self.settings)
		if self.fusion is None:
			self.fusion = RRFFuser()

	def search(
		self,
		query: str,
		top_k: int,
		filters: dict[str, Any] | None = None,
		trace: TraceContext | None = None,
	) -> list[RetrievalResult]:
		"""Run the hybrid retrieval pipeline and return normalized top-k results."""

		normalized_query = str(query or "").strip()
		if not normalized_query:
			raise ValueError("query must not be empty")
		if top_k <= 0:
			raise ValueError("top_k must be greater than 0")

		assert self.query_processor is not None
		assert self.dense_retriever is not None
		assert self.sparse_retriever is not None
		assert self.fusion is not None

		processed = self.query_processor.process(normalized_query, filters=filters)
		if trace is not None:
			trace.record_stage(
				"query_processing",
				{
					"method": "rule_based",
					"provider": "builtin",
					"details": {
						"keywords_count": len(processed.keywords),
						"filters_count": len(processed.filters),
					},
				},
			)
		candidate_top_k = max(self.settings.retrieval.top_k, top_k * 2)

		dense_results: list[RetrievalResult] = []
		sparse_results: list[RetrievalResult] = []
		dense_error: Exception | None = None
		sparse_error: Exception | None = None

		with ThreadPoolExecutor(max_workers=2) as executor:
			dense_future = executor.submit(
				self.dense_retriever.retrieve,
				processed.raw_query,
				candidate_top_k,
				processed.filters,
				trace,
			)
			sparse_future = executor.submit(
				self.sparse_retriever.retrieve,
				processed.keywords,
				candidate_top_k,
				trace,
			)

			try:
				dense_results = dense_future.result()
				if trace is not None:
					trace.record_stage(
						"dense_retrieval",
						{
							"method": "embedding_vector_store",
							"provider": f"{self.settings.embedding.provider}+{self.settings.vector_store.provider}",
							"details": {
								"top_k": candidate_top_k,
								"filters": dict(processed.filters),
								"result_count": len(dense_results),
							},
						},
					)
			except Exception as exc:  # pragma: no cover - exercised by integration tests
				dense_error = exc
				if trace is not None:
					trace.record_stage(
						"dense_retrieval",
						{
							"method": "embedding_vector_store",
							"provider": f"{self.settings.embedding.provider}+{self.settings.vector_store.provider}",
							"details": {
								"top_k": candidate_top_k,
								"filters": dict(processed.filters),
								"result_count": 0,
								"error": str(exc),
							},
						},
					)

			try:
				sparse_results = sparse_future.result()
				if trace is not None:
					trace.record_stage(
						"sparse_retrieval",
						{
							"method": "bm25",
							"provider": "bm25",
							"details": {
								"top_k": candidate_top_k,
								"keywords": list(processed.keywords),
								"result_count": len(sparse_results),
							},
						},
					)
			except Exception as exc:  # pragma: no cover - exercised by integration tests
				sparse_error = exc
				if trace is not None:
					trace.record_stage(
						"sparse_retrieval",
						{
							"method": "bm25",
							"provider": "bm25",
							"details": {
								"top_k": candidate_top_k,
								"keywords": list(processed.keywords),
								"result_count": 0,
								"error": str(exc),
							},
						},
					)

		if dense_error and sparse_error:
			raise RuntimeError("dense and sparse retrieval both failed") from dense_error

		if dense_results and sparse_results:
			candidates = self.fusion.fuse(dense_results, sparse_results, top_k=candidate_top_k)
			if trace is not None:
				trace.record_stage(
					"fusion",
					{
						"method": "rrf",
						"provider": "rrf",
						"details": {
							"dense_count": len(dense_results),
							"sparse_count": len(sparse_results),
							"candidate_count": len(candidates),
						},
					},
				)
		elif dense_results:
			candidates = list(dense_results)
			if trace is not None:
				trace.record_stage(
					"fusion",
					{
						"method": "passthrough",
						"provider": "dense_only",
						"details": {
							"dense_count": len(dense_results),
							"sparse_count": 0,
							"candidate_count": len(candidates),
						},
					},
				)
		else:
			candidates = list(sparse_results)
			if trace is not None:
				trace.record_stage(
					"fusion",
					{
						"method": "passthrough",
						"provider": "sparse_only",
						"details": {
							"dense_count": 0,
							"sparse_count": len(sparse_results),
							"candidate_count": len(candidates),
						},
					},
				)

		filtered = self._apply_metadata_filters(candidates, processed.filters)
		return filtered[:top_k]

	@staticmethod
	def _apply_metadata_filters(
		candidates: list[RetrievalResult], filters: dict[str, Any] | None
	) -> list[RetrievalResult]:
		"""Apply post-filtering for metadata keys unsupported by lower retrieval layers."""

		if not filters:
			return list(candidates)

		normalized_filters = {
			str(key): value for key, value in dict(filters).items() if value not in (None, "")
		}
		if not normalized_filters:
			return list(candidates)

		filtered: list[RetrievalResult] = []
		for candidate in candidates:
			metadata = candidate.metadata or {}
			include = True
			for key, expected in normalized_filters.items():
				if key not in metadata:
					continue
				if str(metadata[key]) != str(expected):
					include = False
					break
			if include:
				filtered.append(candidate)
		return filtered


__all__ = ["HybridSearch"]
