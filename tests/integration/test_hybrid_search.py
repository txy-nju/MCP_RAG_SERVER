"""Integration-style tests for HybridSearch orchestration (D5)."""

from __future__ import annotations

from typing import cast

import pytest

from core.query_engine.dense_retriever import DenseRetriever
from core.query_engine.fusion import RRFFuser
from core.query_engine.hybrid_search import HybridSearch
from core.query_engine.query_processor import QueryProcessor
from core.query_engine.reranker import Reranker
from core.query_engine.sparse_retriever import SparseRetriever
from core.settings import (
	EmbeddingSettings,
	EvaluationSettings,
	LLMSettings,
	ObservabilitySettings,
	RerankSettings,
	RetrievalSettings,
	Settings,
	SplitterSettings,
	VectorStoreSettings,
)
from core.trace.trace_context import TraceContext
from core.types import RetrievalResult


class FakeDenseRetriever:
	def __init__(self, results: list[RetrievalResult] | None = None, error: Exception | None = None) -> None:
		self.results = list(results or [])
		self.error = error
		self.calls: list[dict[str, object]] = []

	def retrieve(self, query: str, top_k: int, filters=None, trace=None) -> list[RetrievalResult]:
		self.calls.append({"query": query, "top_k": top_k, "filters": filters, "trace": trace})
		if self.error is not None:
			raise self.error
		return list(self.results)


class FakeSparseRetriever:
	def __init__(self, results: list[RetrievalResult] | None = None, error: Exception | None = None) -> None:
		self.results = list(results or [])
		self.error = error
		self.calls: list[dict[str, object]] = []

	def retrieve(self, keywords: list[str], top_k: int, trace=None) -> list[RetrievalResult]:
		self.calls.append({"keywords": keywords, "top_k": top_k, "trace": trace})
		if self.error is not None:
			raise self.error
		return list(self.results)


def _make_settings() -> Settings:
	return Settings(
		llm=LLMSettings(provider="openai", model="gpt-4o-mini"),
		embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
		splitter=SplitterSettings(provider="recursive", chunk_size=1000, chunk_overlap=200),
		vector_store=VectorStoreSettings(provider="chroma", collection="demo", persist_path="data/db/chroma"),
		retrieval=RetrievalSettings(top_k=5),
		rerank=RerankSettings(provider="none"),
		evaluation=EvaluationSettings(backend="custom"),
		observability=ObservabilitySettings(log_level="INFO", trace_file="logs/traces.jsonl"),
	)


def _result(chunk_id: str, score: float, **metadata: str) -> RetrievalResult:
	return RetrievalResult(
		chunk_id=chunk_id,
		score=score,
		text=f"text-{chunk_id}",
		metadata={"source_path": f"{chunk_id}.md", **metadata},
	)


@pytest.mark.integration
def test_hybrid_search_fuses_dense_and_sparse_results_with_filters() -> None:
	dense = FakeDenseRetriever(
		results=[
			_result("dense-only", 0.91, collection="demo", doc_type="pdf"),
			_result("shared", 0.87, collection="demo", doc_type="pdf"),
		]
	)
	sparse = FakeSparseRetriever(
		results=[
			_result("shared", 2.0, collection="demo", doc_type="pdf"),
			_result("filtered-out", 1.5, collection="other", doc_type="pdf"),
		]
	)
	search = HybridSearch(
		_make_settings(),
		query_processor=QueryProcessor(),
		dense_retriever=cast(DenseRetriever, dense),
		sparse_retriever=cast(SparseRetriever, sparse),
		fusion=RRFFuser(rrf_k=60),
	)

	results = search.search("find pipeline collection:demo", top_k=3, filters={"doc_type": "pdf"})

	assert dense.calls[0]["query"] == "find pipeline collection:demo"
	assert dense.calls[0]["filters"] == {"collection": "demo", "doc_type": "pdf"}
	assert sparse.calls[0]["keywords"] == ["find", "pipeline"]
	assert [item.chunk_id for item in results] == ["shared", "dense-only"]
	assert all(item.metadata.get("collection") == "demo" for item in results)


@pytest.mark.integration
def test_hybrid_search_degrades_to_sparse_when_dense_route_fails() -> None:
	search = HybridSearch(
		_make_settings(),
		query_processor=QueryProcessor(),
		dense_retriever=cast(DenseRetriever, FakeDenseRetriever(error=RuntimeError("dense unavailable"))),
		sparse_retriever=cast(SparseRetriever, FakeSparseRetriever(
			results=[
				_result("s1", 1.0, collection="demo"),
				_result("s2", 0.8, collection="demo"),
			]
		)),
		fusion=RRFFuser(),
	)

	results = search.search("demo query", top_k=1, filters={"collection": "demo"})

	assert [item.chunk_id for item in results] == ["s1"]
	assert results[0].score == 1.0


@pytest.mark.integration
def test_hybrid_search_raises_when_both_routes_fail() -> None:
	search = HybridSearch(
		_make_settings(),
		query_processor=QueryProcessor(),
		dense_retriever=cast(DenseRetriever, FakeDenseRetriever(error=RuntimeError("dense unavailable"))),
		sparse_retriever=cast(SparseRetriever, FakeSparseRetriever(error=RuntimeError("sparse unavailable"))),
		fusion=RRFFuser(),
	)

	with pytest.raises(RuntimeError, match="dense and sparse retrieval both failed"):
		search.search("demo query", top_k=3)


@pytest.mark.integration
def test_hybrid_search_rejects_invalid_input() -> None:
	search = HybridSearch(
		_make_settings(),
		query_processor=QueryProcessor(),
		dense_retriever=cast(DenseRetriever, FakeDenseRetriever()),
		sparse_retriever=cast(SparseRetriever, FakeSparseRetriever()),
		fusion=RRFFuser(),
	)

	with pytest.raises(ValueError, match="query must not be empty"):
		search.search("   ", top_k=1)

	with pytest.raises(ValueError, match="top_k must be greater than 0"):
		search.search("query", top_k=0)


@pytest.mark.integration
def test_query_trace_contains_all_required_stages() -> None:
	settings = _make_settings()
	trace = TraceContext(trace_type="query")

	dense = FakeDenseRetriever(
		results=[
			_result("dense-only", 0.91, collection="demo", doc_type="pdf"),
			_result("shared", 0.87, collection="demo", doc_type="pdf"),
		]
	)
	sparse = FakeSparseRetriever(
		results=[
			_result("shared", 2.0, collection="demo", doc_type="pdf"),
			_result("sparse-only", 1.5, collection="demo", doc_type="pdf"),
		]
	)

	search = HybridSearch(
		settings,
		query_processor=QueryProcessor(),
		dense_retriever=cast(DenseRetriever, dense),
		sparse_retriever=cast(SparseRetriever, sparse),
		fusion=RRFFuser(rrf_k=60),
	)
	reranker = Reranker(settings=settings)

	candidates = search.search("find pipeline collection:demo", top_k=3, trace=trace)
	rerank_result = reranker.rerank("find pipeline collection:demo", candidates, trace=trace)

	stage_names = [item["stage"] for item in trace.stages]
	assert stage_names == [
		"query_processing",
		"dense_retrieval",
		"sparse_retrieval",
		"fusion",
		"rerank",
	]
	assert len(rerank_result.candidates) >= 1

	for stage in trace.stages:
		assert "elapsed_ms" in stage
		assert isinstance(stage.get("elapsed_ms"), float)
		assert "method" in stage["data"]

	trace_payload = trace.to_dict()
	assert trace_payload["trace_type"] == "query"
