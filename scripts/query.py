"""Online query CLI entrypoint with HybridSearch + Reranker pipeline."""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from core.query_engine import DenseRetriever, HybridSearch, QueryProcessor, Reranker, SparseRetriever
from core.settings import Settings, load_settings
from core.trace.trace_context import TraceContext
from ingestion.storage.bm25_indexer import BM25Indexer
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.vector_store.vector_store_factory import VectorStoreFactory
from observability.logger import get_logger


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	"""Parse command-line options for online query."""

	parser = argparse.ArgumentParser(description="Query the knowledge base with HybridSearch + Reranker.")
	parser.add_argument("--query", required=True, help="Query text (required).")
	parser.add_argument(
		"--top-k",
		type=int,
		default=10,
		help="Number of results to return (default: 10).",
	)
	parser.add_argument(
		"--collection",
		default=None,
		help="Limit search to specific collection. Defaults to vector_store.collection from settings.",
	)
	parser.add_argument(
		"--verbose",
		action="store_true",
		help="Show intermediate results from each pipeline stage.",
	)
	parser.add_argument(
		"--no-rerank",
		action="store_true",
		help="Skip reranking stage.",
	)
	parser.add_argument(
		"--settings",
		default="config/settings.yaml",
		help="Path to settings YAML file.",
	)
	return parser.parse_args(argv)


def build_components(settings: Settings) -> tuple[QueryProcessor, DenseRetriever, SparseRetriever, HybridSearch, Reranker]:
	"""Build all required components for the query pipeline."""

	embedding_client = EmbeddingFactory.create(settings)
	vector_store = VectorStoreFactory.create(settings)
	bm25_indexer = BM25Indexer.load_or_create(settings)

	query_processor = QueryProcessor(settings=settings)
	dense_retriever = DenseRetriever(embedding_client=embedding_client, vector_store=vector_store, settings=settings)
	sparse_retriever = SparseRetriever(bm25_indexer=bm25_indexer, vector_store=vector_store, settings=settings)

	hybrid_search = HybridSearch(
		query_processor=query_processor,
		dense_retriever=dense_retriever,
		sparse_retriever=sparse_retriever,
		settings=settings,
	)

	reranker = Reranker(settings=settings)

	return query_processor, dense_retriever, sparse_retriever, hybrid_search, reranker


def format_text_summary(text: str, max_length: int = 100) -> str:
	"""Truncate text to summary length."""
	text = text.replace("\n", " ").strip()
	if len(text) > max_length:
		return text[:max_length] + "..."
	return text


def print_result(idx: int, result: object, verbose: bool = False) -> None:
	"""Print a single retrieval result."""
	chunk_id = getattr(result, "chunk_id", "unknown")
	score = getattr(result, "score", 0.0)
	text = getattr(result, "text", "")
	metadata = getattr(result, "metadata", {})

	source_path = metadata.get("source_path", "unknown")
	page = metadata.get("page")

	summary = format_text_summary(text)
	if page is not None:
		print(f"  [{idx}] Score: {score:.4f} | Page: {page} | Source: {source_path}")
	else:
		print(f"  [{idx}] Score: {score:.4f} | Source: {source_path}")
	
	print(f"       {summary}")
	if verbose:
		print(f"       chunk_id: {chunk_id}")


def main(argv: Sequence[str] | None = None) -> int:
	"""CLI entrypoint for online query with complete pipeline."""

	args = parse_args(argv)
	logger = get_logger(__name__)

	if not args.query or not str(args.query).strip():
		logger.error("Query cannot be empty.")
		return 1

	if args.top_k <= 0:
		logger.error("Invalid --top-k=%s. It must be greater than 0.", args.top_k)
		return 1

	try:
		settings = load_settings(args.settings)
	except (FileNotFoundError, ValueError) as exc:
		logger.error("Failed to load settings: %s", exc)
		return 1

	try:
		query_processor, dense_retriever, sparse_retriever, hybrid_search, reranker = build_components(settings)
	except Exception as exc:
		logger.error("Failed to build query components: %s", exc)
		return 1

	collection = args.collection or settings.vector_store.collection
	trace = TraceContext()

	try:
		print("\n" + "=" * 80)
		print(f"Querying: {args.query}")
		print(f"Collection: {collection}")
		print("=" * 80 + "\n")

		# Run HybridSearch
		try:
			hybrid_result = hybrid_search.search(
				query=args.query,
				collection=collection,
				top_k=args.top_k,
				trace=trace,
			)
			candidates = hybrid_result
		except Exception as exc:
			logger.error("HybridSearch failed: %s", exc)
			return 1

		if not candidates:
			print("❌ No results found.")
			print("💡 Tip: Please run `python scripts/ingest.py --path <path>` to ingest documents first.")
			return 0

		if args.verbose:
			print(f"[HybridSearch] Retrieved {len(candidates)} candidates\n")

		# Run Reranker (unless --no-rerank)
		if args.no_rerank:
			final_results = candidates
			print("[Reranker] Skipped (--no-rerank flag)")
		else:
			try:
				rerank_result = reranker.rerank(query=args.query, candidates=candidates, trace=trace)
				final_results = rerank_result.candidates
				if rerank_result.fallback:
					print(f"[Reranker] Fallback activated: {rerank_result.fallback_reason}\n")
				else:
					print(f"[Reranker] Reranked {len(final_results)} candidates\n")
			except Exception as exc:
				logger.error("Reranker failed, using HybridSearch results: %s", exc)
				final_results = candidates

		# Display results
		print("📋 TOP-K RESULTS:")
		print("-" * 80)
		for idx, result in enumerate(final_results[:args.top_k], 1):
			print_result(idx, result, verbose=args.verbose)
		print("-" * 80)
		print(f"\nRetrieved {len(final_results[:args.top_k])} result(s).\n")

		return 0

	except KeyboardInterrupt:
		logger.info("Query interrupted by user.")
		return 130
	except Exception as exc:
		logger.error("Unexpected error during query: %s", exc, exc_info=True)
		return 1


if __name__ == "__main__":
	sys.exit(main())
