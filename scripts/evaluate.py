"""CLI entrypoint for retrieval evaluation against a golden test set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Sequence

from core.query_engine import DenseRetriever, HybridSearch, QueryProcessor, SparseRetriever
from core.settings import Settings, load_settings
from ingestion.storage.bm25_indexer import BM25Indexer
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.evaluator.evaluator_factory import EvaluatorFactory
from libs.vector_store.vector_store_factory import VectorStoreFactory
from observability.evaluation import EvalRunner
from observability.logger import get_logger


class _LocalDeterministicEmbedding(BaseEmbedding):
	"""Fallback embedding used when API settings are placeholders."""

	def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
		del trace
		vectors: list[list[float]] = []
		for text in texts:
			digest = hashlib.sha256(text.encode("utf-8")).digest()
			vec = [round(digest[index % len(digest)] / 255.0, 6) for index in range(1536)]
			vectors.append(vec)
		return vectors


def _maybe_enable_local_embedding_fallback(settings: Settings, logger: object) -> None:
	provider = str(settings.embedding.provider or "").strip().lower()
	api_key = str(settings.embedding.api_key or "").strip().lower()
	api_url = str(settings.embedding.api_url or "").strip().lower()

	if provider not in {"openai", "azure"}:
		return
	if api_key != "your-api-key" and api_url != "your-api-url":
		return

	EmbeddingFactory.register("local_fake", _LocalDeterministicEmbedding)
	settings.embedding.provider = "local_fake"
	settings.embedding.model = "deterministic-local"
	settings.embedding.api_key = None
	settings.embedding.api_url = None
	logger.warning("Detected placeholder embedding API settings. Using local deterministic embedding fallback.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate retrieval quality using a golden test set.")
	parser.add_argument(
		"--test-set",
		default="tests/fixtures/golden_test_set.json",
		help="Path to golden test set JSON file.",
	)
	parser.add_argument(
		"--settings",
		default="config/settings.yaml",
		help="Path to settings YAML file.",
	)
	parser.add_argument(
		"--json",
		action="store_true",
		help="Print evaluation report as JSON.",
	)
	return parser.parse_args(argv)


def build_hybrid_search(settings: Settings) -> HybridSearch:
	embedding_client = EmbeddingFactory.create(settings)
	vector_store = VectorStoreFactory.create(settings)
	bm25_indexer = BM25Indexer()
	if bm25_indexer.index_path.exists():
		bm25_indexer.load()

	query_processor = QueryProcessor()
	dense_retriever = DenseRetriever(embedding_client=embedding_client, vector_store=vector_store, settings=settings)
	sparse_retriever = SparseRetriever(bm25_indexer=bm25_indexer, vector_store=vector_store, settings=settings)

	return HybridSearch(
		query_processor=query_processor,
		dense_retriever=dense_retriever,
		sparse_retriever=sparse_retriever,
		settings=settings,
	)


def main(argv: Sequence[str] | None = None) -> int:
	args = parse_args(argv)
	logger = get_logger(__name__)

	try:
		settings = load_settings(args.settings)
	except (FileNotFoundError, ValueError) as exc:
		logger.error("Failed to load settings: %s", exc)
		return 1

	_maybe_enable_local_embedding_fallback(settings, logger)

	test_set_path = Path(args.test_set)
	if not test_set_path.exists():
		logger.error("Golden test set not found: %s", test_set_path)
		return 1

	try:
		hybrid_search = build_hybrid_search(settings)
		evaluator = EvaluatorFactory.create(settings)
		runner = EvalRunner(settings=settings, hybrid_search=hybrid_search, evaluator=evaluator)
		report = runner.run(test_set_path)
	except Exception as exc:
		logger.error("Evaluation failed: %s", exc)
		return 1

	if args.json:
		print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
		return 0

	print("\n" + "=" * 80)
	print("Evaluation Report")
	print("=" * 80)
	print(f"Test set: {test_set_path}")
	print(f"Queries: {len(report.cases)}")
	print("\nAggregate Metrics:")
	for metric_name, metric_value in report.metrics.items():
		print(f"  - {metric_name}: {metric_value:.4f}")

	print("\nPer-query Results:")
	for case in report.cases:
		print(f"  - Query: {case.query}")
		print(f"    Retrieved IDs: {', '.join(case.retrieved_chunk_ids) or 'none'}")
		print(f"    Expected IDs: {', '.join(case.expected_chunk_ids)}")
		metric_summary = ", ".join(
			f"{metric_name}={metric_value:.4f}" for metric_name, metric_value in case.metrics.items()
		)
		print(f"    Metrics: {metric_summary}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
