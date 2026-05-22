"""Offline ingestion CLI entrypoint."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Sequence

from modular_rag.core.settings import Settings, load_settings
from modular_rag.ingestion.pipeline import IngestionPipeline, IngestionPipelineError
from modular_rag.libs.embedding.base_embedding import BaseEmbedding
from modular_rag.libs.embedding.embedding_factory import EmbeddingFactory
from modular_rag.observability.logger import get_logger


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


def build_pipeline(settings: Settings) -> IngestionPipeline:
	"""Build the default ingestion pipeline for CLI usage."""

	return IngestionPipeline(settings)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
	"""Parse command-line options for ingestion."""

	parser = argparse.ArgumentParser(description="Ingest PDF files into the knowledge index.")
	parser.add_argument("--path", required=True, help="Source PDF file or directory path.")
	parser.add_argument(
		"--collection",
		default=None,
		help="Target collection name. Defaults to vector_store.collection from settings.",
	)
	parser.add_argument("--force", action="store_true", help="Force processing even if unchanged.")
	parser.add_argument(
		"--settings",
		default="config/settings.yaml",
		help="Path to settings YAML file.",
	)
	parser.add_argument(
		"--batch-size",
		type=int,
		default=32,
		help="Batch size for embedding stage.",
	)
	return parser.parse_args(argv)


def _collect_sources(source_path: Path) -> list[Path]:
	if not source_path.exists():
		raise FileNotFoundError(f"Source path not found: {source_path}")

	if source_path.is_file():
		return [source_path]

	if source_path.is_dir():
		return sorted(path for path in source_path.rglob("*.pdf") if path.is_file())

	return []


def main(argv: Sequence[str] | None = None) -> int:
	"""CLI entrypoint for offline ingestion."""

	args = parse_args(argv)
	logger = get_logger(__name__)

	if args.batch_size <= 0:
		logger.error("Invalid --batch-size=%s. It must be greater than 0.", args.batch_size)
		return 1

	try:
		settings = load_settings(args.settings)
	except (FileNotFoundError, ValueError) as exc:
		logger.error("Failed to load settings: %s", exc)
		return 1

	_maybe_enable_local_embedding_fallback(settings, logger)

	source_root = Path(args.path)
	try:
		source_files = _collect_sources(source_root)
	except FileNotFoundError as exc:
		logger.error("%s", exc)
		return 1

	if not source_files:
		logger.error("No PDF files found under path: %s", source_root)
		return 1

	collection = args.collection or settings.vector_store.collection
	pipeline = build_pipeline(settings)
	Path("data/db").mkdir(parents=True, exist_ok=True)

	processed = 0
	skipped = 0
	failed = 0

	for source_file in source_files:
		try:
			result = pipeline.run(
				str(source_file),
				collection=collection,
				force=args.force,
				batch_size=args.batch_size,
			)
		except IngestionPipelineError as exc:
			failed += 1
			logger.error("Ingestion failed for %s: %s", source_file, exc)
			continue
		except Exception as exc:  # pragma: no cover - defensive fallback
			failed += 1
			logger.error("Unexpected ingestion failure for %s: %s", source_file, exc)
			continue

		if result.skipped:
			skipped += 1
			logger.info("Skipped unchanged file: %s", source_file)
		else:
			processed += 1
			logger.info(
				"Ingested %s (chunks=%s, records=%s)",
				source_file,
				result.chunk_count,
				result.record_count,
			)

	logger.info(
		"Ingestion summary collection=%s processed=%s skipped=%s failed=%s",
		collection,
		processed,
		skipped,
		failed,
	)
	return 0 if failed == 0 else 2


if __name__ == "__main__":
	raise SystemExit(main())
