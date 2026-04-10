"""Ingestion pipeline orchestration (C14 MVP)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord, Document
from ingestion.chunking.document_chunker import DocumentChunker
from ingestion.embedding.batch_processor import BatchProcessor
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from ingestion.storage.vector_upserter import VectorUpserter
from ingestion.transform.base_transform import BaseTransform
from ingestion.transform.chunk_refiner import ChunkRefiner
from ingestion.transform.image_captioner import ImageCaptioner
from ingestion.transform.metadata_enricher import MetadataEnricher
from libs.loader.base_loader import BaseLoader
from libs.loader.file_integrity import FileIntegrityChecker, SQLiteIntegrityChecker
from libs.loader.pdf_loader import PdfLoader
from observability.logger import get_logger


class ProgressCallback(Protocol):
	"""Callable contract for external progress observers."""

	def __call__(self, stage_name: str, current: int, total: int) -> None: ...


@dataclass(slots=True)
class IngestionResult:
	"""Normalized result payload for one ingestion pipeline run."""

	source_path: str
	file_hash: str
	collection: str
	skipped: bool
	document_id: str | None = None
	chunk_count: int = 0
	record_count: int = 0
	upserted_ids: list[str] = field(default_factory=list)
	saved_image_count: int = 0
	trace: TraceContext | None = None


class IngestionPipelineError(RuntimeError):
	"""Raised when pipeline execution fails at a known stage."""

	def __init__(self, stage: str, message: str) -> None:
		super().__init__(f"Ingestion pipeline failed at stage '{stage}': {message}")
		self.stage = stage


class IngestionPipeline:
	"""MVP ingestion orchestration: integrity -> load -> split -> transform -> encode -> store."""

	_TOTAL_STAGES = 6

	def __init__(
		self,
		settings: Settings,
		*,
		integrity_checker: FileIntegrityChecker | None = None,
		loader: BaseLoader | None = None,
		chunker: DocumentChunker | None = None,
		transforms: list[BaseTransform] | None = None,
		batch_processor: BatchProcessor | None = None,
		vector_upserter: VectorUpserter | None = None,
		bm25_indexer: BM25Indexer | None = None,
		image_storage: ImageStorage | None = None,
	) -> None:
		self._settings = settings
		self._integrity_checker = integrity_checker or SQLiteIntegrityChecker()
		self._loader = loader or PdfLoader()
		self._chunker = chunker or DocumentChunker(settings)
		self._transforms = transforms or [
			ChunkRefiner(settings),
			MetadataEnricher(settings),
			ImageCaptioner(settings),
		]
		self._batch_processor = batch_processor or BatchProcessor(settings)
		self._vector_upserter = vector_upserter or VectorUpserter(settings)
		self._bm25_indexer = bm25_indexer or BM25Indexer()
		self._image_storage = image_storage or ImageStorage()
		self._logger = get_logger(__name__)

	def run(
		self,
		source_path: str,
		collection: str = "default",
		*,
		force: bool = False,
		batch_size: int = 32,
		trace: TraceContext | None = None,
		on_progress: ProgressCallback | None = None,
	) -> IngestionResult:
		"""Execute the full ingestion flow for a source file."""

		trace_ctx = trace or TraceContext(trace_type="ingestion")
		file_hash = self._run_integrity_check(
			source_path=source_path,
			force=force,
			trace=trace_ctx,
			on_progress=on_progress,
		)
		if file_hash is None:
			return IngestionResult(
				source_path=source_path,
				file_hash="",
				collection=collection,
				skipped=True,
				trace=trace_ctx,
			)

		try:
			document = self._run_load_stage(source_path, trace_ctx, on_progress)
			chunks = self._run_split_stage(document, collection, trace_ctx, on_progress)
			transformed_chunks = self._run_transform_stage(chunks, trace_ctx, on_progress)
			records = self._run_encode_stage(transformed_chunks, batch_size, trace_ctx, on_progress)
			upserted_ids, saved_image_count = self._run_store_stage(
				document=document,
				records=records,
				collection=collection,
				trace=trace_ctx,
				on_progress=on_progress,
			)

			self._integrity_checker.mark_success(
				file_hash,
				source_path,
				chunk_count=len(transformed_chunks),
				record_count=len(records),
				collection=collection,
			)
			self._logger.info(
				"Ingestion completed source=%s chunks=%s records=%s collection=%s",
				source_path,
				len(transformed_chunks),
				len(records),
				collection,
			)
			return IngestionResult(
				source_path=source_path,
				file_hash=file_hash,
				collection=collection,
				skipped=False,
				document_id=document.id,
				chunk_count=len(transformed_chunks),
				record_count=len(records),
				upserted_ids=upserted_ids,
				saved_image_count=saved_image_count,
				trace=trace_ctx,
			)
		except Exception as exc:
			self._integrity_checker.mark_failed(file_hash, str(exc))
			raise

	def _run_integrity_check(
		self,
		*,
		source_path: str,
		force: bool,
		trace: TraceContext,
		on_progress: ProgressCallback | None,
	) -> str | None:
		stage = "integrity"
		try:
			file_hash = self._integrity_checker.compute_sha256(source_path)
			if not force and self._integrity_checker.should_skip(file_hash):
				trace.record_stage(stage, {"source_path": source_path, "skip": True})
				self._emit_progress(on_progress, stage, 1)
				self._logger.info("Skipping unchanged file source=%s", source_path)
				return None

			trace.record_stage(stage, {"source_path": source_path, "skip": False})
			self._emit_progress(on_progress, stage, 1)
			return file_hash
		except Exception as exc:
			raise IngestionPipelineError(stage, str(exc)) from exc

	def _run_load_stage(
		self,
		source_path: str,
		trace: TraceContext,
		on_progress: ProgressCallback | None,
	) -> Document:
		stage = "load"
		try:
			document = self._loader.load(source_path)
			trace.record_stage(stage, {"document_id": document.id, "text_chars": len(document.text)})
			self._emit_progress(on_progress, stage, 2)
			return document
		except Exception as exc:
			raise IngestionPipelineError(stage, str(exc)) from exc

	def _run_split_stage(
		self,
		document: Document,
		collection: str,
		trace: TraceContext,
		on_progress: ProgressCallback | None,
	) -> list[Chunk]:
		stage = "split"
		try:
			chunks = self._chunker.split_document(document)
			for chunk in chunks:
				chunk.metadata.setdefault("collection", collection)
			trace.record_stage(stage, {"chunk_count": len(chunks)})
			self._emit_progress(on_progress, stage, 3)
			return chunks
		except Exception as exc:
			raise IngestionPipelineError(stage, str(exc)) from exc

	def _run_transform_stage(
		self,
		chunks: list[Chunk],
		trace: TraceContext,
		on_progress: ProgressCallback | None,
	) -> list[Chunk]:
		stage = "transform"
		try:
			transformed = chunks
			for transform in self._transforms:
				transformed = transform.transform(transformed, trace=trace)
			trace.record_stage(stage, {"chunk_count": len(transformed), "transform_steps": len(self._transforms)})
			self._emit_progress(on_progress, stage, 4)
			return transformed
		except Exception as exc:
			raise IngestionPipelineError(stage, str(exc)) from exc

	def _run_encode_stage(
		self,
		chunks: list[Chunk],
		batch_size: int,
		trace: TraceContext,
		on_progress: ProgressCallback | None,
	) -> list[ChunkRecord]:
		stage = "encode"
		try:
			records = self._batch_processor.process(chunks, batch_size=batch_size, trace=trace)
			trace.record_stage(stage, {"record_count": len(records), "batch_size": batch_size})
			self._emit_progress(on_progress, stage, 5)
			return records
		except Exception as exc:
			raise IngestionPipelineError(stage, str(exc)) from exc

	def _run_store_stage(
		self,
		*,
		document: Document,
		records: list[ChunkRecord],
		collection: str,
		trace: TraceContext,
		on_progress: ProgressCallback | None,
	) -> tuple[list[str], int]:
		stage = "store"
		try:
			upserted_ids = self._vector_upserter.upsert(records, trace=trace)
			self._bm25_indexer.build(records, rebuild=False)
			saved_images = self._persist_images(document=document, collection=collection)
			trace.record_stage(
				stage,
				{
					"upserted_count": len(upserted_ids),
					"bm25_index_path": str(self._bm25_indexer.index_path),
					"saved_image_count": saved_images,
				},
			)
			self._emit_progress(on_progress, stage, 6)
			return upserted_ids, saved_images
		except Exception as exc:
			raise IngestionPipelineError(stage, str(exc)) from exc

	def _persist_images(self, *, document: Document, collection: str) -> int:
		images = document.metadata.get("images", [])
		if not isinstance(images, list):
			return 0

		saved = 0
		for image in images:
			if not isinstance(image, dict):
				continue
			image_id = str(image.get("id", "")).strip()
			image_path = str(image.get("path", "")).strip()
			if not image_id or not image_path:
				continue

			try:
				with open(image_path, "rb") as handle:
					image_bytes = handle.read()
				extension = image_path.rsplit(".", 1)[-1] if "." in image_path else "png"
				self._image_storage.save_image(
					image_id,
					image_bytes,
					collection=collection,
					doc_hash=document.id,
					page_num=image.get("page"),
					extension=extension,
				)
				saved += 1
			except Exception as exc:
				self._logger.warning("Image persist skipped image_id=%s reason=%s", image_id, exc)
		return saved

	def _emit_progress(
		self,
		callback: ProgressCallback | None,
		stage_name: str,
		current: int,
	) -> None:
		if callback is None:
			return
		callback(stage_name, current, self._TOTAL_STAGES)


__all__ = [
	"IngestionPipeline",
	"IngestionPipelineError",
	"IngestionResult",
]
