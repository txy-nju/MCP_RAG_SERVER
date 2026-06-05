"""Ingestion pipeline orchestration (C14 MVP)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

from modular_rag.core.settings import Settings
from modular_rag.core.trace.trace_context import TraceContext
from modular_rag.core.types import Chunk, ChunkRecord, Document
from modular_rag.ingestion.chunking.document_chunker import DocumentChunker
from modular_rag.ingestion.embedding.batch_processor import BatchProcessor
from modular_rag.ingestion.storage.bm25_indexer import BM25Indexer
from modular_rag.ingestion.storage.image_storage import ImageStorage
from modular_rag.ingestion.storage.vector_upserter import VectorUpserter
from modular_rag.ingestion.transform.base_transform import BaseTransform
from modular_rag.ingestion.transform.chunk_refiner import ChunkRefiner
from modular_rag.ingestion.transform.image_captioner import ImageCaptioner
from modular_rag.ingestion.transform.metadata_enricher import MetadataEnricher
from modular_rag.libs.loader.base_loader import BaseLoader
from modular_rag.libs.loader.file_integrity import FileIntegrityChecker, SQLiteIntegrityChecker
from modular_rag.libs.loader.pdf_loader import PdfLoader
from modular_rag.observability.logger import get_logger


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
		# 若磁盘上已有索引，自动加载以支持 rebuild=False 增量追加（否则会覆盖历史数据）
		if bm25_indexer is None and self._bm25_indexer.index_path.exists():
			self._bm25_indexer.load()
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
				trace.record_stage(
					stage,
					{
						"method": "sha256",
						"provider": self._integrity_checker.__class__.__name__.lower(),
						"details": {"source_path": source_path, "skip": True},
					},
				)
				self._emit_progress(on_progress, stage, 1)
				self._logger.info("Skipping unchanged file source=%s", source_path)
				return None

			trace.record_stage(
				stage,
				{
					"method": "sha256",
					"provider": self._integrity_checker.__class__.__name__.lower(),
					"details": {"source_path": source_path, "skip": False},
				},
			)
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
			trace.record_stage(
				stage,
				{
					"method": "load_document",
					"provider": self._loader.__class__.__name__.lower(),
					"details": {
						"document_id": document.id,
						"text_chars": len(document.text),
					},
				},
			)
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
				# Scope chunk ID with collection name to prevent ID collisions
				# when the same document is ingested into multiple collections
				# (e.g. single-video "video_{vid}" vs KB "kb_{kbid}").
				# Both Chroma and BM25 receive this scoped ID, keeping hybrid
				# search consistent.
				chunk.id = f"{collection}_{chunk.id}"
			trace.record_stage(
				stage,
				{
					"method": "split_document",
					"provider": str(self._settings.splitter.provider),
					"details": {
						"chunk_count": len(chunks),
						"collection": collection,
					},
				},
			)
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
			trace.record_stage(
				stage,
				{
					"method": "transform_chain",
					"provider": "->".join(t.__class__.__name__.lower() for t in self._transforms),
					"details": {
						"chunk_count": len(transformed),
						"transform_steps": len(self._transforms),
					},
				},
			)
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
		stage = "embed"
		try:
			records = self._batch_processor.process(chunks, batch_size=batch_size, trace=trace)
			trace.record_stage(
				stage,
				{
					"method": "dense_sparse_batch",
					"provider": f"{self._settings.embedding.provider}+bm25",
					"details": {
						"record_count": len(records),
						"batch_size": batch_size,
					},
				},
			)
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
		stage = "upsert"
		try:
			upserted_ids = self._vector_upserter.upsert(records, trace=trace)
			self._bm25_indexer.build(records, rebuild=False)
			saved_images = self._persist_images(document=document, collection=collection)
			trace.record_stage(
				stage,
				{
					"method": "upsert_indexes",
					"provider": f"{self._settings.vector_store.provider}+bm25+image_storage",
					"details": {
						"upserted_count": len(upserted_ids),
						"bm25_index_path": str(self._bm25_indexer.index_path),
						"saved_image_count": saved_images,
					},
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

	def run_docs(self, docs: list[Document], collection: str) -> None:
		"""跳过 loader/integrity 阶段，直接从已构造好的 Document 列表执行流水线。

		适用于调用方自行构造 Document 对象的场景（如带时间戳的转录 segments）。
		collection 值写入每个 doc 的 metadata["collection"]（与 run() 行为一致）。

		流程：split → transform → encode（embedding）→ upsert（Chroma + BM25）
		不走 _run_store_stage（该方法需要 Document 用于图片持久化）；
		转录文本不含图片，直接调用底层存储组件，行为完全等价。
		"""
		trace_ctx = TraceContext(trace_type="ingestion")
		all_chunks: list = []

		for doc in docs:
			doc.metadata.setdefault("collection", collection)
			chunks = self._run_split_stage(doc, collection, trace_ctx, None)
			all_chunks.extend(chunks)

		if not all_chunks:
			self._logger.info("run_docs: no chunks produced for collection=%s", collection)
			return

		transformed = self._run_transform_stage(all_chunks, trace_ctx, None)
		records = self._run_encode_stage(transformed, 32, trace_ctx, None)
		self._vector_upserter.upsert(records, trace=trace_ctx)
		self._bm25_indexer.build(records, rebuild=False)
		self._logger.info(
			"run_docs: collection=%s docs=%d chunks=%d records=%d",
			collection, len(docs), len(transformed), len(records),
		)

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
