"""Unit tests for ingestion pipeline progress callback behavior (F5)."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

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
from core.types import Chunk, ChunkRecord, Document
from ingestion.chunking.document_chunker import DocumentChunker
from ingestion.embedding.batch_processor import BatchProcessor
from ingestion.pipeline import IngestionPipeline
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from ingestion.storage.vector_upserter import VectorUpserter
from ingestion.transform.base_transform import BaseTransform
from libs.loader.base_loader import BaseLoader
from libs.loader.file_integrity import FileIntegrityChecker


class FakeIntegrityChecker:
    def __init__(self) -> None:
        self.success_calls: list[tuple[str, str, dict]] = []

    def compute_sha256(self, path: str) -> str:
        return f"hash::{Path(path).name}"

    def should_skip(self, file_hash: str) -> bool:
        del file_hash
        return False

    def mark_success(self, file_hash: str, file_path: str, **kwargs) -> None:
        self.success_calls.append((file_hash, file_path, kwargs))

    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        del file_hash, error_msg


class FakeLoader:
    def load(self, path: str) -> Document:
        return Document(
            id="doc-progress-1",
            text="pipeline progress callback unit test",
            metadata={"source_path": path, "images": []},
        )


class FakeChunker:
    def split_document(self, document: Document) -> list[Chunk]:
        return [
            Chunk(
                id="chunk-progress-1",
                text=document.text,
                metadata={"source_path": document.metadata["source_path"], "chunk_index": 0},
                start_offset=0,
                end_offset=len(document.text),
                source_ref={"doc_id": document.id},
            )
        ]


class FakeTransform:
    def transform(self, chunks: list[Chunk], trace: TraceContext | None = None) -> list[Chunk]:
        del trace
        return chunks


class FakeBatchProcessor:
    def process(
        self,
        chunks: list[Chunk],
        *,
        batch_size: int,
        trace: TraceContext | None = None,
    ) -> list[ChunkRecord]:
        del batch_size, trace
        return [
            ChunkRecord.from_chunk(
                chunk,
                dense_vector=[0.1, 0.2, 0.3],
                sparse_vector={"pipeline": 1.0, "progress": 1.0},
            )
            for chunk in chunks
        ]


class FakeVectorUpserter:
    def upsert(self, records: list[ChunkRecord], trace: TraceContext | None = None) -> list[str]:
        del trace
        return [f"upserted-{idx}" for idx, _ in enumerate(records, start=1)]


def _make_settings() -> Settings:
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o-mini"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        splitter=SplitterSettings(provider="recursive", chunk_size=1000, chunk_overlap=200),
        vector_store=VectorStoreSettings(provider="chroma", collection="test", persist_path="data/db/chroma"),
        retrieval=RetrievalSettings(top_k=5),
        rerank=RerankSettings(provider="none"),
        evaluation=EvaluationSettings(backend="custom"),
        observability=ObservabilitySettings(log_level="INFO", trace_file="logs/traces.jsonl"),
    )


def _build_pipeline(tmp_path: Path) -> IngestionPipeline:
    return IngestionPipeline(
        _make_settings(),
        integrity_checker=cast(FileIntegrityChecker, FakeIntegrityChecker()),
        loader=cast(BaseLoader, FakeLoader()),
        chunker=cast(DocumentChunker, FakeChunker()),
        transforms=cast(list[BaseTransform], [FakeTransform()]),
        batch_processor=cast(BatchProcessor, FakeBatchProcessor()),
        vector_upserter=cast(VectorUpserter, FakeVectorUpserter()),
        bm25_indexer=BM25Indexer(index_dir=tmp_path / "bm25"),
        image_storage=ImageStorage(image_root=tmp_path / "images", db_path=tmp_path / "image_index.db"),
    )


@pytest.mark.unit
def test_pipeline_invokes_progress_callback_for_all_stages(tmp_path: Path) -> None:
    source_file = tmp_path / "progress.pdf"
    source_file.write_bytes(b"%PDF-progress")

    pipeline = _build_pipeline(tmp_path)
    progress_events: list[tuple[str, int, int]] = []

    def on_progress(stage_name: str, current: int, total: int) -> None:
        progress_events.append((stage_name, current, total))

    result = pipeline.run(str(source_file), collection="demo", on_progress=on_progress)

    assert result.skipped is False
    assert [event[0] for event in progress_events] == [
        "integrity",
        "load",
        "split",
        "transform",
        "embed",
        "upsert",
    ]
    assert [event[1] for event in progress_events] == [1, 2, 3, 4, 5, 6]
    assert all(total == 6 for _, _, total in progress_events)


@pytest.mark.unit
def test_pipeline_run_without_progress_callback_keeps_behavior(tmp_path: Path) -> None:
    source_file = tmp_path / "no-progress.pdf"
    source_file.write_bytes(b"%PDF-no-progress")

    pipeline = _build_pipeline(tmp_path)

    result = pipeline.run(str(source_file), collection="demo", on_progress=None)

    assert result.skipped is False
    assert result.chunk_count == 1
    assert result.record_count == 1
    assert result.upserted_ids == ["upserted-1"]
