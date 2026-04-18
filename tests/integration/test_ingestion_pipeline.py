"""Integration tests for ingestion pipeline orchestration (C14)."""

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
from ingestion.pipeline import IngestionPipeline, IngestionPipelineError
from ingestion.storage.bm25_indexer import BM25Indexer
from ingestion.storage.image_storage import ImageStorage
from ingestion.storage.vector_upserter import VectorUpserter
from ingestion.transform.base_transform import BaseTransform
from libs.loader.base_loader import BaseLoader
from libs.loader.file_integrity import FileIntegrityChecker


class FakeIntegrityChecker:
    def __init__(self, *, should_skip: bool = False) -> None:
        self._should_skip = should_skip
        self.success_calls: list[tuple[str, str, dict]] = []
        self.failed_calls: list[tuple[str, str]] = []

    def compute_sha256(self, path: str) -> str:
        return f"hash::{Path(path).name}"

    def should_skip(self, file_hash: str) -> bool:
        return self._should_skip

    def mark_success(self, file_hash: str, file_path: str, **kwargs) -> None:
        self.success_calls.append((file_hash, file_path, kwargs))

    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        self.failed_calls.append((file_hash, error_msg))


class FakeLoader:
    def __init__(self, image_path: str) -> None:
        self.image_path = image_path
        self.calls: list[str] = []

    def load(self, path: str) -> Document:
        self.calls.append(path)
        return Document(
            id="doc-001",
            text="pipeline orchestration text",
            metadata={
                "source_path": path,
                "images": [
                    {
                        "id": "img-1",
                        "path": self.image_path,
                        "page": 1,
                        "position": {},
                        "text_offset": 0,
                        "text_length": 8,
                    }
                ],
            },
        )


class FakeChunker:
    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail

    def split_document(self, document: Document) -> list[Chunk]:
        if self._fail:
            raise ValueError("split exploded")
        return [
            Chunk(
                id="chunk-001",
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
                sparse_vector={"pipeline": 1.0, "text": 1.0},
            )
            for chunk in chunks
        ]


class FakeVectorUpserter:
    def __init__(self) -> None:
        self.last_records: list[ChunkRecord] = []

    def upsert(self, records: list[ChunkRecord], trace: TraceContext | None = None) -> list[str]:
        del trace
        self.last_records = list(records)
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


@pytest.mark.integration
def test_ingestion_pipeline_runs_full_flow_and_persists_outputs(tmp_path: Path) -> None:
    source_file = tmp_path / "source.pdf"
    source_file.write_bytes(b"%PDF-test")
    source_image = tmp_path / "source-image.png"
    source_image.write_bytes(b"png-bytes")

    integrity = FakeIntegrityChecker(should_skip=False)
    loader = FakeLoader(str(source_image))
    chunker = FakeChunker()
    batch = FakeBatchProcessor()
    upserter = FakeVectorUpserter()
    bm25 = BM25Indexer(index_dir=tmp_path / "bm25")
    image_storage = ImageStorage(image_root=tmp_path / "images", db_path=tmp_path / "image_index.db")

    pipeline = IngestionPipeline(
        _make_settings(),
        integrity_checker=cast(FileIntegrityChecker, integrity),
        loader=cast(BaseLoader, loader),
        chunker=cast(DocumentChunker, chunker),
        transforms=cast(list[BaseTransform], [FakeTransform()]),
        batch_processor=cast(BatchProcessor, batch),
        vector_upserter=cast(VectorUpserter, upserter),
        bm25_indexer=bm25,
        image_storage=image_storage,
    )

    progress_events: list[tuple[str, int, int]] = []
    def on_progress(stage_name: str, current: int, total: int) -> None:
        progress_events.append((stage_name, current, total))

    result = pipeline.run(
        str(source_file),
        collection="demo",
        on_progress=on_progress,
    )

    assert result.skipped is False
    assert result.document_id == "doc-001"
    assert result.chunk_count == 1
    assert result.record_count == 1
    assert result.upserted_ids == ["upserted-1"]
    assert result.saved_image_count == 1

    assert len(integrity.success_calls) == 1
    assert integrity.failed_calls == []
    assert bm25.index_path.exists()
    assert image_storage.get_image_path("img-1") is not None

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

    assert result.trace is not None
    trace = result.trace
    assert trace.trace_type == "ingestion"

    stage_names = [item["stage"] for item in trace.stages]
    for required in ["load", "split", "transform", "embed", "upsert"]:
        assert required in stage_names

    for stage in trace.stages:
        assert "elapsed_ms" in stage
        assert isinstance(stage["elapsed_ms"], float)
        assert "method" in stage["data"]
        assert "details" in stage["data"]

    trace_payload = trace.to_dict()
    assert trace_payload["trace_type"] == "ingestion"


@pytest.mark.integration
def test_ingestion_pipeline_skips_unchanged_file_when_not_forced(tmp_path: Path) -> None:
    source_file = tmp_path / "already-processed.pdf"
    source_file.write_bytes(b"%PDF-test")

    integrity = FakeIntegrityChecker(should_skip=True)
    loader = FakeLoader(str(tmp_path / "unused.png"))

    pipeline = IngestionPipeline(
        _make_settings(),
        integrity_checker=cast(FileIntegrityChecker, integrity),
        loader=cast(BaseLoader, loader),
        chunker=cast(DocumentChunker, FakeChunker()),
        transforms=cast(list[BaseTransform], [FakeTransform()]),
        batch_processor=cast(BatchProcessor, FakeBatchProcessor()),
        vector_upserter=cast(VectorUpserter, FakeVectorUpserter()),
        bm25_indexer=BM25Indexer(index_dir=tmp_path / "bm25"),
        image_storage=ImageStorage(image_root=tmp_path / "images", db_path=tmp_path / "image_index.db"),
    )

    progress_events: list[tuple[str, int, int]] = []
    def on_progress(stage_name: str, current: int, total: int) -> None:
        progress_events.append((stage_name, current, total))

    result = pipeline.run(
        str(source_file),
        on_progress=on_progress,
    )

    assert result.skipped is True
    assert result.chunk_count == 0
    assert loader.calls == []
    assert integrity.success_calls == []
    assert integrity.failed_calls == []
    assert progress_events == [("integrity", 1, 6)]


@pytest.mark.integration
def test_ingestion_pipeline_raises_clear_stage_error_and_marks_failed(tmp_path: Path) -> None:
    source_file = tmp_path / "broken.pdf"
    source_file.write_bytes(b"%PDF-test")
    source_image = tmp_path / "source-image.png"
    source_image.write_bytes(b"png-bytes")

    integrity = FakeIntegrityChecker(should_skip=False)
    pipeline = IngestionPipeline(
        _make_settings(),
        integrity_checker=cast(FileIntegrityChecker, integrity),
        loader=cast(BaseLoader, FakeLoader(str(source_image))),
        chunker=cast(DocumentChunker, FakeChunker(fail=True)),
        transforms=cast(list[BaseTransform], [FakeTransform()]),
        batch_processor=cast(BatchProcessor, FakeBatchProcessor()),
        vector_upserter=cast(VectorUpserter, FakeVectorUpserter()),
        bm25_indexer=BM25Indexer(index_dir=tmp_path / "bm25"),
        image_storage=ImageStorage(image_root=tmp_path / "images", db_path=tmp_path / "image_index.db"),
    )

    with pytest.raises(IngestionPipelineError, match="stage 'split'"):
        pipeline.run(str(source_file))

    assert len(integrity.failed_calls) == 1
    assert "split" in integrity.failed_calls[0][1]
