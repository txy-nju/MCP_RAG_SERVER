"""Unit tests for ImageCaptioner fallback and enabled behaviors."""

from __future__ import annotations

from typing import Any

import pytest

from modular_rag.core.settings import (
    EmbeddingSettings,
    EvaluationSettings,
    ImageCaptionerSettings,
    IngestionSettings,
    LLMSettings,
    ObservabilitySettings,
    RerankSettings,
    RetrievalSettings,
    Settings,
    SplitterSettings,
    VectorStoreSettings,
)
from modular_rag.core.trace.trace_context import TraceContext
from modular_rag.core.types import Chunk
from modular_rag.ingestion.transform.image_captioner import ImageCaptioner
from modular_rag.libs.llm.base_vision_llm import BaseVisionLLM, ChatResponse


class FakeVisionLLM(BaseVisionLLM):
    def __init__(self, *, provider: str, model: str, caption: str = "diagram of ingestion pipeline") -> None:
        super().__init__(provider=provider, model=model)
        self.caption = caption
        self.calls: list[tuple[str, str]] = []

    def chat_with_image(self, text: str, image_path: str | bytes, trace: TraceContext | None = None) -> ChatResponse:
        assert isinstance(image_path, str)
        self.calls.append((text, image_path))
        return ChatResponse(
            content=self.caption,
            provider=self.provider,
            model=self.model,
            metadata={"trace_provided": trace is not None},
        )


class FailingVisionLLM(BaseVisionLLM):
    def chat_with_image(self, text: str, image_path: str | bytes, trace: TraceContext | None = None) -> ChatResponse:
        raise RuntimeError("vision endpoint unavailable")


def make_settings(*, use_vision_llm: bool) -> Settings:
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o-mini"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        splitter=SplitterSettings(provider="recursive", chunk_size=1000, chunk_overlap=200),
        vector_store=VectorStoreSettings(provider="chroma", collection="default"),
        retrieval=RetrievalSettings(top_k=5),
        rerank=RerankSettings(provider="none"),
        evaluation=EvaluationSettings(backend="custom"),
        observability=ObservabilitySettings(log_level="INFO", trace_file="logs/traces.jsonl"),
        ingestion=IngestionSettings(
            image_captioner=ImageCaptionerSettings(
                use_vision_llm=use_vision_llm,
                prompt_path="config/prompts/image_captioning.txt",
            )
        ),
    )


def make_chunk(*, include_images: bool = True) -> Chunk:
    metadata: dict[str, Any] = {
        "source_path": "docs/sample.pdf",
        "chunk_index": 0,
    }
    if include_images:
        metadata["image_refs"] = ["img-001"]
        metadata["images"] = [
            {
                "id": "img-001",
                "path": "tests/fixtures/sample_documents/diagram.png",
                "text_offset": 10,
                "text_length": 8,
                "page": 1,
                "position": {},
            }
        ]

    return Chunk(
        id="chunk-1",
        text="[IMAGE: img-001] system overview",
        metadata=metadata,
        start_offset=0,
        end_offset=30,
        source_ref={"doc_id": "doc-1"},
    )


@pytest.mark.unit
def test_enabled_mode_generates_caption_and_writes_metadata() -> None:
    vision = FakeVisionLLM(provider="azure", model="gpt-4o")
    captioner = ImageCaptioner(make_settings(use_vision_llm=True), vision_llm=vision)

    out = captioner.transform([make_chunk()])[0]

    assert "image_captions" in out.metadata
    assert out.metadata["image_captions"]["img-001"] == "diagram of ingestion pipeline"
    assert out.metadata.get("has_unprocessed_images") is None
    assert len(vision.calls) == 1


@pytest.mark.unit
def test_disabled_mode_marks_unprocessed_images_without_captioning() -> None:
    vision = FakeVisionLLM(provider="azure", model="gpt-4o")
    captioner = ImageCaptioner(make_settings(use_vision_llm=False), vision_llm=vision)

    out = captioner.transform([make_chunk()])[0]

    assert out.metadata["has_unprocessed_images"] is True
    assert "image_captions" not in out.metadata
    assert vision.calls == []


@pytest.mark.unit
def test_failure_mode_falls_back_without_blocking_batch() -> None:
    captioner = ImageCaptioner(
        make_settings(use_vision_llm=True),
        vision_llm=FailingVisionLLM(provider="azure", model="gpt-4o"),
    )
    trace = TraceContext(trace_type="ingestion")

    out = captioner.transform([make_chunk()], trace=trace)[0]

    assert out.metadata["has_unprocessed_images"] is True
    assert "image_captions" not in out.metadata
    assert "caption_fallback_reason" in out.metadata
    assert any(stage["stage"] == "image_captioner.fallback" for stage in trace.stages)


@pytest.mark.unit
def test_chunk_without_image_refs_is_kept_unchanged() -> None:
    vision = FakeVisionLLM(provider="azure", model="gpt-4o")
    captioner = ImageCaptioner(make_settings(use_vision_llm=True), vision_llm=vision)

    out = captioner.transform([make_chunk(include_images=False)])[0]

    assert "image_captions" not in out.metadata
    assert "has_unprocessed_images" not in out.metadata
    assert vision.calls == []
