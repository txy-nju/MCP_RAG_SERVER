"""Contract tests for metadata enrichment transform stage."""

from __future__ import annotations

from typing import Any

import pytest

from core.settings import (
    EmbeddingSettings,
    EvaluationSettings,
    IngestionSettings,
    LLMSettings,
    MetadataEnricherSettings,
    ObservabilitySettings,
    RerankSettings,
    RetrievalSettings,
    Settings,
    SplitterSettings,
    VectorStoreSettings,
)
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.metadata_enricher import MetadataEnricher
from libs.llm.base_llm import BaseLLM


class FakeLLM(BaseLLM):
    def __init__(self, *, provider: str, model: str, response: str) -> None:
        super().__init__(provider=provider, model=model)
        self.response = response
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]]) -> str:
        self.calls.append(messages)
        return self.response


class FailingLLM(BaseLLM):
    def chat(self, messages: list[dict[str, Any]]) -> str:
        raise RuntimeError("provider unavailable")


def make_settings(*, use_llm: bool = False, max_tags: int = 5) -> Settings:
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
            metadata_enricher=MetadataEnricherSettings(
                use_llm=use_llm,
                prompt_path="config/prompts/metadata_enrichment.txt",
                max_tags=max_tags,
            )
        ),
    )


def make_chunk(text: str, chunk_id: str = "chunk-1") -> Chunk:
    return Chunk(
        id=chunk_id,
        text=text,
        metadata={"source_path": "docs/sample.pdf", "chunk_index": 0},
        start_offset=0,
        end_offset=len(text),
        source_ref={"doc_id": "doc-1"},
    )


@pytest.mark.unit
def test_rule_mode_produces_non_empty_title_summary_tags() -> None:
    enricher = MetadataEnricher(make_settings(use_llm=False))

    out = enricher.transform([make_chunk("## Pipeline Overview\n\nThis section explains chunk processing flow.")])[0]

    assert out.metadata["enriched_by"] == "rule"
    assert out.metadata["title"]
    assert out.metadata["summary"]
    assert isinstance(out.metadata["tags"], list)
    assert len(out.metadata["tags"]) > 0


@pytest.mark.unit
def test_rule_mode_respects_max_tags() -> None:
    enricher = MetadataEnricher(make_settings(use_llm=False, max_tags=3))

    out = enricher.transform([make_chunk("alpha beta gamma delta epsilon alpha beta")])[0]

    assert len(out.metadata["tags"]) <= 3


@pytest.mark.unit
def test_llm_mode_overrides_rule_result_when_json_valid() -> None:
    llm = FakeLLM(
        provider="fake",
        model="demo",
        response='{"title":"LLM Title","summary":"LLM summary","tags":["rag","ingestion","metadata"]}',
    )
    enricher = MetadataEnricher(make_settings(use_llm=True), llm=llm)

    out = enricher.transform([make_chunk("Raw technical chunk")])[0]

    assert out.metadata["enriched_by"] == "llm"
    assert out.metadata["title"] == "LLM Title"
    assert out.metadata["summary"] == "LLM summary"
    assert out.metadata["tags"] == ["rag", "ingestion", "metadata"]
    assert len(llm.calls) == 1


@pytest.mark.unit
def test_llm_mode_accepts_fenced_json() -> None:
    llm = FakeLLM(
        provider="fake",
        model="demo",
        response='```json\n{"title":"T","summary":"S","tags":["a","b","c"]}\n```',
    )
    enricher = MetadataEnricher(make_settings(use_llm=True), llm=llm)

    out = enricher.transform([make_chunk("body")])[0]

    assert out.metadata["enriched_by"] == "llm"
    assert out.metadata["title"] == "T"


@pytest.mark.unit
def test_llm_invalid_payload_falls_back_to_rule_with_reason() -> None:
    llm = FakeLLM(provider="fake", model="demo", response="not-json")
    enricher = MetadataEnricher(make_settings(use_llm=True), llm=llm)

    out = enricher.transform([make_chunk("Pipeline data flow")])[0]

    assert out.metadata["enriched_by"] == "rule"
    assert "fallback_reason" in out.metadata


@pytest.mark.unit
def test_llm_exception_falls_back_and_records_trace() -> None:
    enricher = MetadataEnricher(make_settings(use_llm=True), llm=FailingLLM(provider="fake", model="demo"))
    trace = TraceContext(trace_type="ingestion")

    out = enricher.transform([make_chunk("Pipeline data flow")], trace=trace)[0]

    assert out.metadata["enriched_by"] == "rule"
    assert "fallback_reason" in out.metadata
    assert any(stage["stage"] == "metadata_enricher.llm_fallback" for stage in trace.stages)


@pytest.mark.unit
def test_transform_handles_single_chunk_failure_without_breaking_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    enricher = MetadataEnricher(make_settings(use_llm=False))

    def raise_once(text: str) -> dict[str, Any]:
        if text == "bad":
            raise ValueError("boom")
        return {"title": "ok", "summary": "ok", "tags": ["ok"]}

    monkeypatch.setattr(enricher, "_rule_enrich", raise_once)

    chunks = [make_chunk("good", "a"), make_chunk("bad", "b"), make_chunk("fine", "c")]
    out = enricher.transform(chunks)

    assert [chunk.id for chunk in out] == ["a", "b", "c"]
    assert out[1].metadata["enriched_by"] == "rule"
    assert "enrich_error" in out[1].metadata


@pytest.mark.unit
def test_transform_preserves_chunk_identity_fields() -> None:
    enricher = MetadataEnricher(make_settings(use_llm=False))
    chunk = make_chunk("hello", "chunk-x")

    out = enricher.transform([chunk])[0]

    assert out.id == "chunk-x"
    assert out.source_ref == {"doc_id": "doc-1"}
    assert out.start_offset == chunk.start_offset
    assert out.end_offset == chunk.end_offset
