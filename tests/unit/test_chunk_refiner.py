"""Unit tests for chunk refinement transform stage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from modular_rag.core.settings import (
    ChunkRefinerSettings,
    EmbeddingSettings,
    EvaluationSettings,
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
from modular_rag.ingestion.transform.chunk_refiner import ChunkRefiner
from modular_rag.libs.llm.base_llm import BaseLLM


class FakeLLM(BaseLLM):
    def __init__(self, *, provider: str, model: str, response: str = "refined output") -> None:
        super().__init__(provider=provider, model=model)
        self.response = response
        self.calls: list[list[dict[str, Any]]] = []

    def chat(self, messages: list[dict[str, Any]]) -> str:
        self.calls.append(messages)
        return self.response


class FailingLLM(BaseLLM):
    def chat(self, messages: list[dict[str, Any]]) -> str:
        raise RuntimeError("provider unavailable")


def make_settings(*, use_llm: bool = False) -> Settings:
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
            chunk_refiner=ChunkRefinerSettings(
                use_llm=use_llm,
                prompt_path="config/prompts/chunk_refinement.txt",
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


def load_noisy_fixture() -> dict[str, str]:
    fixture_path = Path("tests/fixtures/noisy_chunks.json")
    return json.loads(fixture_path.read_text(encoding="utf-8"))


@pytest.mark.unit
def test_rule_mode_refines_whitespace_and_footer_noise() -> None:
    noisy = load_noisy_fixture()["typical_noise_scenario"]
    refiner = ChunkRefiner(make_settings(use_llm=False))

    refined = refiner.transform([make_chunk(noisy)])[0]

    assert "Page 1 / 10" not in refined.text
    assert "<!-- footer -->" not in refined.text
    assert "---" not in refined.text
    assert refined.metadata["refined_by"] == "rule"


@pytest.mark.unit
def test_rule_mode_keeps_code_block_content() -> None:
    noisy = load_noisy_fixture()["code_blocks"]
    refiner = ChunkRefiner(make_settings(use_llm=False))

    refined = refiner.transform([make_chunk(noisy)])[0]

    assert "```python" in refined.text
    assert "return a + b" in refined.text
    assert "Page 3" not in refined.text


@pytest.mark.unit
def test_rule_mode_removes_html_comments_and_horizontal_rules() -> None:
    noisy = load_noisy_fixture()["format_markers"]
    refiner = ChunkRefiner(make_settings(use_llm=False))

    refined = refiner.transform([make_chunk(noisy)])[0]

    assert "<!-- hidden -->" not in refined.text
    assert "***" not in refined.text


@pytest.mark.unit
def test_rule_mode_preserves_clean_text_semantics() -> None:
    clean = load_noisy_fixture()["clean_text"]
    refiner = ChunkRefiner(make_settings(use_llm=False))

    refined = refiner.transform([make_chunk(clean)])[0]

    assert "already clean text" in refined.text
    assert refined.text.endswith("stable.")


@pytest.mark.unit
def test_llm_mode_uses_llm_output_and_tags_metadata() -> None:
    llm = FakeLLM(provider="fake", model="demo", response="llm rewritten")
    refiner = ChunkRefiner(make_settings(use_llm=True), llm=llm)

    refined = refiner.transform([make_chunk("raw text")])[0]

    assert refined.text == "llm rewritten"
    assert refined.metadata["refined_by"] == "llm"
    assert len(llm.calls) == 1


@pytest.mark.unit
def test_llm_mode_includes_prompt_as_system_message() -> None:
    llm = FakeLLM(provider="fake", model="demo", response="ok")
    refiner = ChunkRefiner(make_settings(use_llm=True), llm=llm)

    refiner.transform([make_chunk("raw text")])

    assert "{text}" in llm.calls[0][0]["content"]
    assert llm.calls[0][1]["content"] == "raw text"


@pytest.mark.unit
def test_llm_failure_falls_back_to_rule_and_marks_reason() -> None:
    llm = FailingLLM(provider="fake", model="demo")
    refiner = ChunkRefiner(make_settings(use_llm=True), llm=llm)

    refined = refiner.transform([make_chunk("Page 1\n\nbody")])[0]

    assert refined.metadata["refined_by"] == "rule"
    assert "fallback_reason" in refined.metadata
    assert "provider unavailable" in refined.metadata["fallback_reason"]
    assert "Page 1" not in refined.text


@pytest.mark.unit
def test_llm_disabled_never_calls_llm() -> None:
    llm = FakeLLM(provider="fake", model="demo", response="should not be used")
    refiner = ChunkRefiner(make_settings(use_llm=False), llm=llm)

    refined = refiner.transform([make_chunk("normal")])[0]

    assert refined.metadata["refined_by"] == "rule"
    assert refined.text == "normal"
    assert llm.calls == []


@pytest.mark.unit
def test_transform_handles_chunk_exception_without_breaking_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    refiner = ChunkRefiner(make_settings(use_llm=False))

    def raise_once(text: str) -> str:
        if text == "bad":
            raise ValueError("boom")
        return text

    monkeypatch.setattr(refiner, "_rule_based_refine", raise_once)

    chunks = [make_chunk("ok", "chunk-a"), make_chunk("bad", "chunk-b"), make_chunk("fine", "chunk-c")]
    out = refiner.transform(chunks)

    assert [chunk.id for chunk in out] == ["chunk-a", "chunk-b", "chunk-c"]
    assert out[1].text == "bad"
    assert out[1].metadata["refined_by"] == "none"
    assert "refine_error" in out[1].metadata


@pytest.mark.unit
def test_trace_records_llm_stage_on_success() -> None:
    llm = FakeLLM(provider="fake", model="demo", response="ok")
    refiner = ChunkRefiner(make_settings(use_llm=True), llm=llm)
    trace = TraceContext(trace_type="ingestion")

    refiner.transform([make_chunk("alpha")], trace=trace)

    assert any(stage["stage"] == "chunk_refiner.llm" for stage in trace.stages)


@pytest.mark.unit
def test_trace_records_fallback_stage_on_llm_error() -> None:
    llm = FailingLLM(provider="fake", model="demo")
    refiner = ChunkRefiner(make_settings(use_llm=True), llm=llm)
    trace = TraceContext(trace_type="ingestion")

    refiner.transform([make_chunk("alpha")], trace=trace)

    assert any(stage["stage"] == "chunk_refiner.llm_fallback" for stage in trace.stages)


@pytest.mark.unit
def test_load_prompt_appends_placeholder_when_missing(tmp_path: Path) -> None:
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("rewrite this", encoding="utf-8")
    settings = make_settings(use_llm=False)
    refiner = ChunkRefiner(settings, prompt_path=str(prompt_file))

    assert "{text}" in refiner._prompt


@pytest.mark.unit
def test_load_prompt_uses_default_when_file_missing() -> None:
    settings = make_settings(use_llm=False)
    refiner = ChunkRefiner(settings, prompt_path="missing-prompt-file.txt")

    assert "{text}" in refiner._prompt
    assert "retrieval quality" in refiner._prompt


@pytest.mark.unit
def test_transform_preserves_chunk_identity_fields() -> None:
    chunk = make_chunk("hello", "chunk-x")
    refiner = ChunkRefiner(make_settings(use_llm=False))

    refined = refiner.transform([chunk])[0]

    assert refined.id == "chunk-x"
    assert refined.source_ref == {"doc_id": "doc-1"}


@pytest.mark.unit
def test_transform_updates_offsets_after_refinement() -> None:
    chunk = make_chunk("a    b")
    refiner = ChunkRefiner(make_settings(use_llm=False))

    refined = refiner.transform([chunk])[0]

    assert refined.start_offset == 0
    assert refined.end_offset == len(refined.text)


@pytest.mark.unit
def test_mixed_noise_cleanup_removes_page_and_comments() -> None:
    noisy = load_noisy_fixture()["mixed_noise"]
    refiner = ChunkRefiner(make_settings(use_llm=False))

    refined = refiner.transform([make_chunk(noisy)])[0]

    assert "Page 7 / 20" not in refined.text
    assert "<!-- comment -->" not in refined.text
    assert "pipeline works in batches" in refined.text


@pytest.mark.unit
def test_empty_text_remains_empty() -> None:
    refiner = ChunkRefiner(make_settings(use_llm=False))

    refined = refiner.transform([make_chunk("")])[0]

    assert refined.text == ""


@pytest.mark.unit
def test_multiple_chunks_processed_in_order() -> None:
    refiner = ChunkRefiner(make_settings(use_llm=False))
    chunks = [make_chunk("z", "c1"), make_chunk("y", "c2"), make_chunk("x", "c3")]

    out = refiner.transform(chunks)

    assert [chunk.id for chunk in out] == ["c1", "c2", "c3"]


@pytest.mark.unit
def test_page_header_only_lines_removed() -> None:
    refiner = ChunkRefiner(make_settings(use_llm=False))
    chunk = make_chunk("Page 12\n\nMeaningful line")

    refined = refiner.transform([chunk])[0]

    assert "Page 12" not in refined.text
    assert "Meaningful line" in refined.text


@pytest.mark.unit
def test_markdown_heading_preserved() -> None:
    refiner = ChunkRefiner(make_settings(use_llm=False))
    chunk = make_chunk("## Heading\n\nbody")

    refined = refiner.transform([chunk])[0]

    assert refined.text.startswith("## Heading")


@pytest.mark.unit
def test_llm_empty_response_falls_back_to_rule() -> None:
    llm = FakeLLM(provider="fake", model="demo", response="   ")
    refiner = ChunkRefiner(make_settings(use_llm=True), llm=llm)

    refined = refiner.transform([make_chunk("body")])[0]

    assert refined.text == "body"
    assert refined.metadata["refined_by"] == "rule"


@pytest.mark.unit
def test_rule_refine_collapse_multiple_blank_lines() -> None:
    refiner = ChunkRefiner(make_settings(use_llm=False))
    chunk = make_chunk("a\n\n\n\n b")

    refined = refiner.transform([chunk])[0]

    assert "\n\n\n" not in refined.text


@pytest.mark.unit
def test_rule_refine_collapses_excessive_spaces() -> None:
    refiner = ChunkRefiner(make_settings(use_llm=False))
    chunk = make_chunk("a      b")

    refined = refiner.transform([chunk])[0]

    assert refined.text == "a b"


@pytest.mark.unit
def test_rule_mode_handles_ocr_like_text_without_crash() -> None:
    noisy = load_noisy_fixture()["ocr_errors"]
    refiner = ChunkRefiner(make_settings(use_llm=False))

    refined = refiner.transform([make_chunk(noisy)])[0]

    assert "OCR" in refined.text


@pytest.mark.unit
def test_trace_records_chunk_error_when_rule_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    refiner = ChunkRefiner(make_settings(use_llm=False))
    trace = TraceContext(trace_type="ingestion")

    def always_fail(text: str) -> str:
        raise RuntimeError("rule failed")

    monkeypatch.setattr(refiner, "_rule_based_refine", always_fail)
    refiner.transform([make_chunk("a")], trace=trace)

    assert any(stage["stage"] == "chunk_refiner.chunk_error" for stage in trace.stages)


@pytest.mark.unit
def test_chunk_metadata_source_path_preserved() -> None:
    refiner = ChunkRefiner(make_settings(use_llm=False))

    refined = refiner.transform([make_chunk("hello")])[0]

    assert refined.metadata["source_path"] == "docs/sample.pdf"
