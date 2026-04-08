"""Integration tests for ChunkRefiner with real provider wiring.

These tests are opt-in and will skip automatically when required provider
credentials/endpoints are not configured in environment.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.settings import ChunkRefinerSettings, IngestionSettings, load_settings
from core.types import Chunk
from ingestion.transform.chunk_refiner import ChunkRefiner


def _has_real_llm_credentials(settings) -> bool:
    provider = settings.llm.provider.strip().lower()

    if provider == "azure":
        has_endpoint = bool(settings.llm.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"))
        has_api_version = bool(settings.llm.api_version or os.getenv("AZURE_OPENAI_API_VERSION"))
        has_key = bool(settings.llm.api_key or os.getenv("AZURE_OPENAI_API_KEY"))
        return has_endpoint and has_api_version and has_key
    if provider == "openai":
        return bool(settings.llm.api_key or os.getenv("OPENAI_API_KEY"))
    if provider == "deepseek":
        return bool(settings.llm.api_key or os.getenv("DEEPSEEK_API_KEY"))
    if provider == "ollama":
        return bool(os.getenv("OLLAMA_BASE_URL", "").strip())
    return False


def _make_chunk(text: str) -> Chunk:
    return Chunk(
        id="chunk-int-1",
        text=text,
        metadata={"source_path": "docs/integration.pdf", "chunk_index": 0},
        start_offset=0,
        end_offset=len(text),
        source_ref={"doc_id": "doc-int-1"},
    )


@pytest.mark.integration
def test_chunk_refiner_calls_real_llm_when_enabled() -> None:
    settings = load_settings(Path("config/settings.yaml"))
    if not _has_real_llm_credentials(settings):
        pytest.skip("Real LLM credentials not configured for integration test")

    settings.ingestion = IngestionSettings(
        chunk_refiner=ChunkRefinerSettings(
            use_llm=True,
            prompt_path="config/prompts/chunk_refinement.txt",
        )
    )

    refiner = ChunkRefiner(settings)
    noisy = "Page 1 / 9\n\nThis section explains the ingestion pipeline in detail.\n\n---"
    out = refiner.transform([_make_chunk(noisy)])[0]

    assert out.metadata["refined_by"] in {"llm", "rule"}
    assert "ingestion pipeline" in out.text.lower()
    assert len(out.text) > 0


@pytest.mark.integration
def test_chunk_refiner_invalid_model_falls_back_to_rule() -> None:
    settings = load_settings(Path("config/settings.yaml"))
    if not _has_real_llm_credentials(settings):
        pytest.skip("Real LLM credentials not configured for fallback integration test")

    settings.ingestion = IngestionSettings(
        chunk_refiner=ChunkRefinerSettings(
            use_llm=True,
            prompt_path="config/prompts/chunk_refinement.txt",
        )
    )
    settings.llm.model = "definitely-invalid-model-name-for-fallback-check"

    refiner = ChunkRefiner(settings)
    noisy = "Page 8\n\nStable content should survive fallback."
    out = refiner.transform([_make_chunk(noisy)])[0]

    assert out.metadata["refined_by"] == "rule"
    assert "fallback_reason" in out.metadata
    assert "Stable content" in out.text
