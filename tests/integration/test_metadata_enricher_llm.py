"""Integration tests for MetadataEnricher with real provider wiring.

These tests are opt-in and will skip automatically when required provider
credentials/endpoints are not configured in environment.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from core.settings import load_settings
from core.types import Chunk
from ingestion.transform.metadata_enricher import MetadataEnricher


def _is_real_secret(value: str | None) -> bool:
    if not value:
        return False
    lowered = value.strip().lower()
    return lowered not in {"", "your-api-key", "changeme", "replace-me"}


def _has_real_llm_credentials(settings) -> bool:
    provider = settings.llm.provider.strip().lower()

    if provider == "azure":
        has_endpoint = bool(settings.llm.endpoint or os.getenv("AZURE_OPENAI_ENDPOINT"))
        has_api_version = bool(settings.llm.api_version or os.getenv("AZURE_OPENAI_API_VERSION"))
        has_key = _is_real_secret(settings.llm.api_key) or _is_real_secret(os.getenv("AZURE_OPENAI_API_KEY"))
        return has_endpoint and has_api_version and has_key
    if provider == "openai":
        has_key = _is_real_secret(settings.llm.api_key) or _is_real_secret(os.getenv("OPENAI_API_KEY"))
        return has_key
    if provider == "deepseek":
        has_key = _is_real_secret(settings.llm.api_key) or _is_real_secret(os.getenv("DEEPSEEK_API_KEY"))
        return has_key
    if provider == "ollama":
        return bool((settings.llm.api_url or os.getenv("OLLAMA_BASE_URL", "")).strip())
    return False


def _make_chunk(text: str) -> Chunk:
    return Chunk(
        id="chunk-meta-int-1",
        text=text,
        metadata={"source_path": "docs/integration.pdf", "chunk_index": 0},
        start_offset=0,
        end_offset=len(text),
        source_ref={"doc_id": "doc-int-1"},
    )


@pytest.mark.integration
def test_metadata_enricher_calls_real_llm_when_enabled() -> None:
    settings = load_settings(Path("config/settings.yaml"))
    if not _has_real_llm_credentials(settings):
        pytest.skip("Real LLM credentials not configured for integration test")

    settings.ingestion.metadata_enricher.use_llm = True

    enricher = MetadataEnricher(settings)
    chunk = _make_chunk("RAG pipeline coordinates loading, splitting, and retrieval over project documents.")
    out = enricher.transform([chunk])[0]

    assert out.metadata["enriched_by"] in {"llm", "rule"}
    assert out.metadata["title"]
    assert out.metadata["summary"]
    assert isinstance(out.metadata["tags"], list)
    assert len(out.metadata["tags"]) > 0


@pytest.mark.integration
def test_metadata_enricher_invalid_model_falls_back_to_rule() -> None:
    settings = load_settings(Path("config/settings.yaml"))
    if not _has_real_llm_credentials(settings):
        pytest.skip("Real LLM credentials not configured for fallback integration test")

    settings.ingestion.metadata_enricher.use_llm = True
    settings.llm.model = "definitely-invalid-model-name-for-metadata-fallback"

    enricher = MetadataEnricher(settings)
    chunk = _make_chunk("Metadata enrichment should not break ingestion when model calls fail.")
    out = enricher.transform([chunk])[0]

    assert out.metadata["enriched_by"] == "rule"
    assert "fallback_reason" in out.metadata
    assert out.metadata["title"]
    assert out.metadata["summary"]
