"""Tests for configuration loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.settings import Settings, load_settings
from main import main


VALID_CONFIG = """
llm:
  provider: azure
  model: gpt-4o-mini
embedding:
  provider: openai
  model: text-embedding-3-small
splitter:
  provider: recursive
  chunk_size: 1000
  chunk_overlap: 200
vector_store:
  provider: chroma
  collection: test
retrieval:
  top_k: 5
rerank:
  provider: none
evaluation:
  backend: custom
observability:
  log_level: INFO
  trace_file: logs/traces.jsonl
""".strip()


INVALID_CONFIG = """
llm:
  provider: azure
  model: gpt-4o-mini
embedding:
  model: text-embedding-3-small
splitter:
  provider: recursive
  chunk_size: 1000
  chunk_overlap: 200
vector_store:
  provider: chroma
  collection: test
retrieval:
  top_k: 5
rerank:
  provider: none
evaluation:
  backend: custom
observability:
  log_level: INFO
  trace_file: logs/traces.jsonl
""".strip()


@pytest.mark.unit
def test_load_settings_returns_settings_object(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(VALID_CONFIG, encoding="utf-8")

    settings = load_settings(config_path)

    assert isinstance(settings, Settings)
    assert settings.embedding.provider == "openai"
    assert settings.splitter.provider == "recursive"
    assert settings.vector_store.collection == "test"


@pytest.mark.unit
def test_load_settings_reports_missing_field_path(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(INVALID_CONFIG, encoding="utf-8")

    with pytest.raises(ValueError, match="embedding.provider"):
        load_settings(config_path)


@pytest.mark.unit
def test_main_returns_zero_for_valid_config(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(VALID_CONFIG, encoding="utf-8")

    assert main(config_path) == 0


@pytest.mark.unit
def test_main_returns_one_for_invalid_config(tmp_path: Path) -> None:
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(INVALID_CONFIG, encoding="utf-8")

    assert main(config_path) == 1


@pytest.mark.unit
def test_main_returns_one_for_missing_config_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing-settings.yaml"

    assert main(missing_path) == 1


@pytest.mark.unit
def test_load_settings_rejects_overlap_greater_than_chunk_size(tmp_path: Path) -> None:
    invalid_splitter_config = VALID_CONFIG.replace("chunk_size: 1000", "chunk_size: 100").replace(
        "chunk_overlap: 200", "chunk_overlap: 100"
    )
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(invalid_splitter_config, encoding="utf-8")

    with pytest.raises(ValueError, match="splitter.chunk_overlap must be smaller than splitter.chunk_size"):
        load_settings(config_path)
