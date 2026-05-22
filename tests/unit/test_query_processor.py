"""Unit tests for QueryProcessor (D1)."""

from __future__ import annotations

import pytest

from modular_rag.core.query_engine.query_processor import QueryProcessor


@pytest.mark.unit
def test_process_extracts_non_empty_keywords_for_normal_query() -> None:
    processor = QueryProcessor()

    processed = processor.process("How to configure modular rag retrieval pipeline?")

    assert processed.keywords
    assert "modular" in processed.keywords
    assert "retrieval" in processed.keywords
    assert isinstance(processed.filters, dict)


@pytest.mark.unit
def test_process_parses_inline_filters_into_dict() -> None:
    processor = QueryProcessor()

    processed = processor.process("collection:tech doc_type:pdf explain ingestion pipeline")

    assert processed.filters == {"collection": "tech", "doc_type": "pdf"}
    assert "ingestion" in processed.keywords


@pytest.mark.unit
def test_process_merges_explicit_filters_and_overrides_inline() -> None:
    processor = QueryProcessor()

    processed = processor.process(
        "collection:old language:en find bm25 keyword scoring",
        filters={"collection": "new", "access_level": "internal"},
    )

    assert processed.filters["collection"] == "new"
    assert processed.filters["language"] == "en"
    assert processed.filters["access_level"] == "internal"


@pytest.mark.unit
def test_process_falls_back_to_raw_text_when_tokenizer_yields_no_keyword() -> None:
    processor = QueryProcessor()

    processed = processor.process("--- ??? ---")

    assert processed.keywords == ["--- ??? ---"]
