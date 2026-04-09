"""Unit tests for BM25Indexer roundtrip/build/query behavior (C11)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from core.types import ChunkRecord
from ingestion.storage.bm25_indexer import BM25Indexer



def _record(chunk_id: str, sparse: dict[str, float]) -> ChunkRecord:
    return ChunkRecord(
        id=chunk_id,
        text=f"text for {chunk_id}",
        metadata={"source_path": "test.pdf"},
        sparse_vector=sparse,
    )


@pytest.mark.unit
def test_build_query_and_load_roundtrip_returns_stable_top_ids(tmp_path: Path) -> None:
    indexer = BM25Indexer(index_dir=tmp_path)
    records = [
        _record("c1", {"azure": 3.0, "openai": 1.0}),
        _record("c2", {"azure": 1.0, "config": 2.0}),
        _record("c3", {"ollama": 3.0}),
    ]

    indexer.build(records)
    first = indexer.query(["azure", "config"], top_k=3)

    reloaded = BM25Indexer(index_dir=tmp_path)
    reloaded.load()
    second = reloaded.query(["azure", "config"], top_k=3)

    # Only docs containing at least one query term are returned.
    assert [r["chunk_id"] for r in first] == ["c2", "c1"]
    assert first == second


@pytest.mark.unit
def test_idf_matches_formula_for_known_corpus(tmp_path: Path) -> None:
    indexer = BM25Indexer(index_dir=tmp_path)
    indexer.build(
        [
            _record("c1", {"azure": 1.0}),
            _record("c2", {"azure": 1.0, "config": 1.0}),
            _record("c3", {"config": 1.0}),
        ]
    )

    payload = indexer.index_path.read_text(encoding="utf-8")
    assert payload

    indexer_loaded = BM25Indexer(index_dir=tmp_path)
    indexer_loaded.load()

    # N=3, df(azure)=2, df(config)=2
    expected = math.log((3 - 2 + 0.5) / (2 + 0.5))
    azure_idf = indexer_loaded._index["azure"]["idf"]
    config_idf = indexer_loaded._index["config"]["idf"]

    assert azure_idf == pytest.approx(expected, rel=1e-9)
    assert config_idf == pytest.approx(expected, rel=1e-9)


@pytest.mark.unit
def test_incremental_update_adds_new_documents_without_rebuild(tmp_path: Path) -> None:
    indexer = BM25Indexer(index_dir=tmp_path)
    indexer.build([_record("c1", {"azure": 1.0})], rebuild=True)

    indexer.build([_record("c2", {"config": 2.0})], rebuild=False)
    results = indexer.query(["config"], top_k=5)

    assert [row["chunk_id"] for row in results] == ["c2"]


@pytest.mark.unit
def test_rebuild_replaces_existing_index_data(tmp_path: Path) -> None:
    indexer = BM25Indexer(index_dir=tmp_path)
    indexer.build([_record("c1", {"azure": 1.0})], rebuild=True)

    indexer.build([_record("c2", {"ollama": 1.0})], rebuild=True)
    results_old = indexer.query(["azure"], top_k=5)
    results_new = indexer.query(["ollama"], top_k=5)

    assert results_old == []
    assert [row["chunk_id"] for row in results_new] == ["c2"]


@pytest.mark.unit
def test_query_returns_empty_for_unknown_terms_or_invalid_top_k(tmp_path: Path) -> None:
    indexer = BM25Indexer(index_dir=tmp_path)
    indexer.build([_record("c1", {"azure": 1.0})])

    assert indexer.query(["unknown"], top_k=5) == []
    assert indexer.query(["azure"], top_k=0) == []


@pytest.mark.unit
def test_build_skips_empty_sparse_vectors(tmp_path: Path) -> None:
    indexer = BM25Indexer(index_dir=tmp_path)
    records = [
        _record("c1", {}),
        _record("c2", {"azure": 1.0}),
    ]

    indexer.build(records)
    results = indexer.query(["azure"], top_k=5)

    assert [row["chunk_id"] for row in results] == ["c2"]


@pytest.mark.unit
def test_load_raises_when_index_file_missing(tmp_path: Path) -> None:
    indexer = BM25Indexer(index_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        indexer.load()
