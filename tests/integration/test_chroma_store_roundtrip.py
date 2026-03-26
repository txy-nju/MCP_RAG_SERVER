"""Integration tests for the Chroma vector store roundtrip."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.settings import VectorStoreSettings
from libs.vector_store.base_vector_store import VectorStoreQueryResult, VectorStoreRecord
from libs.vector_store.chroma_store import ChromaStore
from libs.vector_store.vector_store_factory import VectorStoreFactory


@pytest.mark.integration
def test_chroma_store_roundtrip_supports_upsert_query_and_filters(tmp_path: Path) -> None:
    persist_path = tmp_path / "chroma"
    vector_store = VectorStoreFactory.create(
        VectorStoreSettings(provider="chroma", collection="roundtrip", persist_path=str(persist_path))
    )

    assert isinstance(vector_store, ChromaStore)
    assert persist_path.exists()

    vector_store.upsert(
        [
            VectorStoreRecord(
                id="chunk-001",
                embedding=[1.0, 0.0, 0.0],
                text="alpha document",
                metadata={"source": "doc-a", "topic": "alpha"},
            ),
            VectorStoreRecord(
                id="chunk-002",
                embedding=[0.0, 1.0, 0.0],
                text="beta document",
                metadata={"source": "doc-b", "topic": "beta"},
            ),
            VectorStoreRecord(
                id="chunk-003",
                embedding=[0.8, 0.2, 0.0],
                text="alpha appendix",
                metadata={"source": "doc-a", "topic": "alpha"},
            ),
        ]
    )

    results = vector_store.query([1.0, 0.0, 0.0], top_k=2)

    assert [result.id for result in results] == ["chunk-001", "chunk-003"]
    assert all(isinstance(result, VectorStoreQueryResult) for result in results)
    assert results[0].text == "alpha document"
    assert results[0].metadata["source"] == "doc-a"
    assert results[0].score >= results[1].score

    filtered_results = vector_store.query([1.0, 0.0, 0.0], top_k=5, filters={"source": "doc-a"})

    assert [result.id for result in filtered_results] == ["chunk-001", "chunk-003"]
    assert all(result.metadata["source"] == "doc-a" for result in filtered_results)

    persisted_store = ChromaStore(provider="chroma", collection="roundtrip", persist_path=str(persist_path))
    persisted_results = persisted_store.query([0.0, 1.0, 0.0], top_k=1)

    assert [result.id for result in persisted_results] == ["chunk-002"]
    assert persisted_results[0].text == "beta document"
