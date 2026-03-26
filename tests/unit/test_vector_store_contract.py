"""Contract tests for the vector store abstraction and factory."""

from __future__ import annotations

import pytest

from core.settings import (
    EmbeddingSettings,
    EvaluationSettings,
    LLMSettings,
    ObservabilitySettings,
    RetrievalSettings,
    RerankSettings,
    Settings,
    SplitterSettings,
    VectorStoreSettings,
)
from libs.vector_store.base_vector_store import BaseVectorStore, VectorStoreQueryResult, VectorStoreRecord
from libs.vector_store.vector_store_factory import VectorStoreFactory


class FakeVectorStore(BaseVectorStore):
    """Simple fake provider used to verify vector store contracts."""

    def __init__(self, *, provider: str, collection: str) -> None:
        super().__init__(provider=provider, collection=collection)
        self.records: list[VectorStoreRecord] = []

    def upsert(self, records: list[VectorStoreRecord], trace: object | None = None) -> None:
        self.records = list(records)

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, object] | None = None,
        trace: object | None = None,
    ) -> list[VectorStoreQueryResult]:
        return [
            VectorStoreQueryResult(
                id=record.id,
                score=1.0 - (index * 0.1),
                text=record.text,
                metadata={**record.metadata, "filters": filters or {}, "query_dim": len(vector)},
            )
            for index, record in enumerate(self.records[:top_k])
        ]


def make_settings(provider: str = "fake", collection: str = "demo") -> Settings:
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o-mini"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        splitter=SplitterSettings(provider="recursive", chunk_size=1000, chunk_overlap=200),
        vector_store=VectorStoreSettings(provider=provider, collection=collection, persist_path="data/db/chroma"),
        retrieval=RetrievalSettings(top_k=5),
        rerank=RerankSettings(provider="none"),
        evaluation=EvaluationSettings(backend="custom"),
        observability=ObservabilitySettings(log_level="INFO", trace_file="logs/traces.jsonl"),
    )


@pytest.fixture(autouse=True)
def reset_vector_store_registry() -> None:
    original = dict(VectorStoreFactory._providers)
    original_loaded = VectorStoreFactory._builtin_providers_loaded
    try:
        VectorStoreFactory._providers.clear()
        VectorStoreFactory._builtin_providers_loaded = False
        yield
    finally:
        VectorStoreFactory._providers.clear()
        VectorStoreFactory._providers.update(original)
        VectorStoreFactory._builtin_providers_loaded = original_loaded


@pytest.mark.unit
def test_factory_routes_provider_from_top_level_settings() -> None:
    VectorStoreFactory.register("fake", FakeVectorStore)

    vector_store = VectorStoreFactory.create(make_settings())

    assert isinstance(vector_store, FakeVectorStore)
    assert vector_store.collection == "demo"


@pytest.mark.unit
def test_factory_accepts_vector_store_settings_directly() -> None:
    VectorStoreFactory.register("fake", FakeVectorStore)

    vector_store = VectorStoreFactory.create(
        VectorStoreSettings(provider="fake", collection="direct", persist_path="data/db/chroma")
    )

    assert isinstance(vector_store, FakeVectorStore)
    assert vector_store.collection == "direct"


@pytest.mark.unit
def test_factory_reports_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported vector store provider: missing"):
        VectorStoreFactory.create(make_settings(provider="missing"))


@pytest.mark.unit
def test_register_rejects_non_vector_store_classes() -> None:
    with pytest.raises(TypeError, match="must inherit from BaseVectorStore"):
        VectorStoreFactory.register("bad", object)


@pytest.mark.unit
def test_vector_store_contract_preserves_record_and_result_shapes() -> None:
    VectorStoreFactory.register("fake", FakeVectorStore)
    vector_store = VectorStoreFactory.create(make_settings())
    record = VectorStoreRecord(
        id="chunk-001",
        embedding=[0.1, 0.2, 0.3],
        text="hello world",
        metadata={"source": "doc-a", "chunk_index": 0},
    )

    vector_store.upsert([record])
    results = vector_store.query([0.3, 0.2, 0.1], top_k=1, filters={"source": "doc-a"})

    assert len(results) == 1
    result = results[0]
    assert isinstance(result, VectorStoreQueryResult)
    assert result.id == "chunk-001"
    assert result.text == "hello world"
    assert result.metadata["source"] == "doc-a"
    assert result.metadata["filters"] == {"source": "doc-a"}
    assert result.metadata["query_dim"] == 3
    assert isinstance(result.score, float)
