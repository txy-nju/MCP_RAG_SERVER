"""Unit tests for the registry-backed embedding factory."""

from __future__ import annotations

import pytest

from modular_rag.core.settings import (
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
from modular_rag.libs.embedding.base_embedding import BaseEmbedding
from modular_rag.libs.embedding.embedding_factory import EmbeddingFactory


class FakeEmbedding(BaseEmbedding):
    """Simple fake provider used to verify factory routing."""

    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        return [[float(index), float(len(text))] for index, text in enumerate(texts)]


def make_settings(provider: str = "fake", model: str = "demo-model") -> Settings:
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o-mini"),
        embedding=EmbeddingSettings(provider=provider, model=model),
        splitter=SplitterSettings(provider="recursive", chunk_size=1000, chunk_overlap=200),
        vector_store=VectorStoreSettings(provider="chroma", collection="default"),
        retrieval=RetrievalSettings(top_k=5),
        rerank=RerankSettings(provider="none"),
        evaluation=EvaluationSettings(backend="custom"),
        observability=ObservabilitySettings(log_level="INFO", trace_file="logs/traces.jsonl"),
    )


@pytest.fixture(autouse=True)
def reset_embedding_registry() -> None:
    original = dict(EmbeddingFactory._providers)
    original_loaded = EmbeddingFactory._builtin_providers_loaded
    try:
        EmbeddingFactory._providers.clear()
        EmbeddingFactory._builtin_providers_loaded = False
        yield
    finally:
        EmbeddingFactory._providers.clear()
        EmbeddingFactory._providers.update(original)
        EmbeddingFactory._builtin_providers_loaded = original_loaded


@pytest.mark.unit
def test_factory_routes_provider_from_top_level_settings() -> None:
    EmbeddingFactory.register("fake", FakeEmbedding)

    embedding = EmbeddingFactory.create(make_settings())

    assert isinstance(embedding, FakeEmbedding)
    assert embedding.embed(["hi", "world"]) == [[0.0, 2.0], [1.0, 5.0]]


@pytest.mark.unit
def test_factory_accepts_embedding_settings_directly() -> None:
    EmbeddingFactory.register("fake", FakeEmbedding)

    embedding = EmbeddingFactory.create(EmbeddingSettings(provider="fake", model="direct-model"))

    assert isinstance(embedding, FakeEmbedding)
    assert embedding.model == "direct-model"


@pytest.mark.unit
def test_factory_reports_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported embedding provider: missing"):
        EmbeddingFactory.create(make_settings(provider="missing"))


@pytest.mark.unit
def test_register_rejects_non_embedding_classes() -> None:
    with pytest.raises(TypeError, match="must inherit from BaseEmbedding"):
        EmbeddingFactory.register("bad", object)
