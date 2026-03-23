"""Unit tests for the registry-backed splitter factory."""

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
from libs.splitter.base_splitter import BaseSplitter
from libs.splitter.splitter_factory import SplitterFactory


class FakeSplitter(BaseSplitter):
    """Simple fake provider used to verify factory routing."""

    def split_text(self, text: str, trace: object | None = None) -> list[str]:
        return [f"{self.provider}:{self.chunk_size}:{self.chunk_overlap}", text]


def make_settings(provider: str = "fake") -> Settings:
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o-mini"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        splitter=SplitterSettings(provider=provider, chunk_size=512, chunk_overlap=64),
        vector_store=VectorStoreSettings(provider="chroma", collection="default"),
        retrieval=RetrievalSettings(top_k=5),
        rerank=RerankSettings(provider="none"),
        evaluation=EvaluationSettings(backend="custom"),
        observability=ObservabilitySettings(log_level="INFO", trace_file="logs/traces.jsonl"),
    )


@pytest.fixture(autouse=True)
def reset_splitter_registry() -> None:
    original = dict(SplitterFactory._providers)
    try:
        SplitterFactory._providers.clear()
        yield
    finally:
        SplitterFactory._providers.clear()
        SplitterFactory._providers.update(original)


@pytest.mark.unit
def test_factory_routes_provider_from_top_level_settings() -> None:
    SplitterFactory.register("fake", FakeSplitter)

    splitter = SplitterFactory.create(make_settings())

    assert isinstance(splitter, FakeSplitter)
    assert splitter.split_text("hello") == ["fake:512:64", "hello"]


@pytest.mark.unit
def test_factory_accepts_splitter_settings_directly() -> None:
    SplitterFactory.register("fake", FakeSplitter)

    splitter = SplitterFactory.create(SplitterSettings(provider="fake", chunk_size=256, chunk_overlap=32))

    assert isinstance(splitter, FakeSplitter)
    assert splitter.chunk_size == 256
    assert splitter.chunk_overlap == 32


@pytest.mark.unit
def test_factory_reports_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported splitter provider: missing"):
        SplitterFactory.create(make_settings(provider="missing"))


@pytest.mark.unit
def test_register_rejects_non_splitter_classes() -> None:
    with pytest.raises(TypeError, match="must inherit from BaseSplitter"):
        SplitterFactory.register("bad", object)
