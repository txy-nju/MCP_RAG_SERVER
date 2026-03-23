"""Unit tests for the registry-backed LLM factory."""

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
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory


class FakeLLM(BaseLLM):
    """Simple fake provider used to verify factory routing."""

    def chat(self, messages: list[dict[str, object]]) -> str:
        return f"{self.provider}:{self.model}:{len(messages)}"


def make_settings(provider: str = "fake", model: str = "demo-model") -> Settings:
    return Settings(
        llm=LLMSettings(provider=provider, model=model),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        splitter=SplitterSettings(provider="recursive", chunk_size=1000, chunk_overlap=200),
        vector_store=VectorStoreSettings(provider="chroma", collection="default"),
        retrieval=RetrievalSettings(top_k=5),
        rerank=RerankSettings(provider="none"),
        evaluation=EvaluationSettings(backend="custom"),
        observability=ObservabilitySettings(log_level="INFO", trace_file="logs/traces.jsonl"),
    )


@pytest.fixture(autouse=True)
def reset_llm_registry() -> None:
    original = dict(LLMFactory._providers)
    try:
        LLMFactory._providers.clear()
        yield
    finally:
        LLMFactory._providers.clear()
        LLMFactory._providers.update(original)


@pytest.mark.unit
def test_factory_routes_provider_from_top_level_settings() -> None:
    LLMFactory.register("fake", FakeLLM)

    llm = LLMFactory.create(make_settings())

    assert isinstance(llm, FakeLLM)
    assert llm.chat([{"role": "user", "content": "hi"}]) == "fake:demo-model:1"


@pytest.mark.unit
def test_factory_accepts_llm_settings_directly() -> None:
    LLMFactory.register("fake", FakeLLM)

    llm = LLMFactory.create(LLMSettings(provider="fake", model="direct-model"))

    assert isinstance(llm, FakeLLM)
    assert llm.model == "direct-model"


@pytest.mark.unit
def test_factory_reports_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported LLM provider: missing"):
        LLMFactory.create(make_settings(provider="missing"))


@pytest.mark.unit
def test_register_rejects_non_llm_classes() -> None:
    with pytest.raises(TypeError, match="must inherit from BaseLLM"):
        LLMFactory.register("bad", object)
