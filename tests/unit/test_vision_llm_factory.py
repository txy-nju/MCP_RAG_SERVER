"""Unit tests for the registry-backed vision LLM factory."""

from __future__ import annotations

from pathlib import Path

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
from modular_rag.libs.llm.base_vision_llm import BaseVisionLLM, ChatResponse
from modular_rag.libs.llm.llm_factory import LLMFactory


class FakeVisionLLM(BaseVisionLLM):
    """Simple fake provider used to verify vision factory routing."""

    def chat_with_image(
        self,
        text: str,
        image_path: str | bytes,
        trace: object | None = None,
    ) -> ChatResponse:
        prepared = self.preprocess_image(image_path)
        return ChatResponse(
            content=f"{self.provider}:{self.model}:{text}:{prepared}",
            provider=self.provider,
            model=self.model,
            metadata={"trace_provided": trace is not None},
        )


def make_settings(
    provider: str = "fake",
    model: str = "vision-demo-model",
    *,
    include_vision: bool = True,
) -> Settings:
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o-mini"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        splitter=SplitterSettings(provider="recursive", chunk_size=1000, chunk_overlap=200),
        vector_store=VectorStoreSettings(provider="chroma", collection="default"),
        retrieval=RetrievalSettings(top_k=5),
        rerank=RerankSettings(provider="none"),
        evaluation=EvaluationSettings(backend="custom"),
        observability=ObservabilitySettings(log_level="INFO", trace_file="logs/traces.jsonl"),
        vision_llm=LLMSettings(provider=provider, model=model) if include_vision else None,
    )


@pytest.fixture(autouse=True)
def reset_llm_registries() -> None:
    original_text = dict(LLMFactory._providers)
    original_vision = dict(LLMFactory._vision_providers)
    original_loaded_flag = LLMFactory._builtin_providers_loaded
    try:
        LLMFactory._providers.clear()
        LLMFactory._vision_providers.clear()
        LLMFactory._builtin_providers_loaded = False
        yield
    finally:
        LLMFactory._providers.clear()
        LLMFactory._providers.update(original_text)
        LLMFactory._vision_providers.clear()
        LLMFactory._vision_providers.update(original_vision)
        LLMFactory._builtin_providers_loaded = original_loaded_flag


@pytest.mark.unit
def test_factory_routes_vision_provider_from_top_level_settings() -> None:
    LLMFactory.register_vision("fake", FakeVisionLLM)

    llm = LLMFactory.create_vision_llm(make_settings())

    assert isinstance(llm, FakeVisionLLM)
    response = llm.chat_with_image("describe", Path("assets") / "cat.png")
    assert response.content == f"fake:vision-demo-model:describe:{Path('assets') / 'cat.png'}"


@pytest.mark.unit
def test_factory_accepts_direct_vision_llm_settings() -> None:
    LLMFactory.register_vision("fake", FakeVisionLLM)

    llm = LLMFactory.create_vision_llm(LLMSettings(provider="fake", model="direct-vision-model"))

    assert isinstance(llm, FakeVisionLLM)
    assert llm.model == "direct-vision-model"


@pytest.mark.unit
def test_factory_requires_vision_settings_on_top_level_settings() -> None:
    with pytest.raises(ValueError, match="Settings.vision_llm must be configured"):
        LLMFactory.create_vision_llm(make_settings(include_vision=False))


@pytest.mark.unit
def test_factory_reports_unknown_vision_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported vision LLM provider: missing"):
        LLMFactory.create_vision_llm(LLMSettings(provider="missing", model="vision-model"))


@pytest.mark.unit
def test_register_vision_rejects_non_vision_llm_classes() -> None:
    with pytest.raises(TypeError, match="must inherit from BaseVisionLLM"):
        LLMFactory.register_vision("bad", object)
