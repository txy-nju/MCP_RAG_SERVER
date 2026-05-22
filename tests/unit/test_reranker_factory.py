"""Unit tests for the registry-backed reranker factory."""

from __future__ import annotations

import pytest

from modular_rag.core.settings import (
    EmbeddingSettings,
    EvaluationSettings,
    LLMSettings,
    ObservabilitySettings,
    RerankSettings,
    RetrievalSettings,
    Settings,
    SplitterSettings,
    VectorStoreSettings,
)
from modular_rag.libs.reranker.base_reranker import BaseReranker, NoneReranker, RerankCandidate
from modular_rag.libs.reranker.reranker_factory import RerankerFactory


class FakeReranker(BaseReranker):
    """Simple fake provider used to verify factory routing."""

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: object | None = None,
    ) -> list[RerankCandidate]:
        return sorted(candidates, key=lambda candidate: candidate.score, reverse=True)


class TrackingLLMReranker(BaseReranker):
    """Reranker used to verify factory passes top-level Settings for llm provider."""

    last_from_settings_type: type | None = None

    @classmethod
    def from_settings(cls, settings):  # type: ignore[override]
        cls.last_from_settings_type = type(settings)
        return cls(provider="llm")

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: object | None = None,
    ) -> list[RerankCandidate]:
        del query, trace
        return list(candidates)


class TrackingCrossEncoderReranker(BaseReranker):
    """Minimal cross-encoder placeholder used to keep registry stable in tests."""

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: object | None = None,
    ) -> list[RerankCandidate]:
        del query, trace
        return list(candidates)


def make_settings(provider: str = "fake") -> Settings:
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o-mini"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        splitter=SplitterSettings(provider="recursive", chunk_size=1000, chunk_overlap=200),
        vector_store=VectorStoreSettings(provider="chroma", collection="default"),
        retrieval=RetrievalSettings(top_k=5),
        rerank=RerankSettings(provider=provider),
        evaluation=EvaluationSettings(backend="custom"),
        observability=ObservabilitySettings(log_level="INFO", trace_file="logs/traces.jsonl"),
    )


def make_candidates() -> list[RerankCandidate]:
    return [
        RerankCandidate(id="a", score=0.2, text="alpha"),
        RerankCandidate(id="b", score=0.8, text="beta"),
    ]


@pytest.fixture(autouse=True)
def reset_reranker_registry() -> None:
    original = dict(RerankerFactory._providers)
    original_flag = RerankerFactory._builtin_providers_loaded
    try:
        RerankerFactory._providers.clear()
        RerankerFactory._providers["none"] = NoneReranker
        RerankerFactory._builtin_providers_loaded = False
        yield
    finally:
        RerankerFactory._providers.clear()
        RerankerFactory._providers.update(original)
        RerankerFactory._builtin_providers_loaded = original_flag


@pytest.mark.unit
def test_factory_routes_provider_from_top_level_settings() -> None:
    RerankerFactory.register("fake", FakeReranker)

    reranker = RerankerFactory.create(make_settings())

    assert isinstance(reranker, FakeReranker)
    assert [candidate.id for candidate in reranker.rerank("hi", make_candidates())] == ["b", "a"]


@pytest.mark.unit
def test_factory_accepts_rerank_settings_directly() -> None:
    RerankerFactory.register("fake", FakeReranker)

    reranker = RerankerFactory.create(RerankSettings(provider="fake"))

    assert isinstance(reranker, FakeReranker)
    assert reranker.provider == "fake"


@pytest.mark.unit
def test_none_provider_returns_stable_fallback() -> None:
    reranker = RerankerFactory.create(make_settings(provider="none"))
    candidates = make_candidates()

    reranked = reranker.rerank("hi", candidates)

    assert isinstance(reranker, NoneReranker)
    assert reranked == candidates
    assert reranked is not candidates


@pytest.mark.unit
def test_factory_reports_unknown_provider() -> None:
    with pytest.raises(ValueError, match="Unsupported reranker provider: missing"):
        RerankerFactory.create(make_settings(provider="missing"))


@pytest.mark.unit
def test_register_rejects_non_reranker_classes() -> None:
    with pytest.raises(TypeError, match="must inherit from BaseReranker"):
        RerankerFactory.register("bad", object)


@pytest.mark.unit
def test_register_rejects_empty_provider_name() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        RerankerFactory.register("   ", FakeReranker)


@pytest.mark.unit
def test_factory_create_rejects_invalid_settings_type() -> None:
    with pytest.raises(TypeError, match="expects Settings or RerankSettings"):
        RerankerFactory.create(object())  # type: ignore[arg-type]


@pytest.mark.unit
def test_unregister_none_keeps_builtin_fallback() -> None:
    RerankerFactory.unregister("none")

    reranker = RerankerFactory.create(RerankSettings(provider="none"))

    assert isinstance(reranker, NoneReranker)


@pytest.mark.unit
def test_llm_provider_uses_top_level_settings_for_creation() -> None:
    RerankerFactory.register("llm", TrackingLLMReranker)
    RerankerFactory.register("cross_encoder", TrackingCrossEncoderReranker)
    RerankerFactory._builtin_providers_loaded = True

    reranker = RerankerFactory.create(make_settings(provider="llm"))

    assert isinstance(reranker, TrackingLLMReranker)
    assert TrackingLLMReranker.last_from_settings_type is Settings
