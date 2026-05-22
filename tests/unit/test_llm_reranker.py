"""Unit tests for the LLM-backed reranker."""

from __future__ import annotations

import json

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
from modular_rag.libs.llm.base_llm import BaseLLM
from modular_rag.libs.llm.llm_factory import LLMFactory
from modular_rag.libs.reranker.base_reranker import NoneReranker, RerankCandidate, RerankerFallbackError
from modular_rag.libs.reranker.llm_reranker import LLMReranker
from modular_rag.libs.reranker.reranker_factory import RerankerFactory


class FakeLLM(BaseLLM):
    """Deterministic fake LLM used for reranker tests."""

    def __init__(self, *, provider: str, model: str, response: str = '{"ranked_ids": ["b", "a"]}') -> None:
        super().__init__(provider=provider, model=model)
        self.response = response
        self.calls: list[list[dict[str, object]]] = []

    def chat(self, messages: list[dict[str, object]]) -> str:
        self.calls.append(messages)
        return self.response


class FailingLLM(BaseLLM):
    """Fake LLM that simulates provider failures."""

    def chat(self, messages: list[dict[str, object]]) -> str:
        raise ValueError("provider unavailable")


class FactoryFakeLLM(FakeLLM):
    """Fake LLM that can be constructed through ``LLMFactory``."""


@pytest.fixture(autouse=True)
def reset_registries() -> None:
    original_rerankers = dict(RerankerFactory._providers)
    original_llms = dict(LLMFactory._providers)
    original_reranker_flag = RerankerFactory._builtin_providers_loaded
    original_llm_flag = LLMFactory._builtin_providers_loaded
    try:
        RerankerFactory._providers.clear()
        RerankerFactory._providers["none"] = NoneReranker
        RerankerFactory._builtin_providers_loaded = False
        LLMFactory._providers.clear()
        LLMFactory._builtin_providers_loaded = False
        yield
    finally:
        RerankerFactory._providers.clear()
        RerankerFactory._providers.update(original_rerankers)
        RerankerFactory._builtin_providers_loaded = original_reranker_flag
        LLMFactory._providers.clear()
        LLMFactory._providers.update(original_llms)
        LLMFactory._builtin_providers_loaded = original_llm_flag


def make_settings() -> Settings:
    return Settings(
        llm=LLMSettings(provider="fake", model="demo-model"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        splitter=SplitterSettings(provider="recursive", chunk_size=1000, chunk_overlap=200),
        vector_store=VectorStoreSettings(provider="chroma", collection="default"),
        retrieval=RetrievalSettings(top_k=5),
        rerank=RerankSettings(provider="llm"),
        evaluation=EvaluationSettings(backend="custom"),
        observability=ObservabilitySettings(log_level="INFO", trace_file="logs/traces.jsonl"),
    )


def make_candidates() -> list[RerankCandidate]:
    return [
        RerankCandidate(id="a", score=0.2, text="alpha"),
        RerankCandidate(id="b", score=0.8, text="beta"),
        RerankCandidate(id="c", score=0.1, text="gamma"),
    ]


@pytest.mark.unit
def test_llm_reranker_reorders_candidates_and_appends_missing_ids() -> None:
    llm = FakeLLM(provider="fake", model="demo-model", response='{"ranked_ids": ["b", "a"]}')
    reranker = LLMReranker(provider="llm", model=llm.model, llm=llm, prompt_template="Rank chunks")

    reranked = reranker.rerank("find beta", make_candidates())

    assert [candidate.id for candidate in reranked] == ["b", "a", "c"]
    assert llm.calls[0][0]["content"] == "Rank chunks"
    user_payload = json.loads(str(llm.calls[0][1]["content"]).split("\n", 1)[1])
    assert user_payload["query"] == "find beta"
    assert [item["id"] for item in user_payload["candidates"]] == ["a", "b", "c"]


@pytest.mark.unit
def test_llm_reranker_reads_prompt_from_file(tmp_path) -> None:
    prompt_path = tmp_path / "rerank.txt"
    prompt_path.write_text("Use this file prompt", encoding="utf-8")
    llm = FakeLLM(provider="fake", model="demo-model")
    reranker = LLMReranker(provider="llm", model=llm.model, llm=llm, prompt_path=str(prompt_path))

    reranker.rerank("find beta", make_candidates())

    assert llm.calls[0][0]["content"] == "Use this file prompt"


@pytest.mark.unit
def test_llm_reranker_raises_fallback_error_for_invalid_schema() -> None:
    llm = FakeLLM(provider="fake", model="demo-model", response='{"ids": ["b"]}')
    reranker = LLMReranker(provider="llm", model=llm.model, llm=llm, prompt_template="Rank chunks")

    with pytest.raises(RerankerFallbackError, match="ranked_ids"):
        reranker.rerank("find beta", make_candidates())


@pytest.mark.unit
def test_llm_reranker_converts_provider_failures_to_fallback_error() -> None:
    llm = FailingLLM(provider="fake", model="demo-model")
    reranker = LLMReranker(provider="llm", model=llm.model, llm=llm, prompt_template="Rank chunks")

    with pytest.raises(RerankerFallbackError, match="provider unavailable"):
        reranker.rerank("find beta", make_candidates())


@pytest.mark.unit
def test_factory_creates_builtin_llm_reranker_from_top_level_settings() -> None:
    LLMFactory.register("fake", FactoryFakeLLM)

    reranker = RerankerFactory.create(make_settings())

    assert isinstance(reranker, LLMReranker)
    assert reranker.provider == "llm"
    assert reranker.model == "demo-model"
