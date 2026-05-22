"""Unit tests for the cross-encoder reranker."""

from __future__ import annotations

import pytest

from modular_rag.core.settings import RerankSettings
from modular_rag.libs.reranker.base_reranker import RerankCandidate, RerankerFallbackError
from modular_rag.libs.reranker.cross_encoder_reranker import CrossEncoderReranker
from modular_rag.libs.reranker.reranker_factory import RerankerFactory


def make_candidates() -> list[RerankCandidate]:
    return [
        RerankCandidate(id="a", score=0.2, text="alpha beta"),
        RerankCandidate(id="b", score=0.8, text="beta gamma"),
        RerankCandidate(id="c", score=0.1, text="delta"),
    ]


@pytest.fixture(autouse=True)
def reset_reranker_registry() -> None:
    original = dict(RerankerFactory._providers)
    original_flag = RerankerFactory._builtin_providers_loaded
    try:
        RerankerFactory._providers.clear()
        RerankerFactory._providers["none"] = __import__(
            "libs.reranker.base_reranker", fromlist=["NoneReranker"]
        ).NoneReranker
        RerankerFactory._builtin_providers_loaded = False
        yield
    finally:
        RerankerFactory._providers.clear()
        RerankerFactory._providers.update(original)
        RerankerFactory._builtin_providers_loaded = original_flag


@pytest.mark.unit
def test_cross_encoder_reranker_reorders_candidates_with_injected_scorer() -> None:
    reranker = CrossEncoderReranker(
        scorer=lambda query, candidates: [0.1, 0.9, 0.3],
    )

    reranked = reranker.rerank("beta", make_candidates())

    assert [candidate.id for candidate in reranked] == ["b", "c", "a"]


@pytest.mark.unit
def test_cross_encoder_reranker_uses_placeholder_lexical_scoring() -> None:
    reranker = CrossEncoderReranker()

    reranked = reranker.rerank("beta gamma", make_candidates())

    assert [candidate.id for candidate in reranked] == ["b", "a", "c"]


@pytest.mark.unit
def test_cross_encoder_reranker_raises_fallback_error_on_scorer_failure() -> None:
    def failing_scorer(query: str, candidates: list[RerankCandidate]) -> list[float]:
        raise RuntimeError("model timeout")

    reranker = CrossEncoderReranker(scorer=failing_scorer)

    with pytest.raises(RerankerFallbackError, match="model timeout"):
        reranker.rerank("beta", make_candidates())


@pytest.mark.unit
def test_cross_encoder_reranker_rejects_invalid_score_shape() -> None:
    reranker = CrossEncoderReranker(scorer=lambda query, candidates: [0.2])

    with pytest.raises(RerankerFallbackError, match="one numeric score per candidate"):
        reranker.rerank("beta", make_candidates())


@pytest.mark.unit
def test_factory_creates_builtin_cross_encoder_reranker() -> None:
    reranker = RerankerFactory.create(RerankSettings(provider="cross_encoder"))

    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.provider == "cross_encoder"
