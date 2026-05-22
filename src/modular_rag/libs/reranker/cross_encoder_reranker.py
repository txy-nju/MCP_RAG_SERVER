"""Cross-encoder-style reranker implementation."""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable

from modular_rag.libs.reranker.base_reranker import BaseReranker, RerankCandidate, RerankerFallbackError


class CrossEncoderReranker(BaseReranker):
    """Rerank candidates with a deterministic scorer and fallback signaling."""

    def __init__(
        self,
        *,
        provider: str = "cross_encoder",
        model: str | None = None,
        scorer: Callable[[str, list[RerankCandidate]], list[float]] | None = None,
    ) -> None:
        super().__init__(provider=provider, model=model)
        self.scorer = scorer or self._score_candidates

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: Any | None = None,
    ) -> list[RerankCandidate]:
        """Return candidates sorted by cross-encoder scores."""

        del trace
        if not candidates:
            return []
        if not isinstance(query, str) or not query.strip():
            raise ValueError("cross_encoder reranker failed: query must be a non-empty string")

        try:
            scores = self.scorer(query, candidates)
        except RerankerFallbackError:
            raise
        except Exception as exc:  # pragma: no cover - defensive conversion to fallback signal
            raise RerankerFallbackError(f"cross_encoder reranker failed: {exc}") from exc

        if not isinstance(scores, list) or len(scores) != len(candidates):
            raise RerankerFallbackError(
                "cross_encoder reranker failed: scorer must return one numeric score per candidate"
            )
        if not all(isinstance(score, (int, float)) for score in scores):
            raise RerankerFallbackError(
                "cross_encoder reranker failed: scorer must return numeric scores"
            )

        ranked_pairs = sorted(
            zip(candidates, scores, strict=False),
            key=lambda item: (float(item[1]), float(item[0].score)),
            reverse=True,
        )
        return [candidate for candidate, _score in ranked_pairs]

    def _score_candidates(self, query: str, candidates: list[RerankCandidate]) -> list[float]:
        """Compute a lightweight lexical-overlap score as a runnable placeholder."""

        query_tokens = Counter(self._tokenize(query))
        scores: list[float] = []
        for candidate in candidates:
            candidate_tokens = Counter(self._tokenize(candidate.text or ""))
            overlap = sum(min(query_tokens[token], candidate_tokens[token]) for token in query_tokens)
            scores.append(float(overlap))
        return scores

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text into normalized whitespace-separated terms."""

        return [token.strip().lower() for token in text.split() if token.strip()]
