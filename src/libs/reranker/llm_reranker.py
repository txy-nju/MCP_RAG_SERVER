"""LLM-backed reranker implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.settings import Settings
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory
from libs.reranker.base_reranker import BaseReranker, RerankCandidate, RerankerFallbackError


class LLMReranker(BaseReranker):
    """Rerank candidates with an LLM that returns structured ranked ids."""

    default_prompt_path = "config/prompts/rerank.txt"
    default_max_candidates = 20

    def __init__(
        self,
        *,
        provider: str = "llm",
        model: str | None,
        llm: BaseLLM,
        prompt_template: str | None = None,
        prompt_path: str = default_prompt_path,
        max_candidates: int = default_max_candidates,
    ) -> None:
        super().__init__(provider=provider, model=model)
        self.llm = llm
        self.prompt_template = prompt_template
        self.prompt_path = prompt_path
        self.max_candidates = max_candidates

    @classmethod
    def from_settings(cls, settings: Any) -> "LLMReranker":
        """Build the reranker from top-level settings and the configured LLM."""

        if not isinstance(settings, Settings):
            raise ValueError("LLMReranker requires top-level Settings so it can initialize the configured LLM")

        llm = LLMFactory.create(settings)
        return cls(
            provider=str(settings.rerank.provider),
            model=llm.model,
            llm=llm,
            prompt_path=str(settings.rerank.prompt_path),
            max_candidates=int(settings.rerank.max_candidates),
        )

    def rerank(
        self,
        query: str,
        candidates: list[RerankCandidate],
        trace: Any | None = None,
    ) -> list[RerankCandidate]:
        """Return candidates reordered according to the LLM response."""

        del trace
        if not candidates:
            return []
        if not isinstance(query, str) or not query.strip():
            raise ValueError("llm reranker failed: query must be a non-empty string")

        limited_candidates = list(candidates[: self.max_candidates])
        try:
            prompt = self._render_prompt(query=query, candidates=limited_candidates)
            response = self.llm.chat(prompt)
            ranked_ids = self._parse_ranked_ids(response=response, candidates=limited_candidates)
        except RerankerFallbackError:
            raise
        except Exception as exc:  # pragma: no cover - defensive conversion to fallback signal
            raise RerankerFallbackError(f"llm reranker failed: {exc}") from exc

        ranked_candidates = self._order_candidates(limited_candidates, ranked_ids)
        if len(candidates) > len(limited_candidates):
            ranked_candidates.extend(candidates[len(limited_candidates) :])
        return ranked_candidates

    def _render_prompt(self, *, query: str, candidates: list[RerankCandidate]) -> list[dict[str, str]]:
        """Build the chat prompt sent to the backing LLM."""

        candidate_payload = [
            {
                "id": candidate.id,
                "score": candidate.score,
                "text": candidate.text or "",
                "metadata": candidate.metadata,
            }
            for candidate in candidates
        ]
        instructions = self._load_prompt_template().strip()
        user_payload = json.dumps(
            {
                "query": query,
                "candidates": candidate_payload,
                "response_schema": {"ranked_ids": [candidate.id for candidate in candidates]},
            },
            ensure_ascii=False,
            indent=2,
        )
        return [
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": (
                    "Return JSON only with the shape {\"ranked_ids\": [\"candidate_id\", ...]}.\n"
                    f"{user_payload}"
                ),
            },
        ]

    def _load_prompt_template(self) -> str:
        """Load the prompt template from an injected string or the configured file."""

        if self.prompt_template is not None:
            if not self.prompt_template.strip():
                raise RerankerFallbackError("llm reranker failed: prompt template cannot be empty")
            return self.prompt_template

        prompt_path = Path(self.prompt_path)
        try:
            content = prompt_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RerankerFallbackError(f"llm reranker failed: unable to read prompt file '{prompt_path}'") from exc
        if not content.strip():
            raise RerankerFallbackError(f"llm reranker failed: prompt file '{prompt_path}' is empty")
        return content

    def _parse_ranked_ids(self, *, response: str, candidates: list[RerankCandidate]) -> list[str]:
        """Parse and validate the JSON response returned by the LLM."""

        raw_response = response.strip()
        if raw_response.startswith("```"):
            raw_response = raw_response.strip("`")
            if raw_response.startswith("json"):
                raw_response = raw_response[4:]
            raw_response = raw_response.strip()

        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise RerankerFallbackError("llm reranker failed: response must be valid JSON") from exc

        ranked_ids = payload.get("ranked_ids")
        if not isinstance(ranked_ids, list) or not ranked_ids or not all(isinstance(item, str) for item in ranked_ids):
            raise RerankerFallbackError("llm reranker failed: response must contain a non-empty 'ranked_ids' string list")

        valid_ids = {candidate.id for candidate in candidates}
        duplicates = {item for item in ranked_ids if ranked_ids.count(item) > 1}
        if duplicates:
            duplicate_text = ", ".join(sorted(duplicates))
            raise RerankerFallbackError(f"llm reranker failed: duplicate candidate ids returned: {duplicate_text}")

        unknown_ids = [item for item in ranked_ids if item not in valid_ids]
        if unknown_ids:
            unknown_text = ", ".join(unknown_ids)
            raise RerankerFallbackError(f"llm reranker failed: unknown candidate ids returned: {unknown_text}")

        return ranked_ids

    @staticmethod
    def _order_candidates(candidates: list[RerankCandidate], ranked_ids: list[str]) -> list[RerankCandidate]:
        """Apply the ranked-id order and append omitted candidates in original order."""

        by_id = {candidate.id: candidate for candidate in candidates}
        ranked = [by_id[candidate_id] for candidate_id in ranked_ids]
        remaining = [candidate for candidate in candidates if candidate.id not in set(ranked_ids)]
        return ranked + remaining
