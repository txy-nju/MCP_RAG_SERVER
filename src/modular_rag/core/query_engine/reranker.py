"""Core reranker orchestration with fallback handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modular_rag.core.settings import Settings
from modular_rag.core.types import RetrievalResult
from modular_rag.libs.reranker.base_reranker import RerankCandidate, RerankerFallbackError
from modular_rag.libs.reranker.reranker_factory import RerankerFactory

if TYPE_CHECKING:
	from modular_rag.core.trace.trace_context import TraceContext


@dataclass(slots=True)
class RerankResult:
	"""Reranking result with fallback flag."""

	candidates: list[RetrievalResult]
	fallback: bool = False
	fallback_reason: str | None = None


@dataclass(slots=True)
class Reranker:
	"""Orchestrate reranking backends with graceful fallback to original ranking."""

	settings: Settings
	timeout_seconds: float = 5.0

	def rerank(
		self,
		query: str,
		candidates: list[RetrievalResult],
		trace: TraceContext | None = None,
	) -> RerankResult:
		"""Rerank candidates with fallback to original order on error."""

		if not candidates:
			if trace is not None:
				trace.record_stage(
					"rerank",
					{
						"method": "skip",
						"provider": "none",
						"details": {"candidate_count": 0, "result_count": 0, "fallback": False},
					},
				)
			return RerankResult(candidates=[])
		if not query or not str(query).strip():
			if trace is not None:
				trace.record_stage(
					"rerank",
					{
						"method": "skip",
						"provider": "none",
						"details": {
							"candidate_count": len(candidates),
							"result_count": len(candidates),
							"fallback": False,
						},
					},
				)
			return RerankResult(candidates=list(candidates))

		backend = RerankerFactory.create(self.settings)

		try:
			rerank_candidates = [
				RerankCandidate(
					id=str(candidate.chunk_id),
					score=float(candidate.score),
					text=str(candidate.text or ""),
					metadata=dict(candidate.metadata or {}),
				)
				for candidate in candidates
			]

			reranked = backend.rerank(str(query).strip(), rerank_candidates, trace=trace)
			
			# Build id_to_original mapping for metadata preservation
			id_to_original = {c.chunk_id: c for c in candidates}
			
			result_candidates = []
			for item in reranked:
				original = id_to_original.get(item.id)
				if original:
					result_candidates.append(
						RetrievalResult(
							chunk_id=item.id,
							score=float(item.score),
							text=item.text or original.text or "",
							metadata=dict(original.metadata or {}),
						)
					)
			
			if trace is not None:
				trace.record_stage(
					"rerank",
					{
						"method": "reranker",
						"provider": str(getattr(backend, "provider", "unknown")),
						"details": {
							"candidate_count": len(candidates),
							"result_count": len(result_candidates),
							"fallback": False,
						},
					},
				)
			return RerankResult(candidates=result_candidates, fallback=False)

		except RerankerFallbackError as exc:
			reason = f"reranker {backend.provider} failed: {str(exc)}"
			if trace is not None:
				trace.record_stage(
					"rerank",
					{
						"method": "reranker",
						"provider": str(getattr(backend, "provider", "unknown")),
						"details": {
							"candidate_count": len(candidates),
							"result_count": len(candidates),
							"fallback": True,
							"reason": reason,
						},
					},
				)
			return RerankResult(candidates=list(candidates), fallback=True, fallback_reason=reason)
		except TimeoutError as exc:
			reason = f"reranker timeout after {self.timeout_seconds} seconds"
			if trace is not None:
				trace.record_stage(
					"rerank",
					{
						"method": "reranker",
						"provider": str(getattr(backend, "provider", "unknown")),
						"details": {
							"candidate_count": len(candidates),
							"result_count": len(candidates),
							"fallback": True,
							"reason": reason,
						},
					},
				)
			return RerankResult(candidates=list(candidates), fallback=True, fallback_reason=reason)
		except Exception as exc:
			reason = f"reranker unexpected error: {str(exc)}"
			if trace is not None:
				trace.record_stage(
					"rerank",
					{
						"method": "reranker",
						"provider": str(getattr(backend, "provider", "unknown")),
						"details": {
							"candidate_count": len(candidates),
							"result_count": len(candidates),
							"fallback": True,
							"reason": reason,
						},
					},
				)
			return RerankResult(candidates=list(candidates), fallback=True, fallback_reason=reason)


__all__ = ["Reranker", "RerankResult"]
