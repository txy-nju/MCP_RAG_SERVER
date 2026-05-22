"""Reciprocal Rank Fusion (RRF) utilities for hybrid retrieval."""

from __future__ import annotations

from dataclasses import dataclass

from modular_rag.core.types import RetrievalResult


@dataclass(slots=True)
class RRFFuser:
	"""Fuse dense and sparse retrieval rankings with Reciprocal Rank Fusion."""

	rrf_k: int = 60

	def fuse(
		self,
		dense_results: list[RetrievalResult],
		sparse_results: list[RetrievalResult],
		top_k: int,
	) -> list[RetrievalResult]:
		"""Return top-k fused results ranked by RRF score."""

		if top_k <= 0:
			raise ValueError("top_k must be greater than 0")
		if self.rrf_k <= 0:
			raise ValueError("rrf_k must be greater than 0")

		if not dense_results and not sparse_results:
			return []

		combined: dict[str, dict[str, object]] = {}
		self._accumulate(combined, dense_results)
		self._accumulate(combined, sparse_results)

		ranked = sorted(
			combined.values(),
			key=lambda item: (
				-float(item["rrf_score"]),
				int(item["best_rank"]),
				str(item["chunk_id"]),
			),
		)

		return [
			RetrievalResult(
				chunk_id=str(item["chunk_id"]),
				score=float(item["rrf_score"]),
				text=str(item["text"]),
				metadata=dict(item["metadata"]),
			)
			for item in ranked[:top_k]
		]

	def _accumulate(
		self,
		combined: dict[str, dict[str, object]],
		results: list[RetrievalResult],
	) -> None:
		for rank, result in enumerate(results, start=1):
			chunk_id = result.chunk_id
			rrf_contribution = 1.0 / (self.rrf_k + rank)
			entry = combined.get(chunk_id)

			if entry is None:
				combined[chunk_id] = {
					"chunk_id": chunk_id,
					"rrf_score": rrf_contribution,
					"best_rank": rank,
					"text": result.text,
					"metadata": dict(result.metadata),
				}
				continue

			entry["rrf_score"] = float(entry["rrf_score"]) + rrf_contribution
			entry["best_rank"] = min(int(entry["best_rank"]), rank)
			if not str(entry["text"]):
				entry["text"] = result.text
			if result.metadata:
				merged_metadata = dict(entry["metadata"])
				merged_metadata.update(result.metadata)
				entry["metadata"] = merged_metadata


__all__ = ["RRFFuser"]
