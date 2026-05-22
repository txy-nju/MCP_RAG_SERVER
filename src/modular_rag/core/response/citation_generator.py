"""Citation generation helpers for MCP structured responses."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from modular_rag.core.types import RetrievalResult


@dataclass(slots=True)
class CitationGenerator:
	"""Generate normalized citation payloads from retrieval results."""

	def generate(self, retrieval_results: list[RetrievalResult]) -> list[dict[str, Any]]:
		"""Build a stable citation list used in structuredContent.citations."""

		citations: list[dict[str, Any]] = []
		for index, result in enumerate(retrieval_results, start=1):
			metadata = dict(result.metadata or {})
			page = metadata.get("page")
			citation = {
				"index": index,
				"source": str(metadata.get("source_path", "unknown")),
				"page": int(page) if isinstance(page, int | float | str) and str(page).strip().isdigit() else page,
				"chunk_id": str(result.chunk_id),
				"score": float(round(float(result.score), 6)),
			}
			citations.append(citation)

		return citations


__all__ = ["CitationGenerator"]
