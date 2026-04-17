"""Build MCP tool responses from retrieval results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.response.citation_generator import CitationGenerator
from core.types import RetrievalResult


@dataclass(slots=True)
class ResponseBuilder:
	"""Construct MCP-compliant response payloads with markdown and citations."""

	citation_generator: CitationGenerator = field(default_factory=CitationGenerator)

	def build(
		self,
		retrieval_results: list[RetrievalResult],
		query: str,
		fallback_reason: str | None = None,
	) -> dict[str, Any]:
		"""Return `content` + `structuredContent` for MCP tools/call responses."""

		normalized_query = str(query or "").strip()
		citations = self.citation_generator.generate(retrieval_results)

		if not retrieval_results:
			text = (
				f"未检索到与问题相关的内容：\"{normalized_query}\"。\n\n"
				"请先执行数据摄取，或尝试更具体的关键词后重试。"
			)
			return {
				"content": [{"type": "text", "text": text}],
				"structuredContent": {
					"query": normalized_query,
					"citations": [],
					"results": [],
				},
			}

		markdown_lines = [f"### 查询结果\n\n问题：{normalized_query}\n"]
		structured_results: list[dict[str, Any]] = []

		for citation, result in zip(citations, retrieval_results, strict=False):
			snippet = self._summarize_text(result.text)
			source = citation.get("source", "unknown")
			page = citation.get("page")
			page_suffix = f" (page {page})" if page not in (None, "") else ""
			markdown_lines.append(f"- {snippet} [{citation['index']}]\\n  来源：{source}{page_suffix}")
			structured_results.append(
				{
					"chunk_id": result.chunk_id,
					"score": float(result.score),
					"text": result.text,
					"metadata": dict(result.metadata or {}),
				}
			)

		if fallback_reason:
			markdown_lines.append(f"\n> 注：已触发重排回退，原因：{fallback_reason}")

		structured_content: dict[str, Any] = {
			"query": normalized_query,
			"citations": citations,
			"results": structured_results,
		}
		if fallback_reason:
			structured_content["reranker_fallback_reason"] = fallback_reason

		return {
			"content": [{"type": "text", "text": "\n".join(markdown_lines)}],
			"structuredContent": structured_content,
		}

	@staticmethod
	def _summarize_text(text: str, max_len: int = 180) -> str:
		compact = " ".join(str(text or "").split())
		if len(compact) <= max_len:
			return compact
		return compact[:max_len].rstrip() + "..."


__all__ = ["ResponseBuilder"]
