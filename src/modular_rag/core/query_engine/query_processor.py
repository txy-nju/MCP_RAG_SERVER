"""Query preprocessing with rule-based keyword extraction and filter parsing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from modular_rag.core.types import ProcessedQuery


@dataclass(slots=True)
class QueryProcessor:
	"""Convert a raw query string to normalized keywords and filters."""

	stopwords: set[str] = field(
		default_factory=lambda: {
			"a",
			"an",
			"and",
			"are",
			"as",
			"at",
			"be",
			"by",
			"for",
			"from",
			"how",
			"in",
			"is",
			"it",
			"of",
			"on",
			"or",
			"that",
			"the",
			"to",
			"what",
			"when",
			"where",
			"which",
			"with",
			"了",
			"和",
			"在",
			"是",
			"的",
			"请",
			"帮我",
		}
	)

	_FILTER_PATTERN = re.compile(
		r"(?P<full>(?P<key>collection|doc_type|language|source|access_level)\s*:\s*(?P<value>[^\s,;]+))",
		re.IGNORECASE,
	)
	_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]{2,}")

	def process(self, query: str, filters: dict[str, Any] | None = None) -> ProcessedQuery:
		"""Return normalized query payload for downstream retrieval stages."""

		normalized_query = str(query or "").strip()
		parsed_filters, query_without_filters = self._extract_inline_filters(normalized_query)

		merged_filters: dict[str, Any] = dict(parsed_filters)
		if filters:
			merged_filters.update(dict(filters))

		keywords = self._extract_keywords(query_without_filters)
		if not keywords and query_without_filters.strip():
			keywords = [query_without_filters.strip()]

		return ProcessedQuery(raw_query=normalized_query, keywords=keywords, filters=merged_filters)

	def _extract_inline_filters(self, query: str) -> tuple[dict[str, Any], str]:
		parsed_filters: dict[str, Any] = {}

		def _replace(match: re.Match[str]) -> str:
			key = match.group("key").lower()
			value = match.group("value").strip()
			if value:
				parsed_filters[key] = value
			return " "

		stripped_query = self._FILTER_PATTERN.sub(_replace, query)
		stripped_query = re.sub(r"\s+", " ", stripped_query).strip()
		return parsed_filters, stripped_query

	def _extract_keywords(self, query: str) -> list[str]:
		if not query:
			return []

		keywords: list[str] = []
		seen: set[str] = set()

		for token in self._TOKEN_PATTERN.findall(query):
			normalized = token.lower()
			if normalized in self.stopwords:
				continue
			if normalized in seen:
				continue
			seen.add(normalized)
			keywords.append(normalized)
		return keywords


__all__ = ["QueryProcessor"]
