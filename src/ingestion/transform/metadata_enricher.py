"""Metadata enrichment transform with rule-based and optional LLM strategies."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory


DEFAULT_METADATA_PROMPT = (
	"You are a metadata extraction assistant for retrieval systems. "
	"Read the chunk and return strict JSON with keys title, summary, tags. "
	"- title: concise and specific\n"
	"- summary: 1-2 sentences\n"
	"- tags: list of 3-5 short topic labels\n"
	"Return JSON only.\n\n"
	"Chunk:\n{text}"
)

_WORD_RE = re.compile(r"[A-Za-z]{3,}")
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


class MetadataEnricher(BaseTransform):
	"""Inject title/summary/tags metadata for chunks with optional LLM enrichment."""

	def __init__(
		self,
		settings: Settings,
		llm: BaseLLM | None = None,
		prompt_path: str | None = None,
	) -> None:
		self._settings = settings
		self._config = getattr(settings.ingestion, "metadata_enricher", None)
		self._use_llm = bool(getattr(self._config, "use_llm", False))
		self._max_tags = int(getattr(self._config, "max_tags", 5))
		self._prompt = self._load_prompt(prompt_path)
		self._llm = llm
		if self._use_llm and self._llm is None:
			self._llm = LLMFactory.create(settings)

	def transform(self, chunks: list[Chunk], trace: TraceContext | None = None) -> list[Chunk]:
		"""Enrich chunk metadata while preserving ordering and source linkage."""

		enriched_chunks: list[Chunk] = []
		for chunk in chunks:
			metadata = dict(chunk.metadata)
			try:
				rule_meta = self._rule_enrich(chunk.text)
				metadata.update(rule_meta)
				metadata["enriched_by"] = "rule"

				if self._use_llm:
					llm_meta, llm_error = self._llm_enrich(chunk.text, trace)
					if llm_meta is not None:
						metadata.update(llm_meta)
						metadata["enriched_by"] = "llm"
					else:
						metadata["enriched_by"] = "rule"
						if llm_error:
							metadata["fallback_reason"] = llm_error

				enriched_chunks.append(
					Chunk(
						id=chunk.id,
						text=chunk.text,
						metadata=metadata,
						start_offset=chunk.start_offset,
						end_offset=chunk.end_offset,
						source_ref=chunk.source_ref,
					)
				)
			except Exception as exc:  # pragma: no cover - defensive safety per chunk
				metadata.setdefault("title", "Untitled chunk")
				metadata.setdefault("summary", "Metadata enrichment failed for this chunk.")
				metadata.setdefault("tags", ["fallback"])
				metadata["enriched_by"] = "rule"
				metadata["enrich_error"] = str(exc)
				enriched_chunks.append(
					Chunk(
						id=chunk.id,
						text=chunk.text,
						metadata=metadata,
						start_offset=chunk.start_offset,
						end_offset=chunk.end_offset,
						source_ref=chunk.source_ref,
					)
				)
				if trace is not None:
					trace.record_stage(
						"metadata_enricher.chunk_error",
						{"chunk_id": chunk.id, "error": str(exc)},
					)

		return enriched_chunks

	def _rule_enrich(self, text: str) -> dict[str, Any]:
		clean_text = (text or "").strip()
		if not clean_text:
			return {
				"title": "Untitled chunk",
				"summary": "No textual content available in this chunk.",
				"tags": ["empty"],
			}

		title = self._derive_title(clean_text)
		summary = self._derive_summary(clean_text)
		tags = self._derive_tags(clean_text)
		return {
			"title": title,
			"summary": summary,
			"tags": tags,
		}

	def _llm_enrich(
		self,
		text: str,
		trace: TraceContext | None = None,
	) -> tuple[dict[str, Any] | None, str | None]:
		"""Run optional LLM extraction. Returns (metadata, error)."""

		if not self._use_llm or self._llm is None:
			return None, None

		try:
			messages = [
				{"role": "system", "content": self._prompt},
				{"role": "user", "content": text},
			]
			raw = self._llm.chat(messages).strip()
			payload = self._parse_llm_payload(raw)
			if trace is not None:
				trace.record_stage(
					"metadata_enricher.llm",
					{
						"input_chars": len(text),
						"output_chars": len(raw),
					},
				)
			return payload, None
		except Exception as exc:
			if trace is not None:
				trace.record_stage(
					"metadata_enricher.llm_fallback",
					{"error": str(exc)},
				)
			return None, str(exc)

	def _parse_llm_payload(self, raw: str) -> dict[str, Any]:
		candidate = _FENCE_RE.sub("", raw).strip()
		if not candidate:
			raise ValueError("LLM returned empty metadata payload")

		data = json.loads(candidate)
		if not isinstance(data, dict):
			raise ValueError("LLM metadata payload must be a JSON object")

		title = str(data.get("title", "")).strip()
		summary = str(data.get("summary", "")).strip()
		tags = self._coerce_tags(data.get("tags"))

		if not title or not summary or not tags:
			raise ValueError("LLM metadata payload missing title/summary/tags")

		return {
			"title": title,
			"summary": summary,
			"tags": tags,
		}

	def _load_prompt(self, prompt_path: str | None) -> str:
		configured_path = prompt_path
		if configured_path is None and self._config is not None:
			configured_path = getattr(self._config, "prompt_path", None)
		configured_path = configured_path or "config/prompts/metadata_enrichment.txt"

		prompt_file = Path(configured_path)
		if not prompt_file.exists():
			return DEFAULT_METADATA_PROMPT

		prompt = prompt_file.read_text(encoding="utf-8").strip()
		if "{text}" not in prompt:
			prompt = f"{prompt}\n\n{{text}}"
		return prompt

	def _derive_title(self, text: str) -> str:
		for line in text.splitlines():
			candidate = line.strip().strip("#")
			if candidate:
				return candidate[:80]
		return text[:80]

	def _derive_summary(self, text: str) -> str:
		one_line = " ".join(part.strip() for part in text.splitlines() if part.strip())
		if len(one_line) <= 220:
			return one_line
		return f"{one_line[:217]}..."

	def _derive_tags(self, text: str) -> list[str]:
		counts: dict[str, int] = {}
		for token in _WORD_RE.findall(text.lower()):
			counts[token] = counts.get(token, 0) + 1

		sorted_tokens = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
		tags = [token for token, _ in sorted_tokens[: self._max_tags]]
		if not tags:
			tags = ["general"]
		return tags

	def _coerce_tags(self, raw_tags: Any) -> list[str]:
		if isinstance(raw_tags, str):
			items = [item.strip() for item in raw_tags.split(",") if item.strip()]
			return items[: self._max_tags]
		if isinstance(raw_tags, list):
			items = [str(item).strip() for item in raw_tags if str(item).strip()]
			return items[: self._max_tags]
		return []
