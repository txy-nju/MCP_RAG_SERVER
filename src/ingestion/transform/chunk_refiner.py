"""Chunk refiner implementation with rule-based cleanup and optional LLM refinement."""

from __future__ import annotations

import re
from pathlib import Path

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk
from ingestion.transform.base_transform import BaseTransform
from libs.llm.base_llm import BaseLLM
from libs.llm.llm_factory import LLMFactory


_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_HORIZONTAL_RULE_RE = re.compile(r"^\s*([-*_])\1{2,}\s*$")
_PAGE_FOOTER_RE = re.compile(r"^\s*page\s+\d+(?:\s*/\s*\d+)?\s*$", re.IGNORECASE)
_CODE_BLOCK_RE = re.compile(r"(```[\s\S]*?```)", re.MULTILINE)


class ChunkRefiner(BaseTransform):
	"""Refine chunk text using deterministic cleanup plus optional LLM post-processing."""

	def __init__(
		self,
		settings: Settings,
		llm: BaseLLM | None = None,
		prompt_path: str | None = None,
	) -> None:
		self._settings = settings
		self._use_llm = bool(getattr(settings.ingestion.chunk_refiner, "use_llm", False))
		self._prompt = self._load_prompt(prompt_path)
		self._llm = llm
		if self._use_llm and self._llm is None:
			self._llm = LLMFactory.create(settings)

	def transform(self, chunks: list[Chunk], trace: TraceContext | None = None) -> list[Chunk]:
		"""Refine chunks in-order, falling back to rule-only output on LLM failures."""

		refined_chunks: list[Chunk] = []
		for chunk in chunks:
			metadata = dict(chunk.metadata)
			try:
				rule_text = self._rule_based_refine(chunk.text)
				refined_text = rule_text
				if self._use_llm:
					llm_text, llm_error = self._llm_refine(rule_text, trace)
					if llm_text:
						refined_text = llm_text
						metadata["refined_by"] = "llm"
					else:
						metadata["refined_by"] = "rule"
						if llm_error:
							metadata["fallback_reason"] = llm_error
				else:
					metadata["refined_by"] = "rule"

				refined_chunks.append(
					Chunk(
						id=chunk.id,
						text=refined_text,
						metadata=metadata,
						start_offset=0,
						end_offset=len(refined_text),
						source_ref=chunk.source_ref,
					)
				)
			except Exception as exc:  # pragma: no cover - defensive isolation per chunk
				metadata["refined_by"] = "none"
				metadata["refine_error"] = str(exc)
				refined_chunks.append(
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
						"chunk_refiner.chunk_error",
						{"chunk_id": chunk.id, "error": str(exc)},
					)

		return refined_chunks

	def _rule_based_refine(self, text: str) -> str:
		"""Apply deterministic denoising while preserving markdown and code blocks."""

		if not text:
			return ""

		parts = _CODE_BLOCK_RE.split(text)
		processed: list[str] = []
		for part in parts:
			if part.startswith("```") and part.endswith("```"):
				processed.append(part)
				continue
			processed.append(self._refine_non_code_text(part))

		merged = "".join(processed)
		merged = re.sub(r"[ \t]{2,}", " ", merged)
		merged = re.sub(r"\n{3,}", "\n\n", merged)
		return merged.strip()

	def _llm_refine(self, text: str, trace: TraceContext | None = None) -> tuple[str | None, str | None]:
		"""Run optional LLM rewrite; returns ``(refined_text, error)``."""

		if not self._use_llm or self._llm is None:
			return None, None

		try:
			messages = [
				{"role": "system", "content": self._prompt},
				{"role": "user", "content": text},
			]
			refined = self._llm.chat(messages).strip()
			if trace is not None:
				trace.record_stage(
					"chunk_refiner.llm",
					{
						"input_chars": len(text),
						"output_chars": len(refined),
					},
				)
			return (refined or None), None
		except Exception as exc:
			if trace is not None:
				trace.record_stage(
					"chunk_refiner.llm_fallback",
					{"error": str(exc)},
				)
			return None, str(exc)

	def _load_prompt(self, prompt_path: str | None) -> str:
		"""Load prompt from file and guarantee ``{text}`` placeholder presence."""

		configured_path = prompt_path or getattr(
			self._settings.ingestion.chunk_refiner,
			"prompt_path",
			"config/prompts/chunk_refinement.txt",
		)

		prompt_file = Path(configured_path)
		if not prompt_file.exists():
			return "Refine the following text for retrieval quality without changing facts:\n{text}"

		prompt = prompt_file.read_text(encoding="utf-8").strip()
		if "{text}" not in prompt:
			prompt = f"{prompt}\n\n{{text}}"
		return prompt

	@staticmethod
	def _refine_non_code_text(text: str) -> str:
		"""Cleanup logic for non-code fragments."""

		text = _HTML_COMMENT_RE.sub("", text)
		lines = text.splitlines()
		cleaned_lines: list[str] = []
		for line in lines:
			if _HORIZONTAL_RULE_RE.match(line):
				continue
			if _PAGE_FOOTER_RE.match(line):
				continue
			cleaned_lines.append(line.rstrip())
		return "\n".join(cleaned_lines)
