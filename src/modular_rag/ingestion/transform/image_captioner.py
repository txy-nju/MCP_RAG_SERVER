"""Image captioner transform with optional vision LLM and graceful fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from modular_rag.core.settings import Settings
from modular_rag.core.trace.trace_context import TraceContext
from modular_rag.core.types import Chunk
from modular_rag.ingestion.transform.base_transform import BaseTransform
from modular_rag.libs.llm.base_vision_llm import BaseVisionLLM
from modular_rag.libs.llm.llm_factory import LLMFactory


DEFAULT_CAPTION_PROMPT = (
	"Describe the image in concise factual language for retrieval. "
	"Focus on entities, chart trends, labels, and code/UI details.\n"
	"Chunk context:\n{text}\n"
)


class ImageCaptioner(BaseTransform):
	"""Generate image captions for chunk-referenced images without blocking ingestion."""

	def __init__(
		self,
		settings: Settings,
		vision_llm: BaseVisionLLM | None = None,
		prompt_path: str | None = None,
	) -> None:
		self._settings = settings
		self._config = getattr(settings.ingestion, "image_captioner", None)
		self._use_vision_llm = bool(getattr(self._config, "use_vision_llm", False))
		self._prompt = self._load_prompt(prompt_path)
		self._vision_llm = vision_llm
		if self._use_vision_llm and self._vision_llm is None and settings.vision_llm is not None:
			self._vision_llm = LLMFactory.create_vision_llm(settings)

	def transform(self, chunks: list[Chunk], trace: TraceContext | None = None) -> list[Chunk]:
		"""Caption chunk-linked images and preserve chunk order and identity."""

		out: list[Chunk] = []
		for chunk in chunks:
			metadata = dict(chunk.metadata)
			refs = self._extract_image_refs(metadata)
			images = self._extract_images(metadata)

			if not refs:
				out.append(self._clone_chunk(chunk, metadata))
				continue

			if not self._use_vision_llm or self._vision_llm is None:
				metadata["has_unprocessed_images"] = True
				out.append(self._clone_chunk(chunk, metadata))
				continue

			id_to_image = {str(image.get("id", "")): image for image in images}
			captions: dict[str, str] = {}
			has_unprocessed = False

			for image_id in refs:
				image_payload = id_to_image.get(image_id)
				image_path = self._resolve_image_path(image_payload)
				if image_path is None:
					has_unprocessed = True
					continue
				try:
					prompt = self._render_prompt(chunk.text)
					response = self._vision_llm.chat_with_image(prompt, image_path, trace=trace)
					caption = response.content.strip()
					if caption:
						captions[image_id] = caption
					else:
						has_unprocessed = True
				except Exception as exc:
					has_unprocessed = True
					metadata["caption_fallback_reason"] = str(exc)
					if trace is not None:
						trace.record_stage(
							"image_captioner.fallback",
							{"chunk_id": chunk.id, "image_id": image_id, "error": str(exc)},
						)

			if captions:
				metadata["image_captions"] = captions
			if has_unprocessed:
				metadata["has_unprocessed_images"] = True
			elif "has_unprocessed_images" in metadata:
				metadata.pop("has_unprocessed_images")

			out.append(self._clone_chunk(chunk, metadata))

		return out

	def _clone_chunk(self, chunk: Chunk, metadata: dict[str, Any]) -> Chunk:
		return Chunk(
			id=chunk.id,
			text=chunk.text,
			metadata=metadata,
			start_offset=chunk.start_offset,
			end_offset=chunk.end_offset,
			source_ref=chunk.source_ref,
		)

	def _load_prompt(self, prompt_path: str | None) -> str:
		configured_path = prompt_path
		if configured_path is None and self._config is not None:
			configured_path = getattr(self._config, "prompt_path", None)
		configured_path = configured_path or "config/prompts/image_captioning.txt"

		prompt_file = Path(configured_path)
		if not prompt_file.exists():
			return DEFAULT_CAPTION_PROMPT

		prompt = prompt_file.read_text(encoding="utf-8").strip()
		if "{text}" not in prompt:
			prompt = f"{prompt}\n\n{{text}}"
		return prompt

	def _render_prompt(self, chunk_text: str) -> str:
		return self._prompt.format(text=chunk_text)

	@staticmethod
	def _extract_image_refs(metadata: dict[str, Any]) -> list[str]:
		refs = metadata.get("image_refs")
		if not isinstance(refs, list):
			return []
		return [str(ref) for ref in refs if str(ref).strip()]

	@staticmethod
	def _extract_images(metadata: dict[str, Any]) -> list[dict[str, Any]]:
		images = metadata.get("images")
		if not isinstance(images, list):
			return []
		return [img for img in images if isinstance(img, dict)]

	@staticmethod
	def _resolve_image_path(image_payload: dict[str, Any] | None) -> str | None:
		if not image_payload:
			return None
		path = image_payload.get("path")
		if path in (None, ""):
			return None
		return str(path)
