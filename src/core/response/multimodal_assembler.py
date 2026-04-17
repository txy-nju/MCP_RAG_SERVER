"""Assemble MCP multimodal content (text + image) from retrieval results."""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.types import RetrievalResult
from ingestion.storage.image_storage import ImageStorage


@dataclass(slots=True)
class MultimodalAssembler:
	"""Build image payloads for MCP content from chunk metadata image refs."""

	image_storage: ImageStorage | None = None
	max_images: int = 4

	def __post_init__(self) -> None:
		if self.image_storage is None:
			self.image_storage = ImageStorage()

	def assemble(self, retrieval_results: list[RetrievalResult]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
		"""Return MCP image contents and structured image metadata."""

		image_contents: list[dict[str, Any]] = []
		structured_images: list[dict[str, Any]] = []
		seen: set[str] = set()

		for result in retrieval_results:
			if len(image_contents) >= self.max_images:
				break
			metadata = dict(result.metadata or {})
			refs = metadata.get("image_refs")
			if not isinstance(refs, list):
				continue

			images = metadata.get("images")
			image_by_id: dict[str, dict[str, Any]] = {}
			if isinstance(images, list):
				for image in images:
					if isinstance(image, dict):
						image_id = str(image.get("id", "")).strip()
						if image_id:
							image_by_id[image_id] = image

			for ref in refs:
				if len(image_contents) >= self.max_images:
					break
				image_id = str(ref).strip()
				if not image_id or image_id in seen:
					continue

				candidate = image_by_id.get(image_id)
				image_path = ""
				if candidate is not None:
					image_path = str(candidate.get("path", "")).strip()
				if not image_path and self.image_storage is not None:
					resolved = self.image_storage.get_image_path(image_id)
					image_path = str(resolved or "").strip()
				if not image_path:
					continue

				path_obj = Path(image_path)
				if not path_obj.exists() or not path_obj.is_file():
					continue

				try:
					raw = path_obj.read_bytes()
				except OSError:
					continue
				if not raw:
					continue

				mime_type, _ = mimetypes.guess_type(path_obj.name)
				if not mime_type:
					mime_type = "application/octet-stream"

				image_contents.append(
					{
						"type": "image",
						"mimeType": mime_type,
						"data": base64.b64encode(raw).decode("ascii"),
					}
				)
				structured_images.append(
					{
						"image_id": image_id,
						"mime_type": mime_type,
						"path": str(path_obj),
						"chunk_id": result.chunk_id,
					}
				)
				seen.add(image_id)

		return image_contents, structured_images


__all__ = ["MultimodalAssembler"]
