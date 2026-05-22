"""PDF loader implementation for ingestion pipeline."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from modular_rag.core.types import IMAGE_PLACEHOLDER_TEMPLATE, Document
from modular_rag.libs.loader.base_loader import BaseLoader
from modular_rag.observability.logger import get_logger

try:
	from pypdf import PdfReader
except Exception:  # pragma: no cover - guarded at runtime for optional dependency states.
	PdfReader = None  # type: ignore[assignment]


class PdfLoader(BaseLoader):
	"""Load PDF files into normalized ``Document`` objects."""

	def __init__(self, image_root: str | Path = "data/images") -> None:
		self.image_root = Path(image_root)
		self.logger = get_logger(__name__)

	def load(self, path: str) -> Document:
		"""Load a PDF file, extracting text and image placeholders."""
		if PdfReader is None:
			raise RuntimeError("pypdf is required for PdfLoader. Install dependency: pypdf>=5,<6")

		pdf_path = Path(path)
		if not pdf_path.exists():
			raise FileNotFoundError(f"PDF file not found: {pdf_path}")

		source_path = str(pdf_path)
		doc_hash = self._compute_sha256(pdf_path)

		text_chunks: list[str] = []
		images_metadata: list[dict[str, Any]] = []
		text_length = 0

		def append_text(segment: str) -> None:
			nonlocal text_length
			text_chunks.append(segment)
			text_length += len(segment)

		reader = PdfReader(str(pdf_path))
		for page_num, page in enumerate(reader.pages, start=1):
			page_text = self._extract_page_text(page)
			if page_text:
				if text_length > 0:
					append_text("\n\n")
				append_text(page_text)

			for image_meta in self._extract_page_images(page=page, page_num=page_num, doc_hash=doc_hash):
				if text_length > 0:
					append_text("\n")
				placeholder = IMAGE_PLACEHOLDER_TEMPLATE.format(image_id=image_meta["id"])
				offset = text_length
				append_text(placeholder)
				image_meta["text_offset"] = offset
				image_meta["text_length"] = len(placeholder)
				images_metadata.append(image_meta)

		metadata: dict[str, Any] = {
			"source_path": source_path,
			"doc_type": "pdf",
			"title": pdf_path.stem,
		}
		if images_metadata:
			metadata["images"] = images_metadata

		return Document(id=doc_hash, text="".join(text_chunks), metadata=metadata)

	@staticmethod
	def _compute_sha256(path: Path) -> str:
		digest = hashlib.sha256()
		with path.open("rb") as fh:
			for chunk in iter(lambda: fh.read(65536), b""):
				digest.update(chunk)
		return digest.hexdigest()

	@staticmethod
	def _extract_page_text(page: Any) -> str:
		text = page.extract_text() or ""
		return text.strip()

	def _extract_page_images(self, *, page: Any, page_num: int, doc_hash: str) -> list[dict[str, Any]]:
		images: list[dict[str, Any]] = []
		try:
			page_images = list(page.images)
		except Exception as exc:
			self.logger.warning("Failed to extract images on page %s: %s", page_num, exc)
			return images

		output_dir = self.image_root / doc_hash
		output_dir.mkdir(parents=True, exist_ok=True)

		for index, image in enumerate(page_images):
			image_id = f"{doc_hash}_{page_num}_{index}"
			suffix = self._resolve_image_suffix(image)
			output_path = output_dir / f"{image_id}{suffix}"
			try:
				data = self._extract_image_bytes(image)
				output_path.write_bytes(data)
				images.append(
					{
						"id": image_id,
						"path": str(output_path),
						"page": page_num,
						"position": {},
					}
				)
			except Exception as exc:
				self.logger.warning("Failed to persist image on page %s: %s", page_num, exc)

		return images

	@staticmethod
	def _resolve_image_suffix(image: Any) -> str:
		name = str(getattr(image, "name", ""))
		suffix = Path(name).suffix.lower()
		if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}:
			return suffix
		return ".png"

	@staticmethod
	def _extract_image_bytes(image: Any) -> bytes:
		data = getattr(image, "data", None)
		if isinstance(data, bytes):
			return data
		if isinstance(data, bytearray):
			return bytes(data)
		raise ValueError("Unsupported image payload from PDF parser")

