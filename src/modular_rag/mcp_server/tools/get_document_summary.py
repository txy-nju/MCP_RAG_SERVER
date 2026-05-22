"""MCP tool: get a document summary by doc_id."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
	from pypdf import PdfReader
except Exception:  # pragma: no cover - optional dependency state
	PdfReader = None  # type: ignore[assignment]


_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")


@dataclass(slots=True)
class DocumentSummary:
	"""Structured summary payload returned by get_document_summary."""

	doc_id: str
	title: str
	summary: str
	tags: list[str]
	source_path: str
	collection: str

	def to_dict(self) -> dict[str, Any]:
		return {
			"doc_id": self.doc_id,
			"title": self.title,
			"summary": self.summary,
			"tags": list(self.tags),
			"source_path": self.source_path,
			"collection": self.collection,
		}


def _find_document_path(doc_id: str, documents_root: Path) -> Path:
	if not documents_root.exists() or not documents_root.is_dir():
		raise ValueError(f"doc_id not found: {doc_id}")

	normalized_doc_id = doc_id.strip()
	if not normalized_doc_id:
		raise ValueError("doc_id must be a non-empty string")

	matches: list[Path] = []
	for candidate in documents_root.rglob("*"):
		if not candidate.is_file() or candidate.name.startswith("."):
			continue
		relative = candidate.relative_to(documents_root).as_posix()
		if (
			candidate.name == normalized_doc_id
			or candidate.stem == normalized_doc_id
			or relative == normalized_doc_id
		):
			matches.append(candidate)

	if not matches:
		raise ValueError(f"doc_id not found: {doc_id}")
	if len(matches) > 1:
		raise ValueError(f"doc_id is ambiguous: {doc_id}")
	return matches[0]


def _extract_pdf_text(path: Path) -> str:
	if PdfReader is None:
		return ""
	try:
		reader = PdfReader(str(path))
	except Exception:
		return ""

	pages: list[str] = []
	for page in reader.pages:
		page_text = (page.extract_text() or "").strip()
		if page_text:
			pages.append(page_text)
	return "\n".join(pages)


def _extract_document_text(path: Path) -> str:
	if path.suffix.lower() == ".pdf":
		pdf_text = _extract_pdf_text(path)
		if pdf_text:
			return pdf_text

	try:
		return path.read_text(encoding="utf-8")
	except Exception:
		return ""


def _normalize_whitespace(text: str) -> str:
	return " ".join(str(text).split())


def _extract_title(text: str, fallback_name: str) -> str:
	for line in str(text).splitlines():
		stripped = line.strip()
		if not stripped:
			continue
		if stripped.startswith("#"):
			return stripped.lstrip("#").strip() or fallback_name
		return stripped[:120]
	return fallback_name


def _build_summary(text: str, *, max_length: int = 280) -> str:
	normalized = _normalize_whitespace(text)
	if not normalized:
		return "No textual content extracted from this document."
	if len(normalized) <= max_length:
		return normalized
	return normalized[:max_length].rstrip() + "..."


def _build_tags(text: str, suffix: str, collection: str, *, max_tags: int = 5) -> list[str]:
	tags: list[str] = []
	if suffix:
		tags.append(suffix.lstrip(".").lower())
	if collection:
		tags.append(collection.lower())

	for token in _TOKEN_PATTERN.findall(text):
		normalized = token.lower()
		if normalized not in tags:
			tags.append(normalized)
		if len(tags) >= max_tags:
			break

	return tags[:max_tags] if tags else ["document"]


def get_document_summary(doc_id: str, documents_root: str | Path = "data/documents") -> DocumentSummary:
	"""Resolve doc_id and return structured title/summary/tags info."""

	root = Path(documents_root)
	path = _find_document_path(doc_id=doc_id, documents_root=root)
	text = _extract_document_text(path)

	try:
		relative = path.relative_to(root)
	except ValueError:
		relative = path

	parts = relative.parts
	collection = str(parts[0]) if len(parts) >= 2 else "default"
	title = _extract_title(text=text, fallback_name=path.stem)
	summary = _build_summary(text=text)
	tags = _build_tags(text=text, suffix=path.suffix, collection=collection)

	return DocumentSummary(
		doc_id=path.name,
		title=title,
		summary=summary,
		tags=tags,
		source_path=str(path),
		collection=collection,
	)


def build_get_document_summary_tool_handler(documents_root: str | Path = "data/documents") -> Any:
	"""Create a protocol-compatible handler for get_document_summary."""

	def _handler(arguments: dict[str, Any]) -> dict[str, Any]:
		doc_id = arguments.get("doc_id")
		if not isinstance(doc_id, str) or not doc_id.strip():
			raise ValueError("doc_id must be a non-empty string")

		summary = get_document_summary(doc_id=doc_id, documents_root=documents_root)
		payload = summary.to_dict()

		text = (
			f"# {summary.title}\n\n"
			f"- doc_id: {summary.doc_id}\n"
			f"- collection: {summary.collection}\n"
			f"- tags: {', '.join(summary.tags)}\n\n"
			f"{summary.summary}"
		)
		return {
			"content": [{"type": "text", "text": text}],
			"structuredContent": payload,
		}

	return _handler


__all__ = ["DocumentSummary", "build_get_document_summary_tool_handler", "get_document_summary"]
