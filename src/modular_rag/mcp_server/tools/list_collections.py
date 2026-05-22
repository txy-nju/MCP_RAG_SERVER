"""MCP tool: list indexed document collections under data/documents."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class CollectionInfo:
	"""Collection metadata returned by list_collections."""

	name: str
	document_count: int
	description: str

	def to_dict(self) -> dict[str, Any]:
		return {
			"name": self.name,
			"description": self.description,
			"document_count": self.document_count,
		}


def list_collections(documents_root: str | Path = "data/documents") -> list[CollectionInfo]:
	"""Discover collection folders and compute per-collection file counts."""

	root = Path(documents_root)
	if not root.exists() or not root.is_dir():
		return []

	collections: list[CollectionInfo] = []
	for collection_dir in root.iterdir():
		if not collection_dir.is_dir() or collection_dir.name.startswith("."):
			continue

		document_count = sum(
			1
			for candidate in collection_dir.rglob("*")
			if candidate.is_file() and not candidate.name.startswith(".")
		)

		collections.append(
			CollectionInfo(
				name=collection_dir.name,
				document_count=document_count,
				description=f"Collection '{collection_dir.name}' with {document_count} document(s).",
			)
		)

	return sorted(collections, key=lambda item: item.name.lower())


def build_list_collections_tool_handler(documents_root: str | Path = "data/documents") -> Any:
	"""Create a protocol-compatible handler for list_collections."""

	def _handler(arguments: dict[str, Any]) -> dict[str, Any]:
		if arguments:
			raise ValueError("list_collections does not accept arguments")

		collections = list_collections(documents_root=documents_root)
		if collections:
			lines = ["Available collections:"]
			for collection in collections:
				lines.append(f"- {collection.name}: {collection.document_count} document(s)")
			text = "\n".join(lines)
		else:
			text = "No collections found under data/documents."

		return {
			"content": [
				{
					"type": "text",
					"text": text,
				}
			],
			"structuredContent": {
				"collections": [collection.to_dict() for collection in collections],
			},
		}

	return _handler


__all__ = ["CollectionInfo", "build_list_collections_tool_handler", "list_collections"]
