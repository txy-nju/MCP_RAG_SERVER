"""Shared core data contracts used across ingestion, retrieval, and MCP layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Metadata = dict[str, Any]
ImageMetadata = dict[str, Any]
SourceRef = dict[str, Any]
SparseVector = dict[str, float]

IMAGE_PLACEHOLDER_TEMPLATE = "[IMAGE: {image_id}]"


def _clone_mapping(value: dict[str, Any] | None) -> dict[str, Any]:
    return dict(value or {})


def _normalize_image_metadata(image: dict[str, Any]) -> ImageMetadata:
    normalized = dict(image)

    for key in ("id", "path", "text_offset", "text_length"):
        if key not in normalized:
            raise ValueError(f"metadata.images[{normalized.get('id', '?')}].{key} is required")

    normalized["id"] = str(normalized["id"])
    normalized["path"] = str(normalized["path"])
    normalized["text_offset"] = int(normalized["text_offset"])
    normalized["text_length"] = int(normalized["text_length"])

    if normalized["text_offset"] < 0:
        raise ValueError("metadata.images[].text_offset must be greater than or equal to 0")
    if normalized["text_length"] < 0:
        raise ValueError("metadata.images[].text_length must be greater than or equal to 0")

    if "page" in normalized and normalized["page"] is not None:
        normalized["page"] = int(normalized["page"])

    position = normalized.get("position")
    if position is None:
        normalized["position"] = {}
    elif not isinstance(position, dict):
        raise ValueError("metadata.images[].position must be a mapping when provided")
    else:
        normalized["position"] = dict(position)

    return normalized


def _normalize_metadata(metadata: Metadata | None) -> Metadata:
    normalized = _clone_mapping(metadata)
    source_path = normalized.get("source_path")
    if not source_path:
        raise ValueError("metadata.source_path is required")

    images = normalized.get("images", [])
    if images is None:
        images = []
    if not isinstance(images, list):
        raise ValueError("metadata.images must be a list when provided")

    normalized["source_path"] = str(source_path)
    normalized["images"] = [_normalize_image_metadata(image) for image in images]
    return normalized


def _normalize_source_ref(source_ref: SourceRef | None) -> SourceRef | None:
    if source_ref is None:
        return None
    return dict(source_ref)


@dataclass(slots=True)
class Document:
    """Canonical document representation produced by loaders."""

    id: str
    text: str
    metadata: Metadata = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.text = str(self.text)
        self.metadata = _normalize_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": _clone_mapping(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Document:
        return cls(
            id=payload["id"],
            text=payload["text"],
            metadata=payload.get("metadata", {}),
        )


@dataclass(slots=True)
class Chunk(Document):
    """A source-grounded slice of a document used during retrieval."""

    start_offset: int = 0
    end_offset: int = 0
    source_ref: SourceRef | None = None

    def __post_init__(self) -> None:
        Document.__post_init__(self)
        self.start_offset = int(self.start_offset)
        self.end_offset = int(self.end_offset)
        self.source_ref = _normalize_source_ref(self.source_ref)

        if self.start_offset < 0:
            raise ValueError("start_offset must be greater than or equal to 0")
        if self.end_offset < self.start_offset:
            raise ValueError("end_offset must be greater than or equal to start_offset")

    def to_dict(self) -> dict[str, Any]:
        payload = Document.to_dict(self)
        payload["start_offset"] = self.start_offset
        payload["end_offset"] = self.end_offset
        payload["source_ref"] = _normalize_source_ref(self.source_ref)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Chunk:
        return cls(
            id=payload["id"],
            text=payload["text"],
            metadata=payload.get("metadata", {}),
            start_offset=payload.get("start_offset", 0),
            end_offset=payload.get("end_offset", 0),
            source_ref=payload.get("source_ref"),
        )


@dataclass(slots=True)
class ChunkRecord:
    """Storage-friendly record that carries chunk content and encoded vectors."""

    id: str
    text: str
    metadata: Metadata
    dense_vector: list[float] | None = None
    sparse_vector: SparseVector | None = None

    def __post_init__(self) -> None:
        self.id = str(self.id)
        self.text = str(self.text)
        self.metadata = _normalize_metadata(self.metadata)

        if self.dense_vector is not None:
            self.dense_vector = [float(value) for value in self.dense_vector]

        if self.sparse_vector is not None:
            self.sparse_vector = {str(term): float(weight) for term, weight in self.sparse_vector.items()}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "metadata": _clone_mapping(self.metadata),
            "dense_vector": None if self.dense_vector is None else list(self.dense_vector),
            "sparse_vector": None if self.sparse_vector is None else dict(self.sparse_vector),
        }

    @classmethod
    def from_chunk(
        cls,
        chunk: Chunk,
        *,
        dense_vector: list[float] | None = None,
        sparse_vector: SparseVector | None = None,
    ) -> ChunkRecord:
        return cls(
            id=chunk.id,
            text=chunk.text,
            metadata=chunk.metadata,
            dense_vector=dense_vector,
            sparse_vector=sparse_vector,
        )


@dataclass(slots=True)
class ProcessedQuery:
    """Structured query payload shared by query processing and retrieval steps."""

    raw_query: str
    keywords: list[str]
    filters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.raw_query = str(self.raw_query)
        self.keywords = [str(keyword) for keyword in self.keywords]
        self.filters = dict(self.filters)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_query": self.raw_query,
            "keywords": list(self.keywords),
            "filters": dict(self.filters),
        }


@dataclass(slots=True)
class RetrievalResult:
    """Normalized retrieval output used by search, rerank, and response builders."""

    chunk_id: str
    score: float
    text: str
    metadata: Metadata

    def __post_init__(self) -> None:
        self.chunk_id = str(self.chunk_id)
        self.score = float(self.score)
        self.text = str(self.text)
        self.metadata = _normalize_metadata(self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "score": self.score,
            "text": self.text,
            "metadata": _clone_mapping(self.metadata),
        }


__all__ = [
    "Chunk",
    "ChunkRecord",
    "Document",
    "IMAGE_PLACEHOLDER_TEMPLATE",
    "ImageMetadata",
    "Metadata",
    "ProcessedQuery",
    "RetrievalResult",
    "SourceRef",
    "SparseVector",
]
