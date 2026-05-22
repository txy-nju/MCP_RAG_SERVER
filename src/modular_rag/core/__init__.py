"""Core application package."""

from modular_rag.core.types import (
    Chunk,
    ChunkRecord,
    Document,
    IMAGE_PLACEHOLDER_TEMPLATE,
    ProcessedQuery,
    RetrievalResult,
)

__all__ = [
    "Chunk",
    "ChunkRecord",
    "Document",
    "IMAGE_PLACEHOLDER_TEMPLATE",
    "ProcessedQuery",
    "RetrievalResult",
]
