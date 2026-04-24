"""Business adapter: Document → List[Chunk] via libs.splitter."""

from __future__ import annotations

import hashlib
import re

from core.settings import Settings
from core.types import Chunk, Document
from libs.splitter.splitter_factory import SplitterFactory
from libs.splitter.base_splitter import BaseSplitter


_IMAGE_PLACEHOLDER_RE = re.compile(r"\[IMAGE:\s*([^\]]+)\]")


class DocumentChunker:
    """Adapter layer between libs.splitter (str→str) and ingestion pipeline (Document→Chunk).

    Added value over raw splitter:
    1. Deterministic Chunk ID generation  ({doc_id}_{index:04d}_{hash_8chars})
    2. Document metadata inheritance
    3. chunk_index tracking
    4. source_ref back-link to parent document
    5. Per-chunk image reference distribution
    6. Type conversion: List[str] → List[Chunk]
    """

    def __init__(self, settings: Settings) -> None:
        self._splitter: BaseSplitter = SplitterFactory.create(settings)

    def split_document(self, document: Document) -> list[Chunk]:
        """Split a Document into a list of Chunks with full metadata setup."""
        normalized_document_text = self._sanitize_text(document.text)
        texts = self._splitter.split_text(normalized_document_text)
        chunks: list[Chunk] = []
        for index, text in enumerate(texts):
            normalized_text = self._sanitize_text(text)
            chunk_id = self._generate_chunk_id(document.id, index, normalized_text)
            metadata = self._inherit_metadata(document, index, normalized_text)
            chunk = Chunk(
                id=chunk_id,
                text=normalized_text,
                metadata=metadata,
                start_offset=0,
                end_offset=len(normalized_text),
                source_ref={"doc_id": document.id},
            )
            chunks.append(chunk)
        return chunks

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """Remove invalid Unicode surrogate code points before downstream encoding/splitting."""
        if not text:
            return text
        return str(text).encode("utf-8", errors="replace").decode("utf-8")

    @staticmethod
    def _generate_chunk_id(doc_id: str, index: int, text: str) -> str:
        """Generate a stable, unique chunk ID."""
        hash_8 = hashlib.sha256(text.encode()).hexdigest()[:8]
        return f"{doc_id}_{index:04d}_{hash_8}"

    @staticmethod
    def _inherit_metadata(document: Document, chunk_index: int, chunk_text: str) -> dict:
        """Copy document metadata, add chunk_index, distribute only referenced images."""
        metadata = dict(document.metadata)
        metadata["chunk_index"] = chunk_index

        # Extract document-level images before chunk-level redistribution
        doc_images: list[dict] = list(metadata.pop("images", []))

        if doc_images:
            image_map = {img["id"]: img for img in doc_images}
            referenced_ids = [m.strip() for m in _IMAGE_PLACEHOLDER_RE.findall(chunk_text)]
            if referenced_ids:
                metadata["images"] = [image_map[id_] for id_ in referenced_ids if id_ in image_map]
                metadata["image_refs"] = referenced_ids
        # Chunks with no image placeholders will have images=[] after normalization

        return metadata
