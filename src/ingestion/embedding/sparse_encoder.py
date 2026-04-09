"""Sparse encoder: computes BM25 term-frequency statistics for chunks."""

from __future__ import annotations

import re
from collections import Counter

from core.settings import Settings
from core.trace.trace_context import TraceContext
from core.types import Chunk, ChunkRecord, SparseVector

# Minimal stop-word list (English high-frequency function words)
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "up", "is", "it", "as", "be", "was", "are",
    "that", "this", "have", "has", "had", "not", "do", "does", "did", "so",
    "if", "its", "my", "we", "he", "she", "they", "you", "me", "us", "him",
    "her", "his", "our", "your", "their", "can", "will", "would", "could",
    "should", "may", "might", "shall", "been", "being", "am",
})


def _tokenize(text: str) -> list[str]:
    """Tokenize *text* into lowercase terms, filtering stop words and short tokens.

    Handles both ASCII words/numbers and CJK characters (each CJK char is a
    separate token).  ASCII tokens shorter than 2 characters are discarded;
    single CJK characters are kept because they carry semantic meaning.
    Tokens present in the stop-word list are always discarded.
    """
    # Match either ASCII word/number sequences or individual CJK characters
    raw_tokens: list[str] = re.findall(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]", text.lower())
    result: list[str] = []
    for t in raw_tokens:
        if t in _STOP_WORDS:
            continue
        # Keep CJK single chars; discard single ASCII chars (noisy)
        is_cjk = "\u4e00" <= t <= "\u9fff"
        if not is_cjk and len(t) < 2:
            continue
        result.append(t)
    return result


class SparseEncoder:
    """Compute per-chunk BM25 term-frequency sparse vectors.

    The output ``sparse_vector`` for each :class:`~core.types.ChunkRecord` is a
    mapping of ``{term: raw_tf_count}`` where TF is the raw occurrence count of
    the term in that chunk.  The IDF component and document-frequency statistics
    are intentionally *not* computed here — they require the full corpus and are
    handled by :class:`~ingestion.storage.bm25_indexer.BM25Indexer` (C11).

    The output structure is compatible with ``BM25Indexer.build()`` which
    expects ``ChunkRecord.sparse_vector`` to carry per-term TF counts.

    Responsibilities:
    1. Tokenize each chunk's text (lowercase, ASCII + CJK, stop-word filtered)
    2. Count raw term frequencies per chunk
    3. Wrap results in :class:`~core.types.ChunkRecord` (``dense_vector=None``)
    """

    def __init__(self, settings: Settings) -> None:
        """Initialise SparseEncoder.

        Args:
            settings: Project settings (reserved for future configuration such
                      as custom stop-word lists or tokeniser settings).
        """
        self._settings = settings

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(
        self, chunks: list[Chunk], trace: TraceContext | None = None
    ) -> list[ChunkRecord]:
        """Compute sparse TF vectors for *chunks*.

        Args:
            chunks: Input chunks to encode.  Must be non-empty.
            trace: Optional trace context for recording stage metrics.

        Returns:
            A list of :class:`~core.types.ChunkRecord` objects in the same
            order as *chunks*, each with ``sparse_vector`` populated and
            ``dense_vector`` left as ``None``.

        Raises:
            ValueError: If *chunks* is empty.
        """
        if not chunks:
            raise ValueError("Cannot encode an empty chunks list")

        records: list[ChunkRecord] = []
        for chunk in chunks:
            sparse_vector = self._compute_tf_vector(chunk.text)
            record = ChunkRecord.from_chunk(chunk, sparse_vector=sparse_vector)
            records.append(record)

        if trace is not None:
            trace.record_stage(
                "sparse_encode",
                {
                    "method": "bm25_tf",
                    "chunk_count": len(chunks),
                    "total_terms": sum(
                        len(r.sparse_vector) for r in records if r.sparse_vector
                    ),
                },
            )

        return records

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_tf_vector(text: str) -> SparseVector:
        """Tokenise *text* and return a ``{term: tf_count}`` mapping.

        Returns an empty dict for blank/whitespace-only input so that
        downstream consumers (e.g. BM25Indexer) can handle sparse chunks
        without raising exceptions.
        """
        if not text or not text.strip():
            return {}

        tokens = _tokenize(text)
        if not tokens:
            return {}

        counts: Counter[str] = Counter(tokens)
        return {term: float(count) for term, count in counts.items()}
    