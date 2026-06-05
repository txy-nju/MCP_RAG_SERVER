"""BM25 inverted index builder, persistence, and query runtime."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from modular_rag.core.types import ChunkRecord


class BM25Indexer:
    """Build and query a BM25 inverted index from sparse chunk statistics.

    Index structure:
    {
        "term": {
            "idf": float,
            "postings": [
                {"chunk_id": str, "tf": float, "doc_length": int}
            ]
        }
    }
    """

    _DEFAULT_INDEX_DIR = Path("data/db/bm25")
    _DEFAULT_INDEX_FILE = "bm25_index.json"

    @classmethod
    def _resolve_index_dir(cls, index_dir: str | Path | None = None) -> Path:
        """Resolve index directory: explicit arg > BM25_INDEX_DIR env > default."""
        import os as _os
        if index_dir is not None:
            return Path(index_dir)
        env_dir = _os.environ.get("BM25_INDEX_DIR")
        if env_dir:
            return Path(env_dir)
        return cls._DEFAULT_INDEX_DIR

    def __init__(
        self,
        index_dir: str | Path | None = None,
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._index_dir = self._resolve_index_dir(index_dir)
        self._index_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self._index_dir / self._DEFAULT_INDEX_FILE

        self._k1 = float(k1)
        self._b = float(b)

        self._index: dict[str, dict[str, Any]] = {}
        self._documents: dict[str, dict[str, Any]] = {}
        self._num_docs: int = 0
        self._avg_doc_length: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def index_path(self) -> Path:
        """Return the persisted index file path."""
        return self._index_path

    def build(self, records: list[ChunkRecord], *, rebuild: bool = True) -> None:
        """Build or incrementally update index from sparse chunk records.

        Args:
            records: Chunk records with ``sparse_vector`` populated.
            rebuild: If True, rebuild from scratch using only ``records``.
                If False, incrementally upsert these records into existing index.
        """
        if rebuild:
            self._documents = {}

        for record in records:
            sparse = record.sparse_vector or {}
            if not sparse:
                continue
            terms = {term: float(tf) for term, tf in sparse.items() if float(tf) > 0.0}
            if not terms:
                continue

            self._documents[record.id] = {
                "chunk_id": record.id,
                "source_path": str(record.metadata.get("source_path", "")),
                "terms": terms,
                "doc_length": int(sum(terms.values())),
            }

        self._recompute_index()
        self.save()

    def save(self) -> None:
        """Persist current index and document statistics to JSON file."""
        payload = {
            "k1": self._k1,
            "b": self._b,
            "num_docs": self._num_docs,
            "avg_doc_length": self._avg_doc_length,
            "index": self._index,
            "documents": self._documents,
        }
        self._index_path.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2),
            encoding="utf-8",
        )

    def load(self) -> None:
        """Load persisted index from disk into memory.

        Raises:
            FileNotFoundError: If the index file does not exist.
            ValueError: If persisted index content is malformed.
        """
        if not self._index_path.exists():
            raise FileNotFoundError(f"BM25 index file not found: {self._index_path}")

        payload = json.loads(self._index_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("Invalid BM25 index payload: expected mapping")

        self._k1 = float(payload.get("k1", self._k1))
        self._b = float(payload.get("b", self._b))
        self._num_docs = int(payload.get("num_docs", 0))
        self._avg_doc_length = float(payload.get("avg_doc_length", 0.0))
        self._index = dict(payload.get("index", {}))
        self._documents = dict(payload.get("documents", {}))

    def query(self, keywords: list[str], top_k: int = 10) -> list[dict[str, float | str]]:
        """Query BM25 index with keywords and return top scored chunk IDs."""
        if top_k <= 0:
            return []

        terms = [str(keyword).strip().lower() for keyword in keywords if str(keyword).strip()]
        if not terms or not self._index:
            return []

        scores: dict[str, float] = {}
        for term in terms:
            term_data = self._index.get(term)
            if term_data is None:
                continue
            idf = float(term_data.get("idf", 0.0))
            postings = term_data.get("postings", [])
            for posting in postings:
                chunk_id = str(posting["chunk_id"])
                tf = float(posting["tf"])
                doc_length = int(posting["doc_length"])
                score = self._bm25_term_score(tf, idf, doc_length)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + score

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [
            {"chunk_id": chunk_id, "score": float(round(score, 6))}
            for chunk_id, score in ranked[:top_k]
        ]

    def remove_document(self, source: str) -> None:
        """Remove all indexed chunks that belong to *source* and persist changes."""
        normalized_source = str(source).strip()
        if not normalized_source:
            return

        remove_ids = [
            chunk_id
            for chunk_id, payload in self._documents.items()
            if str(payload.get("source_path", "")) == normalized_source
        ]
        if not remove_ids:
            return

        for chunk_id in remove_ids:
            self._documents.pop(chunk_id, None)

        self._recompute_index()
        self.save()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _recompute_index(self) -> None:
        self._num_docs = len(self._documents)
        total_doc_length = sum(int(doc["doc_length"]) for doc in self._documents.values())
        self._avg_doc_length = (total_doc_length / self._num_docs) if self._num_docs else 0.0

        doc_freq: dict[str, int] = {}
        postings_map: dict[str, list[dict[str, float | str | int]]] = {}

        for doc in self._documents.values():
            chunk_id = str(doc["chunk_id"])
            terms = dict(doc["terms"])
            doc_length = int(doc["doc_length"])

            for term, tf in terms.items():
                doc_freq[term] = doc_freq.get(term, 0) + 1
                postings_map.setdefault(term, []).append(
                    {
                        "chunk_id": chunk_id,
                        "tf": float(tf),
                        "doc_length": doc_length,
                    }
                )

        new_index: dict[str, dict[str, Any]] = {}
        for term, postings in postings_map.items():
            postings_sorted = sorted(postings, key=lambda row: str(row["chunk_id"]))
            df = doc_freq[term]
            idf = self._compute_idf(self._num_docs, df)
            new_index[term] = {
                "idf": float(idf),
                "postings": postings_sorted,
            }

        self._index = dict(sorted(new_index.items(), key=lambda item: item[0]))

    @staticmethod
    def _compute_idf(num_docs: int, doc_freq: int) -> float:
        if num_docs <= 0 or doc_freq <= 0:
            return 0.0
        return math.log((num_docs - doc_freq + 0.5) / (doc_freq + 0.5))

    def _bm25_term_score(self, tf: float, idf: float, doc_length: int) -> float:
        if tf <= 0.0 or self._avg_doc_length <= 0.0:
            return 0.0
        norm = self._k1 * (1.0 - self._b + self._b * (doc_length / self._avg_doc_length))
        return idf * ((tf * (self._k1 + 1.0)) / (tf + norm))

