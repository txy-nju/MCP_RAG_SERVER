"""Chroma vector store implementation."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from libs.vector_store.base_vector_store import BaseVectorStore, VectorStoreQueryResult, VectorStoreRecord

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class ChromaStore(BaseVectorStore):
    """Persistent Chroma-backed vector store."""

    default_persist_path = "data/db/chroma"

    def __init__(self, *, provider: str, collection: str, persist_path: str) -> None:
        """Initialize a Chroma-backed store.

        Args:
            provider: Vector store backend identifier, expected to be ``chroma``.
            collection: Logical collection name used to group stored records.
            persist_path: Local filesystem path where Chroma persists its data.

        Returns:
            None. Creates the persistent client and ensures the target collection
            is available for future upsert/query operations.
        """

        super().__init__(provider=provider, collection=collection)
        self.persist_path = persist_path
        self._client = self._create_client(persist_path)
        self._collection = self._client.get_or_create_collection(
            name=collection,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, records: list[VectorStoreRecord], trace: TraceContext | None = None) -> None:
        """Insert or update vector records in the configured Chroma collection.

        Args:
            records: Normalized vector-store records containing id, embedding,
                metadata, and optional text payload.
            trace: Optional trace context reserved for later observability
                integration. It is currently unused by this implementation.

        Returns:
            None. The supplied records are validated and then written to Chroma
            using upsert semantics.
        """

        del trace
        if not records:
            return

        ids: list[str] = []
        embeddings: list[list[float]] = []
        metadatas: list[dict[str, Any]] = []
        documents: list[str] = []
        for index, record in enumerate(records):
            self._validate_record(record, index=index)
            ids.append(record.id)
            embeddings.append(record.embedding)
            metadatas.append(dict(record.metadata))
            documents.append(record.text or "")

        self._collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)

    def query(
        self,
        vector: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
        trace: TraceContext | None = None,
    ) -> list[VectorStoreQueryResult]:
        """Query the Chroma collection for the nearest stored vectors.

        Args:
            vector: Query embedding used for similarity search.
            top_k: Maximum number of results to return.
            filters: Optional metadata equality filters passed through to Chroma.
            trace: Optional trace context reserved for later observability
                integration. It is currently unused by this implementation.

        Returns:
            A list of ``VectorStoreQueryResult`` objects ordered by Chroma's
            similarity ranking, including id, score, text, and metadata.
        """

        del trace
        self._validate_query_vector(vector)
        if top_k <= 0:
            raise ValueError(f"{self.provider} vector store query failed: top_k must be greater than 0")
        if filters is not None and not isinstance(filters, dict):
            raise ValueError(f"{self.provider} vector store query failed: filters must be a mapping")

        payload = self._collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            where=filters or None,
            include=["documents", "metadatas", "distances"],
        )
        ids = payload.get("ids", [[]])[0]
        documents = payload.get("documents", [[]])[0]
        metadatas = payload.get("metadatas", [[]])[0]
        distances = payload.get("distances", [[]])[0]

        results: list[VectorStoreQueryResult] = []
        for record_id, document, metadata, distance in zip(ids, documents, metadatas, distances, strict=False):
            numeric_distance = float(distance)
            results.append(
                VectorStoreQueryResult(
                    id=str(record_id),
                    score=1.0 - numeric_distance,
                    text=document,
                    metadata=dict(metadata or {}),
                )
            )
        return results

    def get_by_ids(
        self,
        ids: list[str],
        trace: TraceContext | None = None,
    ) -> list[VectorStoreQueryResult]:
        """Fetch Chroma documents by id while preserving caller order."""

        del trace
        if not ids:
            return []
        normalized_ids = [str(record_id).strip() for record_id in ids if str(record_id).strip()]
        if not normalized_ids:
            return []

        payload = self._collection.get(ids=normalized_ids, include=["documents", "metadatas"])
        result_ids = payload.get("ids", [])
        documents = payload.get("documents", [])
        metadatas = payload.get("metadatas", [])

        by_id: dict[str, VectorStoreQueryResult] = {}
        for record_id, document, metadata in zip(result_ids, documents, metadatas, strict=False):
            by_id[str(record_id)] = VectorStoreQueryResult(
                id=str(record_id),
                score=0.0,
                text=document,
                metadata=dict(metadata or {}),
            )

        return [by_id[record_id] for record_id in normalized_ids if record_id in by_id]

    @classmethod
    def from_settings(cls, settings: Any) -> "ChromaStore":
        """Build a Chroma store from vector-store settings.

        Args:
            settings: Settings-like object exposing ``provider``, ``collection``,
                and optionally ``persist_path``.

        Returns:
            A configured ``ChromaStore`` instance ready for persistence and
            querying.
        """

        return cls(
            provider=str(settings.provider),
            collection=str(settings.collection),
            persist_path=str(settings.persist_path),
        )

    @classmethod
    def _create_client(cls, persist_path: str) -> Any:
        """Create a persistent Chroma client rooted at the given path.

        Args:
            persist_path: Directory used by Chroma to store collection data on
                disk.

        Returns:
            A ``chromadb.PersistentClient`` instance bound to the target path.
        """

        chromadb = cls._load_chromadb_module()
        path = Path(persist_path)
        path.mkdir(parents=True, exist_ok=True)
        return chromadb.PersistentClient(path=str(path))

    @staticmethod
    def _load_chromadb_module() -> Any:
        """Import the optional ``chromadb`` dependency lazily.

        Returns:
            The imported ``chromadb`` module.

        Raises:
            ImportError: If ``chromadb`` is not installed in the active Python
                environment.
        """

        try:
            return importlib.import_module("chromadb")
        except ImportError as exc:
            raise ImportError(
                "ChromaStore requires 'chromadb'. Install project dependencies before using provider 'chroma'."
            ) from exc

    def _validate_record(self, record: VectorStoreRecord, *, index: int) -> None:
        """Validate a record before sending it to Chroma.

        Args:
            record: Record to validate.
            index: Positional index used to produce readable error messages.

        Returns:
            None. Raises if the record is missing required fields or contains an
            invalid embedding or metadata shape.
        """

        if not isinstance(record.id, str) or not record.id.strip():
            raise ValueError(f"{self.provider} vector store upsert failed: record {index} missing id")
        self._validate_query_vector(record.embedding, context=f"record {index} embedding")
        if not isinstance(record.metadata, dict):
            raise ValueError(f"{self.provider} vector store upsert failed: record {index} metadata must be a mapping")

    def _validate_query_vector(self, vector: list[float], *, context: str = "query vector") -> None:
        """Validate an embedding vector used for upsert or query operations.

        Args:
            vector: Candidate embedding vector to validate.
            context: Human-readable label included in validation errors so the
                caller can identify which vector failed validation.

        Returns:
            None. Raises ``ValueError`` when the vector is empty or contains
            non-numeric items.
        """

        if not isinstance(vector, list) or not vector:
            raise ValueError(f"{self.provider} vector store query failed: {context} must be a non-empty list")
        if not all(isinstance(item, (int, float)) for item in vector):
            raise ValueError(f"{self.provider} vector store query failed: {context} must contain numeric values")

    def get_collection_stats(self) -> dict[str, Any]:
        """Return basic statistics for the current collection.

        Returns:
            A dict with keys:
            - ``collection``: collection name
            - ``chunk_count``: number of stored vector records
        """
        return {
            "collection": self.collection,
            "chunk_count": self._collection.count(),
        }
