"""Settings loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class LLMSettings:
    """LLM provider configuration loaded from `settings.yaml`."""

    provider: str
    model: str


@dataclass(slots=True)
class EmbeddingSettings:
    """Embedding model configuration."""

    provider: str
    model: str


@dataclass(slots=True)
class SplitterSettings:
    """Text splitter configuration."""

    provider: str
    chunk_size: int
    chunk_overlap: int


@dataclass(slots=True)
class VectorStoreSettings:
    """Vector store configuration."""

    provider: str
    collection: str
    persist_path: str = "data/db/chroma"


@dataclass(slots=True)
class RetrievalSettings:
    """Retrieval-stage configuration."""

    top_k: int


@dataclass(slots=True)
class RerankSettings:
    """Reranker configuration."""

    provider: str


@dataclass(slots=True)
class EvaluationSettings:
    """Evaluation backend configuration."""

    backend: str


@dataclass(slots=True)
class ObservabilitySettings:
    """Observability and logging configuration."""

    log_level: str
    trace_file: str


@dataclass(slots=True)
class Settings:
    """Top-level normalized application settings object."""

    llm: LLMSettings
    embedding: EmbeddingSettings
    splitter: SplitterSettings
    vector_store: VectorStoreSettings
    retrieval: RetrievalSettings
    rerank: RerankSettings
    evaluation: EvaluationSettings
    observability: ObservabilitySettings


def _require_mapping(data: Any, field_path: str) -> dict[str, Any]:
    """Ensure a parsed YAML node is a mapping."""

    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at '{field_path}'")
    return data


def _require_value(data: dict[str, Any], key: str, field_path: str) -> Any:
    """Read a required key from a mapping."""

    value = data.get(key)
    if value in (None, ""):
        raise ValueError(f"Missing required setting: {field_path}.{key}")
    return value


def validate_settings(settings: Settings) -> None:
    """Validate that the normalized settings object contains all required values."""

    required_values = {
        "llm.provider": settings.llm.provider,
        "llm.model": settings.llm.model,
        "embedding.provider": settings.embedding.provider,
        "embedding.model": settings.embedding.model,
        "splitter.provider": settings.splitter.provider,
        "splitter.chunk_size": settings.splitter.chunk_size,
        "splitter.chunk_overlap": settings.splitter.chunk_overlap,
        "vector_store.provider": settings.vector_store.provider,
        "vector_store.collection": settings.vector_store.collection,
        "vector_store.persist_path": settings.vector_store.persist_path,
        "retrieval.top_k": settings.retrieval.top_k,
        "rerank.provider": settings.rerank.provider,
        "evaluation.backend": settings.evaluation.backend,
        "observability.log_level": settings.observability.log_level,
        "observability.trace_file": settings.observability.trace_file,
    }
    for field_path, value in required_values.items():
        if value in (None, ""):
            raise ValueError(f"Missing required setting: {field_path}")

    if settings.splitter.chunk_size <= 0:
        raise ValueError("splitter.chunk_size must be greater than 0")
    if settings.splitter.chunk_overlap < 0:
        raise ValueError("splitter.chunk_overlap must be greater than or equal to 0")
    if settings.splitter.chunk_overlap >= settings.splitter.chunk_size:
        raise ValueError("splitter.chunk_overlap must be smaller than splitter.chunk_size")


def load_settings(path: str | Path) -> Settings:
    """Load `settings.yaml` and convert it into typed settings objects."""

    settings_path = Path(path)
    with settings_path.open("r", encoding="utf-8") as handle:
        raw_data = yaml.safe_load(handle) or {}

    root = _require_mapping(raw_data, "settings")
    llm = _require_mapping(root.get("llm"), "llm")
    embedding = _require_mapping(root.get("embedding"), "embedding")
    splitter = _require_mapping(root.get("splitter"), "splitter")
    vector_store = _require_mapping(root.get("vector_store"), "vector_store")
    retrieval = _require_mapping(root.get("retrieval"), "retrieval")
    rerank = _require_mapping(root.get("rerank"), "rerank")
    evaluation = _require_mapping(root.get("evaluation"), "evaluation")
    observability = _require_mapping(root.get("observability"), "observability")

    settings = Settings(
        llm=LLMSettings(
            provider=str(_require_value(llm, "provider", "llm")),
            model=str(_require_value(llm, "model", "llm")),
        ),
        embedding=EmbeddingSettings(
            provider=str(_require_value(embedding, "provider", "embedding")),
            model=str(_require_value(embedding, "model", "embedding")),
        ),
        splitter=SplitterSettings(
            provider=str(_require_value(splitter, "provider", "splitter")),
            chunk_size=int(_require_value(splitter, "chunk_size", "splitter")),
            chunk_overlap=int(_require_value(splitter, "chunk_overlap", "splitter")),
        ),
        vector_store=VectorStoreSettings(
            provider=str(_require_value(vector_store, "provider", "vector_store")),
            collection=str(_require_value(vector_store, "collection", "vector_store")),
            persist_path=str(vector_store.get("persist_path", "data/db/chroma")),
        ),
        retrieval=RetrievalSettings(top_k=int(_require_value(retrieval, "top_k", "retrieval"))),
        rerank=RerankSettings(provider=str(_require_value(rerank, "provider", "rerank"))),
        evaluation=EvaluationSettings(backend=str(_require_value(evaluation, "backend", "evaluation"))),
        observability=ObservabilitySettings(
            log_level=str(_require_value(observability, "log_level", "observability")),
            trace_file=str(_require_value(observability, "trace_file", "observability")),
        ),
    )
    validate_settings(settings)
    return settings
