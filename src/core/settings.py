"""Settings loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


DEFAULT_VECTOR_STORE_PERSIST_PATH = "data/db/chroma"
DEFAULT_RERANK_PROMPT_PATH = "config/prompts/rerank.txt"
DEFAULT_RERANK_MAX_CANDIDATES = 20


@dataclass(slots=True)
class LLMSettings:
    """LLM provider configuration loaded from `settings.yaml`."""

    provider: str
    model: str
    endpoint: str | None = None
    api_version: str | None = None
    api_url: str | None = None
    api_key: str | None = None
    deployment_name: str | None = None
    max_image_size: int | None = None
    timeout_seconds: int | None = None


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
    persist_path: str = DEFAULT_VECTOR_STORE_PERSIST_PATH


@dataclass(slots=True)
class RetrievalSettings:
    """Retrieval-stage configuration."""

    top_k: int


@dataclass(slots=True)
class RerankSettings:
    """Reranker configuration."""

    provider: str
    prompt_path: str = DEFAULT_RERANK_PROMPT_PATH
    max_candidates: int = DEFAULT_RERANK_MAX_CANDIDATES


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
    vision_llm: LLMSettings | None = None


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


def _optional_str(data: dict[str, Any], key: str) -> str | None:
    value = data.get(key)
    if value in (None, ""):
        return None
    return str(value)


def _optional_int(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    if value in (None, ""):
        return None
    return int(value)


def _build_llm_settings(data: dict[str, Any], field_path: str) -> LLMSettings:
    return LLMSettings(
        provider=str(_require_value(data, "provider", field_path)),
        model=str(_require_value(data, "model", field_path)),
        endpoint=_optional_str(data, "endpoint") or _optional_str(data, "azure_endpoint"),
        api_version=_optional_str(data, "api_version"),
        api_url=_optional_str(data, "api_url"),
        api_key=_optional_str(data, "api_key"),
        deployment_name=_optional_str(data, "deployment_name"),
        max_image_size=_optional_int(data, "max_image_size"),
        timeout_seconds=_optional_int(data, "timeout_seconds"),
    )


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
        "rerank.prompt_path": settings.rerank.prompt_path,
        "rerank.max_candidates": settings.rerank.max_candidates,
        "evaluation.backend": settings.evaluation.backend,
        "observability.log_level": settings.observability.log_level,
        "observability.trace_file": settings.observability.trace_file,
    }
    for field_path, value in required_values.items():
        if value in (None, ""):
            raise ValueError(f"Missing required setting: {field_path}")

    if settings.vision_llm is not None:
        if settings.vision_llm.provider in (None, ""):
            raise ValueError("Missing required setting: vision_llm.provider")
        if settings.vision_llm.model in (None, ""):
            raise ValueError("Missing required setting: vision_llm.model")
        if settings.vision_llm.max_image_size is not None and settings.vision_llm.max_image_size <= 0:
            raise ValueError("vision_llm.max_image_size must be greater than 0")
        if settings.vision_llm.timeout_seconds is not None and settings.vision_llm.timeout_seconds <= 0:
            raise ValueError("vision_llm.timeout_seconds must be greater than 0")

    if settings.splitter.chunk_size <= 0:
        raise ValueError("splitter.chunk_size must be greater than 0")
    if settings.splitter.chunk_overlap < 0:
        raise ValueError("splitter.chunk_overlap must be greater than or equal to 0")
    if settings.splitter.chunk_overlap >= settings.splitter.chunk_size:
        raise ValueError("splitter.chunk_overlap must be smaller than splitter.chunk_size")
    if settings.rerank.max_candidates <= 0:
        raise ValueError("rerank.max_candidates must be greater than 0")


def load_settings(path: str | Path) -> Settings:
    """Load `settings.yaml` and convert it into typed settings objects."""

    settings_path = Path(path)
    with settings_path.open("r", encoding="utf-8") as handle:
        raw_data = yaml.safe_load(handle) or {}

    root = _require_mapping(raw_data, "settings")
    llm = _require_mapping(root.get("llm"), "llm")
    vision_llm_raw = root.get("vision_llm")
    embedding = _require_mapping(root.get("embedding"), "embedding")
    splitter = _require_mapping(root.get("splitter"), "splitter")
    vector_store = _require_mapping(root.get("vector_store"), "vector_store")
    retrieval = _require_mapping(root.get("retrieval"), "retrieval")
    rerank = _require_mapping(root.get("rerank"), "rerank")
    evaluation = _require_mapping(root.get("evaluation"), "evaluation")
    observability = _require_mapping(root.get("observability"), "observability")

    vision_llm = None
    if vision_llm_raw is not None:
        vision_llm = _build_llm_settings(_require_mapping(vision_llm_raw, "vision_llm"), "vision_llm")

    settings = Settings(
        llm=_build_llm_settings(llm, "llm"),
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
            persist_path=str(vector_store.get("persist_path", DEFAULT_VECTOR_STORE_PERSIST_PATH)),
        ),
        retrieval=RetrievalSettings(top_k=int(_require_value(retrieval, "top_k", "retrieval"))),
        rerank=RerankSettings(
            provider=str(_require_value(rerank, "provider", "rerank")),
            prompt_path=str(rerank.get("prompt_path", DEFAULT_RERANK_PROMPT_PATH)),
            max_candidates=int(rerank.get("max_candidates", DEFAULT_RERANK_MAX_CANDIDATES)),
        ),
        evaluation=EvaluationSettings(backend=str(_require_value(evaluation, "backend", "evaluation"))),
        observability=ObservabilitySettings(
            log_level=str(_require_value(observability, "log_level", "observability")),
            trace_file=str(_require_value(observability, "trace_file", "observability")),
        ),
        vision_llm=vision_llm,
    )
    validate_settings(settings)
    return settings
