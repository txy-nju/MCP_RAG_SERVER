"""Settings loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class LLMSettings:
    """LLM provider configuration loaded from `settings.yaml`.

    Attributes:
        provider: LLM backend identifier such as `azure` or `openai`.
        model: Concrete model name used by the selected provider.
    """

    provider: str
    model: str


@dataclass(slots=True)
class EmbeddingSettings:
    """Embedding model configuration.

    Attributes:
        provider: Embedding backend identifier.
        model: Embedding model name that will generate vectors.
    """

    provider: str
    model: str


@dataclass(slots=True)
class VectorStoreSettings:
    """Vector store configuration.

    Attributes:
        provider: Storage backend name, for example `chroma`.
        collection: Logical collection name used to group indexed data.
    """

    provider: str
    collection: str


@dataclass(slots=True)
class RetrievalSettings:
    """Retrieval-stage configuration.

    Attributes:
        top_k: Number of candidate results to fetch during retrieval.
    """

    top_k: int


@dataclass(slots=True)
class RerankSettings:
    """Reranker configuration.

    Attributes:
        provider: Reranker backend name, or `none` to disable reranking.
    """

    provider: str


@dataclass(slots=True)
class EvaluationSettings:
    """Evaluation backend configuration.

    Attributes:
        backend: Evaluation backend identifier, such as `custom` or `ragas`.
    """

    backend: str


@dataclass(slots=True)
class ObservabilitySettings:
    """Observability and logging configuration.

    Attributes:
        log_level: Logging verbosity used by the application.
        trace_file: Output path for persisted trace records.
    """

    log_level: str
    trace_file: str


@dataclass(slots=True)
class Settings:
    """Top-level normalized application settings object.

    Attributes:
        llm: LLM-related configuration section.
        embedding: Embedding-related configuration section.
        vector_store: Vector store configuration section.
        retrieval: Retrieval behavior configuration section.
        rerank: Reranker configuration section.
        evaluation: Evaluation backend configuration section.
        observability: Logging and trace configuration section.
    """

    llm: LLMSettings
    embedding: EmbeddingSettings
    vector_store: VectorStoreSettings
    retrieval: RetrievalSettings
    rerank: RerankSettings
    evaluation: EvaluationSettings
    observability: ObservabilitySettings


def _require_mapping(data: Any, field_path: str) -> dict[str, Any]:
    """Ensure a parsed YAML node is a mapping.

    Args:
        data: Raw value loaded from YAML.
        field_path: Logical path used in error messages.

    Returns:
        The same value, narrowed to `dict[str, Any]`.

    Raises:
        ValueError: If `data` is not a mapping.
    """

    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping at '{field_path}'")
    return data


def _require_value(data: dict[str, Any], key: str, field_path: str) -> Any:
    """Read a required key from a mapping.

    Args:
        data: Mapping that should contain the required key.
        key: Required field name inside `data`.
        field_path: Parent path used to build a readable error message.

    Returns:
        The raw value associated with `key`.

    Raises:
        ValueError: If the key is missing or its value is empty.
    """

    value = data.get(key)
    if value in (None, ""):
        raise ValueError(f"Missing required setting: {field_path}.{key}")
    return value


def validate_settings(settings: Settings) -> None:
    """Validate that the normalized settings object contains all required values.

    Args:
        settings: Parsed `Settings` instance produced by `load_settings`.

    Returns:
        None. The function succeeds silently when validation passes.

    Raises:
        ValueError: If any required field is empty after parsing.
    """

    required_values = {
        "llm.provider": settings.llm.provider,
        "llm.model": settings.llm.model,
        "embedding.provider": settings.embedding.provider,
        "embedding.model": settings.embedding.model,
        "vector_store.provider": settings.vector_store.provider,
        "vector_store.collection": settings.vector_store.collection,
        "retrieval.top_k": settings.retrieval.top_k,
        "rerank.provider": settings.rerank.provider,
        "evaluation.backend": settings.evaluation.backend,
        "observability.log_level": settings.observability.log_level,
        "observability.trace_file": settings.observability.trace_file,
    }
    for field_path, value in required_values.items():
        if value in (None, ""):
            raise ValueError(f"Missing required setting: {field_path}")


def load_settings(path: str | Path) -> Settings:
    """Load `settings.yaml` and convert it into typed settings objects.

    Args:
        path: File path to the YAML configuration file.

    Returns:
        A validated `Settings` instance containing all major config sections.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If required sections or fields are missing or malformed.
    """

    settings_path = Path(path)
    with settings_path.open("r", encoding="utf-8") as handle:
        raw_data = yaml.safe_load(handle) or {}

    root = _require_mapping(raw_data, "settings")
    llm = _require_mapping(root.get("llm"), "llm")
    embedding = _require_mapping(root.get("embedding"), "embedding")
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
        vector_store=VectorStoreSettings(
            provider=str(_require_value(vector_store, "provider", "vector_store")),
            collection=str(_require_value(vector_store, "collection", "vector_store")),
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
