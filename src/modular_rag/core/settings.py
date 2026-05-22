"""Settings loading and validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


DEFAULT_VECTOR_STORE_PERSIST_PATH = "data/db/chroma"
DEFAULT_RERANK_PROMPT_PATH = "config/prompts/rerank.txt"
DEFAULT_RERANK_MAX_CANDIDATES = 20
DEFAULT_CHUNK_REFINER_PROMPT_PATH = "config/prompts/chunk_refinement.txt"
DEFAULT_METADATA_ENRICHER_PROMPT_PATH = "config/prompts/metadata_enrichment.txt"
DEFAULT_METADATA_ENRICHER_MAX_TAGS = 5
DEFAULT_IMAGE_CAPTIONER_PROMPT_PATH = "config/prompts/image_captioning.txt"


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
    endpoint: str | None = None
    api_version: str | None = None
    api_url: str | None = None
    api_key: str | None = None
    deployment_name: str | None = None


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
    backends: tuple[str, ...] = ()


@dataclass(slots=True)
class ObservabilitySettings:
    """Observability and logging configuration."""

    log_level: str
    trace_file: str


@dataclass(slots=True)
class ChunkRefinerSettings:
    """Chunk refiner behavior configuration."""

    use_llm: bool = False
    prompt_path: str = DEFAULT_CHUNK_REFINER_PROMPT_PATH


@dataclass(slots=True)
class MetadataEnricherSettings:
    """Metadata enricher behavior configuration."""

    use_llm: bool = False
    prompt_path: str = DEFAULT_METADATA_ENRICHER_PROMPT_PATH
    max_tags: int = DEFAULT_METADATA_ENRICHER_MAX_TAGS


@dataclass(slots=True)
class ImageCaptionerSettings:
    """Image captioner behavior configuration."""

    use_vision_llm: bool = False
    prompt_path: str = DEFAULT_IMAGE_CAPTIONER_PROMPT_PATH


@dataclass(slots=True)
class IngestionSettings:
    """Ingestion-stage optional behaviors."""

    chunk_refiner: ChunkRefinerSettings = field(default_factory=ChunkRefinerSettings)
    metadata_enricher: MetadataEnricherSettings = field(default_factory=MetadataEnricherSettings)
    image_captioner: ImageCaptionerSettings = field(default_factory=ImageCaptionerSettings)


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
    ingestion: IngestionSettings = field(default_factory=IngestionSettings)


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


def _optional_bool(data: dict[str, Any], key: str, default: bool = False) -> bool:
    value = data.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    if isinstance(value, int):
        return bool(value)
    raise ValueError(f"Invalid boolean value for key '{key}': {value!r}")


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
    if settings.ingestion.metadata_enricher.max_tags <= 0:
        raise ValueError("ingestion.metadata_enricher.max_tags must be greater than 0")
    if settings.evaluation.backends and any(backend in (None, "") for backend in settings.evaluation.backends):
        raise ValueError("evaluation.backends cannot contain empty values")


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
    ingestion_raw = root.get("ingestion")

    vision_llm = None
    if vision_llm_raw is not None:
        vision_llm = _build_llm_settings(_require_mapping(vision_llm_raw, "vision_llm"), "vision_llm")

    ingestion = IngestionSettings()
    if ingestion_raw is not None:
        ingestion_map = _require_mapping(ingestion_raw, "ingestion")
        chunk_refiner_raw = ingestion_map.get("chunk_refiner")
        metadata_enricher_raw = ingestion_map.get("metadata_enricher")
        image_captioner_raw = ingestion_map.get("image_captioner")

        chunk_refiner = ChunkRefinerSettings()
        if chunk_refiner_raw is not None:
            chunk_refiner_map = _require_mapping(chunk_refiner_raw, "ingestion.chunk_refiner")
            chunk_refiner = ChunkRefinerSettings(
                use_llm=_optional_bool(chunk_refiner_map, "use_llm", default=False),
                prompt_path=str(chunk_refiner_map.get("prompt_path", DEFAULT_CHUNK_REFINER_PROMPT_PATH)),
            )

        metadata_enricher = MetadataEnricherSettings()
        if metadata_enricher_raw is not None:
            metadata_enricher_map = _require_mapping(metadata_enricher_raw, "ingestion.metadata_enricher")
            metadata_enricher = MetadataEnricherSettings(
                use_llm=_optional_bool(metadata_enricher_map, "use_llm", default=False),
                prompt_path=str(
                    metadata_enricher_map.get("prompt_path", DEFAULT_METADATA_ENRICHER_PROMPT_PATH)
                ),
                max_tags=int(metadata_enricher_map.get("max_tags", DEFAULT_METADATA_ENRICHER_MAX_TAGS)),
            )

        image_captioner = ImageCaptionerSettings()
        if image_captioner_raw is not None:
            image_captioner_map = _require_mapping(image_captioner_raw, "ingestion.image_captioner")
            image_captioner = ImageCaptionerSettings(
                use_vision_llm=_optional_bool(image_captioner_map, "use_vision_llm", default=False),
                prompt_path=str(image_captioner_map.get("prompt_path", DEFAULT_IMAGE_CAPTIONER_PROMPT_PATH)),
            )

        ingestion = IngestionSettings(
            chunk_refiner=chunk_refiner,
            metadata_enricher=metadata_enricher,
            image_captioner=image_captioner,
        )

    raw_backends = evaluation.get("backends")
    parsed_backends: tuple[str, ...] = ()
    if raw_backends not in (None, ""):
        if not isinstance(raw_backends, list):
            raise ValueError("Expected list at 'evaluation.backends'")
        parsed_backends = tuple(str(item) for item in raw_backends if item not in (None, ""))
        if not parsed_backends:
            raise ValueError("evaluation.backends must contain at least one backend")

    primary_backend = str(_require_value(evaluation, "backend", "evaluation")) if not parsed_backends else parsed_backends[0]

    settings = Settings(
        llm=_build_llm_settings(llm, "llm"),
        embedding=EmbeddingSettings(
            provider=str(_require_value(embedding, "provider", "embedding")),
            model=str(_require_value(embedding, "model", "embedding")),
            endpoint=_optional_str(embedding, "endpoint") or _optional_str(embedding, "azure_endpoint"),
            api_version=_optional_str(embedding, "api_version"),
            api_url=_optional_str(embedding, "api_url"),
            api_key=_optional_str(embedding, "api_key"),
            deployment_name=_optional_str(embedding, "deployment_name"),
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
        evaluation=EvaluationSettings(backend=primary_backend, backends=parsed_backends),
        observability=ObservabilitySettings(
            log_level=str(_require_value(observability, "log_level", "observability")),
            trace_file=str(_require_value(observability, "trace_file", "observability")),
        ),
        vision_llm=vision_llm,
        ingestion=ingestion,
    )
    validate_settings(settings)
    return settings
