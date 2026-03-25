"""Factory for creating embedding providers from project settings."""

from __future__ import annotations

from core.settings import EmbeddingSettings, Settings
from libs.embedding.base_embedding import BaseEmbedding


class EmbeddingFactory:
    """Registry-backed factory for pluggable embedding providers."""

    _providers: dict[str, type[BaseEmbedding]] = {}
    _builtin_providers_loaded = False

    @classmethod
    def register(cls, provider: str, embedding_cls: type[BaseEmbedding]) -> None:
        """Register a provider implementation for later creation."""

        normalized_provider = provider.strip().lower()
        if not normalized_provider:
            raise ValueError("Embedding provider name cannot be empty")
        if not issubclass(embedding_cls, BaseEmbedding):
            raise TypeError("Registered embedding class must inherit from BaseEmbedding")
        cls._providers[normalized_provider] = embedding_cls

    @classmethod
    def unregister(cls, provider: str) -> None:
        """Remove a provider registration if present."""

        cls._providers.pop(provider.strip().lower(), None)

    @classmethod
    def create(cls, settings: Settings | EmbeddingSettings) -> BaseEmbedding:
        """Create an embedding instance from top-level settings or embedding settings."""

        cls._ensure_builtin_providers_loaded()
        embedding_settings = cls._extract_embedding_settings(settings)
        provider = embedding_settings.provider.strip().lower()
        embedding_cls = cls._providers.get(provider)
        if embedding_cls is None:
            available = ", ".join(sorted(cls._providers)) or "none"
            raise ValueError(
                "Unsupported embedding provider: "
                f"{embedding_settings.provider}. Available providers: {available}"
            )
        return embedding_cls.from_settings(embedding_settings)

    @classmethod
    def _ensure_builtin_providers_loaded(cls) -> None:
        """Load built-in provider modules on first factory use."""

        if cls._builtin_providers_loaded:
            return
        from libs.embedding.azure_embedding import AzureEmbedding
        from libs.embedding.openai_embedding import OpenAIEmbedding

        cls.register("openai", OpenAIEmbedding)
        cls.register("azure", AzureEmbedding)
        cls._builtin_providers_loaded = True

    @staticmethod
    def _extract_embedding_settings(settings: Settings | EmbeddingSettings) -> EmbeddingSettings:
        """Normalize supported input types to an ``EmbeddingSettings`` object."""

        if isinstance(settings, EmbeddingSettings):
            return settings
        if isinstance(settings, Settings):
            return settings.embedding
        raise TypeError("EmbeddingFactory.create expects Settings or EmbeddingSettings")
