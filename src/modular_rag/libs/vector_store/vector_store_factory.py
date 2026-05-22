"""Factory for creating vector store providers from project settings."""

from __future__ import annotations

from modular_rag.core.settings import Settings, VectorStoreSettings
from modular_rag.libs.vector_store.base_vector_store import BaseVectorStore


class VectorStoreFactory:
    """Registry-backed factory for pluggable vector store providers."""

    _providers: dict[str, type[BaseVectorStore]] = {}
    _builtin_providers_loaded = False

    @classmethod
    def register(cls, provider: str, vector_store_cls: type[BaseVectorStore]) -> None:
        """Register a provider implementation for later creation."""

        normalized_provider = provider.strip().lower()
        if not normalized_provider:
            raise ValueError("Vector store provider name cannot be empty")
        if not issubclass(vector_store_cls, BaseVectorStore):
            raise TypeError("Registered vector store class must inherit from BaseVectorStore")
        cls._providers[normalized_provider] = vector_store_cls

    @classmethod
    def unregister(cls, provider: str) -> None:
        """Remove a provider registration if present."""

        cls._providers.pop(provider.strip().lower(), None)

    @classmethod
    def create(cls, settings: Settings | VectorStoreSettings) -> BaseVectorStore:
        """Create a vector store instance from top-level settings or vector-store settings."""

        cls._ensure_builtin_providers_loaded()
        vector_store_settings = cls._extract_vector_store_settings(settings)
        provider = vector_store_settings.provider.strip().lower()
        vector_store_cls = cls._providers.get(provider)
        if vector_store_cls is None:
            available = ", ".join(sorted(cls._providers)) or "none"
            raise ValueError(
                "Unsupported vector store provider: "
                f"{vector_store_settings.provider}. Available providers: {available}"
            )
        return vector_store_cls.from_settings(vector_store_settings)

    @classmethod
    def _ensure_builtin_providers_loaded(cls) -> None:
        """Load built-in provider modules on first factory use."""

        if cls._builtin_providers_loaded:
            return
        from modular_rag.libs.vector_store.chroma_store import ChromaStore

        cls.register("chroma", ChromaStore)
        cls._builtin_providers_loaded = True

    @staticmethod
    def _extract_vector_store_settings(settings: Settings | VectorStoreSettings) -> VectorStoreSettings:
        """Normalize supported input types to a ``VectorStoreSettings`` object."""

        if isinstance(settings, VectorStoreSettings):
            return settings
        if isinstance(settings, Settings):
            return settings.vector_store
        raise TypeError("VectorStoreFactory.create expects Settings or VectorStoreSettings")
