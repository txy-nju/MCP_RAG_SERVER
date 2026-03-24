"""Factory for creating reranker providers from project settings."""

from __future__ import annotations

from core.settings import RerankSettings, Settings
from libs.reranker.base_reranker import BaseReranker, NoneReranker


class RerankerFactory:
    """Registry-backed factory for pluggable reranker providers.

    Returns:
        A factory type that maps configuration provider names to concrete
        ``BaseReranker`` implementations.
    """

    _providers: dict[str, type[BaseReranker]] = {"none": NoneReranker}

    @classmethod
    def register(cls, provider: str, reranker_cls: type[BaseReranker]) -> None:
        """Register a provider implementation for later creation.

        Args:
            provider: Provider key used in configuration, for example ``none`` or
                a future custom backend name.
            reranker_cls: Concrete reranker class implementing ``BaseReranker``.

        Returns:
            None. The provider mapping is stored in the class registry.
        """

        normalized_provider = provider.strip().lower()
        if not normalized_provider:
            raise ValueError("Reranker provider name cannot be empty")
        if not issubclass(reranker_cls, BaseReranker):
            raise TypeError("Registered reranker class must inherit from BaseReranker")
        cls._providers[normalized_provider] = reranker_cls

    @classmethod
    def unregister(cls, provider: str) -> None:
        """Remove a provider registration if present.

        Args:
            provider: Provider key to remove from the registry.

        Returns:
            None. Missing keys are ignored, and the built-in ``none`` provider is
            restored instead of removed.
        """

        normalized_provider = provider.strip().lower()
        if normalized_provider == "none":
            cls._providers[normalized_provider] = NoneReranker
            return
        cls._providers.pop(normalized_provider, None)

    @classmethod
    def create(cls, settings: Settings | RerankSettings) -> BaseReranker:
        """Create a reranker instance from top-level settings or rerank settings.

        Args:
            settings: Either the full project ``Settings`` object or a focused
                ``RerankSettings`` object containing the reranker provider choice.

        Returns:
            A concrete ``BaseReranker`` subclass instance selected from the
            registry by provider name.
        """

        rerank_settings = cls._extract_rerank_settings(settings)
        provider = rerank_settings.provider.strip().lower()
        reranker_cls = cls._providers.get(provider)
        if reranker_cls is None:
            available = ", ".join(sorted(cls._providers)) or "none"
            raise ValueError(
                f"Unsupported reranker provider: {rerank_settings.provider}. Available providers: {available}"
            )
        return reranker_cls.from_settings(rerank_settings)

    @staticmethod
    def _extract_rerank_settings(settings: Settings | RerankSettings) -> RerankSettings:
        """Normalize supported input types to a ``RerankSettings`` object.

        Args:
            settings: Either the full application settings object or a direct
                rerank settings instance.

        Returns:
            The ``RerankSettings`` section that should drive reranker selection.
        """

        if isinstance(settings, RerankSettings):
            return settings
        if isinstance(settings, Settings):
            return settings.rerank
        raise TypeError("RerankerFactory.create expects Settings or RerankSettings")
