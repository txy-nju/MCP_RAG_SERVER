"""Factory for creating splitter providers from project settings."""

from __future__ import annotations

from core.settings import Settings, SplitterSettings
from libs.splitter.base_splitter import BaseSplitter


class SplitterFactory:
    """Registry-backed factory for pluggable splitter providers."""

    _providers: dict[str, type[BaseSplitter]] = {}

    @classmethod
    def register(cls, provider: str, splitter_cls: type[BaseSplitter]) -> None:
        """Register a provider implementation for later creation."""

        normalized_provider = provider.strip().lower()
        if not normalized_provider:
            raise ValueError("Splitter provider name cannot be empty")
        if not issubclass(splitter_cls, BaseSplitter):
            raise TypeError("Registered splitter class must inherit from BaseSplitter")
        cls._providers[normalized_provider] = splitter_cls

    @classmethod
    def unregister(cls, provider: str) -> None:
        """Remove a provider registration if present."""

        cls._providers.pop(provider.strip().lower(), None)

    @classmethod
    def create(cls, settings: Settings | SplitterSettings) -> BaseSplitter:
        """Create a splitter instance from top-level settings or splitter settings."""

        splitter_settings = cls._extract_splitter_settings(settings)
        provider = splitter_settings.provider.strip().lower()
        splitter_cls = cls._providers.get(provider)
        if splitter_cls is None:
            available = ", ".join(sorted(cls._providers)) or "none"
            raise ValueError(
                f"Unsupported splitter provider: {splitter_settings.provider}. Available providers: {available}"
            )
        return splitter_cls.from_settings(splitter_settings)

    @staticmethod
    def _extract_splitter_settings(settings: Settings | SplitterSettings) -> SplitterSettings:
        """Normalize supported input types to a ``SplitterSettings`` object."""

        if isinstance(settings, SplitterSettings):
            return settings
        if isinstance(settings, Settings):
            return settings.splitter
        raise TypeError("SplitterFactory.create expects Settings or SplitterSettings")
