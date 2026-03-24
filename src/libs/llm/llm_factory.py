"""Factory for creating LLM providers from project settings."""

from __future__ import annotations

from core.settings import LLMSettings, Settings
from libs.llm.base_llm import BaseLLM


class LLMFactory:
    """Registry-backed factory for pluggable LLM providers."""

    _providers: dict[str, type[BaseLLM]] = {}
    _builtin_providers_loaded = False

    @classmethod
    def register(cls, provider: str, llm_cls: type[BaseLLM]) -> None:
        """Register a provider implementation for later creation.

        Args:
            provider: Provider key used in configuration, for example
                ``azure`` or ``openai``.
            llm_cls: Concrete class that implements the ``BaseLLM`` contract for
                the provider.

        Returns:
            None. The class is stored in the factory registry for future lookups.
        """

        normalized_provider = provider.strip().lower()
        if not normalized_provider:
            raise ValueError("LLM provider name cannot be empty")
        if not issubclass(llm_cls, BaseLLM):
            raise TypeError("Registered LLM class must inherit from BaseLLM")
        cls._providers[normalized_provider] = llm_cls

    @classmethod
    def unregister(cls, provider: str) -> None:
        """Remove a provider registration if present.

        Args:
            provider: Provider key to remove from the registry.

        Returns:
            None. Missing provider keys are ignored.
        """

        cls._providers.pop(provider.strip().lower(), None)

    @classmethod
    def create(cls, settings: Settings | LLMSettings) -> BaseLLM:
        """Create an LLM instance from top-level settings or llm settings.

        Args:
            settings: Either the full project ``Settings`` object or a focused
                ``LLMSettings`` object containing the provider selection.

        Returns:
            A concrete ``BaseLLM`` subclass instance selected by the configured
            provider.
        """

        cls._ensure_builtin_providers_loaded()
        llm_settings = cls._extract_llm_settings(settings)
        provider = llm_settings.provider.strip().lower()
        llm_cls = cls._providers.get(provider)
        if llm_cls is None:
            available = ", ".join(sorted(cls._providers)) or "none"
            raise ValueError(
                f"Unsupported LLM provider: {llm_settings.provider}. Available providers: {available}"
            )
        return llm_cls.from_settings(llm_settings)

    @classmethod
    def _ensure_builtin_providers_loaded(cls) -> None:
        """Load built-in provider modules on first factory use."""

        if cls._builtin_providers_loaded:
            return
        from libs.llm.azure_llm import AzureLLM
        from libs.llm.deepseek_llm import DeepSeekLLM
        from libs.llm.ollama_llm import OllamaLLM
        from libs.llm.openai_llm import OpenAILLM

        cls.register("openai", OpenAILLM)
        cls.register("azure", AzureLLM)
        cls.register("deepseek", DeepSeekLLM)
        cls.register("ollama", OllamaLLM)
        cls._builtin_providers_loaded = True

    @staticmethod
    def _extract_llm_settings(settings: Settings | LLMSettings) -> LLMSettings:
        """Normalize supported input types to an ``LLMSettings`` object.

        Args:
            settings: Either the full project settings or a direct ``LLMSettings``
                instance.

        Returns:
            The ``LLMSettings`` section that should drive provider selection.
        """

        if isinstance(settings, LLMSettings):
            return settings
        if isinstance(settings, Settings):
            return settings.llm
        raise TypeError("LLMFactory.create expects Settings or LLMSettings")
