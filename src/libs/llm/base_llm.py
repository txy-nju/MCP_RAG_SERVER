"""Base interfaces for pluggable LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseLLM(ABC):
    """Common contract for all LLM providers used by the project."""

    def __init__(self, *, provider: str, model: str) -> None:
        """Initialize shared provider metadata for an LLM instance.

        Args:
            provider: Normalized provider name such as ``azure`` or ``openai``.
            model: Concrete model identifier exposed by the provider.

        Returns:
            None. The constructor stores the shared metadata on the instance.
        """
        self.provider = provider
        self.model = model

    @abstractmethod
    def chat(self, messages: list[dict[str, Any]]) -> str:
        """Generate a text response from a chat-style message list.

        Args:
            messages: Ordered chat messages passed to the provider. Each item is
                expected to contain provider-agnostic keys such as ``role`` and
                ``content``.

        Returns:
            The model's text response for the supplied conversation context.
        """

    @classmethod
    def from_settings(cls, settings: Any) -> "BaseLLM":
        """Build an LLM instance from an llm settings object.

        Args:
            settings: A configuration object that provides at least ``provider``
                and ``model`` attributes, typically ``LLMSettings``.

        Returns:
            A newly constructed provider instance initialized from the settings.
        """

        return cls(provider=str(settings.provider), model=str(settings.model))
