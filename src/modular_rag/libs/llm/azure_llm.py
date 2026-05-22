"""Azure OpenAI LLM provider implementation."""

from __future__ import annotations

import os
from typing import Any

from modular_rag.libs.llm.openai_llm import OpenAICompatibleLLM


class AzureLLM(OpenAICompatibleLLM):
    """Azure OpenAI chat-completions provider implementation."""

    api_url_env_var = "AZURE_OPENAI_API_URL"
    api_key_env_var = "AZURE_OPENAI_API_KEY"
    provider_label = "azure"

    @classmethod
    def from_settings(cls, settings: Any) -> "AzureLLM":
        """Build an Azure provider instance from settings and environment variables."""

        endpoint = getattr(settings, "endpoint", None) or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        api_version = getattr(settings, "api_version", None) or os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
        api_url = getattr(settings, "api_url", None) or os.getenv(
            cls.api_url_env_var,
            cls._build_api_url(endpoint=endpoint, deployment=str(settings.model), api_version=str(api_version)),
        )
        api_key = getattr(settings, "api_key", None) or os.getenv(cls.api_key_env_var, "")
        if not api_key:
            raise ValueError("azure configuration error: missing API key")
        if not api_url:
            raise ValueError("azure configuration error: missing API URL")
        return cls(
            provider=str(settings.provider),
            model=str(settings.model),
            api_key=str(api_key),
            api_url=str(api_url),
            extra_headers=cls._build_extra_headers(settings, api_key=str(api_key)),
        )

    @staticmethod
    def _build_api_url(endpoint: str, deployment: str, api_version: str) -> str:
        """Build the Azure OpenAI chat-completions endpoint from its parts."""

        cleaned_endpoint = endpoint.rstrip("/")
        if not cleaned_endpoint:
            return ""
        return (
            f"{cleaned_endpoint}/openai/deployments/{deployment}"
            f"/chat/completions?api-version={api_version}"
        )

    @classmethod
    def _build_extra_headers(cls, settings: Any, *, api_key: str) -> dict[str, str]:
        """Return Azure-specific request headers."""

        return {"api-key": api_key}

    def _build_payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Build the Azure chat request payload."""

        return {"messages": messages}
