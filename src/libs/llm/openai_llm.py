"""OpenAI-compatible LLM provider implementations."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request

from libs.llm.base_llm import BaseLLM


class OpenAICompatibleLLM(BaseLLM):
    """Shared HTTP client logic for OpenAI-compatible chat providers."""

    api_url_env_var = ""
    api_key_env_var = ""
    default_base_url = ""
    provider_label = ""
    timeout_seconds = 30

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: str,
        api_url: str,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(provider=provider, model=model)
        self.api_key = api_key
        self.api_url = api_url
        self.extra_headers = dict(extra_headers or {})

    def chat(self, messages: list[dict[str, Any]]) -> str:
        """Generate a text response from a chat-style message list."""

        normalized_messages = self._validate_messages(messages)
        payload = self._build_payload(normalized_messages)
        response_data = self._post_json(payload)
        return self._extract_text_response(response_data)

    @classmethod
    def from_settings(cls, settings: Any) -> "OpenAICompatibleLLM":
        """Build a provider instance from settings and environment variables."""

        api_key = getattr(settings, "api_key", None) or os.getenv(cls.api_key_env_var, "")
        api_url = getattr(settings, "api_url", None) or os.getenv(cls.api_url_env_var, cls.default_base_url)
        if not api_key:
            raise ValueError(f"{cls.provider_label} configuration error: missing API key")
        if not api_url:
            raise ValueError(f"{cls.provider_label} configuration error: missing API URL")
        return cls(
            provider=str(settings.provider),
            model=str(settings.model),
            api_key=str(api_key),
            api_url=str(api_url),
            extra_headers=cls._build_extra_headers(settings, api_key=str(api_key)),
        )

    @classmethod
    def _build_extra_headers(cls, settings: Any, *, api_key: str) -> dict[str, str]:
        """Return provider-specific headers for the HTTP request."""

        return {"Authorization": f"Bearer {api_key}"}

    def _build_payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Build the provider request payload."""

        return {
            "model": self.model,
            "messages": messages,
        }

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON POST request and return the parsed response body."""

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        req = request.Request(self.api_url, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw_response = response.read().decode("utf-8")
        except error.HTTPError as exc:
            raise ValueError(f"{self.provider} chat request failed: http_error {exc.code}") from exc
        except error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise ValueError(f"{self.provider} chat request failed: connection_error {reason}") from exc
        except TimeoutError as exc:
            raise ValueError(f"{self.provider} chat request failed: timeout") from exc

        try:
            return json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{self.provider} chat request failed: invalid_json_response") from exc

    def _extract_text_response(self, payload: dict[str, Any]) -> str:
        """Extract assistant text content from an OpenAI-style response payload."""

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError(f"{self.provider} chat request failed: missing choices in response")

        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ValueError(f"{self.provider} chat request failed: missing message in response")

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            text_parts = [item.get("text", "") for item in content if isinstance(item, dict)]
            merged = "".join(part for part in text_parts if part)
            if merged:
                return merged

        raise ValueError(f"{self.provider} chat request failed: empty response content")

    def _validate_messages(self, messages: list[dict[str, Any]]) -> list[dict[str, str]]:
        """Validate the provider-agnostic chat message shape."""

        if not isinstance(messages, list) or not messages:
            raise ValueError(f"{self.provider} chat request failed: messages must be a non-empty list")

        normalized_messages: list[dict[str, str]] = []
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(f"{self.provider} chat request failed: message {index} must be a mapping")
            role = message.get("role")
            content = message.get("content")
            if not isinstance(role, str) or not role.strip():
                raise ValueError(f"{self.provider} chat request failed: message {index} missing role")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"{self.provider} chat request failed: message {index} missing content")
            normalized_messages.append({"role": role, "content": content})
        return normalized_messages


class OpenAILLM(OpenAICompatibleLLM):
    """OpenAI chat-completions provider implementation."""

    api_url_env_var = "OPENAI_API_URL"
    api_key_env_var = "OPENAI_API_KEY"
    default_base_url = "https://api.openai.com/v1/chat/completions"
    provider_label = "openai"
