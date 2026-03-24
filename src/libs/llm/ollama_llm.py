"""Ollama LLM provider implementation."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request

from libs.llm.base_llm import BaseLLM


class OllamaLLM(BaseLLM):
    """Ollama chat provider backed by a local HTTP endpoint."""

    api_url_env_var = "OLLAMA_API_URL"
    default_base_url = "http://localhost:11434/api/chat"
    provider_label = "ollama"
    timeout_seconds = 30

    def __init__(self, *, provider: str, model: str, api_url: str) -> None:
        """Initialize the Ollama provider connection metadata."""

        super().__init__(provider=provider, model=model)
        self.api_url = api_url

    def chat(self, messages: list[dict[str, Any]]) -> str:
        """Generate a text response from a chat-style message list."""

        normalized_messages = self._validate_messages(messages)
        payload = self._build_payload(normalized_messages)
        response_data = self._post_json(payload)
        return self._extract_text_response(response_data)

    @classmethod
    def from_settings(cls, settings: Any) -> "OllamaLLM":
        """Build an Ollama provider instance from settings and environment variables."""

        api_url = getattr(settings, "api_url", None) or os.getenv(cls.api_url_env_var, cls.default_base_url)
        if not api_url:
            raise ValueError("ollama configuration error: missing API URL")
        return cls(provider=str(settings.provider), model=str(settings.model), api_url=str(api_url))

    def _build_payload(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """Build the Ollama chat request payload."""

        return {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON POST request to the Ollama endpoint."""

        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
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
        """Extract assistant text content from an Ollama response payload."""

        message = payload.get("message")
        if not isinstance(message, dict):
            raise ValueError(f"{self.provider} chat request failed: missing message in response")

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content

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
