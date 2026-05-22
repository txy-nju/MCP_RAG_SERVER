"""Ollama embedding provider implementation."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request

from modular_rag.libs.embedding.base_embedding import BaseEmbedding


class OllamaEmbedding(BaseEmbedding):
    """Ollama embeddings provider backed by a local HTTP endpoint."""

    api_url_env_var = "OLLAMA_EMBEDDING_API_URL"
    default_base_url = "http://localhost:11434/api/embed"
    provider_label = "ollama"
    timeout_seconds = 30
    max_text_length = 32768

    def __init__(self, *, provider: str, model: str, api_url: str) -> None:
        """Initialize the Ollama provider connection metadata."""

        super().__init__(provider=provider, model=model)
        self.api_url = api_url

    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        """Convert a batch of text inputs into dense vectors via Ollama."""

        del trace
        normalized_texts = self._validate_texts(texts)
        payload = self._build_payload(normalized_texts)
        response_data = self._post_json(payload)
        return self._extract_embeddings(response_data, expected_count=len(normalized_texts))

    @classmethod
    def from_settings(cls, settings: Any) -> "OllamaEmbedding":
        """Build an Ollama provider instance from settings and environment variables."""

        api_url = getattr(settings, "api_url", None) or os.getenv(cls.api_url_env_var, cls.default_base_url)
        if not api_url:
            raise ValueError("ollama configuration error: missing API URL")
        return cls(provider=str(settings.provider), model=str(settings.model), api_url=str(api_url))

    def _build_payload(self, texts: list[str]) -> dict[str, Any]:
        """Build the Ollama embeddings request payload."""

        return {
            "model": self.model,
            "input": texts,
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
            raise ValueError(f"{self.provider} embedding request failed: http_error {exc.code}") from exc
        except error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise ValueError(f"{self.provider} embedding request failed: connection_error {reason}") from exc
        except TimeoutError as exc:
            raise ValueError(f"{self.provider} embedding request failed: timeout") from exc

        try:
            return json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{self.provider} embedding request failed: invalid_json_response") from exc

    def _extract_embeddings(self, payload: dict[str, Any], *, expected_count: int) -> list[list[float]]:
        """Extract embedding vectors from an Ollama response payload."""

        embeddings = payload.get("embeddings")
        if isinstance(embeddings, list) and embeddings:
            normalized: list[list[float]] = []
            for index, item in enumerate(embeddings):
                normalized.append(self._normalize_embedding(item, index=index))
            if len(normalized) != expected_count:
                raise ValueError(
                    f"{self.provider} embedding request failed: expected {expected_count} embeddings, got {len(normalized)}"
                )
            return normalized

        embedding = payload.get("embedding")
        if embedding is not None:
            normalized = self._normalize_embedding(embedding, index=0)
            if expected_count != 1:
                raise ValueError(
                    f"{self.provider} embedding request failed: expected {expected_count} embeddings, got 1"
                )
            return [normalized]

        raise ValueError(f"{self.provider} embedding request failed: missing embeddings in response")

    def _normalize_embedding(self, value: Any, *, index: int) -> list[float]:
        """Validate and normalize a single embedding vector."""

        if not isinstance(value, list) or not value:
            raise ValueError(f"{self.provider} embedding request failed: response item {index} missing embedding")
        if not all(isinstance(item, (int, float)) for item in value):
            raise ValueError(
                f"{self.provider} embedding request failed: response item {index} has non-numeric embedding values"
            )
        return [float(item) for item in value]

    def _validate_texts(self, texts: list[str]) -> list[str]:
        """Validate the provider-agnostic embedding input shape."""

        if not isinstance(texts, list) or not texts:
            raise ValueError(f"{self.provider} embedding request failed: texts must be a non-empty list")

        normalized_texts: list[str] = []
        for index, text in enumerate(texts):
            if not isinstance(text, str):
                raise ValueError(f"{self.provider} embedding request failed: text {index} must be a string")
            if not text.strip():
                raise ValueError(f"{self.provider} embedding request failed: text {index} must not be empty")
            if len(text) > self.max_text_length:
                raise ValueError(
                    f"{self.provider} embedding request failed: text {index} exceeds max length {self.max_text_length}"
                )
            normalized_texts.append(text)
        return normalized_texts
