"""OpenAI-compatible embedding provider implementations."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request
from urllib.parse import urlparse, urlunparse

from modular_rag.libs.embedding.base_embedding import BaseEmbedding


class OpenAICompatibleEmbedding(BaseEmbedding):
    """Shared HTTP client logic for OpenAI-compatible embedding providers."""

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

    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        """Convert a batch of text inputs into dense vectors."""

        del trace
        normalized_texts = self._validate_texts(texts)
        payload = self._build_payload(normalized_texts)
        response_data = self._post_json(payload)
        return self._extract_embeddings(response_data, expected_count=len(normalized_texts))

    @classmethod
    def from_settings(cls, settings: Any) -> "OpenAICompatibleEmbedding":
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

        del settings
        return {"Authorization": f"Bearer {api_key}"}

    def _build_payload(self, texts: list[str]) -> dict[str, Any]:
        """Build the provider request payload."""

        return {
            "model": self.model,
            "input": texts,
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
        """Extract embedding vectors from an OpenAI-style response payload."""

        data = payload.get("data")
        if not isinstance(data, list) or not data:
            raise ValueError(f"{self.provider} embedding request failed: missing data in response")

        indexed_vectors: dict[int, list[float]] = {}
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                raise ValueError(
                    f"{self.provider} embedding request failed: response item {index} must be a mapping"
                )
            embedding = item.get("embedding")
            if not isinstance(embedding, list) or not embedding:
                raise ValueError(
                    f"{self.provider} embedding request failed: response item {index} missing embedding"
                )
            if not all(isinstance(value, (int, float)) for value in embedding):
                raise ValueError(
                    f"{self.provider} embedding request failed: response item {index} has non-numeric embedding values"
                )
            payload_index = item.get("index", index)
            if not isinstance(payload_index, int):
                raise ValueError(
                    f"{self.provider} embedding request failed: response item {index} has invalid index"
                )
            indexed_vectors[payload_index] = [float(value) for value in embedding]

        if len(indexed_vectors) != expected_count:
            raise ValueError(
                f"{self.provider} embedding request failed: expected {expected_count} embeddings, got {len(indexed_vectors)}"
            )

        try:
            return [indexed_vectors[position] for position in range(expected_count)]
        except KeyError as exc:
            raise ValueError(
                f"{self.provider} embedding request failed: response missing embedding index {exc.args[0]}"
            ) from exc

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
            normalized_texts.append(text)
        return normalized_texts


class OpenAIEmbedding(OpenAICompatibleEmbedding):
    """OpenAI embeddings provider implementation."""

    api_url_env_var = "OPENAI_EMBEDDING_API_URL"
    api_key_env_var = "OPENAI_API_KEY"
    default_base_url = "https://api.openai.com/v1/embeddings"
    provider_label = "openai"

    @classmethod
    def from_settings(cls, settings: Any) -> "OpenAICompatibleEmbedding":
        """Build provider instance and normalize base URL to embeddings endpoint."""

        embedding = super().from_settings(settings)
        embedding.api_url = cls._normalize_openai_embedding_url(str(embedding.api_url))
        return embedding

    @staticmethod
    def _normalize_openai_embedding_url(api_url: str) -> str:
        """Allow both base URLs (.../v1) and full embeddings URLs (.../v1/embeddings)."""

        parsed = urlparse(api_url)
        path = (parsed.path or "").rstrip("/")

        if path.endswith("/embeddings"):
            return api_url

        normalized_path = f"{path}/embeddings" if path else "/embeddings"
        return urlunparse((parsed.scheme, parsed.netloc, normalized_path, parsed.params, parsed.query, parsed.fragment))
