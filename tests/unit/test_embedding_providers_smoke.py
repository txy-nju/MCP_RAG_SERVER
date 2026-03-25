"""Unit tests for OpenAI-compatible embedding providers."""

from __future__ import annotations

import json
from urllib import error

import pytest

from core.settings import EmbeddingSettings
from libs.embedding.azure_embedding import AzureEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.embedding.openai_embedding import OpenAIEmbedding


class DummyResponse:
    """Minimal response object used to mock urllib responses."""

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "DummyResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


@pytest.fixture(autouse=True)
def reset_embedding_registry_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    original = dict(EmbeddingFactory._providers)
    original_loaded = EmbeddingFactory._builtin_providers_loaded
    try:
        EmbeddingFactory._providers.clear()
        EmbeddingFactory._builtin_providers_loaded = False
        for key in [
            "OPENAI_API_KEY",
            "OPENAI_EMBEDDING_API_URL",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_EMBEDDING_API_URL",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_VERSION",
        ]:
            monkeypatch.delenv(key, raising=False)
        yield
    finally:
        EmbeddingFactory._providers.clear()
        EmbeddingFactory._providers.update(original)
        EmbeddingFactory._builtin_providers_loaded = original_loaded


@pytest.mark.unit
def test_factory_creates_openai_embedding_provider_and_parses_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")

    def fake_urlopen(req: object, timeout: int) -> DummyResponse:
        assert getattr(req, "full_url") == "https://api.openai.com/v1/embeddings"
        headers = {key.lower(): value for key, value in req.header_items()}
        assert headers["authorization"] == "Bearer openai-secret"
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["model"] == "text-embedding-3-small"
        assert payload["input"] == ["alpha", "beta"]
        return DummyResponse(
            {
                "data": [
                    {"index": 1, "embedding": [0.3, 0.4]},
                    {"index": 0, "embedding": [0.1, 0.2]},
                ]
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    embedding = EmbeddingFactory.create(
        EmbeddingSettings(provider="openai", model="text-embedding-3-small")
    )

    assert isinstance(embedding, OpenAIEmbedding)
    assert embedding.embed(["alpha", "beta"]) == [[0.1, 0.2], [0.3, 0.4]]


@pytest.mark.unit
def test_factory_creates_azure_embedding_provider_and_builds_deployment_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-secret")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")

    def fake_urlopen(req: object, timeout: int) -> DummyResponse:
        assert getattr(req, "full_url") == (
            "https://example.openai.azure.com/openai/deployments/text-embedding-3-large/"
            "embeddings?api-version=2024-05-01-preview"
        )
        headers = {key.lower(): value for key, value in req.header_items()}
        assert headers["api-key"] == "azure-secret"
        assert "authorization" not in headers
        payload = json.loads(req.data.decode("utf-8"))
        assert "model" not in payload
        assert payload["input"] == ["hello azure"]
        return DummyResponse({"data": [{"index": 0, "embedding": [0.5, 0.6]}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    embedding = EmbeddingFactory.create(
        EmbeddingSettings(provider="azure", model="text-embedding-3-large")
    )

    assert isinstance(embedding, AzureEmbedding)
    assert embedding.embed(["hello azure"]) == [[0.5, 0.6]]


@pytest.mark.unit
def test_embed_validates_non_empty_input() -> None:
    embedding = OpenAIEmbedding(
        provider="openai",
        model="text-embedding-3-small",
        api_key="secret",
        api_url="https://api.openai.com/v1/embeddings",
    )

    with pytest.raises(
        ValueError,
        match="openai embedding request failed: texts must be a non-empty list",
    ):
        embedding.embed([])


@pytest.mark.unit
def test_embed_wraps_connection_errors_with_provider_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")

    def fake_urlopen(req: object, timeout: int) -> DummyResponse:
        raise error.URLError("boom")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    embedding = EmbeddingFactory.create(
        EmbeddingSettings(provider="openai", model="text-embedding-3-small")
    )

    with pytest.raises(ValueError, match="openai embedding request failed: connection_error boom"):
        embedding.embed(["hello"])
