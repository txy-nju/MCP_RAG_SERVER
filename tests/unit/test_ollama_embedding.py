"""Unit tests for the Ollama embedding provider."""

from __future__ import annotations

import json
from urllib import error

import pytest

from modular_rag.core.settings import EmbeddingSettings
from modular_rag.libs.embedding.embedding_factory import EmbeddingFactory
from modular_rag.libs.embedding.ollama_embedding import OllamaEmbedding


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
        monkeypatch.delenv("OLLAMA_EMBEDDING_API_URL", raising=False)
        yield
    finally:
        EmbeddingFactory._providers.clear()
        EmbeddingFactory._providers.update(original)
        EmbeddingFactory._builtin_providers_loaded = original_loaded


@pytest.mark.unit
def test_factory_creates_ollama_embedding_provider_and_parses_batch_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(req: object, timeout: int) -> DummyResponse:
        assert getattr(req, "full_url") == "http://localhost:11434/api/embed"
        headers = {key.lower(): value for key, value in req.header_items()}
        assert headers["content-type"] == "application/json"
        payload = json.loads(req.data.decode("utf-8"))
        assert payload == {
            "model": "nomic-embed-text",
            "input": ["alpha", "beta"],
        }
        return DummyResponse({"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    embedding = EmbeddingFactory.create(
        EmbeddingSettings(provider="ollama", model="nomic-embed-text")
    )

    assert isinstance(embedding, OllamaEmbedding)
    assert embedding.embed(["alpha", "beta"]) == [[0.1, 0.2], [0.3, 0.4]]


@pytest.mark.unit
def test_factory_uses_custom_ollama_embedding_api_url_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OLLAMA_EMBEDDING_API_URL", "http://ollama.local:11434/api/embed")

    embedding = EmbeddingFactory.create(
        EmbeddingSettings(provider="ollama", model="nomic-embed-text")
    )

    assert isinstance(embedding, OllamaEmbedding)
    assert embedding.api_url == "http://ollama.local:11434/api/embed"


@pytest.mark.unit
def test_embed_accepts_single_embedding_response_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req: object, timeout: int) -> DummyResponse:
        return DummyResponse({"embedding": [0.5, 0.6, 0.7]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    embedding = EmbeddingFactory.create(
        EmbeddingSettings(provider="ollama", model="nomic-embed-text")
    )

    assert embedding.embed(["hello"]) == [[0.5, 0.6, 0.7]]


@pytest.mark.unit
def test_embed_rejects_overlong_input() -> None:
    embedding = OllamaEmbedding(
        provider="ollama",
        model="nomic-embed-text",
        api_url="http://localhost:11434/api/embed",
    )

    too_long_text = "a" * (embedding.max_text_length + 1)

    with pytest.raises(ValueError, match="ollama embedding request failed: text 0 exceeds max length"):
        embedding.embed([too_long_text])


@pytest.mark.unit
def test_embed_wraps_connection_errors_without_leaking_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_urlopen(req: object, timeout: int) -> DummyResponse:
        raise error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    embedding = EmbeddingFactory.create(
        EmbeddingSettings(provider="ollama", model="nomic-embed-text")
    )

    with pytest.raises(
        ValueError,
        match="ollama embedding request failed: connection_error connection refused",
    ) as exc_info:
        embedding.embed(["hello"])

    assert "http://localhost:11434/api/embed" not in str(exc_info.value)
    assert "nomic-embed-text" not in str(exc_info.value)


@pytest.mark.unit
def test_embed_wraps_timeout_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req: object, timeout: int) -> DummyResponse:
        raise TimeoutError("timed out")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    embedding = EmbeddingFactory.create(
        EmbeddingSettings(provider="ollama", model="nomic-embed-text")
    )

    with pytest.raises(ValueError, match="ollama embedding request failed: timeout"):
        embedding.embed(["hello"])
