"""Unit tests for the Ollama LLM provider."""

from __future__ import annotations

import json
from urllib import error

import pytest

from modular_rag.core.settings import LLMSettings
from modular_rag.libs.llm.llm_factory import LLMFactory
from modular_rag.libs.llm.ollama_llm import OllamaLLM


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
def reset_llm_registry_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    original = dict(LLMFactory._providers)
    original_loaded = LLMFactory._builtin_providers_loaded
    try:
        LLMFactory._providers.clear()
        LLMFactory._builtin_providers_loaded = False
        monkeypatch.delenv("OLLAMA_API_URL", raising=False)
        yield
    finally:
        LLMFactory._providers.clear()
        LLMFactory._providers.update(original)
        LLMFactory._builtin_providers_loaded = original_loaded


@pytest.mark.unit
def test_factory_creates_ollama_provider_and_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req: object, timeout: int) -> DummyResponse:
        assert getattr(req, "full_url") == "http://localhost:11434/api/chat"
        headers = {key.lower(): value for key, value in req.header_items()}
        assert headers["content-type"] == "application/json"
        payload = json.loads(req.data.decode("utf-8"))
        assert payload == {
            "model": "llama3",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        }
        return DummyResponse({"message": {"content": "ollama-ok"}})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    llm = LLMFactory.create(LLMSettings(provider="ollama", model="llama3"))

    assert isinstance(llm, OllamaLLM)
    assert llm.chat([{"role": "user", "content": "hello"}]) == "ollama-ok"


@pytest.mark.unit
def test_factory_uses_custom_ollama_api_url_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OLLAMA_API_URL", "http://ollama.local:11434/api/chat")

    llm = LLMFactory.create(LLMSettings(provider="ollama", model="llama3"))

    assert isinstance(llm, OllamaLLM)
    assert llm.api_url == "http://ollama.local:11434/api/chat"


@pytest.mark.unit
def test_chat_wraps_connection_errors_without_leaking_config(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req: object, timeout: int) -> DummyResponse:
        raise error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    llm = LLMFactory.create(LLMSettings(provider="ollama", model="llama3"))

    with pytest.raises(ValueError, match="ollama chat request failed: connection_error connection refused") as exc_info:
        llm.chat([{"role": "user", "content": "hello"}])

    assert "http://localhost:11434/api/chat" not in str(exc_info.value)
    assert "llama3" not in str(exc_info.value)


@pytest.mark.unit
def test_chat_wraps_timeout_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req: object, timeout: int) -> DummyResponse:
        raise TimeoutError("timed out")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    llm = LLMFactory.create(LLMSettings(provider="ollama", model="llama3"))

    with pytest.raises(ValueError, match="ollama chat request failed: timeout"):
        llm.chat([{"role": "user", "content": "hello"}])
