"""Unit tests for OpenAI-compatible LLM providers."""

from __future__ import annotations

import json
from urllib import error

import pytest

from core.settings import LLMSettings
from libs.llm.azure_llm import AzureLLM
from libs.llm.deepseek_llm import DeepSeekLLM
from libs.llm.llm_factory import LLMFactory
from libs.llm.openai_llm import OpenAILLM


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
        for key in [
            "OPENAI_API_KEY",
            "OPENAI_API_URL",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_API_URL",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_VERSION",
            "DEEPSEEK_API_KEY",
            "DEEPSEEK_API_URL",
        ]:
            monkeypatch.delenv(key, raising=False)
        yield
    finally:
        LLMFactory._providers.clear()
        LLMFactory._providers.update(original)
        LLMFactory._builtin_providers_loaded = original_loaded


@pytest.mark.unit
def test_factory_creates_openai_provider_and_parses_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")

    def fake_urlopen(req: object, timeout: int) -> DummyResponse:
        assert getattr(req, "full_url") == "https://api.openai.com/v1/chat/completions"
        headers = {key.lower(): value for key, value in req.header_items()}
        assert headers["authorization"] == "Bearer openai-secret"
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["model"] == "gpt-4o-mini"
        assert payload["messages"][0]["role"] == "user"
        return DummyResponse({"choices": [{"message": {"content": "openai-ok"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    llm = LLMFactory.create(LLMSettings(provider="openai", model="gpt-4o-mini"))

    assert isinstance(llm, OpenAILLM)
    assert llm.chat([{"role": "user", "content": "hello"}]) == "openai-ok"


@pytest.mark.unit
def test_factory_creates_azure_provider_and_builds_deployment_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "azure-secret")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")

    def fake_urlopen(req: object, timeout: int) -> DummyResponse:
        assert getattr(req, "full_url") == (
            "https://example.openai.azure.com/openai/deployments/gpt-4o-mini/"
            "chat/completions?api-version=2024-05-01-preview"
        )
        headers = {key.lower(): value for key, value in req.header_items()}
        assert headers["api-key"] == "azure-secret"
        assert "authorization" not in headers
        payload = json.loads(req.data.decode("utf-8"))
        assert "model" not in payload
        return DummyResponse({"choices": [{"message": {"content": "azure-ok"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    llm = LLMFactory.create(LLMSettings(provider="azure", model="gpt-4o-mini"))

    assert isinstance(llm, AzureLLM)
    assert llm.chat([{"role": "user", "content": "hello"}]) == "azure-ok"


@pytest.mark.unit
def test_factory_creates_deepseek_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "deepseek-secret")

    def fake_urlopen(req: object, timeout: int) -> DummyResponse:
        assert getattr(req, "full_url") == "https://api.deepseek.com/chat/completions"
        payload = json.loads(req.data.decode("utf-8"))
        assert payload["model"] == "deepseek-chat"
        return DummyResponse({"choices": [{"message": {"content": "deepseek-ok"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    llm = LLMFactory.create(LLMSettings(provider="deepseek", model="deepseek-chat"))

    assert isinstance(llm, DeepSeekLLM)
    assert llm.chat([{"role": "user", "content": "hello"}]) == "deepseek-ok"


@pytest.mark.unit
def test_chat_validates_message_shape() -> None:
    llm = OpenAILLM(provider="openai", model="gpt-4o-mini", api_key="secret", api_url="https://api.openai.com/v1/chat/completions")

    with pytest.raises(ValueError, match="openai chat request failed: messages must be a non-empty list"):
        llm.chat([])


@pytest.mark.unit
def test_chat_wraps_http_errors_with_provider_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")

    def fake_urlopen(req: object, timeout: int) -> DummyResponse:
        raise error.URLError("boom")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    llm = LLMFactory.create(LLMSettings(provider="openai", model="gpt-4o-mini"))

    with pytest.raises(ValueError, match="openai chat request failed: connection_error boom"):
        llm.chat([{"role": "user", "content": "hello"}])
