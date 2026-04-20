"""Unit tests for the Azure vision LLM provider."""

from __future__ import annotations

import base64
import io
import json
from urllib import error

import pytest

from core.settings import (
    EmbeddingSettings,
    EvaluationSettings,
    LLMSettings,
    ObservabilitySettings,
    RetrievalSettings,
    RerankSettings,
    Settings,
    SplitterSettings,
    VectorStoreSettings,
)
from libs.llm.azure_vision_llm import AzureVisionLLM
from libs.llm.llm_factory import LLMFactory


_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMCAO6n5mQAAAAASUVORK5CYII="
)


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
def reset_llm_registries(monkeypatch: pytest.MonkeyPatch) -> None:
    original_text = dict(LLMFactory._providers)
    original_vision = dict(LLMFactory._vision_providers)
    original_loaded = LLMFactory._builtin_providers_loaded
    try:
        LLMFactory._providers.clear()
        LLMFactory._vision_providers.clear()
        LLMFactory._builtin_providers_loaded = False
        monkeypatch.delenv("AZURE_OPENAI_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
        monkeypatch.delenv("AZURE_OPENAI_VISION_API_URL", raising=False)
        yield
    finally:
        LLMFactory._providers.clear()
        LLMFactory._providers.update(original_text)
        LLMFactory._vision_providers.clear()
        LLMFactory._vision_providers.update(original_vision)
        LLMFactory._builtin_providers_loaded = original_loaded


def make_settings() -> Settings:
    return Settings(
        llm=LLMSettings(provider="openai", model="gpt-4o-mini"),
        embedding=EmbeddingSettings(provider="openai", model="text-embedding-3-small"),
        splitter=SplitterSettings(provider="recursive", chunk_size=1000, chunk_overlap=200),
        vector_store=VectorStoreSettings(provider="chroma", collection="default"),
        retrieval=RetrievalSettings(top_k=5),
        rerank=RerankSettings(provider="none"),
        evaluation=EvaluationSettings(backend="custom"),
        observability=ObservabilitySettings(log_level="INFO", trace_file="logs/traces.jsonl"),
        vision_llm=LLMSettings(
            provider="azure",
            model="gpt-4o",
            endpoint="https://example-resource.openai.azure.com",
            api_version="2024-02-15-preview",
            api_key="secret-key",
            deployment_name="vision-deployment",
            max_image_size=1024,
            timeout_seconds=12,
        ),
    )


@pytest.mark.unit
def test_factory_creates_azure_vision_provider_from_top_level_settings() -> None:
    llm = LLMFactory.create_vision_llm(make_settings())

    assert isinstance(llm, AzureVisionLLM)
    assert llm.api_url.endswith("/openai/deployments/vision-deployment/chat/completions?api-version=2024-02-15-preview")
    assert llm.max_image_size == 1024
    assert llm.timeout_seconds == 12


@pytest.mark.unit
def test_chat_with_image_sends_azure_multimodal_request_for_file_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(_PNG_1X1)

    def fake_urlopen(req: object, timeout: int) -> DummyResponse:
        assert getattr(req, "full_url").endswith("api-version=2024-02-15-preview")
        assert timeout == 12
        headers = {key.lower(): value for key, value in req.header_items()}
        assert headers["api-key"] == "secret-key"
        payload = json.loads(req.data.decode("utf-8"))
        content = payload["messages"][0]["content"]
        assert content[0] == {"type": "text", "text": "describe this image"}
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
        return DummyResponse({"choices": [{"message": {"content": "vision-ok"}}]})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    llm = LLMFactory.create_vision_llm(make_settings())

    response = llm.chat_with_image("describe this image", str(image_path))

    assert response.content == "vision-ok"
    assert response.metadata["deployment_name"] == "vision-deployment"


@pytest.mark.unit
def test_chat_with_image_supports_bytes_input_and_resize_hook(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_resize(self: AzureVisionLLM, image_bytes: bytes) -> bytes:
        assert image_bytes == b"original"
        return b"compressed"

    def fake_urlopen(req: object, timeout: int) -> DummyResponse:
        payload = json.loads(req.data.decode("utf-8"))
        data_url = payload["messages"][0]["content"][1]["image_url"]["url"]
        assert data_url == "data:image/png;base64,Y29tcHJlc3NlZA=="
        return DummyResponse({"choices": [{"message": {"content": [{"type": "text", "text": "bytes-ok"}]}}]})

    monkeypatch.setattr(AzureVisionLLM, "_resize_image_bytes_if_needed", fake_resize)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    llm = LLMFactory.create_vision_llm(make_settings())

    response = llm.chat_with_image("caption", b"original")

    assert response.content == "bytes-ok"


@pytest.mark.unit
def test_chat_with_image_wraps_timeout_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req: object, timeout: int) -> DummyResponse:
        raise TimeoutError("timed out")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    llm = LLMFactory.create_vision_llm(make_settings())

    with pytest.raises(ValueError, match="azure vision request failed: timeout"):
        llm.chat_with_image("caption", _PNG_1X1)


@pytest.mark.unit
def test_chat_with_image_wraps_authentication_errors_with_azure_code(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_urlopen(req: object, timeout: int) -> DummyResponse:
        payload = io.BytesIO(b'{"error": {"code": "Unauthorized", "message": "bad key"}}')
        raise error.HTTPError(getattr(req, "full_url"), 401, "Unauthorized", hdrs=None, fp=payload)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    llm = LLMFactory.create_vision_llm(make_settings())

    with pytest.raises(ValueError, match="azure vision request failed: http_error 401 Unauthorized"):
        llm.chat_with_image("caption", _PNG_1X1)
