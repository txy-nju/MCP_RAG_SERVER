"""Azure OpenAI vision-capable LLM provider implementation."""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import os
from typing import TYPE_CHECKING, Any
from urllib import error, request

from libs.llm.base_vision_llm import BaseVisionLLM, ChatResponse, VisionImageInput

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


class AzureVisionLLM(BaseVisionLLM):
    """Azure OpenAI multimodal chat provider implementation."""

    provider_label = "azure"
    timeout_seconds = 30

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        api_key: str,
        api_url: str,
        api_version: str,
        deployment_name: str,
        max_image_size: int = 2048,
        timeout_seconds: int = 30,
    ) -> None:
        super().__init__(provider=provider, model=model)
        self.api_key = api_key
        self.api_url = api_url
        self.api_version = api_version
        self.deployment_name = deployment_name
        self.max_image_size = max_image_size
        self.timeout_seconds = timeout_seconds

    @classmethod
    def from_settings(cls, settings: Any) -> "AzureVisionLLM":
        endpoint = (
            getattr(settings, "endpoint", None)
            or getattr(settings, "azure_endpoint", None)
            or os.getenv("AZURE_OPENAI_ENDPOINT", "")
        )
        api_version = getattr(settings, "api_version", None) or os.getenv(
            "AZURE_OPENAI_API_VERSION", "2024-02-15-preview"
        )
        deployment_name = getattr(settings, "deployment_name", None) or str(settings.model)
        api_url = getattr(settings, "api_url", None) or os.getenv(
            "AZURE_OPENAI_VISION_API_URL",
            cls._build_api_url(endpoint=endpoint, deployment=deployment_name, api_version=str(api_version)),
        )
        api_key = getattr(settings, "api_key", None) or os.getenv("AZURE_OPENAI_API_KEY", "")
        max_image_size = int(
            getattr(settings, "max_image_size", None)
            or os.getenv("AZURE_OPENAI_VISION_MAX_IMAGE_SIZE", "2048")
        )
        timeout_seconds = int(
            getattr(settings, "timeout_seconds", None)
            or os.getenv("AZURE_OPENAI_VISION_TIMEOUT_SECONDS", str(cls.timeout_seconds))
        )
        if not api_key:
            raise ValueError("azure vision configuration error: missing API key")
        if not api_url:
            raise ValueError("azure vision configuration error: missing API URL")
        return cls(
            provider=str(settings.provider),
            model=str(settings.model),
            api_key=str(api_key),
            api_url=str(api_url),
            api_version=str(api_version),
            deployment_name=str(deployment_name),
            max_image_size=max_image_size,
            timeout_seconds=timeout_seconds,
        )

    @staticmethod
    def _build_api_url(endpoint: str, deployment: str, api_version: str) -> str:
        cleaned_endpoint = endpoint.rstrip("/")
        if not cleaned_endpoint:
            return ""
        return (
            f"{cleaned_endpoint}/openai/deployments/{deployment}"
            f"/chat/completions?api-version={api_version}"
        )

    def chat_with_image(
        self,
        text: str,
        image_path: VisionImageInput,
        trace: TraceContext | None = None,
    ) -> ChatResponse:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("azure vision request failed: text prompt must be a non-empty string")

        data_url = self._build_data_url(image_path)
        payload = self._build_payload(text.strip(), data_url)
        response_payload = self._post_json(payload)
        return ChatResponse(
            content=self._extract_text_response(response_payload),
            provider=self.provider,
            model=self.model,
            raw_response=response_payload,
            metadata={
                "api_version": self.api_version,
                "deployment_name": self.deployment_name,
                "trace_provided": trace is not None,
            },
        )

    def preprocess_image(self, image_path: VisionImageInput) -> bytes:
        image_bytes = self._load_image_bytes(image_path)
        return self._resize_image_bytes_if_needed(image_bytes)

    def _build_data_url(self, image_path: VisionImageInput) -> str:
        image_bytes = self.preprocess_image(image_path)
        mime_type = self._detect_mime_type(image_path)
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    def _load_image_bytes(self, image_path: VisionImageInput) -> bytes:
        if isinstance(image_path, bytes):
            return image_path
        with open(image_path, "rb") as handle:
            return handle.read()

    def _detect_mime_type(self, image_path: VisionImageInput) -> str:
        if isinstance(image_path, str):
            guessed_type, _ = mimetypes.guess_type(image_path)
            if guessed_type:
                return guessed_type
        return "image/png"

    def _resize_image_bytes_if_needed(self, image_bytes: bytes) -> bytes:
        if self.max_image_size <= 0 or Image is None:
            return image_bytes

        with Image.open(io.BytesIO(image_bytes)) as image:
            max_dimension = max(image.size)
            if max_dimension <= self.max_image_size:
                return image_bytes

            resized = image.copy()
            resized.thumbnail((self.max_image_size, self.max_image_size))
            buffer = io.BytesIO()
            output_format = image.format or "PNG"
            save_format = "PNG" if output_format.upper() not in {"PNG", "JPEG", "WEBP", "GIF", "BMP"} else output_format
            resized.save(buffer, format=save_format)
            return buffer.getvalue()

    def _build_payload(self, text: str, data_url: str) -> dict[str, Any]:
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ]
        }

    def _post_json(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self.api_url,
            data=body,
            headers={"Content-Type": "application/json", "api-key": self.api_key},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                raw_response = response.read().decode("utf-8")
        except error.HTTPError as exc:
            azure_code = self._extract_azure_error_code(exc)
            raise ValueError(f"azure vision request failed: http_error {exc.code} {azure_code}") from exc
        except error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise ValueError(f"azure vision request failed: connection_error {reason}") from exc
        except TimeoutError as exc:
            raise ValueError("azure vision request failed: timeout") from exc

        try:
            return json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise ValueError("azure vision request failed: invalid_json_response") from exc

    def _extract_text_response(self, payload: dict[str, Any]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("azure vision request failed: missing choices in response")

        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise ValueError("azure vision request failed: missing message in response")

        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            text_parts = [item.get("text", "") for item in content if isinstance(item, dict)]
            merged = "".join(part for part in text_parts if part)
            if merged:
                return merged

        raise ValueError("azure vision request failed: empty response content")

    def _extract_azure_error_code(self, exc: error.HTTPError) -> str:
        try:
            body = exc.read().decode("utf-8")
        except Exception:  # pragma: no cover - defensive fallback
            return f"http_status_{exc.code}"
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            return f"http_status_{exc.code}"
        error_payload = payload.get("error")
        if not isinstance(error_payload, dict):
            return f"http_status_{exc.code}"
        code = error_payload.get("code")
        if isinstance(code, str) and code.strip():
            return code
        return f"http_status_{exc.code}"
