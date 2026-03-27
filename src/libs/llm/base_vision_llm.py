"""Base interfaces for pluggable vision-capable LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from core.trace.trace_context import TraceContext


VisionImageInput = str | bytes


@dataclass(slots=True)
class ChatResponse:
    """Provider-agnostic multimodal response payload."""

    content: str
    provider: str
    model: str
    raw_response: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseVisionLLM(ABC):
    """Common contract for providers that accept text plus a single image."""

    def __init__(self, *, provider: str, model: str) -> None:
        self.provider = provider
        self.model = model

    @abstractmethod
    def chat_with_image(
        self,
        text: str,
        image_path: VisionImageInput,
        trace: TraceContext | None = None,
    ) -> ChatResponse:
        """Generate a response from text plus an image input."""

    def preprocess_image(self, image_path: VisionImageInput) -> VisionImageInput:
        """Extension point for image compression or format conversion."""

        if isinstance(image_path, str):
            return str(Path(image_path))
        return image_path

    @classmethod
    def from_settings(cls, settings: Any) -> "BaseVisionLLM":
        """Build a vision LLM instance from a settings object."""

        return cls(provider=str(settings.provider), model=str(settings.model))
