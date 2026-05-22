"""Recursive text splitter backed by LangChain."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from modular_rag.libs.splitter.base_splitter import BaseSplitter

if TYPE_CHECKING:
    from modular_rag.core.trace.trace_context import TraceContext


class RecursiveSplitter(BaseSplitter):
    """Markdown-friendly recursive splitter using LangChain semantics."""

    default_separators = [
        "\n# ",
        "\n## ",
        "\n### ",
        "\n#### ",
        "\n##### ",
        "\n###### ",
        "\n```",
        "\n\n",
        "\n",
        " ",
        "",
    ]

    def __init__(
        self,
        *,
        provider: str,
        chunk_size: int,
        chunk_overlap: int,
        separators: list[str] | None = None,
    ) -> None:
        """Initialize splitter configuration and markdown-aware separators."""

        super().__init__(provider=provider, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.separators = list(separators or self.default_separators)

    def split_text(self, text: str, trace: TraceContext | None = None) -> list[str]:
        """Split text into recursive chunks while preserving semantic boundaries."""

        del trace
        if not isinstance(text, str):
            raise ValueError(f"{self.provider} splitter failed: text must be a string")
        if not text.strip():
            raise ValueError(f"{self.provider} splitter failed: text must not be empty")

        splitter_cls = self._load_recursive_text_splitter_class()
        splitter = splitter_cls(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            keep_separator=True,
            length_function=len,
        )
        chunks = splitter.split_text(text)
        return [chunk for chunk in chunks if isinstance(chunk, str) and chunk.strip()]

    @classmethod
    def from_settings(cls, settings: Any) -> "RecursiveSplitter":
        """Build a recursive splitter instance from project settings."""

        return cls(
            provider=str(settings.provider),
            chunk_size=int(settings.chunk_size),
            chunk_overlap=int(settings.chunk_overlap),
        )

    @staticmethod
    def _load_recursive_text_splitter_class() -> type[Any]:
        """Load the LangChain splitter dependency only when needed."""

        try:
            module = importlib.import_module("langchain_text_splitters")
        except ImportError as exc:
            raise ImportError(
                "Recursive splitter requires 'langchain-text-splitters'. "
                "Install project dependencies before using provider 'recursive'."
            ) from exc

        splitter_cls = getattr(module, "RecursiveCharacterTextSplitter", None)
        if splitter_cls is None:
            raise ImportError(
                "Recursive splitter requires langchain_text_splitters.RecursiveCharacterTextSplitter."
            )
        return splitter_cls
