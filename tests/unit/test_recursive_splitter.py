"""Unit tests for the built-in recursive splitter."""

from __future__ import annotations

import pytest

from libs.splitter.recursive_splitter import RecursiveSplitter


class FakeRecursiveCharacterTextSplitter:
    """Minimal LangChain-compatible test double."""

    last_init_kwargs: dict[str, object] | None = None

    def __init__(self, **kwargs: object) -> None:
        self.__class__.last_init_kwargs = kwargs

    def split_text(self, text: str) -> list[str]:
        return [text[:5], "", "  ", text[5:]]


@pytest.mark.unit
def test_recursive_splitter_uses_langchain_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        RecursiveSplitter,
        "_load_recursive_text_splitter_class",
        staticmethod(lambda: FakeRecursiveCharacterTextSplitter),
    )
    splitter = RecursiveSplitter(provider="recursive", chunk_size=12, chunk_overlap=3)

    chunks = splitter.split_text("alpha beta")

    assert chunks == ["alpha", " beta"]
    assert FakeRecursiveCharacterTextSplitter.last_init_kwargs == {
        "chunk_size": 12,
        "chunk_overlap": 3,
        "separators": RecursiveSplitter.default_separators,
        "keep_separator": True,
        "length_function": len,
    }


@pytest.mark.unit
def test_recursive_splitter_preserves_markdown_structure_with_real_splitter() -> None:
    pytest.importorskip("langchain_text_splitters")
    markdown = """# Title\n\n## Section One\n\nParagraph one explains the topic in detail.\n\n```python\nprint('hello')\n```\n\n## Section Two\n\nParagraph two closes the document.\n"""
    splitter = RecursiveSplitter(provider="recursive", chunk_size=70, chunk_overlap=0)

    chunks = splitter.split_text(markdown)

    assert len(chunks) >= 3
    assert chunks[0].startswith("# Title")
    assert any(chunk.startswith("## Section One") for chunk in chunks)
    assert any("```python" in chunk for chunk in chunks)
    assert any(chunk.startswith("## Section Two") for chunk in chunks)


@pytest.mark.unit
def test_recursive_splitter_rejects_blank_text() -> None:
    splitter = RecursiveSplitter(provider="recursive", chunk_size=10, chunk_overlap=2)

    with pytest.raises(ValueError, match="recursive splitter failed: text must not be empty"):
        splitter.split_text("   ")


@pytest.mark.unit
def test_recursive_splitter_reports_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_missing_dependency() -> type[object]:
        raise ImportError("missing dependency")

    monkeypatch.setattr(
        RecursiveSplitter,
        "_load_recursive_text_splitter_class",
        staticmethod(raise_missing_dependency),
    )
    splitter = RecursiveSplitter(provider="recursive", chunk_size=10, chunk_overlap=2)

    with pytest.raises(ImportError, match="missing dependency"):
        splitter.split_text("hello world")
