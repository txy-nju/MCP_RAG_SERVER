"""Contract tests for PdfLoader."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from libs.loader.pdf_loader import PdfLoader
import libs.loader.pdf_loader as pdf_loader_module


class _FakeImage:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self.data = data


class _FakePage:
    def __init__(self, text: str, images: list[_FakeImage] | None = None) -> None:
        self._text = text
        self._images = images or []

    def extract_text(self) -> str:
        return self._text

    @property
    def images(self) -> list[_FakeImage]:
        return self._images


class _ExplodingImagePage(_FakePage):
    @property
    def images(self) -> list[_FakeImage]:
        raise RuntimeError("broken image extractor")


class _FakeReader:
    def __init__(self, pages: list[Any]) -> None:
        self.pages = pages


def _fixture_pdf(name: str) -> Path:
    return Path(__file__).resolve().parent.parent / "fixtures" / "sample_documents" / name


@pytest.mark.unit
def test_pdf_loader_returns_document_with_source_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pdf_loader_module, "PdfReader", lambda _path: _FakeReader([_FakePage("hello from pdf")]))

    loader = PdfLoader(image_root=tmp_path / "images")
    doc = loader.load(str(_fixture_pdf("simple.pdf")))

    assert doc.id
    assert doc.text == "hello from pdf"
    assert doc.metadata["source_path"].endswith("simple.pdf")
    assert doc.metadata["doc_type"] == "pdf"
    assert doc.metadata.get("images", []) == []


@pytest.mark.unit
def test_pdf_loader_extracts_images_and_inserts_placeholders(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_pages = [
        _FakePage(
            "intro section",
            images=[_FakeImage("chart.png", b"fake-image-binary")],
        )
    ]
    monkeypatch.setattr(pdf_loader_module, "PdfReader", lambda _path: _FakeReader(fake_pages))

    loader = PdfLoader(image_root=tmp_path / "images")
    doc = loader.load(str(_fixture_pdf("with_images.pdf")))

    assert "[IMAGE:" in doc.text
    assert len(doc.metadata["images"]) == 1

    image_meta = doc.metadata["images"][0]
    placeholder = f"[IMAGE: {image_meta['id']}]"
    assert doc.text[image_meta["text_offset"] : image_meta["text_offset"] + image_meta["text_length"]] == placeholder
    assert image_meta["page"] == 1
    assert Path(image_meta["path"]).exists()


@pytest.mark.unit
def test_pdf_loader_degrades_when_image_extraction_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        pdf_loader_module,
        "PdfReader",
        lambda _path: _FakeReader([_ExplodingImagePage("text still available")]),
    )

    loader = PdfLoader(image_root=tmp_path / "images")
    doc = loader.load(str(_fixture_pdf("with_images.pdf")))

    assert doc.text == "text still available"
    assert doc.metadata.get("images", []) == []
