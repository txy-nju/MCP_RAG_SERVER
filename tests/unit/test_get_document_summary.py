"""Unit tests for MCP get_document_summary tool (E5)."""

from __future__ import annotations

from pathlib import Path

import pytest

import mcp_server.tools.get_document_summary as summary_module
from mcp_server.tools.get_document_summary import (
    build_get_document_summary_tool_handler,
    get_document_summary,
)


@pytest.mark.unit
def test_get_document_summary_returns_structured_info_for_existing_doc(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    documents_root = tmp_path / "documents"
    target = documents_root / "default" / "simple.pdf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"%PDF-1.4 fake fixture")

    monkeypatch.setattr(
        summary_module,
        "_extract_document_text",
        lambda _path: "Sample Document\nThis fixture explains the loader and retrieval flow.",
    )

    result = get_document_summary("simple.pdf", documents_root=documents_root)

    assert result.doc_id == "simple.pdf"
    assert "Sample Document" in result.title
    assert "loader and retrieval flow" in result.summary
    assert "pdf" in result.tags
    assert result.collection == "default"


@pytest.mark.unit
def test_get_document_summary_raises_for_missing_doc_id(tmp_path: Path) -> None:
    documents_root = tmp_path / "documents"
    documents_root.mkdir(parents=True)

    with pytest.raises(ValueError, match="doc_id not found"):
        get_document_summary("missing.pdf", documents_root=documents_root)


@pytest.mark.unit
def test_get_document_summary_handler_returns_mcp_payload(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    documents_root = tmp_path / "documents"
    target = documents_root / "default" / "simple.pdf"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"%PDF-1.4 fake fixture")

    monkeypatch.setattr(
        summary_module,
        "_extract_document_text",
        lambda _path: "Sample Document\nA concise summary body for MCP response.",
    )

    handler = build_get_document_summary_tool_handler(documents_root=documents_root)
    payload = handler({"doc_id": "simple.pdf"})

    assert payload["content"][0]["type"] == "text"
    assert "Sample Document" in payload["content"][0]["text"]
    assert payload["structuredContent"]["doc_id"] == "simple.pdf"
    assert payload["structuredContent"]["collection"] == "default"


@pytest.mark.unit
def test_get_document_summary_handler_validates_doc_id_argument(tmp_path: Path) -> None:
    handler = build_get_document_summary_tool_handler(documents_root=tmp_path / "documents")

    with pytest.raises(ValueError, match="doc_id must be a non-empty string"):
        handler({"doc_id": ""})
