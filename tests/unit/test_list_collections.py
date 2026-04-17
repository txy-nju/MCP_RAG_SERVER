"""Unit tests for MCP list_collections tool (E4)."""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_server.tools.list_collections import build_list_collections_tool_handler, list_collections


@pytest.mark.unit
def test_list_collections_discovers_directories_and_counts_files(tmp_path: Path) -> None:
    documents_root = tmp_path / "documents"
    alpha = documents_root / "alpha"
    beta = documents_root / "beta"
    alpha.mkdir(parents=True)
    beta.mkdir(parents=True)
    (alpha / "doc1.md").write_text("a", encoding="utf-8")
    (alpha / "nested").mkdir()
    (alpha / "nested" / "doc2.txt").write_text("b", encoding="utf-8")
    (beta / "readme.md").write_text("c", encoding="utf-8")

    collections = list_collections(documents_root)

    assert [item.name for item in collections] == ["alpha", "beta"]
    assert collections[0].document_count == 2
    assert collections[1].document_count == 1


@pytest.mark.unit
def test_list_collections_returns_empty_when_documents_root_missing(tmp_path: Path) -> None:
    missing_root = tmp_path / "not-exists"

    collections = list_collections(missing_root)

    assert collections == []


@pytest.mark.unit
def test_list_collections_handler_returns_mcp_payload(tmp_path: Path) -> None:
    documents_root = tmp_path / "documents"
    (documents_root / "engineering").mkdir(parents=True)
    (documents_root / "engineering" / "design.md").write_text("design", encoding="utf-8")

    handler = build_list_collections_tool_handler(documents_root=documents_root)

    result = handler({})

    assert result["content"][0]["type"] == "text"
    assert "engineering" in result["content"][0]["text"]
    assert result["structuredContent"]["collections"] == [
        {
            "name": "engineering",
            "description": "Collection 'engineering' with 1 document(s).",
            "document_count": 1,
        }
    ]


@pytest.mark.unit
def test_list_collections_handler_rejects_arguments(tmp_path: Path) -> None:
    handler = build_list_collections_tool_handler(documents_root=tmp_path / "documents")

    with pytest.raises(ValueError, match="does not accept arguments"):
        handler({"unexpected": True})
