"""Unit tests for MCP response building and citation generation."""

from __future__ import annotations

from core.response.response_builder import ResponseBuilder
from core.types import RetrievalResult


def test_response_builder_builds_markdown_and_structured_citations() -> None:
    builder = ResponseBuilder()
    results = [
        RetrievalResult(
            chunk_id="chunk-1",
            score=0.92,
            text="Azure OpenAI can be configured with endpoint and deployment name.",
            metadata={"source_path": "docs/azure.md", "page": 3},
        ),
        RetrievalResult(
            chunk_id="chunk-2",
            score=0.87,
            text="Use HybridSearch to fuse dense and sparse retrieval signals.",
            metadata={"source_path": "docs/retrieval.md", "page": 8},
        ),
    ]

    payload = builder.build(retrieval_results=results, query="如何配置 Azure OpenAI")

    text_block = payload["content"][0]["text"]
    assert "[1]" in text_block
    assert "[2]" in text_block

    citations = payload["structuredContent"]["citations"]
    assert len(citations) == 2
    assert citations[0]["source"] == "docs/azure.md"
    assert citations[0]["page"] == 3
    assert citations[0]["chunk_id"] == "chunk-1"
    assert isinstance(citations[0]["score"], float)


def test_response_builder_returns_friendly_message_for_empty_results() -> None:
    builder = ResponseBuilder()

    payload = builder.build(retrieval_results=[], query="不存在的问题")

    assert payload["content"][0]["type"] == "text"
    assert "未检索到与问题相关的内容" in payload["content"][0]["text"]
    assert payload["structuredContent"]["citations"] == []
    assert payload["structuredContent"]["results"] == []
