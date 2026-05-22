"""Integration tests for MCP server stdio behavior (E1)."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from modular_rag.core.query_engine.reranker import RerankResult
from modular_rag.core.types import RetrievalResult
from modular_rag.mcp_server.protocol_handler import ProtocolHandler


@pytest.mark.integration
def test_mcp_server_initialize_over_stdio_without_stdout_pollution() -> None:
	"""Server should answer initialize via stdout and keep logs on stderr."""
	repo_root = Path(__file__).resolve().parents[2]
	src_dir = repo_root / "src"
	env = dict(os.environ)
	pythonpath_parts = [str(src_dir), str(repo_root)]
	existing_pythonpath = env.get("PYTHONPATH")
	if existing_pythonpath:
		pythonpath_parts.append(existing_pythonpath)
	env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

	request = {
		"jsonrpc": "2.0",
		"id": 1,
		"method": "initialize",
		"params": {
			"clientInfo": {"name": "pytest", "version": "1.0.0"},
			"protocolVersion": "2025-06-18",
		},
	}

	process = subprocess.Popen(
		[sys.executable, "-u", "-m", "mcp_server.server"],
		cwd=repo_root,
		stdin=subprocess.PIPE,
		stdout=subprocess.PIPE,
		stderr=subprocess.PIPE,
		text=True,
		env=env,
	)

	stdout, stderr = process.communicate(input=json.dumps(request) + "\n", timeout=10)

	assert process.returncode == 0
	stdout_lines = [line for line in stdout.splitlines() if line.strip()]
	assert len(stdout_lines) == 1

	response = json.loads(stdout_lines[0])
	assert response["jsonrpc"] == "2.0"
	assert response["id"] == 1
	assert response["result"]["serverInfo"]["name"] == "modular-rag-mcp-server"
	assert response["result"]["capabilities"]["tools"] == {}

	# stdout should only contain JSON-RPC payloads, never human-readable logs.
	assert "MCP server started" not in stdout
	assert "MCP server started" in stderr


@pytest.mark.integration
def test_query_knowledge_hub_tool_returns_markdown_and_citations(monkeypatch: pytest.MonkeyPatch) -> None:
	"""Protocol handler should route query_knowledge_hub and return cited MCP payload."""

	settings_stub = SimpleNamespace(retrieval=SimpleNamespace(top_k=3))

	class FakeHybridSearch:
		def search(self, query: str, top_k: int, filters: dict[str, object] | None = None) -> list[RetrievalResult]:
			assert query == "如何配置 Azure OpenAI"
			assert top_k == 2
			assert filters == {"collection": "default"}
			return [
				RetrievalResult(
					chunk_id="c1",
					score=0.9,
					text="在 settings.yaml 中设置 llm.provider 和 llm.model。",
					metadata={"source_path": "docs/setup.md", "page": 2},
				)
			]

	class FakeReranker:
		def rerank(self, query: str, candidates: list[RetrievalResult]) -> RerankResult:
			return RerankResult(candidates=candidates, fallback=False)

	monkeypatch.setattr("mcp_server.tools.query_knowledge_hub.load_settings", lambda _path: settings_stub)
	monkeypatch.setattr(
		"mcp_server.tools.query_knowledge_hub._build_components",
		lambda _settings: (FakeHybridSearch(), FakeReranker()),
	)

	handler = ProtocolHandler()
	response = handler.handle_request(
		{
			"jsonrpc": "2.0",
			"id": 99,
			"method": "tools/call",
			"params": {
				"name": "query_knowledge_hub",
				"arguments": {
					"query": "如何配置 Azure OpenAI",
					"top_k": 2,
					"collection": "default",
				},
			},
		}
	)

	assert "error" not in response
	result = response["result"]
	assert result["content"][0]["type"] == "text"
	assert "[1]" in result["content"][0]["text"]
	assert result["structuredContent"]["citations"][0]["source"] == "docs/setup.md"
	assert result["structuredContent"]["citations"][0]["chunk_id"] == "c1"


@pytest.mark.integration
def test_query_knowledge_hub_returns_image_content_when_chunk_has_image_refs(
	monkeypatch: pytest.MonkeyPatch,
	tmp_path: Path,
) -> None:
	"""When retrieval metadata includes image refs, response should include MCP ImageContent."""

	settings_stub = SimpleNamespace(retrieval=SimpleNamespace(top_k=3))
	image_path = tmp_path / "img-1.png"
	image_path.write_bytes(base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO6n5mQAAAAASUVORK5CYII="))

	class FakeHybridSearch:
		def search(self, query: str, top_k: int, filters: dict[str, object] | None = None) -> list[RetrievalResult]:
			assert query == "embedded image document with images"
			assert top_k == 1
			assert filters == {"collection": "default"}
			return [
				RetrievalResult(
					chunk_id="c-image-1",
					score=0.88,
					text="This chunk includes a referenced image.",
					metadata={
						"source_path": "fixtures/with_images.pdf",
						"page": 1,
						"image_refs": ["img-1"],
						"images": [{"id": "img-1", "path": str(image_path), "text_offset": 0, "text_length": 0}],
					},
				)
			]

	class FakeReranker:
		def rerank(self, query: str, candidates: list[RetrievalResult]) -> RerankResult:
			return RerankResult(candidates=candidates, fallback=False)

	monkeypatch.setattr("mcp_server.tools.query_knowledge_hub.load_settings", lambda _path: settings_stub)
	monkeypatch.setattr(
		"mcp_server.tools.query_knowledge_hub._build_components",
		lambda _settings: (FakeHybridSearch(), FakeReranker()),
	)

	handler = ProtocolHandler()
	response = handler.handle_request(
		{
			"jsonrpc": "2.0",
			"id": 100,
			"method": "tools/call",
			"params": {
				"name": "query_knowledge_hub",
				"arguments": {
					"query": "embedded image document with images",
					"top_k": 1,
					"collection": "default",
				},
			},
		}
	)

	assert "error" not in response
	content = response["result"]["content"]
	assert content[0]["type"] == "text"
	assert any(item.get("type") == "image" for item in content)

	image_item = next(item for item in content if item.get("type") == "image")
	assert image_item["mimeType"] == "image/png"
	assert isinstance(image_item["data"], str) and image_item["data"]
	assert isinstance(base64.b64decode(image_item["data"]), bytes)
