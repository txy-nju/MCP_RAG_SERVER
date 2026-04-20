"""E2E MCP client simulation test for stdio server tool calls."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

from libs.vector_store.base_vector_store import VectorStoreRecord
from libs.vector_store.chroma_store import ChromaStore


class _MockEmbeddingHandler(BaseHTTPRequestHandler):
    """Return deterministic OpenAI-compatible embedding payloads."""

    server_version = "MockEmbedding/1.0"

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8")
        payload = json.loads(body)

        inputs = payload.get("input", [])
        if not isinstance(inputs, list):
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"input must be list"}')
            return

        data = [
            {
                "index": index,
                "embedding": [0.1, 0.2, 0.3],
            }
            for index, _ in enumerate(inputs)
        ]
        response = json.dumps({"object": "list", "data": data}).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def log_message(self, _format: str, *args: Any) -> None:
        del args


def _start_mock_embedding_server() -> tuple[ThreadingHTTPServer, threading.Thread, str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _MockEmbeddingHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address
    return server, thread, f"http://{host}:{port}/v1/embeddings"


def _seed_vector_store(cwd: Path) -> None:
    persist_path = cwd / "data" / "db" / "chroma"
    store = ChromaStore(provider="chroma", collection="default", persist_path=str(persist_path))
    store.upsert(
        [
            VectorStoreRecord(
                id="chunk-mcp-e2e-1",
                embedding=[0.1, 0.2, 0.3],
                metadata={
                    "source_path": "tests/fixtures/sample_documents/e2e_source.pdf",
                    "collection": "default",
                    "page": 1,
                },
                text="This chunk validates MCP E2E query_knowledge_hub citations.",
            )
        ]
    )


def _write_test_settings(cwd: Path, embedding_url: str) -> None:
    (cwd / "config" / "prompts").mkdir(parents=True, exist_ok=True)
    (cwd / "config" / "prompts" / "rerank.txt").write_text("Rerank disabled for E2E test.", encoding="utf-8")

    settings_yaml = "\n".join(
        [
            "llm:",
            "  provider: openai",
            "  model: gpt-4o-mini",
            "  api_key: test-key",
            "  api_url: http://127.0.0.1:1/v1/chat/completions",
            "embedding:",
            "  provider: openai",
            "  model: text-embedding-3-small",
            "  api_key: test-key",
            f"  api_url: {embedding_url}",
            "splitter:",
            "  provider: recursive",
            "  chunk_size: 1000",
            "  chunk_overlap: 200",
            "vector_store:",
            "  provider: chroma",
            "  collection: default",
            "  persist_path: data/db/chroma",
            "retrieval:",
            "  top_k: 5",
            "rerank:",
            "  provider: none",
            "  prompt_path: config/prompts/rerank.txt",
            "  max_candidates: 20",
            "evaluation:",
            "  backend: custom",
            "observability:",
            "  log_level: INFO",
            "  trace_file: logs/traces.jsonl",
        ]
    )
    (cwd / "config").mkdir(parents=True, exist_ok=True)
    (cwd / "config" / "settings.yaml").write_text(settings_yaml + "\n", encoding="utf-8")


def _run_mcp_requests(repo_root: Path, cwd: Path, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    env = dict(os.environ)
    pythonpath_parts = [str(repo_root / "src"), str(repo_root)]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    process = subprocess.Popen(
        [sys.executable, "-u", "-m", "mcp_server.server"],
        cwd=cwd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    input_payload = "\n".join(json.dumps(item, ensure_ascii=True) for item in requests) + "\n"
    stdout, stderr = process.communicate(input=input_payload, timeout=20)

    assert process.returncode == 0, stderr
    lines = [line for line in stdout.splitlines() if line.strip()]
    assert len(lines) == len(requests)
    return [json.loads(line) for line in lines]


@pytest.mark.e2e
def test_mcp_client_simulated_tools_list_and_call_returns_citations(tmp_path: Path) -> None:
    """Simulate an MCP client over stdio and verify query_knowledge_hub citations."""

    repo_root = Path(__file__).resolve().parents[2]
    test_cwd = tmp_path / "mcp_e2e_workspace"
    test_cwd.mkdir(parents=True, exist_ok=True)

    server, thread, embedding_url = _start_mock_embedding_server()
    try:
        _write_test_settings(test_cwd, embedding_url)
        _seed_vector_store(test_cwd)

        responses = _run_mcp_requests(
            repo_root,
            test_cwd,
            requests=[
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "clientInfo": {"name": "pytest-e2e", "version": "1.0.0"},
                        "protocolVersion": "2025-06-18",
                    },
                },
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "tools/list",
                    "params": {},
                },
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "query_knowledge_hub",
                        "arguments": {
                            "query": "mcp e2e citation query",
                            "top_k": 1,
                            "collection": "default",
                        },
                    },
                },
            ],
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    initialize_response, tools_list_response, tools_call_response = responses

    assert initialize_response["id"] == 1
    assert "error" not in initialize_response

    assert tools_list_response["id"] == 2
    tool_names = [item["name"] for item in tools_list_response["result"]["tools"]]
    assert "query_knowledge_hub" in tool_names

    assert tools_call_response["id"] == 3
    assert "error" not in tools_call_response
    result = tools_call_response["result"]
    assert result["content"][0]["type"] == "text"
    citations = result["structuredContent"]["citations"]
    assert len(citations) == 1
    assert citations[0]["chunk_id"] == "chunk-mcp-e2e-1"
    assert citations[0]["source"] == "tests/fixtures/sample_documents/e2e_source.pdf"
