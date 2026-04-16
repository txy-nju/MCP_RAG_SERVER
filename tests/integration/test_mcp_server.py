"""Integration tests for MCP server stdio behavior (E1)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


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
