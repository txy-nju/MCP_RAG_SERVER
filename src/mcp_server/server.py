"""MCP server entrypoint with stdio JSON-RPC handling."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from typing import Any

from observability.logger import get_logger


SERVER_NAME = "modular-rag-mcp-server"
SERVER_VERSION = "0.1.0"
JSONRPC_VERSION = "2.0"
PROTOCOL_VERSION = "2025-06-18"


def _response_result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
	return {
		"jsonrpc": JSONRPC_VERSION,
		"id": request_id,
		"result": result,
	}


def _response_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
	return {
		"jsonrpc": JSONRPC_VERSION,
		"id": request_id,
		"error": {
			"code": code,
			"message": message,
		},
	}


def _handle_request(payload: Mapping[str, Any]) -> dict[str, Any]:
	request_id = payload.get("id")
	method = payload.get("method")

	if payload.get("jsonrpc") != JSONRPC_VERSION or not isinstance(method, str):
		return _response_error(request_id, -32600, "Invalid Request")

	if method != "initialize":
		return _response_error(request_id, -32601, "Method not found")

	params = payload.get("params")
	if params is not None and not isinstance(params, Mapping):
		return _response_error(request_id, -32602, "Invalid params")

	result = {
		"protocolVersion": PROTOCOL_VERSION,
		"serverInfo": {
			"name": SERVER_NAME,
			"version": SERVER_VERSION,
		},
		"capabilities": {
			"tools": {},
		},
	}
	return _response_result(request_id, result)


def main() -> int:
	"""Run the MCP stdio server loop."""
	logger = get_logger("mcp_server.server")
	logger.info("MCP server started (stdio transport)")

	for raw_line in sys.stdin:
		line = raw_line.strip()
		if not line:
			continue

		try:
			payload = json.loads(line)
			if not isinstance(payload, Mapping):
				raise ValueError("JSON payload must be an object")
			response = _handle_request(payload)
		except json.JSONDecodeError:
			response = _response_error(None, -32700, "Parse error")
		except Exception:
			response = _response_error(None, -32603, "Internal error")

		sys.stdout.write(json.dumps(response, ensure_ascii=True) + "\n")
		sys.stdout.flush()

	logger.info("MCP server stopped")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
