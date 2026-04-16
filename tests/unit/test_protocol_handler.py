"""Unit tests for ProtocolHandler JSON-RPC routing (E2)."""

from __future__ import annotations

import pytest

from mcp_server.protocol_handler import ProtocolHandler, ToolDefinition


@pytest.fixture
def handler() -> ProtocolHandler:
	return ProtocolHandler()


def test_initialize_returns_server_info_and_capabilities(handler: ProtocolHandler) -> None:
	response = handler.handle_request(
		{
			"jsonrpc": "2.0",
			"id": 1,
			"method": "initialize",
			"params": {"clientInfo": {"name": "pytest", "version": "1.0.0"}},
		}
	)

	assert response["jsonrpc"] == "2.0"
	assert response["id"] == 1
	assert response["result"]["serverInfo"]["name"] == "modular-rag-mcp-server"
	assert response["result"]["capabilities"]["tools"] == {}


def test_tools_list_returns_registered_tool_schemas(handler: ProtocolHandler) -> None:
	response = handler.handle_request(
		{
			"jsonrpc": "2.0",
			"id": 2,
			"method": "tools/list",
			"params": {},
		}
	)

	tools = response["result"]["tools"]
	tool_names = {tool["name"] for tool in tools}
	assert {"query_knowledge_hub", "list_collections", "get_document_summary"}.issubset(tool_names)
	assert all("inputSchema" in tool for tool in tools)


def test_tools_call_routes_to_registered_handler() -> None:
	def fake_tool(arguments: dict[str, object]) -> dict[str, object]:
		return {"echo": arguments.get("query")}

	handler = ProtocolHandler(
		tools=[
			ToolDefinition(
				name="query_knowledge_hub",
				description="fake",
				input_schema={"type": "object"},
				handler=fake_tool,
			)
		]
	)

	response = handler.handle_request(
		{
			"jsonrpc": "2.0",
			"id": 3,
			"method": "tools/call",
			"params": {"name": "query_knowledge_hub", "arguments": {"query": "hello"}},
		}
	)

	assert response["result"] == {"echo": "hello"}


def test_invalid_method_returns_method_not_found(handler: ProtocolHandler) -> None:
	response = handler.handle_request(
		{
			"jsonrpc": "2.0",
			"id": 4,
			"method": "tools/unknown",
			"params": {},
		}
	)

	assert response["error"]["code"] == -32601
	assert response["error"]["message"] == "Method not found"


def test_invalid_request_returns_invalid_request(handler: ProtocolHandler) -> None:
	response = handler.handle_request(["not-an-object"])

	assert response["id"] is None
	assert response["error"]["code"] == -32600
	assert response["error"]["message"] == "Invalid Request"


def test_tools_call_invalid_params_returns_invalid_params(handler: ProtocolHandler) -> None:
	response = handler.handle_request(
		{
			"jsonrpc": "2.0",
			"id": 5,
			"method": "tools/call",
			"params": {"name": "query_knowledge_hub", "arguments": "bad"},
		}
	)

	assert response["error"]["code"] == -32602
	assert response["error"]["message"] == "Invalid params"


def test_tools_call_unknown_tool_returns_invalid_params(handler: ProtocolHandler) -> None:
	response = handler.handle_request(
		{
			"jsonrpc": "2.0",
			"id": 6,
			"method": "tools/call",
			"params": {"name": "missing", "arguments": {}},
		}
	)

	assert response["error"]["code"] == -32602
	assert response["error"]["message"] == "Invalid params"


def test_internal_error_is_mapped_without_traceback() -> None:
	def crash_tool(arguments: dict[str, object]) -> dict[str, object]:
		raise RuntimeError("secret traceback detail")

	handler = ProtocolHandler(
		tools=[
			ToolDefinition(
				name="query_knowledge_hub",
				description="crash",
				input_schema={"type": "object"},
				handler=crash_tool,
			)
		]
	)

	response = handler.handle_request(
		{
			"jsonrpc": "2.0",
			"id": 7,
			"method": "tools/call",
			"params": {"name": "query_knowledge_hub", "arguments": {}},
		}
	)

	assert response["error"]["code"] == -32603
	assert response["error"]["message"] == "Internal error"
	assert "traceback" not in str(response).lower()
	assert "secret" not in str(response).lower()
