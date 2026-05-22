"""JSON-RPC protocol handler for MCP server core methods."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from modular_rag.mcp_server.tools.get_document_summary import build_get_document_summary_tool_handler
from modular_rag.mcp_server.tools.list_collections import build_list_collections_tool_handler
from modular_rag.mcp_server.tools.query_knowledge_hub import build_query_tool_handler


JSONRPC_VERSION = "2.0"
PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "modular-rag-mcp-server"
SERVER_VERSION = "0.1.0"


class JsonRpcError(Exception):
	"""JSON-RPC error carrying standard error code and public message."""

	def __init__(self, code: int, message: str) -> None:
		super().__init__(message)
		self.code = int(code)
		self.message = str(message)


@dataclass(slots=True)
class ToolDefinition:
	"""Tool schema and callable binding used by tools/list and tools/call."""

	name: str
	description: str
	input_schema: dict[str, Any]
	handler: Callable[[dict[str, Any]], dict[str, Any]]


class ProtocolHandler:
	"""Handle MCP JSON-RPC requests for initialize/tools/list/tools/call."""

	def __init__(
		self,
		tools: list[ToolDefinition] | None = None,
		server_name: str = SERVER_NAME,
		server_version: str = SERVER_VERSION,
		protocol_version: str = PROTOCOL_VERSION,
	) -> None:
		self.server_name = str(server_name)
		self.server_version = str(server_version)
		self.protocol_version = str(protocol_version)

		registered = tools if tools is not None else self._default_tools()
		self._tools: dict[str, ToolDefinition] = {tool.name: tool for tool in registered}

	def _default_tools(self) -> list[ToolDefinition]:
		"""Register placeholder schemas for current E phase tools."""

		def _placeholder_tool(arguments: dict[str, Any]) -> dict[str, Any]:
			return {
				"content": [
					{
						"type": "text",
						"text": "Tool is not implemented yet.",
					}
				]
			}

		return [
			ToolDefinition(
				name="query_knowledge_hub",
				description="Search the knowledge hub and return cited snippets.",
				input_schema={
					"type": "object",
					"properties": {
						"query": {"type": "string"},
						"top_k": {"type": "integer", "minimum": 1},
						"collection": {"type": "string"},
					},
					"required": ["query"],
				},
				handler=build_query_tool_handler(),
			),
			ToolDefinition(
				name="list_collections",
				description="List available knowledge collections.",
				input_schema={
					"type": "object",
					"properties": {},
				},
				handler=build_list_collections_tool_handler(),
			),
			ToolDefinition(
				name="get_document_summary",
				description="Get summary metadata for a document.",
				input_schema={
					"type": "object",
					"properties": {
						"doc_id": {"type": "string"},
					},
					"required": ["doc_id"],
				},
				handler=build_get_document_summary_tool_handler(),
			),
		]

	def handle_initialize(self, params: dict[str, Any] | None) -> dict[str, Any]:
		"""Handle initialize request and return capability negotiation payload."""
		if params is not None and not isinstance(params, dict):
			raise JsonRpcError(-32602, "Invalid params")

		return {
			"protocolVersion": self.protocol_version,
			"serverInfo": {
				"name": self.server_name,
				"version": self.server_version,
			},
			"capabilities": {
				"tools": {},
			},
		}

	def handle_tools_list(self) -> dict[str, Any]:
		"""Return JSON schemas for all registered tools."""
		tools_payload = [
			{
				"name": tool.name,
				"description": tool.description,
				"inputSchema": tool.input_schema,
			}
			for tool in self._tools.values()
		]
		return {"tools": tools_payload}

	def handle_tools_call(self, name: str, arguments: dict[str, Any] | None) -> dict[str, Any]:
		"""Route tools/call to registered tool handlers with safe error mapping."""
		tool = self._tools.get(name)
		if tool is None:
			raise JsonRpcError(-32602, "Invalid params")

		if arguments is None:
			parsed_arguments: dict[str, Any] = {}
		elif isinstance(arguments, dict):
			parsed_arguments = dict(arguments)
		else:
			raise JsonRpcError(-32602, "Invalid params")

		try:
			result = tool.handler(parsed_arguments)
			if not isinstance(result, dict):
				raise JsonRpcError(-32603, "Internal error")
			return result
		except (ValueError, TypeError) as exc:
			raise JsonRpcError(-32602, "Invalid params") from exc
		except JsonRpcError:
			raise
		except Exception as exc:  # pragma: no cover - covered through handle_request mapping
			raise JsonRpcError(-32603, "Internal error") from exc

	def handle_request(self, payload: Any) -> dict[str, Any]:
		"""Parse and dispatch a JSON-RPC request to MCP methods."""
		request_id: Any = None
		try:
			if not isinstance(payload, dict):
				raise JsonRpcError(-32600, "Invalid Request")

			request_id = payload.get("id")
			if payload.get("jsonrpc") != JSONRPC_VERSION:
				raise JsonRpcError(-32600, "Invalid Request")

			method = payload.get("method")
			if not isinstance(method, str):
				raise JsonRpcError(-32600, "Invalid Request")

			params = payload.get("params")

			if method == "initialize":
				result = self.handle_initialize(params if isinstance(params, dict) else params)
			elif method == "tools/list":
				if params not in (None, {}):
					raise JsonRpcError(-32602, "Invalid params")
				result = self.handle_tools_list()
			elif method == "tools/call":
				if not isinstance(params, dict):
					raise JsonRpcError(-32602, "Invalid params")

				name = params.get("name")
				if not isinstance(name, str) or not name.strip():
					raise JsonRpcError(-32602, "Invalid params")

				arguments = params.get("arguments")
				result = self.handle_tools_call(name, arguments if isinstance(arguments, dict) else arguments)
			else:
				raise JsonRpcError(-32601, "Method not found")

			return {
				"jsonrpc": JSONRPC_VERSION,
				"id": request_id,
				"result": result,
			}

		except JsonRpcError as exc:
			return {
				"jsonrpc": JSONRPC_VERSION,
				"id": request_id,
				"error": {
					"code": exc.code,
					"message": exc.message,
				},
			}
		except Exception:
			return {
				"jsonrpc": JSONRPC_VERSION,
				"id": request_id,
				"error": {
					"code": -32603,
					"message": "Internal error",
				},
			}


__all__ = ["ProtocolHandler", "ToolDefinition", "JsonRpcError"]
