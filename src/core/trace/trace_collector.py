"""Trace collection helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core.trace.trace_context import TraceContext


class TraceCollector:
	"""Collect trace payloads and optionally forward them to a persistence sink."""

	def __init__(
		self,
		*,
		writer: Callable[[dict[str, Any]], None] | None = None,
	) -> None:
		self._writer = writer
		self._collected: list[dict[str, Any]] = []

	def collect(self, trace: TraceContext) -> None:
		"""Finalize a trace payload and forward it to the configured writer."""

		trace.finish()
		payload = trace.to_dict()
		self._collected.append(payload)
		if self._writer is not None:
			self._writer(payload)

	@property
	def collected(self) -> list[dict[str, Any]]:
		"""Return collected trace payloads for inspection in tests or callers."""

		return list(self._collected)


__all__ = ["TraceCollector"]
