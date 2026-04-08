"""Trace context utilities for ingestion/query stage instrumentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import uuid4


@dataclass(slots=True)
class TraceContext:
	"""Minimal trace context used before full observability phase is implemented."""

	trace_id: str = field(default_factory=lambda: str(uuid4()))
	trace_type: str = "generic"
	started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
	_t0: float = field(default_factory=perf_counter)
	stages: list[dict[str, Any]] = field(default_factory=list)

	def record_stage(self, stage: str, data: dict[str, Any] | None = None) -> None:
		"""Append a stage-level event to the in-memory trace timeline."""

		self.stages.append(
			{
				"stage": str(stage),
				"timestamp": datetime.now(UTC).isoformat(),
				"elapsed_ms": round((perf_counter() - self._t0) * 1000, 3),
				"data": dict(data or {}),
			}
		)

	def to_dict(self) -> dict[str, Any]:
		"""Serialize trace context into a JSON-friendly payload."""

		return {
			"trace_id": self.trace_id,
			"trace_type": self.trace_type,
			"started_at": self.started_at,
			"stages": list(self.stages),
		}
