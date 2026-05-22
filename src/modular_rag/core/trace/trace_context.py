"""Trace context utilities for ingestion/query stage instrumentation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from time import perf_counter
from typing import Any
from uuid import uuid4


_VALID_TRACE_TYPE_PREFIXES = ("query", "ingestion")


@dataclass(slots=True)
class TraceContext:
	"""Request-scoped trace context for query and ingestion flows."""

	trace_id: str = field(default_factory=lambda: str(uuid4()))
	trace_type: str = "query"
	started_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
	finished_at: str | None = None
	total_elapsed_ms: float | None = None
	_t0: float = field(default_factory=perf_counter)
	stages: list[dict[str, Any]] = field(default_factory=list)

	def __post_init__(self) -> None:
		trace_type = self.trace_type.strip()
		if not trace_type or not trace_type.startswith(_VALID_TRACE_TYPE_PREFIXES):
			raise ValueError(
				"trace_type must start with 'query' or 'ingestion', "
				f"got: {self.trace_type}"
			)
		self.trace_type = trace_type

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

	def finish(self) -> None:
		"""Mark the trace as finished and compute total elapsed time."""

		if self.finished_at is not None and self.total_elapsed_ms is not None:
			return

		self.finished_at = datetime.now(UTC).isoformat()
		self.total_elapsed_ms = round((perf_counter() - self._t0) * 1000, 3)

	def elapsed_ms(self, stage_name: str | None = None) -> float:
		"""Return total elapsed time or the elapsed value recorded for a stage."""

		if stage_name is None:
			if self.total_elapsed_ms is not None:
				return self.total_elapsed_ms
			return round((perf_counter() - self._t0) * 1000, 3)

		for stage in reversed(self.stages):
			if stage.get("stage") == stage_name:
				return float(stage.get("elapsed_ms", 0.0))
		raise KeyError(f"Unknown trace stage: {stage_name}")

	def to_dict(self) -> dict[str, Any]:
		"""Serialize trace context into a JSON-friendly payload."""

		return {
			"trace_id": self.trace_id,
			"trace_type": self.trace_type,
			"started_at": self.started_at,
			"finished_at": self.finished_at,
			"total_elapsed_ms": self.total_elapsed_ms,
			"stages": list(self.stages),
		}
