"""Trace data service for Dashboard ingestion/query trace pages."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.settings import Settings, load_settings

_DEFAULT_CONFIG_PATH = str(Path(__file__).parents[4] / "config" / "settings.yaml")


class TraceService:
	"""Read and filter persisted trace logs for dashboard rendering."""

	def __init__(
		self,
		*,
		settings: Settings | None = None,
		config_path: str | None = None,
		trace_file: str | Path | None = None,
	) -> None:
		if trace_file is not None:
			self._trace_file = Path(trace_file)
		else:
			loaded_settings = settings or load_settings(config_path or _DEFAULT_CONFIG_PATH)
			self._trace_file = Path(loaded_settings.observability.trace_file)

	@property
	def trace_file(self) -> Path:
		return self._trace_file

	def list_traces(self, *, trace_type_prefix: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
		"""Return traces newest-first, optionally filtered by trace type prefix."""

		if limit <= 0:
			raise ValueError("limit must be greater than 0")

		rows = self._load_rows()
		if trace_type_prefix is not None:
			prefix = str(trace_type_prefix).strip()
			rows = [row for row in rows if str(row.get("trace_type", "")).startswith(prefix)]

		rows.sort(key=self._sort_key, reverse=True)
		return rows[:limit]

	def list_ingestion_traces(self, *, limit: int = 200) -> list[dict[str, Any]]:
		"""Return ingestion traces newest-first."""

		return self.list_traces(trace_type_prefix="ingestion", limit=limit)

	def get_trace(self, trace_id: str) -> dict[str, Any] | None:
		"""Return one trace row by trace_id, or None if absent."""

		normalized_trace_id = str(trace_id).strip()
		if not normalized_trace_id:
			return None

		for row in self._load_rows():
			if str(row.get("trace_id", "")).strip() == normalized_trace_id:
				return row
		return None

	def summarize_trace(self, trace: dict[str, Any]) -> dict[str, Any]:
		"""Build compact summary fields used by dashboard table views."""

		stages = trace.get("stages", [])
		stage_names: list[str] = []
		if isinstance(stages, list):
			for stage in stages:
				if isinstance(stage, dict):
					stage_names.append(str(stage.get("stage", "")))

		return {
			"trace_id": str(trace.get("trace_id", "")),
			"trace_type": str(trace.get("trace_type", "")),
			"started_at": trace.get("started_at"),
			"finished_at": trace.get("finished_at"),
			"total_elapsed_ms": trace.get("total_elapsed_ms"),
			"stage_count": len(stage_names),
			"last_stage": stage_names[-1] if stage_names else "",
		}

	def _load_rows(self) -> list[dict[str, Any]]:
		if not self._trace_file.exists():
			return []

		rows: list[dict[str, Any]] = []
		for line in self._trace_file.read_text(encoding="utf-8").splitlines():
			stripped = line.strip()
			if not stripped:
				continue
			try:
				payload = json.loads(stripped)
			except json.JSONDecodeError:
				continue
			if isinstance(payload, dict):
				rows.append(payload)
		return rows

	@staticmethod
	def _sort_key(trace: dict[str, Any]) -> str:
		finished_at = str(trace.get("finished_at", "") or "")
		started_at = str(trace.get("started_at", "") or "")
		return finished_at or started_at


__all__ = ["TraceService"]
