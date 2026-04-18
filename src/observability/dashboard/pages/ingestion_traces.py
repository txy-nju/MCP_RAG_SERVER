"""Ingestion traces page for timeline and stage-level diagnostics."""

from __future__ import annotations

from typing import Any

import streamlit as st

from observability.dashboard.services.trace_service import TraceService


def _render_trace_metrics(rows: list[dict[str, Any]]) -> None:
	total = len(rows)
	elapsed_values: list[float] = []
	for row in rows:
		elapsed = row.get("total_elapsed_ms")
		if isinstance(elapsed, (int, float)):
			elapsed_values.append(float(elapsed))
	avg_elapsed = (sum(elapsed_values) / len(elapsed_values)) if elapsed_values else 0.0
	max_stages = max((int(row.get("stage_count", 0)) for row in rows), default=0)

	col1, col2, col3 = st.columns(3)
	col1.metric("Ingestion Traces", total)
	col2.metric("Avg Elapsed (ms)", round(avg_elapsed, 2) if elapsed_values else 0.0)
	col3.metric("Max Stage Count", max_stages)


def _render_stage_table(stages: list[dict[str, Any]]) -> None:
	if not stages:
		st.caption("No stage data in selected trace.")
		return

	table_rows = []
	for index, stage in enumerate(stages, start=1):
		table_rows.append(
			{
				"#": index,
				"stage": stage.get("stage"),
				"elapsed_ms": stage.get("elapsed_ms"),
				"timestamp": stage.get("timestamp"),
				"data": stage.get("data"),
			}
		)
	st.dataframe(table_rows, use_container_width=True)


st.title("🔍 Ingestion Traces")
st.markdown("Inspect ingestion pipeline traces, stage timings, and raw trace payloads.")

try:
	service = TraceService()

	limit = st.sidebar.slider("Rows", min_value=10, max_value=500, value=200, step=10)
	rows = service.list_ingestion_traces(limit=limit)
	summaries = [service.summarize_trace(row) for row in rows]

	if not rows:
		st.info("No ingestion traces found. Run ingestion first, then refresh this page.")
	else:
		_render_trace_metrics(summaries)
		st.subheader("Recent Ingestion Traces")
		st.dataframe(summaries, use_container_width=True)

		trace_options = [summary["trace_id"] for summary in summaries if str(summary.get("trace_id", "")).strip()]
		if not trace_options:
			st.caption("No trace_id found in current rows.")
			st.stop()

		selected_trace_id = st.selectbox("Trace ID", options=trace_options, index=0)
		if not isinstance(selected_trace_id, str) or not selected_trace_id.strip():
			st.caption("Please select a valid trace id.")
			st.stop()

		selected = service.get_trace(selected_trace_id)
		if selected is not None:
			st.subheader("Stage Timeline")
			_render_stage_table(list(selected.get("stages", [])))

			st.subheader("Raw Trace")
			st.json(selected)
except Exception as exc:  # noqa: BLE001
	st.error(f"Failed to load ingestion traces: {exc}")
