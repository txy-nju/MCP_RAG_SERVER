"""Query traces page for query-pipeline diagnostics and trace inspection."""

from __future__ import annotations

from typing import Any

import streamlit as st

from modular_rag.observability.dashboard.services.trace_service import TraceService


def _render_query_metrics(rows: list[dict[str, Any]]) -> None:
	total = len(rows)
	elapsed_values: list[float] = []
	for row in rows:
		elapsed = row.get("total_elapsed_ms")
		if isinstance(elapsed, (int, float)):
			elapsed_values.append(float(elapsed))
	avg_elapsed = (sum(elapsed_values) / len(elapsed_values)) if elapsed_values else 0.0
	max_elapsed = max(elapsed_values) if elapsed_values else 0.0

	col1, col2, col3 = st.columns(3)
	col1.metric("Query Traces", total)
	col2.metric("Avg Elapsed (ms)", round(avg_elapsed, 2))
	col3.metric("Max Elapsed (ms)", round(max_elapsed, 2))


def _render_stage_table(stages: list[dict[str, Any]]) -> None:
	if not stages:
		st.caption("No stage data in selected trace.")
		return

	table_rows: list[dict[str, Any]] = []
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


st.title("💬 Query Traces")
st.markdown("Inspect query pipeline traces, stage timings, and raw trace payloads.")

try:
	service = TraceService()

	limit = st.sidebar.slider("Rows", min_value=10, max_value=500, value=200, step=10)
	rows = service.list_query_traces(limit=limit)
	summaries = [service.summarize_trace(row) for row in rows]

	if not rows:
		st.info("No query traces found. Run a query first, then refresh this page.")
	else:
		_render_query_metrics(summaries)
		st.subheader("Recent Query Traces")
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
	st.error(f"Failed to load query traces: {exc}")
