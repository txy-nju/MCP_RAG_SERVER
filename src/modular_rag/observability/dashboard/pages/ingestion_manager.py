"""Ingestion manager page for uploading, ingesting, and deleting documents."""

from __future__ import annotations

from typing import Any

import streamlit as st

from modular_rag.observability.dashboard.services.ingestion_service import IngestionService


def _build_progress_callback(progress_bar: Any, status_box: Any) -> Any:
	def on_progress(stage_name: str, current: int, total: int) -> None:
		progress_bar.progress(int(current / total * 100))
		status_box.caption(f"{stage_name}: {current}/{total}")

	return on_progress


def _render_result(result: Any) -> None:
	if getattr(result, "skipped", False):
		st.warning("File unchanged. Existing ingestion artifacts were reused.")
		return

	st.success("Ingestion completed successfully.")
	col1, col2, col3 = st.columns(3)
	col1.metric("Chunks", int(getattr(result, "chunk_count", 0)))
	col2.metric("Records", int(getattr(result, "record_count", 0)))
	col3.metric("Images", int(getattr(result, "saved_image_count", 0)))


def _render_document_actions(service: IngestionService, collection_filter: str | None) -> None:
	rows = service.list_documents(collection=collection_filter)
	st.subheader("Existing Documents")
	if not rows:
		st.info("No ingested documents available for deletion.")
		return

	for row in rows:
		col1, col2, col3, col4, col5 = st.columns([5, 2, 2, 2, 2])
		col1.write(row.source_path)
		col2.write(row.collection)
		col3.write(f"{row.chunk_count} chunks")
		col4.write(f"{row.image_count} images")
		if col5.button("Delete", key=f"delete::{row.collection}::{row.source_path}"):
			result = service.delete_document(row.source_path, collection=row.collection)
			if getattr(result, "success", False):
				st.success(f"Deleted: {row.source_path}")
				st.rerun()
			else:
				st.warning(f"Nothing deleted for: {row.source_path}")


st.title("⚙️ Ingestion Manager")
st.markdown("Upload PDF files into a collection, monitor live progress, and delete ingested documents.")

try:
	service = IngestionService()
	known_collections = service.list_collections()
	default_collection = service.default_collection or "default"

	st.subheader("Upload And Ingest")
	uploaded_file = st.file_uploader("PDF File", type=["pdf"])
	collection = st.selectbox(
		"Target Collection",
		options=known_collections or [default_collection],
		index=(known_collections.index(default_collection) if default_collection in known_collections else 0),
	)
	force = st.checkbox("Force re-ingestion even if unchanged", value=False)

	if st.button("Start Ingestion", type="primary", disabled=uploaded_file is None):
		if uploaded_file is None:
			st.warning("Please select a PDF file before starting ingestion.")
		else:
			progress_bar = st.progress(0)
			status_box = st.empty()
			callback = _build_progress_callback(progress_bar, status_box)
			result = service.ingest_uploaded_file(
				uploaded_file,
				collection=collection,
				force=force,
				on_progress=callback,
			)
			progress_bar.progress(100)
			status_box.caption("done")
			_render_result(result)

	st.divider()
	filter_options = ["All collections"] + known_collections if known_collections else ["All collections"]
	selected_filter = st.selectbox("Filter Deletion List", filter_options, index=0)
	active_filter = None if selected_filter == "All collections" else selected_filter
	_render_document_actions(service, active_filter)

except Exception as exc:  # noqa: BLE001
	st.error(f"Failed to load ingestion manager: {exc}")
