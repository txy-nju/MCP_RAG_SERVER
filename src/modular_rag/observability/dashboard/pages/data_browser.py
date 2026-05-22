"""Data Browser page for browsing documents, chunks, and linked images."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from modular_rag.observability.dashboard.services.data_service import DataService


def _render_document_table(rows: list[dict[str, Any]]) -> None:
	if not rows:
		st.info("No ingested documents were found for this collection.")
		return
	st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_chunk_details(chunks: list[dict[str, Any]]) -> None:
	st.subheader("Chunk Details")
	if not chunks:
		st.info("No chunks available for the selected document.")
		return

	for chunk in chunks:
		metadata = dict(chunk.get("metadata", {}))
		chunk_id = str(chunk.get("chunk_id", ""))
		chunk_index = metadata.get("chunk_index", "?")
		label = f"Chunk {chunk_index} | {chunk_id[:12]}"

		with st.expander(label, expanded=False):
			st.text_area("Content", str(chunk.get("text", "")), height=180)
			st.json(metadata)


def _render_images(images: list[dict[str, Any]]) -> None:
	st.subheader("Image Preview")
	if not images:
		st.caption("No indexed images for the selected document.")
		return

	cols = st.columns(2)
	for idx, image in enumerate(images):
		with cols[idx % 2]:
			image_path = str(image.get("file_path", ""))
			image_id = str(image.get("image_id", ""))
			st.caption(f"{image_id} | {image_path}")
			if image_path and Path(image_path).exists():
				st.image(image_path, use_container_width=True)
			else:
				st.warning("Image file not found on disk.")


st.title("📊 Data Browser")
st.markdown("Browse ingested documents, inspect chunk metadata, and preview linked images.")

try:
	service = DataService()
	collections = service.list_collections()
	collection_options = ["All collections"] + collections
	selected_collection = st.selectbox("Collection", collection_options, index=0)

	active_collection = None if selected_collection == "All collections" else selected_collection
	rows = service.list_documents(collection=active_collection)

	st.subheader("Documents")
	total_docs = len(rows)
	total_chunks = sum(row.chunk_count for row in rows)
	total_images = sum(row.image_count for row in rows)
	col_a, col_b, col_c = st.columns(3)
	col_a.metric("总文档数", total_docs)
	col_b.metric("总 Chunk 数", total_chunks)
	col_c.metric("总图片数", total_images)

	table_rows = [
		{
			"source_path": row.source_path,
			"collection": row.collection,
			"chunks": row.chunk_count,
			"images": row.image_count,
			"file_hash": row.file_hash or "",
		}
		for row in rows
	]
	_render_document_table(table_rows)

	if rows:
		options = [row.source_path for row in rows]
		selected_doc = st.selectbox("Select Document", options, index=0)

		detail = service.get_document_detail(selected_doc)
		col1, col2, col3 = st.columns(3)
		col1.metric("Chunks", int(detail["chunk_count"]))
		col2.metric("Images", int(detail["image_count"]))
		col3.metric("Doc Hash", str(detail.get("file_hash") or "N/A"))

		chunks = service.get_chunks(selected_doc, collection=active_collection)
		_render_chunk_details(chunks)

		images = service.get_images(collection=active_collection, doc_hash=detail.get("file_hash"))
		_render_images(images)

except Exception as exc:  # noqa: BLE001
	st.error(f"Failed to load data browser: {exc}")
