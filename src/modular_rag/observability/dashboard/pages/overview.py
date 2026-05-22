"""System Overview page — displays component configuration and data statistics."""

from __future__ import annotations

import streamlit as st

from modular_rag.observability.dashboard.services.config_service import ComponentCard, ConfigService


def _render_component_card(card: ComponentCard) -> None:
    """Render a single component card inside a Streamlit column."""
    with st.container(border=True):
        st.markdown(f"**{card.name}**")
        st.caption(f"Provider: `{card.provider}`")
        for key, value in card.details.items():
            st.text(f"{key}: {value}")


def _render_data_stats(settings: object) -> None:
    """Try to connect to ChromaStore and render basic collection statistics."""
    st.subheader("Data Statistics")
    try:
        from modular_rag.libs.vector_store.chroma_store import ChromaStore  # type: ignore[import]

        store = ChromaStore.from_settings(settings.vector_store)  # type: ignore[attr-defined]
        stats = store.get_collection_stats()
        col1, col2 = st.columns(2)
        col1.metric("Collection", stats["collection"])
        col2.metric("Chunks", stats["chunk_count"])
    except Exception as exc:  # noqa: BLE001
        st.info(f"Data statistics unavailable: {exc}")


st.title("🏠 System Overview")
st.markdown("Current configuration and data asset summary for the Modular RAG MCP Server.")

try:
    svc = ConfigService()
    settings = svc.get_settings()
    cards = svc.get_component_cards()

    st.subheader("Component Configuration")
    cols = st.columns(min(len(cards), 3))
    for idx, card in enumerate(cards):
        with cols[idx % 3]:
            _render_component_card(card)

    st.divider()
    _render_data_stats(settings)

except Exception as exc:  # noqa: BLE001
    st.error(f"Failed to load configuration: {exc}")
