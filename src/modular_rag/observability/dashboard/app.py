"""Streamlit multi-page Dashboard for Modular RAG MCP Server."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the src/ directory is on sys.path so dashboard pages can import project modules.
_SRC_DIR = Path(__file__).parents[3]  # project root
_SRC_SRC_DIR = _SRC_DIR / "src"
for _p in (_SRC_DIR, _SRC_SRC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import streamlit as st  # noqa: E402

_PAGES_DIR = Path(__file__).parent / "pages"

st.set_page_config(
    page_title="Modular RAG MCP Server",
    page_icon="🔍",
    layout="wide",
)

pages = {
    "System": [
        st.Page(str(_PAGES_DIR / "overview.py"), title="System Overview", icon="🏠", default=True),
    ],
    "Data": [
        st.Page(str(_PAGES_DIR / "data_browser.py"), title="Data Browser", icon="📊"),
    ],
    "Ingestion": [
        st.Page(str(_PAGES_DIR / "ingestion_manager.py"), title="Ingestion Manager", icon="⚙️"),
        st.Page(str(_PAGES_DIR / "ingestion_traces.py"), title="Ingestion Traces", icon="🔍"),
    ],
    "Query": [
        st.Page(str(_PAGES_DIR / "query_traces.py"), title="Query Traces", icon="💬"),
    ],
    "Evaluation": [
        st.Page(str(_PAGES_DIR / "evaluation_panel.py"), title="Evaluation Panel", icon="📈"),
    ],
}

pg = st.navigation(pages)
pg.run()
