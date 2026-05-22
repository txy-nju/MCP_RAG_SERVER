"""E2E dashboard smoke tests using Streamlit AppTest (I2)."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from modular_rag.ingestion.storage.image_storage import ImageStorage
from modular_rag.libs.loader.file_integrity import SQLiteIntegrityChecker
from modular_rag.libs.vector_store.base_vector_store import VectorStoreRecord
from modular_rag.libs.vector_store.chroma_store import ChromaStore


_PNG_1X1_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
    "/x8AAwMCAO6n5mQAAAAASUVORK5CYII="
)


def _seed_dashboard_data() -> None:
    store = ChromaStore(provider="chroma", collection="default", persist_path="data/db/chroma")
    store.upsert(
        [
            VectorStoreRecord(
                id="dashboard-smoke-chunk-1",
                embedding=[0.1, 0.2, 0.3],
                metadata={
                    "source_path": "tests/fixtures/sample_documents/dashboard_smoke.pdf",
                    "collection": "default",
                    "chunk_index": 0,
                    "page": 1,
                    "image_refs": ["img-dashboard-1"],
                    "images": [
                        {
                            "id": "img-dashboard-1",
                            "path": "data/images/default/img-dashboard-1.png",
                            "text_offset": 0,
                            "text_length": 0,
                        }
                    ],
                },
                text="Dashboard smoke test chunk content.",
            )
        ]
    )

    checker = SQLiteIntegrityChecker()
    checker.mark_success(
        "dashboard-smoke-hash-1",
        "tests/fixtures/sample_documents/dashboard_smoke.pdf",
    )

    image_storage = ImageStorage()
    image_storage.save_image(
        "img-dashboard-1",
        base64.b64decode(_PNG_1X1_BASE64),
        collection="default",
        doc_hash="dashboard-smoke-hash-1",
        page_num=1,
    )


def _seed_trace_logs() -> None:
    trace_file = Path("logs/traces.jsonl")
    trace_file.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "trace_id": "ingestion-smoke-1",
            "trace_type": "ingestion.pipeline",
            "started_at": "2026-04-20T10:00:00+00:00",
            "finished_at": "2026-04-20T10:00:01+00:00",
            "total_elapsed_ms": 1000.0,
            "stages": [
                {"stage": "load", "elapsed_ms": 120.0, "timestamp": "2026-04-20T10:00:00+00:00"},
                {"stage": "upsert", "elapsed_ms": 200.0, "timestamp": "2026-04-20T10:00:01+00:00"},
            ],
        },
        {
            "trace_id": "query-smoke-1",
            "trace_type": "query.pipeline",
            "started_at": "2026-04-20T10:01:00+00:00",
            "finished_at": "2026-04-20T10:01:02+00:00",
            "total_elapsed_ms": 2000.0,
            "stages": [
                {
                    "stage": "query_processing",
                    "elapsed_ms": 50.0,
                    "timestamp": "2026-04-20T10:01:00+00:00",
                },
                {"stage": "fusion", "elapsed_ms": 80.0, "timestamp": "2026-04-20T10:01:02+00:00"},
            ],
        },
    ]
    trace_file.write_text("\n".join(json.dumps(row, ensure_ascii=True) for row in rows) + "\n", encoding="utf-8")


@pytest.mark.e2e
def test_dashboard_pages_smoke_render_without_exceptions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Render all 6 dashboard pages with seeded data and assert no runtime exceptions."""

    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    app_test_class = streamlit_testing.AppTest

    monkeypatch.chdir(tmp_path)
    _seed_dashboard_data()
    _seed_trace_logs()

    repo_root = Path(__file__).resolve().parents[2]
    pages_dir = repo_root / "src" / "observability" / "dashboard" / "pages"
    page_files = [
        pages_dir / "overview.py",
        pages_dir / "data_browser.py",
        pages_dir / "ingestion_manager.py",
        pages_dir / "ingestion_traces.py",
        pages_dir / "query_traces.py",
        pages_dir / "evaluation_panel.py",
    ]

    for page_file in page_files:
        at = app_test_class.from_file(str(page_file), default_timeout=20)
        at.run()
        assert len(at.exception) == 0, f"{page_file.name} raised exception(s): {at.exception}"
