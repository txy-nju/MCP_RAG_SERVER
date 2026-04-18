"""Unit tests for Dashboard TraceService (G5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from observability.dashboard.services.trace_service import TraceService


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.mark.unit
def test_list_ingestion_traces_filters_and_sorts_by_latest_timestamp(tmp_path: Path) -> None:
    trace_file = tmp_path / "logs" / "traces.jsonl"
    _write_lines(
        trace_file,
        [
            json.dumps(
                {
                    "trace_id": "q-1",
                    "trace_type": "query",
                    "started_at": "2026-04-18T09:00:00+00:00",
                    "finished_at": "2026-04-18T09:00:01+00:00",
                    "stages": [],
                }
            ),
            "{broken-json",
            json.dumps(
                {
                    "trace_id": "i-1",
                    "trace_type": "ingestion",
                    "started_at": "2026-04-18T10:00:00+00:00",
                    "finished_at": "2026-04-18T10:00:02+00:00",
                    "stages": [{"stage": "load"}],
                }
            ),
            json.dumps(
                {
                    "trace_id": "i-2",
                    "trace_type": "ingestion.pipeline",
                    "started_at": "2026-04-18T11:00:00+00:00",
                    "finished_at": "2026-04-18T11:00:03+00:00",
                    "stages": [{"stage": "upsert"}],
                }
            ),
        ],
    )

    service = TraceService(trace_file=trace_file)

    rows = service.list_ingestion_traces(limit=10)

    assert [row["trace_id"] for row in rows] == ["i-2", "i-1"]


@pytest.mark.unit
def test_get_trace_returns_matching_payload(tmp_path: Path) -> None:
    trace_file = tmp_path / "logs" / "traces.jsonl"
    _write_lines(
        trace_file,
        [
            json.dumps(
                {
                    "trace_id": "ing-42",
                    "trace_type": "ingestion",
                    "started_at": "2026-04-18T12:00:00+00:00",
                    "finished_at": "2026-04-18T12:00:02+00:00",
                    "total_elapsed_ms": 2000.0,
                    "stages": [{"stage": "load"}, {"stage": "upsert"}],
                }
            )
        ],
    )

    service = TraceService(trace_file=trace_file)

    row = service.get_trace("ing-42")

    assert row is not None
    assert row["trace_type"] == "ingestion"
    summary = service.summarize_trace(row)
    assert summary["stage_count"] == 2
    assert summary["last_stage"] == "upsert"


@pytest.mark.unit
def test_list_traces_rejects_non_positive_limit(tmp_path: Path) -> None:
    service = TraceService(trace_file=tmp_path / "logs" / "traces.jsonl")

    with pytest.raises(ValueError, match="limit"):
        service.list_ingestion_traces(limit=0)
