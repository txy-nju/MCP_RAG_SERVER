"""Unit tests for JSON Lines trace logging."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from core.trace.trace_collector import TraceCollector
from core.trace.trace_context import TraceContext
from observability.logger import JSONFormatter, get_trace_logger, write_trace


@pytest.mark.unit
def test_json_formatter_serializes_dict_payload() -> None:
    formatter = JSONFormatter()
    record = logging.LogRecord(
        name="trace-test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg={"trace_id": "trace-1", "trace_type": "query"},
        args=(),
        exc_info=None,
    )

    payload = json.loads(formatter.format(record))

    assert payload["trace_id"] == "trace-1"
    assert payload["trace_type"] == "query"


@pytest.mark.unit
def test_write_trace_appends_json_line(tmp_path: Path) -> None:
    trace_file = tmp_path / "logs" / "traces.jsonl"

    write_trace(
        {
            "trace_id": "trace-1",
            "trace_type": "ingestion",
            "started_at": "2026-04-17T00:00:00+00:00",
            "finished_at": "2026-04-17T00:00:01+00:00",
            "total_elapsed_ms": 1000.0,
            "stages": [],
        },
        trace_file=trace_file,
    )

    lines = trace_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1

    payload = json.loads(lines[0])
    assert payload["trace_type"] == "ingestion"
    assert payload["trace_id"] == "trace-1"


@pytest.mark.unit
def test_get_trace_logger_reuses_existing_handler_for_same_file(tmp_path: Path) -> None:
    trace_file = tmp_path / "logs" / "traces.jsonl"

    logger_a = get_trace_logger(trace_file=trace_file)
    logger_b = get_trace_logger(trace_file=trace_file)

    assert logger_a is logger_b
    assert len(logger_a.handlers) == 1


@pytest.mark.unit
def test_trace_collector_persists_with_default_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trace_file = tmp_path / "logs" / "traces.jsonl"

    monkeypatch.setattr(
        "core.trace.trace_collector.write_trace",
        lambda payload: write_trace(payload, trace_file=trace_file),
    )

    collector = TraceCollector()
    trace = TraceContext(trace_type="query")
    trace.record_stage("fusion", {"method": "rrf"})

    collector.collect(trace)

    lines = trace_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    payload = json.loads(lines[0])
    assert payload["trace_type"] == "query"
    assert payload["stages"][0]["stage"] == "fusion"