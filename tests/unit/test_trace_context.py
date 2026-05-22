"""Unit tests for TraceContext and TraceCollector."""

from __future__ import annotations

import json
from typing import Any

import pytest

from modular_rag.core.trace.trace_collector import TraceCollector
from modular_rag.core.trace.trace_context import TraceContext
import core.trace.trace_context as trace_context_module


@pytest.mark.unit
def test_trace_context_defaults_to_query_type() -> None:
    trace = TraceContext()

    assert trace.trace_type == "query"
    assert trace.finished_at is None
    assert trace.total_elapsed_ms is None


@pytest.mark.unit
def test_trace_context_finish_and_to_dict_are_json_serializable(monkeypatch: pytest.MonkeyPatch) -> None:
    perf_values = iter([10.25, 10.5])
    monkeypatch.setattr(trace_context_module, "perf_counter", lambda: next(perf_values))

    trace = TraceContext(trace_type="ingestion")
    trace._t0 = 10.0
    trace.record_stage("load", {"method": "markitdown"})
    trace.finish()

    payload = trace.to_dict()

    assert payload["trace_id"] == trace.trace_id
    assert payload["trace_type"] == "ingestion"
    assert payload["started_at"] is not None
    assert payload["finished_at"] is not None
    assert payload["total_elapsed_ms"] == 500.0
    assert payload["stages"][0]["stage"] == "load"
    assert payload["stages"][0]["elapsed_ms"] == 250.0
    json.dumps(payload)


@pytest.mark.unit
def test_elapsed_ms_returns_stage_and_total_values(monkeypatch: pytest.MonkeyPatch) -> None:
    perf_values = iter([20.1, 20.4])
    monkeypatch.setattr(trace_context_module, "perf_counter", lambda: next(perf_values))

    trace = TraceContext(trace_type="query")
    trace._t0 = 20.0
    trace.record_stage("dense_retrieval", {"provider": "fake"})
    trace.finish()

    assert trace.elapsed_ms("dense_retrieval") == 100.0
    assert trace.elapsed_ms() == 400.0


@pytest.mark.unit
def test_elapsed_ms_raises_for_unknown_stage() -> None:
    trace = TraceContext(trace_type="query")

    with pytest.raises(KeyError, match="Unknown trace stage"):
        trace.elapsed_ms("missing")


@pytest.mark.unit
def test_trace_context_rejects_invalid_trace_type() -> None:
    with pytest.raises(ValueError, match="trace_type must start with 'query' or 'ingestion'"):
        TraceContext(trace_type="generic")


@pytest.mark.unit
def test_trace_collector_finishes_and_forwards_payload() -> None:
    written: list[dict[str, Any]] = []
    collector = TraceCollector(writer=written.append)
    trace = TraceContext(trace_type="query")
    trace.record_stage("fusion", {"method": "rrf"})

    collector.collect(trace)

    assert trace.finished_at is not None
    assert len(collector.collected) == 1
    assert collector.collected[0]["trace_type"] == "query"
    assert written[0]["stages"][0]["stage"] == "fusion"