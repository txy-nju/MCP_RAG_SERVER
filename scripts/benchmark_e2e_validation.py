"""End-to-end validation benchmark: retrieval quality + latency percentiles.

Runs all queries from a golden test set through the full
HybridSearch → Reranker pipeline, records per-query latency via TraceContext,
evaluates hit_rate / MRR, then emits a combined JSON report with P50/P95/P99.

Typical usage
-------------
python scripts/benchmark_e2e_validation.py \\
    --settings config/settings.perf.yaml \\
    --collection perf_10k \\
    --test-set tests/fixtures/golden_test_set.json \\
    --warmup 5 \\
    --output-report logs/benchmark_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from core.query_engine import DenseRetriever, HybridSearch, QueryProcessor, Reranker, SparseRetriever
from core.settings import Settings, load_settings
from core.trace.trace_context import TraceContext
from ingestion.storage.bm25_indexer import BM25Indexer
from libs.embedding.base_embedding import BaseEmbedding
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.evaluator.custom_evaluator import CustomEvaluator
from libs.vector_store.vector_store_factory import VectorStoreFactory
from observability.logger import get_logger


# ---------------------------------------------------------------------------
# Fallback deterministic embedding (for placeholder API keys)
# ---------------------------------------------------------------------------

class _LocalDeterministicEmbedding(BaseEmbedding):
    """Fallback embedding used when API settings are placeholders."""

    def embed(self, texts: list[str], trace: object | None = None) -> list[list[float]]:
        del trace
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vec = [round(digest[i % len(digest)] / 255.0, 6) for i in range(1536)]
            vectors.append(vec)
        return vectors


def _maybe_enable_local_embedding_fallback(settings: Settings, logger: object) -> None:
    provider = str(settings.embedding.provider or "").strip().lower()
    api_key = str(settings.embedding.api_key or "").strip().lower()
    api_url = str(settings.embedding.api_url or "").strip().lower()

    if provider not in {"openai", "azure"}:
        return
    if api_key != "your-api-key" and api_url != "your-api-url":
        return

    EmbeddingFactory.register("local_fake", _LocalDeterministicEmbedding)
    settings.embedding.provider = "local_fake"
    settings.embedding.model = "deterministic-local"
    settings.embedding.api_key = None
    settings.embedding.api_url = None
    logger.warning(  # type: ignore[attr-defined]
        "Detected placeholder embedding API settings. Using local deterministic embedding fallback."
    )


# ---------------------------------------------------------------------------
# Statistics helpers (no numpy dependency)
# ---------------------------------------------------------------------------

def _percentile(sorted_values: list[float], p: float) -> float:
    """Return the p-th percentile (0–100) from a pre-sorted list."""
    if not sorted_values:
        return 0.0
    k = (len(sorted_values) - 1) * p / 100.0
    lo = int(k)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = k - lo
    return round(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac, 3)


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0, "count": 0}
    s = sorted(values)
    return {
        "min": round(s[0], 3),
        "p50": _percentile(s, 50),
        "p95": _percentile(s, 95),
        "p99": _percentile(s, 99),
        "max": round(s[-1], 3),
        "mean": round(sum(s) / len(s), 3),
        "count": len(s),
    }


# ---------------------------------------------------------------------------
# Component factory
# ---------------------------------------------------------------------------

def _build_components(settings: Settings) -> tuple[HybridSearch, Reranker]:
    embedding_client = EmbeddingFactory.create(settings)
    vector_store = VectorStoreFactory.create(settings)
    bm25_indexer = BM25Indexer()
    if bm25_indexer.index_path.exists():
        bm25_indexer.load()

    query_processor = QueryProcessor()
    dense_retriever = DenseRetriever(
        embedding_client=embedding_client,
        vector_store=vector_store,
        settings=settings,
    )
    sparse_retriever = SparseRetriever(
        bm25_indexer=bm25_indexer,
        vector_store=vector_store,
        settings=settings,
    )

    hybrid_search = HybridSearch(
        query_processor=query_processor,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        settings=settings,
    )
    reranker = Reranker(settings=settings)
    return hybrid_search, reranker


# ---------------------------------------------------------------------------
# Test case loader
# ---------------------------------------------------------------------------

def _load_test_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    raw = payload.get("test_cases")
    if not isinstance(raw, list) or not raw:
        raise ValueError("golden test set must contain a non-empty 'test_cases' list")

    cases: list[dict] = []
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"test_cases[{i}] must be an object")
        query = str(item.get("query", "")).strip()
        expected_chunk_ids = item.get("expected_chunk_ids", [])
        expected_sources = item.get("expected_sources", [])
        if not query:
            raise ValueError(f"test_cases[{i}].query is required")
        if not isinstance(expected_chunk_ids, list) or not expected_chunk_ids:
            raise ValueError(f"test_cases[{i}].expected_chunk_ids must be a non-empty list")
        cases.append({
            "query": query,
            "expected_chunk_ids": [str(c) for c in expected_chunk_ids],
            "expected_sources": [str(s) for s in (expected_sources or [])],
        })
    return cases


# ---------------------------------------------------------------------------
# Single-query benchmark run (returns latency_ms + per-query metrics)
# ---------------------------------------------------------------------------

def _run_query(
    query: str,
    hybrid_search: HybridSearch,
    reranker: Reranker,
    evaluator: CustomEvaluator,
    expected_chunk_ids: list[str],
    collection: str | None,
    top_k: int,
    settings: Settings,
) -> tuple[float, dict[str, float], list[str]]:
    """Execute one query through the full pipeline; return (latency_ms, metrics, retrieved_ids)."""

    filters: dict = {}
    effective_collection = collection or settings.vector_store.collection
    if effective_collection:
        filters["collection"] = effective_collection

    trace = TraceContext(trace_type="query.benchmark")

    results = hybrid_search.search(
        query=query,
        top_k=top_k,
        filters=filters or None,
        trace=trace,
    )

    reranker.rerank(query=query, candidates=results, trace=trace)

    trace.finish()
    latency_ms = trace.total_elapsed_ms or 0.0

    retrieved_ids = [r.chunk_id for r in results]
    metrics = evaluator.evaluate(query, retrieved_ids, expected_chunk_ids)

    return latency_ms, metrics, retrieved_ids


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "End-to-end benchmark: measure retrieval quality (hit_rate, MRR) "
            "and latency percentiles (P50/P95/P99) over a golden test set."
        )
    )
    parser.add_argument(
        "--test-set",
        default="tests/fixtures/golden_test_set.json",
        help="Path to golden test set JSON (default: tests/fixtures/golden_test_set.json).",
    )
    parser.add_argument(
        "--settings",
        default="config/settings.yaml",
        help="Path to settings YAML file (default: config/settings.yaml).",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Collection filter passed to retrieval (default: from settings).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of results to retrieve per query (default: from settings.retrieval.top_k).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warm-up queries before measurement (default: 5).",
    )
    parser.add_argument(
        "--output-report",
        default="logs/benchmark_report.json",
        help="Where to write the JSON report (default: logs/benchmark_report.json).",
    )
    parser.add_argument(
        "--latency-target-ms",
        type=float,
        default=1000.0,
        help="P99 latency target in ms for pass/fail verdict (default: 1000).",
    )
    parser.add_argument(
        "--hit-rate-target",
        type=float,
        default=0.7,
        help="hit_rate@k target for pass/fail verdict (default: 0.70).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    logger = get_logger(__name__)

    # --- Settings ----------------------------------------------------------
    try:
        settings = load_settings(args.settings)
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load settings: %s", exc)
        return 1

    if args.collection:
        settings.vector_store.collection = args.collection

    _maybe_enable_local_embedding_fallback(settings, logger)

    top_k: int = args.top_k if args.top_k and args.top_k > 0 else settings.retrieval.top_k

    # --- Test set ----------------------------------------------------------
    test_set_path = Path(args.test_set)
    if not test_set_path.exists():
        logger.error("Golden test set not found: %s", test_set_path)
        return 1

    try:
        cases = _load_test_cases(test_set_path)
    except ValueError as exc:
        logger.error("Invalid test set: %s", exc)
        return 1

    logger.info("Loaded %d test cases from %s", len(cases), test_set_path)

    # --- Build components --------------------------------------------------
    try:
        hybrid_search, reranker = _build_components(settings)
        evaluator = CustomEvaluator()
    except Exception as exc:
        logger.error("Component initialization failed: %s", exc)
        return 1

    effective_collection = args.collection or settings.vector_store.collection

    # --- Warm-up -----------------------------------------------------------
    warmup_count = min(args.warmup, len(cases))
    if warmup_count > 0:
        logger.info("Running %d warm-up queries (results discarded)...", warmup_count)
        for case in cases[:warmup_count]:
            try:
                _run_query(
                    query=case["query"],
                    hybrid_search=hybrid_search,
                    reranker=reranker,
                    evaluator=evaluator,
                    expected_chunk_ids=case["expected_chunk_ids"],
                    collection=effective_collection,
                    top_k=top_k,
                    settings=settings,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Warm-up query failed (ignored): %s", exc)

    # --- Benchmark loop ----------------------------------------------------
    logger.info("Starting benchmark over %d queries (top_k=%d)...", len(cases), top_k)

    latencies: list[float] = []
    all_metrics: list[dict[str, float]] = []
    case_results: list[dict] = []
    errors: int = 0

    for idx, case in enumerate(cases, start=1):
        try:
            latency_ms, metrics, retrieved_ids = _run_query(
                query=case["query"],
                hybrid_search=hybrid_search,
                reranker=reranker,
                evaluator=evaluator,
                expected_chunk_ids=case["expected_chunk_ids"],
                collection=effective_collection,
                top_k=top_k,
                settings=settings,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%d/%d] Query failed: %s", idx, len(cases), exc)
            errors += 1
            continue

        latencies.append(latency_ms)
        all_metrics.append(metrics)
        case_results.append({
            "index": idx,
            "query": case["query"],
            "latency_ms": latency_ms,
            "metrics": metrics,
            "retrieved_chunk_ids": retrieved_ids,
            "expected_chunk_ids": case["expected_chunk_ids"],
        })

        if idx % 10 == 0 or idx == len(cases):
            logger.info(
                "[%d/%d] last_latency=%.1f ms  running_mean=%.1f ms",
                idx,
                len(cases),
                latency_ms,
                sum(latencies) / len(latencies),
            )

    if not latencies:
        logger.error("All queries failed; no results to report.")
        return 1

    # --- Aggregate metrics -------------------------------------------------
    agg_metrics: dict[str, float] = {}
    if all_metrics:
        for m in all_metrics:
            for k, v in m.items():
                agg_metrics[k] = agg_metrics.get(k, 0.0) + float(v)
        n = len(all_metrics)
        agg_metrics = {k: round(v / n, 4) for k, v in agg_metrics.items()}

    latency_stats = _stats(latencies)

    # --- Pass / fail verdict -----------------------------------------------
    p99_ok = latency_stats["p99"] <= args.latency_target_ms
    hit_rate_ok = agg_metrics.get("hit_rate", 0.0) >= args.hit_rate_target
    passed = p99_ok and hit_rate_ok

    # --- Build report ------------------------------------------------------
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "settings_file": str(args.settings),
        "test_set_file": str(test_set_path),
        "collection": effective_collection,
        "top_k": top_k,
        "warmup_queries": warmup_count,
        "total_queries": len(cases),
        "successful_queries": len(latencies),
        "failed_queries": errors,
        "latency_ms": latency_stats,
        "retrieval_metrics": agg_metrics,
        "targets": {
            "latency_p99_target_ms": args.latency_target_ms,
            "hit_rate_target": args.hit_rate_target,
        },
        "verdict": {
            "passed": passed,
            "p99_latency_ok": p99_ok,
            "hit_rate_ok": hit_rate_ok,
        },
        "cases": case_results,
    }

    # --- Write report file -------------------------------------------------
    output_path = Path(args.output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # --- Console summary ---------------------------------------------------
    sep = "=" * 70
    print(f"\n{sep}")
    print("  End-to-End Benchmark Report")
    print(sep)
    print(f"  Test set  : {test_set_path}  ({len(latencies)} / {len(cases)} queries OK)")
    print(f"  Settings  : {args.settings}")
    print(f"  Collection: {effective_collection}   top_k={top_k}")
    print()
    print("  Latency (ms)")
    print(f"    min  : {latency_stats['min']:>8.1f}")
    print(f"    mean : {latency_stats['mean']:>8.1f}")
    print(f"    P50  : {latency_stats['p50']:>8.1f}")
    print(f"    P95  : {latency_stats['p95']:>8.1f}")
    print(f"    P99  : {latency_stats['p99']:>8.1f}  (target ≤ {args.latency_target_ms:.0f} ms  {'✓ PASS' if p99_ok else '✗ FAIL'})")
    print(f"    max  : {latency_stats['max']:>8.1f}")
    print()
    print("  Retrieval Metrics")
    for metric_name, metric_value in agg_metrics.items():
        flag = ""
        if metric_name == "hit_rate":
            flag = f"  (target ≥ {args.hit_rate_target:.2f}  {'✓ PASS' if hit_rate_ok else '✗ FAIL'})"
        print(f"    {metric_name:<20}: {metric_value:.4f}{flag}")
    print()
    verdict_str = "✓ ALL TARGETS MET" if passed else "✗ SOME TARGETS MISSED"
    print(f"  Overall verdict: {verdict_str}")
    print(f"\n  Report written to: {output_path}")
    print(sep + "\n")

    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
