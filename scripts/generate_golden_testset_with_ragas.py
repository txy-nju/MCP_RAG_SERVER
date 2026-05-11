"""Generate golden test set with Ragas TestsetGenerator.

This script:
1) Loads chunk corpus from the configured vector store collection.
2) Uses Ragas TestsetGenerator to synthesize queries.
3) Maps generated reference contexts back to chunk IDs.
4) Writes project-compatible golden test set JSON.

Output format:
{
  "test_cases": [
    {
      "query": "...",
      "expected_chunk_ids": ["chunk_xxx"],
      "expected_sources": ["source.pdf"]
    }
  ]
}
"""

from __future__ import annotations

import argparse
import json
import random
import re
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from core.query_engine import DenseRetriever, HybridSearch, QueryProcessor, SparseRetriever
from core.settings import Settings, load_settings
from ingestion.storage.bm25_indexer import BM25Indexer
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.vector_store.vector_store_factory import VectorStoreFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate golden testset using Ragas TestsetGenerator.")
    parser.add_argument("--settings", default="config/settings.yaml", help="Path to settings YAML.")
    parser.add_argument(
        "--collection",
        default=None,
        help="Collection name to read chunks from. Defaults to settings.vector_store.collection.",
    )
    parser.add_argument("--num-queries", type=int, default=200, help="Number of queries to generate.")
    parser.add_argument(
        "--output",
        default="tests/fixtures/golden_test_set_ragas_200.json",
        help="Output golden test set path.",
    )
    parser.add_argument(
        "--max-source-docs",
        type=int,
        default=1200,
        help="Max source chunks used to construct generation docs (sampled).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--fallback-top-k",
        type=int,
        default=1,
        help="Top-k used by retrieval fallback when context->chunk mapping fails.",
    )
    parser.add_argument(
        "--network-timeout-sec",
        type=float,
        default=8.0,
        help="Timeout in seconds for preflight endpoint connectivity checks.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="Max retry attempts for Ragas generation when transient network errors occur.",
    )
    parser.add_argument(
        "--retry-base-delay-sec",
        type=float,
        default=2.0,
        help="Base backoff delay in seconds (exponential) between retries.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Persist intermediate output every N accepted test cases.",
    )
    parser.add_argument(
        "--disable-resume",
        action="store_true",
        help="Disable resume mode and ignore existing output file.",
    )
    parser.add_argument(
        "--skip-api-probe",
        action="store_true",
        help="Skip model API probe during preflight checks.",
    )
    return parser.parse_args()


@dataclass(slots=True)
class ChunkEntry:
    chunk_id: str
    text: str
    source_path: str


def _normalize_text(text: str) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
    return normalized


def _coerce_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        s = value.strip()
        return [s] if s else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            s = str(item).strip()
            if s:
                out.append(s)
        return out
    return [str(value).strip()] if str(value).strip() else []


def _build_hybrid_search(settings: Settings) -> HybridSearch:
    embedding_client = EmbeddingFactory.create(settings)
    vector_store = VectorStoreFactory.create(settings)
    bm25_indexer = BM25Indexer()
    if bm25_indexer.index_path.exists():
        bm25_indexer.load()

    query_processor = QueryProcessor()
    dense_retriever = DenseRetriever(embedding_client=embedding_client, vector_store=vector_store, settings=settings)
    sparse_retriever = SparseRetriever(bm25_indexer=bm25_indexer, vector_store=vector_store, settings=settings)

    return HybridSearch(
        query_processor=query_processor,
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        settings=settings,
    )


def _load_chunks(settings: Settings, collection: str) -> list[ChunkEntry]:
    store = VectorStoreFactory.create(settings)
    records = store.get_by_metadata(filters={"collection": collection})

    chunks: list[ChunkEntry] = []
    for record in records:
        chunk_id = str(getattr(record, "id", "")).strip()
        text = str(getattr(record, "text", "")).strip()
        metadata = dict(getattr(record, "metadata", {}))
        source_path = str(metadata.get("source_path", "")).strip()
        if not chunk_id or not text or not source_path:
            continue
        chunks.append(ChunkEntry(chunk_id=chunk_id, text=text, source_path=source_path))
    return chunks


def _select_generation_docs(chunks: list[ChunkEntry], max_source_docs: int, seed: int) -> list[Any]:
    try:
        from langchain_core.documents import Document
    except Exception as exc:
        raise ImportError(
            "Missing dependency 'langchain-core'. Install it before generating with ragas."
        ) from exc

    rnd = random.Random(seed)
    sampled = list(chunks)
    if len(sampled) > max_source_docs:
        sampled = rnd.sample(sampled, k=max_source_docs)

    docs: list[Any] = []
    for chunk in sampled:
        docs.append(
            Document(
                page_content=chunk.text,
                metadata={
                    "chunk_id": chunk.chunk_id,
                    "source_path": chunk.source_path,
                },
            )
        )
    return docs


def _build_langchain_models(settings: Settings) -> tuple[Any, Any]:
    provider = str(settings.llm.provider).strip().lower()
    emb_provider = str(settings.embedding.provider).strip().lower()

    if provider not in {"openai", "azure"} or emb_provider not in {"openai", "azure"}:
        raise ValueError(
            "This script currently supports llm/embedding provider in {'openai','azure'} for ragas generation."
        )

    try:
        from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings, ChatOpenAI, OpenAIEmbeddings
    except Exception as exc:
        raise ImportError(
            "Missing dependency 'langchain-openai'. Install ragas generation dependencies first."
        ) from exc

    try:
        from langchain_core.rate_limiters import InMemoryRateLimiter
        # 中转接口限速：每秒最多 0.5 次请求（即 2s/req），防止触发限流或打爆中转站
        _rate_limiter: Any = InMemoryRateLimiter(
            requests_per_second=0.5,
            check_every_n_seconds=0.1,
            max_bucket_size=2,
        )
    except ImportError:
        _rate_limiter = None

    if provider == "azure":
        llm = AzureChatOpenAI(
            azure_deployment=settings.llm.deployment_name or settings.llm.model,
            api_version=settings.llm.api_version,
            azure_endpoint=settings.llm.endpoint or settings.llm.api_url,
            api_key=settings.llm.api_key,
            temperature=0,
        )
    else:
        llm = ChatOpenAI(
            model=settings.llm.model,
            api_key=settings.llm.api_key,
            base_url=settings.llm.api_url,
            temperature=0,
            timeout=60.0,
            max_retries=5,
            **({"rate_limiter": _rate_limiter} if _rate_limiter is not None else {}),
        )

    if emb_provider == "azure":
        embeddings = AzureOpenAIEmbeddings(
            model=settings.embedding.model,
            azure_deployment=settings.embedding.deployment_name or settings.embedding.model,
            api_version=settings.embedding.api_version,
            azure_endpoint=settings.embedding.endpoint or settings.embedding.api_url,
            api_key=settings.embedding.api_key,
        )
    else:
        embeddings = OpenAIEmbeddings(
            model=settings.embedding.model,
            api_key=settings.embedding.api_key,
            base_url=settings.embedding.api_url,
        )

    return llm, embeddings


def _extract_endpoint_urls(settings: Settings) -> list[str]:
    candidates = [
        str(settings.llm.endpoint or settings.llm.api_url or "").strip(),
        str(settings.embedding.endpoint or settings.embedding.api_url or "").strip(),
    ]
    return [url for url in candidates if url]


def _preflight_tcp_connectivity(urls: list[str], timeout_sec: float) -> None:
    for raw_url in urls:
        parsed = urlparse(raw_url)
        host = parsed.hostname
        if not host:
            raise RuntimeError(f"Invalid endpoint URL (missing host): {raw_url}")
        port = parsed.port
        if port is None:
            port = 443 if parsed.scheme == "https" else 80
        try:
            with socket.create_connection((host, port), timeout=timeout_sec):
                pass
        except OSError as exc:
            raise RuntimeError(
                f"Preflight connectivity failed for endpoint {raw_url} ({host}:{port}): {exc}"
            ) from exc


def _probe_model_clients(llm: Any, embeddings: Any) -> None:
    # Small probes that fail fast on auth/proxy/tls issues before long generation starts.
    if hasattr(embeddings, "embed_query"):
        embeddings.embed_query("preflight connectivity probe")
    if hasattr(llm, "invoke"):
        llm.invoke("Reply with one word: ok")


def _is_retryable_error(exc: Exception) -> bool:
    retryable_signatures = (
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "ConnectError",
        "ReadTimeout",
        "TimeoutException",
        "temporarily unavailable",
        "connection reset",
        "connection aborted",
        "network",
        "tls",
        "proxy",
    )
    joined = f"{exc.__class__.__name__}: {exc}"
    lowered = joined.lower()
    return any(signature.lower() in lowered for signature in retryable_signatures)


def _generate_with_retry(
    docs: list[Any],
    llm: Any,
    embeddings: Any,
    num_queries: int,
    max_retries: int,
    base_delay_sec: float,
) -> list[dict[str, Any]]:
    attempts = max(1, max_retries)
    for attempt in range(1, attempts + 1):
        try:
            return _generate_with_ragas(docs=docs, llm=llm, embeddings=embeddings, num_queries=num_queries)
        except Exception as exc:
            should_retry = _is_retryable_error(exc) and attempt < attempts
            if not should_retry:
                raise
            sleep_sec = base_delay_sec * (2 ** (attempt - 1))
            print(
                f"[WARN] ragas generation failed (attempt {attempt}/{attempts}): {exc}. "
                f"Retrying in {sleep_sec:.1f}s..."
            )
            time.sleep(sleep_sec)
    raise RuntimeError("Unexpected retry flow termination.")


def _load_existing_output(output_path: Path) -> dict[str, Any]:
    if not output_path.exists():
        return {"metadata": {}, "test_cases": []}
    try:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
    except Exception:
        return {"metadata": {}, "test_cases": []}
    metadata = payload.get("metadata")
    test_cases = payload.get("test_cases")
    if not isinstance(metadata, dict):
        metadata = {}
    if not isinstance(test_cases, list):
        test_cases = []
    return {"metadata": metadata, "test_cases": test_cases}


def _write_output(
    output_path: Path,
    *,
    collection: str,
    requested_queries: int,
    test_cases: list[dict[str, Any]],
    fallback_used: int,
    skipped_no_query: int,
    seed: int,
    resumed_from_existing: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": {
            "generated_by": "ragas_testset_generator",
            "collection": collection,
            "requested_queries": requested_queries,
            "usable_queries": len(test_cases),
            "fallback_used": fallback_used,
            "skipped_no_query": skipped_no_query,
            "seed": seed,
            "resumed_from_existing": resumed_from_existing,
        },
        "test_cases": test_cases,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


try:
    from ragas.run_config import RunConfig as _RagasRunConfig
except ImportError:
    _RagasRunConfig = None  # type: ignore


def _generate_with_ragas(docs: list[Any], llm: Any, embeddings: Any, num_queries: int) -> list[dict[str, Any]]:
    try:
        from ragas.testset import TestsetGenerator
    except Exception:
        try:
            from ragas.testset.generator import TestsetGenerator  # type: ignore
        except Exception as exc:
            raise ImportError("Cannot import ragas TestsetGenerator. Check ragas version/install.") from exc

    generator: Any
    if hasattr(TestsetGenerator, "from_langchain"):
        generator = TestsetGenerator.from_langchain(llm=llm, embedding_model=embeddings)
    else:
        generator = TestsetGenerator(llm=llm, embedding_model=embeddings)

    kwargs: dict[str, Any] = {}
    if _RagasRunConfig is not None:
        kwargs["run_config"] = _RagasRunConfig(max_workers=2, max_retries=5)

    if hasattr(generator, "generate_with_langchain_docs"):
        dataset = generator.generate_with_langchain_docs(docs, testset_size=num_queries, **kwargs)
    elif hasattr(generator, "generate"):
        dataset = generator.generate(docs, test_size=num_queries, **kwargs)
    else:
        raise RuntimeError("Unsupported ragas TestsetGenerator API in current version.")

    if hasattr(dataset, "to_pandas"):
        frame = dataset.to_pandas()
        return frame.to_dict(orient="records")

    if isinstance(dataset, list):
        return [dict(item) for item in dataset if isinstance(item, dict)]

    raise RuntimeError("Unsupported ragas dataset output type.")


def _pick_query(row: dict[str, Any]) -> str:
    for key in ("question", "query", "user_input"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _pick_contexts(row: dict[str, Any]) -> list[str]:
    for key in (
        "reference_contexts",
        "contexts",
        "reference",
        "ground_truth_context",
        "ground_truth",
    ):
        values = _coerce_text_list(row.get(key))
        if values:
            return values
    return []


def _build_text_index(chunks: list[ChunkEntry]) -> tuple[dict[str, list[ChunkEntry]], dict[str, list[ChunkEntry]]]:
    exact: dict[str, list[ChunkEntry]] = {}
    prefix: dict[str, list[ChunkEntry]] = {}
    for chunk in chunks:
        norm = _normalize_text(chunk.text)
        if not norm:
            continue
        exact.setdefault(norm, []).append(chunk)
        pref = norm[:120]
        prefix.setdefault(pref, []).append(chunk)
    return exact, prefix


def _match_context_to_chunks(
    context_text: str,
    exact_index: dict[str, list[ChunkEntry]],
    prefix_index: dict[str, list[ChunkEntry]],
) -> list[ChunkEntry]:
    norm = _normalize_text(context_text)
    if not norm:
        return []

    exact_hits = exact_index.get(norm)
    if exact_hits:
        return exact_hits

    pref = norm[:120]
    candidates = prefix_index.get(pref, [])
    if candidates:
        return candidates

    return []


def _dedupe_keep_order(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def main() -> int:
    args = parse_args()

    if args.num_queries <= 0:
        raise ValueError("--num-queries must be > 0")
    if args.max_source_docs <= 0:
        raise ValueError("--max-source-docs must be > 0")
    if args.fallback_top_k <= 0:
        raise ValueError("--fallback-top-k must be > 0")
    if args.network_timeout_sec <= 0:
        raise ValueError("--network-timeout-sec must be > 0")
    if args.max_retries <= 0:
        raise ValueError("--max-retries must be > 0")
    if args.retry_base_delay_sec <= 0:
        raise ValueError("--retry-base-delay-sec must be > 0")
    if args.checkpoint_every <= 0:
        raise ValueError("--checkpoint-every must be > 0")

    settings = load_settings(args.settings)
    collection = args.collection or settings.vector_store.collection
    output_path = Path(args.output)

    chunks = _load_chunks(settings, collection)
    if not chunks:
        raise RuntimeError(f"No chunks found in collection '{collection}'. Run ingestion first.")

    docs = _select_generation_docs(chunks, max_source_docs=args.max_source_docs, seed=args.seed)
    llm, embeddings = _build_langchain_models(settings)

    endpoint_urls = _extract_endpoint_urls(settings)
    if endpoint_urls:
        print(f"[INFO] preflight endpoint tcp check: endpoints={len(endpoint_urls)} timeout={args.network_timeout_sec}s")
        _preflight_tcp_connectivity(endpoint_urls, timeout_sec=args.network_timeout_sec)

    if not args.skip_api_probe:
        print("[INFO] preflight model api probe...")
        _probe_model_clients(llm=llm, embeddings=embeddings)

    print(f"[INFO] generating queries with ragas: target={args.num_queries}, source_docs={len(docs)}")
    rows = _generate_with_retry(
        docs=docs,
        llm=llm,
        embeddings=embeddings,
        num_queries=args.num_queries,
        max_retries=args.max_retries,
        base_delay_sec=args.retry_base_delay_sec,
    )
    if not rows:
        raise RuntimeError("Ragas returned empty dataset.")

    exact_index, prefix_index = _build_text_index(chunks)
    hybrid_search = _build_hybrid_search(settings)

    resumed = False
    existing_cases: list[dict[str, Any]] = []
    skipped_no_query = 0
    fallback_used = 0
    if not args.disable_resume:
        existing = _load_existing_output(output_path)
        existing_cases = [item for item in existing["test_cases"] if isinstance(item, dict)]
        old_meta = existing.get("metadata", {})
        skipped_no_query = int(old_meta.get("skipped_no_query", 0) or 0)
        fallback_used = int(old_meta.get("fallback_used", 0) or 0)
        resumed = len(existing_cases) > 0
        if resumed:
            print(f"[INFO] resume mode: loaded existing cases={len(existing_cases)} from {output_path}")

    test_cases: list[dict[str, Any]] = list(existing_cases)
    existing_query_set: set[str] = set()
    for case in test_cases:
        q = str(case.get("query", "")).strip()
        if q:
            existing_query_set.add(q)

    for row in rows:
        query = _pick_query(row)
        if not query:
            skipped_no_query += 1
            continue
        if query in existing_query_set:
            continue

        contexts = _pick_contexts(row)
        matched: list[ChunkEntry] = []
        for ctx in contexts:
            matched.extend(_match_context_to_chunks(ctx, exact_index, prefix_index))

        expected_chunk_ids = _dedupe_keep_order([item.chunk_id for item in matched])
        expected_sources = _dedupe_keep_order([item.source_path for item in matched])

        if not expected_chunk_ids:
            # Retrieval fallback guarantees project-required expected_chunk_ids field.
            results = hybrid_search.search(
                query=query,
                top_k=args.fallback_top_k,
                filters={"collection": collection},
            )
            expected_chunk_ids = _dedupe_keep_order([result.chunk_id for result in results])
            expected_sources = _dedupe_keep_order(
                [str(result.metadata.get("source_path", "")) for result in results if result.metadata.get("source_path")]
            )
            if expected_chunk_ids:
                fallback_used += 1

        if not expected_chunk_ids:
            # EvalRunner requires non-empty expected_chunk_ids, so skip invalid sample.
            continue

        test_cases.append(
            {
                "query": query,
                "expected_chunk_ids": expected_chunk_ids,
                "expected_sources": expected_sources,
            }
        )
        existing_query_set.add(query)

        if len(test_cases) % args.checkpoint_every == 0:
            _write_output(
                output_path,
                collection=collection,
                requested_queries=args.num_queries,
                test_cases=test_cases,
                fallback_used=fallback_used,
                skipped_no_query=skipped_no_query,
                seed=args.seed,
                resumed_from_existing=resumed,
            )
            print(f"[INFO] checkpoint saved: cases={len(test_cases)} path={output_path}")

        if len(test_cases) >= args.num_queries:
            break

    if len(test_cases) < args.num_queries:
        print(
            f"[WARN] generated usable cases={len(test_cases)} < requested={args.num_queries}. "
            "Try raising --max-source-docs or re-run with another --seed."
        )

    _write_output(
        output_path,
        collection=collection,
        requested_queries=args.num_queries,
        test_cases=test_cases,
        fallback_used=fallback_used,
        skipped_no_query=skipped_no_query,
        seed=args.seed,
        resumed_from_existing=resumed,
    )

    # Hard success criteria to avoid false positive "exit 0 but no usable output".
    written = _load_existing_output(output_path)
    written_cases = written.get("test_cases", [])
    if not isinstance(written_cases, list) or len(written_cases) == 0:
        raise RuntimeError(f"Output validation failed: no usable test_cases written to {output_path}")

    print(
        f"[DONE] wrote golden testset: {output_path} | cases={len(test_cases)} "
        f"fallback_used={fallback_used}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
