"""Evaluation panel page for running and inspecting retrieval evaluations."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

from core.query_engine import DenseRetriever, HybridSearch, QueryProcessor, SparseRetriever
from core.settings import Settings, load_settings
from ingestion.storage.bm25_indexer import BM25Indexer
from libs.embedding.embedding_factory import EmbeddingFactory
from libs.evaluator.evaluator_factory import EvaluatorFactory
from libs.vector_store.vector_store_factory import VectorStoreFactory
from observability.evaluation import EvalReport, EvalRunner


_PROJECT_ROOT = Path(__file__).parents[4]
_DEFAULT_SETTINGS_PATH = _PROJECT_ROOT / "config" / "settings.yaml"
_FIXTURES_DIR = _PROJECT_ROOT / "tests" / "fixtures"


def _list_golden_test_sets() -> list[str]:
	if not _FIXTURES_DIR.exists():
		return []
	return sorted(str(path.relative_to(_PROJECT_ROOT)) for path in _FIXTURES_DIR.glob("*.json"))


def _available_backends() -> list[str]:
	return sorted(EvaluatorFactory._providers.keys())


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


def _build_settings(settings_path: Path, selected_backends: list[str]) -> Settings:
	settings = load_settings(settings_path)
	normalized_backends = tuple(selected_backends)
	primary_backend = normalized_backends[0] if normalized_backends else settings.evaluation.backend
	evaluation = replace(
		settings.evaluation,
		backend=primary_backend,
		backends=normalized_backends if len(normalized_backends) > 1 else (),
	)
	return replace(settings, evaluation=evaluation)


def _render_metric_summary(metrics: dict[str, float]) -> None:
	if not metrics:
		st.info("No aggregate metrics available.")
		return

	columns = st.columns(min(len(metrics), 4))
	for index, (metric_name, metric_value) in enumerate(metrics.items()):
		with columns[index % len(columns)]:
			columns[index % len(columns)].metric(metric_name, round(metric_value, 4))


def _render_case_table(report: EvalReport) -> None:
	rows: list[dict[str, Any]] = []
	for case in report.cases:
		row: dict[str, Any] = {
			"query": case.query,
			"retrieved_chunk_ids": case.retrieved_chunk_ids,
			"expected_chunk_ids": case.expected_chunk_ids,
			"retrieved_sources": case.retrieved_sources,
			"expected_sources": case.expected_sources,
		}
		row.update(case.metrics)
		rows.append(row)

	st.dataframe(rows, use_container_width=True)


def _render_history() -> None:
	history = st.session_state.get("evaluation_history", [])
	st.subheader("Run History")
	if not history:
		st.caption("No evaluation history yet in this session.")
		return
	st.dataframe(history, use_container_width=True)


def _append_history(test_set: str, backends: list[str], report: EvalReport) -> None:
	history = list(st.session_state.get("evaluation_history", []))
	history.append(
		{
			"timestamp": datetime.now().isoformat(timespec="seconds"),
			"test_set": test_set,
			"backends": ", ".join(backends),
			**{metric_name: round(metric_value, 4) for metric_name, metric_value in report.metrics.items()},
		}
	)
	st.session_state["evaluation_history"] = history[-10:]


st.title("📈 Evaluation Panel")
st.markdown("Run golden-set retrieval evaluations and inspect aggregate metrics plus per-query details.")

try:
	golden_sets = _list_golden_test_sets()
	backend_options = _available_backends()

	if not golden_sets:
		st.warning("No golden test set JSON files found under tests/fixtures.")
		st.stop()

	with st.sidebar:
		settings_path_value = st.text_input("Settings Path", value=str(_DEFAULT_SETTINGS_PATH.relative_to(_PROJECT_ROOT)))
		selected_test_set = st.selectbox("Golden Test Set", options=golden_sets, index=0)
		selected_backends = st.multiselect(
			"Evaluation Backends",
			options=backend_options,
			default=backend_options[:1],
		)

	if not selected_backends:
		st.info("Select at least one evaluation backend to run an evaluation.")
		_render_history()
		st.stop()

	run_clicked = st.button("Run Evaluation", type="primary")

	if run_clicked:
		settings_path = (_PROJECT_ROOT / settings_path_value).resolve() if not Path(settings_path_value).is_absolute() else Path(settings_path_value)
		test_set_path = (_PROJECT_ROOT / selected_test_set).resolve()

		with st.spinner("Running evaluation..."):
			settings = _build_settings(settings_path, selected_backends)
			hybrid_search = _build_hybrid_search(settings)
			evaluator = EvaluatorFactory.create(settings)
			runner = EvalRunner(settings=settings, hybrid_search=hybrid_search, evaluator=evaluator)
			report = runner.run(test_set_path)

		st.session_state["latest_evaluation_report"] = report
		st.session_state["latest_evaluation_test_set"] = selected_test_set
		st.session_state["latest_evaluation_backends"] = list(selected_backends)
		_append_history(selected_test_set, selected_backends, report)
		st.success("Evaluation completed successfully.")

	latest_report = st.session_state.get("latest_evaluation_report")
	if latest_report is not None:
		st.subheader("Aggregate Metrics")
		_render_metric_summary(latest_report.metrics)

		st.subheader("Per-query Details")
		_render_case_table(latest_report)

	_render_history()

except Exception as exc:  # noqa: BLE001
	st.error(f"Failed to run evaluation panel: {exc}")
