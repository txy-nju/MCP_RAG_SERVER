"""Evaluation runner for golden test sets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.query_engine.hybrid_search import HybridSearch
from core.settings import Settings
from core.types import RetrievalResult
from libs.evaluator.base_evaluator import BaseEvaluator


@dataclass(slots=True)
class EvalCase:
	"""Single golden evaluation case."""

	query: str
	expected_chunk_ids: list[str]
	expected_sources: list[str] = field(default_factory=list)


@dataclass(slots=True)
class EvalCaseResult:
	"""Per-query evaluation result."""

	query: str
	metrics: dict[str, float]
	retrieved_chunk_ids: list[str]
	retrieved_sources: list[str]
	expected_chunk_ids: list[str]
	expected_sources: list[str]


@dataclass(slots=True)
class EvalReport:
	"""Aggregated evaluation report."""

	metrics: dict[str, float]
	cases: list[EvalCaseResult]

	def to_dict(self) -> dict[str, Any]:
		return {
			"metrics": dict(self.metrics),
			"cases": [
				{
					"query": case.query,
					"metrics": dict(case.metrics),
					"retrieved_chunk_ids": list(case.retrieved_chunk_ids),
					"retrieved_sources": list(case.retrieved_sources),
					"expected_chunk_ids": list(case.expected_chunk_ids),
					"expected_sources": list(case.expected_sources),
				}
				for case in self.cases
			],
		}


class EvalRunner:
	"""Run retrieval against a golden set and aggregate evaluator metrics."""

	def __init__(self, settings: Settings, hybrid_search: HybridSearch, evaluator: BaseEvaluator) -> None:
		self.settings = settings
		self.hybrid_search = hybrid_search
		self.evaluator = evaluator

	def run(self, test_set_path: str | Path) -> EvalReport:
		"""Load a golden test set, run retrieval, and aggregate metrics."""

		cases = self._load_test_cases(test_set_path)
		case_results: list[EvalCaseResult] = []

		for case in cases:
			results = self.hybrid_search.search(
				query=case.query,
				top_k=self.settings.retrieval.top_k,
				filters={"collection": self.settings.vector_store.collection},
			)
			retrieved_chunk_ids = [result.chunk_id for result in results]
			retrieved_sources = [str(result.metadata.get("source_path", "")) for result in results]
			metrics = self.evaluator.evaluate(case.query, retrieved_chunk_ids, case.expected_chunk_ids)

			case_results.append(
				EvalCaseResult(
					query=case.query,
					metrics=metrics,
					retrieved_chunk_ids=retrieved_chunk_ids,
					retrieved_sources=retrieved_sources,
					expected_chunk_ids=list(case.expected_chunk_ids),
					expected_sources=list(case.expected_sources),
				)
			)

		return EvalReport(metrics=self._aggregate_metrics(case_results), cases=case_results)

	@staticmethod
	def _load_test_cases(test_set_path: str | Path) -> list[EvalCase]:
		path = Path(test_set_path)
		with path.open("r", encoding="utf-8") as handle:
			payload = json.load(handle)

		raw_cases = payload.get("test_cases")
		if not isinstance(raw_cases, list) or not raw_cases:
			raise ValueError("golden test set must contain a non-empty 'test_cases' list")

		cases: list[EvalCase] = []
		for index, raw_case in enumerate(raw_cases, start=1):
			if not isinstance(raw_case, dict):
				raise ValueError(f"test_cases[{index}] must be an object")

			query = str(raw_case.get("query", "")).strip()
			expected_chunk_ids = raw_case.get("expected_chunk_ids")
			expected_sources = raw_case.get("expected_sources", [])
			if not query:
				raise ValueError(f"test_cases[{index}].query is required")
			if not isinstance(expected_chunk_ids, list) or not expected_chunk_ids:
				raise ValueError(f"test_cases[{index}].expected_chunk_ids must be a non-empty list")
			if not isinstance(expected_sources, list):
				raise ValueError(f"test_cases[{index}].expected_sources must be a list")

			cases.append(
				EvalCase(
					query=query,
					expected_chunk_ids=[str(item) for item in expected_chunk_ids],
					expected_sources=[str(item) for item in expected_sources],
				)
			)
		return cases

	@staticmethod
	def _aggregate_metrics(case_results: list[EvalCaseResult]) -> dict[str, float]:
		if not case_results:
			return {}

		totals: dict[str, float] = {}
		for case in case_results:
			for metric_name, metric_value in case.metrics.items():
				totals[metric_name] = totals.get(metric_name, 0.0) + float(metric_value)

		case_count = float(len(case_results))
		return {
			metric_name: total / case_count
			for metric_name, total in sorted(totals.items())
		}


__all__ = ["EvalCase", "EvalCaseResult", "EvalReport", "EvalRunner"]
