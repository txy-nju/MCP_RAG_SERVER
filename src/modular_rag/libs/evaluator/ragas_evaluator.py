"""Ragas-backed evaluator implementation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import importlib
from typing import Any

from modular_rag.libs.evaluator.base_evaluator import BaseEvaluator


class RagasEvaluator(BaseEvaluator):
	"""Evaluate retrieval quality with Ragas metrics."""

	def __init__(
		self,
		*,
		backend: str = "ragas",
		ragas_runner: Callable[[str, list[str], list[str]], Mapping[str, Any]] | None = None,
	) -> None:
		super().__init__(backend=backend)
		self._ragas_runner = ragas_runner

	@classmethod
	def from_settings(cls, settings: Any) -> "RagasEvaluator":
		return cls(backend=str(settings.backend))

	def evaluate(
		self,
		query: str,
		retrieved_ids: list[str],
		golden_ids: list[str],
		trace: object | None = None,
	) -> dict[str, float]:
		"""Run ragas evaluation and normalize supported metrics."""

		runner = self._ragas_runner or self._build_default_runner()
		self._ragas_runner = runner
		raw_metrics = runner(query, retrieved_ids, golden_ids)
		return self._normalize_metrics(raw_metrics)

	@staticmethod
	def _normalize_metrics(raw_metrics: Mapping[str, Any]) -> dict[str, float]:
		normalized: dict[str, float] = {}
		aliases = {
			"faithfulness": ["faithfulness"],
			"answer_relevancy": ["answer_relevancy", "answer_relevance"],
			"context_precision": ["context_precision"],
		}

		for canonical_name, possible_names in aliases.items():
			value: Any = 0.0
			for name in possible_names:
				if name in raw_metrics:
					value = raw_metrics[name]
					break
			normalized[canonical_name] = float(value)

		return normalized

	@staticmethod
	def _build_default_runner() -> Callable[[str, list[str], list[str]], Mapping[str, Any]]:
		"""Build the default Ragas execution function with lazy imports."""

		try:
			ragas_module = importlib.import_module("ragas")
			metrics_module = importlib.import_module("ragas.metrics")
			datasets_module = importlib.import_module("datasets")
		except ModuleNotFoundError as exc:
			raise ImportError(
				"RagasEvaluator requires optional dependencies: ragas and datasets. "
				"Install them before using evaluation backend 'ragas'."
			) from exc

		def _runner(query: str, retrieved_ids: list[str], golden_ids: list[str]) -> Mapping[str, Any]:
			dataset = datasets_module.Dataset.from_dict(
				{
					"question": [query],
					"answer": ["\n".join(retrieved_ids)],
					"contexts": [retrieved_ids],
					"ground_truth": ["\n".join(golden_ids)],
				}
			)

			evaluation_result = ragas_module.evaluate(
				dataset=dataset,
				metrics=[
					metrics_module.faithfulness,
					metrics_module.answer_relevancy,
					metrics_module.context_precision,
				],
			)

			if hasattr(evaluation_result, "to_dict"):
				return evaluation_result.to_dict()
			if isinstance(evaluation_result, Mapping):
				return evaluation_result
			raise TypeError("Unexpected ragas evaluation result type")

		return _runner
