"""Composite evaluator implementation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from libs.evaluator.base_evaluator import BaseEvaluator


class CompositeEvaluator(BaseEvaluator):
	"""Run multiple evaluator backends and merge their metrics."""

	def __init__(self, evaluators: list[BaseEvaluator]) -> None:
		super().__init__(backend="composite")
		if not evaluators:
			raise ValueError("CompositeEvaluator requires at least one evaluator")
		self.evaluators = list(evaluators)

	def evaluate(
		self,
		query: str,
		retrieved_ids: list[str],
		golden_ids: list[str],
		trace: object | None = None,
	) -> dict[str, float]:
		"""Evaluate with all child evaluators in parallel and merge metrics."""

		with ThreadPoolExecutor(max_workers=len(self.evaluators)) as executor:
			futures = [
				executor.submit(
					evaluator.evaluate,
					query,
					retrieved_ids,
					golden_ids,
					trace,
				)
				for evaluator in self.evaluators
			]

		merged: dict[str, float] = {}
		for future in futures:
			metrics = future.result()
			duplicated_keys = set(merged).intersection(metrics)
			if duplicated_keys:
				duplicates = ", ".join(sorted(duplicated_keys))
				raise ValueError(f"CompositeEvaluator received duplicate metric keys: {duplicates}")
			merged.update(metrics)
		return merged
