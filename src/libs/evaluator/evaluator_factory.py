"""Factory for creating evaluator providers from project settings."""

from __future__ import annotations

from core.settings import EvaluationSettings, Settings
from libs.evaluator.base_evaluator import BaseEvaluator
from libs.evaluator.custom_evaluator import CustomEvaluator
from libs.evaluator.ragas_evaluator import RagasEvaluator
from observability.evaluation.composite_evaluator import CompositeEvaluator


class EvaluatorFactory:
    """Registry-backed factory for pluggable evaluator providers."""

    _providers: dict[str, type[BaseEvaluator]] = {
        "custom": CustomEvaluator,
        "ragas": RagasEvaluator,
    }

    @classmethod
    def register(cls, backend: str, evaluator_cls: type[BaseEvaluator]) -> None:
        """Register an evaluator backend for later creation."""

        normalized_backend = backend.strip().lower()
        if not normalized_backend:
            raise ValueError("Evaluator backend name cannot be empty")
        if not issubclass(evaluator_cls, BaseEvaluator):
            raise TypeError("Registered evaluator class must inherit from BaseEvaluator")
        cls._providers[normalized_backend] = evaluator_cls

    @classmethod
    def unregister(cls, backend: str) -> None:
        """Remove an evaluator backend registration if present."""

        normalized_backend = backend.strip().lower()
        if normalized_backend == "custom":
            cls._providers[normalized_backend] = CustomEvaluator
            return
        cls._providers.pop(normalized_backend, None)

    @classmethod
    def create(cls, settings: Settings | EvaluationSettings) -> BaseEvaluator:
        """Create an evaluator instance from top-level settings or evaluation settings."""

        evaluation_settings = cls._extract_evaluation_settings(settings)
        if len(evaluation_settings.backends) > 1:
            evaluators = [
                cls.create(EvaluationSettings(backend=backend)) for backend in evaluation_settings.backends
            ]
            return CompositeEvaluator(evaluators)
        backend = evaluation_settings.backend.strip().lower()
        evaluator_cls = cls._providers.get(backend)
        if evaluator_cls is None:
            available = ", ".join(sorted(cls._providers)) or "none"
            raise ValueError(
                f"Unsupported evaluator backend: {evaluation_settings.backend}. Available backends: {available}"
            )
        return evaluator_cls.from_settings(evaluation_settings)

    @staticmethod
    def _extract_evaluation_settings(settings: Settings | EvaluationSettings) -> EvaluationSettings:
        """Normalize supported input types to an ``EvaluationSettings`` object."""

        if isinstance(settings, EvaluationSettings):
            return settings
        if isinstance(settings, Settings):
            return settings.evaluation
        raise TypeError("EvaluatorFactory.create expects Settings or EvaluationSettings")
