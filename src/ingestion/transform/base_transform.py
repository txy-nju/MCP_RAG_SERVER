"""Base contract for ingestion transform stages."""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.trace.trace_context import TraceContext
from core.types import Chunk


class BaseTransform(ABC):
	"""Shared interface for chunk-level transformation components."""

	@abstractmethod
	def transform(self, chunks: list[Chunk], trace: TraceContext | None = None) -> list[Chunk]:
		"""Transform chunks while preserving chunk ordering and source linkage."""
