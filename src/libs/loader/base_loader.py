"""Base interfaces for pluggable document loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod

from core.types import Document


class BaseLoader(ABC):
	"""Common contract for all document loaders used by ingestion."""

	@abstractmethod
	def load(self, path: str) -> Document:
		"""Load a source file and return a normalized ``Document`` object."""

