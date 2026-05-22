"""Unit tests for core reranker orchestration with fallback handling."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from modular_rag.core.query_engine.reranker import Reranker, RerankResult
from modular_rag.core.settings import Settings
from modular_rag.core.types import RetrievalResult
from modular_rag.libs.reranker.base_reranker import RerankCandidate, RerankerFallbackError


@pytest.fixture
def sample_candidates() -> list[RetrievalResult]:
	"""Generate sample retrieval candidates."""
	return [
		RetrievalResult(
			chunk_id="1",
			score=0.9,
			text="First result",
			metadata={"source_path": "document1.pdf"},
		),
		RetrievalResult(
			chunk_id="2",
			score=0.8,
			text="Second result",
			metadata={"source_path": "document2.pdf"},
		),
		RetrievalResult(
			chunk_id="3",
			score=0.7,
			text="Third result",
			metadata={"source_path": "document3.pdf"},
		),
	]


@pytest.fixture
def mock_settings() -> Settings:
	"""Create mock settings."""
	settings = MagicMock(spec=Settings)
	settings.reranker_provider = "llm"
	settings.timeout_seconds = 5.0
	return settings


@pytest.fixture
def reranker(mock_settings: Settings) -> Reranker:
	"""Create reranker instance."""
	return Reranker(settings=mock_settings, timeout_seconds=5.0)


def test_reranker_normal_path(
	reranker: Reranker, sample_candidates: list[RetrievalResult]
) -> None:
	"""Test normal reranking path with successful backend call."""
	mock_backend = MagicMock()
	reranked_items = [
		RerankCandidate(id="2", score=0.95, text="Second result", metadata={}),
		RerankCandidate(id="1", score=0.85, text="First result", metadata={}),
		RerankCandidate(id="3", score=0.70, text="Third result", metadata={}),
	]
	mock_backend.rerank.return_value = reranked_items

	with patch("core.query_engine.reranker.RerankerFactory.create", return_value=mock_backend):
		result = reranker.rerank("query", sample_candidates)

	assert not result.fallback
	assert result.fallback_reason is None
	assert len(result.candidates) == 3
	assert result.candidates[0].chunk_id == "2"
	assert result.candidates[1].chunk_id == "1"
	assert result.candidates[2].chunk_id == "3"


def test_reranker_fallback_on_reranker_error(
	reranker: Reranker, sample_candidates: list[RetrievalResult]
) -> None:
	"""Test fallback to original order on RerankerFallbackError."""
	mock_backend = MagicMock()
	mock_backend.rerank.side_effect = RerankerFallbackError("Backend unavailable")
	mock_backend.provider = "cross_encoder"

	with patch("core.query_engine.reranker.RerankerFactory.create", return_value=mock_backend):
		result = reranker.rerank("query", sample_candidates)

	assert result.fallback
	assert "Backend unavailable" in result.fallback_reason
	assert result.candidates == sample_candidates


def test_reranker_fallback_on_timeout(
	reranker: Reranker, sample_candidates: list[RetrievalResult]
) -> None:
	"""Test fallback on TimeoutError."""
	mock_backend = MagicMock()
	mock_backend.rerank.side_effect = TimeoutError()

	with patch("core.query_engine.reranker.RerankerFactory.create", return_value=mock_backend):
		result = reranker.rerank("query", sample_candidates)

	assert result.fallback
	assert "timeout" in result.fallback_reason.lower()
	assert "5.0" in result.fallback_reason
	assert result.candidates == sample_candidates


def test_reranker_fallback_on_generic_exception(
	reranker: Reranker, sample_candidates: list[RetrievalResult]
) -> None:
	"""Test fallback on generic Exception."""
	mock_backend = MagicMock()
	mock_backend.rerank.side_effect = ValueError("Unexpected error")

	with patch("core.query_engine.reranker.RerankerFactory.create", return_value=mock_backend):
		result = reranker.rerank("query", sample_candidates)

	assert result.fallback
	assert "Unexpected error" in result.fallback_reason
	assert result.candidates == sample_candidates


def test_reranker_empty_candidates(reranker: Reranker) -> None:
	"""Test reranker with empty candidates list."""
	result = reranker.rerank("query", [])

	assert not result.fallback
	assert result.fallback_reason is None
	assert len(result.candidates) == 0


def test_reranker_empty_query(
	reranker: Reranker, sample_candidates: list[RetrievalResult]
) -> None:
	"""Test reranker with empty query."""
	result = reranker.rerank("", sample_candidates)

	assert not result.fallback
	assert result.fallback_reason is None
	assert result.candidates == sample_candidates


def test_reranker_whitespace_query(
	reranker: Reranker, sample_candidates: list[RetrievalResult]
) -> None:
	"""Test reranker with whitespace-only query."""
	result = reranker.rerank("   ", sample_candidates)

	assert not result.fallback
	assert result.fallback_reason is None
	assert result.candidates == sample_candidates


def test_reranker_preserves_metadata(
	reranker: Reranker, sample_candidates: list[RetrievalResult]
) -> None:
	"""Test that reranker preserves metadata through reranking."""
	mock_backend = MagicMock()
	reranked_items = [
		RerankCandidate(id="1", score=0.99, text="First result", metadata={}),
		RerankCandidate(id="2", score=0.88, text="Second result", metadata={}),
	]
	mock_backend.rerank.return_value = reranked_items

	with patch("core.query_engine.reranker.RerankerFactory.create", return_value=mock_backend):
		result = reranker.rerank("query", sample_candidates)

	assert "source_path" in result.candidates[0].metadata
	assert result.candidates[0].metadata["source_path"] == "document1.pdf"
	assert "source_path" in result.candidates[1].metadata
	assert result.candidates[1].metadata["source_path"] == "document2.pdf"


def test_reranker_score_conversion(
	reranker: Reranker, sample_candidates: list[RetrievalResult]
) -> None:
	"""Test score conversion from RerankCandidate to RetrievalResult."""
	mock_backend = MagicMock()
	reranked_items = [
		RerankCandidate(id="1", score=0.95, text="First result", metadata={}),
	]
	mock_backend.rerank.return_value = reranked_items

	with patch("core.query_engine.reranker.RerankerFactory.create", return_value=mock_backend):
		result = reranker.rerank("query", sample_candidates)

	assert result.candidates[0].score == 0.95


def test_reranker_single_candidate(reranker: Reranker) -> None:
	"""Test reranker with single candidate."""
	candidate = RetrievalResult(
		chunk_id="1", score=0.9, text="Single result", metadata={"source_path": "doc.pdf"}
	)

	mock_backend = MagicMock()
	mock_backend.rerank.return_value = [
		RerankCandidate(id="1", score=0.95, text="Single result", metadata={})
	]

	with patch("core.query_engine.reranker.RerankerFactory.create", return_value=mock_backend):
		result = reranker.rerank("query", [candidate])

	assert len(result.candidates) == 1
	assert result.candidates[0].chunk_id == "1"
