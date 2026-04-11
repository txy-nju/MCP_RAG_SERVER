"""Integration tests for scripts/query.py CLI entrypoint."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.query_engine import DenseRetriever, HybridSearch, QueryProcessor, Reranker, SparseRetriever
from core.types import RetrievalResult


@pytest.fixture
def sample_results() -> list[RetrievalResult]:
	"""Generate sample retrieval results."""
	return [
		RetrievalResult(
			chunk_id="chunk_1",
			score=0.95,
			text="This is the first result about Azure configuration.",
			metadata={"source_path": "azure_guide.pdf", "page": 1},
		),
		RetrievalResult(
			chunk_id="chunk_2",
			score=0.87,
			text="Second result discussing deployment options.",
			metadata={"source_path": "deployment.pdf", "page": 5},
		),
	]


def test_query_script_parse_args() -> None:
	"""Test command-line argument parsing."""
	from scripts.query import parse_args

	args = parse_args(["--query", "test query", "--top-k", "5", "--verbose"])
	assert args.query == "test query"
	assert args.top_k == 5
	assert args.verbose is True
	assert args.no_rerank is False


def test_query_script_parse_args_defaults() -> None:
	"""Test default argument values."""
	from scripts.query import parse_args

	args = parse_args(["--query", "test"])
	assert args.top_k == 10
	assert args.verbose is False
	assert args.no_rerank is False
	assert args.collection is None


def test_query_script_parse_args_missing_query() -> None:
	"""Test that missing --query raises error."""
	from scripts.query import parse_args

	with pytest.raises(SystemExit):
		parse_args([])


def test_query_script_format_text_summary() -> None:
	"""Test text summary formatting."""
	from scripts.query import format_text_summary

	long_text = "a" * 150
	summary = format_text_summary(long_text, max_length=100)
	assert len(summary) == 103  # 100 chars + "..."
	assert summary.endswith("...")


def test_query_script_format_text_short() -> None:
	"""Test that short text is not truncated."""
	from scripts.query import format_text_summary

	short_text = "short text"
	summary = format_text_summary(short_text, max_length=100)
	assert summary == short_text


def test_query_script_format_text_newlines() -> None:
	"""Test that newlines are replaced with spaces."""
	from scripts.query import format_text_summary

	text_with_newlines = "line 1\nline 2\nline 3"
	summary = format_text_summary(text_with_newlines, max_length=100)
	assert "\n" not in summary
	assert "line 1 line 2 line 3" in summary


def test_query_script_main_missing_query(capsys: object) -> None:
	"""Test main function with missing query."""
	from scripts.query import main

	result = main(["--query", ""])
	assert result == 1


def test_query_script_main_invalid_top_k(capsys: object) -> None:
	"""Test main function with invalid top-k."""
	from scripts.query import main

	result = main(["--query", "test", "--top-k", "0"])
	assert result == 1


def test_query_script_main_settings_not_found(capsys: object) -> None:
	"""Test main function when settings file not found."""
	from scripts.query import main

	result = main(["--query", "test", "--settings", "/nonexistent/settings.yaml"])
	assert result == 1


def test_query_script_main_no_results(capsys: object) -> None:
	"""Test main function when no results found."""
	from scripts.query import main

	mock_hybrid_search = MagicMock(spec=HybridSearch)
	mock_hybrid_search.search.return_value = []

	with patch("scripts.query.build_components") as mock_build:
		mock_build.return_value = (
			MagicMock(spec=QueryProcessor),
			MagicMock(spec=DenseRetriever),
			MagicMock(spec=SparseRetriever),
			mock_hybrid_search,
			MagicMock(spec=Reranker),
		)
		with patch("scripts.query.load_settings"):
			result = main(["--query", "test"])

	captured = capsys.readouterr()
	assert "No results found" in captured.out
	assert result == 0


def test_query_script_main_success(sample_results: list[RetrievalResult], capsys: object) -> None:
	"""Test main function with successful query."""
	from scripts.query import main

	mock_hybrid_search = MagicMock(spec=HybridSearch)
	mock_hybrid_search.search.return_value = sample_results

	mock_reranker = MagicMock(spec=Reranker)
	mock_reranker.rerank.return_value = MagicMock(
		candidates=sample_results,
		fallback=False,
		fallback_reason=None,
	)

	with patch("scripts.query.build_components") as mock_build:
		mock_build.return_value = (
			MagicMock(spec=QueryProcessor),
			MagicMock(spec=DenseRetriever),
			MagicMock(spec=SparseRetriever),
			mock_hybrid_search,
			mock_reranker,
		)
		with patch("scripts.query.load_settings"):
			result = main(["--query", "test query", "--top-k", "2"])

	captured = capsys.readouterr()
	assert "test query" in captured.out
	assert "TOP-K RESULTS" in captured.out
	assert "chunk_1" not in captured.out or "verbose" not in ["--verbose"]
	assert result == 0


def test_query_script_main_no_rerank(sample_results: list[RetrievalResult], capsys: object) -> None:
	"""Test main function with --no-rerank flag."""
	from scripts.query import main

	mock_hybrid_search = MagicMock(spec=HybridSearch)
	mock_hybrid_search.search.return_value = sample_results

	mock_reranker = MagicMock(spec=Reranker)

	with patch("scripts.query.build_components") as mock_build:
		mock_build.return_value = (
			MagicMock(spec=QueryProcessor),
			MagicMock(spec=DenseRetriever),
			MagicMock(spec=SparseRetriever),
			mock_hybrid_search,
			mock_reranker,
		)
		with patch("scripts.query.load_settings"):
			result = main(["--query", "test", "--no-rerank"])

	captured = capsys.readouterr()
	assert "Skipped" in captured.out
	mock_reranker.rerank.assert_not_called()
	assert result == 0


def test_query_script_main_rerank_fallback(sample_results: list[RetrievalResult], capsys: object) -> None:
	"""Test main function when reranker falls back."""
	from scripts.query import main

	mock_hybrid_search = MagicMock(spec=HybridSearch)
	mock_hybrid_search.search.return_value = sample_results

	mock_reranker = MagicMock(spec=Reranker)
	mock_reranker.rerank.return_value = MagicMock(
		candidates=sample_results,
		fallback=True,
		fallback_reason="Backend timeout",
	)

	with patch("scripts.query.build_components") as mock_build:
		mock_build.return_value = (
			MagicMock(spec=QueryProcessor),
			MagicMock(spec=DenseRetriever),
			MagicMock(spec=SparseRetriever),
			mock_hybrid_search,
			mock_reranker,
		)
		with patch("scripts.query.load_settings"):
			result = main(["--query", "test"])

	captured = capsys.readouterr()
	assert "Fallback" in captured.out
	assert "Backend timeout" in captured.out
	assert result == 0


def test_query_script_main_verbose(sample_results: list[RetrievalResult], capsys: object) -> None:
	"""Test main function with verbose output."""
	from scripts.query import main

	mock_hybrid_search = MagicMock(spec=HybridSearch)
	mock_hybrid_search.search.return_value = sample_results

	mock_reranker = MagicMock(spec=Reranker)
	mock_reranker.rerank.return_value = MagicMock(
		candidates=sample_results,
		fallback=False,
		fallback_reason=None,
	)

	with patch("scripts.query.build_components") as mock_build:
		mock_build.return_value = (
			MagicMock(spec=QueryProcessor),
			MagicMock(spec=DenseRetriever),
			MagicMock(spec=SparseRetriever),
			mock_hybrid_search,
			mock_reranker,
		)
		with patch("scripts.query.load_settings"):
			result = main(["--query", "test", "--verbose"])

	captured = capsys.readouterr()
	assert "[HybridSearch]" in captured.out
	assert "chunk_1" in captured.out  # verbose shows chunk_id
	assert result == 0


def test_query_script_main_collection_param(capsys: object) -> None:
	"""Test main function with custom collection."""
	from scripts.query import main

	mock_hybrid_search = MagicMock(spec=HybridSearch)
	mock_hybrid_search.search.return_value = []

	with patch("scripts.query.build_components") as mock_build:
		mock_build.return_value = (
			MagicMock(spec=QueryProcessor),
			MagicMock(spec=DenseRetriever),
			MagicMock(spec=SparseRetriever),
			mock_hybrid_search,
			MagicMock(spec=Reranker),
		)
		with patch("scripts.query.load_settings"):
			result = main(["--query", "test", "--collection", "custom_collection"])

	mock_hybrid_search.search.assert_called_once()
	call_kwargs = mock_hybrid_search.search.call_args.kwargs
	assert call_kwargs["collection"] == "custom_collection"
