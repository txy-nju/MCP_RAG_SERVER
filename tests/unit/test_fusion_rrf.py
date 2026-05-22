"""Unit tests for Reciprocal Rank Fusion (D4)."""

from __future__ import annotations

import pytest

from modular_rag.core.query_engine.fusion import RRFFuser
from modular_rag.core.types import RetrievalResult


def _result(chunk_id: str, score: float = 0.0) -> RetrievalResult:
	return RetrievalResult(
		chunk_id=chunk_id,
		score=score,
		text=f"text-{chunk_id}",
		metadata={"source_path": f"{chunk_id}.md"},
	)


@pytest.mark.unit
def test_rrf_fusion_merges_dense_and_sparse_rankings() -> None:
	fuser = RRFFuser(rrf_k=60)

	dense = [_result("a"), _result("b"), _result("c")]
	sparse = [_result("b"), _result("d"), _result("a")]

	results = fuser.fuse(dense_results=dense, sparse_results=sparse, top_k=4)

	assert [item.chunk_id for item in results] == ["b", "a", "d", "c"]

	# b: 1/(60+2) + 1/(60+1)
	expected_b = (1 / 62) + (1 / 61)
	assert results[0].score == pytest.approx(expected_b)


@pytest.mark.unit
def test_rrf_fusion_handles_single_route_results() -> None:
	fuser = RRFFuser(rrf_k=10)

	results = fuser.fuse(dense_results=[_result("x"), _result("y")], sparse_results=[], top_k=1)

	assert len(results) == 1
	assert results[0].chunk_id == "x"
	assert results[0].score == pytest.approx(1 / 11)


@pytest.mark.unit
def test_rrf_fusion_returns_empty_when_both_routes_empty() -> None:
	fuser = RRFFuser()

	assert fuser.fuse(dense_results=[], sparse_results=[], top_k=5) == []


@pytest.mark.unit
def test_rrf_fusion_rejects_invalid_top_k() -> None:
	fuser = RRFFuser()

	with pytest.raises(ValueError, match="top_k must be greater than 0"):
		fuser.fuse(dense_results=[_result("a")], sparse_results=[], top_k=0)


@pytest.mark.unit
def test_rrf_fusion_rejects_invalid_rrf_k() -> None:
	fuser = RRFFuser(rrf_k=0)

	with pytest.raises(ValueError, match="rrf_k must be greater than 0"):
		fuser.fuse(dense_results=[_result("a")], sparse_results=[], top_k=1)
