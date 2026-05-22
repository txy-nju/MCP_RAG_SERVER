"""Integration-style tests for scripts/evaluate.py CLI entrypoint."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from modular_rag.observability.evaluation.eval_runner import EvalCaseResult, EvalReport


def test_evaluate_script_parse_args_defaults() -> None:
    from scripts.evaluate import parse_args

    args = parse_args([])
    assert args.test_set == "tests/fixtures/golden_test_set.json"
    assert args.settings == "config/settings.yaml"
    assert args.json is False


def test_evaluate_script_parse_args_json_flag() -> None:
    from scripts.evaluate import parse_args

    args = parse_args(["--json"])
    assert args.json is True


def test_evaluate_script_main_missing_test_set(tmp_path: Path) -> None:
    from scripts.evaluate import main

    result = main(["--test-set", str(tmp_path / "missing.json")])
    assert result == 1


@pytest.mark.integration
def test_evaluate_script_main_success_json_output(capsys: object, tmp_path: Path) -> None:
    from scripts.evaluate import main

    test_set_path = tmp_path / "golden.json"
    test_set_path.write_text('{"test_cases": [{"query": "q", "expected_chunk_ids": ["a"]}]}', encoding="utf-8")

    report = EvalReport(
        metrics={"hit_rate": 1.0, "mrr": 1.0},
        cases=[
            EvalCaseResult(
                query="q",
                metrics={"hit_rate": 1.0, "mrr": 1.0},
                retrieved_chunk_ids=["a"],
                retrieved_sources=["doc.md"],
                expected_chunk_ids=["a"],
                expected_sources=[],
            )
        ],
    )

    with patch("scripts.evaluate.load_settings") as mock_load_settings, patch(
        "scripts.evaluate.build_hybrid_search"
    ) as mock_build_hybrid_search, patch("scripts.evaluate.EvaluatorFactory.create") as mock_create, patch(
        "scripts.evaluate.EvalRunner"
    ) as mock_runner_cls:
        mock_load_settings.return_value = MagicMock()
        mock_build_hybrid_search.return_value = MagicMock()
        mock_create.return_value = MagicMock()
        mock_runner = MagicMock()
        mock_runner.run.return_value = report
        mock_runner_cls.return_value = mock_runner

        result = main(["--test-set", str(test_set_path), "--json"])

    captured = capsys.readouterr()
    assert '"hit_rate": 1.0' in captured.out
    assert result == 0


@pytest.mark.integration
def test_evaluate_script_main_success_text_output(capsys: object, tmp_path: Path) -> None:
    from scripts.evaluate import main

    test_set_path = tmp_path / "golden.json"
    test_set_path.write_text('{"test_cases": [{"query": "q", "expected_chunk_ids": ["a"]}]}', encoding="utf-8")

    report = EvalReport(
        metrics={"hit_rate": 1.0, "mrr": 1.0},
        cases=[
            EvalCaseResult(
                query="q",
                metrics={"hit_rate": 1.0, "mrr": 1.0},
                retrieved_chunk_ids=["a"],
                retrieved_sources=["doc.md"],
                expected_chunk_ids=["a"],
                expected_sources=[],
            )
        ],
    )

    with patch("scripts.evaluate.load_settings") as mock_load_settings, patch(
        "scripts.evaluate.build_hybrid_search"
    ) as mock_build_hybrid_search, patch("scripts.evaluate.EvaluatorFactory.create") as mock_create, patch(
        "scripts.evaluate.EvalRunner"
    ) as mock_runner_cls:
        mock_load_settings.return_value = MagicMock()
        mock_build_hybrid_search.return_value = MagicMock()
        mock_create.return_value = MagicMock()
        mock_runner = MagicMock()
        mock_runner.run.return_value = report
        mock_runner_cls.return_value = mock_runner

        result = main(["--test-set", str(test_set_path)])

    captured = capsys.readouterr()
    assert "Evaluation Report" in captured.out
    assert "hit_rate: 1.0000" in captured.out
    assert result == 0
