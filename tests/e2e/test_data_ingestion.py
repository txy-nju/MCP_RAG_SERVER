"""E2E-style tests for ingestion CLI entrypoint (C15)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.pipeline import IngestionResult
from scripts.ingest import main


VALID_CONFIG = """
llm:
  provider: openai
  model: gpt-4o-mini
embedding:
  provider: openai
  model: text-embedding-3-small
splitter:
  provider: recursive
  chunk_size: 1000
  chunk_overlap: 200
vector_store:
  provider: chroma
  collection: default
  persist_path: data/db/chroma
retrieval:
  top_k: 5
rerank:
  provider: none
  prompt_path: config/prompts/rerank.txt
  max_candidates: 20
evaluation:
  backend: custom
observability:
  log_level: INFO
  trace_file: logs/traces.jsonl
""".strip()


class FakePipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self._seen_paths: set[str] = set()

    def run(
        self,
        source_path: str,
        collection: str = "default",
        *,
        force: bool = False,
        batch_size: int = 32,
        trace=None,
        on_progress=None,
    ) -> IngestionResult:
        del trace, on_progress
        path_key = str(Path(source_path))
        skipped = (path_key in self._seen_paths) and (not force)
        if not skipped:
            self._seen_paths.add(path_key)

        Path("data/db").mkdir(parents=True, exist_ok=True)
        Path("data/db/ingestion_history.db").touch()

        self.calls.append(
            {
                "source_path": source_path,
                "collection": collection,
                "force": force,
                "batch_size": batch_size,
                "skipped": skipped,
            }
        )
        return IngestionResult(
            source_path=source_path,
            file_hash="fake-hash",
            collection=collection,
            skipped=skipped,
            chunk_count=2 if not skipped else 0,
            record_count=2 if not skipped else 0,
        )


@pytest.mark.e2e
def test_ingest_cli_runs_and_skips_on_repeated_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    config_path = tmp_path / "settings.yaml"
    config_path.write_text(VALID_CONFIG, encoding="utf-8")

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir(parents=True)
    (docs_dir / "a.pdf").write_bytes(b"%PDF-1.4 fake")

    fake_pipeline = FakePipeline()
    monkeypatch.setattr("scripts.ingest.build_pipeline", lambda settings: fake_pipeline)

    code = main(["--settings", str(config_path), "--path", str(docs_dir), "--collection", "demo"])
    assert code == 0
    assert fake_pipeline.calls[0]["collection"] == "demo"
    assert fake_pipeline.calls[0]["force"] is False
    assert fake_pipeline.calls[0]["skipped"] is False
    assert (tmp_path / "data/db/ingestion_history.db").exists()

    code = main(["--settings", str(config_path), "--path", str(docs_dir), "--collection", "demo"])
    assert code == 0
    assert fake_pipeline.calls[1]["skipped"] is True


@pytest.mark.e2e
def test_ingest_cli_force_reprocesses_even_if_seen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    config_path = tmp_path / "settings.yaml"
    config_path.write_text(VALID_CONFIG, encoding="utf-8")

    source_file = tmp_path / "single.pdf"
    source_file.write_bytes(b"%PDF-1.4 fake")

    fake_pipeline = FakePipeline()
    monkeypatch.setattr("scripts.ingest.build_pipeline", lambda settings: fake_pipeline)

    assert main(["--settings", str(config_path), "--path", str(source_file)]) == 0
    assert main(["--settings", str(config_path), "--path", str(source_file), "--force"]) == 0

    assert fake_pipeline.calls[0]["skipped"] is False
    assert fake_pipeline.calls[1]["force"] is True
    assert fake_pipeline.calls[1]["skipped"] is False


@pytest.mark.e2e
def test_ingest_cli_returns_error_for_missing_source_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    config_path = tmp_path / "settings.yaml"
    config_path.write_text(VALID_CONFIG, encoding="utf-8")

    code = main(["--settings", str(config_path), "--path", str(tmp_path / "missing")])

    assert code == 1
