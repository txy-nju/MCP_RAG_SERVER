"""Download arXiv PDFs from local metadata snapshot with resume support.

Usage (PowerShell):
  python scripts/download_arxiv_pdfs.py ^
    --snapshot D:\\datasets\\arxiv-metadata-oai-snapshot.json ^
    --output data/documents ^
    --total 10000 ^
    --workers 12 ^
    --by-category
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_STATE_PATH = Path("cache/arxiv_pdf_download_state.json")
DEFAULT_OUTPUT_PATH = Path("data/documents")
DEFAULT_TOTAL = 10_000


@dataclass(slots=True)
class DownloadTask:
	arxiv_id: str
	pdf_url: str
	collection: str
	destination: Path


@dataclass(slots=True)
class DownloadResult:
	arxiv_id: str
	status: str
	destination: Path | None = None
	error: str | None = None


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Download arXiv PDFs from local metadata snapshot.")
	parser.add_argument("--snapshot", required=True, help="Local arxiv-metadata-oai-snapshot.json path.")
	parser.add_argument(
		"--output",
		default=str(DEFAULT_OUTPUT_PATH),
		help="Output root directory. Default: data/documents",
	)
	parser.add_argument(
		"--state-file",
		default=str(DEFAULT_STATE_PATH),
		help="Resume state JSON path. Default: cache/arxiv_pdf_download_state.json",
	)
	parser.add_argument(
		"--total",
		type=int,
		default=DEFAULT_TOTAL,
		help="Target total PDF count under output root. Default: 10000",
	)
	parser.add_argument(
		"--workers",
		type=int,
		default=12,
		help="Concurrent download workers. Default: 12",
	)
	parser.add_argument(
		"--timeout",
		type=float,
		default=30.0,
		help="Per-request timeout seconds. Default: 30",
	)
	parser.add_argument(
		"--save-every",
		type=int,
		default=100,
		help="Persist state every N successful downloads. Default: 100",
	)
	parser.add_argument(
		"--by-category",
		action="store_true",
		help="Store PDFs by primary category as collection folder.",
	)
	parser.add_argument(
		"--default-collection",
		default="default",
		help="Collection folder name when --by-category is not used.",
	)
	return parser.parse_args()


def _sanitize_component(value: str, fallback: str = "unknown") -> str:
	normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
	normalized = normalized.strip("._-")
	return normalized or fallback


def _safe_file_stem(arxiv_id: str) -> str:
	# Old-style IDs may contain '/'; convert to stable single-file name.
	return _sanitize_component(arxiv_id.replace("/", "__"), fallback="paper")


def _primary_category(raw_categories: Any) -> str:
	if not isinstance(raw_categories, str):
		return "unknown"
	tokens = [token for token in raw_categories.split() if token.strip()]
	if not tokens:
		return "unknown"
	return _sanitize_component(tokens[0], fallback="unknown")


def _count_existing_pdfs(output_root: Path) -> int:
	if not output_root.exists():
		return 0
	return sum(1 for path in output_root.rglob("*.pdf") if path.is_file())


def _load_state(state_file: Path) -> set[str]:
	if not state_file.exists():
		return set()
	try:
		payload = json.loads(state_file.read_text(encoding="utf-8"))
	except json.JSONDecodeError:
		return set()
	downloaded_ids = payload.get("downloaded_ids", []) if isinstance(payload, dict) else []
	if not isinstance(downloaded_ids, list):
		return set()
	return {str(item).strip() for item in downloaded_ids if str(item).strip()}


def _save_state(state_file: Path, downloaded_ids: set[str]) -> None:
	state_file.parent.mkdir(parents=True, exist_ok=True)
	payload = {
		"updated_at": datetime.now(UTC).isoformat(),
		"downloaded_ids": sorted(downloaded_ids),
	}
	with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=str(state_file.parent)) as tmp:
		json.dump(payload, tmp, ensure_ascii=False)
		tmp.flush()
		tmp_path = Path(tmp.name)
	tmp_path.replace(state_file)


def _iter_snapshot_records(snapshot_path: Path) -> Iterator[dict[str, Any]]:
	with snapshot_path.open("r", encoding="utf-8") as handle:
		for line_no, line in enumerate(handle, start=1):
			content = line.strip()
			if not content or content in {"[", "]"}:
				continue
			if content.endswith(","):
				content = content[:-1]
			if not content:
				continue
			try:
				payload = json.loads(content)
			except json.JSONDecodeError:
				print(f"[WARN] skip invalid JSON line={line_no}")
				continue
			if isinstance(payload, dict):
				yield payload


def _build_task(record: dict[str, Any], output_root: Path, by_category: bool, default_collection: str) -> DownloadTask | None:
	raw_id = str(record.get("id", "")).strip()
	if not raw_id:
		return None

	collection = _primary_category(record.get("categories")) if by_category else _sanitize_component(default_collection)
	filename = f"{_safe_file_stem(raw_id)}.pdf"
	destination = output_root / collection / filename
	pdf_url = f"https://arxiv.org/pdf/{raw_id}.pdf"

	return DownloadTask(
		arxiv_id=raw_id,
		pdf_url=pdf_url,
		collection=collection,
		destination=destination,
	)


def _download_one(task: DownloadTask, timeout: float) -> DownloadResult:
	if task.destination.exists():
		return DownloadResult(arxiv_id=task.arxiv_id, status="exists", destination=task.destination)

	task.destination.parent.mkdir(parents=True, exist_ok=True)
	tmp_path = task.destination.with_suffix(task.destination.suffix + ".part")
	request = Request(
		task.pdf_url,
		headers={"User-Agent": "modular-rag-mcp-server/benchmark-downloader"},
	)

	try:
		with urlopen(request, timeout=timeout) as resp, tmp_path.open("wb") as out:
			out.write(resp.read())
		tmp_path.replace(task.destination)
		return DownloadResult(arxiv_id=task.arxiv_id, status="downloaded", destination=task.destination)
	except HTTPError as exc:
		if tmp_path.exists():
			tmp_path.unlink(missing_ok=True)
		return DownloadResult(arxiv_id=task.arxiv_id, status="failed", error=f"HTTP {exc.code}")
	except URLError as exc:
		if tmp_path.exists():
			tmp_path.unlink(missing_ok=True)
		return DownloadResult(arxiv_id=task.arxiv_id, status="failed", error=f"URL {exc.reason}")
	except Exception as exc:  # pragma: no cover - defensive fallback
		if tmp_path.exists():
			tmp_path.unlink(missing_ok=True)
		return DownloadResult(arxiv_id=task.arxiv_id, status="failed", error=str(exc))


def main() -> int:
	args = parse_args()

	snapshot_path = Path(args.snapshot)
	output_root = Path(args.output)
	state_file = Path(args.state_file)

	if args.total <= 0:
		print("[ERROR] --total must be greater than 0")
		return 1
	if args.workers <= 0:
		print("[ERROR] --workers must be greater than 0")
		return 1
	if args.timeout <= 0:
		print("[ERROR] --timeout must be greater than 0")
		return 1
	if not snapshot_path.exists():
		print(f"[ERROR] snapshot not found: {snapshot_path}")
		return 1

	output_root.mkdir(parents=True, exist_ok=True)
	known_ids = _load_state(state_file)

	existing_count = _count_existing_pdfs(output_root)
	if existing_count >= args.total:
		print(f"[DONE] existing pdf count={existing_count} already >= target={args.total}")
		return 0

	print(
		f"[START] snapshot={snapshot_path} output={output_root} target={args.total} "
		f"existing={existing_count} workers={args.workers} by_category={bool(args.by_category)}"
	)

	downloaded_new = 0
	skipped_existing = 0
	failed = 0
	scanned = 0
	save_counter = 0
	pending: dict[Future[DownloadResult], DownloadTask] = {}

	def handle_finished(future: Future[DownloadResult]) -> None:
		nonlocal downloaded_new, skipped_existing, failed, save_counter
		task = pending.pop(future)
		result = future.result()
		if result.status == "downloaded":
			downloaded_new += 1
			known_ids.add(task.arxiv_id)
			save_counter += 1
		elif result.status == "exists":
			skipped_existing += 1
			known_ids.add(task.arxiv_id)
		else:
			failed += 1
			print(f"[WARN] download failed id={task.arxiv_id} reason={result.error}")

		if save_counter >= args.save_every:
			_save_state(state_file, known_ids)
			save_counter = 0

	with ThreadPoolExecutor(max_workers=args.workers) as executor:
		for record in _iter_snapshot_records(snapshot_path):
			scanned += 1

			while pending and (existing_count + downloaded_new + len(pending) >= args.total):
				done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
				for item in done:
					handle_finished(item)

			if existing_count + downloaded_new >= args.total:
				break

			task = _build_task(
				record=record,
				output_root=output_root,
				by_category=bool(args.by_category),
				default_collection=str(args.default_collection),
			)
			if task is None:
				continue

			if task.arxiv_id in known_ids:
				continue

			if task.destination.exists():
				known_ids.add(task.arxiv_id)
				skipped_existing += 1
				continue

			future = executor.submit(_download_one, task, float(args.timeout))
			pending[future] = task

			if scanned % 2000 == 0:
				print(
					f"[PROGRESS] scanned={scanned} downloaded_new={downloaded_new} "
					f"skipped_existing={skipped_existing} failed={failed} pending={len(pending)}"
				)

		while pending and (existing_count + downloaded_new < args.total):
			done, _ = wait(set(pending.keys()), return_when=FIRST_COMPLETED)
			for item in done:
				handle_finished(item)

	_save_state(state_file, known_ids)
	final_total = _count_existing_pdfs(output_root)

	print(
		f"[DONE] final_total={final_total} target={args.total} downloaded_new={downloaded_new} "
		f"skipped_existing={skipped_existing} failed={failed} scanned={scanned}"
	)
	if final_total < args.total:
		print(
			"[INFO] target not reached. Re-run the same command to resume, "
			"or increase snapshot coverage / timeout / workers."
		)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
