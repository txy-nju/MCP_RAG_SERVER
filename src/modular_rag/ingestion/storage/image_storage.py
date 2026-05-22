"""Image file storage with SQLite-backed image index mapping."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class ImageStorage:
	"""Persist images to disk and maintain image_id -> file_path index."""

	_DEFAULT_IMAGE_ROOT = Path("data/images")
	_DEFAULT_DB_PATH = Path("data/db/image_index.db")

	def __init__(
		self,
		image_root: str | Path | None = None,
		db_path: str | Path | None = None,
	) -> None:
		self._image_root = Path(image_root) if image_root is not None else self._DEFAULT_IMAGE_ROOT
		self._db_path = Path(db_path) if db_path is not None else self._DEFAULT_DB_PATH

		self._image_root.mkdir(parents=True, exist_ok=True)
		self._db_path.parent.mkdir(parents=True, exist_ok=True)
		self._init_db()

	def save_image(
		self,
		image_id: str,
		image_bytes: bytes,
		*,
		collection: str = "default",
		doc_hash: str | None = None,
		page_num: int | None = None,
		extension: str = ".png",
	) -> str:
		"""Save image bytes and upsert index metadata.

		Returns the saved absolute file path as string.
		"""
		normalized_id = str(image_id).strip()
		if not normalized_id:
			raise ValueError("image_id must be a non-empty string")
		if not isinstance(image_bytes, (bytes, bytearray)) or len(image_bytes) == 0:
			raise ValueError("image_bytes must be non-empty bytes")

		normalized_collection = str(collection or "default").strip() or "default"
		normalized_ext = self._normalize_extension(extension)

		collection_dir = self._image_root / normalized_collection
		collection_dir.mkdir(parents=True, exist_ok=True)
		file_path = collection_dir / f"{normalized_id}{normalized_ext}"
		file_path.write_bytes(bytes(image_bytes))

		with self._connect() as conn:
			conn.execute(
				"""
				INSERT INTO image_index (image_id, file_path, collection, doc_hash, page_num)
				VALUES (?, ?, ?, ?, ?)
				ON CONFLICT(image_id) DO UPDATE SET
					file_path = excluded.file_path,
					collection = excluded.collection,
					doc_hash = excluded.doc_hash,
					page_num = excluded.page_num
				""",
				(normalized_id, str(file_path), normalized_collection, doc_hash, page_num),
			)

		return str(file_path)

	def get_image_path(self, image_id: str) -> str | None:
		"""Return indexed file path for *image_id*, or None if not found."""
		with self._connect() as conn:
			row = conn.execute(
				"SELECT file_path FROM image_index WHERE image_id = ?",
				(str(image_id),),
			).fetchone()
		if row is None:
			return None
		return str(row[0])

	def list_by_collection(self, collection: str) -> list[dict[str, Any]]:
		"""List image index rows for a collection, ordered by created time then id."""
		normalized_collection = str(collection or "default").strip() or "default"
		with self._connect() as conn:
			rows = conn.execute(
				"""
				SELECT image_id, file_path, collection, doc_hash, page_num, created_at
				FROM image_index
				WHERE collection = ?
				ORDER BY created_at ASC, image_id ASC
				""",
				(normalized_collection,),
			).fetchall()

		return [
			{
				"image_id": str(row[0]),
				"file_path": str(row[1]),
				"collection": str(row[2]),
				"doc_hash": row[3],
				"page_num": row[4],
				"created_at": str(row[5]),
			}
			for row in rows
		]

	def list_images(
		self,
		*,
		collection: str | None = None,
		doc_hash: str | None = None,
	) -> list[dict[str, Any]]:
		"""List image rows with optional collection/doc_hash filters."""
		clauses: list[str] = []
		params: list[str] = []

		if collection is not None:
			normalized_collection = str(collection).strip()
			if normalized_collection:
				clauses.append("collection = ?")
				params.append(normalized_collection)

		if doc_hash is not None:
			normalized_doc_hash = str(doc_hash).strip()
			if normalized_doc_hash:
				clauses.append("doc_hash = ?")
				params.append(normalized_doc_hash)

		where_clause = f"WHERE {' AND '.join(clauses)}" if clauses else ""
		query = f"""
			SELECT image_id, file_path, collection, doc_hash, page_num, created_at
			FROM image_index
			{where_clause}
			ORDER BY created_at ASC, image_id ASC
		"""

		with self._connect() as conn:
			rows = conn.execute(query, tuple(params)).fetchall()

		return [
			{
				"image_id": str(row[0]),
				"file_path": str(row[1]),
				"collection": str(row[2]),
				"doc_hash": row[3],
				"page_num": row[4],
				"created_at": str(row[5]),
			}
			for row in rows
		]

	def delete_images(
		self,
		*,
		collection: str | None = None,
		doc_hash: str | None = None,
	) -> int:
		"""Delete indexed images and underlying files by optional filters."""
		rows = self.list_images(collection=collection, doc_hash=doc_hash)
		if not rows:
			return 0

		deleted = 0
		for row in rows:
			try:
				path = Path(str(row["file_path"]))
				if path.exists():
					path.unlink()
			except Exception:
				# Index cleanup should continue even when disk cleanup partially fails.
				pass

		with self._connect() as conn:
			for row in rows:
				conn.execute("DELETE FROM image_index WHERE image_id = ?", (str(row["image_id"]),))
				deleted += 1

		return deleted

	def _connect(self) -> sqlite3.Connection:
		conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
		conn.execute("PRAGMA journal_mode=WAL")
		conn.execute("PRAGMA synchronous=NORMAL")
		return conn

	def _init_db(self) -> None:
		with self._connect() as conn:
			conn.execute(
				"""
				CREATE TABLE IF NOT EXISTS image_index (
					image_id TEXT PRIMARY KEY,
					file_path TEXT NOT NULL,
					collection TEXT,
					doc_hash TEXT,
					page_num INTEGER,
					created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
				)
				"""
			)
			conn.execute(
				"CREATE INDEX IF NOT EXISTS idx_collection ON image_index(collection)"
			)
			conn.execute(
				"CREATE INDEX IF NOT EXISTS idx_doc_hash ON image_index(doc_hash)"
			)

	@staticmethod
	def _normalize_extension(extension: str) -> str:
		normalized = str(extension or ".png").strip() or ".png"
		if not normalized.startswith("."):
			normalized = f".{normalized}"
		return normalized.lower()
