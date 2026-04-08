"""File integrity checker using SHA256 with SQLite backend."""

from __future__ import annotations

import hashlib
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path


class FileIntegrityChecker(ABC):
    """Abstract interface for file integrity checking."""

    @abstractmethod
    def compute_sha256(self, path: str) -> str:
        """Compute SHA256 hash of the file at *path*."""

    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        """Return True if *file_hash* was previously marked as successfully processed."""

    @abstractmethod
    def mark_success(self, file_hash: str, file_path: str, **kwargs) -> None:
        """Record *file_hash* as successfully processed."""

    @abstractmethod
    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        """Record *file_hash* as failed with *error_msg*."""


class SQLiteIntegrityChecker(FileIntegrityChecker):
    """SQLite-backed file integrity checker.

    The database is created automatically at *db_path* (default:
    ``data/db/ingestion_history.db``).  WAL journal mode is enabled so
    concurrent writers don't block each other.
    """

    _DEFAULT_DB = Path("data/db/ingestion_history.db")

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path is not None else self._DEFAULT_DB
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_sha256(self, path: str) -> str:
        """Return the hex-encoded SHA256 digest of the file at *path*."""
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    def should_skip(self, file_hash: str) -> bool:
        """Return True if *file_hash* has a ``success`` record in the DB."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM ingestion_history WHERE file_hash = ? AND status = 'success'",
                (file_hash,),
            ).fetchone()
        return row is not None

    def mark_success(self, file_hash: str, file_path: str, **kwargs) -> None:
        """Upsert a ``success`` record for *file_hash*."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_history (file_hash, file_path, status, extra)
                VALUES (?, ?, 'success', ?)
                ON CONFLICT(file_hash) DO UPDATE SET
                    file_path = excluded.file_path,
                    status    = 'success',
                    extra     = excluded.extra,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (file_hash, file_path, str(kwargs) if kwargs else None),
            )

    def mark_failed(self, file_hash: str, error_msg: str) -> None:
        """Upsert a ``failed`` record for *file_hash*."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO ingestion_history (file_hash, file_path, status, extra)
                VALUES (?, '', 'failed', ?)
                ON CONFLICT(file_hash) DO UPDATE SET
                    status     = 'failed',
                    extra      = excluded.extra,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (file_hash, error_msg),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ingestion_history (
                    file_hash  TEXT PRIMARY KEY,
                    file_path  TEXT NOT NULL DEFAULT '',
                    status     TEXT NOT NULL CHECK(status IN ('success', 'failed')),
                    extra      TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

