"""Unit tests for FileIntegrityChecker / SQLiteIntegrityChecker."""

import hashlib
import tempfile
from pathlib import Path

import pytest

from libs.loader.file_integrity import SQLiteIntegrityChecker


@pytest.fixture()
def checker(tmp_path):
    """Return a fresh SQLiteIntegrityChecker that writes to a temp directory."""
    db = tmp_path / "ingestion_history.db"
    return SQLiteIntegrityChecker(db_path=db)


@pytest.fixture()
def sample_file(tmp_path):
    """Return a temporary file with known content."""
    f = tmp_path / "sample.txt"
    f.write_bytes(b"hello world")
    return f


# ---------------------------------------------------------------------------
# compute_sha256
# ---------------------------------------------------------------------------


def test_compute_sha256_deterministic(checker, sample_file):
    """Same file calculated twice must yield the same digest."""
    h1 = checker.compute_sha256(str(sample_file))
    h2 = checker.compute_sha256(str(sample_file))
    assert h1 == h2


def test_compute_sha256_correct_value(checker, sample_file):
    """Digest must match the reference value computed by hashlib directly."""
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert checker.compute_sha256(str(sample_file)) == expected


def test_compute_sha256_different_files_differ(checker, tmp_path):
    """Two files with different content must have different digests."""
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_bytes(b"aaa")
    f2.write_bytes(b"bbb")
    assert checker.compute_sha256(str(f1)) != checker.compute_sha256(str(f2))


# ---------------------------------------------------------------------------
# should_skip
# ---------------------------------------------------------------------------


def test_should_skip_returns_false_for_unknown_hash(checker):
    assert checker.should_skip("nonexistent_hash") is False


def test_should_skip_returns_true_after_mark_success(checker, sample_file):
    h = checker.compute_sha256(str(sample_file))
    checker.mark_success(h, str(sample_file))
    assert checker.should_skip(h) is True


def test_should_skip_returns_false_after_only_mark_failed(checker, sample_file):
    h = checker.compute_sha256(str(sample_file))
    checker.mark_failed(h, "some error")
    assert checker.should_skip(h) is False


# ---------------------------------------------------------------------------
# mark_success / mark_failed idempotence
# ---------------------------------------------------------------------------


def test_mark_success_idempotent(checker, sample_file):
    """Calling mark_success twice must not raise and should_skip must remain True."""
    h = checker.compute_sha256(str(sample_file))
    checker.mark_success(h, str(sample_file))
    checker.mark_success(h, str(sample_file))  # second call should not fail
    assert checker.should_skip(h) is True


def test_mark_failed_then_success_allows_skip(checker, sample_file):
    """Upgrading from failed → success should make should_skip return True."""
    h = checker.compute_sha256(str(sample_file))
    checker.mark_failed(h, "error")
    checker.mark_success(h, str(sample_file))
    assert checker.should_skip(h) is True


# ---------------------------------------------------------------------------
# DB file creation
# ---------------------------------------------------------------------------


def test_db_file_is_created(tmp_path):
    """The SQLite database file must be created automatically."""
    db_path = tmp_path / "sub" / "ingestion_history.db"
    SQLiteIntegrityChecker(db_path=db_path)
    assert db_path.exists()
