"""Unit tests for ImageStorage file persistence and SQLite index mapping (C13)."""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestion.storage.image_storage import ImageStorage


@pytest.mark.unit
def test_save_image_creates_file_and_index_mapping(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    db_path = tmp_path / "db" / "image_index.db"
    storage = ImageStorage(image_root=image_root, db_path=db_path)

    saved_path = storage.save_image(
        "img-001",
        b"fake-image-bytes",
        collection="demo",
        doc_hash="doc-abc",
        page_num=3,
        extension="png",
    )

    assert Path(saved_path).exists()
    assert Path(saved_path).read_bytes() == b"fake-image-bytes"
    assert storage.get_image_path("img-001") == saved_path


@pytest.mark.unit
def test_mapping_persists_after_reopen(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    db_path = tmp_path / "db" / "image_index.db"

    first = ImageStorage(image_root=image_root, db_path=db_path)
    first.save_image("img-persist", b"abc", collection="demo", doc_hash="hash-1", page_num=1)

    second = ImageStorage(image_root=image_root, db_path=db_path)
    loaded_path = second.get_image_path("img-persist")

    assert loaded_path is not None
    assert Path(loaded_path).exists()
    assert Path(loaded_path).read_bytes() == b"abc"


@pytest.mark.unit
def test_list_by_collection_returns_only_target_collection(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    db_path = tmp_path / "db" / "image_index.db"
    storage = ImageStorage(image_root=image_root, db_path=db_path)

    storage.save_image("img-a", b"a", collection="alpha", doc_hash="d1", page_num=1)
    storage.save_image("img-b", b"b", collection="alpha", doc_hash="d1", page_num=2)
    storage.save_image("img-c", b"c", collection="beta", doc_hash="d2", page_num=1)

    rows = storage.list_by_collection("alpha")

    assert [row["image_id"] for row in rows] == ["img-a", "img-b"]
    assert all(row["collection"] == "alpha" for row in rows)


@pytest.mark.unit
def test_get_image_path_returns_none_for_missing_id(tmp_path: Path) -> None:
    storage = ImageStorage(image_root=tmp_path / "images", db_path=tmp_path / "db" / "image_index.db")
    assert storage.get_image_path("missing") is None


@pytest.mark.unit
def test_save_image_rejects_invalid_inputs(tmp_path: Path) -> None:
    storage = ImageStorage(image_root=tmp_path / "images", db_path=tmp_path / "db" / "image_index.db")

    with pytest.raises(ValueError, match="image_id"):
        storage.save_image("", b"payload")

    with pytest.raises(ValueError, match="image_bytes"):
        storage.save_image("img", b"")
