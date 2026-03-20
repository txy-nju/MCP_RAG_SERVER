"""Smoke tests for the initial package layout."""

from __future__ import annotations

import importlib

import pytest


@pytest.mark.unit
@pytest.mark.parametrize(
    "module_name",
    ["mcp_server", "core", "ingestion", "libs", "observability"],
)
def test_top_level_packages_import(module_name: str) -> None:
    """The clean-start skeleton should expose the spec's top-level packages."""
    module = importlib.import_module(module_name)
    assert module is not None
