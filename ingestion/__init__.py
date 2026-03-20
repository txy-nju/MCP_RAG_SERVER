"""Root shim package for local src-based imports."""

from __future__ import annotations

from pathlib import Path
from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / __name__
if SRC_PACKAGE.exists():
    __path__.append(str(SRC_PACKAGE))
