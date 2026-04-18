"""CLI script to launch the Streamlit Dashboard."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_APP_PATH = Path(__file__).parent.parent / "src" / "observability" / "dashboard" / "app.py"


def main() -> None:
    """Launch the Streamlit dashboard."""
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(_APP_PATH),
        "--server.headless",
        "false",
    ]
    print(f"Starting dashboard: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)  # noqa: S603


if __name__ == "__main__":
    main()
