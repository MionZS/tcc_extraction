from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    if str(PROJECT_DIR) not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR))

    from src.araucaria_sample_pipeline import main as _run

    return _run()


if __name__ == "__main__":
    raise SystemExit(main())
